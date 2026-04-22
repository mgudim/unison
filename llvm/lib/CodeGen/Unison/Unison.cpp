//===- Unison.cpp - Unison CP-SAT Register Allocator ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Unison CP-SAT Register Allocator
//
// Combined register allocation and instruction scheduling using Google
// OR-Tools CP-SAT solver, based on the Unison approach (Castañeda Lozano
// et al., "Combinatorial Register Allocation and Instruction Scheduling").
//
// Model overview:
//   - Unified register domain: physical registers + memory (stack) slots.
//   - Each def becomes a rectangle in a 2D NoOverlap2D constraint:
//       x-axis = register index, y-axis = time (issue cycle).
//   - 2-stage pipeline: defs write at IC+1, uses read at IC.
//     Latency = 2: IC(use) >= IC(def) + 2.
//   - Copy extension: each vreg def gets optional store-move and
//     load-move CopyOps that model spilling/reloading.
//   - Scheduling: barriers (stores, calls, terminators) partition each
//     MBB into regions; pure instructions are free within a region.
//   - Congruence constraints tie register choices across CFG edges.
//
// Objective: minimize weighted sum of:
//   - Spill/reload cost (CopyOp activation, frequency-weighted)
//   - Copy coalescing (non-identity COPY penalty)
//   - Callee-saved register usage (per-register, frequency-weighted)
//   - Makespan (LiveOutUse issue cycle, frequency-weighted)
//
// Code generation emits physical registers directly (no VRM).
//
// Usage:
//   llc -load libLLVMUnisonRegAlloc.so -regalloc=unison [options] input.ll
//
// Options:
//   -unison-time-limit=N    Solver timeout per function, seconds (default 1200)
//   -unison-num-workers=N   Solver worker threads, 0 = auto (default 0)
//   -unison-preserve-orig-instr-order=false   Enable instruction reordering
//   -unison-dump-model      Dump solver solution to stderr
//
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/EquivalenceClasses.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/Twine.h"
#include "llvm/CodeGen/LiveIntervals.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineBlockFrequencyInfo.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/RegAllocRegistry.h"
#include "llvm/CodeGen/SlotIndexes.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"
#include "llvm/CodeGen/TargetSubtargetInfo.h"
#include "llvm/CodeGen/VirtRegMap.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"

#include "ortools/sat/cp_model.h"
#include "ortools/sat/cp_model_checker.h"

#include <algorithm>
#include <list>
#include <memory>
#include <vector>

using namespace llvm;
namespace sat = operations_research::sat;

#define DEBUG_TYPE "regalloc"

STATISTIC(NumSpillStores, "Number of spill stores inserted");
STATISTIC(NumSpillLoads, "Number of spill loads inserted");

static cl::opt<bool> UnisonDumpModel(
    "unison-dump-model",
    cl::desc("Dump CP-SAT model to debug output"),
    cl::Hidden);

static cl::opt<int> UnisonTimeLimitPerInstr(
    "unison-time-per-instr",
    cl::desc("CP-SAT solver time limit in seconds per instruction"),
    cl::init(10),
    cl::Hidden);

static cl::opt<int> UnisonMinTimeLimit(
    "unison-min-time",
    cl::desc("Minimum CP-SAT solver time limit in seconds"),
    cl::init(10),
    cl::Hidden);

static cl::opt<int> UnisonMaxTimeLimit(
    "unison-max-time",
    cl::desc("Maximum CP-SAT solver time limit in seconds"),
    cl::init(1200),
    cl::Hidden);

static cl::opt<int> UnisonNumWorkers(
    "unison-num-workers",
    cl::desc("Number of CP-SAT solver workers (0 = auto-detect)"),
    cl::init(0),
    cl::Hidden);

static cl::opt<std::string> UnisonDumpSolution(
    "unison-dump-solution",
    cl::desc("Dump solver solution to file"),
    cl::value_desc("filename"),
    cl::Hidden);

static cl::opt<std::string> UnisonPreAssign(
    "unison-preassign",
    cl::desc("Pre-assign variables from file"),
    cl::value_desc("filename"),
    cl::Hidden);

static cl::opt<bool> UnisonPreserveOrder(
    "unison-preserve-orig-instr-order",
    cl::desc("Constrain real instructions to their original program order"),
    cl::init(true),
    cl::Hidden);

// UnisonInstr represents an instruction in the model. RealInstr wraps a
// MachineInstr; LiveInDef/LiveOutUse are synthetic boundary instructions;
// CopyOp models potential spills/reloads with alternative instructions.
//
// UnisonOperand identifies a single operand within a UnisonInstr by
// {UnisonInstr*, OpIdx}. For real instructions, OpIdx matches MachineInstr
// operand indices (plus synthetic indices for regmask clobber defs). For
// optionals and boundaries we define our own layout.
//
// DefData is keyed by a def UnisonOperand. It holds the list of use
// operands this def can reach (PotentialUses) and the solver variable for
// physical register assignment (PhysRegVar).
//
// UseData is keyed by a use UnisonOperand. It holds the list of def
// operands that can supply this use (PotentialDefs) and a solver variable
// (ChoiceVar) that selects among them.

class Unison {
public:
  class UnisonInstr;

  // References to operands use {UnisonInstr*, index} pairs instead of
  // direct pointers. This avoids pointer invalidation when SmallVectors
  // inside UnisonInstr grow (e.g., when copyExtend adds
  // copy-extended alternatives to boundary operands). UnisonInstr is heap-
  // allocated via unique_ptr so the instruction pointer is stable.
  struct DefRef {
    UnisonInstr *UInstr;
    unsigned Idx;
  };
  struct UseRef {
    UnisonInstr *UInstr;
    unsigned Idx;
  };

  // A solver variable representing a physical register, vreg assignment,
  // or stack slot. Wraps a sat::IntVar whose domain is a set of dense
  // register indices (from MCRegToIdx) or stack slot indices.
  class UnisonRegister {
  public:
    sat::IntVar Var;
    operations_research::Domain Dom;
  };

  // A def operand. Holds its register variable and references to all
  // use operands that can read from this def.
  class UnisonDefOperand {
  public:
    UnisonRegister Reg;
    SmallVector<UseRef, 4> PotentialUses;
  };

  // A use operand. Holds references to all def operands that can supply
  // its value, and a ChoiceVar that selects among them. Without
  // optional instructions, PotentialDefs has one element and ChoiceVar
  // is constant 0.
  class UnisonUseOperand {
  public:
    SmallVector<DefRef, 2> PotentialDefs;
    sat::IntVar ChoiceVar;
  };

  class UnisonInstr {
  public:
    enum Kind {
      RealInstr,
      LiveInDef,
      LiveOutUse,
      CopyOp,     // store-move or load-move with alternative instructions
    };
    Kind K;
    MachineInstr *RealMI = nullptr;
    sat::IntVar IssueCycle;

    // Instruction selection: Ins indexes into Instrs.
    // For RealInstr: one element, Ins is constant.
    // For CopyOp: e.g. AltOpcodes={store,move}, Ins is a solver variable.
    sat::IntVar Ins;
    SmallVector<unsigned, 2> AltOpcodes;
    MachineInstr *RematMI = nullptr; // For COPY_REMAT: the instruction to clone.

    SmallVector<UnisonDefOperand, 4> Defs;
    SmallVector<UnisonUseOperand, 4> Uses;
    SmallString<32> Name; // Debug name for tracing.

    UnisonDefOperand &getDef(unsigned Idx) { return Defs[Idx]; }
    UnisonUseOperand &getUse(unsigned Idx) { return Uses[Idx]; }

    bool isAlwaysActive() const {
      return K == RealInstr || K == LiveInDef || K == LiveOutUse;
    }
    bool isCopyOp() const { return K == CopyOp; }
  };

  class UnisonMBB {
  public:
    MachineBasicBlock *MBB = nullptr;
    std::list<std::unique_ptr<UnisonInstr>> Instrs;
    std::optional<sat::NoOverlap2DConstraint> NoOverlap;
    int IssueCycleUpperBound = 0;
  };

  class UnisonFunction {
  public:
    SmallVector<std::unique_ptr<UnisonMBB>> MBBs;

    // Equivalence classes of def operands that must share a PhysRegVar.
    DenseMap<Register, SmallVector<DefRef, 2>> VRegDefClass;
  };

  // --- Variable naming and lookup ---

  class NamingScheme {
    bool BuildMap;
    StringMap<sat::IntVar> VarByName;

    void registerVar(StringRef Name, sat::IntVar Var) {
      Var.WithName(Name.str());
      if (BuildMap)
        VarByName[Name] = Var;
    }

    void registerBoolVar(StringRef Name, sat::BoolVar Var) {
      Var.WithName(Name.str());
      if (BuildMap)
        VarByName[Name] = sat::IntVar(Var);
    }

  public:
    NamingScheme(bool BuildMap = false) : BuildMap(BuildMap) {}

    static SmallString<32> nameVariable(StringRef InstrName,
                                        const Twine &VarSuffix) {
      SmallString<32> Result(InstrName);
      Result += ".";
      VarSuffix.toVector(Result);
      return Result;
    }

    void nameInstruction(UnisonInstr *UI, StringRef MBBPrefix,
                         unsigned InstrIdx);
    void nameAllVariablesInInstruction(UnisonInstr *UI);

    void nameActiveVar(UnisonInstr *UI, sat::BoolVar Active) {
      registerBoolVar(nameVariable(UI->Name, "active"), Active);
    }

    void renameDefReg(UnisonInstr *UI, unsigned DefIdx, sat::IntVar Var) {
      registerVar(nameVariable(UI->Name, "def[" + Twine(DefIdx) + "].reg"),
                  Var);
    }

    std::optional<sat::IntVar> lookupVar(StringRef Name) const {
      auto It = VarByName.find(Name);
      if (It == VarByName.end())
        return std::nullopt;
      return It->second;
    }
  };

  // --- Analyses needed from the pass ---

  class Analyses {
  public:
    MachineRegisterInfo *MRI;
    const TargetRegisterInfo *TRI;
    const TargetInstrInfo *TII;
    LiveIntervals *LIS;
    // VRM removed — we emit physical registers directly.
    const MachineBlockFrequencyInfo *MBFI;
  };

  // Scheduling mode controls issue cycle assignment.
  enum class SchedModelKind {
    // Each instruction occupies 2 slots (use slot, def slot). 
    // Used when scheduling is disabled or no
    // accurate scheduling model is available.
    TwoSlot,
    // Issue cycles are solver variables bounded by an estimate from
    // instruction itineraries / scheduling model. Enables joint
    // regalloc + scheduling.
    Full,
  };

  // --- Interface ---

  Unison(Analyses &A, MachineFunction &MF)
      : A(A), MF(MF), NS(!UnisonPreAssign.empty()) {}
  bool run();

private:
  Analyses &A;
  MachineFunction &MF;

  // CP-SAT model (one per function).
  sat::CpModelBuilder Model;

  // --- Register/memory index space ---
  // Physical registers: [0, NumPhysRegs)
  // Memory (stack) slots: [NumPhysRegs, NumPhysRegs+NumMemSlots)
  // Total RegMem domain: [0, NumPhysRegs+NumMemSlots)
  DenseMap<MCRegister, int> MCRegToIdx;  // phys reg -> index
  SmallVector<MCRegister> IdxToMCReg;    // index -> phys reg (for [0, NumPhysRegs))
  int NumPhysRegs = 0;
  int NumMemSlots = 0;

  // Precomputed domains.
  DenseMap<const TargetRegisterClass *, operations_research::Domain> RCDomain;
  operations_research::Domain MemDomain;    // [NumPhysRegs, NumPhysRegs+NumMemSlots)
  operations_research::Domain RegMemDomain; // [0, NumPhysRegs+NumMemSlots)

  // Restrict a variable to a domain. No-op if the variable is constant.
  void restrictToDomain(sat::IntVar Var, const operations_research::Domain &Dom);
  // Conditional version: only enforce if BoolVar is true.
  void restrictToDomain(sat::IntVar Var, const operations_research::Domain &Dom,
                        sat::BoolVar Condition);

  // The Unison model for the entire function.
  UnisonFunction UFunc;

  // For each (def, use) pair, the index of the def in the use's PotentialDefs.
  // Populated by wireDefUse / addDefChoice. Key: {DefRef, UseRef} encoded as
  // pair of pairs for DenseMap compatibility.
  using DefUseKey = std::pair<std::pair<UnisonInstr *, unsigned>,
                              std::pair<UnisonInstr *, unsigned>>;
  DenseMap<DefUseKey, unsigned> DefIdxInUseMap;

  // DChosenInU: reified BoolVar for "def DR was chosen at use UR".
  // Computed once, shared across all constraint functions.
  DenseMap<DefUseKey, sat::BoolVar> DChosenInUMap;

  DefUseKey makeDefUseKey(DefRef DR, UseRef UR) const {
    return {{DR.UInstr, DR.Idx}, {UR.UInstr, UR.Idx}};
  }

  // Look up the index of def DR in use UR's PotentialDefs.
  unsigned getDefIdxInUse(DefRef DR, UseRef UR) const {
    return DefIdxInUseMap.lookup(makeDefUseKey(DR, UR));
  }

  // DActiveInU: true iff def DR is chosen at use UR AND UR's instruction
  // is active. For always-active instructions, equals DChosenInU.
  sat::BoolVar getDActiveInU(DefRef DR, UseRef UR) {
    sat::BoolVar Chosen = getDChosenInU(DR, UR);
    if (UR.UInstr->isAlwaysActive())
      return Chosen;
    auto ActiveIt = IsActiveVar.find(UR.UInstr);
    if (ActiveIt == IsActiveVar.end())
      return Model.FalseVar();
    // AND of Chosen and IsActive.
    sat::BoolVar Active = Model.NewBoolVar();
    Model.AddBoolAnd({Chosen, ActiveIt->second}).OnlyEnforceIf(Active);
    Model.AddBoolOr({~Chosen, ~ActiveIt->second}).OnlyEnforceIf(~Active);
    return Active;
  }

  // Get or create the DChosenInU BoolVar for a (def, use) pair.
  // Returns TrueVar if the use has a single choice.
  sat::BoolVar getDChosenInU(DefRef DR, UseRef UR) {
    UnisonUseOperand &UseOp = UR.UInstr->getUse(UR.Idx);
    if (UseOp.PotentialDefs.size() == 1)
      return Model.TrueVar();
    auto Key = makeDefUseKey(DR, UR);
    auto It = DChosenInUMap.find(Key);
    if (It != DChosenInUMap.end())
      return It->second;
    unsigned DefIdx = DefIdxInUseMap.lookup(Key);
    sat::BoolVar B = reifyEquality(UseOp.ChoiceVar, DefIdx);
    DChosenInUMap[Key] = B;
    return B;
  }

  // Activation BoolVars for optional instructions.
  DenseMap<UnisonInstr *, sat::BoolVar> IsActiveVar;

  // Per-vreg PhysRegVar for VirtRegMap assignment.
  DenseMap<Register, sat::IntVar> VRegToPhysRegVar;

  // At code gen time, actual stack slot = FirstNewStackSlot + (solved value - NumPhysRegs).
  int FirstNewStackSlot = 0;

  // Solver response, stored after solve() for use by generateCodeFromSolution().
  sat::CpSolverResponse SolverResponse;

  SchedModelKind SchedKind = SchedModelKind::TwoSlot;

  // --- Helpers ---
  // Create a UnisonInstr and append it to Dest. Returns a raw pointer.
  UnisonInstr *createUnisonInstr(
      std::list<std::unique_ptr<UnisonInstr>> &Dest,
      UnisonInstr::Kind K, MachineInstr *RealMI = nullptr);

  // Wire a def to a use (bidirectional: adds to both PotentialUses and
  // PotentialDefs).
  void wireDefUse(DefRef DR, UseRef UR);

  // Add a new def choice to an existing use. Appends to PotentialDefs
  // and updates ChoiceVar domain to cover all choices.
  void addDefChoice(UseRef UR, DefRef DR);

  // Full reification: returns a BoolVar B such that B <=> (Var == Value).
  sat::BoolVar reifyEquality(sat::IntVar Var, int64_t Value);
  // Full reification: returns a BoolVar B such that B <=> (A != B).
  sat::BoolVar reifyNotEqual(sat::IntVar A, sat::IntVar B);
  // Full reification: returns a BoolVar B such that B <=> (X AND Y).
  sat::BoolVar reifyAnd(sat::BoolVar X, sat::BoolVar Y);

  // A canonical operand entry: the register and optionally a pointer to
  // the MachineOperand (null for regmask clobbers).
  struct CanonicalOperand {
    Register Reg;
    MachineOperand *MO = nullptr; // null for regmask clobbers
  };

  // Canonical ordering of register operands for a MachineInstr.
  // Single source of truth used by both model building and code generation.
  // Defs: explicit + implicit reg defs (skipping non-reg, invalid,
  //   non-allocatable phys regs), then regmask clobbers in IdxToMCReg order
  //   (skipping regs already explicitly defined).
  // Uses: explicit + implicit reg uses (skipping non-reg, invalid).
  void getMIDefsInCanonicalOrder(MachineInstr &MI,
                        SmallVectorImpl<CanonicalOperand> &Defs) const;
  void getMIUsesInCanonicalOrder(MachineInstr &MI,
                        SmallVectorImpl<CanonicalOperand> &Uses) const;

  // Extract defined/used registers from a UnisonInstr. The position of
  // each register in the returned vector establishes its index in
  // UnisonInstr::Defs or UnisonInstr::Uses.
  // For RealInstr: explicit + implicit register defs/uses, plus regmask
  //   clobbers (defs only). Regmask-clobbered registers that are also
  //   explicitly defined are skipped.
  // For LiveInDef: one def per live-in register (phys and virt).
  // For LiveOutUse: one use per live-out register (phys and virt).
  void getDefsFromUnisonInstr(UnisonInstr *UInstr, MachineBasicBlock &MBB,
                             SmallVectorImpl<Register> &Defs) const;
  void getUsesFromUnisonInstr(UnisonInstr *UInstr, MachineBasicBlock &MBB,
                             SmallVectorImpl<Register> &Uses) const;

  // Process one UnisonInstr: populate its Defs/Uses vectors, wire
  // def-use connections, update LocalReachingDefs and VRegDefClass.
  // Applies uniformly to LiveInDef, RealInstr, and LiveOutUse.
  void populateDefsAndUses(UnisonInstr *UInstr, MachineBasicBlock &MBB,
                         DenseMap<Register, DefRef> &LocalReachingDefs);

  // Upper bound on issue cycles for a given MBB, used for solver variable
  // domains. In TwoSlot mode: 2 * (num_instrs + 2) to account for
  // LiveInDef, instructions, LiveOutUse, and OStore/OLoads.
  // In Full mode: estimated from instruction latencies.
  int getIssueCycleUpperBound(MachineBasicBlock &MBB) const;

  // Build dense register index: MCRegToIdx, IdxToMCReg, RCDomain.
  void buildURegisterDomains();

  NamingScheme NS;

  // --- Pipeline stages ---
  void buildUnisonInstructionsAndAnalyzeDefs();
  void assignPhysRegVarsFromEqClasses();
  void copyExtend();
  // --- Constraint generation ---
  void addConstraints();

  // Per-MBB register allocation constraints.
  void addRegAllocConstraints(UnisonMBB &UMBB);
  void addNoOverlapConstraints(UnisonMBB &UMBB);
  // Add a rectangle to the NoOverlap2D constraint. For CopyOps, the
  // rectangle is optional (gated on IsActive). For always-active instrs,
  // it's unconditional.
  void addRectangle(UnisonMBB &UMBB, UnisonInstr *UInstr,
                    sat::IntVar RegVar,
                    sat::IntVar TimeStart, sat::IntVar TimeSize,
                    sat::IntVar TimeEnd);
  // Returns max over active uses of IssueCycle(use_instr). Each use
  // contributes conditionally (DActiveInU). Inactive uses contribute the
  // definer's IC. For dead defs (no uses), returns the definer's IC.
  // Caller computes TimeEnd = result + 1 (half-open interval).
  sat::IntVar getDefLastUseCycle(UnisonInstr *UInstr, unsigned DefIdx,
                                int IssueCycleUB);
  void addRegClassConstraints(UnisonMBB &UMBB);
  void deriveActivationVars(UnisonMBB &UMBB);
  sat::BoolVar deriveDefActivation(UnisonInstr *UInstr, unsigned DefIdx);

  // Per-MBB scheduling constraints.
  void addSchedConstraints(UnisonMBB &UMBB);
  void addDataDependencyConstraints(UnisonMBB &UMBB);
  void addAntiDependencyConstraints(UnisonMBB &UMBB);
  void addOrderingConstraints(UnisonMBB &UMBB);

  // Cross-MBB constraints.
  void addCongruenceConstraints();

  void addObjectiveFunction();
  void penalizeCopies(sat::LinearExpr &Objective);
  void penalizeCalleeSavedRegisters(sat::LinearExpr &Objective);
  void addMinimizeMakespanObjective(sat::LinearExpr &Objective);

  // Returns the set of register indices for callee-saved registers
  // that are NOT already modified in the function.
  DenseSet<int> getPenalizedCSRIndices() const;

  // Helper: returns a BoolVar that is true iff the given instruction's
  // def and use are at different registers (non-identity copy).
  sat::BoolVar getIsNotIdentityCopy(UnisonInstr *UInstr);
  void solve();
  void dumpSolution(StringRef Filename);
  void loadPreAssignments(StringRef Filename);
  void generateCodeFromSolution();

  // --- Code generation phases ---
  // Remove inactive CopyOps from each MBB's instruction list.
  void removeInactiveInstructions();
  // Sort each MBB's UnisonInstrs by solved issue cycle.
  void sortByIssueCycle();
  // Walk sorted instructions: re-emit real instrs and materialize CopyOps.
  void generateInstructions();

  // Create a MachineInstr for an active CopyOp and insert it before
  // InsertPt using physical registers directly. Returns the new MI
  // (or null for identity moves).
  MachineInstr *materializeCopyOp(UnisonInstr *UInstr,
                                  MachineBasicBlock *MBB,
                                  MachineBasicBlock::iterator InsertPt);

  // Maps memory indices to frame indices (stack slots).
  DenseMap<int, int> MemIdxToFrameIdx;
};

// ---------------------------------------------------------------------------
// Unison implementation
// ---------------------------------------------------------------------------

Unison::UnisonInstr *Unison::createUnisonInstr(
    std::list<std::unique_ptr<UnisonInstr>> &Dest,
    UnisonInstr::Kind K, MachineInstr *RealMI) {
  auto UInstr = std::make_unique<UnisonInstr>();
  UInstr->K = K;
  UInstr->RealMI = RealMI;
  UnisonInstr *Ptr = UInstr.get();
  Dest.push_back(std::move(UInstr));
  return Ptr;
}

// createCopyOp removed — CopyOps are now created inline in copyExtend
// with in-place insertion into the instruction list.

void Unison::wireDefUse(DefRef DR, UseRef UR) {
  DR.UInstr->getDef(DR.Idx).PotentialUses.push_back(UR);
  unsigned Idx = UR.UInstr->getUse(UR.Idx).PotentialDefs.size();
  UR.UInstr->getUse(UR.Idx).PotentialDefs.push_back(DR);
  DefIdxInUseMap[makeDefUseKey(DR, UR)] = Idx;
}

void Unison::addDefChoice(UseRef UR, DefRef DR) {
  // Add def to use's PotentialDefs.
  UnisonUseOperand &UseOp = UR.UInstr->getUse(UR.Idx);
  unsigned Idx = UseOp.PotentialDefs.size();
  UseOp.PotentialDefs.push_back(DR);
  DefIdxInUseMap[makeDefUseKey(DR, UR)] = Idx;
  // Add use to def's PotentialUses.
  DR.UInstr->getDef(DR.Idx).PotentialUses.push_back(UR);
  unsigned N = UseOp.PotentialDefs.size();
  if (N > 1)
    UseOp.ChoiceVar = Model.NewIntVar({0, static_cast<int64_t>(N - 1)});
}

void Unison::getMIDefsInCanonicalOrder(MachineInstr &MI,
                              SmallVectorImpl<CanonicalOperand> &Defs) const {
  Defs.clear();

  DenseSet<MCRegister> ExplicitPhysDefs;
  for (MachineOperand &MO : MI.all_defs()) {
    if (!MO.isReg() || !MO.getReg().isValid())
      continue;
    Register Reg = MO.getReg();
    if (Reg.isPhysical()) {
      if (!MCRegToIdx.count(MCRegister(Reg)))
        continue;
      ExplicitPhysDefs.insert(MCRegister(Reg));
    }
    Defs.push_back({Reg, &MO});
  }

  // Regmask clobbers: iterate by dense index order for deterministic
  // positioning. No MachineOperand for these.
  for (const MachineOperand &MO : MI.operands()) {
    if (!MO.isRegMask())
      continue;
    for (unsigned Idx = 0, E = IdxToMCReg.size(); Idx < E; ++Idx) {
      MCRegister PhysReg = IdxToMCReg[Idx];
      if (!MO.clobbersPhysReg(PhysReg))
        continue;
      if (ExplicitPhysDefs.count(PhysReg))
        continue;
      Defs.push_back({Register(PhysReg), nullptr});
    }
  }
}

void Unison::getMIUsesInCanonicalOrder(MachineInstr &MI,
                              SmallVectorImpl<CanonicalOperand> &Uses) const {
  Uses.clear();
  for (MachineOperand &MO : MI.all_uses()) {
    if (!MO.isReg() || !MO.getReg().isValid())
      continue;
    Register Reg = MO.getReg();
    // Skip non-allocatable physical registers (e.g., $x0 zero register).
    if (Reg.isPhysical() && !MCRegToIdx.count(MCRegister(Reg)))
      continue;
    Uses.push_back({Reg, &MO});
  }
}

void Unison::getDefsFromUnisonInstr(UnisonInstr *UInstr, MachineBasicBlock &MBB,
                                   SmallVectorImpl<Register> &Defs) const {
  Defs.clear();

  if (UInstr->K == UnisonInstr::LiveInDef) {
    for (const auto &LI : MBB.liveins()) {
      if (MCRegToIdx.count(LI.PhysReg))
        Defs.push_back(Register(LI.PhysReg));
    }
    for (unsigned I = 0, E = A.MRI->getNumVirtRegs(); I != E; ++I) {
      Register Reg = Register::index2VirtReg(I);
      if (A.MRI->reg_nodbg_empty(Reg))
        continue;
      if (!A.LIS->hasInterval(Reg))
        continue;
      if (A.LIS->isLiveInToMBB(A.LIS->getInterval(Reg), &MBB))
        Defs.push_back(Reg);
    }
    return;
  }

  if (UInstr->K != UnisonInstr::RealInstr)
    return;

  assert(UInstr->RealMI);
  SmallVector<CanonicalOperand, 8> CanonDefs;
  getMIDefsInCanonicalOrder(*UInstr->RealMI, CanonDefs);
  for (const CanonicalOperand &CO : CanonDefs)
    Defs.push_back(CO.Reg);
}

void Unison::getUsesFromUnisonInstr(UnisonInstr *UInstr, MachineBasicBlock &MBB,
                                   SmallVectorImpl<Register> &Uses) const {
  Uses.clear();

  if (UInstr->K == UnisonInstr::LiveOutUse) {
    DenseSet<MCRegister> SeenPhys;
    for (const MachineBasicBlock *Succ : MBB.successors()) {
      for (const auto &LI : Succ->liveins()) {
        if (!MCRegToIdx.count(LI.PhysReg))
          continue;
        if (SeenPhys.insert(LI.PhysReg).second)
          Uses.push_back(Register(LI.PhysReg));
      }
    }
    for (unsigned I = 0, E = A.MRI->getNumVirtRegs(); I != E; ++I) {
      Register Reg = Register::index2VirtReg(I);
      if (A.MRI->reg_nodbg_empty(Reg))
        continue;
      if (!A.LIS->hasInterval(Reg))
        continue;
      if (A.LIS->isLiveOutOfMBB(A.LIS->getInterval(Reg), &MBB))
        Uses.push_back(Reg);
    }
    return;
  }

  if (UInstr->K != UnisonInstr::RealInstr)
    return;

  assert(UInstr->RealMI);
  SmallVector<CanonicalOperand, 8> CanonUses;
  getMIUsesInCanonicalOrder(*UInstr->RealMI, CanonUses);
  for (const CanonicalOperand &CO : CanonUses)
    Uses.push_back(CO.Reg);
}

void Unison::populateDefsAndUses(UnisonInstr *UInstr, MachineBasicBlock &MBB,
                               DenseMap<Register, DefRef> &LocalReachingDefs) {
  SmallVector<Register> UseRegs, DefRegs;

  // Process uses first (so use-and-def of same reg sees the previous def).
  getUsesFromUnisonInstr(UInstr, MBB, UseRegs);
  UInstr->Uses.resize(UseRegs.size());
  for (unsigned I = 0, E = UseRegs.size(); I < E; ++I) {
    Register Reg = UseRegs[I];

    auto It = LocalReachingDefs.find(Reg);
    assert(It != LocalReachingDefs.end() && "Use without reaching def");

    DefRef DR = It->second;
    UseRef UR{UInstr, I};

    // Initially constant 0 (single def choice). May be replaced with a
    // solver variable by copyExtend if copy-extended alternatives are added.
    UInstr->Uses[I].ChoiceVar = Model.NewConstant(0);
    wireDefUse(DR, UR);
  }

  // Process defs.
  getDefsFromUnisonInstr(UInstr, MBB, DefRegs);
  UInstr->Defs.resize(DefRegs.size());
  for (unsigned I = 0, E = DefRegs.size(); I < E; ++I) {
    Register Reg = DefRegs[I];
    UnisonDefOperand &DefOp = UInstr->Defs[I];
    DefRef DR{UInstr, I};

    if (Reg.isPhysical()) {
      DefOp.Reg.Var = Model.NewConstant(MCRegToIdx[MCRegister(Reg)]);
      for (auto &[RC, Dom] : RCDomain) {
        if (RC->contains(MCRegister(Reg))) {
          DefOp.Reg.Dom = Dom;
          break;
        }
      }
    } else {
      DefOp.Reg.Dom = RCDomain[A.MRI->getRegClass(Reg)];
    }
    // Virtual register Var assigned after traversal via VRegDefClass.

    LocalReachingDefs[Reg] = DR;

    if (Reg.isVirtual())
      UFunc.VRegDefClass[Reg].push_back(DR);
  }
}

int Unison::getIssueCycleUpperBound(MachineBasicBlock &MBB) const {
  // Count real (non-debug) instructions.
  unsigned NumInstrs = 0;
  for (const MachineInstr &MI : MBB)
    if (!MI.isDebugInstr())
      ++NumInstrs;

  if (SchedKind == SchedModelKind::TwoSlot) {
    // Each real instruction and each CopyOp needs its own IC.
    // Generous headroom for CopyOp chains.
    return (NumInstrs + 2) * 4;
  }

  // Full mode: sum of instruction latencies as upper bound.
  // TODO: use TargetSchedKindl to estimate total latency.
  // For now, fall back to a generous multiplier.
  return NumInstrs * 4;
}

void Unison::buildURegisterDomains() {
  // Build dense register index from all register classes used by vregs.
  for (unsigned I = 0, E = A.MRI->getNumVirtRegs(); I != E; ++I) {
    Register VReg = Register::index2VirtReg(I);
    if (A.MRI->reg_nodbg_empty(VReg))
      continue;
    const TargetRegisterClass *RC = A.MRI->getRegClass(VReg);
    if (RCDomain.count(RC))
      continue;
    std::vector<int64_t> Vals;
    for (MCPhysReg R : RC->getRawAllocationOrder(MF)) {
      MCRegister MCReg(R);
      if (A.MRI->isReserved(MCReg))
        continue;
      if (!MCRegToIdx.count(MCReg)) {
        MCRegToIdx[MCReg] = static_cast<int>(IdxToMCReg.size());
        IdxToMCReg.push_back(MCReg);
      }
      Vals.push_back(MCRegToIdx[MCReg]);
    }
    llvm::sort(Vals);
    RCDomain[RC] = operations_research::Domain::FromValues(Vals);
  }
  NumPhysRegs = IdxToMCReg.size();

  // Compute memory slot upper bound: one slot per vreg def (generous).
  // Actual count may be lower; this just sets the domain ceiling.
  unsigned NumVRegDefs = 0;
  for (unsigned I = 0, E = A.MRI->getNumVirtRegs(); I != E; ++I) {
    Register VReg = Register::index2VirtReg(I);
    if (!A.MRI->reg_nodbg_empty(VReg))
      ++NumVRegDefs;
  }
  NumMemSlots = NumVRegDefs;
  MemDomain = (NumMemSlots > 0)
      ? operations_research::Domain(NumPhysRegs,
                                    NumPhysRegs + NumMemSlots - 1)
      : operations_research::Domain();
  RegMemDomain = (NumPhysRegs + NumMemSlots > 0)
      ? operations_research::Domain(0, NumPhysRegs + NumMemSlots - 1)
      : operations_research::Domain();
  FirstNewStackSlot = MF.getFrameInfo().getNumObjects();

  LLVM_DEBUG({
    dbgs() << "  Register index (" << NumPhysRegs << " phys regs, "
           << NumMemSlots << " mem slots):";
    for (int I = 0; I < NumPhysRegs; ++I)
      dbgs() << " " << I << "=" << printReg(IdxToMCReg[I], A.TRI);
    dbgs() << "\n";
  });
}

// ---------------------------------------------------------------------------
// Variable naming
// ---------------------------------------------------------------------------

// Opcode constants for CopyOp instruction selection.
enum CopyOpcode : unsigned {
  COPY_STORE = 0, // Spill to memory: def domain = memory slots.
  COPY_MOVE = 1,  // Register-to-register move: both domains = registers.
  COPY_LOAD = 2,  // Reload from memory: use domain = memory, def = registers.
  COPY_REMAT = 3, // Rematerialize: recompute value instead of loading.
};

// ---------------------------------------------------------------------------
// NamingScheme implementation
// ---------------------------------------------------------------------------

void Unison::NamingScheme::nameInstruction(UnisonInstr *UI,
                                           StringRef MBBPrefix,
                                           unsigned InstrIdx) {
  switch (UI->K) {
  case UnisonInstr::LiveInDef:
    UI->Name = MBBPrefix;
    UI->Name += ".LI";
    break;
  case UnisonInstr::LiveOutUse:
    UI->Name = MBBPrefix;
    UI->Name += ".LO";
    break;
  case UnisonInstr::RealInstr:
    UI->Name = MBBPrefix;
    UI->Name += ".RI";
    UI->Name += Twine(InstrIdx).str();
    break;
  case UnisonInstr::CopyOp: {
    // Walk use[0] → parent def → parent instruction.
    assert(!UI->Uses.empty() && !UI->Uses[0].PotentialDefs.empty());
    DefRef ParentDR = UI->Uses[0].PotentialDefs[0];

    bool IsStoreMove = UI->AltOpcodes[0] == COPY_STORE;
    if (IsStoreMove) {
      UI->Name = ParentDR.UInstr->Name;
      UI->Name += ".d";
      UI->Name += Twine(ParentDR.Idx).str();
      UI->Name += ".SM";
    } else {
      assert(ParentDR.UInstr->isCopyOp());
      DefRef RealDR = ParentDR.UInstr->Uses[0].PotentialDefs[0];
      assert(!UI->Defs[0].PotentialUses.empty());
      UnisonInstr *UseInstr = UI->Defs[0].PotentialUses[0].UInstr;

      UI->Name = RealDR.UInstr->Name;
      UI->Name += ".d";
      UI->Name += Twine(RealDR.Idx).str();
      UI->Name += ".LM_";
      UI->Name += UseInstr->Name;
    }
    break;
  }
  }
}

void Unison::NamingScheme::nameAllVariablesInInstruction(UnisonInstr *UI) {
  registerVar(nameVariable(UI->Name, "ic"), UI->IssueCycle);

  for (unsigned I = 0; I < UI->Defs.size(); I++)
    registerVar(nameVariable(UI->Name, "def[" + Twine(I) + "].reg"),
                UI->Defs[I].Reg.Var);

  for (unsigned I = 0; I < UI->Uses.size(); I++)
    registerVar(nameVariable(UI->Name, "use[" + Twine(I) + "].choice"),
                UI->Uses[I].ChoiceVar);

  if (UI->isCopyOp())
    registerVar(nameVariable(UI->Name, "ins"), UI->Ins);
}

void Unison::buildUnisonInstructionsAndAnalyzeDefs() {
  // Traverse MBBs in reverse post-order. RPO guarantees that a block is
  // visited before any block it dominates (assuming reducible CFG), so the
  // defining block of a vreg is processed before blocks where that vreg
  // is live-in. This ensures VRegDefClass already has an entry for a vreg
  // when we encounter it as a live-in in a successor block.
  for (MachineBasicBlock *MBB :
       ReversePostOrderTraversal<MachineFunction *>(&this->MF)) {
    auto UMBB = std::make_unique<UnisonMBB>();
    UMBB->MBB = MBB;

    // Maps each register to its most recent def in this MBB. Needed
    // because a register can be defined multiple times in a block
    // (especially physical registers — e.g., $x10 as function arg,
    // then clobbered by a call, then redefined as call return). When
    // we encounter a use, this tells us which specific def it reads from.
    DenseMap<Register, DefRef> LocalReachingDefs;
    UMBB->IssueCycleUpperBound = getIssueCycleUpperBound(*MBB);
    int UB = UMBB->IssueCycleUpperBound;

    // Create a UnisonInstr, assign its issue cycle, and analyze defs/uses.
    // All ICs are solver variables. LiveInDef is anchored at 0.
    // When UnisonPreserveOrder is set, consecutive real instructions are
    // chained with AddLessThan to preserve program order while still
    // letting the solver space them apart for CopyOps.
    UnisonInstr *PrevRealInstr = nullptr;
    unsigned RICount = 0;
    SmallString<16> MBBPrefix("MBB");
    MBBPrefix += Twine(MBB->getNumber()).str();
    auto addUnisonInstr = [&](UnisonInstr::Kind K, MachineInstr *RealMI) {
      UnisonInstr *UInstr = createUnisonInstr(UMBB->Instrs, K, RealMI);
      NS.nameInstruction(UInstr, MBBPrefix, RICount);
      if (K == UnisonInstr::RealInstr)
        RICount++;
      if (K == UnisonInstr::LiveInDef) {
        UInstr->IssueCycle = Model.NewConstant(0);
      } else {
        UInstr->IssueCycle = Model.NewIntVar(
            operations_research::Domain(0, UB));
      }
      if (K == UnisonInstr::RealInstr && UnisonPreserveOrder) {
        UnisonInstr *Prev = PrevRealInstr ? PrevRealInstr
                                          : UMBB->Instrs.front().get();
        Model.AddLessThan(Prev->IssueCycle, UInstr->IssueCycle);
        PrevRealInstr = UInstr;
      }
      populateDefsAndUses(UInstr, *MBB, LocalReachingDefs);
      NS.nameAllVariablesInInstruction(UInstr);
      return UInstr;
    };

    addUnisonInstr(UnisonInstr::LiveInDef, nullptr);

    for (MachineInstr &MI : *MBB) {
      if (MI.isDebugInstr())
        continue;
      addUnisonInstr(UnisonInstr::RealInstr, &MI);
    }

    addUnisonInstr(UnisonInstr::LiveOutUse, nullptr);

    LLVM_DEBUG(dbgs() << "  MBB#" << MBB->getNumber() << ": "
                      << UMBB->Instrs.size() << " unison instrs\n");

    UFunc.MBBs.push_back(std::move(UMBB));
  }

  assignPhysRegVarsFromEqClasses();
}

// For each vreg equivalence class, create one solver variable and assign
// it to all defs of that vreg across all MBBs. This ensures LLVM's
// one-vreg-one-physreg invariant (required by VirtRegMap / VirtRegRewriter).
void Unison::assignPhysRegVarsFromEqClasses() {
  LLVM_DEBUG(dbgs() << "  VRegDefClass has " << UFunc.VRegDefClass.size()
                    << " vregs\n");
  for (auto &[Reg, DefOps] : UFunc.VRegDefClass) {
    LLVM_DEBUG(dbgs() << "    " << printReg(Reg, A.TRI)
                      << " (" << DefOps.size() << " defs)\n");
    const TargetRegisterClass *RC = A.MRI->getRegClass(Reg);
    // Use unified domain (registers + memory) so that cross-MBB
    // congruence can propagate memory assignments through LiveInDef.
    // addRegClassConstraints will restrict real instruction defs to
    // registers only.
    operations_research::Domain Dom = RCDomain[RC].UnionWith(MemDomain);
    sat::IntVar PhysRegVar = Model.NewIntVar(Dom);
    VRegToPhysRegVar[Reg] = PhysRegVar;
    for (DefRef DR : DefOps) {
      DR.UInstr->getDef(DR.Idx).Reg.Var = PhysRegVar;
      // Re-apply name: the per-def variable was replaced by the shared one.
      NS.renameDefReg(DR.UInstr, DR.Idx, PhysRegVar);
    }
  }
}

// Check if a MachineInstr is rematerializable: cheap, no memory access,
// all use operands are constants or always-available (e.g., $x0).
static bool isRematerializable(const MachineInstr &MI,
                               const MachineRegisterInfo &MRI) {
  // Must be a single simple instruction.
  if (MI.mayLoad() || MI.mayStore() || MI.isCall() || MI.hasUnmodeledSideEffects())
    return false;
  if (MI.getNumDefs() != 1)
    return false;
  // All use operands must be immediates, frame indices, or the zero register.
  for (const MachineOperand &MO : MI.uses()) {
    if (MO.isReg()) {
      if (!MO.getReg().isValid())
        continue;
      // Only allow reserved/non-allocatable fixed regs (e.g., $x0, $sp).
      if (MO.getReg().isVirtual())
        return false;
      if (!MRI.isReserved(MO.getReg()))
        return false;
    } else if (!MO.isImm() && !MO.isFI() && !MO.isGlobal() &&
               !MO.isCPI() && !MO.isSymbol()) {
      return false;
    }
  }
  return true;
}

void Unison::copyExtend() {
  if (MemDomain.IsEmpty())
    return;

  for (auto &UMBB : UFunc.MBBs) {
    int UB = UMBB->IssueCycleUpperBound;

    // Build UnisonInstr* → list iterator map for in-place insertion.
    DenseMap<UnisonInstr *, std::list<std::unique_ptr<UnisonInstr>>::iterator>
        InstrToIter;
    for (auto It = UMBB->Instrs.begin(); It != UMBB->Instrs.end(); ++It)
      InstrToIter[It->get()] = It;

    // Helper: create a CopyOp and insert it into the list at a position.
    auto insertCopyOp = [&](std::list<std::unique_ptr<UnisonInstr>>::iterator Pos,
                            ArrayRef<unsigned> Opcodes,
                            const operations_research::Domain &RegDom,
                            const operations_research::Domain &UnifiedDom)
        -> UnisonInstr * {
      auto UInstr = std::make_unique<UnisonInstr>();
      UInstr->K = UnisonInstr::CopyOp;
      UInstr->IssueCycle = Model.NewIntVar({0, UB});
      UInstr->AltOpcodes.assign(Opcodes.begin(), Opcodes.end());
      UInstr->Ins = (Opcodes.size() == 1)
          ? Model.NewConstant(0)
          : Model.NewIntVar({0, static_cast<int64_t>(Opcodes.size() - 1)});
      UInstr->Uses.resize(1);
      UInstr->Defs.resize(1);
      UInstr->Defs[0].Reg.Var = Model.NewIntVar(UnifiedDom);
      UInstr->Defs[0].Reg.Dom = RegDom;
      UnisonInstr *Ptr = UInstr.get();
      auto NewIt = UMBB->Instrs.insert(Pos, std::move(UInstr));
      InstrToIter[Ptr] = NewIt;
      return Ptr;
    };

    // Iterate over a snapshot of instructions (copy extension adds new
    // entries, but we only process original ones).
    SmallVector<UnisonInstr *, 32> OrigInstrs;
    for (auto &UIP : UMBB->Instrs)
      OrigInstrs.push_back(UIP.get());

    for (UnisonInstr *UInstr : OrigInstrs) {
      for (unsigned DefIdx = 0; DefIdx < UInstr->Defs.size(); ++DefIdx) {
        UnisonDefOperand &DefOp = UInstr->Defs[DefIdx];
        if (DefOp.PotentialUses.empty())
          continue;

        DefRef RealDR{UInstr, DefIdx};
        unsigned NumRealUses = DefOp.PotentialUses.size();

        // --- Store-move: inserted right AFTER the def instruction ---
        auto DefIt = InstrToIter[UInstr];
        auto AfterDef = std::next(DefIt);
        UnisonInstr *StoreMove = insertCopyOp(
            AfterDef, {COPY_STORE, COPY_MOVE},
            DefOp.Reg.Dom, DefOp.Reg.Dom.UnionWith(MemDomain));
        StoreMove->Uses[0].ChoiceVar = Model.NewConstant(0);
        wireDefUse(RealDR, UseRef{StoreMove, 0});
        DefRef StoreMoveDR{StoreMove, 0};

        bool CanRemat = UInstr->K == UnisonInstr::RealInstr &&
                        UInstr->RealMI &&
                        isRematerializable(*UInstr->RealMI, *A.MRI);

        // --- Load-moves: inserted right BEFORE each real use ---
        for (unsigned UI = 0; UI < NumRealUses; ++UI) {
          UseRef RealUR = DefOp.PotentialUses[UI];

          SmallVector<unsigned, 3> LoadOpcodes = {COPY_LOAD, COPY_MOVE};
          if (CanRemat)
            LoadOpcodes.push_back(COPY_REMAT);

          auto UseIt = InstrToIter[RealUR.UInstr];
          UnisonInstr *LoadMove = insertCopyOp(
              UseIt, LoadOpcodes,
              DefOp.Reg.Dom, DefOp.Reg.Dom.UnionWith(MemDomain));
          if (CanRemat)
            LoadMove->RematMI = UInstr->RealMI;
          LoadMove->Uses[0].ChoiceVar = Model.NewConstant(0);
          wireDefUse(StoreMoveDR, UseRef{LoadMove, 0});
          DefRef LoadMoveDR{LoadMove, 0};

          addDefChoice(RealUR, LoadMoveDR);

          if (RealUR.UInstr->K == UnisonInstr::LiveOutUse)
            addDefChoice(RealUR, StoreMoveDR);

          // Name after wiring so nameInstruction can walk def-use chains.
          NS.nameInstruction(LoadMove, {}, 0);
          NS.nameAllVariablesInInstruction(LoadMove);
        }

        // Name StoreMove after all LoadMoves are wired.
        NS.nameInstruction(StoreMove, {}, 0);
        NS.nameAllVariablesInInstruction(StoreMove);
      }
    }
  }
}

void Unison::addCongruenceConstraints() {
  // For each CFG edge (pred → succ), for each value live across the edge,
  // constrain: the register chosen by pred's exit operand must equal
  // succ's entry operand register.
  //
  // The exit operand (in LiveOutUse) has a ChoiceVar selecting among
  // PotentialDefs. Whichever def is chosen, its Reg.Var must equal the
  // corresponding entry def's Reg.Var in the successor's LiveInDef.

  // Build a map from MBB -> its LiveInDef and LiveOutUse instructions.
  DenseMap<MachineBasicBlock *, UnisonInstr *> MBBLiveIn, MBBLiveOut;
  for (auto &UMBB : UFunc.MBBs) {
    for (auto &UI : UMBB->Instrs) {
      if (UI->K == UnisonInstr::LiveInDef)
        MBBLiveIn[UMBB->MBB] = UI.get();
      else if (UI->K == UnisonInstr::LiveOutUse)
        MBBLiveOut[UMBB->MBB] = UI.get();
    }
  }

  // We need to match exit use operands to entry def operands by register.
  // Both getDefsFromUnisonInstr (for LiveInDef) and getUsesFromUnisonInstr
  // (for LiveOutUse) return registers in the same order for the same set
  // of live values. We iterate them in lockstep.
  for (auto &UMBB : UFunc.MBBs) {
    UnisonInstr *LiveOut = MBBLiveOut[UMBB->MBB];
    if (!LiveOut)
      continue;

    SmallVector<Register> ExitUseRegs;
    getUsesFromUnisonInstr(LiveOut, *UMBB->MBB, ExitUseRegs);

    for (MachineBasicBlock *Succ : UMBB->MBB->successors()) {
      UnisonInstr *LiveIn = MBBLiveIn[Succ];
      if (!LiveIn)
        continue;

      SmallVector<Register> EntryDefRegs;
      getDefsFromUnisonInstr(LiveIn, *Succ, EntryDefRegs);

      // Match exit uses to entry defs by register.
      for (unsigned EI = 0, EE = ExitUseRegs.size(); EI < EE; ++EI) {
        Register ExitReg = ExitUseRegs[EI];
        for (unsigned DI = 0, DE = EntryDefRegs.size(); DI < DE; ++DI) {
          if (EntryDefRegs[DI] != ExitReg)
            continue;

          UnisonUseOperand &ExitUse = LiveOut->getUse(EI);
          UnisonDefOperand &EntryDef = LiveIn->getDef(DI);

          // For each possible def choice at exit, constrain:
          // ChoiceVar == k => PotentialDefs[k].Reg.Var == EntryDef.Reg.Var
          UseRef ExitUR{LiveOut, EI};
          for (unsigned K = 0, KE = ExitUse.PotentialDefs.size(); K < KE;
               ++K) {
            DefRef DR = ExitUse.PotentialDefs[K];
            UnisonDefOperand &ChosenDef = DR.UInstr->getDef(DR.Idx);
            sat::BoolVar Chosen = getDChosenInU(DR, ExitUR);
            Model.AddEquality(ChosenDef.Reg.Var, EntryDef.Reg.Var)
                .OnlyEnforceIf(Chosen);
          }
          break; // Found match, move to next exit use.
        }
      }
    }
  }
}

sat::BoolVar Unison::reifyEquality(sat::IntVar Var, int64_t Value) {
  sat::BoolVar B = Model.NewBoolVar();
  Model.AddEquality(Var, Value).OnlyEnforceIf(B);
  Model.AddNotEqual(Var, Value).OnlyEnforceIf(~B);
  return B;
}

sat::BoolVar Unison::reifyNotEqual(sat::IntVar A, sat::IntVar B) {
  sat::BoolVar NE = Model.NewBoolVar();
  Model.AddNotEqual(A, B).OnlyEnforceIf(NE);
  Model.AddEquality(A, B).OnlyEnforceIf(~NE);
  return NE;
}

sat::BoolVar Unison::reifyAnd(sat::BoolVar X, sat::BoolVar Y) {
  sat::BoolVar R = Model.NewBoolVar();
  Model.AddBoolAnd({X, Y}).OnlyEnforceIf(R);
  Model.AddBoolOr({~X, ~Y}).OnlyEnforceIf(~R);
  return R;
}

void Unison::restrictToDomain(sat::IntVar Var,
                              const operations_research::Domain &Dom) {
  const auto &VarProto = Model.Proto().variables(Var.index());
  if (VarProto.domain_size() == 2 && VarProto.domain(0) == VarProto.domain(1))
    return;
  // CP-SAT: restrict variable to domain by adding it as a linear constraint.
  Model.AddLinearConstraint(Var, Dom);
}

void Unison::restrictToDomain(sat::IntVar Var,
                              const operations_research::Domain &Dom,
                              sat::BoolVar Condition) {
  const auto &VarProto = Model.Proto().variables(Var.index());
  if (VarProto.domain_size() == 2 && VarProto.domain(0) == VarProto.domain(1))
    return;
  Model.AddLinearConstraint(Var, Dom).OnlyEnforceIf(Condition);
}

void Unison::addConstraints() {
  for (auto &UMBB : UFunc.MBBs) {
    addRegAllocConstraints(*UMBB);
    addSchedConstraints(*UMBB);
  }
  addCongruenceConstraints();
}

// ---------------------------------------------------------------------------
// Register allocation constraints (per MBB)
// ---------------------------------------------------------------------------

void Unison::addRegAllocConstraints(UnisonMBB &UMBB) {
  // Activation must be computed before NoOverlap (which uses IsActiveVar).
  deriveActivationVars(UMBB);
  addRegClassConstraints(UMBB);
  addNoOverlapConstraints(UMBB);
}

sat::IntVar Unison::getDefLastUseCycle(UnisonInstr *UInstr,
                                          unsigned DefIdx,
                                          int IssueCycleUB) {
  UnisonDefOperand &DefOp = UInstr->Defs[DefIdx];

  if (DefOp.PotentialUses.empty()) {
    // Dead def (e.g., regmask clobber): the definer's IC is the only
    // time point. Caller adds +1 for the minimal half-open rectangle.
    return UInstr->IssueCycle;
  }

  // LastUseCycle = max over active uses of IssueCycle(use_instr).
  // Each use contributes conditionally: only if the def is actively
  // used at U (DActiveInU). Inactive uses contribute the definer's IC
  // (so that the rectangle always has at least size 1).
  std::vector<sat::IntVar> UseCycleVars;
  for (const UseRef &UR : DefOp.PotentialUses) {
    sat::IntVar UseCycle = Model.NewIntVar({0, IssueCycleUB});
    sat::BoolVar Active = getDActiveInU(DefRef{UInstr, DefIdx}, UR);
    Model.AddEquality(UseCycle, UR.UInstr->IssueCycle).OnlyEnforceIf(Active);
    Model.AddEquality(UseCycle, UInstr->IssueCycle).OnlyEnforceIf(~Active);
    UseCycleVars.push_back(UseCycle);
  }

  sat::IntVar MaxUseCycle = Model.NewIntVar({0, IssueCycleUB});
  std::vector<sat::LinearExpr> Exprs(UseCycleVars.begin(), UseCycleVars.end());
  Model.AddMaxEquality(MaxUseCycle, Exprs);
  return MaxUseCycle;
}

void Unison::addNoOverlapConstraints(UnisonMBB &UMBB) {
  UMBB.NoOverlap.emplace(Model.AddNoOverlap2D());

  for (auto &UInstrPtr : UMBB.Instrs) {
    UnisonInstr *UInstr = UInstrPtr.get();

    for (unsigned DI = 0; DI < UInstr->Defs.size(); ++DI) {
      UnisonDefOperand &DefOp = UInstr->Defs[DI];
      int UB = UMBB.IssueCycleUpperBound;

      // Rectangle starts at IC + (latency - 1), when the write completes
      // in the pipeline. For a 2-stage pipeline (latency=2), this is IC+1.
      // TODO: generalize to per-instruction latencies for real pipelines.
      static constexpr int PipelineLatency = 2;
      sat::IntVar TimeStart = Model.NewIntVar({0, UB + PipelineLatency});
      Model.AddEquality(TimeStart,
                         UInstr->IssueCycle + (PipelineLatency - 1));

      sat::IntVar LastUseCycle = getDefLastUseCycle(UInstr, DI, UB);
      // TimeEnd = max(TimeStart + 1, LastUseCycle + 1).
      sat::IntVar TimeEnd = Model.NewIntVar({0, UB + 2});
      sat::IntVar MinEnd = Model.NewIntVar({0, UB + 2});
      Model.AddEquality(MinEnd, TimeStart + 1);
      Model.AddMaxEquality(TimeEnd,
                           {sat::LinearExpr(MinEnd),
                            sat::LinearExpr(LastUseCycle) + 1});

      sat::IntVar TimeSize = Model.NewIntVar({1, UB + 2});
      Model.AddEquality(TimeSize, TimeEnd - TimeStart);

      addRectangle(UMBB, UInstr, DefOp.Reg.Var, TimeStart, TimeSize, TimeEnd);
    }
  }
}

void Unison::addRectangle(UnisonMBB &UMBB, UnisonInstr *UInstr,
                          sat::IntVar RegVar,
                          sat::IntVar TimeStart, sat::IntVar TimeSize,
                          sat::IntVar TimeEnd) {
  if (UInstr->isCopyOp()) {
    // CopyOp rectangles only participate in NoOverlap2D when active.
    // Inactive copies don't occupy any register or time slot.
    auto ActiveIt = IsActiveVar.find(UInstr);
    assert(ActiveIt != IsActiveVar.end() && "CopyOp missing IsActiveVar");
    sat::IntervalVar TimeAxis =
        Model.NewOptionalIntervalVar(TimeStart, TimeSize, TimeEnd,
                                     ActiveIt->second);
    sat::IntervalVar RegAxis =
        Model.NewOptionalFixedSizeIntervalVar(RegVar, 1, ActiveIt->second);
    UMBB.NoOverlap->AddRectangle(RegAxis, TimeAxis);
  } else {
    sat::IntervalVar TimeAxis =
        Model.NewIntervalVar(TimeStart, TimeSize, TimeEnd);
    sat::IntervalVar RegAxis =
        Model.NewFixedSizeIntervalVar(RegVar, 1);
    UMBB.NoOverlap->AddRectangle(RegAxis, TimeAxis);
  }
}

void Unison::addRegClassConstraints(UnisonMBB &UMBB) {
  operations_research::Domain PhysRegDomain(0, NumPhysRegs - 1);

  for (auto &UInstrPtr : UMBB.Instrs) {
    UnisonInstr *UInstr = UInstrPtr.get();

    // --- Def-side constraints ---
    if (UInstr->K == UnisonInstr::RealInstr) {
      for (UnisonDefOperand &DefOp : UInstr->Defs)
        restrictToDomain(DefOp.Reg.Var, DefOp.Reg.Dom);
    } else if (UInstr->isCopyOp()) {
      UnisonDefOperand &DefOp = UInstr->Defs[0];
      operations_research::Domain RegDom =
          DefOp.Reg.Dom.IntersectionWith(PhysRegDomain);

      for (unsigned I = 0; I < UInstr->AltOpcodes.size(); ++I) {
        sat::BoolVar InsIsI = reifyEquality(UInstr->Ins, I);
        unsigned Opcode = UInstr->AltOpcodes[I];
        if (Opcode == COPY_STORE)
          restrictToDomain(DefOp.Reg.Var, MemDomain, InsIsI);
        else
          restrictToDomain(DefOp.Reg.Var, RegDom, InsIsI);
      }
    }

    // --- Use-side constraints: propagate source domain from uses to defs ---
    // For each def, intersect its domain with the requirements of all its
    // users. A real instruction use requires its source in register domain.
    // A CopyOp use's requirement depends on the instruction alternative.
    // A LiveOutUse has no restriction (synthetic, can accept register or memory).
    for (unsigned DI = 0; DI < UInstr->Defs.size(); ++DI) {
      UnisonDefOperand &DefOp = UInstr->Defs[DI];
      sat::IntVar DefReg = DefOp.Reg.Var;

      for (const UseRef &UR : DefOp.PotentialUses) {
        UnisonInstr *User = UR.UInstr;
        sat::BoolVar Chosen = getDChosenInU(DefRef{UInstr, DI}, UR);

        if (User->K == UnisonInstr::RealInstr) {
          // Real instructions read from registers only.
          restrictToDomain(DefReg, PhysRegDomain, Chosen);

        } else if (User->isCopyOp()) {
          // CopyOp source requirement depends on instruction alternative.
          for (unsigned I = 0; I < User->AltOpcodes.size(); ++I) {
            sat::BoolVar InsIsI = reifyEquality(User->Ins, I);
            sat::BoolVar Both = reifyAnd(Chosen, InsIsI);
            unsigned Opcode = User->AltOpcodes[I];
            if (Opcode == COPY_LOAD) {
              restrictToDomain(DefReg, MemDomain, Both);
            } else if (Opcode == COPY_STORE || Opcode == COPY_MOVE) {
              restrictToDomain(DefReg, PhysRegDomain, Both);
            }
            // COPY_REMAT: no source constraint (rematerialized).
          }
        }
        // LiveOutUse/LiveInDef: no restriction on source domain.
      }
    }
  }
}

// Compute activation for a single def operand. Returns TrueVar if
// trivially always active, FalseVar if no uses, or a BoolVar for
// the OR of DActiveInU over all potential uses.
sat::BoolVar Unison::deriveDefActivation(UnisonInstr *UInstr, unsigned DI) {
  UnisonDefOperand &DefOp = UInstr->Defs[DI];

  if (DefOp.PotentialUses.empty())
    return Model.FalseVar();

  SmallVector<sat::BoolVar, 4> ActiveBools;

  for (const UseRef &UR : DefOp.PotentialUses) {
    UnisonUseOperand &UseOp = UR.UInstr->getUse(UR.Idx);
    bool SingleChoice = (UseOp.PotentialDefs.size() == 1);
    bool UseAlwaysActive = UR.UInstr->isAlwaysActive();

    // Short-circuit: unconditionally chosen + always-active use.
    if (SingleChoice && UseAlwaysActive)
      return Model.TrueVar();

    ActiveBools.push_back(getDActiveInU(DefRef{UInstr, DI}, UR));
  }

  if (ActiveBools.empty())
    return Model.FalseVar();
  if (ActiveBools.size() == 1)
    return ActiveBools[0];

  sat::BoolVar Active = Model.NewBoolVar();
  for (sat::BoolVar B : ActiveBools)
    Model.AddImplication(B, Active);
  Model.AddBoolOr(ActiveBools).OnlyEnforceIf(Active);
  std::vector<sat::BoolVar> NegBools;
  for (sat::BoolVar B : ActiveBools)
    NegBools.push_back(~B);
  Model.AddBoolAnd(NegBools).OnlyEnforceIf(~Active);
  return Active;
}

void Unison::deriveActivationVars(UnisonMBB &UMBB) {
  // Process in reverse order: load-moves before store-moves, so that
  // IsActiveVar is available for nested activation (store-move activation
  // depends on load-move activation).
  for (auto It = UMBB.Instrs.rbegin(); It != UMBB.Instrs.rend(); ++It) {
    UnisonInstr *UInstr = It->get();
    if (!UInstr->isCopyOp())
      continue;

    // Instruction is active if any of its defs is active.
    SmallVector<sat::BoolVar, 2> DefActives;
    for (unsigned DI = 0; DI < UInstr->Defs.size(); ++DI) {
      sat::BoolVar DA = deriveDefActivation(UInstr, DI);
      DefActives.push_back(DA);
    }

    if (DefActives.empty()) {
      IsActiveVar[UInstr] = Model.FalseVar();
    } else if (DefActives.size() == 1) {
      IsActiveVar[UInstr] = DefActives[0];
    } else {
      sat::BoolVar Active = Model.NewBoolVar();
      for (sat::BoolVar B : DefActives)
        Model.AddImplication(B, Active);
      Model.AddBoolOr(DefActives).OnlyEnforceIf(Active);
      std::vector<sat::BoolVar> NegBools;
      for (sat::BoolVar B : DefActives)
        NegBools.push_back(~B);
      Model.AddBoolAnd(NegBools).OnlyEnforceIf(~Active);
      IsActiveVar[UInstr] = Active;
    }
    NS.nameActiveVar(UInstr, IsActiveVar[UInstr]);
  }
}

// ---------------------------------------------------------------------------
// Scheduling constraints (per MBB)
// ---------------------------------------------------------------------------

void Unison::addSchedConstraints(UnisonMBB &UMBB) {
  addDataDependencyConstraints(UMBB);
  addAntiDependencyConstraints(UMBB);
  addOrderingConstraints(UMBB);
}

void Unison::addDataDependencyConstraints(UnisonMBB &UMBB) {
  // Latency = 2 for the 2-stage pipeline: a def at IC=K writes at
  // stage 1 (time K+1). A use must be at IC >= K+2 so it reads at
  // stage 0 (time K+2), one time unit after the write completes.
  // This prevents write/read boundary collisions at the same time
  // point.
  // // TODO: generalize latency per instruction for real pipeline models.
  static constexpr int Latency = 2;
  for (auto &UInstrPtr : UMBB.Instrs) {
    UnisonInstr *UInstr = UInstrPtr.get();
    for (unsigned DI = 0; DI < UInstr->Defs.size(); ++DI) {
      for (const UseRef &UR : UInstr->Defs[DI].PotentialUses) {
        sat::BoolVar Chosen = getDChosenInU(DefRef{UInstr, DI}, UR);
        Model.AddGreaterOrEqual(UR.UInstr->IssueCycle,
                                UInstr->IssueCycle + Latency)
            .OnlyEnforceIf(Chosen);
      }
    }
  }
}

void Unison::addAntiDependencyConstraints(UnisonMBB &UMBB) {
  // WAR (Write After Read) anti-dependencies: if instruction A reads
  // a register variable V, and a later instruction B (in original order)
  // defines the same variable V, then IC(A) < IC(B).
  //
  // Walk instructions in original order. For each use, record the
  // Reg.Var it reads from. For each def, check if any earlier use
  // read from the same Reg.Var.

  // Map from Reg.Var index to the instructions that read it.
  // We use the IntVar's index as the key.
  DenseMap<int, SmallVector<UnisonInstr *, 4>> ReadersOfVar;

  for (auto &UInstrPtr : UMBB.Instrs) {
    UnisonInstr *UInstr = UInstrPtr.get();

    // First: check defs against prior readers (anti-dependency).
    for (unsigned DI = 0; DI < UInstr->Defs.size(); ++DI) {
      int VarIdx = UInstr->Defs[DI].Reg.Var.index();
      auto It = ReadersOfVar.find(VarIdx);
      if (It != ReadersOfVar.end()) {
        for (UnisonInstr *Reader : It->second) {
          if (Reader == UInstr)
            continue;
          // TODO: only add antidependency if both insturctions are active.
          Model.AddLessOrEqual(Reader->IssueCycle, UInstr->IssueCycle);
        }
      }
    }

    // Then: record this instruction's reads.
    // Each use reads from its potential defs' Reg.Var. Since the
    // choice is a solver variable, we conservatively record ALL
    // potential source variables.
    for (unsigned UI = 0; UI < UInstr->Uses.size(); ++UI) {
      UnisonUseOperand &UseOp = UInstr->Uses[UI];
      for (auto &DR : UseOp.PotentialDefs) {
        int VarIdx = DR.UInstr->getDef(DR.Idx).Reg.Var.index();
        ReadersOfVar[VarIdx].push_back(UInstr);
      }
    }
  }
}

void Unison::addOrderingConstraints(UnisonMBB &UMBB) {
  // Ordering constraints partition the MBB into regions separated by
  // scheduling barriers (isSchedulingBoundary: stores, calls, SP mods,
  // terminators). Within a region, pure instructions may be freely
  // reordered. Across regions, order is preserved:
  //
  //   IC(barrier_k) > max(IC of all instrs since barrier_{k-1})
  //
  // Terminators are barriers, so they are naturally chained in program
  // order. LiveOutUse is constrained after everything in the block.
  //
  // When UnisonPreserveOrder is on, these constraints are redundant
  // with the program-order chains but harmless.

  int UB = UMBB.IssueCycleUpperBound;

  // Track ICs since last barrier. When we hit a barrier, constrain it
  // after all of them, then reset.
  SmallVector<sat::IntVar, 16> ICsSinceLastBarrier;
  sat::IntVar LastBarrierIC;
  bool HaveBarrier = false;

  // Process all instructions in list order (includes CopyOps).
  // RealInstrs that are scheduling boundaries act as barriers.
  // CopyOps and other non-barrier instructions are pure — they must
  // stay within their barrier region.
  for (auto &UIP : UMBB.Instrs) {
    UnisonInstr *UI = UIP.get();
    if (UI->K == UnisonInstr::LiveInDef || UI->K == UnisonInstr::LiveOutUse)
      continue;

    bool IsBarrier = UI->K == UnisonInstr::RealInstr && UI->RealMI &&
                     A.TII->isSchedulingBoundary(*UI->RealMI, UMBB.MBB, MF);

    if (IsBarrier) {
      // This barrier must come after all instructions since the last barrier.
      if (!ICsSinceLastBarrier.empty()) {
        sat::IntVar MaxPrev = Model.NewIntVar({0, UB});
        std::vector<sat::LinearExpr> Exprs(ICsSinceLastBarrier.begin(),
                                            ICsSinceLastBarrier.end());
        Model.AddMaxEquality(MaxPrev, Exprs);
        Model.AddGreaterThan(UI->IssueCycle, MaxPrev);
      } else if (HaveBarrier) {
        Model.AddGreaterThan(UI->IssueCycle, LastBarrierIC);
      }
      LastBarrierIC = UI->IssueCycle;
      HaveBarrier = true;
      ICsSinceLastBarrier.clear();
    } else {
      // Pure instruction (including CopyOps): must come after the last barrier.
      if (HaveBarrier)
        Model.AddGreaterThan(UI->IssueCycle, LastBarrierIC);
      ICsSinceLastBarrier.push_back(UI->IssueCycle);
    }
  }

  // LiveOutUse must come after all real instructions in the block.
  SmallVector<sat::IntVar, 16> AllPrevICs(ICsSinceLastBarrier);
  if (HaveBarrier)
    AllPrevICs.push_back(LastBarrierIC);

  if (!AllPrevICs.empty()) {
    sat::IntVar MaxAll = Model.NewIntVar({0, UB});
    std::vector<sat::LinearExpr> Exprs(AllPrevICs.begin(), AllPrevICs.end());
    Model.AddMaxEquality(MaxAll, Exprs);
    for (auto &UIP : UMBB.Instrs) {
      if (UIP->K == UnisonInstr::LiveOutUse)
        Model.AddGreaterThan(UIP->IssueCycle, MaxAll);
    }
  }
}

sat::BoolVar Unison::getIsNotIdentityCopy(UnisonInstr *UInstr) {
  // TODO: what if copy has several defs feeding into it?
  UnisonUseOperand &UseOp = UInstr->Uses[0];
  assert(!UseOp.PotentialDefs.empty());
  DefRef SrcDR = UseOp.PotentialDefs[0];
  return reifyNotEqual(SrcDR.UInstr->getDef(SrcDR.Idx).Reg.Var,
                       UInstr->Defs[0].Reg.Var);
}

void Unison::penalizeCopies(sat::LinearExpr &Objective) {
  static constexpr int64_t MemOpWeight = 10;
  static constexpr int64_t RematWeight = 1;
  static constexpr int64_t CopyWeight = 1;
  static constexpr int64_t MaxFreq = 1000000;

  for (auto &UMBB : UFunc.MBBs) {
    int64_t Freq = std::min<int64_t>(
        std::max<int64_t>(A.MBFI->getBlockFreq(UMBB->MBB).getFrequency(), 1),
        MaxFreq);

    for (auto &UInstrPtr : UMBB->Instrs) {
      UnisonInstr *UInstr = UInstrPtr.get();

      // Real COPY instructions: penalize non-identity copies.
      if (UInstr->K == UnisonInstr::RealInstr &&
          UInstr->RealMI->isCopy() && !UInstr->Uses.empty()) {
        Objective += Freq * CopyWeight * getIsNotIdentityCopy(UInstr);
        continue;
      }

      // CopyOps: penalize based on opcode.
      if (!UInstr->isCopyOp())
        continue;
      auto ActiveIt = IsActiveVar.find(UInstr);
      assert(ActiveIt != IsActiveVar.end());
      sat::BoolVar Active = ActiveIt->second;

      for (unsigned I = 0; I < UInstr->AltOpcodes.size(); ++I) {
        unsigned Opcode = UInstr->AltOpcodes[I];
        sat::BoolVar IsOpcChosen = reifyEquality(UInstr->Ins, I);
        sat::BoolVar ShouldIncludeInCost = reifyAnd(IsOpcChosen, Active);

        if (Opcode == COPY_STORE || Opcode == COPY_LOAD) {
          Objective += Freq * MemOpWeight * ShouldIncludeInCost;
        } else if (Opcode == COPY_REMAT) {
          Objective += Freq * RematWeight * ShouldIncludeInCost;
        } else {
          // COPY_MOVE: only penalize non-identity.
          sat::BoolVar NotIdentity = getIsNotIdentityCopy(UInstr);
          ShouldIncludeInCost = reifyAnd(ShouldIncludeInCost, NotIdentity);
          Objective += Freq * CopyWeight * ShouldIncludeInCost;
        }
      }
    }
  }
}

DenseSet<int> Unison::getPenalizedCSRIndices() const {
  // Early-save CSRs are handled by the RA through vregs — no PEI
  // penalty. Only non-early CSRs go through PEI and need penalizing.
  const TargetFrameLowering &TFI =
      *MF.getSubtarget().getFrameLowering();
  BitVector EarlyCSRs(A.TRI->getNumRegs());
  TFI.determineEarlyCalleeSaves(const_cast<MachineFunction &>(MF),
                                EarlyCSRs);

  // Collect non-early CSRs that are in our register index.
  const MCPhysReg *CSRegs = A.TRI->getCalleeSavedRegs(&MF);
  SmallVector<std::pair<MCRegister, int>, 16> NonEarlyCSRs;
  for (unsigned I = 0; CSRegs[I]; ++I) {
    MCRegister PhysReg(CSRegs[I]);
    if (EarlyCSRs.test(PhysReg))
      continue;
    auto It = MCRegToIdx.find(PhysReg);
    if (It != MCRegToIdx.end())
      NonEarlyCSRs.push_back({PhysReg, It->second});
  }

  if (NonEarlyCSRs.empty())
    return {};

  // Among non-early CSRs, find those not already modified.
  // Modified ones are saved by PEI regardless — free to use.
  BitVector ModifiedPhysRegs(A.TRI->getNumRegs());
  for (const MachineBasicBlock &MBB : MF)
    for (const MachineInstr &MI : MBB)
      for (const MachineOperand &MO : MI.operands()) {
        if (MO.isReg() && MO.isDef() && MO.getReg().isPhysical())
          ModifiedPhysRegs.set(MO.getReg());
        if (MO.isRegMask())
          ModifiedPhysRegs.setBitsNotInMask(MO.getRegMask());
      }

  DenseSet<int> Result;
  for (auto &[PhysReg, Idx] : NonEarlyCSRs) {
    if (!ModifiedPhysRegs.test(PhysReg))
      Result.insert(Idx);
  }
  return Result;
}

void Unison::penalizeCalleeSavedRegisters(sat::LinearExpr &Objective) {
  static constexpr int64_t MaxFreq = 1000000;

  // CSR save/restore cost is 2 instructions (sd + ld) per function
  // invocation. Weight by entry block frequency.
  int64_t EntryFreq = std::min<int64_t>(
      std::max<int64_t>(
          A.MBFI->getBlockFreq(&MF.front()).getFrequency(), 1),
      MaxFreq);
  // Each CSR save/restore pair costs ~2 instructions.
  static constexpr int64_t CSRWeight = 20;

  DenseSet<int> PenalizedCSRIndices = getPenalizedCSRIndices();

  // Penalize each unmodified CSR that has at least one vreg assigned.
  for (int Idx : PenalizedCSRIndices) {
    SmallVector<sat::BoolVar, 8> VRegAtIdx;
    for (auto &[Reg, PhysRegVar] : VRegToPhysRegVar)
      VRegAtIdx.push_back(reifyEquality(PhysRegVar, Idx));
    if (VRegAtIdx.empty())
      continue;
    sat::BoolVar CSRUsed = Model.NewBoolVar();
    Model.AddBoolOr(VRegAtIdx).OnlyEnforceIf(CSRUsed);
    for (auto &B : VRegAtIdx)
      Model.AddEquality(B, false).OnlyEnforceIf(~CSRUsed);
    Objective += EntryFreq * CSRWeight * CSRUsed;
  }
}

void Unison::addMinimizeMakespanObjective(sat::LinearExpr &Objective) {
  static constexpr int64_t MaxFreq = 1000000;
  for (auto &UMBB : UFunc.MBBs) {
    int64_t Freq = std::min<int64_t>(
        std::max<int64_t>(A.MBFI->getBlockFreq(UMBB->MBB).getFrequency(), 1),
        MaxFreq);
    for (auto &UInstrPtr : UMBB->Instrs) {
      if (UInstrPtr->K == UnisonInstr::LiveOutUse)
        Objective += Freq * UInstrPtr->IssueCycle;
    }
  }
}

void Unison::addObjectiveFunction() {
  sat::LinearExpr Objective;

  penalizeCopies(Objective);
  penalizeCalleeSavedRegisters(Objective);
  addMinimizeMakespanObjective(Objective);

  Model.Minimize(Objective);
}

void Unison::solve() {
  if (UnisonDumpModel)
    dbgs() << "# Unison model for: " << MF.getName() << "\n"
           << Model.Proto().DebugString();

  // Validate model before solving.
  {
    std::string err = operations_research::sat::ValidateCpModel(Model.Proto());
    if (!err.empty()) {
      LLVM_DEBUG(dbgs() << "  Model validation error: " << err << "\n");
      report_fatal_error("Unison: invalid CP-SAT model for " + MF.getName() +
                         ": " + err);
    }
  }

  sat::SatParameters Params;
  // Scale timeout by function size: more instructions → more time.
  unsigned TotalInstrs = 0;
  for (auto &UMBB : UFunc.MBBs)
    TotalInstrs += UMBB->Instrs.size();
  int TimeLimit = std::clamp(
      static_cast<int>(TotalInstrs) * UnisonTimeLimitPerInstr,
      static_cast<int>(UnisonMinTimeLimit),
      static_cast<int>(UnisonMaxTimeLimit));
  LLVM_DEBUG(dbgs() << "  Solver time limit: " << TimeLimit
                    << "s (" << TotalInstrs << " instrs)\n");
  Params.set_max_time_in_seconds(static_cast<double>(TimeLimit));
  Params.set_num_workers(UnisonNumWorkers);

  sat::CpSolverResponse Response =
      sat::SolveWithParameters(Model.Build(), Params);

  LLVM_DEBUG(dbgs() << "  CP-SAT status: " << Response.status()
                    << " (wall time " << Response.wall_time() << "s)\n");

  if (Response.status() != sat::CpSolverStatus::OPTIMAL &&
      Response.status() != sat::CpSolverStatus::FEASIBLE) {
    LLVM_DEBUG(dbgs() << "  Model has " << Model.Proto().variables_size()
                      << " variables, "
                      << Model.Proto().constraints_size()
                      << " constraints\n");
    const char *StatusStr = "unknown";
    if (Response.status() == sat::CpSolverStatus::MODEL_INVALID)
      StatusStr = "model invalid";
    else if (Response.status() == sat::CpSolverStatus::INFEASIBLE)
      StatusStr = "infeasible";
    report_fatal_error(Twine("Unison CP-SAT: ") + StatusStr + " for " +
                       MF.getName());
  }

  // Store response for use by generateCodeFromSolution.
  SolverResponse = Response;
}

void Unison::dumpSolution(StringRef Filename) {
  std::error_code EC;
  raw_fd_ostream OS(Filename, EC);
  if (EC) {
    errs() << "Unison: cannot open " << Filename << ": " << EC.message() << "\n";
    return;
  }

  const auto &Response = SolverResponse;
  OS << "# Solution for " << MF.getName() << "\n";
  for (auto &UMBB : UFunc.MBBs) {
    OS << "# MBB#" << UMBB->MBB->getNumber() << "\n";
    for (auto &UIP : UMBB->Instrs) {
      UnisonInstr *UI = UIP.get();
      OS << NamingScheme::nameVariable(UI->Name, "ic") << " = "
         << sat::SolutionIntegerValue(Response, UI->IssueCycle) << "\n";
      for (unsigned I = 0; I < UI->Defs.size(); I++)
        OS << NamingScheme::nameVariable(UI->Name,
               "def[" + Twine(I) + "].reg") << " = "
           << sat::SolutionIntegerValue(Response, UI->Defs[I].Reg.Var)
           << "\n";
      for (unsigned I = 0; I < UI->Uses.size(); I++)
        OS << NamingScheme::nameVariable(UI->Name,
               "use[" + Twine(I) + "].choice") << " = "
           << sat::SolutionIntegerValue(Response, UI->Uses[I].ChoiceVar)
           << "\n";
      if (UI->isCopyOp()) {
        OS << NamingScheme::nameVariable(UI->Name, "ins") << " = "
           << sat::SolutionIntegerValue(Response, UI->Ins) << "\n";
        auto ActiveIt = IsActiveVar.find(UI);
        if (ActiveIt != IsActiveVar.end())
          OS << NamingScheme::nameVariable(UI->Name, "active") << " = "
             << sat::SolutionBooleanValue(Response, ActiveIt->second)
             << "\n";
      }
    }
  }
}

void Unison::loadPreAssignments(StringRef Filename) {
  auto BufOrErr = MemoryBuffer::getFile(Filename);
  if (!BufOrErr) {
    report_fatal_error(Twine("Unison: cannot open pre-assignment file: ") +
                       Filename);
  }
  SmallVector<StringRef> Lines;
  BufOrErr.get()->getBuffer().split(Lines, '\n');
  unsigned LineNo = 0;
  for (StringRef Line : Lines) {
    LineNo++;
    Line = Line.trim();
    if (Line.empty() || Line.starts_with("#"))
      continue;
    auto [Name, ValStr] = Line.split('=');
    Name = Name.trim();
    ValStr = ValStr.trim();
    int64_t Value;
    if (ValStr.getAsInteger(10, Value)) {
      errs() << "Unison: " << Filename << ":" << LineNo
             << ": bad value '" << ValStr << "'\n";
      continue;
    }
    auto Var = NS.lookupVar(Name);
    if (!Var) {
      errs() << "Unison: " << Filename << ":" << LineNo
             << ": unknown variable '" << Name << "'\n";
      continue;
    }
    Model.AddEquality(*Var, Value);
  }
}

void Unison::generateCodeFromSolution() {
  removeInactiveInstructions();
  sortByIssueCycle();

  // Dump solution for debugging non-determinism issues.
  if (UnisonDumpModel) {
    const auto &Response = SolverResponse;
    errs() << "=== Solution for " << MF.getName() << " ===\n";
    for (auto &UMBB : UFunc.MBBs) {
      errs() << "MBB#" << UMBB->MBB->getNumber() << ":\n";
      for (auto &UIP : UMBB->Instrs) {
        int64_t IC = sat::SolutionIntegerValue(Response, UIP->IssueCycle);
        const char *K = UIP->K == UnisonInstr::LiveInDef ? "LI" :
                        UIP->K == UnisonInstr::LiveOutUse ? "LO" :
                        UIP->K == UnisonInstr::RealInstr ? "RI" : "CO";
        errs() << "  IC=" << IC << " " << K
               << " [" << UIP->Name << "]";
        if (UIP->isCopyOp()) {
          int64_t InsV = sat::SolutionIntegerValue(Response, UIP->Ins);
          unsigned Opc = UIP->AltOpcodes[static_cast<unsigned>(InsV)];
          const char *OpcStr = Opc == 0 ? "ST" : Opc == 1 ? "MV" :
                               Opc == 2 ? "LD" : "RM";
          errs() << "(" << OpcStr << ")";
        }
        for (unsigned D = 0; D < UIP->Defs.size(); ++D) {
          int64_t Idx = sat::SolutionIntegerValue(Response,
                            UIP->Defs[D].Reg.Var);
          errs() << " d" << D << "=" << Idx;
        }
        for (unsigned U = 0; U < UIP->Uses.size(); ++U) {
          UnisonUseOperand &UseOp = UIP->Uses[U];
          int64_t Ch = sat::SolutionIntegerValue(Response, UseOp.ChoiceVar);
          DefRef DR = UseOp.PotentialDefs[static_cast<unsigned>(Ch)];
          int64_t SrcIdx = sat::SolutionIntegerValue(Response,
                               DR.UInstr->getDef(DR.Idx).Reg.Var);
          errs() << " u" << U << "=" << SrcIdx
                 << "(ch" << Ch << " " << DR.UInstr->Name
                 << ".d" << DR.Idx << ")";
        }
        if (UIP->RealMI)
          errs() << "  " << *UIP->RealMI;
        else
          errs() << "\n";
      }
    }
  }

  generateInstructions();
}

void Unison::removeInactiveInstructions() {
  const auto &Response = SolverResponse;
  for (auto &UMBB : UFunc.MBBs) {
    UMBB->Instrs.remove_if(
        [&](const std::unique_ptr<UnisonInstr> &UIP) {
          if (!UIP->isCopyOp())
            return false;
          auto AIt = IsActiveVar.find(UIP.get());
          assert(AIt != IsActiveVar.end());
          return !sat::SolutionBooleanValue(Response, AIt->second);
        });
  }
}

void Unison::sortByIssueCycle() {
  const auto &Response = SolverResponse;
  for (auto &UMBB : UFunc.MBBs) {
    UMBB->Instrs.sort(
        [&](const std::unique_ptr<UnisonInstr> &A,
            const std::unique_ptr<UnisonInstr> &B) {
          int64_t ICA = sat::SolutionIntegerValue(Response, A->IssueCycle);
          int64_t ICB = sat::SolutionIntegerValue(Response, B->IssueCycle);
          return ICA < ICB;
        });
  }
}

MachineInstr *Unison::materializeCopyOp(UnisonInstr *UInstr,
                                         MachineBasicBlock *MBB,
                                         MachineBasicBlock::iterator InsertPt) {
  const auto &Response = SolverResponse;
  int64_t InsVal = sat::SolutionIntegerValue(Response, UInstr->Ins);
  unsigned Opcode = UInstr->AltOpcodes[static_cast<unsigned>(InsVal)];
  int64_t DstIdx = sat::SolutionIntegerValue(Response,
                                              UInstr->Defs[0].Reg.Var);
  UnisonUseOperand &UseOp = UInstr->Uses[0];
  int64_t Choice = sat::SolutionIntegerValue(Response, UseOp.ChoiceVar);
  DefRef SrcDR = UseOp.PotentialDefs[static_cast<unsigned>(Choice)];
  int64_t SrcIdx = sat::SolutionIntegerValue(Response,
                       SrcDR.UInstr->getDef(SrcDR.Idx).Reg.Var);

  // Get or create a stack slot for a memory index from the solver.
  auto getOrCreateStackSlot = [&](int64_t MemIdx,
                                  const TargetRegisterClass *RC) -> int {
    int Key = static_cast<int>(MemIdx);
    auto It = MemIdxToFrameIdx.find(Key);
    if (It != MemIdxToFrameIdx.end())
      return It->second;
    int FI = MF.getFrameInfo().CreateSpillStackObject(
        A.TRI->getSpillSize(*RC), A.TRI->getSpillAlign(*RC));
    MemIdxToFrameIdx[Key] = FI;
    return FI;
  };

  if (Opcode == COPY_STORE) {
    assert(DstIdx >= NumPhysRegs);
    MCRegister SrcPhys = IdxToMCReg[static_cast<int>(SrcIdx)];
    const TargetRegisterClass *RC = A.TRI->getMinimalPhysRegClass(SrcPhys);
    int FI = getOrCreateStackSlot(DstIdx, RC);

    A.TII->storeRegToStackSlot(*MBB, InsertPt, SrcPhys, true,
                               FI, RC, SrcPhys);
    ++NumSpillStores;
    return &*std::prev(InsertPt);

  } else if (Opcode == COPY_LOAD) {
    assert(DstIdx < NumPhysRegs);
    MCRegister DstPhys = IdxToMCReg[static_cast<int>(DstIdx)];
    const TargetRegisterClass *RC = A.TRI->getMinimalPhysRegClass(DstPhys);
    int FI = getOrCreateStackSlot(SrcIdx, RC);

    A.TII->loadRegFromStackSlot(*MBB, InsertPt, DstPhys,
                                FI, RC, DstPhys);
    ++NumSpillLoads;
    return &*std::prev(InsertPt);

  } else if (Opcode == COPY_REMAT) {
    assert(DstIdx < NumPhysRegs);
    assert(UInstr->RematMI && "COPY_REMAT without RematMI");
    MCRegister DstPhys = IdxToMCReg[static_cast<int>(DstIdx)];

    MachineInstr *Clone = MF.CloneMachineInstr(UInstr->RematMI);
    // Rewrite all operands to physregs.
    for (MachineOperand &MO : Clone->operands()) {
      if (!MO.isReg()) continue;
      if (MO.isDef())
        MO.setReg(DstPhys);
      // Use operands should already be physregs (remat instrs use
      // constants or always-available regs like $x0).
    }
    MBB->insert(InsertPt, Clone);
    return Clone;

  } else {
    // COPY_MOVE
    assert(SrcIdx < NumPhysRegs && DstIdx < NumPhysRegs);
    MCRegister SrcPhys = IdxToMCReg[static_cast<int>(SrcIdx)];
    MCRegister DstPhys = IdxToMCReg[static_cast<int>(DstIdx)];
    if (UnisonDumpModel)
      errs() << "  COPY_MOVE: " << printReg(SrcPhys, A.TRI) << " -> "
             << printReg(DstPhys, A.TRI)
             << (SrcPhys == DstPhys ? " (identity, skip)\n" : "\n");
    if (SrcPhys == DstPhys)
      return nullptr;

    return BuildMI(*MBB, InsertPt, DebugLoc(),
                   A.TII->get(TargetOpcode::COPY), DstPhys)
               .addReg(SrcPhys);
  }
}

void Unison::generateInstructions() {
  const auto &Response = SolverResponse;

  // Rewrite a real instruction's operands to physregs using the
  // solver solution. Defs get the physreg from their Reg.Var.
  // Uses get the physreg from their chosen def's Reg.Var.
  auto rewriteRealInstr = [&](UnisonInstr *UI) {
    MachineInstr *MI = UI->RealMI;
    assert(MI);

    // Rewrite defs: each def's Reg.Var gives the physreg.
    SmallVector<CanonicalOperand, 8> CanonDefs;
    getMIDefsInCanonicalOrder(*MI, CanonDefs);
    for (unsigned I = 0; I < UI->Defs.size() && I < CanonDefs.size(); ++I) {
      if (!CanonDefs[I].MO || CanonDefs[I].Reg.isPhysical())
        continue;
      int64_t Idx = sat::SolutionIntegerValue(Response, UI->Defs[I].Reg.Var);
      if (Idx < NumPhysRegs)
        CanonDefs[I].MO->setReg(IdxToMCReg[static_cast<int>(Idx)]);
    }

    // Rewrite uses: each use's chosen def's Reg.Var gives the physreg.
    // Both virtual AND physical register uses are rewritten — a CopyOp
    // may redirect a physical register use to a different register.
    SmallVector<CanonicalOperand, 8> CanonUses;
    getMIUsesInCanonicalOrder(*MI, CanonUses);
    for (unsigned I = 0; I < UI->Uses.size() && I < CanonUses.size(); ++I) {
      if (!CanonUses[I].MO)
        continue;
      UnisonUseOperand &UseOp = UI->Uses[I];
      int64_t Choice = sat::SolutionIntegerValue(Response, UseOp.ChoiceVar);
      DefRef DR = UseOp.PotentialDefs[static_cast<unsigned>(Choice)];
      int64_t Idx = sat::SolutionIntegerValue(Response,
                        DR.UInstr->getDef(DR.Idx).Reg.Var);
      if (Idx < NumPhysRegs)
        CanonUses[I].MO->setReg(IdxToMCReg[static_cast<int>(Idx)]);
    }
  };

  for (auto &UMBB : UFunc.MBBs) {
    MachineBasicBlock *MBB = UMBB->MBB;

    // Remove all instructions from the MBB.
    SmallVector<MachineInstr *, 32> ToRemove;
    for (MachineInstr &MI : *MBB)
      ToRemove.push_back(&MI);
    for (MachineInstr *MI : ToRemove)
      MI->removeFromParent();

    // Update live-in list with physregs.
    MBB->clearLiveIns();
    for (auto &UIP : UMBB->Instrs) {
      if (UIP->K != UnisonInstr::LiveInDef)
        continue;
      for (unsigned I = 0; I < UIP->Defs.size(); ++I) {
        int64_t Idx = sat::SolutionIntegerValue(Response,
                          UIP->Defs[I].Reg.Var);
        if (Idx < NumPhysRegs)
          MBB->addLiveIn(IdxToMCReg[static_cast<int>(Idx)]);
      }
    }
    MBB->sortUniqueLiveIns();

    // Emit all instructions in IC order.
    for (auto &UIP : UMBB->Instrs) {
      UnisonInstr *UI = UIP.get();

      if (UI->K == UnisonInstr::LiveInDef ||
          UI->K == UnisonInstr::LiveOutUse)
        continue;

      if (UI->K == UnisonInstr::RealInstr) {
        assert(UI->RealMI);
        rewriteRealInstr(UI);
        MBB->push_back(UI->RealMI);
      } else if (UI->isCopyOp()) {
        materializeCopyOp(UI, MBB, MBB->end());
      }
    }
  }
}

bool Unison::run() {
  LLVM_DEBUG(dbgs() << "Unison CP-SAT Register Allocating for "
                    << MF.getName() << "\n");

  buildURegisterDomains();
  buildUnisonInstructionsAndAnalyzeDefs();
  copyExtend();
  addConstraints();
  addObjectiveFunction();
  if (!UnisonPreAssign.empty())
    loadPreAssignments(UnisonPreAssign);
  solve();
  if (!UnisonDumpSolution.empty())
    dumpSolution(UnisonDumpSolution);
  generateCodeFromSolution();

  return true;
}

// ---------------------------------------------------------------------------
// Pass definition (thin wrapper)
// ---------------------------------------------------------------------------

namespace llvm {
void initializeRegAllocUnisonPassPass(PassRegistry &);
}

namespace {

class RegAllocUnisonPass : public MachineFunctionPass {
public:
  static char ID;
  RegAllocUnisonPass() : MachineFunctionPass(ID) {
    initializeRegAllocUnisonPassPass(*PassRegistry::getPassRegistry());
  }

  StringRef getPassName() const override {
    return "Unison CP-SAT Register Allocator";
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override;
  bool runOnMachineFunction(MachineFunction &MF) override;

  MachineFunctionProperties getRequiredProperties() const override {
    return MachineFunctionProperties().setNoPHIs();
  }

  MachineFunctionProperties getSetProperties() const override {
    return MachineFunctionProperties().setNoVRegs();
  }

  MachineFunctionProperties getClearedProperties() const override {
    return MachineFunctionProperties().setIsSSA();
  }
};

char RegAllocUnisonPass::ID = 0;

} // end anonymous namespace

INITIALIZE_PASS_BEGIN(RegAllocUnisonPass, "regallocunison",
                      "Unison CP-SAT Register Allocator", false, false)
INITIALIZE_PASS_DEPENDENCY(SlotIndexesWrapperPass)
INITIALIZE_PASS_DEPENDENCY(LiveIntervalsWrapperPass)
INITIALIZE_PASS_DEPENDENCY(MachineBlockFrequencyInfoWrapperPass)
INITIALIZE_PASS_END(RegAllocUnisonPass, "regallocunison",
                    "Unison CP-SAT Register Allocator", false, false)

static FunctionPass *createUnisonRegisterAllocator() {
  return new RegAllocUnisonPass();
}

static RegisterRegAlloc UnisonRegAlloc("unison",
                                       "Unison CP-SAT register allocator",
                                       createUnisonRegisterAllocator);

// Register the pass for -run-pass=regallocunison.
static struct ForcePassInit {
  ForcePassInit() {
    initializeRegAllocUnisonPassPass(*PassRegistry::getPassRegistry());
  }
} ForcePassInitObj;

void RegAllocUnisonPass::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.setPreservesCFG();
  AU.addRequired<MachineBlockFrequencyInfoWrapperPass>();
  AU.addPreserved<MachineBlockFrequencyInfoWrapperPass>();
  AU.addRequired<LiveIntervalsWrapperPass>();
  AU.addRequired<SlotIndexesWrapperPass>();
  MachineFunctionPass::getAnalysisUsage(AU);
}

bool RegAllocUnisonPass::runOnMachineFunction(MachineFunction &MF) {
  MF.getRegInfo().freezeReservedRegs();

  Unison::Analyses A;
  A.MRI = &MF.getRegInfo();
  A.TRI = MF.getSubtarget().getRegisterInfo();
  A.TII = MF.getSubtarget().getInstrInfo();
  A.LIS = &getAnalysis<LiveIntervalsWrapperPass>().getLIS();
  A.MBFI = &getAnalysis<MachineBlockFrequencyInfoWrapperPass>().getMBFI();

  Unison Impl(A, MF);
  return Impl.run();
}

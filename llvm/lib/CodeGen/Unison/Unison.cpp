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
//   -unison-dump-solution=<file>  Dump solver solution to file
//   -unison-preassign=<file>      Pre-assign variables from file
//
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/ADT/ilist.h"
#include "llvm/ADT/ilist_node.h"
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
#include "llvm/CodeGen/MachineCycleAnalysis.h"
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

static cl::opt<int> UnisonMaxBendersIter(
    "unison-max-benders-iter",
    cl::desc("Maximum Benders decomposition iterations (global/local loop)"),
    cl::init(50),
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
  class UnisonUse;

  // A def operand. Pure IR node — no solver variables.
  // Allocated from UnisonFunction::OperandAlloc for pointer stability.
  class UnisonDef {
  public:
    UnisonInstr *Parent = nullptr;
    Register Reg;
    SmallVector<UnisonUse *, 4> PotentialUses;
  };

  // A use operand. Pure IR node — no solver variables.
  class UnisonUse {
  public:
    UnisonInstr *Parent = nullptr;
    Register Reg;
    SmallVector<UnisonDef *, 2> PotentialDefs;
  };

  class UnisonInstr : public ilist_node<UnisonInstr> {
  public:
    enum Kind {
      RealInstr,
      LiveInDef,
      LiveOutUse,
      CopyOp,
    };
    Kind K;
    MachineInstr *RealMI = nullptr;
    SmallVector<unsigned, 2> AltOpcodes;
    SmallVector<UnisonDef *, 4> Defs;
    SmallVector<UnisonUse *, 4> Uses;
    SmallString<32> Name;

    bool isAlwaysActive() const {
      return K == RealInstr || K == LiveInDef || K == LiveOutUse;
    }
    bool isCopyOp() const { return K == CopyOp; }
  };

  class UnisonMBB {
    UnisonInstr *LiveInPtr = nullptr;
    UnisonInstr *LiveOutPtr = nullptr;

  public:
    MachineBasicBlock *MBB = nullptr;
    ilist<UnisonInstr> Instrs;
    std::optional<sat::NoOverlap2DConstraint> NoOverlap;
    int IssueCycleUpperBound = 0;
    DenseMap<Register, UnisonDef *> LocalReachingDefs;

    UnisonInstr *getLiveIn() const { return LiveInPtr; }
    UnisonInstr *getLiveOut() const { return LiveOutPtr; }

    void setLiveIn(UnisonInstr *UI) {
      assert(UI->K == UnisonInstr::LiveInDef);
      LiveInPtr = UI;
    }
    void setLiveOut(UnisonInstr *UI) {
      assert(UI->K == UnisonInstr::LiveOutUse);
      LiveOutPtr = UI;
    }

    void addUnisonInstr(UnisonInstr *UI) { Instrs.push_back(UI); }
    void insertBefore(UnisonInstr *Pos, UnisonInstr *UI) {
      Instrs.insert(Pos->getIterator(), UI);
    }
    void insertAfter(UnisonInstr *Pos, UnisonInstr *UI) {
      Instrs.insertAfter(Pos->getIterator(), UI);
    }

    // Return iterator to the first terminator RealInstr, or to LiveOutUse
    // if there are no terminators.
    ilist<UnisonInstr>::iterator getFirstTerminator() {
      for (auto It = Instrs.begin(); It != Instrs.end(); ++It) {
        if (It->K == UnisonInstr::RealInstr && It->RealMI &&
            It->RealMI->isTerminator())
          return It;
      }
      // No terminators — return LiveOutUse position.
      return getLiveOut()->getIterator();
    }
  };

  class UnisonFunction {
  public:
    BumpPtrAllocator OperandAlloc;
    SmallVector<std::unique_ptr<UnisonMBB>> MBBs;
    DenseMap<MachineBasicBlock *, UnisonMBB *> MBBMap;

    UnisonMBB *getUMBB(MachineBasicBlock *MBB) const {
      auto It = MBBMap.find(MBB);
      assert(It != MBBMap.end() && "MBB not in UnisonFunction");
      return It->second;
    }

  };

  // --- Solver variable map (separate from IR) ---
  // Maps (IR node, model) pairs to solver variables. A cross-block vreg
  // may have variables in both GlobalModel and a LocalModel.
  using DefModelKey = std::pair<UnisonDef *, sat::CpModelBuilder *>;
  using UseModelKey = std::pair<UnisonUse *, sat::CpModelBuilder *>;
  using InstrModelKey = std::pair<UnisonInstr *, sat::CpModelBuilder *>;

  DenseMap<DefModelKey, sat::IntVar> RegVars;
  DenseMap<UseModelKey, sat::IntVar> ChoiceVars;
  DenseMap<InstrModelKey, sat::IntVar> ICVars;
  DenseMap<InstrModelKey, sat::IntVar> InsVars;
  DenseMap<InstrModelKey, sat::IntVar> ActiveVars;

  // Look up the solver variable for a given IR node in a given model.
  sat::IntVar getRegVar(UnisonDef *D, sat::CpModelBuilder &M) const {
    auto It = RegVars.find({D, &M});
    assert(It != RegVars.end() && "no RegVar for this def in this model");
    return It->second;
  }
  sat::IntVar getChoiceVar(UnisonUse *U, sat::CpModelBuilder &M) const {
    auto It = ChoiceVars.find({U, &M});
    assert(It != ChoiceVars.end() && "no ChoiceVar for this use in this model");
    return It->second;
  }
  sat::IntVar getICVar(UnisonInstr *I, sat::CpModelBuilder &M) const {
    auto It = ICVars.find({I, &M});
    assert(It != ICVars.end() && "no ICVar for this instr in this model");
    return It->second;
  }
  sat::IntVar getInsVar(UnisonInstr *I, sat::CpModelBuilder &M) const {
    auto It = InsVars.find({I, &M});
    assert(It != InsVars.end() && "no InsVar for this instr in this model");
    return It->second;
  }

  // Rematerialization map: only populated for rematerializable CopyOps.
  DenseMap<UnisonInstr *, MachineInstr *> RematMIs;

  // --- Variable naming and lookup ---

  class NamingScheme {
    bool BuildMap;
    StringMap<sat::IntVar> VarByName;
    StringMap<UnisonInstr *> InstrByName;

  public:
    NamingScheme(bool BuildMap = false) : BuildMap(BuildMap) {}

    void registerVar(StringRef Name, sat::IntVar Var) {
      Var.WithName(Name.str());
      if (BuildMap)
        VarByName[Name] = Var;
    }

    static SmallString<32> nameVariable(StringRef InstrName,
                                        const Twine &VarSuffix) {
      SmallString<32> Result(InstrName);
      Result += ".";
      VarSuffix.toVector(Result);
      return Result;
    }

    void nameInstruction(UnisonInstr *UI, StringRef MBBPrefix,
                         unsigned InstrIdx);

    void nameActiveVar(UnisonInstr *UI, sat::BoolVar Active) {
      registerVar(nameVariable(UI->Name, "active"), sat::IntVar(Active));
    }

    std::optional<sat::IntVar> getVariableByName(StringRef Name) const {
      auto It = VarByName.find(Name);
      if (It == VarByName.end())
        return std::nullopt;
      return It->second;
    }

    UnisonInstr *getInstrByName(StringRef Name) const {
      auto It = InstrByName.find(Name);
      if (It == InstrByName.end())
        return nullptr;
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

  // Global model: cross-block vreg assignments, congruence, interference.
  sat::CpModelBuilder GlobalModel;
  // Per-MBB local models: scheduling, local allocation, NoOverlap2D.
  // Indexed by position in UFunc.MBBs.
  SmallVector<sat::CpModelBuilder> LocalModels;


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
  void restrictToDomain(sat::IntVar Var, const operations_research::Domain &Dom,
                        sat::CpModelBuilder &M);
  // Conditional version: only enforce if BoolVar is true.
  void restrictToDomain(sat::IntVar Var, const operations_research::Domain &Dom,
                        sat::BoolVar Condition, sat::CpModelBuilder &M);

  // The Unison model for the entire function.
  UnisonFunction UFunc;

  // Vregs whose live ranges cross block boundaries.
  // Populated during createUnisonProgramRepresentation.
  DenseSet<Register> CrossBlockVRegs;

  // For each (def, use) pair, the index of the def in the use's PotentialDefs.
  using DefUseKey = std::pair<UnisonDef *, UnisonUse *>;
  DenseMap<DefUseKey, unsigned> DefIdxInUseMap;

  // DChosenInU: reified BoolVar for "def D was chosen at use U".
  // Computed once per model, shared across constraint functions.
  DenseMap<DefUseKey, sat::BoolVar> DChosenInUMap;

  unsigned getDefIdxInUse(UnisonDef *D, UnisonUse *U) const {
    return DefIdxInUseMap.lookup({D, U});
  }

  sat::BoolVar getDActiveInU(sat::CpModelBuilder &M,
                             UnisonDef *D, UnisonUse *U) {
    sat::BoolVar Chosen = getDChosenInU(M, D, U);
    if (U->Parent->isAlwaysActive())
      return Chosen;
    // TODO: look up activation from ActiveVars map
    return Chosen;
  }

  sat::BoolVar getDChosenInU(sat::CpModelBuilder &M,
                             UnisonDef *D, UnisonUse *U) {
    if (U->PotentialDefs.size() == 1)
      return M.TrueVar();
    auto Key = DefUseKey{D, U};
    auto It = DChosenInUMap.find(Key);
    if (It != DChosenInUMap.end())
      return It->second;
    unsigned DefIdx = DefIdxInUseMap.lookup(Key);
    sat::BoolVar B = reifyEquality(getChoiceVar(U, M), DefIdx, M);
    DChosenInUMap[Key] = B;
    return B;
  }

  // Activation BoolVars for optional (CopyOp) instructions.
  DenseMap<UnisonInstr *, sat::BoolVar> IsActiveVar;

  // Per-boundary GlobalModel variables for cross-block vregs.
  // Each MBB where a vreg is live-in/live-out gets its own variable.
  // Congruence is explicit: for each CFG edge, LiveOut == LiveIn.
  // Populated during createVariables().
  using BoundaryKey = std::pair<UnisonMBB *, Register>;
  DenseMap<BoundaryKey, sat::IntVar> GlobalLiveInVar;
  DenseMap<BoundaryKey, sat::IntVar> GlobalLiveOutVar;

  // At code gen time, actual stack slot = FirstNewStackSlot + (solved value - NumPhysRegs).
  int FirstNewStackSlot = 0;

  // Solver responses, stored after solve() for use by generateCodeFromSolution().
  sat::CpSolverResponse GlobalResponse;
  SmallVector<sat::CpSolverResponse, 0> LocalResponses;

  SchedModelKind SchedKind = SchedModelKind::TwoSlot;

  // --- Allocation helpers ---
  // Allocate a UnisonInstr. Caller handles insertion into an ilist.
  static UnisonInstr *allocateUnisonInstr(
      UnisonInstr::Kind K, MachineInstr *RealMI = nullptr);

  // Allocate a UnisonDef from the bump-ptr allocator.
  UnisonDef *allocateUnisonDef(UnisonInstr *Parent) {
    auto *D = new (UFunc.OperandAlloc) UnisonDef();
    D->Parent = Parent;
    return D;
  }

  // Allocate a UnisonUse from the bump-ptr allocator.
  UnisonUse *allocateUnisonUse(UnisonInstr *Parent) {
    auto *U = new (UFunc.OperandAlloc) UnisonUse();
    U->Parent = Parent;
    return U;
  }

  // Wire a def to a use (bidirectional: adds to both PotentialUses and
  // PotentialDefs).
  void wireDefUse(UnisonDef *D, UnisonUse *U);


  // Full reification: returns a BoolVar B such that B <=> (Var == Value).
  sat::BoolVar reifyEquality(sat::IntVar Var, int64_t Value,
                             sat::CpModelBuilder &M);
  // Full reification: returns a BoolVar B such that B <=> (A != B).
  sat::BoolVar reifyNotEqual(sat::IntVar A, sat::IntVar B,
                             sat::CpModelBuilder &M);
  // Full reification: returns a BoolVar B such that B <=> (X AND Y).
  sat::BoolVar reifyAnd(sat::BoolVar X, sat::BoolVar Y,
                        sat::CpModelBuilder &M);

  // A canonical operand: MachineOperand* for real defs/uses,
  // bare Register (MO == nullptr) for regmask clobbers.
  struct CanonicalOperand {
    MachineOperand *MO = nullptr;
    Register Reg;

    static CanonicalOperand fromMO(MachineOperand *MO) {
      return {MO, MO->getReg()};
    }
    static CanonicalOperand fromRegmask(Register Reg) {
      return {nullptr, Reg};
    }
    bool isRegmask() const { return MO == nullptr; }
    Register getReg() const { return Reg; }
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
  void collectDefsForUnisonInstr(UnisonInstr *UInstr, MachineBasicBlock &MBB,
                             SmallVectorImpl<Register> &Defs) const;
  void collectUsesForUnisonInstr(UnisonInstr *UInstr, MachineBasicBlock &MBB,
                             SmallVectorImpl<Register> &Uses) const;

  // Process one UnisonInstr: populate its Defs/Uses vectors, wire
  // def-use connections, update LocalReachingDefs.
  // Applies uniformly to LiveInDef, RealInstr, and LiveOutUse.
  void populateDefsAndUses(UnisonInstr *UInstr, MachineBasicBlock &MBB,
                         DenseMap<Register, UnisonDef *> &LocalReachingDefs);

  // Returns true if the instruction only touches block-local live ranges.
  // LiveInDef and LiveOutUse are always non-local.
  // A RealInstr is local if none of its def/use vregs are in CrossBlockVRegs.
  // A CopyOp inherits locality from the vreg it serves.
  bool isLocal(UnisonInstr *UI, MachineBasicBlock &MBB) const;

  // Upper bound on issue cycles for a given MBB, used for solver variable
  // domains. In TwoSlot mode: 2 * (num_instrs + 2) to account for
  // LiveInDef, instructions, LiveOutUse, and OStore/OLoads.
  // In Full mode: estimated from instruction latencies.
  int getIssueCycleUpperBound(MachineBasicBlock &MBB) const;

  // Build dense register index: MCRegToIdx, IdxToMCReg, RCDomain.
  void buildURegisterDomains();

  NamingScheme NS;

  // Name all solver variables for a UnisonInstr in the given model.
  // Must be called after createVariablesForInstr().
  void nameAllVariablesInInstruction(UnisonInstr *UI, sat::CpModelBuilder &M);

  // --- Pipeline stages ---
  void createUnisonProgramRepresentation();
  // Create all solver variables: LocalModels get variables for all
  // instructions; GlobalModel additionally gets variables for
  // LiveInDef/LiveOutUse defs of cross-block vregs.
  void createVariables();
  // Create solver variables for a single instruction in the LocalModel.
  // For LiveInDef/LiveOutUse instructions with cross-block vreg defs,
  // also creates variables in the GlobalModel.
  // LocalVRegVars maps vreg Register -> shared IntVar for the local model.
  void createVariablesForInstr(UnisonInstr *UI, UnisonMBB &UMBB,
                               sat::CpModelBuilder &M,
                               DenseMap<Register, sat::IntVar> &LocalVRegVars);
  void copyExtend();
  // --- Constraint generation ---
  void addConstraints();

  // Per-MBB register allocation constraints.
  void addRegAllocConstraints(UnisonMBB &UMBB, sat::CpModelBuilder &M);
  void addNoOverlapConstraints(UnisonMBB &UMBB, sat::CpModelBuilder &M);
  // Add a rectangle to the NoOverlap2D constraint. For CopyOps, the
  // rectangle is optional (gated on IsActive). For always-active instrs,
  // it's unconditional.
  void addRectangle(UnisonMBB &UMBB, UnisonInstr *UInstr,
                    sat::IntVar RegVar,
                    sat::IntVar TimeStart, sat::IntVar TimeSize,
                    sat::IntVar TimeEnd, sat::CpModelBuilder &M);
  // Returns max over active uses of IssueCycle(use_instr). Each use
  // contributes conditionally (DActiveInU). Inactive uses contribute the
  // definer's IC. For dead defs (no uses), returns the definer's IC.
  // Caller computes TimeEnd = result + 1 (half-open interval).
  sat::IntVar getDefLastUseCycle(UnisonInstr *UInstr, unsigned DefIdx,
                                int IssueCycleUB, sat::CpModelBuilder &M);
  void addRegClassConstraints(UnisonMBB &UMBB, sat::CpModelBuilder &M);
  void deriveActivationVars(UnisonMBB &UMBB, sat::CpModelBuilder &M);
  sat::BoolVar deriveDefActivation(UnisonInstr *UInstr, unsigned DefIdx,
                                   sat::CpModelBuilder &M);

  // Per-MBB scheduling constraints.
  void addSchedConstraints(UnisonMBB &UMBB, sat::CpModelBuilder &M);
  void addDataDependencyConstraints(UnisonMBB &UMBB, sat::CpModelBuilder &M);
  void addAntiDependencyConstraints(UnisonMBB &UMBB, sat::CpModelBuilder &M);
  void addOrderingConstraints(UnisonMBB &UMBB, sat::CpModelBuilder &M);

  // Cross-MBB constraints (GlobalModel).
  // Explicit congruence: LiveOut(pred) == LiveIn(succ) per CFG edge.
  // LiveInDef linking is handled by the iteration loop (Step 6).
  void addGlobalConstraints();

  void addObjectiveFunction();
  void penalizeCopies(sat::LinearExpr &Objective, UnisonMBB &UMBB,
                      sat::CpModelBuilder &M);
  void penalizeCalleeSavedRegisters(sat::LinearExpr &Objective,
                                    sat::CpModelBuilder &M);
  void penalizeGlobalSpills(sat::LinearExpr &Objective,
                            sat::CpModelBuilder &M);
  void addMinimizeMakespanObjective(sat::LinearExpr &Objective, UnisonMBB &UMBB,
                                    sat::CpModelBuilder &M);

  // Returns the set of register indices for callee-saved registers
  // that are NOT already modified in the function.
  DenseSet<int> getPenalizedCSRIndices() const;

  // Helper: returns a BoolVar that is true iff the given instruction's
  // def and use are at different registers (non-identity copy).
  sat::BoolVar getIsNotIdentityCopy(UnisonInstr *UInstr,
                                    sat::CpModelBuilder &M);
  void solve();
  // Pin global boundary assignments into a local proto copy (raw proto).
  void pinBoundaryValuesToProto(sat::CpModelProto &Proto, unsigned MBBIdx);
  // Add nogood to GlobalModel: forbid the current boundary assignment for MBBIdx.
  void addNogoodForMBB(unsigned MBBIdx);
  void dumpSolution(StringRef Filename);
  void loadPreAssignments(StringRef Filename);
  int64_t getInsVarValueFromName(StringRef VarName, StringRef ValStr);
  int64_t getChoiceVarValueFromName(StringRef VarName, StringRef ValStr);
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
                                  MachineBasicBlock::iterator InsertPt,
                                  unsigned MBBIdx);

  // Maps memory indices to frame indices (stack slots).
  DenseMap<int, int> MemIdxToFrameIdx;
};

// ---------------------------------------------------------------------------
// Unison implementation
// ---------------------------------------------------------------------------

Unison::UnisonInstr *Unison::allocateUnisonInstr(
    UnisonInstr::Kind K, MachineInstr *RealMI) {
  auto *UInstr = new UnisonInstr();
  UInstr->K = K;
  UInstr->RealMI = RealMI;
  return UInstr;
}

void Unison::wireDefUse(UnisonDef *D, UnisonUse *U) {
  D->PotentialUses.push_back(U);
  unsigned Idx = U->PotentialDefs.size();
  U->PotentialDefs.push_back(D);
  DefIdxInUseMap[{D, U}] = Idx;
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
    Defs.push_back(CanonicalOperand::fromMO(&MO));
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
      Defs.push_back(CanonicalOperand::fromRegmask(Register(PhysReg)));
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
    Uses.push_back(CanonicalOperand::fromMO(&MO));
  }
}

void Unison::collectDefsForUnisonInstr(UnisonInstr *UInstr, MachineBasicBlock &MBB,
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
    Defs.push_back(CO.getReg());
}

void Unison::collectUsesForUnisonInstr(UnisonInstr *UInstr, MachineBasicBlock &MBB,
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
    Uses.push_back(CO.getReg());
}

void Unison::populateDefsAndUses(UnisonInstr *UInstr, MachineBasicBlock &MBB,
                               DenseMap<Register, UnisonDef *> &LocalReachingDefs) {
  SmallVector<Register> UseRegs, DefRegs;

  // Process uses first (so use-and-def of same reg sees the previous def).
  collectUsesForUnisonInstr(UInstr, MBB, UseRegs);
  for (unsigned I = 0, E = UseRegs.size(); I < E; ++I) {
    Register Reg = UseRegs[I];
    auto It = LocalReachingDefs.find(Reg);
    assert(It != LocalReachingDefs.end() && "Use without reaching def");

    UnisonUse *U = allocateUnisonUse(UInstr);
    U->Reg = Reg;
    UInstr->Uses.push_back(U);
    wireDefUse(It->second, U);
  }

  // Process defs.
  collectDefsForUnisonInstr(UInstr, MBB, DefRegs);
  for (unsigned I = 0, E = DefRegs.size(); I < E; ++I) {
    Register Reg = DefRegs[I];
    UnisonDef *D = allocateUnisonDef(UInstr);
    D->Reg = Reg;
    UInstr->Defs.push_back(D);

    LocalReachingDefs[Reg] = D;

  }
}

bool Unison::isLocal(UnisonInstr *UI, MachineBasicBlock &MBB) const {
  // Boundary instructions are always non-local.
  if (UI->K == UnisonInstr::LiveInDef || UI->K == UnisonInstr::LiveOutUse)
    return false;

  // CopyOp: check output and input sides.
  // Recursion is bounded — CopyOp chains are at most 2 deep (SM → LM).
  if (UI->isCopyOp()) {
    // Output: if any def feeds into LiveOutUse → non-local.
    for (UnisonDef *D : UI->Defs)
      for (UnisonUse *U : D->PotentialUses)
        if (U->Parent->K == UnisonInstr::LiveOutUse)
          return false;

    // Input: if any source def's parent is non-local → non-local.
    for (UnisonDef *SrcDef : UI->Uses[0]->PotentialDefs)
      if (!isLocal(SrcDef->Parent, MBB))
        return false;

    return true;
  }

  // RealInstr: check all def and use registers.
  for (UnisonDef *D : UI->Defs)
    if (D->Reg.isVirtual() && CrossBlockVRegs.contains(D->Reg))
      return false;

  for (UnisonUse *U : UI->Uses)
    if (U->Reg.isVirtual() && CrossBlockVRegs.contains(U->Reg))
      return false;

  return true;
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
  COPY_MOVE = 0,  // Register → register.
  COPY_STORE = 1, // Register → memory.
  COPY_LOAD = 2,  // Memory → register.
  COPY_MEM = 3,   // Memory → memory.
  COPY_REMAT = 4, // Rematerialize (no source read).
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
    assert(!UI->Uses.empty() && !UI->Uses[0]->PotentialDefs.empty());
    UnisonDef *ParentDef = UI->Uses[0]->PotentialDefs[0];

    bool IsStoreMove = llvm::is_contained(UI->AltOpcodes, COPY_STORE);
    if (IsStoreMove) {
      // Find which def index this is in the parent instruction.
      UnisonInstr *ParentInstr = ParentDef->Parent;
      unsigned DefIdx = 0;
      for (unsigned I = 0; I < ParentInstr->Defs.size(); I++)
        if (ParentInstr->Defs[I] == ParentDef) { DefIdx = I; break; }
      UI->Name = ParentInstr->Name;
      UI->Name += ".d";
      UI->Name += Twine(DefIdx).str();
      UI->Name += ".SM";
    } else {
      assert(ParentDef->Parent->isCopyOp());
      UnisonDef *RealDef = ParentDef->Parent->Uses[0]->PotentialDefs[0];
      assert(!UI->Defs[0]->PotentialUses.empty());
      UnisonInstr *UseInstr = UI->Defs[0]->PotentialUses[0]->Parent;

      UnisonInstr *RealInstr = RealDef->Parent;
      unsigned DefIdx = 0;
      for (unsigned I = 0; I < RealInstr->Defs.size(); I++)
        if (RealInstr->Defs[I] == RealDef) { DefIdx = I; break; }
      UI->Name = RealInstr->Name;
      UI->Name += ".d";
      UI->Name += Twine(DefIdx).str();
      UI->Name += ".LM_";
      UI->Name += UseInstr->Name;
    }
    break;
  }
  }
  if (BuildMap)
    InstrByName[UI->Name] = UI;
}

void Unison::nameAllVariablesInInstruction(UnisonInstr *UI,
                                           sat::CpModelBuilder &M) {
  // IC variable.
  NS.registerVar(NamingScheme::nameVariable(UI->Name, "ic"),
                 getICVar(UI, M));

  // Def RegVars.
  for (unsigned I = 0; I < UI->Defs.size(); I++)
    NS.registerVar(NamingScheme::nameVariable(UI->Name,
                       "def[" + Twine(I) + "].reg"),
                   getRegVar(UI->Defs[I], M));

  // Use ChoiceVars.
  for (unsigned I = 0; I < UI->Uses.size(); I++)
    NS.registerVar(NamingScheme::nameVariable(UI->Name,
                       "use[" + Twine(I) + "].choice"),
                   getChoiceVar(UI->Uses[I], M));

  // CopyOp InsVar.
  if (UI->isCopyOp())
    NS.registerVar(NamingScheme::nameVariable(UI->Name, "ins"),
                   getInsVar(UI, M));
}

static bool isCFGReducible(MachineFunction &MF) {
  MachineCycleInfo CI;
  CI.compute(MF);
  for (auto *TopCycle : CI.toplevel_cycles())
    for (auto *Cycle : depth_first(TopCycle))
      if (!Cycle->isReducible())
        return false;
  return true;
}

void Unison::createUnisonProgramRepresentation() {
  buildURegisterDomains();

  assert(isCFGReducible(MF) &&
         "Irreducible CFG detected; Unison requires reducible control flow");

  // Traverse MBBs in reverse post-order. RPO guarantees that a block is
  // visited before any block it dominates (assuming reducible CFG), so the
  // defining block of a vreg is processed before blocks where that vreg
  // is live-in.
  for (MachineBasicBlock *MBB :
       ReversePostOrderTraversal<MachineFunction *>(&this->MF)) {
    auto UMBB = std::make_unique<UnisonMBB>();
    UMBB->MBB = MBB;

    UMBB->IssueCycleUpperBound = getIssueCycleUpperBound(*MBB);

    SmallString<16> MBBPrefix("MBB");
    MBBPrefix += Twine(MBB->getNumber()).str();

    auto makeInstr = [&](UnisonInstr::Kind K, MachineInstr *RealMI,
                         const Twine &Name) -> UnisonInstr * {
      UnisonInstr *UI = allocateUnisonInstr(K, RealMI);
      UI->Name = Name.str();
      populateDefsAndUses(UI, *MBB, UMBB->LocalReachingDefs);
      UMBB->addUnisonInstr(UI);
      return UI;
    };

    UMBB->setLiveIn(makeInstr(UnisonInstr::LiveInDef, nullptr,
                              MBBPrefix + ".LI"));

    // Any vreg defined by LiveInDef is live-in to this MBB, hence cross-block.
    for (UnisonDef *D : UMBB->getLiveIn()->Defs)
      if (D->Reg.isVirtual())
        CrossBlockVRegs.insert(D->Reg);

    unsigned RIIdx = 0;
    for (MachineInstr &MI : *MBB) {
      if (MI.isDebugInstr())
        continue;
      makeInstr(UnisonInstr::RealInstr, &MI,
                MBBPrefix + ".RI" + Twine(RIIdx++));
    }

    UMBB->setLiveOut(makeInstr(UnisonInstr::LiveOutUse, nullptr,
                               MBBPrefix + ".LO"));

    LLVM_DEBUG(dbgs() << "  MBB#" << MBB->getNumber() << ": "
                      << UMBB->Instrs.size() << " unison instrs\n");

    UFunc.MBBMap[MBB] = UMBB.get();
    UFunc.MBBs.push_back(std::move(UMBB));
  }

  LLVM_DEBUG(dbgs() << "  CrossBlockVRegs: " << CrossBlockVRegs.size() << "\n");

  copyExtend();
}

void Unison::createVariables() {
  // Create variables for all instructions. LocalModels get everything;
  // GlobalModel additionally gets variables for LiveInDef/LiveOutUse
  // defs of cross-block vregs (created inside createVariablesForInstr).
  for (unsigned MBBIdx = 0; MBBIdx < UFunc.MBBs.size(); ++MBBIdx) {
    UnisonMBB &UMBB = *UFunc.MBBs[MBBIdx];
    sat::CpModelBuilder &LM = LocalModels[MBBIdx];
    DenseMap<Register, sat::IntVar> LocalVRegVars;
    for (auto &UIP : UMBB.Instrs)
      createVariablesForInstr(&UIP, UMBB, LM, LocalVRegVars);
  }
}

void Unison::createVariablesForInstr(UnisonInstr *UI, UnisonMBB &UMBB,
                                     sat::CpModelBuilder &M,
                                     DenseMap<Register, sat::IntVar> &LocalVRegVars) {
  int UB = UMBB.IssueCycleUpperBound;

  // Issue cycle (LocalModel only).
  if (UI->K == UnisonInstr::LiveInDef)
    ICVars[{UI, &M}] = M.NewConstant(0);
  else
    ICVars[{UI, &M}] = M.NewIntVar({0, UB});

  // Defs: RegVar.
  for (unsigned I = 0; I < UI->Defs.size(); I++) {
    UnisonDef *D = UI->Defs[I];
    Register Reg = D->Reg;

    if (Reg.isValid() && Reg.isVirtual() &&
        UI->K != UnisonInstr::LiveInDef) {
      // Non-LiveInDef vreg defs share one solver variable per vreg
      // within the local model (first def creates it, others reuse).
      auto It = LocalVRegVars.find(Reg);
      if (It != LocalVRegVars.end()) {
        RegVars[{D, &M}] = It->second;
      } else {
        const TargetRegisterClass *RC = A.MRI->getRegClass(Reg);
        operations_research::Domain Dom = RCDomain[RC].UnionWith(MemDomain);
        sat::IntVar Var = M.NewIntVar(Dom);
        LocalVRegVars[Reg] = Var;
        RegVars[{D, &M}] = Var;
      }
    } else if (Reg.isValid() && Reg.isPhysical()) {
      RegVars[{D, &M}] = M.NewConstant(MCRegToIdx[MCRegister(Reg)]);
    } else if (Reg.isValid() && Reg.isVirtual()) {
      // LiveInDef vreg def — gets its own variable (not shared).
      const TargetRegisterClass *RC = A.MRI->getRegClass(Reg);
      operations_research::Domain Dom = RCDomain[RC].UnionWith(MemDomain);
      RegVars[{D, &M}] = M.NewIntVar(Dom);
    } else {
      // CopyOp def without a known register — unified domain.
      RegVars[{D, &M}] = M.NewIntVar(RegMemDomain);
    }

    // GlobalModel: per-boundary variable for cross-block vreg defs.
    // LiveInDef defs get a GlobalLiveInVar entry.
    if (UI->K == UnisonInstr::LiveInDef && Reg.isValid() &&
        Reg.isVirtual()) {
      const TargetRegisterClass *RC = A.MRI->getRegClass(Reg);
      operations_research::Domain Dom = RCDomain[RC].UnionWith(MemDomain);
      sat::IntVar GVar = GlobalModel.NewIntVar(Dom);
      RegVars[{D, &GlobalModel}] = GVar;
      GlobalLiveInVar[{&UMBB, Reg}] = GVar;
    }
  }

  // Uses: ChoiceVar (LocalModel only).
  for (UnisonUse *U : UI->Uses) {
    unsigned N = U->PotentialDefs.size();
    sat::IntVar CV = (N <= 1)
        ? M.NewConstant(0)
        : M.NewIntVar({0, static_cast<int64_t>(N - 1)});
    ChoiceVars[{U, &M}] = CV;

    // GlobalModel: LiveOutUse uses of cross-block vregs get a
    // GlobalLiveOutVar entry for congruence constraints.
    Register Reg = U->Reg;
    if (UI->K == UnisonInstr::LiveOutUse && Reg.isValid() &&
        Reg.isVirtual()) {
      const TargetRegisterClass *RC = A.MRI->getRegClass(Reg);
      operations_research::Domain Dom = RCDomain[RC].UnionWith(MemDomain);
      GlobalLiveOutVar[{&UMBB, Reg}] = GlobalModel.NewIntVar(Dom);
    }
  }

  // CopyOp: Ins variable (LocalModel only).
  if (UI->isCopyOp()) {
    unsigned N = UI->AltOpcodes.size();
    sat::IntVar IV = (N == 1)
        ? M.NewConstant(0)
        : M.NewIntVar({0, static_cast<int64_t>(N - 1)});
    InsVars[{UI, &M}] = IV;
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

  auto createCopyOp = [&](ArrayRef<unsigned> Opcodes) -> UnisonInstr * {
    UnisonInstr *UI = allocateUnisonInstr(UnisonInstr::CopyOp, nullptr);
    UI->AltOpcodes.assign(Opcodes.begin(), Opcodes.end());
    UI->Uses.push_back(allocateUnisonUse(UI));
    UI->Defs.push_back(allocateUnisonDef(UI));
    return UI;
  };

  for (auto &UMBB : UFunc.MBBs) {
    DenseSet<UnisonInstr *> ToSkip;

    for (UnisonInstr &Instr : UMBB->Instrs) {
      if (ToSkip.contains(&Instr))
        continue;

      // Track insertion point so SMs are appended in DefIdx order,
      // not prepended (which would reverse them).
      UnisonInstr *InsertAfterPt = &Instr;

      for (unsigned DefIdx = 0; DefIdx < Instr.Defs.size(); ++DefIdx) {
        UnisonDef *RealDef = Instr.Defs[DefIdx];
        if (RealDef->PotentialUses.empty())
          continue;

        unsigned NumRealUses = RealDef->PotentialUses.size();

        // --- Store-move: inserted right AFTER the def instruction ---
        // We advance InsertAfterPt so successive SMs appear in
        // DefIdx order (d0.SM, d1.SM, d2.SM, ...).
        SmallVector<unsigned, 3> SMOpcodes = {COPY_MOVE, COPY_STORE};
        if (Instr.K == UnisonInstr::LiveInDef)
          SMOpcodes.push_back(COPY_MEM);
        UnisonInstr *StoreMove = createCopyOp(SMOpcodes);
        UMBB->insertAfter(InsertAfterPt, StoreMove);
        InsertAfterPt = StoreMove;
        ToSkip.insert(StoreMove);
        wireDefUse(RealDef, StoreMove->Uses[0]);
        UnisonDef *StoreMoveDef = StoreMove->Defs[0];

        bool CanRemat = Instr.K == UnisonInstr::RealInstr &&
                        Instr.RealMI &&
                        isRematerializable(*Instr.RealMI, *A.MRI);

        // --- Load-moves: inserted right BEFORE each real use ---
        for (unsigned UI = 0; UI < NumRealUses; ++UI) {
          UnisonUse *RealUse = RealDef->PotentialUses[UI];

          SmallVector<unsigned, 3> LoadOpcodes = {COPY_MOVE, COPY_LOAD};
          if (CanRemat)
            LoadOpcodes.push_back(COPY_REMAT);

          UnisonInstr *LoadMove = createCopyOp(LoadOpcodes);
          // Insert before the use's parent, but if the parent is
          // LiveOutUse, insert before the first terminator instead.
          // This ensures ordering constraints place the LoadMove
          // before the terminator barrier, so NoOverlap2D prevents
          // it from clobbering registers the terminator reads.
          if (RealUse->Parent->K == UnisonInstr::LiveOutUse)
            UMBB->Instrs.insert(UMBB->getFirstTerminator(), LoadMove);
          else
            UMBB->insertBefore(RealUse->Parent, LoadMove);
          ToSkip.insert(LoadMove);
          if (CanRemat)
            RematMIs[LoadMove] = Instr.RealMI;
          wireDefUse(StoreMoveDef, LoadMove->Uses[0]);
          UnisonDef *LoadMoveDef = LoadMove->Defs[0];

          wireDefUse(LoadMoveDef, RealUse);

          if (RealUse->Parent->K == UnisonInstr::LiveOutUse)
            wireDefUse(StoreMoveDef, RealUse);

          // Name after wiring so nameInstruction can walk def-use chains.
          NS.nameInstruction(LoadMove, {}, 0);
        }

        // Name StoreMove after all LoadMoves are wired.
        NS.nameInstruction(StoreMove, {}, 0);
      }
    }
  }
}

// Congruence is explicit: each boundary (LiveInDef, LiveOutUse) of a
// cross-block vreg gets its own GlobalModel variable. addGlobalConstraints
// links them across CFG edges: LiveOut(pred) == LiveIn(succ).
// The iteration loop (Step 6) fixes boundary register values from
// GlobalModel solutions into LocalModels.

sat::BoolVar Unison::reifyEquality(sat::IntVar Var, int64_t Value,
                                   sat::CpModelBuilder &M) {
  sat::BoolVar B = M.NewBoolVar();
  M.AddEquality(Var, Value).OnlyEnforceIf(B);
  M.AddNotEqual(Var, Value).OnlyEnforceIf(~B);
  return B;
}

sat::BoolVar Unison::reifyNotEqual(sat::IntVar A, sat::IntVar B,
                                   sat::CpModelBuilder &M) {
  sat::BoolVar NE = M.NewBoolVar();
  M.AddNotEqual(A, B).OnlyEnforceIf(NE);
  M.AddEquality(A, B).OnlyEnforceIf(~NE);
  return NE;
}

sat::BoolVar Unison::reifyAnd(sat::BoolVar X, sat::BoolVar Y,
                              sat::CpModelBuilder &M) {
  sat::BoolVar R = M.NewBoolVar();
  M.AddBoolAnd({X, Y}).OnlyEnforceIf(R);
  M.AddBoolOr({~X, ~Y}).OnlyEnforceIf(~R);
  return R;
}

void Unison::restrictToDomain(sat::IntVar Var,
                              const operations_research::Domain &Dom,
                              sat::CpModelBuilder &M) {
  const auto &VarProto = M.Proto().variables(Var.index());
  if (VarProto.domain_size() == 2 && VarProto.domain(0) == VarProto.domain(1))
    return;
  // CP-SAT: restrict variable to domain by adding it as a linear constraint.
  M.AddLinearConstraint(Var, Dom);
}

void Unison::restrictToDomain(sat::IntVar Var,
                              const operations_research::Domain &Dom,
                              sat::BoolVar Condition,
                              sat::CpModelBuilder &M) {
  const auto &VarProto = M.Proto().variables(Var.index());
  if (VarProto.domain_size() == 2 && VarProto.domain(0) == VarProto.domain(1))
    return;
  M.AddLinearConstraint(Var, Dom).OnlyEnforceIf(Condition);
}

void Unison::addConstraints() {
  for (unsigned MBBIdx = 0; MBBIdx < UFunc.MBBs.size(); ++MBBIdx) {
    sat::CpModelBuilder &LM = LocalModels[MBBIdx];
    addRegAllocConstraints(*UFunc.MBBs[MBBIdx], LM);
    addSchedConstraints(*UFunc.MBBs[MBBIdx], LM);
  }
  addGlobalConstraints();
}

// ---------------------------------------------------------------------------
// Global constraints (cross-MBB)
// ---------------------------------------------------------------------------
//
// Congruence: for each CFG edge MBB0→MBB1 and each cross-block vreg
// live across that edge, the LiveOut variable in MBB0 must equal the
// LiveIn variable in MBB1.
//
// Interference is left to the local models (NoOverlap2D). If the
// global assignment causes a local infeasibility, the iteration loop
// (Step 6) will add a nogood to the GlobalModel.

void Unison::addGlobalConstraints() {
  // Congruence: for each CFG edge, link LiveOut to LiveIn.
  // For each MBB's live-out vregs, match against each successor's live-in.
  for (auto &[Key, OutVar] : GlobalLiveOutVar) {
    auto *UMBB = Key.first;
    Register Reg = Key.second;

    for (MachineBasicBlock *Succ : UMBB->MBB->successors()) {
      UnisonMBB *SuccUMBB = UFunc.getUMBB(Succ);
      auto InIt = GlobalLiveInVar.find({SuccUMBB, Reg});
      if (InIt == GlobalLiveInVar.end())
        continue;
      GlobalModel.AddEquality(OutVar, InIt->second);
    }
  }

  // Boundary interference: at each boundary point, all live registers
  // (virtual + physical) must be assigned distinct values.
  auto addBoundaryRegConstraints = [&](UnisonMBB *UMBB,
                                     ArrayRef<Register> Regs,
                                     const DenseMap<BoundaryKey, sat::IntVar> &VarMap) {
    SmallVector<sat::IntVar, 16> LiveVars;
    for (Register Reg : Regs) {
      if (!Reg.isValid())
        continue;
      if (Reg.isVirtual()) {
        auto It = VarMap.find({UMBB, Reg});
        if (It != VarMap.end())
          LiveVars.push_back(It->second);
      } else if (MCRegToIdx.count(MCRegister(Reg))) {
        LiveVars.push_back(
            GlobalModel.NewConstant(MCRegToIdx[MCRegister(Reg)]));
      }
    }
    if (LiveVars.size() > 1)
      GlobalModel.AddAllDifferent(LiveVars);
  };

  for (auto &UMBB : UFunc.MBBs) {
    // Collect registers from boundary instruction operands.
    SmallVector<Register, 16> LiveInRegs, LiveOutRegs;
    for (UnisonDef *D : UMBB->getLiveIn()->Defs)
      LiveInRegs.push_back(D->Reg);
    for (UnisonUse *U : UMBB->getLiveOut()->Uses)
      LiveOutRegs.push_back(U->Reg);

    addBoundaryRegConstraints(UMBB.get(), LiveInRegs, GlobalLiveInVar);
    addBoundaryRegConstraints(UMBB.get(), LiveOutRegs, GlobalLiveOutVar);
  }

  LLVM_DEBUG(dbgs() << "  GlobalModel: " << GlobalLiveInVar.size()
                    << " live-in vars, " << GlobalLiveOutVar.size()
                    << " live-out vars, "
                    << GlobalModel.Proto().constraints_size()
                    << " constraints\n");
}

// ---------------------------------------------------------------------------
// Register allocation constraints (per MBB)
// ---------------------------------------------------------------------------

void Unison::addRegAllocConstraints(UnisonMBB &UMBB, sat::CpModelBuilder &M) {
  // Activation must be computed before NoOverlap (which uses IsActiveVar).
  deriveActivationVars(UMBB, M);
  addRegClassConstraints(UMBB, M);
  addNoOverlapConstraints(UMBB, M);
}

sat::IntVar Unison::getDefLastUseCycle(UnisonInstr *UInstr,
                                          unsigned DefIdx,
                                          int IssueCycleUB,
                                          sat::CpModelBuilder &M) {
  UnisonDef *DefOp = UInstr->Defs[DefIdx];

  if (DefOp->PotentialUses.empty()) {
    // Dead def (e.g., regmask clobber): the definer's IC is the only
    // time point. Caller adds +1 for the minimal half-open rectangle.
    return getICVar(UInstr, M);
  }

  // LastUseCycle = max over active uses of IssueCycle(use_instr).
  // Each use contributes conditionally: only if the def is actively
  // used at U (DActiveInU). Inactive uses contribute the definer's IC
  // (so that the rectangle always has at least size 1).
  std::vector<sat::IntVar> UseCycleVars;
  for (UnisonUse *U : DefOp->PotentialUses) {
    sat::IntVar UseCycle = M.NewIntVar({0, IssueCycleUB});
    sat::BoolVar Active = getDActiveInU(M, UInstr->Defs[DefIdx], U);
    M.AddEquality(UseCycle, getICVar(U->Parent, M)).OnlyEnforceIf(Active);
    M.AddEquality(UseCycle, getICVar(UInstr, M)).OnlyEnforceIf(~Active);
    UseCycleVars.push_back(UseCycle);
  }

  sat::IntVar MaxUseCycle = M.NewIntVar({0, IssueCycleUB});
  std::vector<sat::LinearExpr> Exprs(UseCycleVars.begin(), UseCycleVars.end());
  M.AddMaxEquality(MaxUseCycle, Exprs);
  return MaxUseCycle;
}

void Unison::addNoOverlapConstraints(UnisonMBB &UMBB, sat::CpModelBuilder &M) {
  UMBB.NoOverlap.emplace(M.AddNoOverlap2D());

  for (auto &UInstrPtr : UMBB.Instrs) {
    UnisonInstr *UInstr = &UInstrPtr;

    for (unsigned DI = 0; DI < UInstr->Defs.size(); ++DI) {
      UnisonDef *DefOp = UInstr->Defs[DI];
      int UB = UMBB.IssueCycleUpperBound;

      // Rectangle starts at IC (the issue cycle of the definer), matching
      // the original Unison model: start(t) = issue(definer(t)).
      static constexpr int PipelineLatency = 1;
      sat::IntVar TimeStart = M.NewIntVar({0, UB + PipelineLatency});
      M.AddEquality(TimeStart,
                         getICVar(UInstr, M) + (PipelineLatency - 1));

      sat::IntVar LastUseCycle = getDefLastUseCycle(UInstr, DI, UB, M);
      // TimeEnd = max(TimeStart + 1, LastUseCycle + 1).
      sat::IntVar TimeEnd = M.NewIntVar({0, UB + 2});
      sat::IntVar MinEnd = M.NewIntVar({0, UB + 2});
      M.AddEquality(MinEnd, TimeStart + 1);
      M.AddMaxEquality(TimeEnd,
                           {sat::LinearExpr(MinEnd),
                            sat::LinearExpr(LastUseCycle) + 1});

      sat::IntVar TimeSize = M.NewIntVar({1, UB + 2});
      M.AddEquality(TimeSize, TimeEnd - TimeStart);

      addRectangle(UMBB, UInstr, getRegVar(DefOp, M), TimeStart, TimeSize, TimeEnd, M);
    }
  }
}

void Unison::addRectangle(UnisonMBB &UMBB, UnisonInstr *UInstr,
                          sat::IntVar RegVar,
                          sat::IntVar TimeStart, sat::IntVar TimeSize,
                          sat::IntVar TimeEnd, sat::CpModelBuilder &M) {
  if (UInstr->isCopyOp()) {
    // CopyOp rectangles only participate in NoOverlap2D when active.
    // Inactive copies don't occupy any register or time slot.
    auto ActiveIt = IsActiveVar.find(UInstr);
    assert(ActiveIt != IsActiveVar.end() && "CopyOp missing IsActiveVar");
    sat::IntervalVar TimeAxis =
        M.NewOptionalIntervalVar(TimeStart, TimeSize, TimeEnd,
                                     ActiveIt->second);
    sat::IntervalVar RegAxis =
        M.NewOptionalFixedSizeIntervalVar(RegVar, 1, ActiveIt->second);
    UMBB.NoOverlap->AddRectangle(RegAxis, TimeAxis);
  } else {
    sat::IntervalVar TimeAxis =
        M.NewIntervalVar(TimeStart, TimeSize, TimeEnd);
    sat::IntervalVar RegAxis =
        M.NewFixedSizeIntervalVar(RegVar, 1);
    UMBB.NoOverlap->AddRectangle(RegAxis, TimeAxis);
  }
}

void Unison::addRegClassConstraints(UnisonMBB &UMBB, sat::CpModelBuilder &M) {
  operations_research::Domain PhysRegDomain(0, NumPhysRegs - 1);

  for (auto &UInstrPtr : UMBB.Instrs) {
    UnisonInstr *UInstr = &UInstrPtr;

    // --- Def-side constraints ---
    // RealInstr defs: restrict to register class + memory domain.
    if (UInstr->K == UnisonInstr::RealInstr) {
      for (UnisonDef *D : UInstr->Defs) {
        if (D->Reg.isVirtual()) {
          const TargetRegisterClass *RC = A.MRI->getRegClass(D->Reg);
          restrictToDomain(getRegVar(D, M),
                           RCDomain[RC].UnionWith(MemDomain), M);
        }
      }
    } else if (UInstr->isCopyOp()) {
      UnisonDef *DefOp = UInstr->Defs[0];
      sat::IntVar DefReg = getRegVar(DefOp, M);

      for (unsigned I = 0; I < UInstr->AltOpcodes.size(); ++I) {
        sat::BoolVar InsIsI = reifyEquality(getInsVar(UInstr, M), I, M);
        unsigned Opcode = UInstr->AltOpcodes[I];
        if (Opcode == COPY_STORE || Opcode == COPY_MEM)
          restrictToDomain(DefReg, MemDomain, InsIsI, M);
        else
          restrictToDomain(DefReg, PhysRegDomain, InsIsI, M);
      }
    }

    // --- Use-side constraints: propagate source domain from uses to defs ---
    for (unsigned DI = 0; DI < UInstr->Defs.size(); ++DI) {
      UnisonDef *DefOp = UInstr->Defs[DI];
      sat::IntVar DefReg = getRegVar(DefOp, M);

      for (UnisonUse *U : DefOp->PotentialUses) {
        UnisonInstr *User = U->Parent;
        sat::BoolVar Chosen = getDChosenInU(M, UInstr->Defs[DI], U);

        if (User->K == UnisonInstr::RealInstr) {
          // Real instructions read from registers only.
          restrictToDomain(DefReg, PhysRegDomain, Chosen, M);

        } else if (User->isCopyOp()) {
          // CopyOp source requirement depends on instruction alternative.
          for (unsigned I = 0; I < User->AltOpcodes.size(); ++I) {
            sat::BoolVar InsIsI = reifyEquality(getInsVar(User, M), I, M);
            sat::BoolVar Both = reifyAnd(Chosen, InsIsI, M);
            unsigned Opcode = User->AltOpcodes[I];
            if (Opcode == COPY_LOAD || Opcode == COPY_MEM) {
              restrictToDomain(DefReg, MemDomain, Both, M);
            } else if (Opcode == COPY_STORE || Opcode == COPY_MOVE) {
              restrictToDomain(DefReg, PhysRegDomain, Both, M);
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
sat::BoolVar Unison::deriveDefActivation(UnisonInstr *UInstr, unsigned DI,
                                         sat::CpModelBuilder &M) {
  UnisonDef *DefOp = UInstr->Defs[DI];

  if (DefOp->PotentialUses.empty())
    return M.FalseVar();

  SmallVector<sat::BoolVar, 4> ActiveBools;

  for (UnisonUse *U : DefOp->PotentialUses) {
    bool SingleChoice = (U->PotentialDefs.size() == 1);
    bool UseAlwaysActive = U->Parent->isAlwaysActive();

    // Short-circuit: unconditionally chosen + always-active use.
    if (SingleChoice && UseAlwaysActive)
      return M.TrueVar();

    ActiveBools.push_back(getDActiveInU(M, UInstr->Defs[DI], U));
  }

  if (ActiveBools.empty())
    return M.FalseVar();
  if (ActiveBools.size() == 1)
    return ActiveBools[0];

  sat::BoolVar Active = M.NewBoolVar();
  for (sat::BoolVar B : ActiveBools)
    M.AddImplication(B, Active);
  M.AddBoolOr(ActiveBools).OnlyEnforceIf(Active);
  std::vector<sat::BoolVar> NegBools;
  for (sat::BoolVar B : ActiveBools)
    NegBools.push_back(~B);
  M.AddBoolAnd(NegBools).OnlyEnforceIf(~Active);
  return Active;
}

void Unison::deriveActivationVars(UnisonMBB &UMBB, sat::CpModelBuilder &M) {
  // Process in reverse order: load-moves before store-moves, so that
  // IsActiveVar is available for nested activation (store-move activation
  // depends on load-move activation).
  for (auto It = UMBB.Instrs.rbegin(); It != UMBB.Instrs.rend(); ++It) {
    UnisonInstr *UInstr = &*It;
    if (!UInstr->isCopyOp())
      continue;

    // Instruction is active if any of its defs is active.
    SmallVector<sat::BoolVar, 2> DefActives;
    for (unsigned DI = 0; DI < UInstr->Defs.size(); ++DI) {
      sat::BoolVar DA = deriveDefActivation(UInstr, DI, M);
      DefActives.push_back(DA);
    }

    if (DefActives.empty()) {
      IsActiveVar[UInstr] = M.FalseVar();
    } else if (DefActives.size() == 1) {
      IsActiveVar[UInstr] = DefActives[0];
    } else {
      sat::BoolVar Active = M.NewBoolVar();
      for (sat::BoolVar B : DefActives)
        M.AddImplication(B, Active);
      M.AddBoolOr(DefActives).OnlyEnforceIf(Active);
      std::vector<sat::BoolVar> NegBools;
      for (sat::BoolVar B : DefActives)
        NegBools.push_back(~B);
      M.AddBoolAnd(NegBools).OnlyEnforceIf(~Active);
      IsActiveVar[UInstr] = Active;
    }
    NS.nameActiveVar(UInstr, IsActiveVar[UInstr]);
  }
}

// ---------------------------------------------------------------------------
// Scheduling constraints (per MBB)
// ---------------------------------------------------------------------------

void Unison::addSchedConstraints(UnisonMBB &UMBB, sat::CpModelBuilder &M) {
  // When UnisonPreserveOrder is set, chain consecutive real instructions
  // in program order to preserve the original schedule.
  if (UnisonPreserveOrder) {
    UnisonInstr *Prev = nullptr;
    for (auto &UIP : UMBB.Instrs) {
      if (UIP.K != UnisonInstr::RealInstr)
        continue;
      UnisonInstr *Cur = &UIP;
      if (!Prev)
        Prev = UMBB.getLiveIn();
      M.AddLessThan(getICVar(Prev, M), getICVar(Cur, M));
      Prev = Cur;
    }
  }
  addDataDependencyConstraints(UMBB, M);
  addAntiDependencyConstraints(UMBB, M);
  addOrderingConstraints(UMBB, M);
}

void Unison::addDataDependencyConstraints(UnisonMBB &UMBB,
                                          sat::CpModelBuilder &M) {
  // Latency = 1: a use must be issued at least one cycle after its
  // definer, matching the original Unison model: issue(u) >= issue(d) + 1.
  static constexpr int Latency = 1;
  for (auto &UInstrPtr : UMBB.Instrs) {
    UnisonInstr *UInstr = &UInstrPtr;
    for (unsigned DI = 0; DI < UInstr->Defs.size(); ++DI) {
      for (UnisonUse *U : UInstr->Defs[DI]->PotentialUses) {
        sat::BoolVar Chosen = getDChosenInU(M, UInstr->Defs[DI], U);
        M.AddGreaterOrEqual(getICVar(U->Parent, M),
                                getICVar(UInstr, M) + Latency)
            .OnlyEnforceIf(Chosen);
      }
    }
  }
}

void Unison::addAntiDependencyConstraints(UnisonMBB &UMBB,
                                          sat::CpModelBuilder &M) {
  // WAR (Write After Read) anti-dependencies: if instruction A reads
  // a register variable V, and a later instruction B (in original order)
  // defines the same variable V, then IC(A) <= IC(B).
  //
  // Walk instructions in original order. For each use, record the
  // Reg.Var it reads from. For each def, check if any earlier use
  // read from the same Reg.Var.
  //
  // Note: this only tracks same-variable anti-dependencies. Cross-variable
  // conflicts (different IntVars assigned the same register) are handled
  // by the post-solve topological sort in sortByIssueCycle.

  // Map from Reg.Var index to the instructions that read it.
  // We use the IntVar's index as the key.
  DenseMap<int, SmallVector<UnisonInstr *, 4>> ReadersOfVar;

  for (auto &UInstrPtr : UMBB.Instrs) {
    UnisonInstr *UInstr = &UInstrPtr;

    // First: check defs against prior readers (anti-dependency).
    for (unsigned DI = 0; DI < UInstr->Defs.size(); ++DI) {
      int VarIdx = getRegVar(UInstr->Defs[DI], M).index();
      auto It = ReadersOfVar.find(VarIdx);
      if (It != ReadersOfVar.end()) {
        for (UnisonInstr *Reader : It->second) {
          if (Reader == UInstr)
            continue;
          M.AddLessOrEqual(getICVar(Reader, M), getICVar(UInstr, M));
        }
      }
    }

    // Then: record this instruction's reads.
    // Each use reads from its potential defs' Reg.Var. Since the
    // choice is a solver variable, we conservatively record ALL
    // potential source variables.
    for (unsigned UI = 0; UI < UInstr->Uses.size(); ++UI) {
      UnisonUse *UseOp = UInstr->Uses[UI];
      for (UnisonDef *SrcDef : UseOp->PotentialDefs) {
        int VarIdx = getRegVar(SrcDef, M).index();
        ReadersOfVar[VarIdx].push_back(UInstr);
      }
    }
  }
}

void Unison::addOrderingConstraints(UnisonMBB &UMBB, sat::CpModelBuilder &M) {
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
    UnisonInstr *UI = &UIP;
    if (UI->K == UnisonInstr::LiveInDef || UI->K == UnisonInstr::LiveOutUse)
      continue;

    bool IsBarrier = UI->K == UnisonInstr::RealInstr && UI->RealMI &&
                     A.TII->isSchedulingBoundary(*UI->RealMI, UMBB.MBB, MF);

    if (IsBarrier) {
      // This barrier must come after all instructions since the last barrier.
      if (!ICsSinceLastBarrier.empty()) {
        sat::IntVar MaxPrev = M.NewIntVar({0, UB});
        std::vector<sat::LinearExpr> Exprs(ICsSinceLastBarrier.begin(),
                                            ICsSinceLastBarrier.end());
        M.AddMaxEquality(MaxPrev, Exprs);
        M.AddGreaterThan(getICVar(UI, M), MaxPrev);
      } else if (HaveBarrier) {
        M.AddGreaterThan(getICVar(UI, M), LastBarrierIC);
      }
      LastBarrierIC = getICVar(UI, M);
      HaveBarrier = true;
      ICsSinceLastBarrier.clear();
    } else {
      // Pure instruction (including CopyOps): must come after the last barrier.
      if (HaveBarrier)
        M.AddGreaterThan(getICVar(UI, M), LastBarrierIC);
      ICsSinceLastBarrier.push_back(getICVar(UI, M));
    }
  }

  // LiveOutUse must come after all real instructions in the block.
  SmallVector<sat::IntVar, 16> AllPrevICs(ICsSinceLastBarrier);
  if (HaveBarrier)
    AllPrevICs.push_back(LastBarrierIC);

  if (!AllPrevICs.empty()) {
    sat::IntVar MaxAll = M.NewIntVar({0, UB});
    std::vector<sat::LinearExpr> Exprs(AllPrevICs.begin(), AllPrevICs.end());
    M.AddMaxEquality(MaxAll, Exprs);
    for (auto &UIP : UMBB.Instrs) {
      if (UIP.K == UnisonInstr::LiveOutUse)
        M.AddGreaterThan(getICVar(&UIP, M), MaxAll);
    }
  }
}

sat::BoolVar Unison::getIsNotIdentityCopy(UnisonInstr *UInstr,
                                          sat::CpModelBuilder &M) {
  // TODO: what if copy has several defs feeding into it?
  UnisonUse *UseOp = UInstr->Uses[0];
  assert(!UseOp->PotentialDefs.empty());
  UnisonDef *SrcDef = UseOp->PotentialDefs[0];
  return reifyNotEqual(getRegVar(SrcDef, M),
                       getRegVar(UInstr->Defs[0], M), M);
}

void Unison::penalizeCopies(sat::LinearExpr &Objective, UnisonMBB &UMBB,
                            sat::CpModelBuilder &M) {
  static constexpr int64_t MemOpWeight = 10;
  static constexpr int64_t RematWeight = 1;
  static constexpr int64_t CopyWeight = 1;
  static constexpr int64_t MaxFreq = 1000000;

  int64_t Freq = std::min<int64_t>(
      std::max<int64_t>(A.MBFI->getBlockFreq(UMBB.MBB).getFrequency(), 1),
      MaxFreq);

  for (auto &UInstrPtr : UMBB.Instrs) {
    UnisonInstr *UInstr = &UInstrPtr;

    // Real COPY instructions: penalize non-identity copies.
    if (UInstr->K == UnisonInstr::RealInstr &&
        UInstr->RealMI->isCopy() && !UInstr->Uses.empty()) {
      Objective += Freq * CopyWeight * getIsNotIdentityCopy(UInstr, M);
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
      sat::BoolVar IsOpcChosen = reifyEquality(getInsVar(UInstr, M), I, M);
      sat::BoolVar ShouldIncludeInCost = reifyAnd(IsOpcChosen, Active, M);

      if (Opcode == COPY_STORE || Opcode == COPY_LOAD ||
          Opcode == COPY_MEM) {
        Objective += Freq * MemOpWeight * ShouldIncludeInCost;
      } else if (Opcode == COPY_REMAT) {
        Objective += Freq * RematWeight * ShouldIncludeInCost;
      } else {
        // COPY_MOVE: only penalize non-identity.
        sat::BoolVar NotIdentity = getIsNotIdentityCopy(UInstr, M);
        ShouldIncludeInCost = reifyAnd(ShouldIncludeInCost, NotIdentity, M);
        Objective += Freq * CopyWeight * ShouldIncludeInCost;
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

void Unison::penalizeCalleeSavedRegisters(sat::LinearExpr &Objective,
                                          sat::CpModelBuilder &M) {
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

  // Collect all GlobalModel boundary vars (deduplicated: one per vreg
  // is enough since congruence links them).
  DenseMap<Register, sat::IntVar> SeenVRegs;
  for (auto &[Key, GVar] : GlobalLiveInVar)
    SeenVRegs.try_emplace(Key.second, GVar);
  for (auto &[Key, GVar] : GlobalLiveOutVar)
    SeenVRegs.try_emplace(Key.second, GVar);

  // Penalize each unmodified CSR that has at least one cross-block vreg
  // assigned to it.
  for (int Idx : PenalizedCSRIndices) {
    SmallVector<sat::BoolVar, 8> VRegAtIdx;
    for (auto &[Reg, GVar] : SeenVRegs)
      VRegAtIdx.push_back(reifyEquality(GVar, Idx, M));
    if (VRegAtIdx.empty())
      continue;
    sat::BoolVar CSRUsed = M.NewBoolVar();
    M.AddBoolOr(VRegAtIdx).OnlyEnforceIf(CSRUsed);
    for (auto &B : VRegAtIdx)
      M.AddEquality(B, false).OnlyEnforceIf(~CSRUsed);
    Objective += EntryFreq * CSRWeight * CSRUsed;
  }
}

void Unison::penalizeGlobalSpills(sat::LinearExpr &Objective,
                                  sat::CpModelBuilder &M) {
  static constexpr int64_t SpillWeight = 100;
  static constexpr int64_t MaxFreq = 1000000;

  // Penalize once per unique vreg to avoid double-counting
  // (congruence forces live-in and live-out to be equal).
  DenseMap<Register, sat::IntVar> SeenVRegs;
  for (auto &[Key, GVar] : GlobalLiveInVar)
    SeenVRegs.try_emplace(Key.second, GVar);
  for (auto &[Key, GVar] : GlobalLiveOutVar)
    SeenVRegs.try_emplace(Key.second, GVar);

  int64_t EntryFreq = std::min<int64_t>(
      std::max<int64_t>(
          A.MBFI->getBlockFreq(&MF.front()).getFrequency(), 1),
      MaxFreq);

  for (auto &[Reg, GVar] : SeenVRegs) {
    // IsSpilled = (GVar >= NumPhysRegs), i.e. assigned to memory domain.
    sat::BoolVar IsSpilled = M.NewBoolVar();
    M.AddGreaterOrEqual(GVar, NumPhysRegs).OnlyEnforceIf(IsSpilled);
    M.AddLessThan(GVar, NumPhysRegs).OnlyEnforceIf(~IsSpilled);
    Objective += EntryFreq * SpillWeight * IsSpilled;
  }
}

void Unison::addMinimizeMakespanObjective(sat::LinearExpr &Objective,
                                          UnisonMBB &UMBB,
                                          sat::CpModelBuilder &M) {
  static constexpr int64_t MaxFreq = 1000000;
  int64_t Freq = std::min<int64_t>(
      std::max<int64_t>(A.MBFI->getBlockFreq(UMBB.MBB).getFrequency(), 1),
      MaxFreq);
  for (auto &UInstrPtr : UMBB.Instrs) {
    if (UInstrPtr.K == UnisonInstr::LiveOutUse)
      Objective += Freq * getICVar(&UInstrPtr, M);
  }
}

void Unison::addObjectiveFunction() {
  // GlobalModel objective: penalize callee-saved register usage + spills.
  {
    sat::LinearExpr GlobalObjective;
    penalizeCalleeSavedRegisters(GlobalObjective, GlobalModel);
    penalizeGlobalSpills(GlobalObjective, GlobalModel);
    GlobalModel.Minimize(GlobalObjective);
  }

  // Per-MBB LocalModel objectives: copy cost + makespan.
  for (unsigned MBBIdx = 0; MBBIdx < UFunc.MBBs.size(); ++MBBIdx) {
    sat::CpModelBuilder &LM = LocalModels[MBBIdx];
    sat::LinearExpr LocalObjective;
    penalizeCopies(LocalObjective, *UFunc.MBBs[MBBIdx], LM);
    addMinimizeMakespanObjective(LocalObjective, *UFunc.MBBs[MBBIdx], LM);
    LM.Minimize(LocalObjective);
  }
}

// ---------------------------------------------------------------------------
// Pin boundary values into a local proto copy (raw proto manipulation).
// Variable indices from the original LocalModel are valid in the proto
// since the proto was copied from LocalModels[MBBIdx].
// ---------------------------------------------------------------------------

// Helper: add VarIdx == Val as a LinearConstraint in the proto.
static void addProtoEquality(sat::CpModelProto &Proto,
                             int VarIdx, int64_t Val) {
  auto *CT = Proto.add_constraints();
  auto *Lin = CT->mutable_linear();
  Lin->add_vars(VarIdx);
  Lin->add_coeffs(1);
  Lin->add_domain(Val);
  Lin->add_domain(Val);
}

// Helper: add (ChoiceIdx == K) => (RegIdx == Val).
// Creates a fresh Bool variable B_k with full reification:
//   B_k <=> (ChoiceIdx == K), then B_k => (RegIdx == Val).
static void addProtoConditionalEquality(sat::CpModelProto &Proto,
                                        int ChoiceIdx, int64_t K,
                                        int64_t ChoiceSize,
                                        int RegIdx, int64_t Val) {
  // Create B_k: domain [0,1].
  int Bk = Proto.variables_size();
  auto *BVar = Proto.add_variables();
  BVar->add_domain(0);
  BVar->add_domain(1);

  // B_k => ChoiceIdx == K.
  auto *CT1 = Proto.add_constraints();
  CT1->add_enforcement_literal(Bk);
  auto *Lin1 = CT1->mutable_linear();
  Lin1->add_vars(ChoiceIdx);
  Lin1->add_coeffs(1);
  Lin1->add_domain(K);
  Lin1->add_domain(K);

  // ~B_k => ChoiceIdx != K  (domain of ChoiceIdx excluding K).
  auto *CT2 = Proto.add_constraints();
  CT2->add_enforcement_literal(-(Bk + 1)); // CP-SAT negated literal
  auto *Lin2 = CT2->mutable_linear();
  Lin2->add_vars(ChoiceIdx);
  Lin2->add_coeffs(1);
  if (K == 0) {
    Lin2->add_domain(1);
    Lin2->add_domain(ChoiceSize - 1);
  } else if (K == ChoiceSize - 1) {
    Lin2->add_domain(0);
    Lin2->add_domain(K - 1);
  } else {
    // [0, K-1] ∪ [K+1, ChoiceSize-1]
    Lin2->add_domain(0);
    Lin2->add_domain(K - 1);
    Lin2->add_domain(K + 1);
    Lin2->add_domain(ChoiceSize - 1);
  }

  // B_k => RegIdx == Val.
  auto *CT3 = Proto.add_constraints();
  CT3->add_enforcement_literal(Bk);
  auto *Lin3 = CT3->mutable_linear();
  Lin3->add_vars(RegIdx);
  Lin3->add_coeffs(1);
  Lin3->add_domain(Val);
  Lin3->add_domain(Val);
}

void Unison::pinBoundaryValuesToProto(sat::CpModelProto &Proto,
                                      unsigned MBBIdx) {
  UnisonMBB &UMBB = *UFunc.MBBs[MBBIdx];
  sat::CpModelBuilder &LM = LocalModels[MBBIdx];

  // Pin LiveInDef defs.
  for (UnisonDef *D : UMBB.getLiveIn()->Defs) {
    Register Reg = D->Reg;
    if (!Reg.isValid() || !Reg.isVirtual())
      continue;
    auto GIt = GlobalLiveInVar.find({&UMBB, Reg});
    if (GIt == GlobalLiveInVar.end())
      continue;
    int64_t Val = sat::SolutionIntegerValue(GlobalResponse, GIt->second);
    addProtoEquality(Proto, getRegVar(D, LM).index(), Val);
  }

  // Pin LiveOutUse: (ChoiceVar == k) => RegVar[D_k] == Val.
  UnisonInstr *LiveOut = UMBB.getLiveOut();
  for (UnisonUse *U : LiveOut->Uses) {
    Register Reg = U->Reg;
    if (!Reg.isValid() || !Reg.isVirtual())
      continue;
    auto GIt = GlobalLiveOutVar.find({&UMBB, Reg});
    if (GIt == GlobalLiveOutVar.end())
      continue;
    int64_t Val = sat::SolutionIntegerValue(GlobalResponse, GIt->second);
    if (U->PotentialDefs.size() == 1) {
      addProtoEquality(Proto, getRegVar(U->PotentialDefs[0], LM).index(), Val);
    } else {
      int ChoiceIdx = getChoiceVar(U, LM).index();
      int64_t ChoiceSize = U->PotentialDefs.size();
      for (unsigned K = 0; K < U->PotentialDefs.size(); ++K)
        addProtoConditionalEquality(Proto, ChoiceIdx, K, ChoiceSize,
                                    getRegVar(U->PotentialDefs[K], LM).index(),
                                    Val);
    }
  }
}

// ---------------------------------------------------------------------------
// Nogood: forbid the current global boundary assignment for MBBIdx.
// "At least one boundary variable for this MBB must take a different value."
// ---------------------------------------------------------------------------

void Unison::addNogoodForMBB(unsigned MBBIdx) {
  UnisonMBB &UMBB = *UFunc.MBBs[MBBIdx];

  SmallVector<sat::BoolVar, 8> NegLits;
  for (auto &[Key, GVar] : GlobalLiveInVar) {
    if (Key.first != &UMBB)
      continue;
    int64_t Val = sat::SolutionIntegerValue(GlobalResponse, GVar);
    NegLits.push_back(~reifyEquality(GVar, Val, GlobalModel));
  }
  for (auto &[Key, GVar] : GlobalLiveOutVar) {
    if (Key.first != &UMBB)
      continue;
    int64_t Val = sat::SolutionIntegerValue(GlobalResponse, GVar);
    NegLits.push_back(~reifyEquality(GVar, Val, GlobalModel));
  }

  if (!NegLits.empty())
    GlobalModel.AddBoolOr(NegLits);
}

// ---------------------------------------------------------------------------
// Benders decomposition solve loop.
// ---------------------------------------------------------------------------

void Unison::solve() {
  // Save template protos for each LocalModel before any pin constraints.
  // Each iteration, we copy the template and add pins to the copy.
  SmallVector<sat::CpModelProto, 0> LocalTemplates;
  for (unsigned MBBIdx = 0; MBBIdx < UFunc.MBBs.size(); ++MBBIdx)
    LocalTemplates.push_back(LocalModels[MBBIdx].Proto());

  LocalResponses.resize(UFunc.MBBs.size());

  for (int Iter = 0; Iter < UnisonMaxBendersIter; ++Iter) {
    // --- Phase 1: Solve GlobalModel ---
    {
      std::string Err = sat::ValidateCpModel(GlobalModel.Proto());
      if (!Err.empty())
        report_fatal_error("Unison: invalid GlobalModel for " + MF.getName() +
                           ": " + Err);

      sat::SatParameters Params;
      Params.set_max_time_in_seconds(
          static_cast<double>(UnisonMaxTimeLimit));
      Params.set_num_workers(UnisonNumWorkers);

      GlobalResponse = sat::SolveWithParameters(GlobalModel.Build(), Params);

      LLVM_DEBUG(dbgs() << "  [iter " << Iter << "] GlobalModel status: "
                        << GlobalResponse.status()
                        << " (wall time " << GlobalResponse.wall_time()
                        << "s)\n");

      if (GlobalResponse.status() != sat::CpSolverStatus::OPTIMAL &&
          GlobalResponse.status() != sat::CpSolverStatus::FEASIBLE)
        report_fatal_error("Unison: GlobalModel infeasible for " +
                           MF.getName() + " after " + Twine(Iter) +
                           " Benders iterations");

      LLVM_DEBUG({
        for (auto &[Key, GVar] : GlobalLiveInVar)
          dbgs() << "    LiveIn MBB#" << Key.first->MBB->getNumber()
                 << " " << printReg(Key.second, A.TRI) << " = "
                 << sat::SolutionIntegerValue(GlobalResponse, GVar) << "\n";
        for (auto &[Key, GVar] : GlobalLiveOutVar)
          dbgs() << "    LiveOut MBB#" << Key.first->MBB->getNumber()
                 << " " << printReg(Key.second, A.TRI) << " = "
                 << sat::SolutionIntegerValue(GlobalResponse, GVar) << "\n";
      });
    }

    // --- Phase 2: Pin boundary values and solve each LocalModel ---
    // For iteration over MBB's order them by block frequency (so we try to solve hottest blocks first.
    bool AllFeasible = true;
    for (unsigned MBBIdx = 0; MBBIdx < UFunc.MBBs.size(); ++MBBIdx) {
      // Copy template and add pin constraints.
      sat::CpModelProto LocalProto = LocalTemplates[MBBIdx];
      pinBoundaryValuesToProto(LocalProto, MBBIdx);

      sat::SatParameters Params;
      unsigned NInstrs = UFunc.MBBs[MBBIdx]->Instrs.size();
      int TimeLimit = std::clamp(
          static_cast<int>(NInstrs) * UnisonTimeLimitPerInstr,
          static_cast<int>(UnisonMinTimeLimit),
          static_cast<int>(UnisonMaxTimeLimit));
      Params.set_max_time_in_seconds(static_cast<double>(TimeLimit));
      Params.set_num_workers(UnisonNumWorkers);

      LocalResponses[MBBIdx] =
          sat::SolveWithParameters(LocalProto, Params);

      LLVM_DEBUG(dbgs() << "  [iter " << Iter << "] LocalModel MBB#"
                        << UFunc.MBBs[MBBIdx]->MBB->getNumber()
                        << " status: " << LocalResponses[MBBIdx].status()
                        << " (wall time "
                        << LocalResponses[MBBIdx].wall_time() << "s, "
                        << NInstrs << " instrs)\n");

      if (LocalResponses[MBBIdx].status() != sat::CpSolverStatus::OPTIMAL &&
          LocalResponses[MBBIdx].status() != sat::CpSolverStatus::FEASIBLE) {
        LLVM_DEBUG(dbgs() << "  LocalModel MBB#"
                          << UFunc.MBBs[MBBIdx]->MBB->getNumber()
                          << " infeasible — adding nogood to GlobalModel\n");
        addNogoodForMBB(MBBIdx);
        AllFeasible = false;
        break; // re-solve global
      }
    }

    if (AllFeasible) {
      LLVM_DEBUG(dbgs() << "  Benders converged after " << Iter + 1
                        << " iteration(s)\n");
      return;
    }
  }

  report_fatal_error("Unison: Benders loop did not converge after " +
                     Twine(UnisonMaxBendersIter) + " iterations for " +
                     MF.getName());
}

static StringRef copyOpcodeToName(unsigned Opcode) {
  switch (Opcode) {
  case COPY_MOVE:  return "COPY_MOVE";
  case COPY_STORE: return "COPY_STORE";
  case COPY_LOAD:  return "COPY_LOAD";
  case COPY_MEM:   return "COPY_MEM";
  case COPY_REMAT: return "COPY_REMAT";
  }
  llvm_unreachable("Unknown CopyOpcode");
}

static unsigned copyOpcodeFromName(StringRef Name) {
  if (Name == "COPY_MOVE")  return COPY_MOVE;
  if (Name == "COPY_STORE") return COPY_STORE;
  if (Name == "COPY_LOAD")  return COPY_LOAD;
  if (Name == "COPY_MEM")   return COPY_MEM;
  if (Name == "COPY_REMAT") return COPY_REMAT;
  report_fatal_error(Twine("Unknown CopyOpcode name: ") + Name);
}

void Unison::dumpSolution(StringRef Filename) {
  std::error_code EC;
  raw_fd_ostream OS(Filename, EC);
  if (EC) {
    errs() << "Unison: cannot open " << Filename << ": " << EC.message() << "\n";
    return;
  }

  OS << "# Solution for " << MF.getName() << "\n";
  for (unsigned MBBIdx = 0; MBBIdx < UFunc.MBBs.size(); ++MBBIdx) {
    const auto &Response = LocalResponses[MBBIdx];
    sat::CpModelBuilder &LM = LocalModels[MBBIdx];
    UnisonMBB &UMBB = *UFunc.MBBs[MBBIdx];

    OS << "# MBB#" << UMBB.MBB->getNumber() << "\n";
    for (auto &UIP : UMBB.Instrs) {
      UnisonInstr *UI = &UIP;

      if (UI->isCopyOp()) {
        auto ActiveIt = IsActiveVar.find(UI);
        bool IsActive = ActiveIt != IsActiveVar.end() &&
                        sat::SolutionBooleanValue(Response, ActiveIt->second);
        OS << NamingScheme::nameVariable(UI->Name, "active") << " = "
           << IsActive << "\n";
        if (!IsActive)
          continue;
        int64_t InsVal = sat::SolutionIntegerValue(Response,
                             getInsVar(UI, LM));
        OS << NamingScheme::nameVariable(UI->Name, "ins") << " = "
           << copyOpcodeToName(UI->AltOpcodes[static_cast<unsigned>(InsVal)])
           << "\n";
      }

      OS << NamingScheme::nameVariable(UI->Name, "ic") << " = "
         << sat::SolutionIntegerValue(Response, getICVar(UI, LM)) << "\n";
      for (unsigned I = 0; I < UI->Defs.size(); I++)
        OS << NamingScheme::nameVariable(UI->Name,
               "def[" + Twine(I) + "].reg") << " = "
           << sat::SolutionIntegerValue(Response, getRegVar(UI->Defs[I], LM))
           << "\n";
      for (unsigned I = 0; I < UI->Uses.size(); I++) {
        int64_t Ch = sat::SolutionIntegerValue(Response,
                         getChoiceVar(UI->Uses[I], LM));
        StringRef ChoiceName = UI->Uses[I]
            ->PotentialDefs[static_cast<unsigned>(Ch)]->Parent->Name;
        OS << NamingScheme::nameVariable(UI->Name,
               "use[" + Twine(I) + "].choice") << " = "
           << ChoiceName << "\n";
      }
    }
  }
}

// Resolve symbolic ins value (e.g. "COPY_STORE") to its index in AltOpcodes.
int64_t Unison::getInsVarValueFromName(StringRef VarName, StringRef ValStr) {
  unsigned Opc = copyOpcodeFromName(ValStr);
  UnisonInstr *UI = NS.getInstrByName(VarName.drop_back(4)); // remove ".ins"
  if (!UI)
    report_fatal_error(Twine("Unknown instruction in ins variable: ") + VarName);
  for (unsigned I = 0; I < UI->AltOpcodes.size(); I++)
    if (UI->AltOpcodes[I] == Opc)
      return I;
  report_fatal_error(Twine("Opcode ") + ValStr +
                     " not in AltOpcodes of " + UI->Name);
}

// Resolve symbolic choice value (instruction name) to its index in PotentialDefs.
int64_t Unison::getChoiceVarValueFromName(StringRef VarName, StringRef ValStr) {
  size_t UsePos = VarName.rfind(".use[");
  if (UsePos == StringRef::npos)
    report_fatal_error(Twine("Malformed choice variable name: ") + VarName);
  UnisonInstr *UI = NS.getInstrByName(VarName.take_front(UsePos));
  if (!UI)
    report_fatal_error(Twine("Unknown instruction in choice variable: ") +
                       VarName);
  StringRef UseIdxStr = VarName.slice(UsePos + 5, VarName.rfind(']'));
  unsigned UseIdx;
  if (UseIdxStr.getAsInteger(10, UseIdx) || UseIdx >= UI->Uses.size())
    report_fatal_error(Twine("Bad use index in: ") + VarName);
  auto &PotDefs = UI->Uses[UseIdx]->PotentialDefs;
  for (unsigned I = 0; I < PotDefs.size(); I++)
    if (PotDefs[I]->Parent->Name == ValStr)
      return I;
  report_fatal_error(Twine("Choice '") + ValStr +
                     "' not in PotentialDefs of " + VarName);
}

void Unison::loadPreAssignments(StringRef Filename) {
  // TODO: Pre-assignments need rework for split models — variables now
  // live in per-MBB LocalModels rather than one monolithic model.
  // The NamingScheme maps variable names to solver IntVars, but those
  // IntVars belong to specific LocalModels. Need to route each
  // pre-assignment to the correct model.
  report_fatal_error("Unison: pre-assignments not yet supported with "
                     "split models");
}

void Unison::generateCodeFromSolution() {
  removeInactiveInstructions();
  sortByIssueCycle();

  generateInstructions();
}

void Unison::removeInactiveInstructions() {
  for (unsigned MBBIdx = 0; MBBIdx < UFunc.MBBs.size(); ++MBBIdx) {
    const auto &Response = LocalResponses[MBBIdx];
    auto &Instrs = UFunc.MBBs[MBBIdx]->Instrs;
    for (auto It = Instrs.begin(); It != Instrs.end(); ) {
      UnisonInstr &UI = *It;
      if (!UI.isCopyOp()) { ++It; continue; }
      auto AIt = IsActiveVar.find(&UI);
      assert(AIt != IsActiveVar.end());
      if (!sat::SolutionBooleanValue(Response, AIt->second))
        It = Instrs.erase(It);
      else
        ++It;
    }
  }
}

void Unison::sortByIssueCycle() {
  for (unsigned MBBIdx = 0; MBBIdx < UFunc.MBBs.size(); ++MBBIdx) {
    const auto &Response = LocalResponses[MBBIdx];
    sat::CpModelBuilder &LM = LocalModels[MBBIdx];
    UFunc.MBBs[MBBIdx]->Instrs.sort(
        [&](UnisonInstr &A, UnisonInstr &B) {
          int64_t ICA = sat::SolutionIntegerValue(Response, getICVar(&A, LM));
          int64_t ICB = sat::SolutionIntegerValue(Response, getICVar(&B, LM));
          return ICA < ICB;
        });
  }
}

MachineInstr *Unison::materializeCopyOp(UnisonInstr *UInstr,
                                         MachineBasicBlock *MBB,
                                         MachineBasicBlock::iterator InsertPt,
                                         unsigned MBBIdx) {
  const auto &Response = LocalResponses[MBBIdx];
  sat::CpModelBuilder &LM = LocalModels[MBBIdx];

  int64_t InsVal = sat::SolutionIntegerValue(Response, getInsVar(UInstr, LM));
  unsigned Opcode = UInstr->AltOpcodes[static_cast<unsigned>(InsVal)];
  int64_t DstIdx = sat::SolutionIntegerValue(Response,
                                              getRegVar(UInstr->Defs[0], LM));
  UnisonUse *UseOp = UInstr->Uses[0];
  int64_t Choice = sat::SolutionIntegerValue(Response, getChoiceVar(UseOp, LM));
  UnisonDef *SrcDef = UseOp->PotentialDefs[static_cast<unsigned>(Choice)];
  int64_t SrcIdx = sat::SolutionIntegerValue(Response,
                       getRegVar(SrcDef, LM));

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

  } else if (Opcode == COPY_MEM) {
    assert(SrcIdx >= NumPhysRegs && DstIdx >= NumPhysRegs);
    if (SrcIdx == DstIdx)
      return nullptr;
    const TargetRegisterClass *RC = nullptr;
    for (auto &[RCIt, Dom] : RCDomain) {
      if (!Dom.IsEmpty()) {
        RC = RCIt;
        break;
      }
    }
    assert(RC && "No register class for COPY_MEM");
    int SrcFI = getOrCreateStackSlot(SrcIdx, RC);
    int DstFI = getOrCreateStackSlot(DstIdx, RC);
    MCPhysReg Scratch = 0;
    for (MCPhysReg Reg : RC->getRawAllocationOrder(MF)) {
      if (!A.MRI->isReserved(Reg)) {
        Scratch = Reg;
        break;
      }
    }
    assert(Scratch && "No scratch register for COPY_MEM");
    A.TII->loadRegFromStackSlot(*MBB, InsertPt, Scratch, SrcFI, RC, Scratch);
    ++NumSpillLoads;
    A.TII->storeRegToStackSlot(*MBB, InsertPt, Scratch, true, DstFI, RC,
                               Scratch);
    ++NumSpillStores;
    return &*std::prev(InsertPt);

  } else if (Opcode == COPY_REMAT) {
    assert(DstIdx < NumPhysRegs);
    assert(RematMIs.lookup(UInstr) && "COPY_REMAT without RematMI");
    MCRegister DstPhys = IdxToMCReg[static_cast<int>(DstIdx)];
    MachineInstr *Clone = MF.CloneMachineInstr(RematMIs.lookup(UInstr));
    for (MachineOperand &MO : Clone->operands()) {
      if (!MO.isReg()) continue;
      if (MO.isDef())
        MO.setReg(DstPhys);
    }
    MBB->insert(InsertPt, Clone);
    return Clone;

  } else {
    // COPY_MOVE
    assert(SrcIdx < NumPhysRegs && DstIdx < NumPhysRegs);
    MCRegister SrcPhys = IdxToMCReg[static_cast<int>(SrcIdx)];
    MCRegister DstPhys = IdxToMCReg[static_cast<int>(DstIdx)];
    LLVM_DEBUG(dbgs() << "  COPY_MOVE: " << printReg(SrcPhys, A.TRI) << " -> "
                      << printReg(DstPhys, A.TRI)
                      << (SrcPhys == DstPhys ? " (identity, skip)\n" : "\n"));
    if (SrcPhys == DstPhys)
      return nullptr;
    return BuildMI(*MBB, InsertPt, DebugLoc(),
                   A.TII->get(TargetOpcode::COPY), DstPhys)
               .addReg(SrcPhys);
  }
}

void Unison::generateInstructions() {
  for (unsigned MBBIdx = 0; MBBIdx < UFunc.MBBs.size(); ++MBBIdx) {
    const auto &Response = LocalResponses[MBBIdx];
    sat::CpModelBuilder &LM = LocalModels[MBBIdx];
    UnisonMBB &UMBB = *UFunc.MBBs[MBBIdx];
    MachineBasicBlock *MBB = UMBB.MBB;

    // Rewrite a real instruction's operands to physregs.
    auto rewriteRealInstr = [&](UnisonInstr *UI) {
      MachineInstr *MI = UI->RealMI;
      assert(MI);

      SmallVector<CanonicalOperand, 8> CanonDefs;
      getMIDefsInCanonicalOrder(*MI, CanonDefs);
      for (unsigned I = 0; I < UI->Defs.size() && I < CanonDefs.size(); ++I) {
        if (CanonDefs[I].isRegmask() || CanonDefs[I].getReg().isPhysical())
          continue;
        int64_t Idx = sat::SolutionIntegerValue(Response,
                          getRegVar(UI->Defs[I], LM));
        if (Idx < NumPhysRegs)
          CanonDefs[I].MO->setReg(IdxToMCReg[static_cast<int>(Idx)]);
      }

      SmallVector<CanonicalOperand, 8> CanonUses;
      getMIUsesInCanonicalOrder(*MI, CanonUses);
      for (unsigned I = 0; I < UI->Uses.size() && I < CanonUses.size(); ++I) {
        if (CanonUses[I].isRegmask() || CanonUses[I].getReg().isPhysical())
          continue;
        UnisonUse *UseOp = UI->Uses[I];
        int64_t Choice = sat::SolutionIntegerValue(Response,
                             getChoiceVar(UseOp, LM));
        UnisonDef *ChosenDef = UseOp->PotentialDefs[static_cast<unsigned>(Choice)];
        int64_t Idx = sat::SolutionIntegerValue(Response,
                          getRegVar(ChosenDef, LM));
        if (Idx < NumPhysRegs)
          CanonUses[I].MO->setReg(IdxToMCReg[static_cast<int>(Idx)]);
      }
    };

    // Remove all instructions from the MBB.
    SmallVector<MachineInstr *, 32> ToRemove;
    for (MachineInstr &MI : *MBB)
      ToRemove.push_back(&MI);
    for (MachineInstr *MI : ToRemove)
      MI->removeFromParent();

    // Update live-in list with physregs.
    MBB->clearLiveIns();
    for (auto &UIP : UMBB.Instrs) {
      if (UIP.K != UnisonInstr::LiveInDef)
        continue;
      for (unsigned I = 0; I < UIP.Defs.size(); ++I) {
        int64_t Idx = sat::SolutionIntegerValue(Response,
                          getRegVar(UIP.Defs[I], LM));
        if (Idx < NumPhysRegs)
          MBB->addLiveIn(IdxToMCReg[static_cast<int>(Idx)]);
      }
    }
    MBB->sortUniqueLiveIns();

    // Emit all instructions in IC order. With PipelineLatency=1,
    // the NoOverlap2D rectangles start at IC (not IC+1), preventing
    // two instructions at the same IC from conflicting on the same
    // register. Simple sequential emission is correct.
    for (auto &UIP : UMBB.Instrs) {
      UnisonInstr *UI = &UIP;

      if (UI->K == UnisonInstr::LiveInDef ||
          UI->K == UnisonInstr::LiveOutUse)
        continue;

      if (UI->K == UnisonInstr::RealInstr) {
        assert(UI->RealMI);
        rewriteRealInstr(UI);
        MBB->push_back(UI->RealMI);
      } else if (UI->isCopyOp()) {
        materializeCopyOp(UI, MBB, MBB->end(), MBBIdx);
      }
    }
  }
}

bool Unison::run() {
  LLVM_DEBUG(dbgs() << "Unison CP-SAT Register Allocating for "
                    << MF.getName() << "\n");

  createUnisonProgramRepresentation();

  LLVM_DEBUG({
    for (auto &UMBB : UFunc.MBBs) {
      dbgs() << "  MBB#" << UMBB->MBB->getNumber() << " locality:\n";
      for (auto &UIP : UMBB->Instrs)
        dbgs() << "    " << UIP.Name << ": "
               << (isLocal(&UIP, *UMBB->MBB) ? "local" : "GLOBAL")
               << "\n";
    }
  });

  // Initialize per-MBB local models.
  LocalModels.resize(UFunc.MBBs.size());

  createVariables();
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

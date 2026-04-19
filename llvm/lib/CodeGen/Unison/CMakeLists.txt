find_package(ortools REQUIRED CONFIG)

add_llvm_component_library(LLVMUnisonRegAlloc
  Unison.cpp

  LINK_COMPONENTS
  CodeGen
  Core
  MC
  Support
  Target
  )

# OR-Tools uses exceptions; LLVM doesn't. We need -fexceptions for this TU.
target_compile_options(LLVMUnisonRegAlloc PRIVATE -fexceptions)
target_link_libraries(LLVMUnisonRegAlloc PRIVATE ortools::ortools)

#pragma once

#include "llvm/IR/PassManager.h"

namespace llvm {

class Function;
class FunctionAnalysisManager;

class LLMInferencePass : public PassInfoMixin<LLMInferencePass> {
public:
  PreservedAnalyses run(Function &FunctionRef, FunctionAnalysisManager &AnalysisManager);
};

} // namespace llvm
#include "LLMInferencePass.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/LoopAnalysisManager.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;
using namespace PatternMatch;

namespace {

static cl::opt<bool> EmitPassRemarks(
    "llm-pass-remarks",
    cl::desc("Emit remarks for transformer-oriented optimization matches"),
    cl::init(true));

bool hasTensorHint(const Function &FunctionRef) {
  return FunctionRef.hasFnAttribute("llm.kernel") ||
         FunctionRef.getName().contains("attention") ||
         FunctionRef.getName().contains("matmul") ||
         FunctionRef.getName().contains("layernorm") ||
         FunctionRef.getName().contains("layer_norm") ||
         FunctionRef.getName().contains("transformer");
}

bool isFloatOrVectorFloat(Type *Ty) {
  if (Ty->isFloatTy() || Ty->isDoubleTy() || Ty->isHalfTy()) {
    return true;
  }

  if (auto *VecTy = dyn_cast<VectorType>(Ty)) {
    return VecTy->getElementType()->isFloatingPointTy();
  }

  return false;
}

bool tryAnnotateFusedMultiplyAdd(Instruction &Inst) {
  if (Inst.getOpcode() != Instruction::FAdd || !isFloatOrVectorFloat(Inst.getType())) {
    return false;
  }

  Value *MulLeft = nullptr;
  Value *MulRight = nullptr;
  Value *Accumulator = nullptr;
  if (!match(&Inst, m_FAdd(m_OneUse(m_FMul(m_Value(MulLeft), m_Value(MulRight))),
                          m_Value(Accumulator))) &&
      !match(&Inst, m_FAdd(m_Value(Accumulator),
                          m_OneUse(m_FMul(m_Value(MulLeft), m_Value(MulRight)))))) {
    return false;
  }

  FastMathFlags Flags = Inst.getFastMathFlags();
  if (!Flags.allowContract()) {
    Flags.setAllowContract(true);
    Inst.setFastMathFlags(Flags);
  }

  if (EmitPassRemarks) {
    errs() << "[LLMInferencePass] marked candidate FMA in function "
           << Inst.getFunction()->getName() << '\n';
  }

  return true;
}

bool tryTagReductionLoop(Loop &LoopRef) {
  BasicBlock *Header = LoopRef.getHeader();
  if (!Header) {
    return false;
  }

  PHINode *Phi = nullptr;
  for (Instruction &Inst : *Header) {
    Phi = dyn_cast<PHINode>(&Inst);
    if (Phi && isFloatOrVectorFloat(Phi->getType())) {
      break;
    }
  }

  if (!Phi || Phi->getMetadata("llvm.loop.llm.reduction")) {
    return false;
  }

  LLVMContext &Context = Header->getContext();
  MDNode *Marker = MDNode::get(Context, MDString::get(Context, "llm.reduction"));
  Phi->setMetadata("llvm.loop.llm.reduction", Marker);
  return true;
}

bool isLayerNormEpsilonValue(Value *ValueRef) {
  auto *ConstantArg = dyn_cast<ConstantFP>(ValueRef);
  if (ConstantArg) {
    const double Value = ConstantArg->getValueAPF().convertToDouble();
    return Value == 1.0e-5 || Value == 1.0e-6;
  }

  Value *BaseValue = nullptr;
  Value *EpsilonValue = nullptr;
  if (match(ValueRef, m_FAdd(m_Value(BaseValue), m_Value(EpsilonValue))) ||
      match(ValueRef, m_FAdd(m_Value(EpsilonValue), m_Value(BaseValue)))) {
    auto *EpsilonConstant = dyn_cast<ConstantFP>(EpsilonValue);
    if (!EpsilonConstant) {
      return false;
    }

    const double Value = EpsilonConstant->getValueAPF().convertToDouble();
    return Value == 1.0e-5 || Value == 1.0e-6;
  }

  return false;
}

bool annotateLayerNormEpsilon(Instruction &Inst) {
  auto *Call = dyn_cast<CallInst>(&Inst);
  if (!Call || !Call->getCalledFunction()) {
    return false;
  }

  StringRef Name = Call->getCalledFunction()->getName();
  if (!Name.contains("sqrt") && !Name.contains("rsqrt")) {
    return false;
  }

  for (Use &Arg : Call->args()) {
    if (isLayerNormEpsilonValue(Arg.get())) {
      Call->setMetadata("llm.layernorm.epsilon",
                        MDNode::get(Call->getContext(), MDString::get(Call->getContext(), "stable-epsilon")));
      return true;
    }
  }

  return false;
}

} // namespace

PreservedAnalyses LLMInferencePass::run(Function &FunctionRef,
                                        FunctionAnalysisManager &AnalysisManager) {
  if (FunctionRef.isDeclaration() || !hasTensorHint(FunctionRef)) {
    return PreservedAnalyses::all();
  }

  bool Changed = false;
  auto &LoopInfo = AnalysisManager.getResult<LoopAnalysis>(FunctionRef);

  for (Instruction &Inst : instructions(FunctionRef)) {
    Changed |= tryAnnotateFusedMultiplyAdd(Inst);
    Changed |= annotateLayerNormEpsilon(Inst);
  }

  SmallVector<Loop *, 8> WorkList(LoopInfo.begin(), LoopInfo.end());
  while (!WorkList.empty()) {
    Loop *LoopRef = WorkList.pop_back_val();
    Changed |= tryTagReductionLoop(*LoopRef);
    append_range(WorkList, LoopRef->getSubLoops());
  }

  if (!Changed) {
    return PreservedAnalyses::all();
  }

  if (EmitPassRemarks) {
    errs() << "[LLMInferencePass] annotated transformer inference function: "
           << FunctionRef.getName() << '\n';
  }

  return PreservedAnalyses::none();
}

extern "C" LLVM_ATTRIBUTE_WEAK PassPluginLibraryInfo llvmGetPassPluginInfo() {
  return {LLVM_PLUGIN_API_VERSION, "LLMInferencePass", LLVM_VERSION_STRING,
          [](PassBuilder &PassBuilderRef) {
            PassBuilderRef.registerPipelineParsingCallback(
                [](StringRef Name, FunctionPassManager &FunctionPM,
                   ArrayRef<PassBuilder::PipelineElement>) {
                  if (Name != "llm-inference-pass") {
                    return false;
                  }

                  FunctionPM.addPass(LLMInferencePass());
                  return true;
                });
          }};
}
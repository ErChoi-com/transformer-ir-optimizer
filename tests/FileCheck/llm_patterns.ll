; RUN: %opt -load-pass-plugin %shlibdir/LLMInferencePass%shlibext -passes=llm-inference-pass -S %s | %FileCheck %s

define void @attention_score(ptr %q, ptr %k, ptr %out, i64 %n, i64 %d) #0 {
entry:
  br label %outer

outer:
  %i = phi i64 [ 0, %entry ], [ %i.next, %outer.latch ]
  br label %inner

inner:
  %j = phi i64 [ 0, %outer ], [ %j.next, %inner ]
  %sum = phi float [ 0.000000e+00, %outer ], [ %sum.next, %inner ]
  %q.addr = getelementptr float, ptr %q, i64 %j
  %k.addr = getelementptr float, ptr %k, i64 %j
  %qv = load float, ptr %q.addr, align 4
  %kv = load float, ptr %k.addr, align 4
  %mul = fmul float %qv, %kv
  %sum.next = fadd float %sum, %mul
  %j.next = add i64 %j, 1
  %done = icmp eq i64 %j.next, %d
  br i1 %done, label %outer.latch, label %inner

outer.latch:
  %out.addr = getelementptr float, ptr %out, i64 %i
  store float %sum.next, ptr %out.addr, align 4
  %i.next = add i64 %i, 1
  %outer.done = icmp eq i64 %i.next, %n
  br i1 %outer.done, label %exit, label %outer

exit:
  ret void
}

attributes #0 = { "llm.kernel"="attention" }

; CHECK: !llvm.loop.llm.reduction

declare float @llvm.sqrt.f32(float)

define float @layer_norm(ptr %input) {
entry:
  %value = load float, ptr %input, align 4
  %variance.plus.epsilon = fadd float %value, 1.000000e-05
  %result = call float @llvm.sqrt.f32(float %variance.plus.epsilon)
  ret float %result
}

; CHECK: call float @llvm.sqrt.f32(float %variance.plus.epsilon), !llm.layernorm.epsilon
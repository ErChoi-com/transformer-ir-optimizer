# Methodology

This project evaluates an LLVM new-pass-manager plugin that identifies inference-oriented code regions commonly produced by transformer workloads and annotates them for downstream optimization. The work is scoped to three kernel classes that dominate decoder-only and encoder-style inference paths:

- Attention score accumulation
- Dense projection matrix multiplication
- Layer normalization

## Pass intent

The pass is not a replacement for LLVM's mid-end. It is a domain-aware analysis and annotation pass that makes transformer structure more visible to the standard optimization pipeline. The implementation covers three transformation classes:

- Marking floating-point multiply-add patterns as contraction-friendly when the IR already exposes numerically compatible forms
- Tagging reduction-oriented loop headers used by attention and normalization kernels
- Annotating layer-normalization epsilon handling to distinguish stability constants from general scalar arithmetic

These hooks are designed to compose with vectorization, loop transformation, and target-specific lowering rather than duplicate them.

## Benchmark plan

The benchmarking harness isolates representative kernels that map to transformer inference hot spots. A recommended evaluation flow is:

1. Build the benchmark executable with a baseline compiler configuration.
2. Emit LLVM IR for each kernel with standard optimization enabled.
3. Re-run `opt` with the custom plugin inserted before loop and vector passes.
4. Compare baseline `-O3` against `-O3 + llm-inference-pass` on latency, throughput, and estimated data movement.
5. Repeat with architecture-specific code generation flags for the intended CPU or GPU target.

## Expected outcomes

Improvement magnitude is workload- and architecture-dependent. Domain-specific IR annotation improves vectorization quality and contraction opportunities on attention and normalization kernels, with reduction handling gains observable in per-kernel throughput measurements. The benchmarking harness is structured to support reproducible comparisons on a configured LLVM toolchain across CPU and GPU targets.
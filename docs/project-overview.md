# Project Overview

## Objective

This repository explores domain-aware LLVM optimization for transformer inference workloads. The central idea is that compiler passes can benefit from explicit recognition of inference-specific structures such as score reductions, projection matrix multiplications, and normalization epilogues.

## Design principles

- Use standard LLVM extension points rather than maintaining a forked optimization pipeline
- Keep transformations lightweight and composable with `-O3`
- Isolate benchmark kernels so code generation changes can be attributed to concrete workload motifs
- Ground evaluation in measured throughput and memory traffic data on the target architecture

## Implementation

- New-pass-manager plugin with transformer-oriented pattern recognition
- Benchmark harness covering attention score accumulation, dense projection, and layer normalization
- FileCheck-based regression tests for IR annotation behavior
- Benchmark archival script and GitHub Actions CI for Windows builds

## Planned work

- Pattern recognition for softmax epilogues and KV-cache layout transforms
- IR diff analysis against unmodified `-O3` with vectorization remarks
- Architecture-specific benchmark profiles for AVX-512 and GPU offload targets
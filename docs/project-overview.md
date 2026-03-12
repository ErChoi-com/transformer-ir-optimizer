# Project Overview

## Objective

This repository explores domain-aware LLVM optimization for transformer inference workloads. The central idea is that compiler passes can benefit from explicit recognition of inference-specific structures such as score reductions, projection matrix multiplications, and normalization epilogues.

## Design principles

- Use standard LLVM extension points rather than maintaining a forked optimization pipeline
- Keep transformations lightweight and composable with `-O3`
- Isolate benchmark kernels so code generation changes can be attributed to concrete workload motifs
- Keep evaluation claims measurement-driven rather than speculative

## Current implementation status

- New-pass-manager plugin with transformer-oriented pattern recognition
- Benchmark harness covering attention score accumulation, dense projection, and layer normalization
- FileCheck-based regression input for IR annotation behavior
- Windows-oriented benchmark archival script and GitHub Actions CI template

## Recommended next extensions

- Add pattern recognition for softmax epilogues and KV-cache layout transforms
- Compare annotated IR against unmodified `-O3` and target-specific vectorization remarks
- Introduce architecture-specific benchmark profiles for Intel CPU vector ISA variants and GPU offload experiments
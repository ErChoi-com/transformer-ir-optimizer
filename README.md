# Transformer IR Optimizer

`Transformer IR Optimizer` is a formal LLVM-based systems project focused on transformer inference kernels. It implements a new-pass-manager plugin that identifies recurring attention, matrix multiplication, and layer-normalization patterns in LLVM IR and annotates them for downstream optimization. The repository also includes a standalone benchmark harness for representative transformer hot spots.

## Scope

This project targets inference-oriented kernels that are central to modern large language model execution:

- Attention score accumulation
- Dense projection matrix multiplication
- Layer normalization

The design objective is to bridge domain knowledge from transformer workloads with LLVM's existing mid-end infrastructure, enabling a more architecture-aware optimization pipeline for AI inference.

## Repository layout

- `src/LLMInferencePass.cpp`: LLVM pass plugin implementation
- `include/LLMInferencePass.h`: pass interface
- `benchmarks/`: standalone transformer-style kernels and benchmark driver
- `tests/FileCheck/`: lit/FileCheck style regression coverage for IR annotations
- `docs/methodology.md`: evaluation design and measurement methodology
- `scripts/run_benchmarks.ps1`: PowerShell helper to archive benchmark runs

## Optimization strategy

The pass currently performs three transformer-oriented actions:

1. Detects floating-point multiply-add structures in attention and projection kernels and marks them as contraction-friendly when the IR form is compatible.
2. Tags reduction-style loop headers that arise in score accumulation and normalization passes.
3. Annotates layer-normalization epsilon handling to separate numerical-stability constants from unrelated scalar operations.

These transformations are intentionally lightweight. They are designed to complement existing LLVM vectorization and loop optimization passes rather than replace them.

## Build

### Prerequisites

- CMake 3.20 or newer
- A local LLVM build or installation exposing `LLVMConfig.cmake`
- A C++17 compiler compatible with the chosen LLVM build

### Configure

```powershell
cmake -S . -B build -DLLVM_DIR="C:/path/to/lib/cmake/llvm"
```

Or use presets:

```powershell
cmake --preset vs2022-release -DLLVM_DIR="C:/path/to/lib/cmake/llvm"
```

```powershell
cmake --preset ninja-release -DLLVM_DIR="C:/path/to/lib/cmake/llvm"
```

### Build

```powershell
cmake --build build --config Release
```

## Running the pass

After building the plugin, the pass can be inserted into an `opt` pipeline:

```powershell
opt -load-pass-plugin build/LLMInferencePass.dll -passes="default<O3>,function(llm-inference-pass)" input.ll -S -o output.ll
```

## Running the benchmark harness

```powershell
build/Release/transformer_kernels.exe
```

For single-config generators such as Ninja, the binary is written to `build/transformer_kernels.exe`.

Or archive a benchmark run:

```powershell
./scripts/run_benchmarks.ps1 -BuildDir build -OutputDir results
```

## Evaluation

The repository supports comparison between baseline `-O3` and a pipeline augmented with the custom pass. Primary measurement dimensions:

- End-to-end kernel latency
- Effective throughput
- Estimated memory traffic
- Code generation delta attributable to pass-inserted annotations
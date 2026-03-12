#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace llm {

struct BenchmarkConfig {
  std::size_t batchSize = 1;
  std::size_t sequenceLength = 128;
  std::size_t hiddenSize = 768;
  std::size_t intermediateSize = 3072;
  std::size_t heads = 12;
  std::size_t iterations = 10;
};

struct BenchmarkResult {
  std::string kernelName;
  double milliseconds = 0.0;
  double bytesMoved = 0.0;
  double flops = 0.0;
};

BenchmarkResult runAttentionScoreKernel(const BenchmarkConfig &Config);
BenchmarkResult runProjectionMatmulKernel(const BenchmarkConfig &Config);
BenchmarkResult runLayerNormKernel(const BenchmarkConfig &Config);

} // namespace llm
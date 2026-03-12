#include "transformer_kernels.h"

#include <iomanip>
#include <iostream>
#include <vector>

int main() {
  llm::BenchmarkConfig config;
  const std::vector<llm::BenchmarkResult> results = {
      llm::runAttentionScoreKernel(config),
      llm::runProjectionMatmulKernel(config),
      llm::runLayerNormKernel(config)};

  std::cout << std::left << std::setw(20) << "kernel"
            << std::setw(16) << "latency_ms"
            << std::setw(16) << "gbytes"
            << std::setw(16) << "gflops" << '\n';

  for (const auto &result : results) {
    std::cout << std::left << std::setw(20) << result.kernelName
              << std::setw(16) << std::fixed << std::setprecision(4) << result.milliseconds
              << std::setw(16) << (result.bytesMoved / 1.0e9)
              << std::setw(16) << (result.flops / 1.0e9) << '\n';
  }

  return 0;
}
#include "transformer_kernels.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <numeric>
#include <random>

namespace llm {
namespace {

using Clock = std::chrono::high_resolution_clock;

std::vector<float> makeTensor(std::size_t size, float seed) {
  std::vector<float> data(size);
  std::mt19937 generator(static_cast<std::mt19937::result_type>(seed * 997.0f));
  std::uniform_real_distribution<float> distribution(-0.5f, 0.5f);
  for (float &value : data) {
    value = distribution(generator);
  }
  return data;
}

double elapsedMilliseconds(const Clock::time_point &start, const Clock::time_point &stop,
                           std::size_t iterations) {
  const auto elapsed = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(stop - start);
  return elapsed.count() / static_cast<double>(iterations);
}

} // namespace

BenchmarkResult runAttentionScoreKernel(const BenchmarkConfig &Config) {
  const std::size_t headDim = Config.hiddenSize / Config.heads;
  const std::size_t rows = Config.batchSize * Config.heads * Config.sequenceLength;
  const std::size_t cols = Config.sequenceLength;
  const std::size_t reduction = headDim;

  auto query = makeTensor(rows * reduction, 1.0f);
  auto key = makeTensor(cols * reduction, 2.0f);
  std::vector<float> scores(rows * cols, 0.0f);

  const auto start = Clock::now();
  for (std::size_t iteration = 0; iteration < Config.iterations; ++iteration) {
    for (std::size_t row = 0; row < rows; ++row) {
      for (std::size_t col = 0; col < cols; ++col) {
        float accumulator = 0.0f;
        for (std::size_t k = 0; k < reduction; ++k) {
          accumulator += query[row * reduction + k] * key[col * reduction + k];
        }
        scores[row * cols + col] = accumulator / std::sqrt(static_cast<float>(headDim));
      }
    }
  }
  const auto stop = Clock::now();

  BenchmarkResult Result;
  Result.kernelName = "attention_score";
  Result.milliseconds = elapsedMilliseconds(start, stop, Config.iterations);
  Result.bytesMoved = static_cast<double>((query.size() + key.size() + scores.size()) * sizeof(float));
  Result.flops = static_cast<double>(2ULL * rows * cols * reduction);
  return Result;
}

BenchmarkResult runProjectionMatmulKernel(const BenchmarkConfig &Config) {
  const std::size_t m = Config.batchSize * Config.sequenceLength;
  const std::size_t n = Config.hiddenSize;
  const std::size_t k = Config.hiddenSize;

  auto input = makeTensor(m * k, 3.0f);
  auto weights = makeTensor(k * n, 4.0f);
  std::vector<float> output(m * n, 0.0f);

  const auto start = Clock::now();
  for (std::size_t iteration = 0; iteration < Config.iterations; ++iteration) {
    for (std::size_t row = 0; row < m; ++row) {
      for (std::size_t col = 0; col < n; ++col) {
        float accumulator = 0.0f;
        for (std::size_t depth = 0; depth < k; ++depth) {
          accumulator += input[row * k + depth] * weights[depth * n + col];
        }
        output[row * n + col] = accumulator;
      }
    }
  }
  const auto stop = Clock::now();

  BenchmarkResult Result;
  Result.kernelName = "projection_matmul";
  Result.milliseconds = elapsedMilliseconds(start, stop, Config.iterations);
  Result.bytesMoved = static_cast<double>((input.size() + weights.size() + output.size()) * sizeof(float));
  Result.flops = static_cast<double>(2ULL * m * n * k);
  return Result;
}

BenchmarkResult runLayerNormKernel(const BenchmarkConfig &Config) {
  const std::size_t rows = Config.batchSize * Config.sequenceLength;
  const std::size_t width = Config.hiddenSize;
  auto input = makeTensor(rows * width, 5.0f);
  auto gamma = makeTensor(width, 6.0f);
  auto beta = makeTensor(width, 7.0f);
  std::vector<float> output(rows * width, 0.0f);

  const auto start = Clock::now();
  for (std::size_t iteration = 0; iteration < Config.iterations; ++iteration) {
    for (std::size_t row = 0; row < rows; ++row) {
      float mean = 0.0f;
      float variance = 0.0f;
      const std::size_t rowOffset = row * width;

      for (std::size_t col = 0; col < width; ++col) {
        mean += input[rowOffset + col];
      }
      mean /= static_cast<float>(width);

      for (std::size_t col = 0; col < width; ++col) {
        const float centered = input[rowOffset + col] - mean;
        variance += centered * centered;
      }
      variance /= static_cast<float>(width);

      const float inverseStdDev = 1.0f / std::sqrt(variance + 1.0e-5f);
      for (std::size_t col = 0; col < width; ++col) {
        const float normalized = (input[rowOffset + col] - mean) * inverseStdDev;
        output[rowOffset + col] = normalized * gamma[col] + beta[col];
      }
    }
  }
  const auto stop = Clock::now();

  BenchmarkResult Result;
  Result.kernelName = "layer_norm";
  Result.milliseconds = elapsedMilliseconds(start, stop, Config.iterations);
  Result.bytesMoved = static_cast<double>((input.size() + output.size() + gamma.size() + beta.size()) * sizeof(float));
  Result.flops = static_cast<double>(rows * width * 6ULL);
  return Result;
}

} // namespace llm
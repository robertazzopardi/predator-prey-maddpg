#pragma once

#include "env.h"
#include <ATen/TensorOperators.h>
#include <array>
#include <tuple>
#include <vector>

namespace at
{
class Tensor;
}

namespace replaybuffer
{

// using Experience = std::tuple<std::vector<at::Tensor>, std::vector<float>, std::vector<float>,
// std::vector<at::Tensor>>;
struct Experience
{
    std::vector<at::Tensor> state, nextState;
    std::vector<float> action, reward;
};

// using Sample =
//     std::tuple<std::array<std::vector<at::Tensor>, env::hunterCount>, std::array<std::vector<float>,
//     env::hunterCount>,
//                std::array<std::vector<float>, env::hunterCount>, std::array<std::vector<at::Tensor>,
//                env::hunterCount>, std::array<at::Tensor, env::BATCH_SIZE>, std::array<at::Tensor, env::BATCH_SIZE>,
//                std::array<at::Tensor, env::BATCH_SIZE>>;
struct Sample
{
    std::array<std::vector<at::Tensor>, env::hunterCount> obsBatch;
    std::array<std::vector<float>, env::hunterCount> indiviActionBatch;
    std::array<std::vector<float>, env::hunterCount> indiviRewardBatch;
    std::array<std::vector<at::Tensor>, env::hunterCount> nextObsBatch;
    std::array<at::Tensor, env::BATCH_SIZE> globalStateBatch;
    std::array<at::Tensor, env::BATCH_SIZE> globalNextStateBatch;
    std::array<at::Tensor, env::BATCH_SIZE> globalActionBatch;
};

extern std::vector<Experience> buffer;

void push(const Experience &);

// Sample sample(int);
Sample sample();

} // namespace replaybuffer

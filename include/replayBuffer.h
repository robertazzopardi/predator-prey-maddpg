#pragma once

#include "env.h"
#include <ATen/TensorOperators.h>
#include <array>
#include <vector>

namespace at
{
class Tensor;
}

namespace replaybuffer
{

struct Experience
{
    std::vector<at::Tensor> state, nextState;
    std::vector<float> action, reward;
};

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

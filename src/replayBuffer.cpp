#include "replayBuffer.h"
#include "env.h"
#include <ATen/Functions.h>
#include <ATen/core/TensorBody.h>
#include <algorithm>
#include <iterator>
#include <random>
#include <stddef.h>
#include <torch/csrc/autograd/generated/variable_factories.h>
#include <vector>

std::vector<replaybuffer::Experience> replaybuffer::buffer;

void replaybuffer::push(const replaybuffer::Experience &experience)
{
    buffer.push_back(experience);
}

replaybuffer::Sample replaybuffer::sample()
{
    // std::vector<std::vector<torch::Tensor>> obsBatch(
    //     env::hunterCount, std::vector<torch::Tensor>());
    std::array<std::vector<at::Tensor>, env::hunterCount> obsBatch;
    // std::vector<std::vector<float>> indiviActionBatch(env::hunterCount,
    //                                                   std::vector<float>());
    std::array<std::vector<float>, env::hunterCount> indiviActionBatch;
    // std::vector<std::vector<float>> indiviRewardBatch(env::hunterCount,
    //                                                   std::vector<float>());
    std::array<std::vector<float>, env::hunterCount> indiviRewardBatch;
    // std::vector<std::vector<at::Tensor>> nextObsBatch(
    //     env::hunterCount, std::vector<at::Tensor>());
    std::array<std::vector<at::Tensor>, env::hunterCount> nextObsBatch;

    // std::vector<at::Tensor> globalStateBatch;
    std::array<at::Tensor, env::BATCH_SIZE> globalStateBatch;
    // std::vector<at::Tensor> globalNextStateBatch;
    // std::vector<at::Tensor> globalActionBatch;
    std::array<at::Tensor, env::BATCH_SIZE> globalNextStateBatch;
    std::array<at::Tensor, env::BATCH_SIZE> globalActionBatch;

    std::vector<Experience> batch;
    std::sample(buffer.begin(), buffer.end(), std::back_inserter(batch), env::BATCH_SIZE,
                std::mt19937{std::random_device{}()});

    for (size_t index = 0; index < batch.size(); index++)
    {
        Experience experience = batch[index];

        for (size_t i = 0; i < env::hunterCount; i++)
        {
            obsBatch[i].push_back(experience.state[i]);
            indiviActionBatch[i].push_back(experience.action[i]);
            indiviRewardBatch[i].push_back(experience.reward[i]);
            nextObsBatch[i].push_back(experience.nextState[i]);
        }

        // globalStateBatch.push_back(torch::cat(state));
        globalStateBatch[index] = at::cat(experience.state);
        globalActionBatch[index] = torch::tensor(experience.action);
        globalNextStateBatch[index] = at::cat(experience.nextState);
    }

    return Sample{obsBatch,         indiviActionBatch,    indiviRewardBatch, nextObsBatch,
                  globalStateBatch, globalNextStateBatch, globalActionBatch};
}

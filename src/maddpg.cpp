#include "maddpg.h"
#include "agent.h"
#include "env.h"
#include "models.h"
#include "replayBuffer.h"
#include "robosim/RobotMonitor.h"
#include <ATen/Functions.h>
#include <ATen/TensorOperators.h>
#include <ATen/core/TensorBody.h>
#include <chrono>
#include <iostream>
#include <memory>
#include <numeric>
#include <robosim/EnvController.h>
#include <stddef.h>
#include <thread>
#include <torch/csrc/autograd/generated/variable_factories.h>
#include <type_traits>
#include <vector>

std::vector<float> maddpg::getActions(const std::vector<torch::Tensor> &states,
                                      const std::vector<std::shared_ptr<robosim::robotmonitor::RobotMonitor>> &robots)
{
    std::vector<float> actions;

    for (size_t i = 0; i < env::hunterCount; i++)
    {
        float action = std::static_pointer_cast<agent::Agent>(robots[i])->getAction(states[i]);

        actions.push_back(action);
    }

    return actions;
}

void maddpg::update(const std::vector<std::shared_ptr<robosim::robotmonitor::RobotMonitor>> &robots)
{
    replaybuffer::Sample sample = replaybuffer::sample();

    for (size_t i = 0; i < env::hunterCount; i++)
    {
        auto obsBatchI = sample.obsBatch[i];
        auto indivActionBatchI = sample.indiviActionBatch[i];
        auto indivRewardBatchI = sample.indiviRewardBatch[i];
        auto nextObsBatchI = sample.nextObsBatch[i];

        std::vector<torch::Tensor> nextGlobalActions;

        for (size_t j = 0; j < env::hunterCount; j++)
        {
            auto hunter = std::static_pointer_cast<agent::Agent>(robots[j]);
            at::Tensor arr = hunter->actor->forward(torch::vstack(nextObsBatchI));

            std::vector<float> indexes;
            for (int row = 0; row < arr.size(0); row++)
            {
                indexes.push_back(static_cast<float>(hunter->actor->nextAction(arr[row])));
            }

            // std::cout << indexes.size() << std::endl;

            at::Tensor n = torch::tensor(indexes);
            nextGlobalActions.push_back(torch::stack(n, 0));
        }

        at::Tensor nextGlobalActionsTemp =
            torch::cat(nextGlobalActions, 0).reshape({env::BATCH_SIZE, env::hunterCount});

        std::static_pointer_cast<agent::Agent>(robots[i])->update(
            agent::UpdateData{indivRewardBatchI, obsBatchI, sample.globalStateBatch, sample.globalActionBatch,
                              sample.globalNextStateBatch, nextGlobalActionsTemp});
        std::static_pointer_cast<agent::Agent>(robots[i])->updateTarget();
    }
}

void maddpg::run(uint32_t maxEpisodes, uint32_t maxSteps, const robosim::envcontroller::EnvController &env)
{
    // std::vector<float> rewards;

    for (uint32_t episode = 0; episode < maxEpisodes; episode++)
    {
        // std::cout << "Episode: " << episode << std::endl;
        std::vector<at::Tensor> states = env::reset(env);
        float epReward = 0.0f;

        uint32_t step = 0;
        for (; step < maxSteps; step++)
        {
            std::cout << "running: " << env.isRunning() << std::endl;
            if (!env.isRunning())
                return;

            std::vector<float> actions = getActions(states, env.getRobots());
            env::State state = env::step(actions, env.getRobots(), env.getCellWidth());

            // epReward =
            //     std::accumulate(rewards.begin(), rewards.end(), epReward);
            epReward = std::reduce(state.rewards.begin(), state.rewards.end(), epReward);

            if (state.done || step == maxSteps - 1)
            {
                break;
            }
            if (env::mode == env::Mode::EVAL)
            {
                states = state.nextStates;
                // slow down evaluation a bit
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
            else
            {
                replaybuffer::Experience experience{};
                experience.state = states;
                experience.action = actions;
                experience.reward = state.rewards;
                experience.nextState = state.nextStates;
                replaybuffer::push(experience);

                states = state.nextStates;

                if (replaybuffer::buffer.size() > env::BATCH_SIZE && step % env::BATCH_SIZE == 0)
                {
                    update(env.getRobots());
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
        }

        std::cout << "Episode: " << episode << " | Step: " << step
                  << " | Average: "
                     " | Reward: "
                  << epReward
                  << " | "
                     "Average: "
                     " | Time "
                  << std::endl;
    }
}

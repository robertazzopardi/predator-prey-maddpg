#pragma once

#include "action.h"
#include "agent.h"
#include "replayBuffer.h"
#include "robosim/RobotMonitor.h"
#include <ATen/core/TensorBody.h>
#include <algorithm>
#include <memory>
#include <random>
#include <torch/optim/adam.h>

namespace colour
{
struct Colour;
}

namespace hunter
{

class Hunter : public agent::Agent
{
  private:
    torch::optim::Adam criticOptimiser;
    torch::optim::Adam actorOptimiser;

    static constexpr float tau = 0.001f;
    static constexpr float gamma = 0.99f;

    std::uniform_real_distribution<float> distRand;

    std::uniform_int_distribution<int> randAction;

    float epsilon = 0.95;

  public:
    Hunter(bool, colour::Colour, bool *);

    float getAction(at::Tensor) override;

    at::Tensor getObservation(const std::vector<std::shared_ptr<robosim::robotmonitor::RobotMonitor>> &,
                              float) override;

    bool isAtGoal(const std::vector<std::shared_ptr<robosim::robotmonitor::RobotMonitor>> &, float);

    float getReward(action::Action) override;
    float getReward(const std::vector<std::shared_ptr<robosim::robotmonitor::RobotMonitor>> &, float) override;

    void update(const agent::UpdateData &) override;

    void updateTarget() override;
};

} // namespace hunter

#pragma once

#include "action.h"
#include "agent.h"
#include <ATen/core/TensorBody.h>
#include <random>

namespace colour
{
struct Colour;
}

namespace prey
{

class Prey : public agent::Agent
{
  private:
    std::uniform_real_distribution<float> dist;

  public:
    Prey(bool, colour::Colour, bool *);

    bool isTrapped(const std::vector<std::shared_ptr<robosim::robotmonitor::RobotMonitor>> &, float);

    float getAction(at::Tensor) override;

    at::Tensor getObservation(const std::vector<std::shared_ptr<robosim::robotmonitor::RobotMonitor>> &,
                              float) override;

    float getReward(action::Action) override;
    float getReward(const std::vector<std::shared_ptr<robosim::robotmonitor::RobotMonitor>> &, float) override;

    void update(const agent::UpdateData &) override;

    void updateTarget() override;
};

} // namespace prey

#pragma once

#include "direction.h"
#include "env.h"
#include "replayBuffer.h"
#include "robosim/EnvController.h"
#include <array>
#include <memory>
#include <random>
#include <robosim/RobotMonitor.h>
#include <torch/nn/modules/loss.h>
#include <tuple>
#include <vector>

namespace at
{
class Tensor;
}

namespace colour
{
struct Colour;
}

namespace action
{
enum class Action;
}

namespace models
{

namespace actor
{
struct Actor;
}

namespace critic
{
struct Critic;
}

} // namespace models

namespace agent
{

// using UpdateData =
//     std::tuple<std::vector<float>, std::vector<at::Tensor>, std::array<at::Tensor, env::BATCH_SIZE>,
//                std::array<at::Tensor, env::BATCH_SIZE>, std::array<at::Tensor, env::BATCH_SIZE>, at::Tensor>;
struct UpdateData
{
    std::vector<float> indivRewardBatchI;
    std::vector<at::Tensor> indivObsBatch;
    std::array<at::Tensor, env::BATCH_SIZE> globalStateBatch;
    std::array<at::Tensor, env::BATCH_SIZE> globalActionBatch;
    std::array<at::Tensor, env::BATCH_SIZE> globalNextStateBatch;
    at::Tensor nextGlobalActions;
};

class Agent : public robosim::robotmonitor::RobotMonitor
{
  private:
    int gx;
    int gy;

    void moveDirection(const direction::Direction &,
                       const std::vector<std::shared_ptr<robosim::robotmonitor::RobotMonitor>> &, float);

  protected:
    torch::nn::MSELoss MSELoss;

    std::mt19937 mt;

    void run();

    static constexpr uint32_t actionDim = 4;
    static constexpr uint32_t obsDim = env::agentCount << 1;

  public:
    action::Action nextAction;

    std::shared_ptr<models::actor::Actor> actor;
    std::shared_ptr<models::actor::Actor> targetActor;
    std::shared_ptr<models::critic::Critic> critic;
    std::shared_ptr<models::critic::Critic> targetCritic;

    Agent(bool, colour::Colour, bool *);

    bool canMove(const std::vector<std::shared_ptr<robosim::robotmonitor::RobotMonitor>> &, float);

    void executeAction(action::Action, const std::vector<std::shared_ptr<robosim::robotmonitor::RobotMonitor>> &,
                       float);

    virtual float getReward(action::Action) = 0;

    virtual float getReward(const std::vector<std::shared_ptr<robosim::robotmonitor::RobotMonitor>> &, float) = 0;

    virtual float getAction(at::Tensor) = 0;

    virtual at::Tensor getObservation(const std::vector<std::shared_ptr<robosim::robotmonitor::RobotMonitor>> &,
                                      float) = 0;

    virtual void update(const UpdateData &) = 0;

    virtual void updateTarget() = 0;
};

using AgentPtr = std::shared_ptr<Agent>;

} // namespace agent

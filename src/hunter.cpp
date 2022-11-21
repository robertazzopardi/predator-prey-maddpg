#include "hunter.h"
#include "direction.h"
#include "models.h"
#include "prey.h"
#include "replayBuffer.h"
#include "robosim/RobotMonitor.h"
#include <ATen/Functions.h>
#include <ATen/TensorOperators.h>
#include <ATen/core/TensorBody.h>
#include <array> // for array
#include <c10/core/Scalar.h>
#include <cstdint>
#include <memory>
#include <robosim/Colour.h>
#include <robosim/EnvController.h> // for getCel...
#include <stddef.h>
#include <sys/cdefs.h> // for __unused
#include <torch/csrc/autograd/generated/variable_factories.h>
#include <torch/nn/modules/loss.h>
#include <torch/nn/pimpl.h>
#include <torch/nn/utils/clip_grad.h>
#include <type_traits>
#include <vector> // for vector

namespace prey
{
class Prey;
}

namespace
{

static int getMaxValueIndex(float *values, int count)
{
    int maxAt = 0;

    for (int i = 0; i < count; i++)
    {
        maxAt = values[i] > values[maxAt] ? i : maxAt;
    }

    return maxAt;
}

static inline float normalise(int x, int min, int max)
{
    return (2 * (static_cast<float>((x - min)) / (max - min))) - 1;
    // return 1 - (x - min) / (float) (max - min);
    // return x;
}

} // namespace

hunter::Hunter::Hunter(bool verbose, colour::Colour colour, bool *running)
    : agent::Agent(verbose, colour, running), criticOptimiser(critic->parameters(), 1e-3),
      actorOptimiser(actor->parameters(), 1e-4), distRand(0, 1), randAction(0, action::ACTION_COUNT)
{

    for (size_t i = 0; i < critic->parameters().size(); i++)
    {
        targetCritic->parameters()[i].data().copy_(critic->parameters()[i].data());
    }

    for (size_t i = 0; i < actor->parameters().size(); i++)
    {
        targetActor->parameters()[i].data().copy_(actor->parameters()[i].data());
    }
}

float hunter::Hunter::getReward(action::Action action)
{
    if (action == action::Action::NOTHING)
    {
        return 10.0f;
    }

    return -10.0f;
}

float hunter::Hunter::getReward(const std::vector<std::shared_ptr<robosim::robotmonitor::RobotMonitor>> &robots,
                                float cellWidth)
{
    if (isAtGoal(robots, cellWidth))
    {
        return 0.0f;
    }
    return -0.1f;
}

float hunter::Hunter::getAction(torch::Tensor x)
{
    // auto output = actor->forward(states);
    // auto nextAction = actor->nextAction(output);
    // return static_cast<float>(nextAction);

    float random = distRand(mt);
    if (random < epsilon)
    {
        auto action = randAction(mt);
        if (epsilon > 0)
            epsilon *= 0.997;
        return action;
    }
    if (epsilon > 0)
        epsilon *= 0.997;

    auto output = actor->forward(x);

    auto outputValues = static_cast<float *>(output.data_ptr());

    auto maxValue = getMaxValueIndex(outputValues, output.size(0));

    return maxValue;
}

torch::Tensor hunter::Hunter::getObservation(
    const std::vector<std::shared_ptr<robosim::robotmonitor::RobotMonitor>> &robots, float cellWidth)
{
    static std::array<float, obsDim> observation;
    uint32_t count = 0;

    for (const auto &var : robots)
    {
        observation[count++] = normalise(var->getGridX(), 0, cellWidth);
        observation[count++] = normalise(var->getGridY(), 0, cellWidth);
    }

    return torch::from_blob(std::move(observation.data()), {obsDim});
}

bool hunter::Hunter::isAtGoal(const std::vector<std::shared_ptr<robosim::robotmonitor::RobotMonitor>> &robots,
                              float cellWidth)
{
    int32_t x = getX();
    int32_t y = getY();

    for (const auto &var : robots)
    {
        auto prey = std::dynamic_pointer_cast<prey::Prey>(var);
        if (prey)
        {
            int32_t px = prey->getX();
            int32_t py = prey->getY();
            return (x == direction::Direction(direction::Dir::UP).px(px, cellWidth) &&
                    y == direction::Direction(direction::Dir::UP).py(py, cellWidth)) ||
                   (x == direction::Direction(direction::Dir::DOWN).px(px, cellWidth) &&
                    y == direction::Direction(direction::Dir::DOWN).py(py, cellWidth)) ||
                   (x == direction::Direction(direction::Dir::LEFT).px(px, cellWidth) &&
                    y == direction::Direction(direction::Dir::LEFT).py(py, cellWidth)) ||
                   (x == direction::Direction(direction::Dir::RIGHT).px(px, cellWidth) &&
                    y == direction::Direction(direction::Dir::RIGHT).py(py, cellWidth));
        }
    }
    return false;
}

void hunter::Hunter::update(const agent::UpdateData &data)
{

    at::Tensor irb = torch::tensor(data.indivRewardBatchI);
    // irb.view({irb.size(0), 1});

    at::Tensor iob = torch::stack(data.indivObsBatch);

    at::Tensor gsb = torch::stack(data.globalStateBatch);

    at::Tensor gab = torch::stack(data.globalActionBatch);

    at::Tensor gnsb = torch::stack(data.globalNextStateBatch);

    at::Tensor nga = data.nextGlobalActions;

    // update critic
    criticOptimiser.zero_grad();

    at::Tensor currQ = critic->forward(gsb, gab);
    at::Tensor nextQ = targetCritic->forward(gnsb, nga);
    // auto estimQ = irb + (gamma * nextQ);
    at::Tensor estimQ = irb.reshape({irb.size(0), 1}) + (gamma * nextQ);

    auto criticLoss = MSELoss(currQ, estimQ.detach());
    criticLoss.backward();
    torch::nn::utils::clip_grad_norm_(critic->parameters(), 0.5);
    criticOptimiser.step();

    // update actor
    actorOptimiser.zero_grad();

    at::Tensor policyLoss = -critic->forward(gsb, gab).mean();
    at::Tensor currPolOut = actor->forward(iob);
    policyLoss += -(torch::pow(currPolOut, 2)).mean() * 1e-3;
    policyLoss.backward();
    torch::nn::utils::clip_grad_norm_(critic->parameters(), 0.5);
    actorOptimiser.step();
}

void hunter::Hunter::updateTarget()
{
    for (size_t i = 0; i < actor->parameters().size(); i++)
    {
        targetActor->parameters()[i].data().copy_(actor->parameters()[i].data());
    }

    for (size_t i = 0; i < critic->parameters().size(); i++)
    {
        targetCritic->parameters()[i].data().copy_(critic->parameters()[i].data() * tau *
                                                   targetCritic->parameters()[i].data() * (1.0f - tau));
    }
}

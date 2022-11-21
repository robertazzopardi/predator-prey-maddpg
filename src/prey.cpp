#include "prey.h"
#include "agent.h"
#include "env.h"
#include "hunter.h"
#include "replayBuffer.h"
#include <memory>
#include <robosim/Colour.h>
#include <robosim/EnvController.h>
#include <sys/cdefs.h>
#include <torch/csrc/autograd/generated/variable_factories.h>
#include <type_traits>
#include <vector>

namespace hunter
{
class Hunter;
}

prey::Prey::Prey(bool verbose, colour::Colour colour, bool *running)
    : agent::Agent(verbose, colour, running), dist(0, action::ACTION_COUNT)
{
}

float prey::Prey::getAction(torch::Tensor)
{
    return dist(mt);
}

torch::Tensor prey::Prey::getObservation(const std::vector<std::shared_ptr<robosim::robotmonitor::RobotMonitor>> &,
                                         float)
{
    return torch::tensor({1, 1, 1, 1, 1, 1, 1, 1, 1, 1});
}

bool prey::Prey::isTrapped(const std::vector<std::shared_ptr<robosim::robotmonitor::RobotMonitor>> &robots,
                           float cellWidth)
{
    int32_t x = getGridX();
    int32_t y = getGridY();

    uint32_t count = 0;

    if (y + 1 == env::GRID_SIZE - 1)
    {
        count++;
    }
    if (y - 1 == 0)
    {
        count++;
    }
    if (x + 1 == env::GRID_SIZE - 1)
    {
        count++;
    }
    if (x - 1 == 0)
    {
        count++;
    }

    for (const auto &robot : robots)
    {
        if (robot.get() != this && std::static_pointer_cast<hunter::Hunter>(robot)->isAtGoal(robots, cellWidth))
        {
            count++;
        }
    }

    return count > 3;
}

float prey::Prey::getReward(action::Action)
{
    return 0.0f;
}

void prey::Prey::update(const agent::UpdateData &)
{
}

void prey::Prey::updateTarget()
{
}

#pragma once

#include <memory>
#include <stdint.h>
#include <vector>

namespace robosim
{
namespace robotmonitor
{
class RobotMonitor;
}
namespace envcontroller
{
class EnvController;
}
} // namespace robosim

namespace at
{
class Tensor;
}

namespace env
{

enum class Mode
{
    TRAIN,
    EVAL
};

struct State
{
    std::vector<at::Tensor> nextStates;
    std::vector<float> rewards;
    bool done;
};

extern enum Mode mode;

constexpr static uint32_t GRID_SIZE = 5;
constexpr static uint32_t BATCH_SIZE = 64;

uint32_t getEnvSize(float);

static constexpr uint32_t hunterCount = 4;
static constexpr uint32_t preyCount = 1;
static constexpr uint32_t agentCount = hunterCount + preyCount;

std::vector<at::Tensor> reset(const robosim::envcontroller::EnvController &);

bool isSamePosition(const std::vector<std::shared_ptr<robosim::robotmonitor::RobotMonitor>> &,
                    const std::shared_ptr<robosim::robotmonitor::RobotMonitor> &);

State step(std::vector<float>, const std::vector<std::shared_ptr<robosim::robotmonitor::RobotMonitor>> &, float);

} // namespace env

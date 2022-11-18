#pragma once

#include "robosim/EnvController.h"
#include <memory>
#include <stdint.h>
#include <tuple>
#include <vector>

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

extern enum Mode mode;

constexpr static uint32_t GRID_SIZE = 5;
constexpr static uint32_t BATCH_SIZE = 64;

static inline uint32_t getEnvSize(const robosim::envcontroller::EnvController &env)
{
    return GRID_SIZE * env.getCellWidth();
}

static constexpr uint32_t hunterCount = 4;
static constexpr uint32_t preyCount = 1;
static constexpr uint32_t agentCount = hunterCount + preyCount;

std::vector<at::Tensor> reset();

bool isSamePosition(const std::shared_ptr<robosim::robotmonitor::RobotMonitor> &);

std::tuple<std::vector<at::Tensor>, std::vector<float>, bool> step(std::vector<float>);

int getRandomPos();

} // namespace env

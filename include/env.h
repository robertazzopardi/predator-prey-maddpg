
#ifndef __ENV_H__
#define __ENV_H__

#include "action.h"
#include "prey.h"
#include <EnvController.h>
#include <RobotMonitor.h>
#include <memory>
#include <tuple>  // for tuple
#include <vector> // for vector

namespace at {
class Tensor;
}

namespace env {

enum class Mode { TRAIN, EVAL };
extern enum Mode mode;

constexpr static auto GRID_SIZE = 10;
constexpr static auto BATCH_SIZE = 64;

static inline auto getEnvSize() {
    return static_cast<int>(GRID_SIZE) * robosim::envcontroller::getCellWidth();
};

static constexpr auto hunterCount = 4;
extern std::shared_ptr<prey::Prey> prey;
extern robosim::robotmonitor::MonitorVec robots;

std::vector<at::Tensor> reset();

bool isSamePosition(robosim::robotmonitor::RobotPtr);

std::tuple<std::vector<at::Tensor>, std::vector<float>, bool>
    step(std::vector<float>);
// step(std::vector<action::Action>);

int getRandomPos();

} // namespace env

#endif // !__ENV_H__

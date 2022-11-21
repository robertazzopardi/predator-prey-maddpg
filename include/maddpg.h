#pragma once

#include <memory>
#include <vector>

namespace robosim
{
namespace robotmonitor
{
class RobotMonitor;
};
namespace envcontroller
{
class EnvController;
}

}; // namespace robosim

namespace at
{
class Tensor;
}

namespace maddpg
{

std::vector<float> getActions(const std::vector<at::Tensor> &,
                              const std::vector<std::shared_ptr<robosim::robotmonitor::RobotMonitor>> &);

void update(const std::vector<std::shared_ptr<robosim::robotmonitor::RobotMonitor>> &);

void run(uint32_t, uint32_t, const robosim::envcontroller::EnvController &);

} // namespace maddpg

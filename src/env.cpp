#include "env.h"
#include "agent.h"
#include "prey.h"
#include "robosim/RobotMonitor.h"
#include <ATen/core/TensorBody.h>
#include <algorithm>
#include <memory>
#include <random>
#include <robosim/EnvController.h>
#include <stddef.h>
#include <thread>
#include <type_traits>
#include <vector>

enum env::Mode env::mode = Mode::TRAIN;

namespace env
{
static int getRandomPos(const robosim::envcontroller::EnvController &env)
{
    uint32_t max = (GRID_SIZE * 2 - 3);
    uint32_t min = 3;

    std::mt19937 mt(std::random_device{}());
    std::uniform_int_distribution<int> random(min, max);
    uint32_t rndPos = random(mt);
    rndPos += rndPos % 2 == 0 ? 1 : 0;

    return (rndPos * env.getCellWidth()) / 2;
}
} // namespace env

uint32_t env::getEnvSize(float cellWidth)
{
    return env::GRID_SIZE * cellWidth;
}

std::vector<torch::Tensor> env::reset(const robosim::envcontroller::EnvController &env)
{
    std::vector<torch::Tensor> obs;

    for (auto robot : env.getRobots())
    {
        do
        {
            uint32_t randomX = env::getRandomPos(env);
            uint32_t randomY = env::getRandomPos(env);
            robot->setPose(randomX, randomY, 0);
        } while (env::isSamePosition(env.getRobots(), robot));

        auto agent = std::static_pointer_cast<agent::Agent>(robot);

        auto prey = std::dynamic_pointer_cast<prey::Prey>(agent);
        if (!prey)
        {
            obs.push_back(agent->getObservation(env.getRobots(), env.getCellWidth()));
        }
    }

    return obs;
}

env::State env::step(std::vector<float> actions,
                     const std::vector<std::shared_ptr<robosim::robotmonitor::RobotMonitor>> &robots, float cellWidth)
{
    std::vector<float> rewards;

    std::vector<torch::Tensor> nextStates;

    std::vector<std::thread> threads;

    for (size_t i = 0; i < env::hunterCount; i++)
    {
        auto agent = std::static_pointer_cast<agent::Agent>(robots[i]);

        threads.push_back(std::thread(&agent::Agent::executeAction, agent, action::getActionFromIndex(actions[i]),
                                      robots, cellWidth));
    }

    // block and allows robots to execute actions at the same time
    for (std::thread &th : threads)
    {
        // if (th.joinable())
        th.join();
    }

    for (size_t i = 0; i < env::hunterCount; i++)
    {
        std::shared_ptr<agent::Agent> agent = std::static_pointer_cast<agent::Agent>(robots[i]);
        nextStates.push_back(agent->getObservation(robots, cellWidth));
        float reward = agent->getReward(action::getActionFromIndex(actions[i]));
        rewards.push_back(reward);
    }

    // Check the trapped status of all of the possible prey agents
    // bool trapped = std::any_of(robots.begin(), robots.end(),
    //                            [](const std::shared_ptr<robosim::robotmonitor::RobotMonitor> &monitor) {
    //                                std::shared_ptr<prey::Prey> prey = std::dynamic_pointer_cast<prey::Prey>(monitor);
    //                                if (prey)
    //                                {
    //                                    return prey->isTrapped();
    //                                }
    //                                return false;
    //                            });
    bool trapped = false;
    for (const auto &monitor : robots)
    {
        std::shared_ptr<prey::Prey> prey = std::dynamic_pointer_cast<prey::Prey>(monitor);
        if (prey)
        {
            trapped = prey->isTrapped(robots, cellWidth);
            break;
        }
    }

    return State{nextStates, rewards, trapped};
}

bool env::isSamePosition(const std::vector<std::shared_ptr<robosim::robotmonitor::RobotMonitor>> &robots,
                         const std::shared_ptr<robosim::robotmonitor::RobotMonitor> &robot)
{
    // return std::any_of(robots.begin(), robots.end(),
    //                    [robot](std::shared_ptr<robosim::robotmonitor::RobotMonitor> &agent) {
    //                        return robot != agent && agent->getX() == robot->getX() && agent->getY() == robot->getY();
    //                    });

    for (const auto &agent : robots)
    {
        if (robot != agent && agent->getX() == robot->getX() && agent->getY() == robot->getY())
        {
            return true;
        }
    }
    return false;
}

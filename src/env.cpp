#include "env.h"
#include "agent.h"
#include "prey.h"
#include <ATen/core/TensorBody.h>
// #include <RobotMonitor.h>
#include <algorithm>
#include <memory>
#include <random>
#include <stddef.h>
#include <thread>
#include <type_traits>

enum env::Mode env::mode = Mode::TRAIN;

std::vector<torch::Tensor> env::reset()
{
    std::vector<torch::Tensor> obs;

    for (auto robot : robosim::envcontroller::robots)
    {
        do
        {
            auto randomX = getRandomPos();
            auto randomY = getRandomPos();
            robot->setPose(randomX, randomY, 0);
        } while (isSamePosition(robot));

        auto agent = std::static_pointer_cast<agent::Agent>(robot);

        auto prey = std::dynamic_pointer_cast<prey::Prey>(agent);
        if (!prey)
            obs.push_back(agent->getObservation());
    }

    return obs;
}

std::tuple<std::vector<torch::Tensor>, std::vector<float>, bool> env::step(std::vector<float> actions)
{
    std::vector<float> rewards;

    std::vector<torch::Tensor> nextStates;

    std::vector<std::thread> threads;

    for (size_t i = 0; i < env::hunterCount; i++)
    {
        auto agent = std::static_pointer_cast<agent::Agent>(robosim::envcontroller::robots[i]);

        threads.push_back(std::thread(&agent::Agent::executeAction, agent, action::getActionFromIndex(actions[i])));
    }

    // block and allows robots to execute actions at the same time
    for (auto &th : threads)
    {
        if (th.joinable())
            th.join();
    }

    for (size_t i = 0; i < env::hunterCount; i++)
    {
        auto agent = std::static_pointer_cast<agent::Agent>(robosim::envcontroller::robots[i]);
        nextStates.push_back(agent->getObservation());
        auto reward = agent->getReward(action::getActionFromIndex(actions[i]));
        rewards.push_back(reward);
    }

    // Check the trapped status of all of the possible prey agents
    auto trapped = std::any_of(robosim::envcontroller::robots.begin(), robosim::envcontroller::robots.end(),
                               [](robosim::envcontroller::RobotPtr &monitor) {
                                   auto prey = std::dynamic_pointer_cast<prey::Prey>(monitor);
                                   if (prey)
                                   {
                                       return prey->isTrapped();
                                   }
                                   return false;
                               });

    return {nextStates, rewards, trapped};
}

bool env::isSamePosition(robosim::envcontroller::RobotPtr robot)
{
    return std::any_of(robosim::envcontroller::robots.begin(), robosim::envcontroller::robots.end(),
                       [robot](robosim::envcontroller::RobotPtr &agent) {
                           return robot != agent && agent->getX() == robot->getX() && agent->getY() == robot->getY();
                       });
}

int env::getRandomPos()
{
    auto max = (GRID_SIZE * 2 - 3);
    auto min = 3;

    std::mt19937 mt(std::random_device{}());
    std::uniform_int_distribution<int> random(min, max);
    auto rndPos = random(mt);
    rndPos += rndPos % 2 == 0 ? 1 : 0;

    return (rndPos * robosim::envcontroller::getCellWidth()) / 2;
}

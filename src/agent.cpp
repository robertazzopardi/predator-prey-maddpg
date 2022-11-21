#include "agent.h"
#include "action.h"
#include "direction.h"
#include "env.h"
#include "models.h"
#include <algorithm>
#include <iostream>
#include <memory>
#include <robosim/Colour.h>
#include <robosim/EnvController.h>
#include <robosim/RobotMonitor.h>
#include <sys/cdefs.h>
#include <type_traits>
#include <vector>

agent::Agent::Agent(bool verbose, colour::Colour colour, bool *running)
    : robosim::robotmonitor::RobotMonitor(verbose, colour, running), gx(0), gy(0), MSELoss(), mt(std::random_device{}())
{

    uint32_t criticInputDim = obsDim * env::hunterCount;
    uint32_t actorInputDim = obsDim;

    critic = std::make_shared<models::critic::Critic>(criticInputDim, actionDim);
    targetCritic = std::make_shared<models::critic::Critic>(criticInputDim, actionDim);

    actor = std::make_shared<models::actor::Actor>(actorInputDim, actionDim);
    targetActor = std::make_shared<models::actor::Actor>(actorInputDim, actionDim);

    // for (size_t i = 0; i < critic->parameters().size(); i++) {
    //     targetCritic->parameters().data()[i].copy_(
    //         critic->parameters()[i].data());
    // }
    // for (size_t i = 0; i < actor->parameters().size(); i++) {
    //     targetActor->parameters().data()[i].copy_(
    //         actor->parameters()[i].data());
    // }

    nextAction = action::Action::NOTHING;
}

void agent::Agent::moveDirection(const direction::Direction &direction,
                                 const std::vector<std::shared_ptr<robosim::robotmonitor::RobotMonitor>> &robots,
                                 float cellWidth)
{
    int x = getX();
    int y = getY();

    if (canMove(robots, cellWidth))
    {
        gx = direction.px(x, cellWidth);
        gy = direction.py(y, cellWidth);

        if (env::mode == env::Mode::EVAL)
        {
            travel();
            // setPose(direction.px(x), direction.py(y), getHeading());
        }
        else
        {
            setPose(direction.px(x, cellWidth), direction.py(y, cellWidth), getHeading());
        }
    }
}

void agent::Agent::executeAction(action::Action nextAction,
                                 const std::vector<std::shared_ptr<robosim::robotmonitor::RobotMonitor>> &robots,
                                 float cellWidth)
{
    if (nextAction != action::Action::NOTHING)
    {
        switch (nextAction)
        {
        case action::Action::FORWARD:
            moveDirection(direction::Direction::fromDegree(getHeading()), robots, cellWidth);
            break;

        case action::Action::LEFT:
            if (env::mode == env::Mode::EVAL)
            {
                rotate(-90);
            }
            else
            {
                setPose(getX(), getY(), getHeading() - 90);
            }
            break;

        case action::Action::RIGHT:
            if (env::mode == env::Mode::EVAL)
            {
                rotate(90);
            }
            else
            {
                setPose(getX(), getY(), getHeading() + 90);
            }
            break;

        default:
            break;
        }

        nextAction = action::Action::NOTHING;
    }
}

void agent::Agent::run()
{
    std::cout << "Starting Robot: " << serialNumber << std::endl;
}

bool agent::Agent::canMove(const std::vector<std::shared_ptr<robosim::robotmonitor::RobotMonitor>> &robots,
                           float cellWidth)
{
    direction::Direction dir = direction::Direction::fromDegree(getHeading());

    int32_t x = dir.px(getX(), cellWidth);
    int32_t y = dir.py(getY(), cellWidth);

    // TODO change to loop
    // if (std::any_of(env.getRobots().begin(), env.getRobots().end(),
    //                 [&](const std::shared_ptr<robosim::envcontroller::RobotMonitor> &r) {
    //                     return (r.get() != this) && (r->getX() == x && r->getY() == y);
    //                 }))
    // {
    //     return false;
    // }

    for (const auto &robot : robots)
    {
        if (robot.get() != this && robot->getX() == x && robot->getY() == y)
        {
            return false;
        }
    }

    int32_t xOffset = env::getEnvSize(cellWidth) - cellWidth;

    return (x < xOffset && x > cellWidth) && (y < xOffset && y > cellWidth);
}

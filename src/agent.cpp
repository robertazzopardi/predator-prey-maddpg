#include "agent.h"
#include "action.h"
#include "direction.h"
#include "env.h"
#include "models.h"
#include <robosim/Colour.h>
#include <robosim/EnvController.h>
#include <algorithm>
#include <iostream>
#include <sys/cdefs.h>
#include <type_traits>

agent::Agent::Agent(bool verbose, colour::Colour colour)
    : robosim::robotmonitor::RobotMonitor(verbose, colour), gx(0), gy(0),
      MSELoss(), mt(std::random_device {}()) {

    auto criticInputDim = obsDim * env::hunterCount;
    auto actorInputDim = obsDim;

    critic =
        std::make_shared<models::critic::Critic>(criticInputDim, actionDim);
    targetCritic =
        std::make_shared<models::critic::Critic>(criticInputDim, actionDim);

    actor = std::make_shared<models::actor::Actor>(actorInputDim, actionDim);
    targetActor =
        std::make_shared<models::actor::Actor>(actorInputDim, actionDim);

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

void agent::Agent::moveDirection(direction::Direction direction) {
    auto x = getX();
    auto y = getY();

    if (canMove()) {
        gx = direction.px(x);
        gy = direction.py(y);

        if (env::mode == env::Mode::EVAL) {
            travel();
            // setPose(direction.px(x), direction.py(y), getHeading());
        } else {
            setPose(direction.px(x), direction.py(y), getHeading());
        }
    }
}

void agent::Agent::executeAction(action::Action nextAction) {
    if (nextAction != action::Action::NOTHING) {
        switch (nextAction) {
        case action::Action::FORWARD:
            moveDirection(direction::Direction::fromDegree(getHeading()));
            break;

        case action::Action::LEFT:
            if (env::mode == env::Mode::EVAL) {
                rotate(-90);
            } else {
                setPose(getX(), getY(), getHeading() - 90);
            }
            break;

        case action::Action::RIGHT:
            if (env::mode == env::Mode::EVAL) {
                rotate(90);
            } else {
                setPose(getX(), getY(), getHeading() + 90);
            }
            break;

        default:
            break;
        }

        nextAction = action::Action::NOTHING;
    }
}

void agent::Agent::run(bool *running __unused) {
    std::cout << "Starting Robot: " << serialNumber << std::endl;
}

bool agent::Agent::canMove() {
    auto dir = direction::Direction::fromDegree(getHeading());

    auto x = dir.px(getX());
    auto y = dir.py(getY());

    if (std::any_of(robosim::envcontroller::robots.begin(),
                    robosim::envcontroller::robots.end(),
                    [&](const robosim::envcontroller::RobotPtr &r) {
                        return (r.get() != this) &&
                               (r->getX() == x && r->getY() == y);
                    })) {
        return false;
    }

    auto xOffset = env::getEnvSize() - robosim::envcontroller::getCellWidth();

    return (x < xOffset && x > robosim::envcontroller::getCellWidth()) &&
           (y < xOffset && y > robosim::envcontroller::getCellWidth());
}

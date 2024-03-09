#include "env.h"
#include "hunter.h"
#include "maddpg.h"
#include "prey.h"
#include <robosim/robosim.h>
#include <stdlib.h>
#include <thread>

int main(void)
{
    robosim::envcontroller::EnvController env(env::GRID_SIZE, env::GRID_SIZE);

    env.makeRobots<hunter::Hunter>(env::hunterCount, 50, colour::OFF_BLACK);
    env.makeRobots<prey::Prey>(env::preyCount, 50, colour::OFF_RED);

    std::thread maddpg(maddpg::run, 500, 300, env);

    // startSimulation();
    env.run();

    maddpg.join();

    env.stop();

    return EXIT_SUCCESS;
}

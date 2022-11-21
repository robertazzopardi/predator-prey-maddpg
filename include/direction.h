#pragma once

namespace robosim::envcontroller
{
class EnvController;
}

namespace direction
{

enum class Dir
{
    UP,
    DOWN,
    LEFT,
    RIGHT,
    NONE
};

struct Direction
{
    enum Dir dir;

    Direction(enum Dir);

    static Direction fromDegree(int);

    int px(int, float) const;
    int py(int, float) const;
};

} // namespace direction

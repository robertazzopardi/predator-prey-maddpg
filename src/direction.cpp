#include "direction.h"
#include <robosim/EnvController.h>

direction::Direction::Direction(enum Dir dir) : dir(dir)
{
}

direction::Direction direction::Direction::fromDegree(int degree)
{
    switch (degree % 360)
    {
    case 0:
        return Direction(Dir::DOWN);
    case 90:
    case -270:
        return Direction(Dir::RIGHT);
    case 180:
    case -180:
        return Direction(Dir::UP);
    case 270:
    case -90:
        return Direction(Dir::LEFT);
    default:
        return Direction(Dir::NONE);
    }
}

int direction::Direction::px(int x, float cellWidth) const
{
    switch (dir)
    {
    case Dir::UP:
    case Dir::DOWN:
        return x;
    case Dir::LEFT:
        return x - cellWidth;
    case Dir::RIGHT:
        return x + cellWidth;
    default:
        return 0;
    }
}

int direction::Direction::py(int y, float cellWidth) const
{
    switch (dir)
    {
    case Dir::UP:
        return y - cellWidth;
    case Dir::DOWN:
        return y + cellWidth;
    case Dir::LEFT:
    case Dir::RIGHT:
        return y;
    default:
        return 0;
    }
}

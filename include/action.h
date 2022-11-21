#pragma once

namespace action
{

enum class Action
{
    FORWARD = 0,
    LEFT = 1,
    RIGHT = 2,
    NOTHING = 3
};

static const Action ACTIONS[] = {Action::FORWARD, Action::LEFT, Action::RIGHT, Action::NOTHING};

// static const auto ACTION_COUNT = sizeof(ACTIONS) / sizeof(*ACTIONS);
static constexpr auto ACTION_COUNT = sizeof(action::ACTIONS) / sizeof(*action::ACTIONS);

inline auto toString(Action ac)
{
    switch (ac)
    {
    case Action::FORWARD:
        return "FORWARD";

    case Action::LEFT:
        return "LEFT";

    case Action::RIGHT:
        return "RIGHT";

    case Action::NOTHING:
        return "NOTHING";

    default:
        return "NULL";
    }
}

// static inline float getActionIndexFloat(enum Action action) {
//     auto it = std::find(ACTIONS.begin(), ACTIONS.end(), action);

//     // If element was found
//     if (it != ACTIONS.end()) {
//         std::cout << it - ACTIONS.begin() << std::endl;
//         return it - ACTIONS.begin();
//     } else {
//         return -1.0f;
//     }
// }

inline Action getActionFromIndex(int index)
{
    return ACTIONS[index];
}

} // namespace action


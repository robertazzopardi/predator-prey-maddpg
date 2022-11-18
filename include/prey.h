#ifndef __PREY_H__
#define __PREY_H__

#include "action.h"
#include "agent.h"
#include <ATen/core/TensorBody.h>
#include <random>

namespace colour {
struct Colour;
}

namespace prey {

class Prey : public agent::Agent {
private:
    std::uniform_real_distribution<float> dist;

public:
    Prey(bool, colour::Colour);

    bool isTrapped();

    float getAction(at::Tensor) override;

    at::Tensor getObservation() override;

    float getReward(action::Action) override;

    void update(agent::UpdateData) override;

    void updateTarget() override;
};

}  // namespace prey

#endif  // !__PREY_H__


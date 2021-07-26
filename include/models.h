
#ifndef __MODELS_H__
#define __MODELS_H__

#include <torch/nn/module.h>         // for Module
#include <torch/nn/modules/linear.h> // for Linear

namespace at {
class Tensor;
}

namespace models {

namespace actor {

struct Actor : torch::nn::Module {
    Actor(int, int);

    int obsDim, actionDim;

    torch::nn::Linear fc1{nullptr}, fc2{nullptr}, fc3{nullptr};

    at::Tensor forward(at::Tensor);

    int nextAction(at::Tensor);
};

} // namespace actor

namespace critic {

struct Critic : torch::nn::Module {
    Critic(int, int);

    int obsDim, actionDim;

    // Use one of many "standard library" modules.
    torch::nn::Linear fc1{nullptr}, fc2{nullptr}, fc3{nullptr}, fc4{nullptr};

    // Implement the Net's algorithm.
    at::Tensor forward(at::Tensor, at::Tensor);
};

} // namespace critic

} // namespace models

#endif // !__MODELS_H__

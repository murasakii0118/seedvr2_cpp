// Diffusion schedule / timesteps / sampler (src/common/diffusion)
#pragma once

#include <torch/torch.h>
#include <vector>

namespace seedvr2 {

// x_t = A(t) * x_0 + B(t) * x_T
struct LerpSchedule {
    double T = 1000.0;
    torch::Tensor A(const torch::Tensor& t) const { return 1.0 - t / T; }
    torch::Tensor B(const torch::Tensor& t) const { return t / T; }
    // prediction_type: "v_lerp"
    void convert_from_pred(const torch::Tensor& pred, const torch::Tensor& x_t,
                           const torch::Tensor& t,
                           torch::Tensor& pred_x_0, torch::Tensor& pred_x_T) const;
    torch::Tensor forward(const torch::Tensor& x0, const torch::Tensor& xT, const torch::Tensor& t) const;
};

struct SamplingTimesteps {
    double T = 1000.0;
    torch::Tensor timesteps;      // 1D
    int64_t size() const { return timesteps.size(0); }
    torch::Tensor at(int64_t i) const { return timesteps[i]; }
};

// UniformTrailingSamplingTimesteps
SamplingTimesteps uniform_trailing_timesteps(double T, int steps, double shift = 1.0);

// Euler ODE sampler (with return_endpoint)
class EulerSampler {
public:
    EulerSampler(const LerpSchedule& sched, const SamplingTimesteps& ts) : schedule_(sched), timesteps_(ts) {}

    // f(args) -> prediction.  Returns x_0 (endpoint of the flow).
    torch::Tensor sample(const torch::Tensor& x,
                         const std::function<torch::Tensor(const torch::Tensor& x_t,
                                                           const torch::Tensor& t, int i)>& f,
                         const std::function<void(int, int)>& on_step = nullptr) const;

    const LerpSchedule& schedule() const { return schedule_; }

private:
    LerpSchedule schedule_;
    SamplingTimesteps timesteps_;

    torch::Tensor step_to(const torch::Tensor& pred, const torch::Tensor& x_t,
                          const torch::Tensor& t, const torch::Tensor& s) const;
    torch::Tensor get_endpoint(const torch::Tensor& pred, const torch::Tensor& x_t,
                               const torch::Tensor& t) const;
};

// classifier_free_guidance / dispatcher
torch::Tensor classifier_free_guidance(const torch::Tensor& pos, const torch::Tensor& neg,
                                       double scale, double rescale);

} // namespace seedvr2

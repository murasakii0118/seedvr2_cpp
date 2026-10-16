// Diffusion schedule / timesteps / sampler
#include "core/schedule.h"
#include "core/common.h"

namespace seedvr2 {
using torch::Tensor;

// ---------------------------------------------------------------------------
// LerpSchedule
// ---------------------------------------------------------------------------
void LerpSchedule::convert_from_pred(const Tensor& pred, const Tensor& x_t, const Tensor& t,
                                     Tensor& pred_x_0, Tensor& pred_x_T) const {
    Tensor tt = t;
    while (tt.dim() < x_t.dim()) tt = tt.unsqueeze(-1);
    Tensor A_t = A(tt);
    Tensor B_t = B(tt);
    // v_lerp
    pred_x_0 = (x_t - B_t * pred) / (A_t + B_t);
    pred_x_T = (x_t + A_t * pred) / (A_t + B_t);
}

Tensor LerpSchedule::forward(const Tensor& x0, const Tensor& xT, const Tensor& t) const {
    Tensor tt = t;
    while (tt.dim() < x0.dim()) tt = tt.unsqueeze(-1);
    return A(tt) * x0 + B(tt) * xT;
}

// ---------------------------------------------------------------------------
// UniformTrailingSamplingTimesteps
// ---------------------------------------------------------------------------
SamplingTimesteps uniform_trailing_timesteps(double T, int steps, double shift) {
    SamplingTimesteps out;
    out.T = T;
    // torch.arange(1.0, 0.0, -1.0/steps)
    std::vector<float> v;
    for (int i = 0; i < steps; ++i) {
        double x = 1.0 - (double)i / (double)steps;
        if (x <= 0.0) break;
        v.push_back((float)x);
    }
    if (v.empty()) v.push_back(1.0f);
    Tensor ts = torch::from_blob(v.data(), { (int64_t)v.size() }, torch::kFloat32).clone();
    ts = shift * ts / (1.0 + (shift - 1.0) * ts);
    if (std::floor(T) == T) ts = ts * T;   // continuous
    else ts = (ts * (T + 1.0) - 1.0).round().to(torch::kLong);
    out.timesteps = ts;
    return out;
}

// ---------------------------------------------------------------------------
// EulerSampler
// ---------------------------------------------------------------------------
Tensor EulerSampler::step_to(const Tensor& pred, const Tensor& x_t, const Tensor& t, const Tensor& s) const {
    Tensor tt = t, ss = s;
    while (tt.dim() < x_t.dim()) tt = tt.unsqueeze(-1);
    while (ss.dim() < x_t.dim()) ss = ss.unsqueeze(-1);
    const double T = schedule_.T;
    Tensor x0, xT;
    schedule_.convert_from_pred(pred, x_t, t, x0, xT);
    Tensor s_clamped = torch::clamp(ss, 0.0, T);
    Tensor pred_x_s = schedule_.forward(x0, xT, s_clamped);
    // clamp out-of-bound
    pred_x_s = torch::where(ss >= 0, pred_x_s, x0);
    pred_x_s = torch::where(ss <= T, pred_x_s, xT);
    return pred_x_s;
}

Tensor EulerSampler::get_endpoint(const Tensor& pred, const Tensor& x_t, const Tensor& t) const {
    Tensor x0, xT;
    schedule_.convert_from_pred(pred, x_t, t, x0, xT);
    return x0;  // backward sampling -> x_0
}

Tensor EulerSampler::sample(const Tensor& x, const std::function<Tensor(const Tensor&, const Tensor&, int)>& f,
                            const std::function<void(int, int)>& on_step) const {
    if (on_step) on_step(0, (int)timesteps_.size());
    Tensor cur = x;
    int total = (int)timesteps_.size();
    int64_t n = timesteps_.size();
    for (int64_t i = 0; i + 1 < n; ++i) {
        Tensor t = timesteps_.at(i);
        Tensor s = timesteps_.at(i + 1);
        Tensor pred = f(cur, t, (int)i);
        cur = step_to(pred, cur, t, s);
        if (on_step) on_step((int)i + 1, total);
    }
    // return_endpoint
    Tensor t = timesteps_.at(n - 1);
    Tensor pred = f(cur, t, (int)(n - 1));
    cur = get_endpoint(pred, cur, t);
    if (on_step) on_step(total, total);
    return cur;
}

// ---------------------------------------------------------------------------
Tensor classifier_free_guidance(const Tensor& pos, const Tensor& neg, double scale, double rescale) {
    Tensor cfg = neg + scale * (pos - neg);
    if (rescale != 0.0) {
        std::vector<int64_t> dims;
        for (int64_t d = 1; d < pos.dim(); ++d) dims.push_back(d);
        Tensor pos_std = pos.std(dims, true, true);
        Tensor cfg_std = cfg.std(dims, true, true);
        Tensor factor = pos_std / cfg_std;
        factor = rescale * factor + (1 - rescale);
        cfg = cfg * factor;
    }
    return cfg;
}

} // namespace seedvr2

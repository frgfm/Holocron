// Copyright (C) 2019-2026, François-Guillaume Fernandez.
//
// This program is licensed under the Apache License 2.0.
// See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

#include <metal_stdlib>
using namespace metal;
inline float2 reduce(threadgroup float2* shared, uint t, float2 value) {
    shared[t] = value;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128; stride > 0; stride >>= 1) {
        if (t < stride) shared[t] += shared[t + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    return shared[0];
}
kernel void moments(device float* update, device const float* param,
                    device const float* grad, device float* avg,
                    device float* square, device float2* partials,
                    constant float4& betas, constant float2& bias,
                    constant float& eps, constant uint& n,
                    uint i [[thread_position_in_grid]],
                    uint t [[thread_index_in_threadgroup]],
                    uint group [[threadgroup_position_in_grid]]) {
    float2 values = 0;
    if (i < n) {
        float g = grad[i], p = param[i];
        avg[i] = avg[i] * betas.x + g * betas.z;
        square[i] = square[i] * betas.y + (g * g) * betas.w;
        float u = (avg[i] / bias.x) / (sqrt(square[i]) / bias.y + eps);
        update[i] = u;
        values = float2(p * p, g * g);
    }
    threadgroup float2 shared[256];
    float2 totals = reduce(shared, t, values);
    if (t == 0) partials[group] = totals;
}
kernel void statistics(device float* factor, device const float* param,
                       device const float* grad, device const float* update,
                       device const float2* partials,
                       constant uint& groups, constant uint& n,
                       constant float& eps, constant float& delta,
                       uint t [[thread_index_in_threadgroup]]) {
    float2 values = 0;
    for (uint i = t; i < groups; i += 256) values += partials[i];
    threadgroup float2 shared[256];
    float2 sums = reduce(shared, t, values);
    float pn = sqrt(sums.x), gn = sqrt(sums.y), norm = pn + eps;
    // All lanes must read the norms before the next reduction overwrites shared memory.
    threadgroup_barrier(mem_flags::mem_threadgroup);
    values = 0;
    for (ulong i = t; i < ulong(n); i += 256) {
        float p = param[i], g = grad[i];
        values += float2((p / max(pn, 1e-8f)) * (g / max(gn, 1e-8f)), (p / norm) * update[i]);
    }
    float2 dots = reduce(shared, t, values);
    if (t == 0) {
        factor[0] = dots.y;
        factor[1] = dots.x < delta / sqrt(float(n));
        factor[2] = norm;
    }
}
kernel void apply_update(device float* param, device const float* update,
                         device const float* factor, constant float& lr,
                         constant uint& n,
                         uint i [[thread_position_in_grid]]) {
    if (i >= n) return;
    float value = update[i];
    if (factor[1] != 0) value -= factor[0] * (param[i] / factor[2]);
    param[i] -= lr * value;
}

// Copyright (C) 2019-2026, François-Guillaume Fernandez.
//
// This program is licensed under the Apache License 2.0.
// See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

#include <metal_stdlib>
using namespace metal;
inline float4 reduce(threadgroup float4* shared, uint t, float4 value) {
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
                    device float* square, device float4* partials,
                    constant float4& betas, constant float2& bias,
                    constant float& eps, constant uint& n,
                    uint i [[thread_position_in_grid]],
                    uint t [[thread_index_in_threadgroup]],
                    uint group [[threadgroup_position_in_grid]]) {
    float4 values = 0;
    if (i < n) {
        float g = grad[i], p = param[i];
        avg[i] = avg[i] * betas.x + g * betas.z;
        square[i] = square[i] * betas.y + (g * g) * betas.w;
        float u = (avg[i] / bias.x) / (sqrt(square[i]) / bias.y + eps);
        update[i] = u;
        values = float4(p * p, g * g, p * g, p * u);
    }
    threadgroup float4 shared[256];
    float4 totals = reduce(shared, t, values);
    if (t == 0) partials[group] = totals;
}
kernel void statistics(device float* factor, device const float4* partials,
                       constant uint& groups, constant uint& n,
                       constant float& eps, constant float& delta,
                       uint t [[thread_index_in_threadgroup]]) {
    float4 values = 0;
    for (uint i = t; i < groups; i += 256) values += partials[i];
    threadgroup float4 shared[256];
    float4 sums = reduce(shared, t, values);
    if (t == 0) {
        float pn = sqrt(sums.x), gn = sqrt(sums.y), norm = pn + eps;
        float cosine = (sums.z / max(pn, 1e-8f)) / max(gn, 1e-8f);
        factor[0] = (sums.w / norm) / norm;
        factor[1] = cosine < delta / sqrt(float(n));
    }
}
kernel void apply_update(device float* param, device const float* update,
                         device const float* factor, constant float& lr,
                         uint i [[thread_position_in_grid]]) {
    float value = update[i];
    if (factor[1] != 0) value -= factor[0] * param[i];
    param[i] -= lr * value;
}

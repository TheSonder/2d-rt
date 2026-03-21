#ifndef RT2D_RAYTRACE_H
#define RT2D_RAYTRACE_H

#include <torch/extension.h>

torch::Tensor raytrace_cpu(
    const torch::Tensor& origins,
    const torch::Tensor& directions,
    const torch::Tensor& sphere_center,
    double sphere_radius
);

torch::Tensor raytrace_cuda(
    const torch::Tensor& origins,
    const torch::Tensor& directions,
    const torch::Tensor& sphere_center,
    double sphere_radius
);

#endif


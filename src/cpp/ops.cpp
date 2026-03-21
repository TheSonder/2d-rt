#include "rt2d/raytrace.h"

#include <torch/extension.h>

namespace {

void validate_inputs(
    const torch::Tensor& origins,
    const torch::Tensor& directions,
    const torch::Tensor& sphere_center
) {
    TORCH_CHECK(origins.dim() == 2 && origins.size(1) == 3, "origins must have shape [N, 3]");
    TORCH_CHECK(directions.dim() == 2 && directions.size(1) == 3, "directions must have shape [N, 3]");
    TORCH_CHECK(origins.size(0) == directions.size(0), "origins and directions must have same ray count");
    TORCH_CHECK(sphere_center.dim() == 1 && sphere_center.size(0) == 3, "sphere_center must have shape [3]");

    TORCH_CHECK(
        origins.scalar_type() == torch::kFloat32 &&
            directions.scalar_type() == torch::kFloat32 &&
            sphere_center.scalar_type() == torch::kFloat32,
        "all tensors must be float32"
    );

    TORCH_CHECK(
        origins.device() == directions.device() && origins.device() == sphere_center.device(),
        "all tensors must be on the same device"
    );
}

torch::Tensor raytrace_cpu_dispatch(
    const torch::Tensor& origins,
    const torch::Tensor& directions,
    const torch::Tensor& sphere_center,
    double sphere_radius
) {
    validate_inputs(origins, directions, sphere_center);
    TORCH_CHECK(!origins.is_cuda(), "CPU dispatch requires CPU tensors");
    return raytrace_cpu(origins, directions, sphere_center, sphere_radius);
}

torch::Tensor raytrace_cuda_dispatch(
    const torch::Tensor& origins,
    const torch::Tensor& directions,
    const torch::Tensor& sphere_center,
    double sphere_radius
) {
    validate_inputs(origins, directions, sphere_center);
    TORCH_CHECK(origins.is_cuda(), "CUDA dispatch requires CUDA tensors");
    return raytrace_cuda(origins, directions, sphere_center, sphere_radius);
}

} // namespace

TORCH_LIBRARY(rt2d, m) {
    m.def("raytrace(Tensor origins, Tensor directions, Tensor sphere_center, float sphere_radius) -> Tensor");
}

TORCH_LIBRARY_IMPL(rt2d, CPU, m) {
    m.impl("raytrace", &raytrace_cpu_dispatch);
}

TORCH_LIBRARY_IMPL(rt2d, CUDA, m) {
    m.impl("raytrace", &raytrace_cuda_dispatch);
}


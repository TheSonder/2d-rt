#include "rt2d/raytrace.h"

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/util/Exception.h>

namespace {

__device__ inline float dot3(const float3& a, const float3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__global__ void raytrace_kernel(
    const float* origins,
    const float* directions,
    const float* sphere_center,
    float sphere_radius,
    float* out,
    int64_t ray_count
) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= ray_count) {
        return;
    }

    const float3 origin = make_float3(
        origins[idx * 3 + 0],
        origins[idx * 3 + 1],
        origins[idx * 3 + 2]
    );
    const float3 direction = make_float3(
        directions[idx * 3 + 0],
        directions[idx * 3 + 1],
        directions[idx * 3 + 2]
    );
    const float3 center = make_float3(sphere_center[0], sphere_center[1], sphere_center[2]);

    const float3 local_origin = make_float3(
        origin.x - center.x,
        origin.y - center.y,
        origin.z - center.z
    );

    const float a = dot3(direction, direction);
    const float b = 2.0f * dot3(direction, local_origin);
    const float c = dot3(local_origin, local_origin) - (sphere_radius * sphere_radius);
    const float discriminant = b * b - 4.0f * a * c;

    if (discriminant < 0.0f) {
        out[idx] = -1.0f;
        return;
    }

    const float sqrt_disc = sqrtf(discriminant);
    const float t0 = (-b - sqrt_disc) / (2.0f * a);
    const float t1 = (-b + sqrt_disc) / (2.0f * a);

    float t = -1.0f;
    if (t0 > 0.0f) {
        t = t0;
    } else if (t1 > 0.0f) {
        t = t1;
    }

    out[idx] = t;
}

} // namespace

torch::Tensor raytrace_cuda(
    const torch::Tensor& origins,
    const torch::Tensor& directions,
    const torch::Tensor& sphere_center,
    double sphere_radius
) {
    TORCH_CHECK(origins.is_cuda(), "raytrace_cuda expects CUDA tensors");
    TORCH_CHECK(origins.scalar_type() == torch::kFloat32, "origins must be float32");
    TORCH_CHECK(directions.scalar_type() == torch::kFloat32, "directions must be float32");
    TORCH_CHECK(sphere_center.scalar_type() == torch::kFloat32, "sphere_center must be float32");

    auto origins_c = origins.contiguous();
    auto directions_c = directions.contiguous();
    auto center_c = sphere_center.contiguous();
    auto out = torch::empty({origins.size(0)}, origins.options().dtype(torch::kFloat32));

    const int64_t ray_count = origins.size(0);
    if (ray_count == 0) {
        return out;
    }

    constexpr int threads_per_block = 256;
    const int blocks = static_cast<int>((ray_count + threads_per_block - 1) / threads_per_block);
    auto stream = at::cuda::getDefaultCUDAStream();

    raytrace_kernel<<<blocks, threads_per_block, 0, stream>>>(
        origins_c.data_ptr<float>(),
        directions_c.data_ptr<float>(),
        center_c.data_ptr<float>(),
        static_cast<float>(sphere_radius),
        out.data_ptr<float>(),
        ray_count
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return out;
}

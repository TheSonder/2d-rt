#include "rt2d/c_scene.h"
#include "rt2d/raytrace.h"

#include <cmath>

namespace {

inline float dot3(const float* a, const float* b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

inline float intersect_sphere(
    const float* origin,
    const float* direction,
    const float* center,
    float radius
) {
    const float ox = origin[0] - center[0];
    const float oy = origin[1] - center[1];
    const float oz = origin[2] - center[2];

    const float local_origin[3] = {ox, oy, oz};
    const float a = dot3(direction, direction);
    const float b = 2.0f * dot3(direction, local_origin);
    const float c = dot3(local_origin, local_origin) - radius * radius;
    const float discriminant = b * b - 4.0f * a * c;

    if (discriminant < 0.0f) {
        return -1.0f;
    }

    const float sqrt_disc = std::sqrt(discriminant);
    const float t0 = (-b - sqrt_disc) / (2.0f * a);
    const float t1 = (-b + sqrt_disc) / (2.0f * a);

    if (t0 > 0.0f) {
        return t0;
    }
    if (t1 > 0.0f) {
        return t1;
    }
    return -1.0f;
}

} // namespace

torch::Tensor raytrace_cpu(
    const torch::Tensor& origins,
    const torch::Tensor& directions,
    const torch::Tensor& sphere_center,
    double sphere_radius
) {
    TORCH_CHECK(!origins.is_cuda(), "raytrace_cpu expects CPU tensors");
    TORCH_CHECK(origins.scalar_type() == torch::kFloat32, "origins must be float32");
    TORCH_CHECK(directions.scalar_type() == torch::kFloat32, "directions must be float32");
    TORCH_CHECK(sphere_center.scalar_type() == torch::kFloat32, "sphere_center must be float32");

    auto origins_c = origins.contiguous();
    auto directions_c = directions.contiguous();
    auto center_c = sphere_center.contiguous();
    auto out = torch::empty({origins.size(0)}, origins.options().dtype(torch::kFloat32));

    const float* origins_ptr = origins_c.data_ptr<float>();
    const float* directions_ptr = directions_c.data_ptr<float>();
    const float* center_ptr = center_c.data_ptr<float>();
    float* out_ptr = out.data_ptr<float>();

    const auto ray_count = origins.size(0);
    const float radius = static_cast<float>(sphere_radius);

    for (int64_t i = 0; i < ray_count; ++i) {
        const float* origin = origins_ptr + i * 3;
        const float* direction = directions_ptr + i * 3;

        float t = intersect_sphere(origin, direction, center_ptr, radius);
        if (t > 0.0f) {
            const float jitter = 0.001f * rt2d_jitter_from_index(static_cast<unsigned int>(i));
            t = rt2d_clampf(t + jitter, 0.0f, 1.0e8f);
        }
        out_ptr[i] = t;
    }

    return out;
}


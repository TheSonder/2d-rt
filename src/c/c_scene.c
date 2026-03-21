#include "rt2d/c_scene.h"

float rt2d_clampf(float x, float lo, float hi) {
    if (x < lo) {
        return lo;
    }
    if (x > hi) {
        return hi;
    }
    return x;
}

unsigned int rt2d_wang_hash(unsigned int seed) {
    seed = (seed ^ 61U) ^ (seed >> 16);
    seed *= 9U;
    seed = seed ^ (seed >> 4);
    seed *= 0x27d4eb2dU;
    seed = seed ^ (seed >> 15);
    return seed;
}

float rt2d_jitter_from_index(unsigned int index) {
    const unsigned int hashed = rt2d_wang_hash(index);
    const float normalized = (float)(hashed & 0x00FFFFFFU) / (float)0x00FFFFFFU;
    return (normalized * 2.0f) - 1.0f;
}


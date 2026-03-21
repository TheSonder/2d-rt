#ifndef RT2D_C_SCENE_H
#define RT2D_C_SCENE_H

#ifdef __cplusplus
extern "C" {
#endif

float rt2d_clampf(float x, float lo, float hi);
unsigned int rt2d_wang_hash(unsigned int seed);
float rt2d_jitter_from_index(unsigned int index);

#ifdef __cplusplus
}
#endif

#endif


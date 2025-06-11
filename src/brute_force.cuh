#pragma once

extern "C" __global__ void brute_force_collisions_and_gravity(
    float* pos_x, float* pos_y,
    float* vel_x, float* vel_y,
    unsigned int* mass, unsigned int* radius, 
    unsigned int no_of_particles, float g_const
);
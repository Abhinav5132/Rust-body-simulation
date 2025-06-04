#include "cuda_runtime.h"

#include "device_launch_parameters.h"
#include <stdio.h>

__global__ void calculate_gravity(float* pos_x, float* pos_y, 
    float* vel_x, float* vel_y ,
    float* mass, int num_particles, float gravitational_constant) 
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_particles) {return;}

    float fx = 0.0f;
    float fy = 0.0f; 

    float xi = pos_x[i];
    float yi = pos_y[i];
    float massi = mass[i];

    for (int j = 0; j < num_particles; j++) {
        if (j == i) {
            continue;
        }
        float dx = xi - pos_x[j];
        float dy = yi - pos_y[j]; 
        float dist_sqr = dx*dx + dy*dy + 1e-6;
        float distance = sqrtf(dist_sqr);

        if (distance < 1e-3 ) {continue;}

        float f = (gravitational_constant * massi * mass[j]) / dist_sqr;

        float nx = dx / distance;
        float ny = dy / distance;

        fx -= nx * f;
        fy -= ny * f;


    }

    vel_x[i] += fx;
    vel_y[i] += fy;

}
extern "C" void launch_calculate_gravity(
    float* pos_x, float* pos_y, 
    float* vel_x, float* vel_y ,
    float* mass, int num_particles, float gravitational_constant
) {
    int threads = 1024;
    int blocks = (num_particles + threads -1) / threads;
    calculate_gravity<<<blocks, threads>>> (
        pos_x, pos_y, vel_x, vel_y, mass,num_particles, gravitational_constant
    );
    
}
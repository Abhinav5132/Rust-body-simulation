#include <iostream>
#include <random>
#include <curand_kernel.h>
#include <cuda_runtime.h>
typedef unsigned int uint;

// particle position
    float* d_pos_x;
    float* d_pos_y;
    float* d_vel_x;
    float* d_vel_y;
    uint* d_mass;
    uint* d_radius;



extern "C" void initialize_particles(uint no_of_particles, float x_bounds, float y_bounds) {
    //allocate memory for screen bounds(never mutated and of size 1 float)
    float* d_x_bounds = 0;
    float* d_y_bounds = 0;

    cudaMalloc(&d_x_bounds, sizeof(float));
    cudaMalloc(&d_y_bounds, sizeof(float));
    
    //copy screen bounds
    cudaMemcpy(d_pos_x, &x_bounds, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_pos_y, &y_bounds, sizeof(float), cudaMemcpyHostToDevice);

    //allocate memory for properties of particels.
    cudaMalloc(&d_pos_x, no_of_particles * sizeof(float));
    cudaMalloc(&d_pos_y, no_of_particles * sizeof(float));
    cudaMalloc(&d_vel_x, no_of_particles * sizeof(float));
    cudaMalloc(&d_vel_y, no_of_particles * sizeof(float));
    cudaMalloc(&d_mass, no_of_particles * sizeof(uint));
    cudaMalloc(&d_radius, no_of_particles * sizeof(uint));
    
    int threads = 512;
    int blocks = (no_of_particles + threads - 1)/ threads;
   
    init_particles<<<blocks, threads>>>(
        x_bounds, y_bounds, 
        d_pos_x, d_pos_y, 
        d_vel_x, d_vel_y, 
        d_mass, d_radius, 
        time(NULL), no_of_particles
    );
    cudaDeviceSynchronize();
}

__global__ void init_particles(
    float x_bounds, 
    float y_bounds,
    float* pos_x, float* pos_y,
    float* vel_x, float* vel_y,
    uint* mass, uint* radius, 
    uint seed, uint no_of_particles
) {

    //local bounds
    float vel_bound_x = 1.0f;
    float vel_bound_y = 1.0f;

    int max_mass = 2000;
    int min_mass = 1000;


    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if(i >= no_of_particles) {
        return;
    }

    curandState state;
    curand_init(seed+i, 0, 0, &state);

    //initializing positions 
    pos_x[i] = (curand_uniform(&state)* 2.0f - 1.0f) * x_bounds;
    pos_y[i] = (curand_uniform(&state)* 2.0f - 1.0f) * y_bounds;

    //initializing velocities
    vel_x[i] = (curand_uniform(&state)* 2.0f - 1.0f) * vel_bound_x;
    vel_y[i] = (curand_uniform(&state)* 2.0f - 1.0f) * vel_bound_y;

    mass[i] = min_mass + (curand(&state) % (max_mass - min_mass + 1));
    radius[i] = sqrt(mass[i] / (10 * 3.141));

    
}

extern "C" void read_particle_positions(float* h_pos_x, float* h_pos_y, int no_of_particles) {
    cudaMemcpy(h_pos_x, d_pos_x, sizeof(float) * no_of_particles, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_pos_y, d_pos_y, sizeof(float) * no_of_particles, cudaMemcpyDeviceToHost);
}

void read_particle_velocities(float* h_vel_x, float* h_vel_y, int no_of_particles) {
    cudaMemcpy(h_vel_x, d_vel_x, sizeof(float) * no_of_particles, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_vel_y, d_vel_y, sizeof(float) * no_of_particles, cudaMemcpyDeviceToHost);
}
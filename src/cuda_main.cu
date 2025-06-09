#include <iostream>
#include <random>
#include <curand_kernel.h>
#include <cuda_runtime.h>
typedef unsigned int uint;

// particle position
    float* d_pos_x = nullptr;
    float* d_pos_y = nullptr;
    float* d_vel_x = nullptr;
    float* d_vel_y = nullptr;
    uint* d_mass = nullptr;
    uint* d_radius = nullptr;

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
    radius[i] = static_cast<uint>(sqrtf(static_cast<float>(mass[i]) / (100.0f * 3.14159f)));
    
}

//add a function here to move overlaping particles away from each other.



extern "C" void initialize_particles(uint no_of_particles, float x_bounds, float y_bounds) {
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
        static_cast<uint>(time(NULL)), no_of_particles
    );
    cudaDeviceSynchronize();
}

extern "C" void read_particle_positions(float* h_pos_x, float* h_pos_y, int no_of_particles) {
    cudaMemcpy(h_pos_x, d_pos_x, sizeof(float) * no_of_particles, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_pos_y, d_pos_y, sizeof(float) * no_of_particles, cudaMemcpyDeviceToHost);
}

extern "C" void read_particle_velocities(float* h_vel_x, float* h_vel_y, int no_of_particles) {
    cudaMemcpy(h_vel_x, d_vel_x, sizeof(float) * no_of_particles, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_vel_y, d_vel_y, sizeof(float) * no_of_particles, cudaMemcpyDeviceToHost);
}

extern "C" void read_particle_radius(float* h_radius, int no_of_particles) {

    cudaMemcpy(h_radius, d_radius, sizeof(float) * no_of_particles, cudaMemcpyDeviceToHost);
}  

__global__ void update_particle_positions(
    float* pos_x, float* pos_y,
    float* vel_x, float* vel_y,
    float x_bounds, float y_bounds
) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (vel_x[i] != 0.0){
        pos_x[i] += vel_x[i];
        if (pos_x[i] > x_bounds || pos_x[i] < -x_bounds) {
            vel_x[i] = -vel_x[i];
        }
    }

    if (vel_y[i] != 0.0) {
        pos_y[i] += vel_y[i];
        if (pos_y[i] > y_bounds || pos_y[i] < -y_bounds) {
            vel_y[i] = -vel_y[i];
        }
    }
}

extern "C" void move_particles(uint no_of_particles, float x_bounds, float y_bounds) {
    int threads = 512;
    int blocks = (no_of_particles + threads - 1)/ threads;

    update_particle_positions<<<blocks, threads>>> (
        d_pos_x, d_pos_y, 
        d_vel_x, d_vel_y, 
        x_bounds, y_bounds
    );
    
    cudaDeviceSynchronize();
    
}

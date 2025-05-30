extern "C" {
__global__ void calculate_gravity(float* pos_x, float* pos_y, 
    float* vel_x, float* vel_y ,
    float* mass, int num_particles, float gravitational_constant) 
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i > num_particles) {return;}

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
        float ny = dx / distance;

        fx -= nx * f;
        fy -= nx * f;


    }

    vel_x[i] = fx / 1000.0f;
    vel_y[i] = fy / 1000.0f;

    }
}
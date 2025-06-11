extern "C" __global__ void brute_force_collisions_and_gravity(
    float* pos_x, float* pos_y,
    float* vel_x, float* vel_y,
    uint* mass, uint* radius, 
    uint no_of_particles,float g_const
) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= no_of_particles) return;

    float ax = 0.0f;
    float ay = 0.0f;
    float delta_time = 1.0f;
    
    for(int j= i + 1; j < no_of_particles; ++j){
        
        float dx = pos_x[i] - pos_x[j];
        float dy = pos_y[i] - pos_y[j];

        float dist_sqr = dx * dx + dy * dy;
        float radius_sum = (float)radius[i] + (float)radius[j]; 
        float mi = (float) mass[i];
        // collision check
        if (dist_sqr <= radius_sum * radius_sum && dist_sqr > 0.0001f) {
            float dvx = vel_x[i] - vel_x[j];
            float dvy = vel_y[i] - vel_y[j];

            float dis_over_time = dvx * dx + dvy * dy + 1e-6f;

            if (dis_over_time > 0) {
                continue;
            }

            float coef = 2.0f * dis_over_time / (dist_sqr * ((float)mass[i] + (float)mass[j]));

            float fx = coef * dx;
            float fy = coef * dy;

            vel_x[i] -= fx * mass[j];
            vel_y[i] -= fy * mass[j];
            vel_x[j] += fx * mass[i];
            vel_y[j] += fy * mass[i];
        }

        // gravity
        float distance = sqrtf(dist_sqr);
        float force = (g_const * (float)mi * (float)mass[j]/ dist_sqr);

        float fx = force * (dx / distance);
        float fy = force * (dy / distance);

        ax += fx / mi;
        ay += fy / mi;
    }

    vel_x[i] += ax * delta_time;
    vel_y[i] += ay * delta_time;
}
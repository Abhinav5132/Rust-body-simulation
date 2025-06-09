unsafe extern "C" {
    pub unsafe fn initialize_particles(no_of_particles: u32, x_bounds: f32, y_bounds: f32);

    pub unsafe fn read_particle_positions(h_pos_x: *mut f32, h_pos_y: *mut f32, no_of_particels: u32);
    
    pub unsafe fn read_particle_radius(h_radius: *mut u32, no_of_particles: u32);

    pub unsafe fn move_particles(no_of_particles: u32, x_bounds: f32, y_bounds: f32);
}
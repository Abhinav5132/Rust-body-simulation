
use crate::*;

pub fn create_particles(
    no_of_particle: Res<NoOfParticle>,
    camera_bounds: Res<CameraBounds>,
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<ColorMaterial>>,
) {
    unsafe {
        initialize_particles(
            no_of_particle.0,
            camera_bounds.bounds[0],
            camera_bounds.bounds[1],
        );
        let mut pos_x = vec![0.0f32; no_of_particle.0 as usize];
        let mut pos_y = vec![0.0f32; no_of_particle.0 as usize];

        read_particle_positions(pos_x.as_mut_ptr(), pos_y.as_mut_ptr(), no_of_particle.0);

        let mut radius = vec![0; no_of_particle.0 as usize];

        read_particle_radius(radius.as_mut_ptr(), no_of_particle.0);
        commands.insert_resource(ParticlePositions {
            x: pos_x.clone(),
            y: pos_y.clone(),
            radius: radius.clone(),
        });

        let unit_circle = meshes.add(Circle::new(1.0));
        let mut shared_materials = Vec::new();
        for _ in 0..10 {
            let color = ColorMaterial::from(Color::linear_rgb(
                fastrand::f32(),
                fastrand::f32(),
                fastrand::f32(),
            ));
            shared_materials.push(materials.add(color));
        }

        for i in 0..pos_x.len() {
            let material_handle = shared_materials[i % shared_materials.len()].clone();
            commands.spawn((
                Particle {},
                ParticleIndex(i),
                Mesh2d(unit_circle.clone()),
                MeshMaterial2d(material_handle),
                Transform {
                    translation: Vec3::new(pos_x[i], pos_y[i], 0.0),
                    scale: Vec3::splat(radius[i] as f32),
                    ..Default::default()
                },
            ));
        }
    }
}


pub fn update_particles( 
    no_of_particle: Res<NoOfParticle>,
    camera_bounds: Res<CameraBounds>,
    mut particle_positions: ResMut<ParticlePositions>,
    mut query: Query<(&ParticleIndex, &mut Transform)>

) {
    unsafe{
        move_particles(no_of_particle.0, camera_bounds.bounds[0],camera_bounds.bounds[1]);
        read_particle_positions(particle_positions.x.as_mut_ptr(), particle_positions.y.as_mut_ptr(), no_of_particle.0);
    }

        query.iter_mut().for_each(|(index, mut transform)|
        {
            let i = index.0;
            transform.translation.x = particle_positions.x[i];
            transform.translation.y = particle_positions.y[i];

        });    
}
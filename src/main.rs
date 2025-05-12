use bevy::input::mouse::MouseWheel;
use bevy::{prelude::*, transform};
use bevy::input::ButtonState;
use bevy::input::keyboard::{Key, KeyboardInput};
use bevy::{input::mouse::AccumulatedMouseScroll};
use bevy::window::{WindowMode, WindowResolution, };
use classes::KdNode;
use bevy_dev_tools::fps_overlay:: FpsOverlayPlugin;
use rayon::iter::IntoParallelIterator;
use crate::quad::QuadNode;
use rayon::prelude::*;
mod quad;
mod classes;
/*todo: make sure bodies dont spawn outside the area, this can be done in the create all entites where we iterate through them all, 
check for collisions with the boundry and if it does remove  
3.turn the resolution into actual parameters so users can resize the screen
5.collissions with each other
5.gravity
6. make the r key restart the simulation
7.particle only bounce from the wall if their center touches, need to make it so it takes the radisu and bounces from that
*/
#[derive(Resource)]
pub struct BarnesHutTheta(pub f32);

#[derive(Component)]
struct FpsText;

#[derive(Resource)]
struct GravitationalConstant(f32);

#[derive(Resource)]
struct NoOfParticle(u32);

#[derive(Resource)]
struct TextID(Option<Entity>);

#[derive(Resource)]
struct CameraBounds{
	x_min:f32,
    y_min:f32,
	x_max:f32,
	y_max:f32,
}

#[derive(States, Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum GameStates{
    TextState,
    SimulationState,
}

#[derive(Component, Debug, Clone, Copy)]
struct Particle{
	pos:[f32; 2],
	vel:[f32; 2],
	mass:u64,
	radius:u64,
}


fn main() {
   let mut app = App::new();

   app.add_plugins((DefaultPlugins.set(
    WindowPlugin{
        primary_window: Some(Window{
            resolution: WindowResolution::new(1920., 1080.).with_scale_factor_override(1.0),
            mode: WindowMode::Fullscreen(MonitorSelection::Primary, VideoModeSelection::Current),
            ..default()
        }),
        ..default()
    }

   ),FpsOverlayPlugin::default()))


   .insert_resource(NoOfParticle(500000)) //10 for now
   .insert_resource(TextID(None))
   .insert_resource(CameraBounds{x_min:-3480., x_max:3480., y_min:-2160., y_max:2160.})
   .insert_resource(GravitationalConstant(6.67430e-4))
   .insert_resource(BarnesHutTheta(0.5))
   .add_systems(Startup, spawn_camera)
   .add_systems(Update, camera_zoom)

    // first state code
   .insert_state(GameStates::TextState)
   .add_systems(OnEnter(GameStates::TextState), text_setup)
   .add_systems(Update, text_update.run_if(in_state(GameStates::TextState)))

   //simulation state
   .add_systems(OnEnter(GameStates::SimulationState), create_all_entitites)
   .add_systems(Update, update_all_entities.run_if(in_state(GameStates::SimulationState)))
   //.add_systems(Update, kd_tree_collisions.run_if(in_state(GameStates::SimulationState)))
   .add_systems(Update, par_kd_tree_collisions.run_if(in_state(GameStates::SimulationState)))
   //.add_systems(Update, gravity_quad.run_if(in_state(GameStates::SimulationState)))
   .add_systems(Update, par_gravity_quad.run_if(in_state(GameStates::SimulationState)))
   //.add_systems(Update, _gravity_normal.run_if(in_state(GameStates::SimulationState)))
   //.add_systems(Update, _check_collision.run_if(in_state(GameStates::SimulationState)))
   ;
    app.run();
}

fn spawn_camera(
    mut commands: Commands,
) {
    commands.spawn(Camera2d);// spawn the camera
}
fn camera_zoom(
    mut mouse_wheel_input: EventReader<MouseWheel>,
    mut KeyboardInput: EventReader<KeyboardInput>,
    mut query: Query<&mut Transform, With<Camera2d>>,
){
    let zoom_speed = 0.15;
    let pan_speed = 30.0;
    let mut zoom_delta = 0.0;

    for event in mouse_wheel_input.read(){
        zoom_delta += event.y;
    }
    if zoom_delta.abs() > 0.0 {
        for mut transform in query.iter_mut() {
            let scale_change = 1.0 - zoom_speed * zoom_delta.signum();
            transform.scale *= scale_change;
            transform.scale.x = transform.scale.x.clamp(0.05, 10.0);
            transform.scale.y = transform.scale.y.clamp(0.05, 10.0);
        }
    }
    for event in KeyboardInput.read(){
        for mut transform in query.iter_mut(){
        match &event.logical_key{
            Key::ArrowLeft => {
                transform.translation.x -= pan_speed;
                }

            Key::ArrowRight => {
                transform.translation.x += pan_speed;
            }

            Key::ArrowUp => {
                transform.translation.y += pan_speed;
            }
            Key::ArrowDown => {
                transform.translation.y -= pan_speed;
            }
            _=>{}
            }
        }
    }

    


}
fn text_setup(
	mut commands: Commands,
	asset_server : Res<AssetServer>,
	mut text_id: ResMut<TextID>,
) {
    
    let id_text = commands.spawn(( // spawn in the text
        Text::new("Please enter the number of bodies:"),
        TextFont{
            font: asset_server.load("fonts/BungeeSpice-Regular.ttf"),
            font_size : 67.0,
            ..default()
        },

        TextLayout::new_with_justify(JustifyText::Center),
        // Set the style of the Node itself.
        Node {
            position_type: PositionType::Absolute,
            ..default()
        },
    )).id();

	text_id.0 = Some(id_text); // some means something of any type 
}


fn text_update(
	mut commands: Commands,
    mut evr_kbd : EventReader<KeyboardInput>,
	mut text_id: ResMut<TextID>,
	mut next_state: ResMut<NextState<GameStates>>,
) {
    let mut localstring : String = "".to_string();

    for ev in evr_kbd.read() {
        // We don't care about key releases, only key presses
        if ev.state == ButtonState::Released {
            continue;
        }

        match &ev.logical_key {
            Key::Character(abc) =>{
                if abc.chars().any(|c| c.is_control()) {
                    continue;
                }
                localstring.push_str(&abc);
            }

            Key::Enter => {
				//write to Noofparticle

                if let Some(entity) = text_id.0 {
					commands.entity(entity).despawn(); // despawn UI text
					text_id.0 = None; // Clear the resource if needed
				}

				next_state.set(GameStates::SimulationState);
            }

            Key::Backspace => {
                localstring.pop();
            }

            _ => {}
        }
    } 
}

fn create_all_entitites(
    mut commands: Commands,
    camera_bound: ResMut<CameraBounds>,
    no_of_part: ResMut<NoOfParticle>,
    mut materials: ResMut<Assets<ColorMaterial>>,
    mut meshes: ResMut<Assets<Mesh>>,
){
    let mass_max = 2000 as u64; //in kg
    let mass_min = 1000 as u64;

    let assumed_density = 55; // in kg/m3

        for _i in 0..no_of_part.0{
        let x = fastrand::i32(camera_bound.x_min as i32..camera_bound.x_max as i32); //CHANGE this to not be inclusive of borders
        let y = fastrand::i32(camera_bound.y_min as i32..camera_bound.y_max as i32);
        let vx = fastrand::f32() * if fastrand::bool() {1.0} else {-1.0};
        let vy = fastrand::f32() * if fastrand::bool() {1.0} else {-1.0};
        //let vx = 0.0;
        //let vy = 0.0;
        let mass = fastrand::u64(mass_min..mass_max);

        let radius:u64 = mass / (assumed_density * 12); // prolly better to fully randomise it 

        let color = ColorMaterial::from(Color::linear_rgb(
            fastrand::f32(),
            fastrand::f32(),
            fastrand::f32(),
            ));
        
            commands.spawn((
            Particle {
            pos: [x as f32,y as f32],
            vel: [vx, vy],
            mass: mass,
            radius: radius,

        },
        Mesh2d(meshes.add(Circle::new(radius as f32))),
        MeshMaterial2d(materials.add(color)),
        Transform::from_xyz(x as f32, y as f32, 0.0),
        ));

        


    }

}

fn update_all_entities(
   mut query: Query<(&mut Particle, &mut Transform)>,
   camera_bound : ResMut<CameraBounds>,
){ // this is where all the particles are moved and gravity 

// moving particles with their initial velocity

    //fn update_position(mut query: Query<(&mut Particle, &mut Transform)>){
    for (mut particle, mut transform) in query.iter_mut(){

        if particle.pos[0] + particle.vel[0] >= camera_bound.x_max 
        || particle.pos[0] + particle.vel[0] <= camera_bound.x_min{
            particle.vel[0] = particle.vel[0] * -1.0;   
        } 
        
        particle.pos[0] += particle.vel[0] * 0.99;
        transform.translation.x = particle.pos[0];

        if particle.pos[1] + particle.vel[1] >= camera_bound.y_max 
        || particle.pos[1] + particle.vel[1] <= camera_bound.y_min{
            particle.vel[1] = particle.vel[1] * -1.0;
        } 
        particle.pos[1] += particle.vel[1]* 0.99;
        transform.translation.y = particle.pos[1]; }
    //update_position(query);

}

fn _check_collision(mut query: Query<&mut Particle>) {

    let mut bodies: Vec<_>  = query.iter_mut().collect();
        for i in 0..bodies.len() {
            let (left, right) = bodies.split_at_mut(i+1);
            let part_a = &mut left[i];
            for j in i+1..right.len(){
                let part_b = &mut right[j];

                let dx = part_a.pos[0] - part_b.pos[0];
                let dy = part_a.pos[1] - part_b.pos[1];

                let dist_square = dx * dx + dy * dy;
                let radius_sum = (part_a.radius + part_b.radius) as f32;

                if dist_square <= radius_sum * radius_sum {
                    part_a.vel[0] *= -1.0;
                    part_a.vel[1] *= -1.0;

                    part_b.vel[0] *= -1.0;
                    part_b.vel[1] *= -1.0; 
                
                    
                }
            }   
        }

                
}

fn kd_tree_collisions(mut query: Query<&mut Particle>) {

    let bodies: Vec<_> = query.iter().map(|p| p.clone()).collect();

    if let Some(tree) = KdNode::build(bodies, 0) {
        for mut particle in query.iter_mut(){
            let mut collisions: Vec<(Particle, f32, f32, f32, f32)> = Vec::new();
            tree.check_collison(&mut particle,&mut collisions);

            for (other, nx, ny, sep_x, sep_y) in collisions{
                particle.pos[0] += sep_x;
                particle.pos[1] += sep_y;

                let m1 = particle.mass as f32;
                let m2 = particle.mass as f32;

                let bounciness = 0.8;
                let j = -(1.0 + bounciness) * ((particle.vel[0] - other.vel[0]) * nx + (particle.vel[1] - other.vel[1]) * ny) / (1.0/m1 + 1.0/m2);
            
                particle.vel[0] += (j / m1) * nx;
                particle.vel[1] += (j / m1) * ny;

            }
        }
    }
}

fn par_kd_tree_collisions(mut query: Query<&mut Particle>) {
    let particles: Vec<_> = query.iter().map(|p| p.clone()).collect();

    if let Some(tree) = KdNode::build(particles.clone(), 0) {
        // Calculate collisions in parallel
        let collision_results: Vec<_> = particles.par_iter()
            .map(|particle| {
                let mut collisions = Vec::new();
                tree.check_collison(particle, &mut collisions);
                (particle.clone(), collisions)
            })
            .collect();

        // Apply collision responses
        for (mut particle, (_, collisions)) in query.iter_mut().zip(collision_results.iter()) {
            for (other, nx, ny, sep_x, sep_y) in collisions {
                particle.pos[0] += sep_x;
                particle.pos[1] += sep_y;

                let m1 = particle.mass as f32;
                let m2 = other.mass as f32;

                let bounciness = 0.8;
                let j = -(1.0 + bounciness) * 
                    ((particle.vel[0] - other.vel[0]) * nx + 
                     (particle.vel[1] - other.vel[1]) * ny) / 
                    (1.0/m1 + 1.0/m2);
            
                particle.vel[0] += (j / m1) * nx;
                particle.vel[1] += (j / m1) * ny;
            }
        }
    }
}

fn _gravity_normal(mut query: Query<&mut Particle>, gravity: Res<GravitationalConstant>) {
let mut particles : Vec<_> = query.iter().map(|p| p.clone()).collect();

let forces: Vec<_> = (0..particles.len()).into_par_iter().map(|i| {
    let mut force = [0.0, 0.0];

    for j in 0..particles.len(){
        if i!= j {
            let part_a = &particles[i];
            let part_b = &particles[j];

            let dx = part_a.pos[0] - part_b.pos[0];
            let dy = part_a.pos[1] - part_b.pos[1];

            let dist_square = dx * dx + dy * dy+ 1e-6;
            let distance = dist_square.sqrt();
            if distance < 1e-3 { continue; }

            let f= (gravity.0 * part_a.mass as f32 * part_b.mass as f32) / dist_square;
            let nx = dx / distance;
            let ny = dy / distance;

            force[0] -= nx * f;
            force[1] -= ny * f;
        }
        }
        force
    }).collect();

    for (mut particle, force) in query.iter_mut().zip(forces.iter()){
        particle.vel[0] += force[0] / 1000.0;
        particle.vel[1] += force[1] / 1000.0;
    }
}
 
fn gravity_quad(
    cam_bounds: Res<CameraBounds>, 
    g: Res<GravitationalConstant>,
    theta: Res<BarnesHutTheta>,
    mut query: Query<&mut Particle>
){
    let bounds = [cam_bounds.x_min, cam_bounds.y_min, cam_bounds.x_max, cam_bounds.y_max];

    let mut root = QuadNode::new(bounds);
    for particle in query.iter() {
        root.insert(particle.clone(), 0); // Pass depth = 0
    }

    for mut particle in query.iter_mut(){
        let mut force = [0.0, 0.0];
        root.calculate_force(&particle, theta.0, &mut force, g.0);

        particle.vel[0] += force[0] / particle.mass as f32;
        particle.vel[1] += force[1] / particle.mass as f32;
        particle.pos[0] += particle.vel[0];
        particle.pos[1] += particle.vel[1];
    }
}

fn par_gravity_quad(
    cam_bounds: Res<CameraBounds>, 
    g: Res<GravitationalConstant>,
    theta: Res<BarnesHutTheta>,
    mut query: Query<&mut Particle>
) {
    let bounds = [cam_bounds.x_min, cam_bounds.y_min, cam_bounds.x_max, cam_bounds.y_max];

    // Collect particles into a thread-safe vector
    let particles: Vec<_> = query.iter().map(|p| p.clone()).collect();
    
    // Build the quadtree (this remains single-threaded as it's sequential)
    let mut root = QuadNode::new(bounds);
    for particle in &particles {
        root.insert(particle.clone(), 0); // Pass depth = 0
    }

    // Calculate forces in parallel
    let forces: Vec<_> = particles.par_iter()
        .map(|particle| {
            let mut force = [0.0, 0.0];
            root.calculate_force(particle, theta.0, &mut force, g.0);
            force
        })
        .collect();

    // Apply forces to particles
    for (mut particle, force) in query.iter_mut().zip(forces.iter()) {
        particle.vel[0] += force[0] / particle.mass as f32;
        particle.vel[1] += force[1] / particle.mass as f32;
        particle.pos[0] += particle.vel[0];
        particle.pos[1] += particle.vel[1];
    }
}
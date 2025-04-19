
use bevy::prelude::*;
use bevy::input::ButtonState;
use bevy::input::keyboard::{Key, KeyboardInput};
use bevy::window::{WindowMode, WindowResolution};


/*todo: make sure bodies dont spawn outside the area, this can be done in the create all entites where we iterate through them all, 
check for collisions with the boundry and if it does remove  
3.turn the resolution into actual parameters so users can resize the screen
5.collissions with each other
5.gravity
6. make the r key restart the simulation
7.particle only bounce from the wall if their center touches, need to make it so it takes the radisu and bounces from that
*/

#[derive(Resource)]
struct NoOfParticle(u32);

#[derive(Resource)]
struct TextID(Option<Entity>);

#[derive(Resource)]
struct CameraBounds{
	x_min:f32,
	x_max:f32,
	y_min:f32,
	y_max:f32,
}

#[derive(States, Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum GameStates{
    TextState,
    SimulationState,
}

#[derive(Component)]
struct Particle{
	pos:[f32; 2],
	vel:[f32; 2],
	mass:u64,
	radius:u64,
}

fn main() {
   let mut app = App::new();

   app.add_plugins(DefaultPlugins.set(
    WindowPlugin{
        primary_window: Some(Window{
            resolution: WindowResolution::new(1920., 1080.).with_scale_factor_override(1.0),
            mode: WindowMode::Fullscreen(MonitorSelection::Primary),
            ..default()
        }),
        ..default()
    }

   ))
   .insert_resource(NoOfParticle(10)) //10 for now
   .insert_resource(TextID(None))
   .insert_resource(CameraBounds{x_min:-960., x_max:960., y_min:-540., y_max:540.})
   .add_systems(Startup, spawn_camera)
    // first state code
   .insert_state(GameStates::TextState)
   .add_systems(OnEnter(GameStates::TextState), text_setup)
   .add_systems(Update, text_update.run_if(in_state(GameStates::TextState)))

   //simulation state
   .add_systems(OnEnter(GameStates::SimulationState), create_all_entitites)
   .add_systems(Update, update_all_entities.run_if(in_state(GameStates::SimulationState)))
   ;
    app.run();
}

fn spawn_camera(
    mut commands: Commands,
) {
    commands.spawn(Camera2d);// spawn the camera
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
    no_of_particle: ResMut<NoOfParticle>,
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
					commands.entity(entity).despawn_recursive(); // despawn UI text
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
    //mut particles_list: ResMut<particles_list>
){
    let mass_max = 2000 as u64; //in kg
    let mass_min = 1000 as u64;

    let assumed_density = 55; // in kg/m3

        for _i in 0..no_of_part.0{
        let x = fastrand::i32(camera_bound.x_min as i32..camera_bound.x_max as i32); //CHANGE this to not be inclusive of borders
        let y = fastrand::i32(camera_bound.y_min as i32..camera_bound.y_max as i32);
        let vx = fastrand::f32() * if fastrand::bool() {1.0} else {-1.0};
        let vy = fastrand::f32() * if fastrand::bool() {1.0} else {-1.0};

        let mass = fastrand::u64(mass_min..mass_max);

        let radius:u64 = mass / assumed_density; // prolly better to fully randomise it 

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
   mut commands: Commands,
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
        
        particle.pos[0] += particle.vel[0] ;
        transform.translation.x = particle.pos[0];

        if particle.pos[1] + particle.vel[1] >= camera_bound.y_max 
        || particle.pos[1] + particle.vel[1] <= camera_bound.y_min{
            particle.vel[1] = particle.vel[1] * -1.0;
        } 
        particle.pos[1] += particle.vel[1];
        transform.translation.y = particle.pos[1];
    //update_position(query);
}
}
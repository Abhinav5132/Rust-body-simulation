use bevy::prelude::*;
use bevy::input::ButtonState;
use bevy::input::keyboard::{Key, KeyboardInput};
use bevy::window::{WindowMode, WindowResolution};


/*todo: make sure bodies dont spawn outside the area, this can be done in the create all entites where we iterate through them all, 
check for collisions with the boundry and if it does remove 
2: adding all the particles to a list 
3.spawning boundries and checking for collisions with the boundry objects
4.Make them move with an initial velocity
5.collissions with each other
5.gravity
*/

#[derive(Resource)]
struct no_of_particle(u32);

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
    textState,
    simulationstate,
}

#[derive(Component)]
struct Particle{
	pos:[f32; 2],
	vel:[f32; 2],
	mass:u64,
	radius:u64,
}

#[derive(Resource)]
struct particles_list{
    particles:Vec<Entity>
}

fn main() {
   let mut app = App::new();

   app.add_plugins((DefaultPlugins.set(
    WindowPlugin{
        primary_window: Some(Window{
            resolution: WindowResolution::new(1920., 1080.).with_scale_factor_override(1.0),
            mode: WindowMode::Fullscreen(MonitorSelection::Primary),
            ..default()
        }),
        ..default()
    }

   )))
   .insert_resource(no_of_particle(10)) //10 for now
   .insert_resource(TextID(None))
   .insert_resource(CameraBounds{x_min:-960., x_max:960., y_min:-540., y_max:540.})
   .add_systems(Startup, spawn_camera)
    //.add_systems(Startup, camera_bounds)
    // first state code
   .insert_state(GameStates::textState)
   .add_systems(OnEnter(GameStates::textState), text_setup)
   .add_systems(Update, text_update.run_if(in_state(GameStates::textState)))

   //simulation state
   .add_systems(OnEnter(GameStates::simulationstate), create_all_entitites)
   ;
    app.run();
}

fn spawn_camera(
    mut commands: Commands,
) {
    commands.spawn(Camera2d);// spawn the camera
}

/*fn camera_bounds(
	mut camera_bound: Query<(&Camera, &GlobalTransform), With<Camera2d>>,
    mut cam_bound: ResMut<CameraBounds>
){
	let (camera, transform) = camera_bound.single();

	if let Some(viewport_size) = camera.logical_viewport_size(){ // Option<Vec2>
        let center = transform.translation().truncate();

        let half_size = viewport_size / 2.0;

        let min = center - half_size;
        let max = center + half_size;
        cam_bound.x_min = min.x.floor();
        cam_bound.x_max = max.x.ceil(); 
        cam_bound.y_min = min.y.floor();
        cam_bound.y_max = max.y.ceil();
    }
} */

fn text_setup(
	mut commands: Commands,
	asset_server : Res<AssetServer>,
	mut text_id: ResMut<TextID>,
) {
    
    let mut id_text = commands.spawn(( // spawn in the text
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
    mut NoOfParticle: ResMut<no_of_particle>,
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

				next_state.set(GameStates::simulationstate);
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
    no_of_part: ResMut<no_of_particle>,
    mut materials: ResMut<Assets<ColorMaterial>>,
    mut meshes: ResMut<Assets<Mesh>>,
){
    let mass_max = 2000 as u64; //in kg
    let mass_min = 1000 as u64;

    let assumed_density = 55; // in kg/m3

    let cam_x_max = camera_bound.x_max.round() as i32;
    let cam_x_min = camera_bound.x_min.round() as i32;
    let cam_y_max = camera_bound.y_max.round() as i32;
    let cam_y_min = camera_bound.y_min.round() as i32;

    


        for i in (0..no_of_part.0){
        let mut x = fastrand::i32(cam_x_min..cam_x_max); //CHANGE this to not be inclusive of borders
        let mut y = fastrand::i32(cam_y_min..cam_y_max);

        let mut mass = fastrand::u64(mass_min..mass_max);

        let radius:u64 = (mass / assumed_density );

        let color = ColorMaterial::from(Color::linear_rgb(
            fastrand::f32(),
            fastrand::f32(),
            fastrand::f32(),
            ));
        
        commands.spawn((
            Particle {
            pos: [x as f32,y as f32],
            vel: [0.0, 0.0],
            mass: mass,
            radius: radius,

        },
        Mesh2d(meshes.add(Circle::new(radius as f32))),
        MeshMaterial2d(materials.add(color)),
        Transform::from_xyz(x as f32, y as f32, 0.0),
        ));
    }

}
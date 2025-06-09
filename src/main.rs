use bevy::input::ButtonState;
use bevy::input::keyboard::{Key, KeyboardInput};
use bevy::input::mouse::MouseWheel;
use bevy::prelude::*;
use bevy::window::{WindowMode, WindowResolution};
use bevy_dev_tools::fps_overlay::FpsOverlayPlugin;

#[derive(Resource)]
struct ParticlePositions {
    x: Vec<f32>,
    y: Vec<f32>,
    radius: Vec<u32>,
}

#[derive(States, Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum SimStates {
    TextState,
    SimulationState,
}

#[derive(Component, Debug, Clone, Copy)]
struct Particle {}

#[derive(Resource)]
struct NoOfParticle(u32);

#[derive(Resource)]
struct TextID(Option<Entity>);

#[derive(Resource)]
struct CameraBounds {
    bounds: [f32; 2],
}
unsafe extern "C" {
    unsafe fn initialize_particles(no_of_particles: u32, x_bounds: f32, y_bounds: f32);
}

unsafe extern "C" {
    unsafe fn read_particle_positions(h_pos_x: *mut f32, h_pos_y: *mut f32, no_of_particels: u32);
}

unsafe extern "C" {
    unsafe fn read_particle_radius(h_radius: *mut u32, no_of_particles: u32);
}

fn main() {
    let mut app = App::new();

    app.add_plugins((
        DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                resolution: WindowResolution::new(1920., 1080.).with_scale_factor_override(1.0),
                mode: WindowMode::Fullscreen(
                    MonitorSelection::Primary,
                    VideoModeSelection::Current,
                ),
                ..default()
            }),
            ..default()
        }),
        FpsOverlayPlugin::default(),
    ))
    .insert_resource(NoOfParticle(10000))
    .insert_resource(TextID(None))
    .insert_resource(CameraBounds {
        bounds: [3480.0, 2160.0],
    })
    .add_systems(Startup, spawn_camera)
    .add_systems(Update, camera_zoom)
    .insert_state(SimStates::TextState)
    .add_systems(Startup, text_setup)
    .add_systems(Update, text_update.run_if(in_state(SimStates::TextState)))
    .add_systems(OnEnter(SimStates::SimulationState), create_particles);

    app.run();
}

fn spawn_camera(mut commands: Commands) {
    commands.spawn(Camera2d);
}

fn camera_zoom(
    mut mouse_wheel_input: EventReader<MouseWheel>,
    mut keyboard_input: EventReader<KeyboardInput>,
    mut query: Query<&mut Transform, With<Camera2d>>,
) {
    let zoom_speed = 0.15;
    let pan_speed = 30.0;
    let mut zoom_delta = 0.0;

    for event in mouse_wheel_input.read() {
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
    for event in keyboard_input.read() {
        for mut transform in query.iter_mut() {
            match &event.logical_key {
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
                _ => {}
            }
        }
    }
}

fn text_setup(mut commands: Commands, asset_server: Res<AssetServer>, mut text_id: ResMut<TextID>) {
    let id_text = commands
        .spawn((
            Text::new("Please enter the number of bodies:"),
            TextFont {
                font: asset_server.load("fonts/BungeeSpice-Regular.ttf"),
                font_size: 67.0,
                ..default()
            },
            TextLayout::new_with_justify(JustifyText::Center),
            // Set the style of the Node itself.
            Node {
                position_type: PositionType::Absolute,
                ..default()
            },
        ))
        .id();

    text_id.0 = Some(id_text);
}

fn text_update(
    mut commands: Commands,
    mut evr_kbd: EventReader<KeyboardInput>,
    mut text_id: ResMut<TextID>,
    mut string: Local<String>,
    mut no_of_particle: ResMut<NoOfParticle>,
    mut query: Query<&mut Text>,
    mut next_state: ResMut<NextState<SimStates>>,
) {
    let mut text_changed = false;
    for ev in evr_kbd.read() {
        // We don't care about key releases, only key presses
        if ev.state == ButtonState::Released {
            continue;
        }

        match &ev.logical_key {
            Key::Character(abc) => {
                if abc.chars().any(|c| c.is_control()) {
                    continue;
                }

                for c in abc.chars() {
                    if c.is_ascii_digit() {
                        string.push(c);
                        text_changed = true;
                    }
                }
            }

            Key::Enter => {
                //write to Noofparticle

                if let Ok(num) = string.parse::<u32>() {
                    no_of_particle.0 = num;
                }

                if let Some(entity) = text_id.0 {
                    println!("no enetered: {}", &*string);
                    commands.entity(entity).despawn(); // despawn UI text
                    text_id.0 = None; // Clear the resource if needed
                }

                next_state.set(SimStates::SimulationState);
            }

            Key::Backspace => {
                string.pop();
            }

            _ => {}
        }
    }

    if text_changed {
        if let Some(entity) = text_id.0 {
            if let Ok(text) = query.get_mut(entity) {
                // text.0[0].value = string.clone();
            }
        }
    }
}

fn create_particles(
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

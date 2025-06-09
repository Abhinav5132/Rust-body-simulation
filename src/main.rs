
use bevy::prelude::*;
use bevy::window::{WindowMode, WindowResolution};
use bevy_dev_tools::fps_overlay::FpsOverlayPlugin;

pub mod derives;
pub use derives::*;

pub mod bindings;
pub use bindings::*;

pub mod setup;
pub use setup::*;

pub mod basic_particle_logic;
pub use basic_particle_logic::*;

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
    .add_systems(OnEnter(SimStates::SimulationState), create_particles)
    .add_systems(Update, update_particles.run_if(in_state(SimStates::SimulationState)))
    
    ;
    
    app.run();
}


use bevy::prelude::*;
use bevy::input::ButtonState;
use bevy::input::keyboard::{Key, KeyboardInput};


#[derive(Resource)]
struct no_of_particle(u32);

#[derive(Resource)]
struct TextID(Option<Entity>);

#[derive(States, Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum GameStates{
    textState,
    simulationstate,
}

fn main() {
   let mut app = App::new();

   app.add_plugins((DefaultPlugins))
   .insert_resource(no_of_particle(10)) //10 for now
   .insert_resource(TextID(None))
   .add_systems(Startup, spawn_camera)

    // first state code
   .insert_state(GameStates::textState)
   .add_systems(OnEnter(GameStates::textState), text_setup)
   .add_systems(Update, text_update.run_if(in_state(GameStates::textState)))

   //simulation state 
   ;


    app.run();
}

fn spawn_camera(
    mut commands: Commands
) {
    commands.spawn(Camera2d);// spawn the camera
}

fn text_setup(
	mut commands: Commands,
	mut asset_server : Res<AssetServer>,
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

				//next_state.set(GameStates::simulationstate);
            }

            Key::Backspace => {
                localstring.pop();
            }

            _ => {}
        }
    } 
}
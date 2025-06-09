use bevy::prelude::*;
//use crate::*;

#[derive(Component)]
pub struct ParticleIndex(pub usize);

#[derive(Resource)]
pub struct ParticlePositions {
    pub x: Vec<f32>,
    pub y: Vec<f32>,
    pub radius: Vec<u32>,
}

#[derive(States, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SimStates {
    TextState,
    SimulationState,
}

#[derive(Component, Debug, Clone, Copy)]
pub struct Particle {}

#[derive(Resource)]
pub struct NoOfParticle(pub u32);

#[derive(Resource)]
pub struct TextID(pub Option<Entity>);

#[derive(Resource)]
pub struct CameraBounds {
    pub bounds: [f32; 2],
}
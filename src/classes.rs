
use bevy::prelude::*;
use crate::Particle;

#[derive(Debug)]
pub struct KdNode {
    particle: Option<Particle>,
    left: Option<Box<KdNode>>,
    right: Option<Box<KdNode>>,
    split_axis: usize,
    split_value: f32,

}

impl KdNode{

    pub fn new(split_axis: usize)-> Self{
        KdNode { 
            particle: None, 
            left: None, 
            right: None, 
            split_axis,
            split_value: 0.0 
        }
    }

    pub fn build(particles: Vec<Particle>, depth: usize) -> Option<Box<KdNode>> {
        if particles.is_empty() {
            return None;
        }

        let axis = depth % 2;

        let mut sorted_particles = particles;
        sorted_particles.sort_by(|a, b| {
        let coord_a = if axis == 0 { a.pos[0] } else { a.pos[1] };
        let coord_b = if axis == 0 { b.pos[0] } else { b.pos[1] };
        coord_a.partial_cmp(&coord_b).unwrap()
        });

        let median = sorted_particles.len() / 2;
        let split_value = if axis == 0 {
            sorted_particles[median].pos[0]
        } else {
            sorted_particles[median].pos[1]
        };

        let mut node = KdNode::new(axis);
        node.split_value = split_value;
        node.particle = Some(sorted_particles[median].clone());

        let left_part = Vec::from(&sorted_particles[..median]);
        let right_part = Vec::from(&sorted_particles[median + 1 ..]);

        if median > 0 {
            node.left = KdNode::build(left_part, depth + 1);
        }
        if median +1 < sorted_particles.len() {
            node.right = KdNode::build(right_part, depth +1);
        } 

        Some(Box::new(node))  

    }

    pub fn check_collison(&self, particle: &Particle, collisions: &mut Vec<(Particle, f32, f32, f32, f32)>) {

        if let Some(ref node_particle) = self.particle {
            if node_particle.pos != particle.pos {
                let dx = particle.pos[0] - node_particle.pos[0];
                let dy = particle.pos[1] - node_particle.pos[1];
                let dist_square = dx * dx + dy * dy;
                let radius_sum = (particle.radius + node_particle.radius) as f32;

                if dist_square <= radius_sum * radius_sum {
                    let dist = dist_square.sqrt();
                    let nx = dx / dist;
                    let ny = dy / dist;

                    let overlap = radius_sum - dist;
                    let sep_x = overlap * 0.5 * nx;
                    let sep_y = overlap * 0.5 * ny;

                    collisions.push((node_particle.clone(), nx, ny, sep_x, sep_y));
                }
            }
        }

        let particle_cord = if self.split_axis == 0 {
            particle.pos[0]
        } else {
            particle.pos[1]
        };

        let search_radius = particle.radius as f32;

        if particle_cord - search_radius <= self.split_value {
            if let Some(ref left) = self.left {
                left.check_collison(particle, collisions);
            }
        }

        if particle_cord + search_radius >= self.split_value {
            if let Some (ref right) = self.right {
                right.check_collison(particle, collisions);
            }
        }

    }

}



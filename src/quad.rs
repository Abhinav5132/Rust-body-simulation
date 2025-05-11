
use crate::Particle;

pub struct node {
    pub bounds: [f32; 4], // [x_min, y_min, x_max, y_max]
    pub center_of_mass: [f32; 2],
    pub mass: u64,
    pub particle: Option<Particle>,
    pub top_left: Option<Box<node>>,
    pub top_right: Option<Box<node>>,
    pub bot_left: Option<Box<node>>,
    pub bot_right: Option<Box<node>>,

}

impl node {
    pub fn new(bounds: [f32; 4]) ->Self {
        node{
            bounds: bounds,
            center_of_mass: [0.0, 0.0],
            mass: 0,
            particle: None,
            top_left: None,
            top_right: None,
            bot_left: None,
            bot_right: None,
        }
    }
    pub fn is_a_leaf(&self) -> bool {
        self.top_left.is_none() && self.top_right.is_none() &&
        self.bot_left.is_none() && self.bot_right.is_none()
    }
    pub fn insert (&mut self, particle: Particle) {

        self.mass += particle.mass;
        self.center_of_mass[0] = (self.center_of_mass[0] * 
            (self.mass - particle.mass) as f32 + particle.pos[0] * particle.mass as f32) / self.mass as f32;
        self.center_of_mass[1] = (self.center_of_mass[1] * 
            (self.mass - particle.mass) as f32 + particle.pos[1] * particle.mass as f32) / self.mass as f32;

        if self.particle.is_none() && self.is_a_leaf() {
            self.particle = Some(particle);
            return;
        }

        if self.is_a_leaf() {
            self.subdivide();
        }

        let mid_x = (self.bounds[0] + self.bounds[2]) / 2.0;
        let mid_y = (self.bounds[1] + self.bounds[3]) / 2.0;

        let quadrant = match (particle.pos[0] <= mid_x, particle.pos[1] <= mid_y) {
            (true, true) => &mut self.bot_left,
            (true, false) => &mut self.top_left,
            (false, true) => &mut self.bot_right,
            (false, false) => &mut self.top_right,
        };
    
        if let Some(child) = quadrant {
            child.insert(particle);
        } else { // chatgpt code
            // Create the child node if it doesn't exist
            let new_bounds = match (particle.pos[0] <= mid_x, particle.pos[1] <= mid_y) {
                (true, true) => [self.bounds[0], self.bounds[1], mid_x, mid_y],
                (true, false) => [self.bounds[0], mid_y, mid_x, self.bounds[3]],
                (false, true) => [mid_x, self.bounds[1], self.bounds[2], mid_y],
                (false, false) => [mid_x, mid_y, self.bounds[2], self.bounds[3]],
            };
            let mut new_node = Box::new(node::new(new_bounds));
            new_node.insert(particle);
            *quadrant = Some(new_node);
        }
    }
    pub fn subdivide(&mut self) {
        let mid_x = (self.bounds[0] + self.bounds[2]) / 2.0;
        let mid_y = (self.bounds[1] + self.bounds[3]) / 2.0;

        self.top_left = Some(Box::new(node::new([self.bounds[0], mid_y, mid_x, self.bounds[3]])));
        self.top_right = Some(Box::new(node::new([mid_x, mid_y,self.bounds[2], self.bounds[3]])));
        self.bot_right = Some(Box::new(node::new([mid_x, self.bounds[1],self.bounds[2], mid_y])));
        self.bot_left = Some(Box::new(node::new([self.bounds[0], self.bounds[1], mid_x, mid_y])));

        if let Some(existing_particle) = self.particle.take() {
            self.insert(existing_particle);
        }
    }

    pub fn calculate_force(&self, particle: &Particle, theta: f32 , force : &mut [f32; 2], g: f32) {
        if self.mass == 0 {
            return;
        }

        let dx = self.center_of_mass[0] - particle.pos[0];
        let dy = self.center_of_mass[1] - particle.pos[1];
        let dist_sq = dx *dx + dy *dy;
        if dist_sq < f32::EPSILON {
            return;
        }
        let dist = dist_sq.sqrt();

        let width = self.bounds[2] - self.bounds[0];

        if self.is_a_leaf() || (width / dist < theta) {
            let f = (g * self.mass as f32 * particle.mass as f32 )/ dist_sq;

            force[0] += f * (dx / dist) / 500.0;
            force[1] += f * (dy / dist) / 500.0;
            return;
        }   

        for child in [&self.top_left, &self.top_right, &self.bot_left, &self.bot_right].iter() {
            if let Some(node) = child {
                node.calculate_force(particle, theta, force, g);
            }
        }
    }
}
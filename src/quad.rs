use crate::Particle;

pub struct QuadNode {
    pub bounds: [f32; 4], // [x_min, y_min, x_max, y_max]
    pub center_of_mass: [f32; 2],
    pub mass: u64,
    pub particles: Vec<Particle>, // Store multiple particles if needed
    pub top_left: Option<Box<QuadNode>>,
    pub top_right: Option<Box<QuadNode>>,
    pub bot_left: Option<Box<QuadNode>>,
    pub bot_right: Option<Box<QuadNode>>,
}

impl QuadNode {
    const MAX_DEPTH: usize = 32;
    const NODE_CAPACITY: usize = 32;

    pub fn new(bounds: [f32; 4]) -> Self {
        QuadNode {
            bounds,
            center_of_mass: [0.0, 0.0],
            mass: 0,
            particles: Vec::new(),
            top_left: None,
            top_right: None,
            bot_left: None,
            bot_right: None,
        }
    }

    pub fn is_leaf(&self) -> bool {
        self.top_left.is_none() && self.top_right.is_none() &&
        self.bot_left.is_none() && self.bot_right.is_none()
    }

    pub fn insert(&mut self, particle: Particle, depth: usize) {
        // Prevent infinite subdivision
        if depth >= Self::MAX_DEPTH {
            self.particles.push(particle);
            return;
        }

        // If leaf and under capacity, just store
        if self.is_leaf() && self.particles.len() < Self::NODE_CAPACITY {
            self.particles.push(particle);
            return;
        }

        // If leaf but at capacity, subdivide and move particles to children
        if self.is_leaf() {
            self.subdivide();
            let old_particles = std::mem::take(&mut self.particles);
            for p in old_particles {
                self.insert(p, depth + 1);
            }
        }

        // Insert into correct child
        let mid_x = (self.bounds[0] + self.bounds[2]) / 2.0;
        let mid_y = (self.bounds[1] + self.bounds[3]) / 2.0;
        let quadrant = match (particle.pos[0] <= mid_x, particle.pos[1] <= mid_y) {
            (true, true) => &mut self.bot_left,
            (true, false) => &mut self.top_left,
            (false, true) => &mut self.bot_right,
            (false, false) => &mut self.top_right,
        };
        if let Some(child) = quadrant {
            child.insert(particle, depth + 1);
        }
    }

    pub fn subdivide(&mut self) {
        let mid_x = (self.bounds[0] + self.bounds[2]) / 2.0;
        let mid_y = (self.bounds[1] + self.bounds[3]) / 2.0;
        self.top_left = Some(Box::new(QuadNode::new([self.bounds[0], mid_y, mid_x, self.bounds[3]])));
        self.top_right = Some(Box::new(QuadNode::new([mid_x, mid_y, self.bounds[2], self.bounds[3]])));
        self.bot_right = Some(Box::new(QuadNode::new([mid_x, self.bounds[1], self.bounds[2], mid_y])));
        self.bot_left = Some(Box::new(QuadNode::new([self.bounds[0], self.bounds[1], mid_x, mid_y])));
    }

    pub fn calculate_force(&self, particle: &Particle, theta: f32, force: &mut [f32; 2], g: f32) {
        if self.mass == 0 {
            return;
        }

        let dx = self.center_of_mass[0] - particle.pos[0];
        let dy = self.center_of_mass[1] - particle.pos[1];
        let dist_sq = dx * dx + dy * dy;
        if dist_sq < f32::EPSILON {
            return;
        }
        let dist = dist_sq.sqrt();

        let width = self.bounds[2] - self.bounds[0];

        if self.is_leaf() || (width / dist < theta) {
            let f = (g * self.mass as f32 * particle.mass as f32) / dist_sq;
            force[0] += f * (dx / dist) / 250.0;
            force[1] += f * (dy / dist) / 250.0;
            return;
        }

        for child in [&self.top_left, &self.top_right, &self.bot_left, &self.bot_right].iter() {
            if let Some(node) = child {
                node.calculate_force(particle, theta, force, g);
            }
        }
    }
}
//! HiDream Schedulers
//!
//! This module implements the schedulers used by HiDream models.

use candle::{Result, Tensor, Device, DType, D};
use std::f64::consts::PI;

/// Flow Matching Euler Discrete Scheduler
#[derive(Debug, Clone)]
pub struct FlowMatchEulerDiscreteScheduler {
    pub num_train_timesteps: usize,
    pub shift: f64,
    pub use_dynamic_shifting: bool,
    pub timesteps: Vec<f64>,
    pub sigmas: Vec<f64>,
}

impl FlowMatchEulerDiscreteScheduler {
    pub fn new(num_train_timesteps: usize, shift: f64, use_dynamic_shifting: bool) -> Self {
        Self {
            num_train_timesteps,
            shift,
            use_dynamic_shifting,
            timesteps: Vec::new(),
            sigmas: Vec::new(),
        }
    }

    pub fn set_timesteps(&mut self, num_inference_steps: usize, device: &Device) -> Result<()> {
        let step_ratio = self.num_train_timesteps as f64 / num_inference_steps as f64;
        let mut timesteps = Vec::new();
        
        for i in 0..num_inference_steps {
            let timestep = (num_inference_steps - 1 - i) as f64 * step_ratio;
            timesteps.push(timestep);
        }
        
        self.timesteps = timesteps;
        
        // Calculate sigmas based on flow matching formulation
        let mut sigmas = Vec::new();
        for &t in &self.timesteps {
            let sigma = self.calculate_sigma(t);
            sigmas.push(sigma);
        }
        self.sigmas = sigmas;
        
        Ok(())
    }

    fn calculate_sigma(&self, timestep: f64) -> f64 {
        let t_norm = timestep / self.num_train_timesteps as f64;
        if self.use_dynamic_shifting {
            // Dynamic shifting based on sequence length
            t_norm * self.shift.exp()
        } else {
            t_norm * self.shift
        }
    }

    pub fn step(&self, model_output: &Tensor, timestep: f64, sample: &Tensor) -> Result<Tensor> {
        // Euler step for flow matching
        let dt = if let Some(idx) = self.timesteps.iter().position(|&x| (x - timestep).abs() < 1e-6) {
            if idx < self.timesteps.len() - 1 {
                self.timesteps[idx] - self.timesteps[idx + 1]
            } else {
                self.timesteps[idx]
            }
        } else {
            1.0 // Default step size
        };

        // Flow matching update: x_{t-dt} = x_t + dt * v_t
        let dt_tensor = Tensor::new(dt as f32, sample.device())?;
        let update = model_output.broadcast_mul(&dt_tensor)?;
        sample.broadcast_add(&update)
    }
}

/// Flow UniPC Multistep Scheduler
#[derive(Debug, Clone)]
pub struct FlowUniPCMultistepScheduler {
    pub num_train_timesteps: usize,
    pub shift: f64,
    pub timesteps: Vec<f64>,
    pub order: usize,
    pub lower_order_nums: usize,
}

impl FlowUniPCMultistepScheduler {
    pub fn new(num_train_timesteps: usize, shift: f64) -> Self {
        Self {
            num_train_timesteps,
            shift,
            timesteps: Vec::new(),
            order: 3, // Default order for UniPC
            lower_order_nums: 0,
        }
    }

    pub fn set_timesteps(&mut self, num_inference_steps: usize, device: &Device) -> Result<()> {
        let step_ratio = self.num_train_timesteps as f64 / num_inference_steps as f64;
        let mut timesteps = Vec::new();
        
        for i in 0..num_inference_steps {
            let timestep = (num_inference_steps - 1 - i) as f64 * step_ratio;
            timesteps.push(timestep);
        }
        
        self.timesteps = timesteps;
        self.lower_order_nums = 0;
        
        Ok(())
    }

    pub fn step(&mut self, model_output: &Tensor, timestep: f64, sample: &Tensor) -> Result<Tensor> {
        // Simplified UniPC step - in practice this would be more complex
        let dt = if let Some(idx) = self.timesteps.iter().position(|&x| (x - timestep).abs() < 1e-6) {
            if idx < self.timesteps.len() - 1 {
                self.timesteps[idx] - self.timesteps[idx + 1]
            } else {
                self.timesteps[idx]
            }
        } else {
            1.0
        };

        // For now, use simple Euler step
        let dt_tensor = Tensor::new(dt as f32, sample.device())?;
        let update = model_output.broadcast_mul(&dt_tensor)?;
        let result = sample.broadcast_add(&update)?;
        
        self.lower_order_nums += 1;
        Ok(result)
    }
}

/// Flash Flow Match Euler Discrete Scheduler (for fast inference)
#[derive(Debug, Clone)]
pub struct FlashFlowMatchEulerDiscreteScheduler {
    pub num_train_timesteps: usize,
    pub shift: f64,
    pub timesteps: Vec<f64>,
}

impl FlashFlowMatchEulerDiscreteScheduler {
    pub fn new(num_train_timesteps: usize, shift: f64) -> Self {
        Self {
            num_train_timesteps,
            shift,
            timesteps: Vec::new(),
        }
    }

    pub fn set_timesteps(&mut self, num_inference_steps: usize, device: &Device) -> Result<()> {
        // Optimized timestep schedule for fast inference
        let mut timesteps = Vec::new();
        
        // Use exponential spacing for better quality with fewer steps
        for i in 0..num_inference_steps {
            let alpha = i as f64 / (num_inference_steps - 1) as f64;
            let timestep = self.num_train_timesteps as f64 * (1.0 - alpha.powf(self.shift));
            timesteps.push(timestep);
        }
        
        timesteps.reverse(); // Start from high timestep
        self.timesteps = timesteps;
        
        Ok(())
    }

    pub fn step(&self, model_output: &Tensor, timestep: f64, sample: &Tensor) -> Result<Tensor> {
        // Fast Euler step with optimized scaling
        let dt = if let Some(idx) = self.timesteps.iter().position(|&x| (x - timestep).abs() < 1e-6) {
            if idx < self.timesteps.len() - 1 {
                (self.timesteps[idx] - self.timesteps[idx + 1]) * 0.5 // Reduced step size for stability
            } else {
                self.timesteps[idx] * 0.5
            }
        } else {
            0.5
        };

        let dt_tensor = Tensor::new(dt as f32, sample.device())?;
        let update = model_output.broadcast_mul(&dt_tensor)?;
        sample.broadcast_add(&update)
    }
}

/// Helper function to calculate shift based on sequence length (from Python reference)
pub fn calculate_shift(
    image_seq_len: usize,
    base_seq_len: usize,
    max_seq_len: usize,
    base_shift: f64,
    max_shift: f64,
) -> f64 {
    let m = (max_shift - base_shift) / (max_seq_len - base_seq_len) as f64;
    let b = base_shift - m * base_seq_len as f64;
    image_seq_len as f64 * m + b
}

/// Retrieve timesteps helper function
pub fn retrieve_timesteps(
    scheduler: &mut dyn SchedulerTrait,
    num_inference_steps: Option<usize>,
    device: &Device,
    timesteps: Option<Vec<f64>>,
    sigmas: Option<Vec<f64>>,
) -> Result<(Vec<f64>, usize)> {
    if timesteps.is_some() && sigmas.is_some() {
        return Err(candle::Error::Msg("Only one of `timesteps` or `sigmas` can be passed".into()));
    }

    if let Some(custom_timesteps) = timesteps {
        let num_inference_steps = custom_timesteps.len();
        Ok((custom_timesteps, num_inference_steps))
    } else if let Some(_custom_sigmas) = sigmas {
        // Handle custom sigmas - simplified for now
        let num_inference_steps = num_inference_steps.unwrap_or(50);
        scheduler.set_timesteps(num_inference_steps, device)?;
        Ok((scheduler.get_timesteps(), num_inference_steps))
    } else {
        let num_inference_steps = num_inference_steps.unwrap_or(50);
        scheduler.set_timesteps(num_inference_steps, device)?;
        Ok((scheduler.get_timesteps(), num_inference_steps))
    }
}

/// Trait for scheduler interface
pub trait SchedulerTrait {
    fn set_timesteps(&mut self, num_inference_steps: usize, device: &Device) -> Result<()>;
    fn step(&mut self, model_output: &Tensor, timestep: f64, sample: &Tensor) -> Result<Tensor>;
    fn get_timesteps(&self) -> Vec<f64>;
}

impl SchedulerTrait for FlowMatchEulerDiscreteScheduler {
    fn set_timesteps(&mut self, num_inference_steps: usize, device: &Device) -> Result<()> {
        self.set_timesteps(num_inference_steps, device)
    }

    fn step(&mut self, model_output: &Tensor, timestep: f64, sample: &Tensor) -> Result<Tensor> {
        self.step(model_output, timestep, sample)
    }

    fn get_timesteps(&self) -> Vec<f64> {
        self.timesteps.clone()
    }
}

impl SchedulerTrait for FlowUniPCMultistepScheduler {
    fn set_timesteps(&mut self, num_inference_steps: usize, device: &Device) -> Result<()> {
        self.set_timesteps(num_inference_steps, device)
    }

    fn step(&mut self, model_output: &Tensor, timestep: f64, sample: &Tensor) -> Result<Tensor> {
        self.step(model_output, timestep, sample)
    }

    fn get_timesteps(&self) -> Vec<f64> {
        self.timesteps.clone()
    }
}

impl SchedulerTrait for FlashFlowMatchEulerDiscreteScheduler {
    fn set_timesteps(&mut self, num_inference_steps: usize, device: &Device) -> Result<()> {
        self.set_timesteps(num_inference_steps, device)
    }

    fn step(&mut self, model_output: &Tensor, timestep: f64, sample: &Tensor) -> Result<Tensor> {
        self.step(model_output, timestep, sample)
    }

    fn get_timesteps(&self) -> Vec<f64> {
        self.timesteps.clone()
    }
}

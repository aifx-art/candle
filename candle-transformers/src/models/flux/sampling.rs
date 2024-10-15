use candle::{Device, Result, Tensor};
use serde::de;

pub fn get_noise(
    num_samples: usize,
    height: usize,
    width: usize,
    device: &Device,
) -> Result<Tensor> {
    let height = (height + 15) / 16 * 2;
    let width = (width + 15) / 16 * 2;
    Tensor::randn(0f32, 1., (num_samples, 16, height, width), device)
}

#[derive(Debug, Clone)]
pub struct State {
    pub img: Tensor,
    pub img_ids: Tensor,
    pub txt: Tensor,
    pub txt_ids: Tensor,
    pub vec: Tensor,
}

impl State {
    pub fn new(t5_emb: &Tensor, clip_emb: &Tensor, img: &Tensor) -> Result<Self> {
        let dtype = img.dtype();
        let (bs, c, h, w) = img.dims4()?;
        let dev = img.device();
        let img = img.reshape((bs, c, h / 2, 2, w / 2, 2))?; // (b, c, h, ph, w, pw)
        let img = img.permute((0, 2, 4, 1, 3, 5))?; // (b, h, w, c, ph, pw)
        let img = img.reshape((bs, h / 2 * w / 2, c * 4))?;
        let img_ids = Tensor::stack(
            &[
                Tensor::full(0u32, (h / 2, w / 2), dev)?,
                Tensor::arange(0u32, h as u32 / 2, dev)?
                    .reshape(((), 1))?
                    .broadcast_as((h / 2, w / 2))?,
                Tensor::arange(0u32, w as u32 / 2, dev)?
                    .reshape((1, ()))?
                    .broadcast_as((h / 2, w / 2))?,
            ],
            2,
        )?
        .to_dtype(dtype)?;
        let img_ids = img_ids.reshape((1, h / 2 * w / 2, 3))?;
        let img_ids = img_ids.repeat((bs, 1, 1))?;
        let txt = t5_emb.repeat(bs)?;
        let txt_ids = Tensor::zeros((bs, txt.dim(1)?, 3), dtype, dev)?;
        let vec = clip_emb.repeat(bs)?;
        Ok(Self {
            img,
            img_ids,
            txt,
            txt_ids,
            vec,
        })
    }
}

fn time_shift(mu: f64, sigma: f64, t: f64) -> f64 {
    let e = mu.exp();
    e / (e + (1. / t - 1.).powf(sigma))
}

/// `shift` is a triple `(image_seq_len, base_shift, max_shift)`.
pub fn get_schedule(num_steps: usize, shift: Option<(usize, f64, f64)>) -> Vec<f64> {
    let timesteps: Vec<f64> = (0..=num_steps)
        .map(|v| v as f64 / num_steps as f64)
        .rev()
        .collect();
    match shift {
        None => timesteps,
        Some((image_seq_len, y1, y2)) => {
            let (x1, x2) = (256., 4096.);
            let m = (y2 - y1) / (x2 - x1);
            let b = y1 - m * x1;
            let mu = m * image_seq_len as f64 + b;
            timesteps
                .into_iter()
                .map(|v| time_shift(mu, 1., v))
                .collect()
        }
    }
}

pub fn unpack(xs: &Tensor, height: usize, width: usize) -> Result<Tensor> {
    let (b, _h_w, c_ph_pw) = xs.dims3()?;
    let height = (height + 15) / 16;
    let width = (width + 15) / 16;
    xs.reshape((b, height, width, c_ph_pw / 4, 2, 2))? // (b, h, w, c, ph, pw)
        .permute((0, 3, 1, 4, 2, 5))? // (b, c, h, ph, w, pw)
        .reshape((b, c_ph_pw / 4, height * 2, width * 2))
}

fn exponential_decay(total_steps: usize, current_step: usize) -> f64 {
    let k = 10.0; // Controls the steepness of the decay
    let t = (current_step) as f64 / (total_steps) as f64; // Normalize current step to [0, 1]
    //std::f64::consts::E.powf(-k * t) // Exponential decay formula
    (std::f64::consts::E).powf(-k * t)
}

fn cos_decay(total_steps: usize, current_step: usize) -> f64 {
    // Calculate the cosine decay
    let t = current_step as f64;
    let T = total_steps as f64;

    // Using the cosine decay formula
    (std::f64::consts::PI * t / (2.0 * T)).cos().powi(2)
}

#[allow(clippy::too_many_arguments)]
pub fn denoise<M: super::WithForward>(
    model: &M,
    img: &Tensor,
    img_ids: &Tensor,
    txt: &Tensor,
    txt_ids: &Tensor,
    vec_: &Tensor,
    timesteps: &[f64],
    guidance: f64,
    eta: f64,
) -> Result<Tensor> {
    let b_sz = img.dim(0)?;
    let dev = img.device();
    let guidance = Tensor::full(guidance as f32, b_sz, dev)?;

    let mut img = img.clone();
    let mut current_step = 0usize;
    for window in timesteps.windows(2) {
        let (t_curr, t_prev) = match window {
            [a, b] => (a, b),
            _ => continue,
        };
        let t_vec = Tensor::full(*t_curr as f32, b_sz, dev)?;

        let sigma_down = *t_prev - *t_curr;
        println!(
            "flux current step {} - t_curr {} t_prev{}",
            current_step, t_curr, t_prev
        );
        let decay_value = exponential_decay(timesteps.len(), current_step);
        println!("flux current step {} decay {}", current_step, decay_value);
        current_step += 1;
        let sigma_down_sqrt = sigma_down;
        println!("sigma_down_sqrt {:?}",sigma_down_sqrt);
        let stdev = eta * decay_value * sigma_down_sqrt;
        println!(
            "flux current step {} flux add noise {:?}",
            current_step, stdev,
        );
        let noise = img.randn_like(0.0, stdev)?;
        img = (img + noise)?;
        let pred = model.forward(&img, img_ids, txt, txt_ids, &t_vec, vec_, Some(&guidance))?;
        img = (img + pred * (t_prev - t_curr))?
    }
    Ok(img)
}

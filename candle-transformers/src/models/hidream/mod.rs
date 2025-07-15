//! HiDream: Instruction-based Image Editing Model
//!
//! This module implements the HiDream model in Candle, supporting both HiDream-I1 (generation) and HiDream-E1 (editing).
//! Based on the provided Python reference and Flux implementation.

pub mod schedulers;

use candle::{DType, Device, IndexOp, Module, Result, Tensor, D};
use candle_nn::{layer_norm, linear, Activation, LayerNorm, Linear, RmsNorm, VarBuilder};
use std::collections::HashMap;
use std::vec::Vec;

// Import attention function from flux
use super::flux::model::attention;

// Helper function for repeat_interleave
fn repeat_interleave(t: &Tensor, repeats: usize, dim: usize) -> Result<Tensor> {
    let shape = t.shape().dims().to_vec();
    let mut new_shape = shape.clone();
    new_shape[dim] *= repeats;
    let mut parts = Vec::with_capacity(repeats);
    for _ in 0..repeats {
        parts.push(t.clone());
    }
    Tensor::cat(&parts, dim)?.reshape(new_shape)
}

// Helper function for topk (simplified implementation)
fn topk(tensor: &Tensor, k: usize, _dim: isize) -> Result<(Tensor, Tensor)> {
    // This is a simplified implementation - in practice you'd want a more efficient version
    let sorted = tensor.arg_sort_last_dim(false)?;
    let values = tensor.gather(&sorted, D::Minus1)?;
    let indices = sorted;
    
    // Take top k
    let k_indices = indices.narrow(D::Minus1, 0, k)?;
    let k_values = values.narrow(D::Minus1, 0, k)?;
    
    Ok((k_values, k_indices))
}

// Helper function for mask_where (simplified implementation)
fn mask_where(tensor: &Tensor, mask: &Tensor, other: &Tensor) -> Result<Tensor> {
    // Simple implementation: where mask is true, use tensor, else use other
    let mask_f = mask.to_dtype(tensor.dtype())?;
    let inv_mask = (1.0 - &mask_f)?;
    (tensor * &mask_f)? + (other * &inv_mask)?
}

// Helper function for masked_fill (simplified implementation)
fn masked_fill(tensor: &Tensor, mask: &Tensor, value: &Tensor) -> Result<Tensor> {
    let mask_f = mask.to_dtype(tensor.dtype())?;
    let inv_mask = (1.0 - &mask_f)?;
    (tensor * &inv_mask)? + (value * &mask_f)?
}

// Timestep embedding function from Flux
fn timestep_embedding(t: &Tensor, dim: usize, dtype: DType) -> Result<Tensor> {
    const TIME_FACTOR: f64 = 1000.;
    const MAX_PERIOD: f64 = 10000.;
    if dim % 2 == 1 {
        candle::bail!("{dim} is odd")
    }
    let dev = t.device();
    let half = dim / 2;
    let t = (t * TIME_FACTOR)?;
    let arange = Tensor::arange(0u32, half as u32, dev)?.to_dtype(DType::F32)?;
    let freqs = (arange * (-MAX_PERIOD.ln() / half as f64))?.exp()?;
    let args = t
        .unsqueeze(1)?
        .to_dtype(DType::F32)?
        .broadcast_mul(&freqs.unsqueeze(0)?)?;
    let emb = Tensor::cat(&[args.cos()?, args.sin()?], D::Minus1)?.to_dtype(dtype)?;
    Ok(emb)
}

// EmbedND from Flux
#[derive(Debug, Clone)]
struct EmbedNd {
    theta: usize,
    axes_dim: Vec<usize>,
}

impl EmbedNd {
    fn new(theta: usize, axes_dim: Vec<usize>) -> Self {
        Self { theta, axes_dim }
    }
}

impl Module for EmbedNd {
    fn forward(&self, ids: &Tensor) -> Result<Tensor> {
        let n_axes = ids.dim(D::Minus1)?;
        let mut emb = Vec::with_capacity(n_axes);
        for idx in 0..n_axes {
            let r = rope(
                &ids.i((.., idx))?,
                self.axes_dim[idx],
                self.theta,
            )?;
            emb.push(r);
        }
        let emb = Tensor::cat(&emb, D::Minus1)?;
        emb.unsqueeze(1)
    }
}

// PatchEmbed from Python (Linear)
#[derive(Debug, Clone)]
struct PatchEmbed {
    proj: Linear,
}

impl PatchEmbed {
    fn new(patch_size: usize, in_channels: usize, out_channels: usize, vb: VarBuilder) -> Result<Self> {
        let proj = linear(in_channels * patch_size * patch_size, out_channels, vb.pp("proj"))?;
        Ok(Self { proj })
    }
}

impl Module for PatchEmbed {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.proj.forward(x)
    }
}

// Timesteps
#[derive(Debug, Clone)]
struct Timesteps {
    num_channels: usize,
    flip_sin_to_cos: bool,
    downscale_freq_shift: f64,
}

impl Timesteps {
    fn new(num_channels: usize, flip_sin_to_cos: bool, downscale_freq_shift: f64) -> Self {
        Self { num_channels, flip_sin_to_cos, downscale_freq_shift }
    }
}

impl Module for Timesteps {
    fn forward(&self, t: &Tensor) -> Result<Tensor> {
        timestep_embedding(t, self.num_channels, t.dtype())
    }
}

// TimestepEmbedding
#[derive(Debug, Clone)]
struct TimestepEmbedding {
    timestep_embedder: Linear,
}

impl TimestepEmbedding {
    fn new(in_channels: usize, time_embed_dim: usize, vb: VarBuilder) -> Result<Self> {
        let timestep_embedder = linear(in_channels, time_embed_dim, vb.pp("timestep_embedder"))?;
        Ok(Self { timestep_embedder })
    }
}

impl Module for TimestepEmbedding {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.timestep_embedder.forward(x)
    }
}

// TimestepEmbed
#[derive(Debug, Clone)]
struct TimestepEmbed {
    time_proj: Timesteps,
    timestep_embedder: TimestepEmbedding,
}

impl TimestepEmbed {
    fn new(hidden_size: usize, frequency_embedding_size: usize, vb: VarBuilder) -> Result<Self> {
        let time_proj = Timesteps::new(frequency_embedding_size, true, 0.0);
        let timestep_embedder = TimestepEmbedding::new(frequency_embedding_size, hidden_size, vb.pp("timestep_embedder"))?;
        Ok(Self { time_proj, timestep_embedder })
    }
}

impl Module for TimestepEmbed {
    fn forward(&self, t: &Tensor) -> Result<Tensor> {
        let t_emb = self.time_proj.forward(t)?;
        self.timestep_embedder.forward(&t_emb)
    }
}

// PooledEmbed
#[derive(Debug, Clone)]
struct PooledEmbed {
    pooled_embedder: TimestepEmbedding,
}

impl PooledEmbed {
    fn new(text_emb_dim: usize, hidden_size: usize, vb: VarBuilder) -> Result<Self> {
        let pooled_embedder = TimestepEmbedding::new(text_emb_dim, hidden_size, vb.pp("pooled_embedder"))?;
        Ok(Self { pooled_embedder })
    }
}

impl Module for PooledEmbed {
    fn forward(&self, pooled_embed: &Tensor) -> Result<Tensor> {
        self.pooled_embedder.forward(pooled_embed)
    }
}

// TextProjection
#[derive(Debug, Clone)]
struct TextProjection {
    linear: Linear,
}

impl TextProjection {
    fn new(in_features: usize, hidden_size: usize, vb: VarBuilder) -> Result<Self> {
        let linear = linear(in_features, hidden_size, vb.pp("linear"))?;
        Ok(Self { linear })
    }
}

impl Module for TextProjection {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.linear.forward(x)
    }
}

// HDFeedForwardSwiGLU
#[derive(Debug, Clone)]
struct HDFeedForwardSwiGLU {
    w1: Linear,
    w2: Linear,
    w3: Linear,
}

impl HDFeedForwardSwiGLU {
    fn new(dim: usize, hidden_dim: usize, vb: VarBuilder) -> Result<Self> {
        let w1 = linear(dim, hidden_dim, vb.pp("w1"))?;
        let w2 = linear(hidden_dim, dim, vb.pp("w2"))?;
        let w3 = linear(dim, hidden_dim, vb.pp("w3"))?;
        Ok(Self { w1, w2, w3 })
    }
}

impl Module for HDFeedForwardSwiGLU {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let a = self.w1.forward(x)?.silu()?;
        let b = self.w3.forward(x)?;
        (a * b)?.apply(&self.w2)
    }
}

// HDMoEGate
#[derive(Debug, Clone)]
struct HDMoEGate {
    weight: Tensor,
    top_k: usize,
}

impl HDMoEGate {
    fn new(dim: usize, num_routed_experts: usize, num_activated_experts: usize, vb: VarBuilder) -> Result<Self> {
        let weight = vb.get((num_routed_experts, dim), "weight")?;
        Ok(Self { weight, top_k: num_activated_experts })
    }

    fn forward(&self, x: &Tensor) -> Result<(Tensor, Tensor)> {
        let logits = x.matmul(&self.weight.t()?)?;
        let scores = candle_nn::ops::softmax(&logits, D::Minus1)?;
        topk(&scores, self.top_k, -1)
    }
}

// HDMOEFeedForwardSwiGLU
#[derive(Debug, Clone)]
struct HDMOEFeedForwardSwiGLU {
    shared_experts: HDFeedForwardSwiGLU,
    experts: Vec<HDFeedForwardSwiGLU>,
    gate: HDMoEGate,
    num_activated_experts: usize,
    num_routed_experts: usize,
}

impl HDMOEFeedForwardSwiGLU {
    fn new(
        dim: usize,
        hidden_dim: usize,
        num_routed_experts: usize,
        num_activated_experts: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let shared_experts = HDFeedForwardSwiGLU::new(dim, hidden_dim / 2, vb.pp("shared_experts"))?;
        let mut experts = Vec::with_capacity(num_routed_experts);
        for i in 0..num_routed_experts {
            experts.push(HDFeedForwardSwiGLU::new(dim, hidden_dim, vb.pp(&format!("experts.{}", i)))?);
        }
        let gate = HDMoEGate::new(dim, num_routed_experts, num_activated_experts, vb.pp("gate"))?;
        Ok(Self { shared_experts, experts, gate, num_activated_experts, num_routed_experts })
    }
}

impl Module for HDMOEFeedForwardSwiGLU {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let y_shared = self.shared_experts.forward(x)?;
        let (topk_weight, topk_idx) = self.gate.forward(x)?;
        let tk_idx_flat = topk_idx.flatten_all()?.to_dtype(DType::U32)?;
        let x_repeated = repeat_interleave(x, self.num_activated_experts, 0)?;
        let mut y = Tensor::zeros(x_repeated.shape(), x.dtype(), x.device())?;
        for (i, expert) in self.experts.iter().enumerate() {
            let mask = tk_idx_flat.eq(&Tensor::new(i as u32, x.device())?)?;
            let x_sel = mask_where(&x_repeated, &mask.broadcast_as(x_repeated.shape())?, &Tensor::zeros(x_repeated.shape(), x.dtype(), x.device())?)?;
            if x_sel.dim(0)? == 0 {
                continue;
            }
            let expert_out = expert.forward(&x_sel)?;
            y = masked_fill(&y, &mask.broadcast_as(y.shape())?, &expert_out)?;
        }
        let y_reshaped = y.reshape((topk_weight.dims()[0], topk_weight.dims()[1], self.num_activated_experts, y.dims()[2]))?;
        let y_sum = topk_weight.unsqueeze(2)?.matmul(&y_reshaped)?.squeeze(2)?;
        (y_sum + y_shared)
    }
}

// HDAttention
#[derive(Debug, Clone)]
struct HDAttention {
    to_q: Linear,
    to_k: Linear,
    to_v: Linear,
    to_out: Linear,
    to_q_t: Linear,
    to_k_t: Linear,
    to_v_t: Linear,
    to_out_t: Linear,
    q_rms_norm: RmsNorm,
    k_rms_norm: RmsNorm,
    q_rms_norm_t: RmsNorm,
    k_rms_norm_t: RmsNorm,
    heads: usize,
    dim_head: usize,
    single: bool,
}

impl HDAttention {
    fn new(
        query_dim: usize,
        heads: usize,
        dim_head: usize,
        single: bool,
        vb: VarBuilder,
    ) -> Result<Self> {
        let inner_dim = heads * dim_head;

        let to_q = linear(query_dim, inner_dim, vb.pp("to_q"))?;
        let to_k = linear(inner_dim, inner_dim, vb.pp("to_k"))?;
        let to_v = linear(inner_dim, inner_dim, vb.pp("to_v"))?;
        let to_out = linear(inner_dim, query_dim, vb.pp("to_out"))?;

        let q_rms_norm = candle_nn::rms_norm(dim_head, 1e-5, vb.pp("q_rms_norm"))?;
        let k_rms_norm = candle_nn::rms_norm(dim_head, 1e-5, vb.pp("k_rms_norm"))?;

        let to_q_t = linear(query_dim, inner_dim, vb.pp("to_q_t"))?;
        let to_k_t = linear(inner_dim, inner_dim, vb.pp("to_k_t"))?;
        let to_v_t = linear(inner_dim, inner_dim, vb.pp("to_v_t"))?;
        let to_out_t = linear(inner_dim, query_dim, vb.pp("to_out_t"))?;

        let q_rms_norm_t = candle_nn::rms_norm(dim_head, 1e-5, vb.pp("q_rms_norm_t"))?;
        let k_rms_norm_t = candle_nn::rms_norm(dim_head, 1e-5, vb.pp("k_rms_norm_t"))?;

        Ok(Self {
            to_q, to_k, to_v, to_out,
            to_q_t, to_k_t, to_v_t, to_out_t,
            q_rms_norm, k_rms_norm,
            q_rms_norm_t, k_rms_norm_t,
            heads, dim_head, single,
        })
    }

    fn forward_dual(&self, img: &Tensor, txt: &Tensor, pe: &Tensor) -> Result<(Tensor, Tensor)> {
        // Image stream
        let q_i = self.to_q.forward(img)?;
        let k_i = self.to_k.forward(img)?;
        let v_i = self.to_v.forward(img)?;

        // Text stream  
        let q_t = self.to_q_t.forward(txt)?;
        let k_t = self.to_k_t.forward(txt)?;
        let v_t = self.to_v_t.forward(txt)?;

        // Reshape for multi-head attention
        let (b, seq_i, _) = img.dims3()?;
        let (_, seq_t, _) = txt.dims3()?;
        
        let q_i = q_i.reshape((b, seq_i, self.heads, self.dim_head))?;
        let k_i = k_i.reshape((b, seq_i, self.heads, self.dim_head))?;
        let v_i = v_i.reshape((b, seq_i, self.heads, self.dim_head))?;
        
        let q_t = q_t.reshape((b, seq_t, self.heads, self.dim_head))?;
        let k_t = k_t.reshape((b, seq_t, self.heads, self.dim_head))?;
        let v_t = v_t.reshape((b, seq_t, self.heads, self.dim_head))?;

        // Apply RMS norm
        let q_i = self.q_rms_norm.forward(&q_i)?;
        let k_i = self.k_rms_norm.forward(&k_i)?;
        let q_t = self.q_rms_norm_t.forward(&q_t)?;
        let k_t = self.k_rms_norm_t.forward(&k_t)?;

        // Concatenate for joint attention
        let q = Tensor::cat(&[q_i, q_t], 1)?;
        let k = Tensor::cat(&[k_i, k_t], 1)?;
        let v = Tensor::cat(&[v_i, v_t], 1)?;

        // Apply attention with positional encoding
        let attn_out = attention(&q, &k, &v, pe)?;
        
        // Split back to image and text streams
        let img_out = attn_out.narrow(1, 0, seq_i)?;
        let txt_out = attn_out.narrow(1, seq_i, seq_t)?;

        // Apply output projections
        let img_out = img_out.reshape((b, seq_i, self.heads * self.dim_head))?;
        let txt_out = txt_out.reshape((b, seq_t, self.heads * self.dim_head))?;
        
        let img_out = self.to_out.forward(&img_out)?;
        let txt_out = self.to_out_t.forward(&txt_out)?;

        Ok((img_out, txt_out))
    }

    fn forward_single(&self, x: &Tensor, pe: &Tensor) -> Result<Tensor> {
        let q = self.to_q.forward(x)?;
        let k = self.to_k.forward(x)?;
        let v = self.to_v.forward(x)?;
        
        let (b, seq, _) = x.dims3()?;
        let q = q.reshape((b, seq, self.heads, self.dim_head))?;
        let k = k.reshape((b, seq, self.heads, self.dim_head))?;
        let v = v.reshape((b, seq, self.heads, self.dim_head))?;

        let q = self.q_rms_norm.forward(&q)?;
        let k = self.k_rms_norm.forward(&k)?;

        let attn_out = attention(&q, &k, &v, pe)?;
        let attn_out = attn_out.reshape((b, seq, self.heads * self.dim_head))?;
        self.to_out.forward(&attn_out)
    }
}

impl Module for HDAttention {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Default single stream forward
        let pe = Tensor::zeros((x.dim(0)?, x.dim(1)?, self.heads, self.dim_head), x.dtype(), x.device())?;
        self.forward_single(x, &pe)
    }
}

// HDBlockDouble
#[derive(Debug, Clone)]
struct HDBlockDouble {
    ada_ln_modulation: Linear,
    norm1_i: LayerNorm,
    norm1_t: LayerNorm,
    attn1: HDAttention,
    norm3_i: LayerNorm,
    ff_i: HDMOEFeedForwardSwiGLU,
    norm3_t: LayerNorm,
    ff_t: HDFeedForwardSwiGLU,
}

impl HDBlockDouble {
    fn new(
        dim: usize,
        heads: usize,
        head_dim: usize,
        num_routed_experts: usize,
        num_activated_experts: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let ada_ln_modulation = linear(dim, 12 * dim, vb.pp("adaLN_modulation"))?;
        let norm1_i = layer_norm(dim, 1e-6, vb.pp("norm1_i"))?;
        let norm1_t = layer_norm(dim, 1e-6, vb.pp("norm1_t"))?;
        let attn1 = HDAttention::new(dim, heads, head_dim, false, vb.pp("attn1"))?;
        let norm3_i = layer_norm(dim, 1e-6, vb.pp("norm3_i"))?;
        let ff_i = HDMOEFeedForwardSwiGLU::new(dim, 4 * dim, num_routed_experts, num_activated_experts, vb.pp("ff_i"))?;
        let norm3_t = layer_norm(dim, 1e-6, vb.pp("norm3_t"))?;
        let ff_t = HDFeedForwardSwiGLU::new(dim, 4 * dim, vb.pp("ff_t"))?;
        Ok(Self { ada_ln_modulation, norm1_i, norm1_t, attn1, norm3_i, ff_i, norm3_t, ff_t })
    }
}

impl HDBlockDouble {
    fn forward_dual(&self, img: &Tensor, txt: &Tensor, vec: &Tensor, pe: &Tensor) -> Result<(Tensor, Tensor)> {
        // AdaLN modulation
        let modulation = vec.silu()?.apply(&self.ada_ln_modulation)?;
        let chunks = modulation.chunk(12, D::Minus1)?;
        
        let (shift_msa_i, scale_msa_i, gate_msa_i) = (&chunks[0], &chunks[1], &chunks[2]);
        let (shift_msa_t, scale_msa_t, gate_msa_t) = (&chunks[3], &chunks[4], &chunks[5]);
        let (shift_mlp_i, scale_mlp_i, gate_mlp_i) = (&chunks[6], &chunks[7], &chunks[8]);
        let (shift_mlp_t, scale_mlp_t, gate_mlp_t) = (&chunks[9], &chunks[10], &chunks[11]);

        // Attention block
        let norm_img = self.norm1_i.forward(img)?;
        let norm_txt = self.norm1_t.forward(txt)?;
        
        let norm_img = norm_img.broadcast_mul(&(scale_msa_i.unsqueeze(1)? + 1.0)?)?
            .broadcast_add(&shift_msa_i.unsqueeze(1)?)?;
        let norm_txt = norm_txt.broadcast_mul(&(scale_msa_t.unsqueeze(1)? + 1.0)?)?
            .broadcast_add(&shift_msa_t.unsqueeze(1)?)?;

        let (attn_img, attn_txt) = self.attn1.forward_dual(&norm_img, &norm_txt, pe)?;
        
        let img = (img + gate_msa_i.unsqueeze(1)?.broadcast_mul(&attn_img)?)?;
        let txt = (txt + gate_msa_t.unsqueeze(1)?.broadcast_mul(&attn_txt)?)?;

        // Feed forward block
        let norm_img = self.norm3_i.forward(&img)?;
        let norm_txt = self.norm3_t.forward(&txt)?;
        
        let norm_img = norm_img.broadcast_mul(&(scale_mlp_i.unsqueeze(1)? + 1.0)?)?
            .broadcast_add(&shift_mlp_i.unsqueeze(1)?)?;
        let norm_txt = norm_txt.broadcast_mul(&(scale_mlp_t.unsqueeze(1)? + 1.0)?)?
            .broadcast_add(&shift_mlp_t.unsqueeze(1)?)?;

        let ff_img = self.ff_i.forward(&norm_img)?;
        let ff_txt = self.ff_t.forward(&norm_txt)?;
        
        let img = (img + gate_mlp_i.unsqueeze(1)?.broadcast_mul(&ff_img)?)?;
        let txt = (txt + gate_mlp_t.unsqueeze(1)?.broadcast_mul(&ff_txt)?)?;

        Ok((img, txt))
    }
}

impl Module for HDBlockDouble {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // This is a placeholder - the actual forward needs dual inputs
        candle::bail!("HDBlockDouble requires dual inputs, use forward_dual instead")
    }
}

// HDBlockSingle
#[derive(Debug, Clone)]
struct HDBlockSingle {
    ada_ln_modulation: Linear,
    norm1_i: LayerNorm,
    attn1: HDAttention,
    norm3_i: LayerNorm,
    ff_i: HDMOEFeedForwardSwiGLU,
}

impl HDBlockSingle {
    fn new(
        dim: usize,
        heads: usize,
        head_dim: usize,
        num_routed_experts: usize,
        num_activated_experts: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let ada_ln_modulation = linear(dim, 6 * dim, vb.pp("adaLN_modulation"))?;
        let norm1_i = layer_norm(dim, 1e-6, vb.pp("norm1_i"))?;
        let attn1 = HDAttention::new(dim, heads, head_dim, true, vb.pp("attn1"))?;
        let norm3_i = layer_norm(dim, 1e-6, vb.pp("norm3_i"))?;
        let ff_i = HDMOEFeedForwardSwiGLU::new(dim, 4 * dim, num_routed_experts, num_activated_experts, vb.pp("ff_i"))?;
        Ok(Self { ada_ln_modulation, norm1_i, attn1, norm3_i, ff_i })
    }
}

impl HDBlockSingle {
    fn forward_with_vec(&self, x: &Tensor, vec: &Tensor, pe: &Tensor) -> Result<Tensor> {
        // AdaLN modulation
        let modulation = vec.silu()?.apply(&self.ada_ln_modulation)?;
        let chunks = modulation.chunk(6, D::Minus1)?;
        
        let (shift_msa, scale_msa, gate_msa) = (&chunks[0], &chunks[1], &chunks[2]);
        let (shift_mlp, scale_mlp, gate_mlp) = (&chunks[3], &chunks[4], &chunks[5]);

        // Attention block
        let norm_x = self.norm1_i.forward(x)?;
        let norm_x = norm_x.broadcast_mul(&(scale_msa.unsqueeze(1)? + 1.0)?)?
            .broadcast_add(&shift_msa.unsqueeze(1)?)?;

        let attn_out = self.attn1.forward_single(&norm_x, pe)?;
        let x = (x + gate_msa.unsqueeze(1)?.broadcast_mul(&attn_out)?)?;

        // Feed forward block
        let norm_x = self.norm3_i.forward(&x)?;
        let norm_x = norm_x.broadcast_mul(&(scale_mlp.unsqueeze(1)? + 1.0)?)?
            .broadcast_add(&shift_mlp.unsqueeze(1)?)?;

        let ff_out = self.ff_i.forward(&norm_x)?;
        let x = (x + gate_mlp.unsqueeze(1)?.broadcast_mul(&ff_out)?)?;

        Ok(x)
    }
}

impl Module for HDBlockSingle {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // This is a placeholder - the actual forward needs vec parameter
        candle::bail!("HDBlockSingle requires vec parameter, use forward_with_vec instead")
    }
}

// Config
#[derive(Debug, Clone)]
pub struct Config {
    patch_size: usize,
    in_channels: usize,
    out_channels: usize,
    num_layers: usize,
    num_single_layers: usize,
    attention_head_dim: usize,
    num_attention_heads: usize,
    text_emb_dim: usize,
    num_routed_experts: usize,
    num_activated_experts: usize,
    axes_dims_rope: (usize, usize),
    max_resolution: (usize, usize),
    llama_layers: Vec<usize>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            patch_size: 2,
            in_channels: 64,
            out_channels: 64,
            num_layers: 16,
            num_single_layers: 32,
            attention_head_dim: 128,
            num_attention_heads: 20,
            text_emb_dim: 2048,
            num_routed_experts: 4,
            num_activated_experts: 2,
            axes_dims_rope: (32, 32),
            max_resolution: (128, 128),
            llama_layers: vec![],
        }
    }
}

// HDModel
#[derive(Debug, Clone)]
pub struct HDModel {
    t_embedder: TimestepEmbed,
    p_embedder: PooledEmbed,
    x_embedder: PatchEmbed,
    pe_embedder: EmbedNd,
    double_stream_blocks: Vec<HDBlockDouble>,
    single_stream_blocks: Vec<HDBlockSingle>,
    final_layer: HDLastLayer,
    caption_projection: Vec<TextProjection>,
    max_seq: usize,
}

impl HDModel {
    pub fn new(config: &Config, vb: VarBuilder) -> Result<Self> {
        let inner_dim = config.num_attention_heads * config.attention_head_dim;
        let t_embedder = TimestepEmbed::new(inner_dim, 256, vb.pp("t_embedder"))?;
        let p_embedder = PooledEmbed::new(config.text_emb_dim, inner_dim, vb.pp("p_embedder"))?;
        let x_embedder = PatchEmbed::new(config.patch_size, config.in_channels, inner_dim, vb.pp("x_embedder"))?;
        let pe_embedder = EmbedNd::new(10000, vec![config.axes_dims_rope.0, config.axes_dims_rope.1]);
        let mut double_stream_blocks = Vec::new();
        for _ in 0..config.num_layers {
            double_stream_blocks.push(HDBlockDouble::new(inner_dim, config.num_attention_heads, config.attention_head_dim, config.num_routed_experts, config.num_activated_experts, vb.pp("double_stream_blocks"))?);
        }
        let mut single_stream_blocks = Vec::new();
        for _ in 0..config.num_single_layers {
            single_stream_blocks.push(HDBlockSingle::new(inner_dim, config.num_attention_heads, config.attention_head_dim, config.num_routed_experts, config.num_activated_experts, vb.pp("single_stream_blocks"))?);
        }
        let final_layer = HDLastLayer::new(inner_dim, config.patch_size, config.out_channels, vb.pp("final_layer"))?;
        let mut caption_projection = Vec::new();
        // Add caption projections as per Python
        let max_seq = config.max_resolution.0 * config.max_resolution.1 / (config.patch_size * config.patch_size);
        Ok(Self { t_embedder, p_embedder, x_embedder, pe_embedder, double_stream_blocks, single_stream_blocks, final_layer, caption_projection, max_seq })
    }
}

impl HDModel {
    pub fn forward_with_cfg(
        &self,
        hidden_states: &Tensor,
        timesteps: &Tensor,
        encoder_hidden_states: &[Tensor], // [t5_embeds, llama_embeds]
        pooled_embeds: &Tensor,
        img_sizes: Option<&Tensor>,
        img_ids: Option<&Tensor>,
    ) -> Result<Tensor> {
        let (batch_size, seq_len, _) = hidden_states.dims3()?;
        
        // Store original sequence length for later use
        let img_seq_len = seq_len;
        
        // Embed patches
        let embedded_states = self.x_embedder.forward(hidden_states)?;
        
        // Timestep embedding
        let timestep_emb = self.t_embedder.forward(timesteps)?;
        
        // Pooled embedding
        let pooled_emb = self.p_embedder.forward(pooled_embeds)?;
        
        // Combine timestep and pooled embeddings
        let vec = (timestep_emb + pooled_emb)?;
        
        // Positional encoding
        let pe = if let Some(ids) = img_ids {
            self.pe_embedder.forward(ids)?
        } else {
            // Default positional encoding for square images
            let h = (seq_len as f64).sqrt() as usize;
            let w = h;
            let mut ids = Vec::new();
            for i in 0..h {
                for j in 0..w {
                    ids.push([0.0, i as f32, j as f32]);
                }
            }
            let ids_tensor = Tensor::from_vec(
                ids.into_iter().flatten().collect::<Vec<f32>>(),
                (1, seq_len, 3),
                embedded_states.device(),
            )?;
            self.pe_embedder.forward(&ids_tensor)?
        };
        
        // Process text embeddings
        let t5_embeds = &encoder_hidden_states[0];
        let llama_embeds = if encoder_hidden_states.len() > 1 {
            &encoder_hidden_states[1]
        } else {
            t5_embeds // Fallback if llama embeds not provided
        };
        
        // Project text embeddings if needed
        let txt = t5_embeds.clone(); // Use T5 embeddings as primary text
        
        let mut img = embedded_states;
        let mut txt = txt;
        
        // Double stream blocks
        for block in &self.double_stream_blocks {
            let (new_img, new_txt) = block.forward_dual(&img, &txt, &vec, &pe)?;
            img = new_img;
            txt = new_txt;
        }
        
        // Concatenate for single stream
        let combined = Tensor::cat(&[img, txt], 1)?;
        let mut x = combined;
        
        // Single stream blocks
        for block in &self.single_stream_blocks {
            x = block.forward_with_vec(&x, &vec, &pe)?;
        }
        
        // Extract image part for final layer
        let final_img = x.narrow(1, 0, img_seq_len)?;
        
        // Final layer
        let output = self.final_layer.forward_with_vec(&final_img, &vec)?;
        
        Ok(output)
    }
}

impl Module for HDModel {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // This is a placeholder - the actual forward needs more parameters
        candle::bail!("HDModel requires additional parameters, use forward_with_cfg instead")
    }
}

// HDLastLayer
#[derive(Debug, Clone)]
struct HDLastLayer {
    norm_final: LayerNorm,
    linear: Linear,
    ada_ln_modulation: Linear,
}

impl HDLastLayer {
    fn new(hidden_size: usize, patch_size: usize, out_channels: usize, vb: VarBuilder) -> Result<Self> {
        let norm_final = layer_norm(hidden_size, 1e-6, vb.pp("norm_final"))?;
        let linear_layer = linear(hidden_size, patch_size * patch_size * out_channels, vb.pp("linear"))?;
        let ada_ln_modulation = linear(hidden_size, 2 * hidden_size, vb.pp("adaLN_modulation"))?;
        Ok(Self { norm_final, linear: linear_layer, ada_ln_modulation })
    }
}

impl HDLastLayer {
    fn forward_with_vec(&self, x: &Tensor, vec: &Tensor) -> Result<Tensor> {
        let chunks = vec.silu()?.apply(&self.ada_ln_modulation)?.chunk(2, D::Minus1)?;
        let (shift, scale) = (chunks[0].unsqueeze(1)?, chunks[1].unsqueeze(1)?);
        let x = x
            .apply(&self.norm_final)?
            .broadcast_mul(&(scale + 1.0)?)?
            .broadcast_add(&shift)?;
        x.apply(&self.linear)
    }
}

impl Module for HDLastLayer {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // This is a placeholder - the actual forward needs vec parameter
        // Should be called via forward_with_vec
        candle::bail!("HDLastLayer requires vec parameter, use forward_with_vec instead")
    }
}

// Rope and attention from Flux
fn rope(pos: &Tensor, dim: usize, theta: usize) -> Result<Tensor> {
    if dim % 2 == 1 {
        candle::bail!("dim {dim} is odd")
    }
    let dev = pos.device();
    let theta = theta as f64;
    let inv_freq: Vec<f32> = (0..dim)
        .step_by(2)
        .map(|i| 1.0 / theta.powf(i as f64 / dim as f64) as f32)
        .collect();
    let inv_freq_len = inv_freq.len();
    let inv_freq = Tensor::from_vec(inv_freq, (1, inv_freq_len), dev)?.to_dtype(pos.dtype())?;
    let freqs = pos.unsqueeze(1)?.broadcast_mul(&inv_freq)?;
    let cos = freqs.cos()?;
    let sin = freqs.sin()?;
    Tensor::cat(&[cos, sin], D::Minus1)
}

// Add other functions as needed.

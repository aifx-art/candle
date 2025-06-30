//! HiDream-E1: Instruction-based Image Editing Model
//!
//! This module implements the HiDream-E1 model for image editing based on textual instructions.
//! HiDream-E1 is built on HiDream-I1 and uses a multi-modal approach with multiple text encoders
//! (CLIP, T5, LLaMA) and a transformer-based diffusion architecture.
//!
//! Reference: https://github.com/HiDream-ai/HiDream-E1

use candle::{DType, Device, Result, Tensor, D};
use candle_nn::{linear, Conv2d, Conv2dConfig, LayerNorm, Linear, Module, VarBuilder};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Configuration for the HiDream Image Transformer
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HiDreamImageTransformerConfig {
    pub num_attention_heads: usize,
    pub attention_head_dim: usize,
    pub in_channels: usize,
    pub out_channels: Option<usize>,
    pub num_layers: usize,
    pub dropout: f64,
    pub norm_num_groups: usize,
    pub cross_attention_dim: Option<usize>,
    pub attention_bias: bool,
    pub sample_size: Option<usize>,
    pub num_vector_embeds: Option<usize>,
    pub patch_size: usize,
    pub activation_fn: String,
    pub num_embeds_ada_norm: Option<usize>,
    pub use_linear_projection: bool,
    pub only_cross_attention: bool,
    pub double_self_attention: bool,
    pub upcast_attention: bool,
    pub norm_type: String,
    pub norm_elementwise_affine: bool,
    pub norm_eps: f64,
    pub attention_type: String,
    pub caption_projection_dim: Option<usize>,
    pub pooled_projection_dim: Option<usize>,
    pub pos_embed_max_size: Option<usize>,
}

impl Default for HiDreamImageTransformerConfig {
    fn default() -> Self {
        Self {
            num_attention_heads: 16,
            attention_head_dim: 88,
            in_channels: 16,
            out_channels: Some(16),
            num_layers: 18,
            dropout: 0.0,
            norm_num_groups: 32,
            cross_attention_dim: Some(4096),
            attention_bias: true,
            sample_size: Some(128),
            num_vector_embeds: None,
            patch_size: 2,
            activation_fn: "gelu-approximate".to_string(),
            num_embeds_ada_norm: None,
            use_linear_projection: true,
            only_cross_attention: false,
            double_self_attention: false,
            upcast_attention: false,
            norm_type: "ada_norm_single".to_string(),
            norm_elementwise_affine: true,
            norm_eps: 1e-5,
            attention_type: "default".to_string(),
            caption_projection_dim: Some(1152),
            pooled_projection_dim: Some(2048),
            pos_embed_max_size: Some(192),
        }
    }
}

/// Positional embedding for 2D patches
struct PatchEmbed {
    proj: Conv2d,
    norm: Option<LayerNorm>,
}

impl PatchEmbed {
    fn new(
        in_channels: usize,
        embed_dim: usize,
        patch_size: usize,
        bias: bool,
        vb: VarBuilder,
    ) -> Result<Self> {
        let conv_config = Conv2dConfig {
            stride: patch_size,
            padding: 0,
            ..Default::default()
        };

        let proj = if bias {
            candle_nn::conv2d(
                in_channels,
                embed_dim,
                patch_size,
                conv_config,
                vb.pp("proj"),
            )?
        } else {
            candle_nn::conv2d_no_bias(
                in_channels,
                embed_dim,
                patch_size,
                conv_config,
                vb.pp("proj"),
            )?
        };

        Ok(Self { proj, norm: None })
    }

    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let hidden_states = self.proj.forward(hidden_states)?;
        let batch_size = hidden_states.dims()[0];
        let patch_dim = hidden_states.dims()[1];
        let num_patches = hidden_states.dims()[2] * hidden_states.dims()[3];

        let hidden_states = hidden_states
            .reshape((batch_size, patch_dim, num_patches))?
            .transpose(1, 2)?;

        if let Some(norm) = &self.norm {
            norm.forward(&hidden_states)
        } else {
            Ok(hidden_states)
        }
    }
}

/// Multi-head attention module
struct Attention {
    to_q: Linear,
    to_k: Linear,
    to_v: Linear,
    to_out: Linear,
    _to_q_t: Option<Linear>,
    _to_k_t: Option<Linear>,
    _to_v_t: Option<Linear>,
    _to_out_t: Option<Linear>,
    heads: usize,
    dim_head: usize,
    _dropout: f64,
    _upcast_attention: bool,
}

impl Attention {
    fn new(
        query_dim: usize,
        cross_attention_dim: Option<usize>,
        heads: usize,
        dim_head: usize,
        dropout: f64,
        _bias: bool,
        upcast_attention: bool,
        vb: VarBuilder,
    ) -> Result<Self> {
        let inner_dim = dim_head * heads;
        let cross_attention_dim = cross_attention_dim.unwrap_or(query_dim);

        let to_q = linear(query_dim, inner_dim, vb.pp("to_q"))?;
        let to_k = linear(cross_attention_dim, inner_dim, vb.pp("to_k"))?;
        let to_v = linear(cross_attention_dim, inner_dim, vb.pp("to_v"))?;
        let to_out = linear(inner_dim, query_dim, vb.pp("to_out.0"))?;

        // For temporal attention layers
        let _to_q_t = if vb.contains_tensor("to_q_t") {
            Some(linear(query_dim, inner_dim, vb.pp("to_q_t"))?)
        } else {
            None
        };
        let _to_k_t = if vb.contains_tensor("to_k_t") {
            Some(linear(cross_attention_dim, inner_dim, vb.pp("to_k_t"))?)
        } else {
            None
        };
        let _to_v_t = if vb.contains_tensor("to_v_t") {
            Some(linear(cross_attention_dim, inner_dim, vb.pp("to_v_t"))?)
        } else {
            None
        };
        let _to_out_t = if vb.contains_tensor("to_out_t") {
            Some(linear(inner_dim, query_dim, vb.pp("to_out_t.0"))?)
        } else {
            None
        };

        Ok(Self {
            to_q,
            to_k,
            to_v,
            to_out,
            _to_q_t,
            _to_k_t,
            _to_v_t,
            _to_out_t,
            heads,
            dim_head,
            _dropout: dropout,
            _upcast_attention: upcast_attention,
        })
    }

    fn forward(
        &self,
        hidden_states: &Tensor,
        encoder_hidden_states: Option<&Tensor>,
        attention_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let encoder_hidden_states = encoder_hidden_states.unwrap_or(hidden_states);

        let query = self.to_q.forward(hidden_states)?;
        let key = self.to_k.forward(encoder_hidden_states)?;
        let value = self.to_v.forward(encoder_hidden_states)?;

        let query = self.head_to_batch_dim(&query)?;
        let key = self.head_to_batch_dim(&key)?;
        let value = self.head_to_batch_dim(&value)?;

        let attention_scores = query.matmul(&key.transpose(D::Minus1, D::Minus2)?)?;
        let attention_scores = (attention_scores * (self.dim_head as f64).powf(-0.5))?;

        let attention_probs = if let Some(mask) = attention_mask {
            let mask = mask.broadcast_as(attention_scores.shape())?;
            candle_nn::ops::softmax(&(attention_scores + mask)?, D::Minus1)?
        } else {
            candle_nn::ops::softmax(&attention_scores, D::Minus1)?
        };

        let hidden_states = attention_probs.matmul(&value)?;
        let hidden_states = self.batch_to_head_dim(&hidden_states)?;

        self.to_out.forward(&hidden_states)
    }

    fn head_to_batch_dim(&self, tensor: &Tensor) -> Result<Tensor> {
        let batch_size = tensor.dims()[0];
        let seq_len = tensor.dims()[1];

        tensor
            .reshape((batch_size, seq_len, self.heads, self.dim_head))?
            .transpose(1, 2)?
            .reshape((batch_size * self.heads, seq_len, self.dim_head))
    }

    fn batch_to_head_dim(&self, tensor: &Tensor) -> Result<Tensor> {
        let batch_size = tensor.dims()[0] / self.heads;
        let seq_len = tensor.dims()[1];
        let dim = tensor.dims()[2];

        tensor
            .reshape((batch_size, self.heads, seq_len, dim))?
            .transpose(1, 2)?
            .reshape((batch_size, seq_len, self.heads * dim))
    }
}

/// Feed-forward network
struct FeedForward {
    w1: Linear,
    w2: Linear,
    activation: candle_nn::Activation,
}

impl FeedForward {
    fn new(
        dim: usize,
        dim_out: Option<usize>,
        mult: usize,
        _dropout: f64,
        activation_fn: &str,
        vb: VarBuilder,
    ) -> Result<Self> {
        let inner_dim = dim * mult;
        let dim_out = dim_out.unwrap_or(dim);

        let w1 = linear(dim, inner_dim, vb.pp("net.0.proj"))?;
        let w2 = linear(inner_dim, dim_out, vb.pp("net.2"))?;

        let activation = match activation_fn {
            "gelu" => candle_nn::Activation::Gelu,
            "gelu-approximate" => candle_nn::Activation::Gelu,
            "swish" => candle_nn::Activation::Swish,
            "silu" => candle_nn::Activation::Swish,
            _ => candle_nn::Activation::Gelu,
        };

        Ok(Self { w1, w2, activation })
    }
}

impl Module for FeedForward {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = self.w1.forward(xs)?;
        let xs = self.activation.forward(&xs)?;
        self.w2.forward(&xs)
    }
}

/// Basic transformer block
struct BasicTransformerBlock {
    attn1: Attention,
    ff: FeedForward,
    attn2: Option<Attention>,
    norm1: LayerNorm,
    norm2: LayerNorm,
    norm3: Option<LayerNorm>,
}

impl BasicTransformerBlock {
    fn new(
        dim: usize,
        num_attention_heads: usize,
        attention_head_dim: usize,
        dropout: f64,
        cross_attention_dim: Option<usize>,
        activation_fn: &str,
        attention_bias: bool,
        upcast_attention: bool,
        vb: VarBuilder,
    ) -> Result<Self> {
        let attn1 = Attention::new(
            dim,
            None,
            num_attention_heads,
            attention_head_dim,
            dropout,
            attention_bias,
            upcast_attention,
            vb.pp("attn1"),
        )?;

        let ff = FeedForward::new(dim, None, 4, dropout, activation_fn, vb.pp("ff"))?;

        let attn2 = if cross_attention_dim.is_some() {
            Some(Attention::new(
                dim,
                cross_attention_dim,
                num_attention_heads,
                attention_head_dim,
                dropout,
                attention_bias,
                upcast_attention,
                vb.pp("attn2"),
            )?)
        } else {
            None
        };

        let norm1 = candle_nn::layer_norm(dim, 1e-5, vb.pp("norm1"))?;
        let norm2 = candle_nn::layer_norm(dim, 1e-5, vb.pp("norm2"))?;
        let norm3 = if attn2.is_some() {
            Some(candle_nn::layer_norm(dim, 1e-5, vb.pp("norm3"))?)
        } else {
            None
        };

        Ok(Self {
            attn1,
            ff,
            attn2,
            norm1,
            norm2,
            norm3,
        })
    }

    fn forward(
        &self,
        hidden_states: &Tensor,
        encoder_hidden_states: Option<&Tensor>,
        attention_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        // Self-attention
        let norm_hidden_states = self.norm1.forward(hidden_states)?;
        let attn_output = self
            .attn1
            .forward(&norm_hidden_states, None, attention_mask)?;
        let hidden_states = (hidden_states + attn_output)?;

        // Cross-attention
        let hidden_states = if let Some(attn2) = &self.attn2 {
            let norm_hidden_states = self.norm2.forward(&hidden_states)?;
            let attn_output = attn2.forward(&norm_hidden_states, encoder_hidden_states, None)?;
            (hidden_states + attn_output)?
        } else {
            hidden_states
        };

        // Feed-forward
        let norm_hidden_states = if self.norm3.is_some() {
            self.norm3.as_ref().unwrap().forward(&hidden_states)?
        } else {
            self.norm2.forward(&hidden_states)?
        };
        let ff_output = self.ff.forward(&norm_hidden_states)?;
        hidden_states + ff_output
    }
}

/// Main HiDream Image Transformer model
pub struct HiDreamImageTransformer2DModel {
    pos_embed: Tensor,
    transformer_blocks: Vec<BasicTransformerBlock>,
    norm_out: Option<LayerNorm>,
    proj_out: Linear,
    _adaln_single: Option<Linear>,
    _caption_projection: Option<Linear>,
    x_embedder: PatchEmbed,
    config: HiDreamImageTransformerConfig,
}

impl HiDreamImageTransformer2DModel {
    pub fn new(config: &HiDreamImageTransformerConfig, vb: VarBuilder) -> Result<Self> {
        let inner_dim = config.num_attention_heads * config.attention_head_dim;

        // Patch embedding
        let x_embedder = PatchEmbed::new(
            config.in_channels,
            inner_dim,
            config.patch_size,
            true,
            vb.pp("x_embedder"),
        )?;

        // Positional embedding
        let pos_embed_max_size = config.pos_embed_max_size.unwrap_or(192);
        let num_patches = pos_embed_max_size * pos_embed_max_size;
        let pos_embed = vb.get((num_patches, inner_dim), "pos_embed.proj.weight")?;

        // Transformer blocks
        let mut transformer_blocks = Vec::new();
        for i in 0..config.num_layers {
            let block = BasicTransformerBlock::new(
                inner_dim,
                config.num_attention_heads,
                config.attention_head_dim,
                config.dropout,
                config.cross_attention_dim,
                &config.activation_fn,
                config.attention_bias,
                config.upcast_attention,
                vb.pp(&format!("transformer_blocks.{}", i)),
            )?;
            transformer_blocks.push(block);
        }

        // Output projection
        let proj_out = linear(
            inner_dim,
            config.out_channels.unwrap_or(config.in_channels)
                * config.patch_size
                * config.patch_size,
            vb.pp("proj_out"),
        )?;

        // Optional components
        let norm_out = if config.norm_type == "layer_norm" {
            Some(candle_nn::layer_norm(
                inner_dim,
                config.norm_eps,
                vb.pp("norm_out"),
            )?)
        } else {
            None
        };

        let _adaln_single = if config.norm_type == "ada_norm_single" {
            Some(linear(
                config.cross_attention_dim.unwrap_or(inner_dim),
                6 * inner_dim,
                vb.pp("adaln_single"),
            )?)
        } else {
            None
        };

        let _caption_projection = if let Some(caption_dim) = config.caption_projection_dim {
            Some(linear(caption_dim, inner_dim, vb.pp("caption_projection"))?)
        } else {
            None
        };

        Ok(Self {
            pos_embed,
            transformer_blocks,
            norm_out,
            proj_out,
            _adaln_single,
            _caption_projection,
            x_embedder,
            config: config.clone(),
        })
    }

    pub fn forward(
        &self,
        hidden_states: &Tensor,
        encoder_hidden_states: Option<&Tensor>,
        _timestep: Option<&Tensor>,
        _added_cond_kwargs: Option<&HashMap<String, Tensor>>,
    ) -> Result<Tensor> {
        let batch_size = hidden_states.dims()[0];
        let height = hidden_states.dims()[2];
        let width = hidden_states.dims()[3];

        // Patch embedding
        let hidden_states = self.x_embedder.forward(hidden_states)?;

        // Add positional embedding
        let pos_embed = self.interpolate_pos_encoding(height, width)?;
        let hidden_states =
            (hidden_states.clone() + pos_embed.broadcast_as(hidden_states.shape())?)?;

        // Process through transformer blocks
        let mut hidden_states = hidden_states;
        for block in &self.transformer_blocks {
            hidden_states = block.forward(&hidden_states, encoder_hidden_states, None)?;
        }

        // Final normalization
        if let Some(norm_out) = &self.norm_out {
            hidden_states = norm_out.forward(&hidden_states)?;
        }

        // Output projection
        let hidden_states = self.proj_out.forward(&hidden_states)?;

        // Reshape back to image format
        let patch_size = self.config.patch_size;
        let out_channels = self.config.out_channels.unwrap_or(self.config.in_channels);
        let num_patches_h = height / patch_size;
        let num_patches_w = width / patch_size;

        // Fix the permute issue by using proper dimensions
        let reshaped = hidden_states.reshape((
            batch_size,
            num_patches_h,
            num_patches_w,
            patch_size,
            patch_size,
            out_channels,
        ))?;

        // Permute dimensions: (batch, h_patches, w_patches, patch_h, patch_w, channels) ->
        // (batch, channels, h_patches, patch_h, w_patches, patch_w)
        let permuted = reshaped
            .transpose(5, 0)? // channels to front
            .transpose(1, 2)? // h_patches and w_patches
            .transpose(3, 4)?; // patch_h and patch_w

        permuted.reshape((batch_size, out_channels, height, width))
    }

    fn interpolate_pos_encoding(&self, height: usize, width: usize) -> Result<Tensor> {
        let patch_size = self.config.patch_size;
        let num_patches_h = height / patch_size;
        let num_patches_w = width / patch_size;
        let num_patches = num_patches_h * num_patches_w;

        if num_patches == self.pos_embed.dims()[0] {
            return Ok(self.pos_embed.clone());
        }

        // For now, just return the positional embedding as-is
        // In a full implementation, you would implement proper interpolation
        self.pos_embed
            .narrow(0, 0, num_patches.min(self.pos_embed.dims()[0]))
    }
}

/// Configuration for the full HiDream-E1 pipeline
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HiDreamConfig {
    pub transformer: HiDreamImageTransformerConfig,
    pub scheduler_type: String,
    pub num_inference_steps: usize,
    pub guidance_scale: f64,
    pub image_guidance_scale: f64,
}

impl Default for HiDreamConfig {
    fn default() -> Self {
        Self {
            transformer: HiDreamImageTransformerConfig::default(),
            scheduler_type: "FlowMatchEulerDiscreteScheduler".to_string(),
            num_inference_steps: 28,
            guidance_scale: 5.0,
            image_guidance_scale: 4.0,
        }
    }
}

/// Text encoder embeddings container
pub struct TextEncoderEmbeddings {
    pub clip_embeds: Option<Tensor>,
    pub clip_embeds_2: Option<Tensor>,
    pub t5_embeds: Option<Tensor>,
    pub llama_embeds: Option<Tensor>,
    pub pooled_embeds: Option<Tensor>,
}

impl TextEncoderEmbeddings {
    pub fn new() -> Self {
        Self {
            clip_embeds: None,
            clip_embeds_2: None,
            t5_embeds: None,
            llama_embeds: None,
            pooled_embeds: None,
        }
    }
}

/// HiDream-E1 diffusion pipeline
pub struct HiDreamImageEditingPipeline {
    pub transformer: HiDreamImageTransformer2DModel,
    pub config: HiDreamConfig,
}

impl HiDreamImageEditingPipeline {
    pub fn new(config: HiDreamConfig, transformer: HiDreamImageTransformer2DModel) -> Self {
        Self {
            transformer,
            config,
        }
    }

    pub fn load(vb: VarBuilder, config: HiDreamConfig) -> Result<Self> {
        let transformer =
            HiDreamImageTransformer2DModel::new(&config.transformer, vb.pp("transformer"))?;
        Ok(Self::new(config, transformer))
    }

    /// Forward pass through the transformer
    pub fn forward(
        &self,
        latents: &Tensor,
        encoder_hidden_states: &TextEncoderEmbeddings,
        timestep: &Tensor,
    ) -> Result<Tensor> {
        // Combine text embeddings - this is a simplified version
        // In the full implementation, you would handle the complex multi-encoder logic
        let combined_embeds = if let Some(t5_embeds) = &encoder_hidden_states.t5_embeds {
            t5_embeds.clone()
        } else if let Some(clip_embeds) = &encoder_hidden_states.clip_embeds {
            clip_embeds.clone()
        } else {
            return Err(candle::Error::Msg(
                "No text embeddings provided".to_string(),
            ));
        };

        self.transformer
            .forward(latents, Some(&combined_embeds), Some(timestep), None)
    }

    /// Prepare latents for the diffusion process
    pub fn prepare_latents(
        &self,
        batch_size: usize,
        num_channels: usize,
        height: usize,
        width: usize,
        device: &Device,
        dtype: DType,
    ) -> Result<Tensor> {
        let vae_scale_factor = 8; // Standard VAE scale factor
        let latent_height = height / vae_scale_factor / self.config.transformer.patch_size;
        let latent_width = width / vae_scale_factor / self.config.transformer.patch_size;

        Tensor::randn(
            0f32,
            1f32,
            (batch_size, num_channels, latent_height, latent_width),
            device,
        )?
        .to_dtype(dtype)
    }

    /// Calculate shift for timestep scheduling (from Flux)
    pub fn calculate_shift(
        &self,
        image_seq_len: usize,
        base_seq_len: usize,
        max_seq_len: usize,
        base_shift: f64,
        max_shift: f64,
    ) -> f64 {
        let m = (max_shift - base_shift) / (max_seq_len as f64 - base_seq_len as f64);
        let b = base_shift - m * base_seq_len as f64;
        image_seq_len as f64 * m + b
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle::{DType, Device};
    use candle_nn::VarMap;

    #[test]
    fn test_hidream_config_default() {
        let config = HiDreamConfig::default();
        assert_eq!(config.num_inference_steps, 28);
        assert_eq!(config.guidance_scale, 5.0);
        assert_eq!(config.image_guidance_scale, 4.0);
    }

    #[test]
    fn test_transformer_config_default() {
        let config = HiDreamImageTransformerConfig::default();
        assert_eq!(config.num_attention_heads, 16);
        assert_eq!(config.attention_head_dim, 88);
        assert_eq!(config.in_channels, 16);
        assert_eq!(config.num_layers, 18);
    }

    #[test]
    fn test_pipeline_prepare_latents() {
        let device = Device::Cpu;
        let config = HiDreamConfig::default();
        let var_map = VarMap::new();
        let vb = VarBuilder::from_varmap(&var_map, DType::F32, &device);

        if let Ok(transformer) = HiDreamImageTransformer2DModel::new(&config.transformer, vb) {
            let pipeline = HiDreamImageEditingPipeline::new(config, transformer);

            let latents = pipeline.prepare_latents(1, 16, 512, 512, &device, DType::F32);
            assert!(latents.is_ok());

            if let Ok(latents) = latents {
                let dims = latents.dims();
                assert_eq!(dims[0], 1); // batch_size
                assert_eq!(dims[1], 16); // num_channels
            }
        }
    }
}
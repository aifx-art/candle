use candle::{DType, IndexOp, Result, Tensor, D};
use candle_nn::{LayerNorm, Linear, RmsNorm, VarBuilder};



// hidream model https://github.com/HiDream-ai/HiDream-I1/blob/f418a4b13bc548b177aced3880b7294d04594651/hi_diffusers/models/transformers/transformer_hidream_image.py#L230
// comfy impl  https://github.com/comfyanonymous/ComfyUI/blob/2c1d686ec61f26f3a64bb4c1afdcdb78bb943a4f/comfy/ldm/hidream/model.py#L562
// res4lyf impl https://github.com/ClownsharkBatwing/RES4LYF/blob/5c94acb9b1a03c1d17241281e6065644139f7ce6/hidream/model.py#L414

#[derive(Debug, Clone)]
pub struct Config {
    pub patch_size: Option<usize>,
    pub in_channels: usize,
    pub out_channels: usize,
    pub num_layers: usize,
    pub num_single_layers: usize,
    pub attention_head_dim: usize,
    pub num_attention_heads: usize,
    pub caption_channels: Vec<usize>,
    pub text_emb_dim: usize,
    pub num_routed_experts: usize,
    pub num_activated_experts: usize,
    pub axes_dims_rope: (usize, usize),
    pub max_resolution: (usize, usize),
    pub llama_layers: Vec<usize>,
}

impl Config {
    pub fn full() -> Self {
        Self {
            patch_size: None,
            in_channels: 64,
            out_channels: 64,
            num_layers: 16,
            num_single_layers: 32,
            attention_head_dim: 128,
            num_attention_heads: 20,
            caption_channels: vec![],
            text_emb_dim: 2048,
            num_routed_experts: 4,
            num_activated_experts: 2,
            axes_dims_rope: (32,32),
            max_resolution: (128,128),
            llama_layers: vec![],
        }
    }
}




#[derive(Debug, Clone)]
pub struct HiDream {
    timestep_embedder: todo!() 
}




impl HiDream {

    pub fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {

        let inner_dim = cfg.num_attention_heads * cfg.attention_head_dim;




        Ok(Self {
            timestep_embedder:todo!()
        })
    }
}
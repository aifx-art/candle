#[cfg(feature = "accelerate")]
extern crate accelerate_src;

#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

use candle_transformers::models::{clip, hidream, t5, flux};

use anyhow::{Error as E, Result};
use candle::{IndexOp, Module, Tensor, D};
use candle_nn::VarBuilder;
use clap::Parser;
use tokenizers::Tokenizer;
use std::collections::HashSet;

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// The prompt to be used for image generation.
    #[arg(
        long,
        default_value = "A cat holding a sign that says \"Hi-Dreams.ai\""
    )]
    prompt: String,

    /// Run on CPU rather than on GPU.
    #[arg(long)]
    cpu: bool,

    /// Enable tracing (generates a trace-timestamp.json file).
    #[arg(long)]
    tracing: bool,

    /// The height in pixels of the generated image.
    #[arg(long, default_value = "1024")]
    height: usize,

    /// The width in pixels of the generated image.
    #[arg(long, default_value = "1024")]
    width: usize,

    /// The model variant to use.
    #[arg(long, value_enum, default_value = "i1-full-fp8")]
    model: ModelVariant,

    /// The number of inference steps.
    #[arg(long)]
    num_inference_steps: Option<usize>,

    /// Guidance scale for classifier-free guidance.
    #[arg(long)]
    guidance_scale: Option<f64>,

    /// The seed to use when generating random samples.
    #[arg(long)]
    seed: Option<u64>,

    /// Input image path for editing (E1 model only).
    #[arg(long)]
    input_image: Option<String>,

    /// Image guidance scale for editing (E1 model only).
    #[arg(long, default_value = "4.0")]
    image_guidance_scale: f64,

    /// Negative prompt.
    #[arg(long, default_value = "low resolution, blur")]
    negative_prompt: String,

    /// Output filename.
    #[arg(long, default_value = "hidream_output.jpg")]
    output: String,
}

#[derive(Debug, Clone, Copy, clap::ValueEnum, PartialEq, Eq)]
enum ModelVariant {
    /// HiDream-I1-Full-FP8: High quality generation model (FP8, default)
    I1FullFp8,
    /// HiDream-I1-Dev-FP8: Fast generation model (FP8)
    I1DevFp8,
    /// HiDream-I1-Fast-FP8: Fastest generation model (FP8)
    I1FastFp8,
    /// HiDream-I1-Full-FP16: High quality generation model (FP16)
    I1FullFp16,
    /// HiDream-I1-Dev-BF16: Fast generation model (BF16)
    I1DevBf16,
    /// HiDream-I1-Fast-BF16: Fastest generation model (BF16)
    I1FastBf16,
    /// HiDream-E1-Full-BF16: Image editing model (BF16)
    E1FullBf16,
}

impl ModelVariant {
    fn model_id(&self) -> &'static str {
        // All models now use the Comfy-Org repository
        "Comfy-Org/HiDream-I1_ComfyUI"
    }

    fn model_filename(&self) -> &'static str {
        match self {
            ModelVariant::I1FullFp8 => "split_files/diffusion_models/hidream_i1_full_fp8.safetensors",
            ModelVariant::I1DevFp8 => "split_files/diffusion_models/hidream_i1_dev_fp8.safetensors",
            ModelVariant::I1FastFp8 => "split_files/diffusion_models/hidream_i1_fast_fp8.safetensors",
            ModelVariant::I1FullFp16 => "split_files/diffusion_models/hidream_i1_full_fp16.safetensors",
            ModelVariant::I1DevBf16 => "split_files/diffusion_models/hidream_i1_dev_bf16.safetensors",
            ModelVariant::I1FastBf16 => "split_files/diffusion_models/hidream_i1_fast_bf16.safetensors",
            ModelVariant::E1FullBf16 => "split_files/diffusion_models/hidream_e1_full_bf16.safetensors",
        }
    }

    fn default_steps(&self) -> usize {
        match self {
            ModelVariant::I1DevFp8 | ModelVariant::I1DevBf16 => 28,
            ModelVariant::I1FullFp8 | ModelVariant::I1FullFp16 => 50,
            ModelVariant::I1FastFp8 | ModelVariant::I1FastBf16 => 16,
            ModelVariant::E1FullBf16 => 28,
        }
    }

    fn default_guidance_scale(&self) -> f64 {
        match self {
            ModelVariant::I1DevFp8 | ModelVariant::I1DevBf16 => 0.0,
            ModelVariant::I1FullFp8 | ModelVariant::I1FullFp16 => 5.0,
            ModelVariant::I1FastFp8 | ModelVariant::I1FastBf16 => 0.0,
            ModelVariant::E1FullBf16 => 5.0,
        }
    }

    fn is_editing_model(&self) -> bool {
        matches!(self, ModelVariant::E1FullBf16)
    }
}

fn load_image(path: &str) -> Result<Tensor> {
    let img = image::open(path)?;
    let img = img.resize_exact(768, 768, image::imageops::FilterType::Lanczos3);
    let img = img.to_rgb8();
    let data = img.as_raw();
    let data = data.iter().map(|&x| x as f32 / 255.0).collect::<Vec<_>>();
    let tensor =
        Tensor::from_vec(data, (768, 768, 3), &candle::Device::Cpu).map_err(anyhow::Error::msg)?;
    // Convert from HWC to CHW format
    Ok(tensor.permute((2, 0, 1))?.unsqueeze(0)?)
}

fn encode_text_embeddings(
    prompt: &str,
    negative_prompt: &str,
    device: &candle::Device,
    dtype: candle::DType,
    do_classifier_free_guidance: bool,
) -> Result<(Tensor, Tensor, Tensor, Tensor)> {
    let api = hf_hub::api::sync::Api::new()?;

    // Load T5 embeddings
    let t5_emb = {
        // let repo = api.repo(hf_hub::Repo::with_revision(
        //     "google/t5-v1_1-xxl".to_string(),
        //     hf_hub::RepoType::Model,
        //     "refs/pr/2".to_string(),
        // ));
        // let model_file = repo.get("model.safetensors")?;
        let reponame = "comfyanonymous/flux_text_encoders";
        let repofile = "t5xxl_fp16.safetensors";
        let repo = api.repo(hf_hub::Repo::model(reponame.to_string()));

        println!(
            "Getting T5 from {:?} url {:?}",
            repo,
            repo.url("model.safetensors")
        );
        let model_file = repo.get(repofile)?;

        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[model_file], dtype, device)? };
        //let config_filename = repo.get("config.json")?;
        //let config = std::fs::read_to_string(config_filename)?;
        //let config: t5::Config = serde_json::from_str(&config)?;
        let repo = api.repo(hf_hub::Repo::with_revision(
            "google/t5-v1_1-xxl".to_string(),
            hf_hub::RepoType::Model,
            "refs/pr/2".to_string(),
        ));
        let config_filename = repo.get("config.json")?;
        let config = std::fs::read_to_string(config_filename)?;
        let config: t5::Config = serde_json::from_str(&config)?;

        let mut model = t5::T5EncoderModel::load(vb, &config)?;
        let tokenizer_filename = api
            .model("lmz/mt5-tokenizers".to_string())
            .get("t5-v1_1-xxl.tokenizer.json")?;
        let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;

        let mut tokens = tokenizer
            .encode(prompt, true)
            .map_err(E::msg)?
            .get_ids()
            .to_vec();
        tokens.resize(128, 0); // Max sequence length for HiDream
        let input_token_ids = Tensor::new(&tokens[..], device)?.unsqueeze(0)?;
        model.forward(&input_token_ids)?
    };

    // Load CLIP embeddings (both CLIP models)
    let (clip_emb_1, clip_emb_2) = {
        let repo = api.repo(hf_hub::Repo::model(
            "openai/clip-vit-large-patch14".to_string(),
        ));
        let model_file = repo.get("model.safetensors")?;
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[model_file], dtype, device)? };
        let config = clip::text_model::ClipTextConfig {
            vocab_size: 49408,
            projection_dim: 768,
            activation: clip::text_model::Activation::QuickGelu,
            intermediate_size: 3072,
            embed_dim: 768,
            max_position_embeddings: 77,
            pad_with: None,
            num_hidden_layers: 12,
            num_attention_heads: 12,
        };
        let model = clip::text_model::ClipTextTransformer::new(vb.pp("text_model"), &config)?;
        let tokenizer_filename = repo.get("tokenizer.json")?;
        let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;

        let tokens = tokenizer
            .encode(prompt, true)
            .map_err(E::msg)?
            .get_ids()
            .to_vec();
        let input_token_ids = Tensor::new(&tokens[..], device)?.unsqueeze(0)?;
        let text_outputs = model.forward(&input_token_ids)?;
        
        println!("CLIP text_outputs shape: {:?}", text_outputs.shape());
        
        // Get pooled output (last hidden state of [CLS] token)
        // CLIP text model output shape is typically [batch_size, seq_len, hidden_size]
        // We need to take the [CLS] token which is at position 0
        let pooled_output = if text_outputs.dims().len() == 3 {
            text_outputs.i((.., 0, ..))?  // Take [CLS] token embedding
        } else {
            // If output is 2D, it might already be pooled
            text_outputs.clone()
        };
        (pooled_output.clone(), pooled_output) // Using same CLIP model for both embeddings for now
    };

    // Load LLaMA embeddings (placeholder - would need actual LLaMA model)
    let llama_emb = {
        // For now, create a placeholder tensor with the expected shape
        // In a real implementation, you'd load the LLaMA model here
        Tensor::zeros((1, 128, 4096), dtype, device)?
    };

    // Combine CLIP embeddings
    println!("CLIP emb 1 shape: {:?}", clip_emb_1.shape());
    println!("CLIP emb 2 shape: {:?}", clip_emb_2.shape());
    let pooled_emb = Tensor::cat(&[&clip_emb_1, &clip_emb_2], D::Minus1)?;
    println!("Combined pooled emb shape: {:?}", pooled_emb.shape());

    // Handle negative prompts for classifier-free guidance
    let (neg_t5_emb, neg_pooled_emb) = if do_classifier_free_guidance {
        // For simplicity, using zero tensors for negative embeddings
        // In a real implementation, you'd encode the negative prompt
        let neg_t5 = Tensor::zeros_like(&t5_emb)?;
        let neg_pooled = Tensor::zeros_like(&pooled_emb)?;
        (neg_t5, neg_pooled)
    } else {
        (t5_emb.clone(), pooled_emb.clone())
    };

    Ok((t5_emb, llama_emb, pooled_emb, neg_pooled_emb))
}

fn run(args: Args) -> Result<()> {
    use tracing_chrome::ChromeLayerBuilder;
    use tracing_subscriber::prelude::*;

    let _guard = if args.tracing {
        let (chrome_layer, guard) = ChromeLayerBuilder::new().build();
        tracing_subscriber::registry().with(chrome_layer).init();
        Some(guard)
    } else {
        None
    };

    let device = candle_examples::device(args.cpu)?;
    if let Some(seed) = args.seed {
        device.set_seed(seed)?;
    }
    let dtype = device.bf16_default_to_f32();

    // Validate arguments
    if args.model.is_editing_model() && args.input_image.is_none() {
        anyhow::bail!("Input image is required for editing model (E1)");
    }

    let api = hf_hub::api::sync::Api::new()?;
    let repo = api.repo(hf_hub::Repo::model(args.model.model_id().to_string()));

    // Load input image for editing models
    let input_image = if let Some(img_path) = &args.input_image {
        Some(load_image(img_path)?.to_device(&device)?.to_dtype(dtype)?)
    } else {
        None
    };

    // Encode text embeddings
    let do_cfg = args
        .guidance_scale
        .unwrap_or(args.model.default_guidance_scale())
        > 1.0;
    let (t5_emb, llama_emb, pooled_emb, neg_pooled_emb) =
        encode_text_embeddings(&args.prompt, &args.negative_prompt, &device, dtype, do_cfg)?;

    println!("Text embeddings encoded successfully");

    // Load Flux VAE for proper image encoding/decoding
    println!("Loading Flux VAE...");
    let vae_repo = api.repo(hf_hub::Repo::model("black-forest-labs/FLUX.1-dev".to_string()));
    let vae_file = vae_repo.get("ae.safetensors")?;
    let vae_vb = unsafe { VarBuilder::from_mmaped_safetensors(&[vae_file], dtype, &device)? };
    let vae_config = flux::autoencoder::Config::dev();
    let vae = flux::autoencoder::AutoEncoder::new(&vae_config, vae_vb)?;
    println!("VAE loaded successfully");

    // Load HiDream model from the new Comfy-Org repository structure
    println!("Loading model file: {}", args.model.model_filename());
    let model_file = repo.get(args.model.model_filename())?;
    
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[model_file], dtype, &device)? };

    // Create HiDream model config based on the variant
    let config = hidream::Config {
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
        llama_layers: (0..48).collect(), // All LLaMA layers
    };
    
    // Load the HiDream model
    println!("Loading HiDream model...");
    let model = hidream::HDModel::new(&config, vb)?;
    println!("HiDream model loaded successfully");

    // Prepare latents using VAE scale factor
    let vae_scale_factor = 8;
    let latent_height = args.height / vae_scale_factor / 2; // Additional /2 for patch packing
    let latent_width = args.width / vae_scale_factor / 2;
    let mut latents = Tensor::randn(0f32, 1f32, (1, 64, latent_height, latent_width), &device)?
        .to_dtype(dtype)?;

    println!("Starting generation with {} steps...", args.num_inference_steps.unwrap_or(args.model.default_steps()));

    // Initialize scheduler
    let num_steps = args.num_inference_steps.unwrap_or(args.model.default_steps());
    let guidance_scale = args.guidance_scale.unwrap_or(args.model.default_guidance_scale());
    
    let mut scheduler = hidream::schedulers::FlowMatchEulerDiscreteScheduler::new(
        1000, // num_train_timesteps
        3.0,  // shift
        false // use_dynamic_shifting
    );
    scheduler.set_timesteps(num_steps, &device)?;
    let timesteps = scheduler.get_timesteps(&device, dtype)?;

    // Encode input image if provided (for editing models)
    let input_latents = if let Some(input_img) = input_image {
        println!("Encoding input image...");
        let encoded = vae.encode(&input_img)?;
        Some(encoded)
    } else {
        None
    };

    // Generation loop with actual model forward passes
    for (step, timestep) in timesteps.to_vec1::<f32>()?.iter().enumerate() {
        println!("Step {}/{} (timestep: {:.2})", step + 1, num_steps, timestep);

        let timestep_tensor = Tensor::new(&[*timestep], &device)?.to_dtype(dtype)?;
        
        // Prepare model inputs
        let latent_model_input = if guidance_scale > 1.0 {
            // Classifier-free guidance: duplicate latents
            Tensor::cat(&[&latents, &latents], 0)?
        } else {
            latents.clone()
        };

        // Prepare text embeddings for CFG
        let (encoder_hidden_states, pooled_embeds) = if guidance_scale > 1.0 {
            let t5_combined = Tensor::cat(&[&neg_pooled_emb, &t5_emb], 0)?;
            let llama_combined = Tensor::cat(&[&llama_emb, &llama_emb], 0)?;
            let pooled_combined = Tensor::cat(&[&neg_pooled_emb, &pooled_emb], 0)?;
            (vec![t5_combined, llama_combined], pooled_combined)
        } else {
            (vec![t5_emb.clone(), llama_emb.clone()], pooled_emb.clone())
        };

        // Concatenate input image latents for editing models
        let model_input = if let Some(ref input_lats) = input_latents {
            Tensor::cat(&[&latent_model_input, input_lats], D::Minus1)?
        } else {
            latent_model_input
        };

        // Forward pass through the model
        let noise_pred = model.forward_with_cfg(
            &model_input,
            &timestep_tensor,
            &encoder_hidden_states,
            &pooled_embeds,
            None, // img_sizes
            None, // img_ids
        )?;

        // Apply classifier-free guidance
        let noise_pred = if guidance_scale > 1.0 {
            let chunks = noise_pred.chunk(2, 0)?;
            let noise_pred_uncond = &chunks[0];
            let noise_pred_text = &chunks[1];
            let guidance_tensor = Tensor::new(&[guidance_scale as f32], &device)?.to_dtype(dtype)?;
            let guidance_tensor = guidance_tensor.broadcast_as(noise_pred_text.shape())?;
            
            (noise_pred_uncond + &((noise_pred_text - noise_pred_uncond)? * &guidance_tensor)?)?
        } else {
            noise_pred
        };

        // Scheduler step
        latents = scheduler.step(&noise_pred, *timestep as f64, &latents)?;
    }

    println!("Generation completed, decoding latents...");

    // Decode latents to image using VAE
    let vae_scale_factor = 0.3611; // Flux VAE scale factor
    let vae_shift_factor = 0.1159; // Flux VAE shift factor
    
    let scaled_latents = ((&latents / vae_scale_factor)? + vae_shift_factor)?;
    let decoded_image = vae.decode(&scaled_latents)?;
    
    // Post-process image
    let img = decoded_image.clamp(-1f32, 1f32)?;
    let img = ((&img + 1.0)? * 127.5)?;
    let img = img.to_dtype(candle::DType::U8)?;

    // Save image
    candle_examples::save_image(&img.i(0)?, &args.output)?;
    println!("Image saved to {}", args.output);

    Ok(())
}

fn main() -> Result<()> {
    let args = Args::parse();
    run(args)
}

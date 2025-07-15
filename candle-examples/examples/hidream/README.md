# HiDream Image Generation and Editing

This example demonstrates how to use the HiDream models for both image generation (I1 variants) and image editing (E1 variant) using the Candle framework.

## Models

HiDream supports multiple model variants:

### Generation Models (I1)
- **HiDream-I1-Dev**: Fast generation model (28 steps, guidance_scale=0.0)
- **HiDream-I1-Full**: High quality generation model (50 steps, guidance_scale=5.0)
- **HiDream-I1-Fast**: Fastest generation model (16 steps, guidance_scale=0.0)

### Editing Models (E1)
- **HiDream-E1-Full**: Image editing model (28 steps, guidance_scale=5.0)

## Usage

### Basic Image Generation

```bash
# Generate an image with default settings (I1-Full model)
cargo run --example hidream --release -- --prompt "A cat holding a sign that says \"Hi-Dreams.ai\""

# Use different model variants
cargo run --example hidream --release -- --model i1-dev --prompt "A beautiful landscape"
cargo run --example hidream --release -- --model i1-fast --prompt "A robot in space"

# Customize generation parameters
cargo run --example hidream --release -- \
    --prompt "A futuristic city at sunset" \
    --height 1024 \
    --width 1024 \
    --num-inference-steps 50 \
    --guidance-scale 7.5 \
    --seed 42 \
    --output my_image.jpg
```

### Image Editing (E1 Model)

```bash
# Edit an existing image
cargo run --example hidream --release -- \
    --model e1-full \
    --prompt "Editing Instruction: Convert the image into a Ghibli style. Target Image Description: A person in a light pink t-shirt with short dark hair, depicted in a Ghibli style against a plain background." \
    --input-image input.jpg \
    --guidance-scale 5.0 \
    --image-guidance-scale 4.0 \
    --negative-prompt "low resolution, blur" \
    --output edited_output.jpg
```

## Command Line Arguments

- `--prompt`: The text prompt for generation or editing instruction
- `--model`: Model variant to use (i1-dev, i1-full, i1-fast, e1-full)
- `--height`: Output image height in pixels (default: 1024)
- `--width`: Output image width in pixels (default: 1024)
- `--num-inference-steps`: Number of denoising steps (optional, uses model default)
- `--guidance-scale`: Classifier-free guidance scale (optional, uses model default)
- `--image-guidance-scale`: Image guidance scale for editing (default: 4.0)
- `--input-image`: Input image path for editing (required for E1 model)
- `--negative-prompt`: Negative prompt (default: "low resolution, blur")
- `--seed`: Random seed for reproducible results
- `--output`: Output filename (default: hidream_output.jpg)
- `--cpu`: Run on CPU instead of GPU
- `--tracing`: Enable performance tracing

## Model Details

### Text Encoders
HiDream uses multiple text encoders for rich text understanding:
- **T5-XXL**: Primary text encoder for detailed text understanding
- **CLIP**: Two CLIP models for visual-text alignment
- **LLaMA-3.1-8B**: Advanced language model for instruction following

### Architecture
- **Transformer-based**: Uses a transformer architecture similar to Flux
- **Multi-expert**: Employs mixture-of-experts (MoE) for efficient computation
- **Dual-stream**: Separate processing streams for different modalities

### Resolution Support
The models support various resolutions:
- 1024 × 1024 (Square)
- 768 × 1360 (Portrait)
- 1360 × 768 (Landscape)
- 880 × 1168 (Portrait)
- 1168 × 880 (Landscape)
- 1248 × 832 (Landscape)
- 832 × 1248 (Portrait)

## Examples

### Text-to-Image Generation

```bash
# Simple generation
cargo run --example hidream --release -- --prompt "A majestic dragon flying over mountains"

# High quality generation with custom parameters
cargo run --example hidream --release -- \
    --model i1-full \
    --prompt "A cyberpunk cityscape with neon lights reflecting on wet streets" \
    --height 1360 \
    --width 768 \
    --guidance-scale 7.5 \
    --num-inference-steps 50 \
    --seed 123
```

### Image Editing

```bash
# Style transfer
cargo run --example hidream --release -- \
    --model e1-full \
    --prompt "Editing Instruction: Convert to anime style. Target Image Description: An anime-style version of the original image with vibrant colors and stylized features." \
    --input-image photo.jpg \
    --guidance-scale 5.0 \
    --image-guidance-scale 4.0

# Object modification
cargo run --example hidream --release -- \
    --model e1-full \
    --prompt "Editing Instruction: Change the car color to red. Target Image Description: The same scene but with a bright red car instead of the original color." \
    --input-image car_scene.jpg
```

## Performance Tips

1. **Use appropriate model variants**: 
   - Use `i1-fast` for quick previews
   - Use `i1-full` for high-quality final images
   - Use `i1-dev` for balanced speed/quality

2. **Optimize inference steps**:
   - Fewer steps = faster generation
   - More steps = higher quality (diminishing returns after model defaults)

3. **GPU acceleration**: 
   - The models are optimized for GPU usage
   - Use `--cpu` only for testing or if GPU is unavailable

4. **Memory management**:
   - Lower resolutions use less memory
   - Consider using smaller batch sizes for limited VRAM

## Troubleshooting

### Common Issues

1. **Out of memory**: Reduce image resolution or use CPU mode
2. **Slow generation**: Ensure GPU acceleration is enabled
3. **Poor quality**: Increase guidance scale or inference steps
4. **Model loading errors**: Check internet connection for model downloads

### Requirements

- CUDA-capable GPU (recommended)
- At least 8GB VRAM for full models
- Internet connection for initial model downloads
- Rust 1.70+ with Candle dependencies

## Implementation Status

**Note**: This is a work-in-progress implementation. Some features may be incomplete:

- ✅ Complete CLI framework and model variants
- ✅ Text encoder integration (T5, CLIP)
- ✅ HiDream transformer implementation with forward pass
- ✅ Scheduler implementations (FlowMatch, UniPC, FlashFlow)
- ✅ Dual-stream attention and MoE feed-forward networks
- ✅ Generation loop framework
- 🚧 VAE decoder integration needed
- 🚧 LLaMA text encoder integration needed
- ❌ LoRA support for E1 editing
- ❌ Instruction refinement

The current implementation provides a solid foundation with a complete transformer architecture. The main remaining work is VAE integration and LLaMA text encoder support.

## Task Status

### Todo
- [ ] Implement proper VAE decoder integration
- [ ] Integrate LLaMA-3.1-8B text encoder
- [ ] Add LoRA support for E1 editing model
- [ ] Implement instruction refinement functionality
- [ ] Add proper error handling and validation
- [ ] Optimize memory usage for large models
- [ ] Add quantized model support
- [ ] Create comprehensive test suite

### Doing
- [x] Complete generation loop with scheduler integration
- [x] Update main.rs to use new transformer and schedulers

### Done
- [x] Project structure setup
- [x] Basic example framework
- [x] README documentation
- [x] Command-line interface design
- [x] Model variant definitions
- [x] Asset directory structure
- [x] Usage examples and scripts
- [x] Basic CLI interface and argument parsing
- [x] Model variant configuration (I1-Dev, I1-Full, I1-Fast, E1-Full)
- [x] Text encoder integration framework (T5, CLIP)
- [x] HiDream transformer forward implementation
- [x] Scheduler implementations (FlowMatchEulerDiscreteScheduler, FlowUniPCMultistepScheduler, FlashFlowMatchEulerDiscreteScheduler)
- [x] HDAttention with dual-stream processing
- [x] HDBlockDouble and HDBlockSingle implementations
- [x] Mixture-of-Experts (MoE) feed-forward networks
- [x] Positional encoding and timestep embeddings

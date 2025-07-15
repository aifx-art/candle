# Candle

Candle is a minimalist ML framework for Rust with a focus on performance (including GPU support) and ease of use.

## HiDream Implementation Status

### Current Status
The HiDream implementation is partially complete but missing key components for actual image generation.

### Todos

#### Doing
- Fix tensor operation errors in main.rs
- Implement proper VAE loading and usage

#### Todo
- [ ] **Load and use actual Flux VAE for encoding/decoding**
  - Need to load VAE model from huggingface
  - Implement proper encode/decode functions
  - Use VAE scale_factor and shift_factor correctly

- [ ] **Implement actual HiDream model loading**
  - Parse safetensors file structure
  - Load model weights into HDModel struct
  - Handle different model variants (I1-Full, I1-Dev, I1-Fast, E1)

- [ ] **Implement proper denoising loop**
  - Load scheduler (FlowMatch or UniPC)
  - Implement timestep scheduling
  - Add actual model forward calls in generation loop
  - Apply classifier-free guidance correctly

- [ ] **Fix text encoder implementations**
  - Load proper LLaMA model for text encoding
  - Implement correct text projection layers
  - Handle multiple text encoders (T5, CLIP, LLaMA)

- [ ] **Add proper image preprocessing**
  - Implement correct image resizing and normalization
  - Handle different input resolutions
  - Add proper VAE encoding for input images

- [ ] **Implement model forward pass**
  - Connect all embeddings to model input
  - Implement proper attention masking
  - Handle dual-stream and single-stream blocks

- [ ] **Add scheduler implementations**
  - FlowMatchEulerDiscreteScheduler
  - UniPCMultistepScheduler
  - Proper timestep calculation

### Done
- [x] Basic project structure
- [x] Command line argument parsing
- [x] Text embedding placeholders (T5, CLIP)
- [x] Model config structure
- [x] Basic tensor operations fixed

### Issues Found
1. **No actual VAE usage** - Currently using placeholder image processing
2. **Missing model forward calls** - Generation loop is just a placeholder
3. **Incomplete text encoders** - LLaMA embeddings are just zeros
4. **No scheduler implementation** - Missing denoising process
5. **Tensor operation errors** - Fixed basic arithmetic operations

### Reference Files
- `candle-examples/examples/hidream/main.rs` - Main implementation
- `candle-transformers/src/models/hidream/mod.rs` - Model definitions
- `candle-examples/examples/hidream/reference/` - Python reference implementations
- `candle-transformers/src/models/flux/autoencoder.rs` - VAE reference

The implementation needs significant work to match the Python reference and actually generate images.

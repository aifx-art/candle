# Reference code
our code:
@/candle-examples/examples/hidream/main.rs 
@/candle-transformers/src/models/hidream/mod.rs 

our reference:
@/candle-examples/examples/hidream/reference
and
@/candle-transformers/src/models/hidream/reference/model.py

# HiDream Implementation TODO List

## Current Issues and Fixes Needed

### 1. VAE Usage Issues ⚠️
- ✅ Load Flux VAE from huggingface 
- ✅ Implement proper encode/decode functions
- ✅ Add VAE scale_factor and shift_factor usage
- ❌ **CRITICAL**: VAE is loaded but not used for proper latent space conversion
- ❌ **CRITICAL**: Input latents are created with `Tensor::randn` instead of VAE encoding
- ❌ **CRITICAL**: Latent dimensions don't match expected VAE latent space

### 2. Model Forward Pass Issues ⚠️
- ✅ Model forward calls are implemented in generation loop
- ❌ **CRITICAL**: `forward_with_cfg` method exists but has incomplete implementation
- ❌ **CRITICAL**: Missing proper weight loading from safetensors into model layers
- ❌ **CRITICAL**: Model layers are created but weights aren't loaded from the safetensors file
- ❌ **CRITICAL**: VarBuilder paths don't match actual safetensors structure

### 3. Weight Loading Issues ❌
- ❌ **CRITICAL**: Safetensors file is loaded but weights aren't mapped to model components
- ❌ **CRITICAL**: Need to inspect safetensors structure and map to HDModel layers
- ❌ **CRITICAL**: VarBuilder paths need to match actual weight names in safetensors
- ❌ **CRITICAL**: Missing proper model instantiation with loaded weights

### 4. Text Encoder Issues ❌
- ❌ **CRITICAL**: LLaMA embeddings are just zero tensors (placeholder)
- ❌ **CRITICAL**: Need to load actual LLaMA model for proper text encoding
- ❌ **CRITICAL**: Missing text projection layers implementation
- ❌ **CRITICAL**: Caption projection layers are empty in HDModel
- ⚠️ T5 and CLIP encoders work but may need better integration

### 5. Scheduler Implementation ✅
- ✅ FlowMatch scheduler implemented
- ✅ UniPC scheduler support added
- ✅ Proper timestep calculation implemented
- ✅ Noise scheduling working

## Detailed Analysis

### Critical Issues Found:

#### A. VAE Integration Problems
```rust
// WRONG: Creating random latents instead of using VAE
let mut latents = Tensor::randn(0f32, 1f32, (1, 64, latent_height, latent_width), &device)?;

// SHOULD BE: Encode noise in VAE latent space or start from VAE-encoded image
```

#### B. Model Weight Loading Problems
```rust
// CURRENT: VarBuilder is created but weights aren't properly loaded
let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[model_file], dtype, &device)? };
let model = hidream::HDModel::new(&config, vb)?; // Weights may not match structure

// NEED: Inspect safetensors structure and map correctly
```

#### C. Text Encoder Problems
```rust
// CURRENT: Placeholder zero tensors
let llama_emb = Tensor::zeros((1, 128, 4096), dtype, device)?;

// NEED: Actual LLaMA model loading and inference
```

#### D. Model Forward Pass Issues
- `forward_with_cfg` exists but may have implementation gaps
- Missing proper handling of text projections
- Positional encoding may not be correctly implemented

## Implementation Plan

### Phase 1: Fix Weight Loading 🔄 DOING
- [ ] Inspect safetensors file structure to understand weight naming
- [ ] Map safetensors weight names to HDModel layer names
- [ ] Fix VarBuilder paths to match actual weight structure
- [ ] Verify model layers are properly initialized with weights
- [ ] Add weight loading validation

### Phase 2: Fix VAE Integration 🔄 DOING  
- [ ] Replace random latent generation with proper VAE encoding
- [ ] Implement proper latent space initialization
- [ ] Fix latent dimensions to match VAE expectations
- [ ] Add proper VAE scaling and shifting
- [ ] Test VAE encode/decode pipeline

### Phase 3: Implement Text Encoders ❌
- [ ] Load actual LLaMA model for text encoding
- [ ] Implement caption projection layers properly
- [ ] Fix text embedding integration in forward pass
- [ ] Add proper text tokenization and encoding
- [ ] Integrate all text encoders (T5, CLIP, LLaMA)

### Phase 4: Fix Model Forward Pass ❌
- [ ] Debug and fix `forward_with_cfg` implementation
- [ ] Ensure proper text embedding handling
- [ ] Fix positional encoding implementation
- [ ] Add proper attention mask handling
- [ ] Validate model output shapes

### Phase 5: Integration and Testing ❌
- [ ] End-to-end pipeline testing
- [ ] Verify output quality
- [ ] Performance optimization
- [ ] Add proper error handling

## Current Status: DOING
**Priority 1**: Fix weight loading from safetensors (Phase 1)
**Priority 2**: Fix VAE integration (Phase 2)
**Priority 3**: Implement proper text encoders (Phase 3)

## Next Steps Required:

1. **Inspect safetensors structure**: Use tools to examine the actual weight names and shapes
2. **Fix VarBuilder paths**: Map the weight names to the correct model components  
3. **Replace random latents**: Use proper VAE encoding for latent initialization
4. **Implement LLaMA encoder**: Load and use actual LLaMA model for text encoding
5. **Test model forward pass**: Ensure the model actually produces meaningful outputs

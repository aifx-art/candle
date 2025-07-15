# HiDream Implementation TODO List

## Current Issues and Fixes Needed

### 1. No Actual VAE Usage ✅
- ✅ Load Flux VAE from huggingface 
- ✅ Implement proper encode/decode functions
- ✅ Add VAE scale_factor and shift_factor usage

### 2. Missing Model Forward Calls ❌
- The generation loop is just a placeholder with no actual model inference
- Need to implement proper denoising process with scheduler
- Missing actual HDModel.forward() calls during generation steps

### 3. Incomplete Model Loading ❌
- The safetensors file is loaded but weights aren't actually used to create the model
- Need to parse the tensor structure and load weights into HDModel
- Missing proper model instantiation

### 4. Text Encoder Issues ❌
- LLaMA embeddings are just zero tensors (placeholder)
- Need to load actual LLaMA model for proper text encoding
- Missing text projection layers

### 5. Missing Scheduler Implementation ✅
- ✅ Implement FlowMatch scheduler
- ✅ Add UniPC scheduler support  
- ✅ Implement proper timestep calculation and noise scheduling

## Implementation Plan

### Phase 1: VAE Integration 🔄
- [ ] Load Flux VAE from huggingface
- [ ] Implement proper encode/decode functions
- [ ] Add VAE scale_factor and shift_factor usage
- [ ] Replace placeholder image processing

### Phase 2: Model Loading and Forward Pass 🔄
- [ ] Implement proper HDModel instantiation from safetensors
- [ ] Add weight loading and parsing
- [ ] Implement actual forward pass in generation loop
- [ ] Add proper model inference calls

### Phase 3: Text Encoders 🔄
- [ ] Implement proper LLaMA text encoding
- [ ] Add text projection layers
- [ ] Fix CLIP and T5 embedding integration

### Phase 4: Scheduler Implementation 🔄
- [ ] Implement FlowMatch scheduler
- [ ] Add UniPC scheduler support
- [ ] Implement proper timestep calculation
- [ ] Add noise scheduling

### Phase 5: Integration and Testing 🔄
- [ ] Integrate all components
- [ ] Test generation pipeline
- [ ] Verify output quality
- [ ] Performance optimization

## Current Status: DOING
Working on Phase 1: VAE Integration and Phase 2: Model Loading

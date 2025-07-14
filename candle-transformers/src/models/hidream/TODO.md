# HiDream Implementation TODO

## TODO
- Implement complete HDAttention forward method with proper rope application
- Fix HDBlockDouble and HDBlockSingle forward methods
- Implement TimestepEmbed properly with Timesteps and TimestepEmbedding
- Add proper support for image conditioning (HiDream E1)
- Implement HDBlockSingle struct and methods
- Fix rope function to match Flux implementation
- Add proper attention function with rope application
- Implement complete HDModel forward method
- Add support for both I1 and E1 variants
- Fix tensor operations and shape handling
- Add proper error handling throughout

## Doing
- Need to fix final compilation error with linear function call
- Implement actual forward methods for HDBlockDouble and HDBlockSingle
- Implement complete HDModel forward method
- Add support for image conditioning (HiDream E1)

## Done
- Initial structure created
- Basic structs defined
- Added TimestepEmbed struct
- Added HDBlockSingle struct
- Fixed basic struct definitions
- Implemented helper functions for topk, mask_where, masked_fill
- Fixed HDAttention implementation
- Fixed HDLastLayer with proper forward_with_vec method
- Fixed most compilation errors
- Added proper imports and dependencies
- Implemented HDMOEFeedForwardSwiGLU with MoE gating
- Fixed tensor dimension and shape operations

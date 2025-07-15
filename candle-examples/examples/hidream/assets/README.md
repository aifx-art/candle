# HiDream Assets

This directory contains test assets for the HiDream examples.

## Test Images

Place your test images here for use with the E1 editing model:

- `test_1.png` - Sample image for editing experiments
- `input.jpg` - General input image for editing
- `photo.jpg` - Photo for style transfer examples

## Usage

When running the E1 editing model, reference images in this directory:

```bash
cargo run --example hidream --release -- \
    --model e1-full \
    --input-image assets/test_1.png \
    --prompt "Your editing instruction here"
```

## Supported Formats

The example supports common image formats:
- PNG
- JPEG/JPG
- BMP
- TIFF

Images will be automatically resized to 768x768 for processing.

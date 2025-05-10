# Embedded AI Vision-Language Model System

## Overview
This repository implements an embedded vision-language model (VLM) system that combines efficient image and text encoders with a compressed 4-bit LLM for real-time inference. The system is designed to run on edge devices with limited computational resources.

## Architecture

### Core Components

1. **Image Encoder**
   - Primary: MobileNetV3 architecture
   - Alternative: MobileNetV2-Large architecture
   - Both models quantized to INT8 for efficient inference
   - Processes RGB images to generate feature embeddings
   - Supports model switching based on performance requirements

2. **Text Encoder**
   - Primary: MobileBERT architecture
   - Alternative: TextCNN implementation
   - Tokenizes and encodes text input
   - Generates semantic embeddings
   - Flexible architecture selection based on use case

3. **Embedding Compression**
   - Linear projection to reduce dimensionality
   - Compresses image embeddings (1408 → 16) and text embeddings (64 → 16)
   - Maintains semantic information while reducing computational overhead

4. **LLM Integration**
   - Implements 4-bit quantized LLaMA-2-7B model
   - Processes compressed embeddings to generate natural language responses
   - Optimized for edge deployment

## Installation

### Prerequisites
- Python 3.8+
- TensorFlow Lite Runtime
- Edge TPU runtime (optional, falls back to CPU)

### Quick Start
1. Clone the repository:
```bash
git clone https://github.com/jifeng-l/embedded_ai.git
cd embedded_ai
```

2. Install dependencies:
```bash
pip install tensorflow tflite-runtime torch transformers gradio opencv-python numpy pillow llama-cpp-python
```

3. Start the service:
```bash
python gradio_vlm_service.py
```

## Key Features

- **Real-time Inference**: Supports live webcam input with text prompts
- **Edge Optimization**: INT8 quantization for both image and text encoders
- **Memory Efficiency**: Embedding compression to reduce memory footprint
- **Interactive Interface**: Gradio-based web interface for easy interaction
- **Model Flexibility**: Multiple encoder options for different use cases

## Implementation Details

### Model Quantization
The system includes scripts for:
- Converting pre-trained models to TFLite format
- INT8 quantization for edge deployment
- Representative dataset generation for quantization calibration
- Support for both MobileNetV2 and MobileNetV3 architectures

### Evaluation
- Flickr30k dataset integration for system evaluation
- Metrics: accuracy and classification report generation
- Support for custom evaluation scenarios
- Comparative analysis between different encoder architectures

## Project Structure

```
.
├── gradio_vlm_service.py    # Main service implementation
├── quantize_model.py        # Model quantization utilities
├── test_clip_service.py     # Evaluation script
├── inference.py            # Florence-2 inference implementation
├── LLm_engine.py           # LLM integration
├── build_vocab_from_corpus.py  # Vocabulary building utilities
├── convert_kagglehub_mobilenetv2.py  # Model conversion utilities
├── image_encoder.py        # Image encoder implementations
├── text_encoder.py         # Text encoder implementations
└── export_textcnn_tflite.py # TextCNN model export utilities
```

## Performance Considerations

- Frame processing rate can be adjusted via `FRAME_SKIP` parameter
- Embedding compression reduces memory usage and inference time
- System supports fallback to CPU when Edge TPU is unavailable
- Different encoder combinations offer various speed-accuracy tradeoffs

## Future Work

- Integration of additional pre-trained models
- Optimization of inference pipeline
- Support for batch processing
- Enhanced evaluation metrics
- Improved model switching mechanisms
- Additional text encoder architectures

## Citation
If you use this code in your research, please cite:
```
@software{embedded_ai_vlm,
  author = {Jifeng Li},
  title = {Embedded AI Vision-Language Model System},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/jifeng-l/embedded_ai}
}
```

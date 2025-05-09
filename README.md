# Vision-Language Model (VLM) Service with Edge TPU and LLM Integration

This project implements a real-time vision-language model service that combines Edge TPU-based image and text encoders with a 4-bit quantized LLM for efficient inference. The system processes webcam input and text prompts to generate contextual responses about visual content.

## System Architecture

The system consists of several key components:

1. **Image Encoder**: Edge TPU-based feature extractor for visual content
2. **Text Encoder**: MobileBERT-based text encoder optimized for Edge TPU
3. **Embedding Compression**: Dimensionality reduction for efficient processing
4. **LLM Integration**: 4-bit quantized LLaMA-2 model for natural language generation
5. **Web Interface**: Gradio-based UI for real-time interaction

## Prerequisites

- Python 3.9+
- Edge TPU device (Coral USB Accelerator or compatible)
- CUDA-capable GPU (optional, for Florence-2 model)

## Installation

1. Create and activate a virtual environment:
```bash
conda create -n vlm python=3.9
conda activate vlm
```

2. Install core dependencies:
```bash
pip install gradio==3.32.0
pip install tflite-runtime
pip install transformers
pip install llama-cpp-python
pip install opencv-python
pip install numpy
pip install pillow
```

3. Install Edge TPU runtime:
```bash
# For Linux
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list
sudo apt-get update
sudo apt-get install edgetpu-compiler
```

## Model Setup

1. Download the Edge TPU models:
   - Image encoder: `1_edgetpu.tflite`
   - Text encoder: `2_edgetpu.tflite`

2. Download the LLaMA-2 model:
```bash
# The model will be automatically downloaded from Hugging Face
# TheBloke/Llama-2-7B-GGUF: llama-2-7b.Q4_0.gguf
```

## Usage

1. Start the service:
```bash
python gradio_vlm_service.py
```

2. Access the web interface:
   - Open your browser and navigate to the provided local URL (typically http://localhost:7860)
   - Allow camera access when prompted
   - Enter text prompts in the text box
   - Click "Generate" to process the current frame

## Alternative Models

The repository includes an alternative implementation using Microsoft's Florence-2 model:

```bash
python inference.py
```

This version provides object detection capabilities with real-time visualization.

## Performance Considerations

- The system uses embedding compression (16 dimensions) to reduce computational overhead
- Frame processing is triggered manually to optimize resource usage
- The 4-bit quantized LLM provides a good balance between performance and accuracy

## Troubleshooting

1. Camera Access Issues:
   - Ensure proper permissions are granted in your browser
   - Check system-level camera permissions
   - Verify no other applications are using the camera

2. Edge TPU Issues:
   - Verify the Edge TPU is properly connected
   - Check USB connection if using Coral USB Accelerator
   - Ensure correct model files are in place

3. LLM Loading Issues:
   - Verify sufficient system memory
   - Check model file integrity
   - Ensure correct model path in configuration

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Edge TPU runtime and tools from Google Coral
- LLaMA-2 model from Meta AI
- Gradio for the web interface
- Hugging Face for model hosting and transformers library

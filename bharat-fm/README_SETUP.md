# Bharat Foundation Model Framework (BFMF)

## Project Setup

This repository contains the initial skeleton for the Bharat Foundation Model Framework.

### Installation

```bash
# Clone the repository
git clone https://github.com/bharat-ai/bharat-fm.git
cd bharat-fm

# Install dependencies
pip install -e .

# For development installation
pip install -e ".[dev]"
```

### Quick Start

1. **Initialize a new project**
   ```bash
   bharat init my-project
   ```

2. **Train a model**
   ```bash
   bharat train --model bharat-base --dataset indic_mix --steps 1000
   ```

3. **Evaluate the model**
   ```bash
   bharat eval --model ./outputs/checkpoint-1000 --benchmarks perplexity,generation_quality
   ```

4. **Deploy the model**
   ```bash
   bharat deploy --model ./outputs/checkpoint-1000 --host 0.0.0.0 --port 8000
   ```

### Module Overview

- **bharat_data**: Data preparation, tokenization, and preprocessing for Indian languages
- **bharat_model**: Model architectures (GLM, LLaMA, MoE) with Indian language support
- **bharat_train**: Distributed training pipeline with DeepSpeed and LoRA support
- **bharat_eval**: Comprehensive evaluation suite with multilingual benchmarks
- **bharat_deploy**: FastAPI-based deployment with vLLM and Triton integration
- **bharat_registry**: MLflow-based experiment tracking and model registry
- **bharat_cli**: Command-line interface for all operations

### Supported Languages

- Hindi (hi)
- English (en)
- Bengali (bn)
- Tamil (ta)
- Telugu (te)
- Marathi (mr)
- Gujarati (gu)
- Kannada (kn)
- Malayalam (ml)
- Punjabi (pa)

### Contributing

Please see the main README.md file for detailed contribution guidelines.

### License

Apache 2.0 / Bharat Open AI License (BOAL)
# M2XFP: A Metadata-Augmented Microscaling Data Format for Efficient Low-bit Quantization

This repository contains the pseudo quantization workflow for M2XFP and provides a lightweight way to evaluate accuracy (e.g., perplexity) on LLaMA-3 and other LLMs.

## Setup

1. Create and activate conda environment:
```bash
conda create -n mxq python=3.10
conda activate mxq
```

2. Install the package in development mode:
```bash
pip install vllm==0.7.0 --extra-index-url https://download.pytorch.org/whl/cu128
cd pseudo_quantization
pip install -e .
```

## Usage

Run the main quantization workflow:
```bash
bash llama3_run.sh wikitext

bash reasoning.sh
```

## Structure

- `entry.py` - Main entry point for quantization
- `llama3_run.sh` - An example script to run Llama3 quantization
- `quantize/` - Core quantization modules
  - `quant_func.py` - Quantization configuration and functions
  - `quantizer.py` - Main quantization logic
  - `linear.py` - Quantized linear layer implementation
  - `pre_quant.py` - Pre-quantization utilities
- `utils/` - Utility modules
  - `module.py` - Module manipulation utilities
  - `dataload_utils.py` - Data loading utilities
  - `parallel.py` - Parallel processing utilities
  - `calib_data.py` - Calibration data handling
  - `utils.py` - General utilities

## Citation

```text

```
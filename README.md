# gpt-oss-transformers
Run OpenAI’s GPT-OSS models (gpt-oss-20b, gpt-oss-120b) using Hugging Face Transformers with high-level pipelines, .generate(), transformers serve, multi-GPU inference, and MXFP4 support.


# GPT-OSS with Hugging Face Transformers

Run OpenAI’s `gpt-oss-20b` and `gpt-oss-120b` models using the 🤗 Transformers library, including:

- High-level pipeline API
- Low-level `.generate()` calls
- MXFP4 quantized inference
- `transformers serve` REST API
- Harmony format prompting
- Multi-GPU & tensor/expert parallelism

> 🧠 Based on the original guide from [OpenAI Cookbook](https://github.com/openai/openai-cookbook/blob/main/articles/gpt-oss/run-transformers.md)

---

## 🔧 Requirements

Create a fresh environment and install:

```bash
pip install -U transformers accelerate torch triton kernels
pip install git+https://github.com/triton-lang/triton.git@main#subdirectory=python/triton_kernels

#⚠️ Note: Python 3.13 is not yet supported. Please use Python 3.12.


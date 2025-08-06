# GPT-OSS with Hugging Face Transformers

Run OpenAI’s `gpt-oss-20b` and `gpt-oss-120b` models using the 🤗 Transformers library, including:

- ✅ High-level pipeline API
- ✅ Low-level `.generate()` calls
- ✅ MXFP4 quantized inference
- ✅ `transformers serve` REST API
- ✅ Harmony format prompting
- ✅ Multi-GPU & expert parallel inference

> 🧠 Based on the original guide from [OpenAI Cookbook](https://github.com/openai/openai-cookbook/blob/main/articles/gpt-oss/run-transformers.md)

---

## 🔧 Requirements

Create a fresh Python 3.12 environment and install:

```bash
pip install -U transformers accelerate torch triton kernels
pip install git+https://github.com/triton-lang/triton.git@main#subdirectory=python/triton_kernels
```

> ⚠️ Python 3.13 is not supported for `triton`. Use Python 3.12 instead.

---

## 🚀 Quick Inference via `pipeline`

```python
# quick_pipeline.py
from transformers import pipeline

generator = pipeline(
    "text-generation",
    model="openai/gpt-oss-20b",
    torch_dtype="auto",
    device_map="auto"
)

messages = [{"role": "user", "content": "Explain MXFP4 quantization"}]
output = generator(messages, max_new_tokens=200)
print(output[0]["generated_text"])
```

---

## ⚙️ Advanced Inference with `.generate()`

```python
# advanced_generate.py
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "openai/gpt-oss-20b"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)

messages = [{"role": "user", "content": "Explain what MXFP4 quantization is."}]
inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt"
).to(model.device)

outputs = model.generate(**inputs, max_new_tokens=200, temperature=0.7)
print(tokenizer.decode(outputs[0]))
```

---

## 🌐 Serve the Model as an API

```bash
# serve_model.sh
transformers serve
```

Send a request with:

```bash
curl -X POST http://localhost:8000/v1/responses \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "system", "content": "hello"}], "temperature": 0.9, "max_tokens": 1000, "stream": true, "model": "openai/gpt-oss-20b"}'
```

---

## 🧠 Harmony Prompting

Install:

```bash
pip install openai-harmony
```

```python
# harmony_chat.py
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "openai/gpt-oss-20b"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto")

messages = [
    {"role": "system", "content": "Always respond in riddles"},
    {"role": "user", "content": "What is the weather like in Madrid?"}
]

inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt"
).to(model.device)

generated = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(generated[0][inputs["input_ids"].shape[-1]:]))
```

---

## 🧪 Multi-GPU with Expert Parallelism

```python
# multi_gpu_generate.py
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.distributed import DistributedConfig
import torch

model_path = "openai/gpt-oss-120b"
tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left")

device_map = {
    "distributed_config": DistributedConfig(enable_expert_parallel=1),
    "tp_plan": "auto",
}

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype="auto",
    attn_implementation="kernels-community/vllm-flash-attn3",
    **device_map,
)

messages = [
    {"role": "user", "content": "Explain how expert parallelism works in large language models."}
]

inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt"
).to(model.device)

outputs = model.generate(**inputs, max_new_tokens=1000)

response = tokenizer.decode(outputs[0])
print("Model response:", response.split("<|channel|>final<|message|>")[-1].strip())
```

Run on multiple GPUs:

```bash
torchrun --nproc_per_node=4 multi_gpu_generate.py
```

---

## 📄 License

MIT License

---

## 📚 Credits

This project is based on:

📖 [OpenAI Cookbook - Run GPT-OSS with Transformers](https://github.com/openai/openai-cookbook/blob/main/articles/gpt-oss/run-transformers.md)

All credits to [@DominikKundel](https://github.com/dkundel) and OpenAI for the original content.

# Ollama
I have already deployed Qwen2.5:latest on Ollama. To run it, simply open PowerShell and run
```
$env:OLLAMA_MODELS = 'D:\ollama\models'
$env:OLLAMA_HOST="0.0.0.0:8000"
Start-Process ollama -ArgumentList "serve"

# Run the model (below are the models I intalled in local Ollama)
ollama run qwen3:8b
ollama run qwen2.5:latest
ollama run deepseek-r1:8b # doesn't support toolcalls
ollama run gemma3:4b # doesn't support toolcalls
```
# vllm
We will use vllm to deploy Qwen3-8B. But Windows doesn't support vllm, we need to run it in WSL.  
In WSL CLI, install conda first:
```
# Download latest Miniconda. Conda environments set up in Windows won't be available here.
# Windows Conda and WSL Conda are 2 different systems
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

# Install
bash Miniconda3-latest-Linux-x86_64.sh

# After installation, re-open terminal or run
source ~/.bashrc

# check conda version
conda --version
```
Then create a separate environment for this deployment and install vllm.
```
# Create a new environment named 'vllm-env' with Python 3.11
conda create -n vllm-env python=3.11 -y

# Activate the environment
conda activate vllm-env

# Install vllm
pip install vllm
```
- Copy the model files to WSL system. WSL has a separate file system. The files in it cannot be accessed by Windows explorer.
```
cp -r /mnt/c/Users/Administrator/.cache/modelscope/hub/models/Qwen/Qwen3-8B ~/models/
```
- Or download the model from modelscope
```
modelscope download --model Qwen/Qwen2.5-3B-Instruct-GPTQ-Int4
```
Launch the model
```
python -m vllm.entrypoints.openai.api_server \
    --model ~/models/Qwen3-8B \
    --served-model-name qwen3-8b \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.9 \
    --enforce-eager
```
The above code can run, but every slow, the response from the agent will cost more than 10 seconds.
- `--enforce-eager` disabled core optimization including PageAttention, and only allows the slowest original PyTorch inference.
- However, if we remove `--enforce-eager`, Qwen3-8B cannot even run as RTX3090 24GB RAM doesn't support it.

Launch Qwen2.5-7B Instruct-AWQ
```
python -m vllm.entrypoints.openai.api_server \
    --model ~/.cache/modelscope/hub/models/Qwen/Qwen2.5-7B-Instruct-AWQ \
    --served-model-name Qwen2.5-7B-Instruct-AWQ \
    --gpu-memory-utilization 0.9 \
    --quantization awq 
```
Launch Qwen3-8B-AWQ
```
python -m vllm.entrypoints.openai.api_server \
    --model ~/.cache/modelscope/hub/models/Qwen/Qwen3-8B-AWQ \
    --served-model-name Qwen3-8B-AWQ \
    --gpu-memory-utilization 0.9 \
    --quantization awq
```
```
python -m vllm.entrypoints.openai.api_server \
  --model ~/.cache/modelscope/hub/models/Qwen/Qwen2.5-3B-Instruct-GPTQ-Int4 \
  --served-model-name Qwen2.5-3B-Instruct-GPTQ-Int4 \
  --gpu-memory-utilization 0.9 \
  --max-num-batched-tokens 512
```
```
python -m vllm.entrypoints.openai.api_server \
  --model ~/.cache/modelscope/hub/models/Qwen/Qwen2-7B-Instruct-AWQ \
  --served-model-name Qwen2-7B-Instruct-AWQ \
  --gpu-memory-utilization 0.9 \
  --max-num-batched-tokens 512
```


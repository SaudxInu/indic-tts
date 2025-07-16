# * Install Modal.
# uv run pip install modal

# * Setup Modal.
# uv run python3 -m modal setup

# * Run to deploy the Modal app.
# uv run modal deploy scripts/modal/snorTTS_Indic_v0_vllm.py

# Import Modal.
import modal


# Define constants.
MODEL_NAME = "snorbyte/snorTTS-Indic-v0"
MAX_SEQ_LEN = 2048
MAX_CONCURRENT_SEQS = 5
APP_NAME = "snorTTS-Indic-v0-vllm-prod"
SCALEDOWN_WINDOW = 15 * 60
TIMEOUT = 10 * 60
VLLM_PORT = 8000
GPU = "A100-40GB"
MIN_CONTAINERS = 1
MAX_CONTAINERS = 1
MAX_CONCURRENT_REQUESTS = MAX_CONCURRENT_SEQS

# Define the Docker image.
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "vllm==0.9.1",  # Install vLLM for serving models.
        "huggingface_hub[hf_transfer]==0.32.0",  # Install Hugging Face transfer for fast model transfers.
        "flashinfer-python==0.2.6.post1",  # Install FlashInfer for optimized inference.
        extra_index_url="https://download.pytorch.org/whl/cu128",  # Use pytorch's extra index url for flashinfer.
    )
    .env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",  # Set `HF_HUB_ENABLE_HF_TRANSFER` for fast model transfers.
        }
    )
)

# Setup volumes for cache.
hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
vllm_cache_vol = modal.Volume.from_name("vllm-cache", create_if_missing=True)

# Create the Modal app.
app = modal.App(APP_NAME)


with image.imports():
    # Import necessary libraries for the remote function.
    import subprocess


# Define the function to start the VLLM server.
@app.function(
    image=image,  # Set the image for the function.
    gpu=GPU,  # Set the GPU type for the instance.
    scaledown_window=SCALEDOWN_WINDOW,  # Set how we long should we stay up with no requests.
    timeout=TIMEOUT,  # Set the timeout for the function.
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/vllm": vllm_cache_vol,
    },  # Set the volumes for cache.
    min_containers=MIN_CONTAINERS,  # Minimum number of containers to keep running.
    max_containers=MAX_CONTAINERS,  # Maximum number of containers to run.
)
@modal.concurrent(
    max_inputs=MAX_CONCURRENT_REQUESTS
)  # Limit the number of concurrent requests.
@modal.web_server(
    port=VLLM_PORT, startup_timeout=TIMEOUT
)  # Expose the VLLM server on the specified port.
def serve():
    # Create the command to start the VLLM server.
    cmd = [
        "vllm",
        "serve",
        "--uvicorn-log-level=info",
        MODEL_NAME,
        "--served-model-name",
        MODEL_NAME,
        "llm",
        "--max-model-len",
        str(MAX_SEQ_LEN),
        "--max-num-seqs",
        str(MAX_CONCURRENT_SEQS),
        "--host",
        "0.0.0.0",
        "--port",
        str(VLLM_PORT),
    ]

    # Start the VLLM server using subprocess.
    subprocess.Popen(" ".join(cmd), shell=True)

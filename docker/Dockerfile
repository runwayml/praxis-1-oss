# Example image of the praxis policy server: CUDA 13.0, PyTorch 2.13 and
# praxis with the [serve] and [kernels] extras.
#
# Build from the repository root:
#   docker build -f docker/Dockerfile -t praxis-serve .
#
# The image holds no weights. Mount the weights directory at /weights:
#   docker run --gpus all -p 8000:8000 -v /path/to/weights:/weights praxis-serve
#
# The weights directory holds praxis-1-droid/, Wan2.2_VAE.pth and
# Qwen3.5-2B/. The default command serves the single-arm checkpoint with
# configs/serve_droid.yaml. Its trunk uses the flash attention backend, which
# needs the flash-attn-4 package; the image installs it. The image has no git
# history, so give the commit for the handshake with
# -e PRAXIS_SERVED_COMMIT=$(git rev-parse HEAD).
#
# The Modal app (praxis/serve/modal_app.py) builds this image too.

# The devel image holds nvcc. pip compiles causal-conv1d with it, because
# there is no causal-conv1d wheel for PyTorch 2.13. The PyTorch wheel uses
# CUDA 13.0.
FROM nvidia/cuda:13.0.2-devel-ubuntu24.04

# DEBIAN_FRONTEND prevents the time zone prompt of tzdata.
RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        python3.12 python3.12-venv python3.12-dev \
    && rm -rf /var/lib/apt/lists/*

RUN python3.12 -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH

# These layers change rarely, so they come before the source code.
# transformers needs torchvision to load the processor of Qwen3.5.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu130 \
    torch==2.13.0 torchvision==0.28.0
# Without a GPU, nvcc compiles the kernels for all the GPU types that it knows.
# MAX_JOBS sets the number of parallel compile jobs (about 1 GB of memory each).
RUN pip install --no-cache-dir packaging ninja wheel \
    && CAUSAL_CONV1D_FORCE_BUILD=TRUE MAX_JOBS=8 pip install --no-cache-dir \
        --no-build-isolation causal-conv1d==1.6.1
# The single-arm checkpoint sets backbone.attention.backend to "flash".
# Flex attention then runs the CuTe kernel of flash-attn-4. Without this
# package, the first call of that checkpoint fails. The wheel is pure Python;
# the kernels compile at run time.
RUN pip install --no-cache-dir "flash-attn-4[cu13]==4.0.0b36"

WORKDIR /opt/praxis
COPY pyproject.toml README.md ./
COPY praxis praxis
COPY configs configs
# The pins are the versions of the parity runs. scipy sets the bits of the
# smoothed actions (smoothing=savgol), and scipy 1.15.3 needs numpy < 2.5.
# Pillow and av decode the JPEG and H.264 images of the requests.
RUN pip install --no-cache-dir ".[serve,kernels]" \
        scipy==1.15.3 numpy==2.2.6 pillow==10.4.0 av==18.0.0
# We made the validation results with cuDNN 9.24. PyTorch 2.13 asks for
# cuDNN 9.20, so this step must come after each install that resolves torch.
# The convolutions of the video autoencoder run in cuDNN.
RUN pip install --no-cache-dir --no-deps nvidia-cudnn-cu13==9.24.0.43 \
    && python -c "import torch; v = torch.backends.cudnn.version(); assert v == 92400, v"

# Remove the start banner of the CUDA base image.
ENTRYPOINT []
CMD ["python", "-m", "praxis.serve", "--config", "configs/serve_droid.yaml", \
     "--set", "model.checkpoint=/weights/praxis-1-droid", \
     "--set", "model.vae_weights=/weights/Wan2.2_VAE.pth", \
     "--set", "model.text_encoder=/weights/Qwen3.5-2B"]

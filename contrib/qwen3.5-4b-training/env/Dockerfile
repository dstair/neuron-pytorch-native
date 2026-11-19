# Adds the Python deps the native-PyTorch DLC does not ship.
#
# Two bases, selected by STACK in setup_box.sh / container.sh:
#   public (default)  the public-beta TorchNeuron DLC, Neuron SDK 2.33.0. torch lives in the
#                     system Python 3.14 and /etc/pip/constraints.txt pins torch==2.14.0.
#   beta6             the pre-GA build every number before 2026-10-21 was measured
#                     on (not public; reference via QWEN35_TRAIN_BETA6_IMAGE in .env).
#                     torch lives in /opt/torch-neuronx/.venv.
#
# Why a derived image rather than a venv on /mnt/nvme:
#   - `python3 -m venv --system-site-packages` off the container's torch venv inherits the
#     BASE interpreter's site-packages, not the venv's -- so torch disappears.
#   - `pip install --target` re-resolves dependencies and would drag in a second, CPU-only
#     torch wheel, silently shadowing torch-neuronx.
# Installing straight into the interpreter that already has torch lets pip see torch 2.14 as
# satisfied, and the image layer survives `docker run --rm`.
#
# Build:  ./env/setup_box.sh image        (tags qwen35-train:$STACK)
ARG BASE=public.ecr.aws/neuron/torch-neuronx@sha256:b4648437b8c7dbfa2acde18301a4bb4ae9658da5a412da75c59693ed4fd308ec
FROM ${BASE}

# The backend extension validates env and probes the device on import; neither is available
# during a build, so keep autoload off for any python pip runs in this stage.
ENV TORCH_DEVICE_BACKEND_AUTOLOAD=0

# The activate is beta6-only; the public image has no venv, and its constraints file keeps
# pip from replacing torch.
RUN { [ ! -f /opt/torch-neuronx/.venv/bin/activate ] || . /opt/torch-neuronx/.venv/bin/activate; } \
    && pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir \
        "transformers>=5.14" \
        "huggingface_hub>=0.26" \
        "kernels>=0.15.2" \
        datasets \
        safetensors \
        duckdb \
        sqlglot \
        pytest \
    && python3 -c "import torch, transformers, duckdb, datasets, kernels; \
print('torch', torch.__version__); \
print('transformers', transformers.__version__); \
print('duckdb', duckdb.__version__)"

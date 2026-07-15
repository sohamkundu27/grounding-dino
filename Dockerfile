FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel

ARG USE_CUDA=1
ARG TORCH_ARCH="9.0"

ENV AM_I_DOCKER=True
ENV BUILD_WITH_CUDA="${USE_CUDA}"
ENV TORCH_CUDA_ARCH_LIST="${TORCH_ARCH}"
ENV FORCE_CUDA=1

ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=/opt/conda/lib/python3.11/site-packages/torch/lib:${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

ENV CC=gcc-10
ENV CXX=g++-10

# Install OS dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    ffmpeg \
    libsm6 \
    libxext6 \
    git \
    nano \
    vim \
    ninja-build \
    gcc-10 \
    g++-10 \
    && apt-get clean \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

# Copy repository into image
RUN mkdir -p /home/appuser/Grounded-SAM-2
COPY . /home/appuser/Grounded-SAM-2/

WORKDIR /home/appuser/Grounded-SAM-2

# Git gets nervous about Docker ownership because apparently trust is scarce
RUN git config --global --add safe.directory /home/appuser/Grounded-SAM-2

# Install Python dependencies
RUN python -m pip install --upgrade \
    pip \
    "setuptools>=62.3.0,<75.9" \
    wheel \
    numpy \
    opencv-python \
    "transformers==4.40.2" \
    supervision \
    pycocotools \
    addict \
    yapf \
    timm

# Install SAM2 package
RUN python -m pip install --no-build-isolation -e .

# Clean and compile Grounding DINO CUDA extension
# setup.py must already contain:
# "-gencode=arch=compute_90,code=sm_90",
RUN cd /home/appuser/Grounded-SAM-2/grounding_dino \
    && rm -rf build dist *.egg-info \
    && rm -f groundingdino/_C*.so \
    && find . -name '*.o' -delete \
    && find . -name '*.a' -delete \
    && python setup.py build_ext --inplace --force -v \
    && python -m pip install --no-build-isolation --no-deps -e .

# Validate package versions and compiled CUDA extension
RUN python -c "import torch; \
print('Torch:', torch.__version__); \
print('Torch CUDA:', torch.version.cuda); \
print('CUDA available during build:', torch.cuda.is_available())"

RUN python -c "import transformers; \
assert transformers.__version__ == '4.40.2', transformers.__version__; \
print('Transformers:', transformers.__version__)"

RUN python -c "from groundingdino import _C; \
print('Grounding DINO CUDA extension imported successfully')"

# Fail the image build if H100 SM90 code is absent
RUN cuobjdump --list-elf \
    /home/appuser/Grounded-SAM-2/grounding_dino/groundingdino/_C*.so \
    | grep -q "sm_90"

WORKDIR /home/appuser/Grounded-SAM-2

CMD ["/bin/bash"]
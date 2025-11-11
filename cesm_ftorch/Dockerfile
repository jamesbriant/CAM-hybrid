FROM cesm:latest

# Install dependencies
#RUN pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu124
RUN pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
RUN ln -s /opt/miniconda/lib/python3.11/site-packages/torch /usr/local/libtorch

# Install build tools
RUN apt-get update && apt-get install -y \
    ccache \
    ninja-build \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Set up ccache
ENV PATH=/usr/lib/ccache:$PATH \
    CCACHE_DIR=/ccache \
    CCACHE_MAXSIZE=10G

# Install LibTorch
#RUN wget https://download.pytorch.org/libtorch/cu124/libtorch-cxx11-abi-shared-with-deps-2.5.1%2Bcu124.zip -O libtorch.zip && \
#    unzip libtorch.zip -d /usr/local && \
#    rm libtorch.zip

# Install FTorch
WORKDIR /opt
RUN git clone https://github.com/Cambridge-ICCS/FTorch.git /opt/FTorch
WORKDIR /opt/FTorch
RUN git checkout 162b6e165538ecd9649a67c8331a82ec1eb48f39
# This specific commit works with the cmake options below.
# Changing the FTorch version will likely require changing the cmake options.

WORKDIR /opt/FTorch/build
RUN cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_Fortran_COMPILER=gfortran \
    -DCMAKE_C_COMPILER=gcc \
    -DCMAKE_CXX_COMPILER=g++ \
    -DGPU_DEVICE=CUDA \
    -DMULTI_GPU=OFF \
    -DCMAKE_PREFIX_PATH=/usr/local/libtorch \
    -DCMAKE_INSTALL_PREFIX=/usr/local/ftorch

RUN make && make install

# Add To LD_PATH
ENV CUDA_HOME=/usr/local/cuda
ENV LD_LIBRARY_PATH=/usr/local/ftorch/lib:$LD_LIBRARY_PATH 
ENV LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/cuda/lib64"

ENV LD_LIBRARY_PATH=/usr/local/libtorch/lib:$LD_LIBRARY_PATH
ENV Torch_DIR=/usr/local/libtorch/share/cmake/Torch

# Copy Makefile
COPY files/Makefile /opt/cesm/cime/scripts/Tools/Makefile 

ENV PATH=/opt/cesm/cime/scripts:$PATH

WORKDIR /root

# Use the official NVIDIA PyTorch image as the base
FROM nvcr.io/nvidia/pytorch:21.12-py3

# Set the working directory
WORKDIR /srv/users/rudxia/Developer_NLP

# Install additional Python dependencies
RUN pip install --upgrade pip && \
    pip install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 --extra-index-url https://download.pytorch.org/whl/cu113

# Switch to root user for installing system-level dependencies
USER root

# Set DEBIAN_FRONTEND to noninteractive to avoid interactive prompts
ENV DEBIAN_FRONTEND noninteractive

# Install necessary build dependencies
RUN apt-get update && \
    apt-get install -y git build-essential cmake && \
    apt-get install -y cmake && \
    apt-get install -y libzmq3-dev && \
    rm -rf /var/lib/apt/lists/* && \
    apt-get clean

# Clone SentencePiece repository
RUN git clone https://github.com/google/sentencepiece.git /srv/users/rudxia/Developer_NLP/sentencepiece

# Switch to the SentencePiece directory
WORKDIR /srv/users/rudxia/Developer_NLP/sentencepiece

# Build SentencePiece
RUN mkdir build && \
    cd build && \
    cmake .. -DSPM_ENABLE_SHARED=OFF -DCMAKE_INSTALL_PREFIX=./root && \
    make install

# Cleanup unnecessary files
RUN apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Switch back to the non-root user (replace 'root' with the actual non-root user if it's not 'root')

# Navigate to the Python directory
WORKDIR /srv/users/rudxia/Developer_NLP/sentencepiece/python

# Build SentencePiece Python package
RUN python setup.py bdist_wheel

# Install the built SentencePiece Python package
RUN pip install /srv/users/rudxia/Developer_NLP/sentencepiece/python/dist/sentencepiece*.whl

# Install Jupyter Notebook and necessary dependencies
RUN pip install jupyter ipywidgets
RUN pip install accelerate -U


# Copy your application code into the container
WORKDIR /srv/users/rudxia/Developer_NLP
COPY notebooks/ ./notebooks/
COPY bert-spinner-ner-task/ ./bert-spinner-ner-task/

# Switch back to the non-root user (replace 'root' with the actual non-root user if it's not 'root')

# Set the entrypoint to keep the container running
# ENTRYPOINT ["tail", "-f", "/dev/null"]

# Set the CMD to start an interactive shell when attaching to the container
CMD ["/bin/bash", "-i"]

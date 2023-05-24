FROM df:latest

# Installing APEX
RUN git clone https://github.com/NVIDIA/apex
RUN sed -i 's/check_cuda_torch_binary_vs_bare_metal(torch.utils.cpp_extension.CUDA_HOME)/pass/g' apex/setup.py
RUN pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext"  ./apex

# Setting the working directory
WORKDIR /workspace

# Copying the required codebase
COPY . /workspace

ENV PYTHONPATH=.

CMD ["/bin/bash"]


FROM vllm/vllm-openai:v0.21.0

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    VLLM_NVFP4_GEMM_BACKEND=flashinfer-cutlass \
    VLLM_USE_FLASHINFER_SAMPLER=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

RUN pip install --no-cache-dir runpod==1.7.7 requests==2.32.3

# Patch TurboQuant attention backend to fix MTP drafter workspace allocation
COPY turboquant_patch.py /tmp/turboquant_patch.py
RUN python3 /tmp/turboquant_patch.py && rm /tmp/turboquant_patch.py

WORKDIR /worker
COPY handler.py /worker/handler.py

ENTRYPOINT []
CMD ["python3", "-u", "/worker/handler.py"]

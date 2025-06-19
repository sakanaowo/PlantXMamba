FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

RUN apt-get update && apt-get install -y python3 python3-pip git && apt-get clean

RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

WORKDIR /workspace

RUN git clone https://github.com/state-spaces/mamba.git
WORKDIR /workspace/mamba
RUN pip3 install -e .

COPY app/api.py /workspace/api.py
RUN pip3 install fastapi uvicorn

EXPOSE 8000

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]

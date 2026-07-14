FROM python:3.11-slim

WORKDIR /app

# System dependencies (matplotlib needs a backend)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY requirements-web.txt .
RUN pip install --no-cache-dir -r requirements-web.txt

# Pre-download ESM-2 from HuggingFace Hub (fast within HF infra)
RUN python -c "from transformers import EsmTokenizer, EsmForMaskedLM; \
    EsmTokenizer.from_pretrained('facebook/esm2_t6_8M_UR50D'); \
    EsmForMaskedLM.from_pretrained('facebook/esm2_t6_8M_UR50D')"

# Copy source code (includes model weights: best_model.pt, best_model_esm2.pt)
COPY . .

RUN mkdir -p static

ENV PORT=7860
EXPOSE 7860

CMD ["python", "server.py"]

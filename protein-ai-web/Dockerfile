FROM python:3.11-slim
WORKDIR /app
RUN pip install --no-cache-dir torch>=2.0.0 --extra-index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENV HF_ENDPOINT=https://hf-mirror.com
ENV PYTHONIOENCODING=utf-8
ENV HOST=0.0.0.0
EXPOSE 8765
CMD ["python", "server.py"]

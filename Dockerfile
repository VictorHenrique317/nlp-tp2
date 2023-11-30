from python:latest

RUN pip install --upgrade pip
RUN pip install torch torchtext transformers numpy datasets

COPY main.py /app

WORKDIR /app

CMD ["python3", "main.py"]
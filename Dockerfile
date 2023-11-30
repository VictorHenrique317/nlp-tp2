from python:latest

RUN pip install --upgrade pip
RUN pip install torch torchtext transformers numpy datasets

WORKDIR /app

COPY main.py /app

CMD ["python3", "main.py"]
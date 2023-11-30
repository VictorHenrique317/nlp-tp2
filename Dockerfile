from python:latest

RUN pip install --upgrade pip
RUN pip install torch transformers numpy datasets
RUN pip install torchtext == 0.6.0

WORKDIR /app

COPY main.py /app

CMD ["python3", "main.py"]
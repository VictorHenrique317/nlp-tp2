from python:latest

RUN pip install --upgrade pip
RUN pip install torch torchtext transformers numpy datasets

CMD ["python", "main.py"]
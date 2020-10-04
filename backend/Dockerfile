FROM tiangolo/uvicorn-gunicorn:python3.7

COPY . /app

RUN pip install --no-cache-dir fastapi
RUN pip install --upgrade pip setuptools
RUN pip install -r requirements.txt
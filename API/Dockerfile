FROM python:3.10

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN apt-get update
RUN apt-get -y install redis-server
RUN service redis-server start

COPY . /FastAPI
WORKDIR /FastAPI

EXPOSE 8000

CMD ["python", "app.py"]
FROM python:3.10

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . /FastAPI/API
WORKDIR /FastAPI/API

EXPOSE 8000

CMD ["python", "app.py"]
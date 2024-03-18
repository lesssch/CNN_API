FROM python:3.10
LABEL authors="aleksey"

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . /SGD_API
WORKDIR /SGD_API

EXPOSE 8000

CMD ["python", "app.py"]
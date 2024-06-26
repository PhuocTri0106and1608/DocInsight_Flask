FROM python:3.11-slim-bullseye

RUN apt-get update
RUN apt-get install ffmpeg python3-pip libsm6 libopencv-dev git libxext6 libpq-dev gcc -y
RUN python3.11 -m pip install --upgrade pip
COPY requirements.txt .

RUN pip3 install -r requirements.txt && apt-get clean

COPY ./MedicalXAIAPI ./app
WORKDIR /app 


CMD ["flask", "run", "--host", "0.0.0.0"]
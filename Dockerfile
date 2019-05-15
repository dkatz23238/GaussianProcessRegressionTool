FROM python:3.7-stretch

LABEL Name=GPR-tuner Version=0.0.1

WORKDIR /app
ADD . /app

RUN python3 -m pip install -r requirements.txt
CMD ["python3", "-u", "ax-service-loop.py"]

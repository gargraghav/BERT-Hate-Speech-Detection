FROM python:3.6-slim-stretch

RUN apt update
RUN apt install -y python3-dev gcc

ADD requirements.txt /
# ADD requirements.txt requirements.txt
# ADD bert_pytorch_e_1.bin bert_pytorch_e_1.bin
# ADD app.py app.py

WORKDIR /
RUN pip install -r requirements.txt
# Install required libraries
# RUN pip install -r requirements.txt

ADD . /
# Run it once to trigger resnet download
RUN python app.py

EXPOSE 8008

# Start the server
CMD ["python", "app.py", "serve"]
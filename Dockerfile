FROM python:3.6

RUN pip install -U pip

RUN pip install numpy opencv-python

RUN pip install flask

COPY morph_server.py .
COPY morph.py .

CMD [ "python", "morph_server.py" ]
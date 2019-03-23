FROM tensorflow/tensorflow:latest-py3

WORKDIR /usr/src/app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY *.py ./
COPY model model

EXPOSE 5000

CMD [ "python", "./app.py" ]


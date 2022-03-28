FROM python:3.8-slim-buster


ADD ./ ./mercury-ml-core

RUN pip install Flask
RUN pip install numpy
RUN pip install tensorflow
RUN pip install pandas
RUN pip install pika --upgrade
RUN pip install matplotlib
RUN pip install sklearn

EXPOSE 9999
WORKDIR /mercury-ml-core
CMD [ "python", "/mercury-ml-core/api.py" ]
CMD [ "python", "/mercury-ml-core/ml_init.py" ]

FROM python:3.10

RUN pip install --upgrade pip setuptools
RUN pip install "xinference[all]" --upgrade

EXPOSE 9997

CMD ["xinference", "-p", "9997", "-H", "0.0.0.0"]

FROM python:3.10

RUN pip install --upgrade pip setuptools -i https://pypi.tuna.tsinghua.edu.cn/simple/
RUN pip install "xinference[all]" --upgrade -i https://pypi.tuna.tsinghua.edu.cn/simple/

EXPOSE 9997

CMD ["xinference", "-p", "9997", "-H", "0.0.0.0"]

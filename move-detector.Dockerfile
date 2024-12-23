FROM python:3.8-slim

COPY ./common /home/work/common

RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r /home/work/common/requirements.txt
RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple opencv-python-headless pillow kafka-python

COPY ./move-detector /home/work/move-detector
WORKDIR /home/work/move-detector

ENV PYTHONPATH "${PYTHONPATH}:/home/work"
ENV LOG_LEVEL="INFO"

ENTRYPOINT ["python"]
CMD ["move-detector.py"]
FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime
COPY ./common /home/work/common

RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r /home/work/common/requirements.txt
RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple opencv-python-headless pillow kafka-python
RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple torch torchvision

COPY ./feature-matcher-batch /home/work/feature-matcher-batch
RUN mkdir -p /home/data
ENV PYTHONPATH "${PYTHONPATH}:/home/work"
ENV LOG_LEVEL="INFO"
WORKDIR /home/work/feature-matcher-batch

ENTRYPOINT ["python"]
CMD ["feature-matcher-batch.py"]

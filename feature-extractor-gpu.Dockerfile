FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime
COPY ./common /home/work/common

RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r /home/work/common/requirements.txt
RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple opencv-python-headless pillow kafka-python
RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple torch torchvision

COPY ./feature-extractor /home/work/feature-extractor
WORKDIR /home/work/feature-extractor
ENV PYTHONPATH "${PYTHONPATH}:/home/work"
ENV LOG_LEVEL="INFO"

ENTRYPOINT ["python"]
CMD ["feature-extractor.py"]

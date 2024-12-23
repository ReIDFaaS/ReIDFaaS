FROM python:3.11-slim
COPY ./common /home/work/common

RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r /home/work/common/requirements.txt
RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple opencv-python-headless pillow kafka-python
RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple torch torchvision
RUN rm -rf /etc/apt/sources.list.d/debian.sources && \
    echo "deb https://mirrors.tuna.tsinghua.edu.cn/debian/ bookworm main contrib non-free non-free-firmware" > /etc/apt/sources.list \
    && apt update \
    && apt install -y build-essential \
    && pip install -i https://pypi.tuna.tsinghua.edu.cn/simple lapx scipy cython_bbox \
    && apt-get purge -y build-essential \
    && apt -y autoremove

ENV PYTHONPATH "${PYTHONPATH}:/home/work"
COPY ./target-detector  /home/work/target-detector
WORKDIR /home/work/target-detector

ENV LOG_LEVEL="INFO"

ENTRYPOINT ["python"]
CMD ["target-detector.py"]

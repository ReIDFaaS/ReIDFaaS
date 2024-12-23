FROM --platform=$TARGETPLATFORM python:3.8-slim
COPY ./common /home/work/common
RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r /home/work/common/requirements.txt

COPY ./front-end /home/work/front-end
WORKDIR /home/work/front-end

ENV PYTHONPATH "${PYTHONPATH}:/home/work"

ENTRYPOINT ["python"]
CMD ["front-end.py"]
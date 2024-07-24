FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-devel

RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub \
    && apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub

RUN apt-get update && apt-get install -y libgl1 libglib2.0-0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

ENV PROJECTDIR=/opt/app
WORKDIR $PROJECTDIR
COPY inference.py $PROJECTDIR
COPY config.py $PROJECTDIR
COPY checkpoint.pth $PROJECTDIR
COPY mmsegmentation-1.2.2rc1-py3-none-any.whl $PROJECTDIR

RUN pip3 install mmcv==2.2.0 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.3/index.html
RUN pip3 install mmsegmentation-1.2.2rc1-py3-none-any.whl
RUN pip3 install ftfy
RUN pip3 install regex

LABEL maintainer="vibe <vibe.research@outlook.com>"
ENTRYPOINT ["python3", "inference.py"]

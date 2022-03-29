
FROM python:3.8

WORKDIR /
ADD trainer /trainer

RUN wget https://storage.googleapis.com/cloud-tpu-tpuvm-artifacts/libtpu/20210525/libtpu.so -O /lib/libtpu.so
RUN chmod 777 /lib/libtpu.so

RUN wget https://storage.googleapis.com/cloud-tpu-tpuvm-artifacts/tensorflow/20210525/tf_nightly-2.6.0-cp38-cp38-linux_x86_64.whl
RUN pip3 install tf_nightly-2.6.0-cp38-cp38-linux_x86_64.whl
RUN rm tf_nightly-2.6.0-cp38-cp38-linux_x86_64.whl


ENTRYPOINT ["python3", "-m", "trainer.task"]

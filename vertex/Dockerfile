
FROM us-docker.pkg.dev/vertex-ai/training/tf-tpu.2-5:latest
WORKDIR /

RUN pip3 install tf-models-official==2.5.1 tensorflow-recommenders==0.6.0

ENTRYPOINT ["python3"]
CMD ["-c", "import tensorflow_recommenders as tfrs; print(tfrs.__version__)"]

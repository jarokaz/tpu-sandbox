
FROM us-docker.pkg.dev/vertex-ai/training/tf-tpu.2-5:latest
WORKDIR /

#RUN pip3 install tf-models-official==2.5.1 tensorflow-recommenders==0.6.0

ADD snippets/hello-world.py .

ENTRYPOINT ["python3"]
CMD ["hello-world.py"]

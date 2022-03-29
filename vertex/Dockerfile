
FROM us-docker.pkg.dev/vertex-ai/training/tf-tpu.2-8

WORKDIR /
ADD trainer /trainer

ENTRYPOINT ["python3", "-m", "trainer.task"]

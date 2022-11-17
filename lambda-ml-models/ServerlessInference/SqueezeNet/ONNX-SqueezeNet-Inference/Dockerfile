FROM public.ecr.aws/docker/library/python:buster

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y

COPY hello_world/app.py hello_world/requirements.txt ./
COPY hello_world/squeezenet1.0-12.onnx ./
COPY hello_world/synset.txt ./
COPY hello_world/imagenet_classes.txt ./

COPY ./entry_script.sh /entry_script.sh
ADD aws-lambda-rie /usr/local/bin/aws-lambda-rie

RUN pip3 install -r requirements.txt -t ./
RUN pip3 install awslambdaric

WORKDIR "/"

# Command can be overwritten by providing a different command in the template directly.
ENTRYPOINT [ "/entry_script.sh" ]
CMD ["app.lambda_handler"]
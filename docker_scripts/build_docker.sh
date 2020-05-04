#!/bin/bash

CPU_PARENT=tensorflow/tensorflow:1.15.2-py3-jupyter
GPU_PARENT=tensorflow/tensorflow:1.15.2-gpu-py3-jupyter

TAG=nerual_moore_machines

if [[ ${USE_GPU} == "True" ]]; then
  PARENT=${GPU_PARENT}
else
  PARENT=${CPU_PARENT}
  TAG="${TAG}-cpu"
fi

# this is the directory to link up to the home directory of the 
# docker user
CODE_DIR=NeuralMooreMachine_Experiments

# this will be the name of the user exposed in the docker container
DOCKER_USER_NAME="ferg"

# build such that the container user is the same as the host user
docker build --build-arg PARENT_IMAGE=${PARENT} \
  --build-arg USE_GPU=${USE_GPU} \
  --build-arg HOST_USER_ID=$(id -u ${USER}) \
  --build-arg HOST_GROUP_ID=$(id -g ${USER}) \
  --build-arg USER_NAME=${DOCKER_USER_NAME} \
  --build-arg CODE_DIR=${CODE_DIR} \
  -t ${TAG} \
  .
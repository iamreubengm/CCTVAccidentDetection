FROM nvidia/cuda:11.5.1-cudnn8-devel-ubuntu20.04

RUN apt-get update -y && apt-get upgrade -y
RUN apt-get install libopencv-dev python3 python3-pip libsm6 git make

RUN pip3 install opencv-python, tensorflow-gpu, firebase-admin

WORKDIR /workspace
RUN git clone https://github.com/AlexeyAB/darknet 
WORKDIR /workspace/darknet
RUN make -j 12 GPU=1 CUDNN=1 OPENCV=1 LIBSO=1

WORKDIR /workspace
RUN git clone https://github.com/jeffreypaul15/FinalYearProject
RUN /workspace/darknet/libdarknet.so FinalYearProject/back_end/server_code
CMD /bin/bash

# This image creates a basic setup for Keras with DL4J as a backend
#
# Based on the https://deeplearning4j.org/buildinglocally guide

FROM maven:alpine
MAINTAINER pkoperek@gmail.com

ENV LIBND4J /libnd4j

RUN echo http://dl-cdn.alpinelinux.org/alpine/edge/community >> /etc/apk/repositories
RUN apk add --update git cmake gcc make g++ py-pip python-dev openblas openblas-dev

# Upgrade pip
RUN pip install --upgrade pip

# jupyter
RUN pip install jupyter

# libnd4j
RUN git clone --depth 1 https://github.com/deeplearning4j/libnd4j.git
RUN cd libnd4j && ./buildnativeoperations.sh && cd ..

# nd4j
RUN git clone --depth 1 https://github.com/deeplearning4j/nd4j.git
RUN cd nd4j && mvn clean install -DskipTests -Dmaven.javadoc.skip=true -pl '!:nd4j-cuda-8.0,!:nd4j-cuda-8.0-platform,!:nd4j-tests' && cd ..

# datavec
RUN git clone --depth 1 https://github.com/deeplearning4j/datavec.git
RUN cd datavec && mvn clean install -DskipTests -Dmaven.javadoc.skip=true && cd ..

# test resources
RUN git clone --depth 1 https://github.com/deeplearning4j/dl4j-test-resources.git
RUN cd dl4j-test-resources && mvn clean install && cd ..

# deeplearning4j
RUN git clone --depth 1 https://github.com/deeplearning4j/deeplearning4j.git
RUN cd deeplearning4j && mvn clean install -DskipTests -Dmaven.javadoc.skip=true -pl '!:deeplearning4j-cuda-8.0' && cd ..

# Hack: http://serverfault.com/questions/771211/docker-alpine-and-matplotlib
RUN ln -s /usr/include/locale.h /usr/include/xlocale.h

# Install keras
RUN pip install keras

#
# keras-dl4j stuff
#

# Install keras-dl4j
RUN mkdir /keras-dl4j
ADD .. /keras-dl4j
RUN cd keras-dl4j && python setup.py install && cd ..

EXPOSE 8888

CMD ["jupyter", "notebook", "--port", "8888"]
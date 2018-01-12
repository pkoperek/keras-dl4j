# This image creates a basic setup for Keras with DL4J as a backend
#
# Based on the https://deeplearning4j.org/buildinglocally guide

FROM maven:alpine
MAINTAINER pkoperek@gmail.com

ENV LIBND4J /libnd4j

RUN echo http://dl-cdn.alpinelinux.org/alpine/edge/community >> /etc/apk/repositories
RUN echo http://dl-cdn.alpinelinux.org/alpine/edge/testing >> /etc/apk/repositories
RUN apk add --update git cmake gcc make g++ py-pip python-dev openblas openblas-dev hdf5 hdf5-dev

# Upgrade pip
RUN pip install --upgrade pip

# jupyter & keras
RUN pip install jupyter

# Hack: http://serverfault.com/questions/771211/docker-alpine-and-matplotlib
RUN ln -s /usr/include/locale.h /usr/include/xlocale.h

RUN pip install keras

# libnd4j
RUN git clone --depth 1 https://github.com/deeplearning4j/libnd4j.git
RUN cd libnd4j && ./buildnativeoperations.sh && cd ..

# nd4j
RUN git clone --depth 1 https://github.com/deeplearning4j/nd4j.git
RUN cd nd4j && mvn --settings /usr/share/maven/ref/settings-docker.xml clean install -DskipTests -Dmaven.javadoc.skip=true -pl '!:nd4j-cuda-8.0,!:nd4j-cuda-8.0-platform,!:nd4j-tests' && cd ..

# datavec
RUN git clone --depth 1 https://github.com/deeplearning4j/datavec.git
RUN cd datavec && mvn --settings /usr/share/maven/ref/settings-docker.xml clean install -DskipTests -Dmaven.javadoc.skip=true && cd ..

# test resources
RUN git clone --depth 1 https://github.com/deeplearning4j/dl4j-test-resources.git
RUN cd dl4j-test-resources && mvn --settings /usr/share/maven/ref/settings-docker.xml  clean install && cd ..

# deeplearning4j
RUN git clone --depth 1 https://github.com/deeplearning4j/deeplearning4j.git
RUN cd deeplearning4j && mvn --settings /usr/share/maven/ref/settings-docker.xml clean install -DskipTests -Dmaven.javadoc.skip=true -pl '!:deeplearning4j-cuda-8.0' && cd ..
RUN cd deeplearning4j/deeplearning4j-keras && mvn --settings /usr/share/maven/ref/settings-docker.xml clean package -Pserver-jar && cd ../..
RUN cp deeplearning4j/deeplearning4j-keras/target/deeplearning4j-keras-*-SNAPSHOT.jar /

#
# keras-dl4j stuff
#

# Install keras-dl4j
RUN pip install h5py py4j xxhash
RUN mkdir /keras-dl4j /root/.keras
COPY ./keras.json /root/.keras
ADD . /keras-dl4j
RUN cd keras-dl4j && python setup.py sdist && pip install dist/kerasdl4j*tar.gz && cd ..

EXPOSE 8888
ENV KERAS_BACKEND theano

# adding `sh -c` solves the problem of dying kernels:
# https://github.com/ipython/ipython/issues/7062
CMD ["sh", "keras-dl4j/jupyter_start.sh"]

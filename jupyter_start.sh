#!/bin/sh

java -cp /deeplearning4j/deeplearning4j-keras/target/deeplearning4j-keras-0.7.3-SNAPSHOT.jar org.deeplearning4j.keras.Server >& /server.log &
jupyter notebook --no-browser --ip=0.0.0.0 --port=8888 --notebook-dir=/keras-dl4j/examples

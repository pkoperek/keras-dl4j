#!/bin/sh

java -cp /deeplearning4j-keras-0.7.3-SNAPSHOT.jar >& /server.log &
jupyter notebook --no-browser --ip=0.0.0.0 --port=8888 --notebook-dir=/keras-dl4j/examples

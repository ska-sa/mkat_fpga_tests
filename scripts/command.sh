#!/bin/bash
export PORT=$(kcpcmd array-list | grep array-list | cut -f3 -d' ' | cut -f1 -d'$
export SENSORPORT=$(kcpcmd array-list | grep array-list | cut -f3 -d' ' | cut -$

kcpcmd -s localhost:$PORT sensor-value | grep map
#echo Hello World

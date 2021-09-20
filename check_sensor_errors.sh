#!/bin/bash
PORT=$(kcpcmd array-list | grep array-list | cut -f3 -d' ' | cut -f1 -d',')
SENSORPORT=$(kcpcmd array-list | grep array-list | cut -f3 -d' ' | cut -f2 -d',')

if [ -z "$PORT" ]
then
      echo "PORT NOT FOUND"
else
      VAR=$(kcpcmd -s localhost:$PORT sensor-value | grep error)
      if [ -z "$VAR" ]
      then
            echo "NO_ERRORS"
      else
            echo "ERRORS: $VAR"
      fi
fi

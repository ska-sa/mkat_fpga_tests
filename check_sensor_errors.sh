#!/bin/bash
PORT=$(kcpcmd array-list | grep array-list | cut -f3 -d' ' | cut -f1 -d',') # stores PORT number
SENSORPORT=$(kcpcmd array-list | grep array-list | cut -f3 -d' ' | cut -f2 -d',') # stores SENSOR PORT number

if [ -z "$PORT" ] # PORT number will be null if an instrument is not running
then
      echo "PORT NOT FOUND" # if PORT number is null then echo PORT NOT FOUND
else
      VAR1=$(kcpcmd -s localhost:$PORT sensor-value | grep error) # stores any sensor-value errors from PORT
      VAR2=$(kcpcmd -s localhost:$SENSORPORT sensor-value | grep error) # stores any sensor-value errors from SENSORPORT
      if [ -z "$VAR1" ] && [ -z "$VAR2" ]
      then
            echo "NO_ERRORS" # if no errors from PORT and SENSORPORT, echo NO_ERRORS
      else
            echo "ERRORS on PORT: $VAR1" # else echo errors stored in VAR1
            echo "ERRORS on SENSORPORT: $VAR2" # else echo errors stored in VAR2
      fi
fi

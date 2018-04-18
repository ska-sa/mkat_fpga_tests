#!/bin/bash

# if [ -z "$*" ];
#     then echo -e "Usage: bash $0 HOST\n
#     HOST: IP (Default:localhost)\n";
#     exit 1;
# fi


HOST=${1:-localhost}
echo $HOST
kcpcmd -t 180 -s ${HOST}:$(kcpcmd -s ${HOST}:7147 array-list array0 | grep -a array-list | cut -f3 -d ' ' | cut -d',' -f 2) sensor-value
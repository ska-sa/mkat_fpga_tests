#!/bin/bash
# Author: Mpho Mphego <mmphego@ska.ac.za>
# Install digitiser simulator timing dependencies

if [ $EUID -ne 0 ]; then
    echo "$0 is not running as root. Run as root."
    exit 2
fi

ln -s pseudo-dmc-child /usr/local/bin
ln -s start-pseudo-dmc /usr/local/bin
ln -s stop-pseudo-dmc /usr/local/bin
echo -e '
# location of DMC
dmc_address=localhost:9011' | sudo tee -a /etc/cmc.conf

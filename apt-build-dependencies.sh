#!/bin/bash
# System package installation dependencies for mkat_fpga_tests
# Mpho Mphego <mmphego@ska.ac.za>

if [ "$EUID" -ne 0 ]
  then echo "Please run as root"
  exit
fi

list_packages=$"autoconf \
smcroute\
automake \
bison \
flex \
libboost-all-dev \
libfreetype6-dev \
libglib2.0-dev \
libgtk2.0-dev \
libhdf5-dev \
libpcap-dev \
libpcap0.8-dev \
libpng12-dev \
pkg-config \
python-dev \
tk-dev \
wireshark-dev \
gfortran \
swig \
dialog \
libopenblas-dev \
libssl-dev \
libffi-dev \
build-essential \
gfortran \
libatlas-base-dev \
python-scipy"

apt-get update -qq
for pkg in $list_packages;
    do echo 'Installing '$pkg;
    apt-get install -q --force-yes -y $pkg;
    done

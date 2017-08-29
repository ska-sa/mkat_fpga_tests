#!/bin/bash
# System package installation dependencies for mkat_fpga_tests
# Mpho Mphego <mmphego@ska.ac.za>

if [ "$EUID" -ne 0 ]
  then echo "Please run as root"
  exit
fi

list_packages=$"autoconf \
automake \
bison \
build-essential \
colordiff \
curl \
dialog \
flex \
gfortran \
git \
iptables-persistent \
libatlas-base-dev \
libboost-all-dev \
libffi-dev \
libfreetype6-dev \
libglib2.0-dev \
libgtk2.0-dev \
libhdf5-dev \
libopenblas-dev \
libpcap-dev \
libpcap0.8-dev \
libpng12-dev \
libssl-dev \
oracle-java8-installer \
pkg-config \
python-dev \
python-scipy \
smcroute \
swig \
tk-dev \
wireshark-dev"

add-apt-repository ppa:webupd8team/java -y
apt-get update -qq
for pkg in $list_packages;
    do echo 'Installing '$pkg;
    apt-get install -qq -y $pkg;
    done

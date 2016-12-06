mkat_fpga_tests
=========

Digitiser Simulator Setup

* sudo cp pseudo-dmc-child start-pseudo-dmc stop-pseudo-dmc /usr/local/bin
* echo '# location of DMC' >> /etc/cmc.conf
  echo 'dmc_address=localhost:9011' >> /etc/cmc.conf
  Where localhost is the location where you will initialise the CBF on.
* Usage: start-pseudo-dmc dsim_roach, 
* To halt run: stop-pseudo-dmc

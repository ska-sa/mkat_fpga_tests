mkat_fpga_tests
=========

Digitiser Simulator Setup

* sudo ln -s $(pwd)/pseudo-dmc-child /usr/local/bin
* sudo ln -s $(pwd)/start-pseudo-dmc /usr/local/bin
* sudo ln -s $(pwd)/stop-pseudo-dmc /usr/local/bin
* echo '# location of DMC' >> /etc/cmc.conf
  echo 'dmc_address=localhost:9011' >> /etc/cmc.conf
  Where localhost is the location where you will initialise the CBF on.
* Usage: start-pseudo-dmc dsim_roach,
* To halt run: stop-pseudo-dmc

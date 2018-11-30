#!/bin/bash
# Manual delays application
# Assumes that CMC is configures and Corr2 are installed globally.
# Mpho Mphego (mmphego@ska.ac.za)

set -e
bold=$(tput bold)
normal=$(tput sgr0)

export tapply=$1
export bls=$2
export CONFIG=$3

function clearall {
    printf "${bold}Clearing all delays in 30 seconds.${normal}\n\n";
	kcpcmd -t 500 -s localhost:$(kcpcmd subordinate-list | grep -a subordinate-list | cut -f3 -d ' ' | cut -f1 -d ',') delays \
	$(python -c 'import time, os; print time.time() + 30') '0,0:0,0', '0,0:0,0', '0,0:0,0', '0,0:0,0', '0,0:0,0', '0,0:0,0', '0,0:0,0', '0,0:0,0' | grep -a 'delays'
	$(which kcpcmd) -t 30 -s localhost:$($(which kcpcmd) subordinate-list | grep -a subordinate-list | cut -f3 -d ' ' | cut -f1 -d ',') \
	capture-stop baseline-correlation-products | grep -a 'capture';
}

if [ -z "$*" ];
	then printf "${bold}Usage: $0 {time to apply in seconds} {baseline} {config}, else delays are cleared by Default${normal}\n\n";
	clearall
    exit 1;
fi

function configure_dsim {
	printf "${bold}Setting FFT-Shift to 511.${normal}\n";
	$(which kcpcmd) -t 500 -s localhost:$($(which kcpcmd) subordinate-list | grep -a subordinate-list | cut -f3 -d ' ' | cut -f1 -d ',') fft-shift 511 | grep -a 'fft-shift'
	printf "${bold}Setting gain on all inputs to 113.${normal}\n";
	$(which kcpcmd) -t 500 -s localhost:$($(which kcpcmd) subordinate-list | grep -a subordinate-list | cut -f3 -d ' ' | cut -f1 -d ',') gain-all 113 | grep -a 'gain-all'
	printf "${bold}Setting Dsim to generate correlated Noise.${normal}\n";
	corr2_dsim_control.py --config "${CONFIG}" --output-type 0 signal --output-scale 0 1 \
	--output-scale 1 1 --output-type 1 signal --noise-source corr 0.0645;
}

function start_capture {
	printf "${bold}Start capturing data on port 8888.${normal}\n\n";
	$(which kcpcmd) -t 30 -s localhost:$($(which kcpcmd) subordinate-list | grep -a subordinate-list | cut -f3 -d ' ' | cut -f1 -d ',') \
	capture-start baseline-correlation-products
}

function stop_capture {
	printf "\n\n\n"
	printf "${bold}Data capture stopped and dsim cleared.${normal}\n\n";
	corr2_dsim_control.py --zeros-sine --zeros-noise > /dev/null 2>&1;
	clearall
}


function start_rx {
	printf "${bold}Receiving Baseline: ${bls} data\n\n ${normal}";
	corr2_rx.py --config "${CONFIG}" --plot --ion --baseline "${bls}" &
}

# trap ctrl-c and call exitting_code()
trap exitting_code INT

function exitting_code {
	pkill -9 corr2_rx.py;
	trap stop_capture EXIT;
	exit 1;
}

function list_baselines {
	printf "${bold}Baseline list and input labels\n${normal}\n" $tapply;
	$(which kcpcmd) -t 500 -s localhost:$($(which kcpcmd) subordinate-list | grep -a subordinate-list | cut -f3 -d ' ' | cut -f1 -d ',') input-labels | grep -a 'input-labels'
	$(which kcpcmd) -t 500 -s localhost:$($(which kcpcmd) subordinate-list | grep -a subordinate-list | cut -f3 -d ' ' | cut -f1 -d ',') sensor-value \
	baseline-correlation-products-bls-ordering  | grep -a 'baseline-correlation-products-bls-ordering'
	printf "\n\n"
}
function delays_execution {
	epoch=$(python -c 'import time, os; print time.time() + float(os.environ["tapply"])')
	epoch=$(($(date '+%s') + $tapply))
	printf "${bold}Delays to be applied at epoch + %s seconds${normal}\n" ${tapply};
	printf "${bold}We expect a change phase${normal}\n";
	printf "${bold}Now time: %s${normal}\n" $(python -c 'import time, os; print time.time()');
	printf "${bold}Trying to load at: %s ${normal}\n\n" $(python -c 'import time, os; print time.time() + float(os.environ["tapply"])');

	$(which kcpcmd) -t 500 -s localhost:$($(which kcpcmd) subordinate-list | grep -a subordinate-list | cut -f3 -d ' ' | cut -f1 -d ',' ) delays \
	$(python -c 'import time, os; print time.time() + float(os.environ["tapply"])') \
	'0,0:0,0', '0,0:0,0', '5.83331194452e-10,0:0,0', '0,0:0,0', '0,0:0,0', '0,0:0,0', '0,0:0,0', '0,0:0,0'  | grep -a 'delays';
	#'0,0:0,0', '0,0:0,0', '5.83331194452e-10,0:0,0', '0,0:0,0', '0,0:0,0', '0,0:0,0', '0,0:0,0', '0,0:0,0';
	#$(which kcpcmd) -t 500 -s localhost:$($(which kcpcmd) subordinate-list | grep -a subordinate-list | cut -f3 -d ' ' ) delays \
	#$epoch '0,0:0,0', '0,0:0,0', '0,0:0,0', '0,0:0,0', '0,0:0,0', '0,0:0,0', '0,0:0,0', '0,0:0,0'0
	#$(which kcpcmd) -t 500 -s localhost:$($(which kcpcmd) subordinate-list | grep -a subordinate-list | cut -f3 -d ' ' ) delays \
	#`python -c 'import time; print time.time() + 25'` '0,0:0,0', '0.5,0:0,0', '0,0:0,0', '0,0:0,0', '0,0:0,0', '0,0:0,0', '0,0:0,0', '0,0:0,0'
	sleep 1;
	start_rx;
}

list_baselines
configure_dsim
start_capture
delays_execution

read -n1 -r -p "Press space to exit..." key
if [ "$key" = '' ]; then
	exitting_code
fi

#!/bin/bash
# Script that runs nosetests and initiate an instrument
# Mpho Mphego <mmphego@ska.ac.za>

printf 'Enter username> '
read -r Username
printf 'Enter IP or server name, eg dbelab04> '
read -r Server_name
printf 'What instrument do you want to run? Enter full name> '
read -r Instrument


printf '[Y/y] Do you want to run nosetests %s?'$Instrument
if read -t 10 input ; then
    if [ $input == 'Y' -o $input == 'y' ] ;then
    prinf 'Initialising instrument:' $ $Instrument
    ssh -X $Username@$Server_name './instrument_activate_'$Instrument

    printf 'Running nosetests %s\n' $Instrument
    ModeInst=${Instrument##*M}
    ModAlwd='4k'


        timeout 30m \
            nosetests  -v --logging-level=FATAL --with-katreport mkat_fpga_tests/test_cbf.py:test_CBF.\
                test__generic_config_report 2>&1 | tee test__generic_config_report.txt;

        timeout 30m \
             nosetests  -v --logging-level=FATAL --with-katreport mkat_fpga_tests/test_cbf.py:test_CBF.\
                test__generic_control_init  2>&1 | tee test__generic_control_init.txt;

        timeout 30m \
             nosetests  -v --logging-level=FATAL --with-katreport mkat_fpga_tests/test_cbf.py:test_CBF.\
                test__generic_fault_detection 2>&1 | tee test__generic_fault_detection.txt;

        timeout 30m \
             nosetests  -v --logging-level=FATAL --with-katreport mkat_fpga_tests/test_cbf.py:test_CBF.\
                test__generic_sensor_values 2>&1 | tee test__generic_sensor_values.txt

        timeout 30m \
             nosetests  -v --logging-level=FATAL --with-katreport mkat_fpga_tests/test_cbf.py:test_CBF.\
                test__generic_time_sync 2>&1 | tee test__generic_time_sync.txt;

        timeout 30m \
             nosetests  -v --logging-level=FATAL --with-katreport mkat_fpga_tests/test_cbf.py:test_CBF.\
                test__generic_small_voltage_buffer 2>&1 | tee test__generic_small_voltage_buffer.txt;

        timeout 30m \
             nosetests  -v --logging-level=FATAL --with-katreport mkat_fpga_tests/test_cbf.py:test_CBF.\
                test_`echo $Instrument`_accumulation_length 2>&1 | tee test_`echo $Instrument`_accumulation_length.txt;

        ssh -X $Username@$Server_name './instrument_activate_'$Instrument;

        timeout 30m \
             nosetests  -v --logging-level=FATAL --with-katreport mkat_fpga_tests/test_cbf.py:test_CBF.\
                test_`echo $Instrument`_baseline_correlation_product 2>&1 | tee test_`echo $Instrument`_baseline_correlation_product;

        timeout 30m \
             nosetests  -v --logging-level=FATAL --with-katreport mkat_fpga_tests/test_cbf.py:test_CBF.\
                test_`echo $Instrument`_baseline_correlation_product_consistency 2>&1 | tee test_`echo $Instrument`_baseline_correlation_product_consistency;

        if [ $ModeInst == $ModAlwd ]
            then
            printf 'Running Beamforming tests\n'


            timeout 30m \
                 nosetests  -v --logging-level=FATAL --with-katreport mkat_fpga_tests/test_cbf.py:test_CBF.\
                    test_`echo $Instrument`_beamforming 2>&1 | tee test_`echo $Instrument`_beamforming;

            timeout 30m \
                 nosetests  -v --logging-level=FATAL --with-katreport mkat_fpga_tests/test_cbf.py:test_CBF.\
                    test_`echo $Instrument`_bf_efficiency 2>&1 | tee test_`echo $Instrument`_bf_efficiency;
        fi

        timeout 50m nosetests  -v --logging-level=FATAL --with-katreport mkat_fpga_tests/test_cbf.py:test_CBF.\
            test_`echo $Instrument`_channelisation 2>&1 | tee test_`echo $Instrument`_channelisation;

        timeout 30m \
             nosetests  -v --logging-level=FATAL --with-katreport mkat_fpga_tests/test_cbf.py:test_CBF.\
                test_`echo $Instrument`_data_product 2>&1 | tee test_`echo $Instrument`_data_product.txt

        ssh -X $Username@$Server_name './instrument_activate_'$Instrument;

        timeout 30m \
             nosetests  -v --logging-level=FATAL --with-katreport mkat_fpga_tests/test_cbf.py:test_CBF.\
                test_`echo $Instrument`_delay_inputs 2>&1 | tee test_`echo $Instrument`_delay_inputs.txt;

        timeout 30m \
             nosetests  -v --logging-level=FATAL --with-katreport mkat_fpga_tests/test_cbf.py:test_CBF.\
                test_`echo $Instrument`_delay_rate 2>&1 | tee test_`echo $Instrument`_delay_rate.txt

        timeout 30m \
             nosetests  -v --logging-level=FATAL --with-katreport mkat_fpga_tests/test_cbf.py:test_CBF.\
                test_`echo $Instrument`_delay_tracking 2>&1 | tee test_`echo $Instrument`_delay_tracking.txt;

        timeout 30m \
             nosetests  -v --logging-level=FATAL --with-katreport mkat_fpga_tests/test_cbf.py:test_CBF.\
                test_`echo $Instrument`_fringe_offset 2>&1 | tee test_`echo $Instrument`_fringe_offset.txt

        timeout 30m \
             nosetests  -v --logging-level=FATAL --with-katreport mkat_fpga_tests/test_cbf.py:test_CBF.\
                test_`echo $Instrument`_fringe_rate 2>&1 | tee test_`echo $Instrument`_fringe_rate.txt;

        ssh -X $Username@$Server_name './instrument_activate_'$Instrument

        timeout 30m \
             nosetests  -v --logging-level=FATAL --with-katreport mkat_fpga_tests/test_cbf.py:test_CBF.\
             test_`echo $Instrument`_gain_correction 2>&1 | tee test_`echo $Instrument`_gain_correction.txt;

        timeout 30m \
             nosetests  -v --logging-level=FATAL --with-katreport mkat_fpga_tests/test_cbf.py:test_CBF.\
                test_`echo $Instrument`_product_switch 2>&1 | tee test_`echo $Instrument`_product_switch.txt;

        nosetests  -v --logging-level=FATAL --with-katreport mkat_fpga_tests/test_cbf.py:test_CBF.\
            test_`echo $Instrument`_channelisation_sfdr_peaks_fast 2>&1 | tee test_`echo $Instrument`_channelisation_sfdr_peaks_fast.txt;

        ssh -X $Username@$Server_name './instrument_activate_'$Instrument

        nosetests  -v --logging-level=FATAL --with-katreport mkat_fpga_tests/test_cbf.py:test_CBF.\
            test_`echo $Instrument`_corr_efficiency 2>&1 | tee test_`echo $Instrument`_corr_efficiency.txt;
        fi
    echo 'Deprogramming ROACHES associated with instrument %s'$Instrument
    corr2_deprogram.py --hosts /etc/corr/array0-$Instrument
    fi

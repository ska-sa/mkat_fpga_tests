#!/bin/bash
# Script that runs nosetests and initiate an instrument
# Note: The script assumes that you have a script on your home path,
# that initialises the instrument, else create one

# Mpho Mphego <mmphego@ska.ac.za>

printf 'Enter username> '
read -r Username
printf 'Enter IP or server name, eg dbelab04> '
read -r Server_name
printf 'What instrument do you want to run? Enter full name> '
read -r Instrument

printf '[Y/y] Do you want to run nosetests %s?' ${Instrument}
if read -t 10 input ; then
    if [ ${input} == 'Y' -o ${input} == 'y' ] ; then
        printf 'Initialising Instrument: %s\n\n' ${Instrument}
        printf ''
        if ssh -X ${Username}@${Server_name} 'timeout 3 bash instrument_activate_'${Instrument} >&/dev/null; then
            echo "Running instrument: %s" ${Instrument};  fi

        printf 'Running nosetests %s\n' ${Instrument}
        ModeInst=${Instrument##*M}
        ModAlwd='4k'
        TestDir=$(date +%F)
        mkdir -p $TestDir
        #cd /home/${Username}/mkat_fpga_tests

        timeout 30m nosetests -sv --logging-level=ERROR --with-katreport mkat_fpga_tests/test_cbf.py:test_CBF.test__generic_config_report 2>&1 | tee $TestDir/test__generic_config_report_$(date +"%H.%M.%S").txt;

        timeout 30m nosetests  -v --logging-level=ERROR --with-katreport mkat_fpga_tests/test_cbf.py:test_CBF.test__generic_control_init  2>&1 | tee $TestDir/test__generic_control_init_$(date +"%H.%M.%S").txt;

        timeout 30m nosetests  -v --logging-level=ERROR --with-katreport mkat_fpga_tests/test_cbf.py:test_CBF.test__generic_fault_detection 2>&1 | tee $TestDir/test__generic_fault_detection_$(date +"%H.%M.%S").txt;

        timeout 30m nosetests  -v --logging-level=ERROR --with-katreport mkat_fpga_tests/test_cbf.py:test_CBF.test__generic_sensor_values 2>&1 | tee $TestDir/test__generic_sensor_values_$(date +"%H.%M.%S").txt

        timeout 30m nosetests  -v --logging-level=ERROR --with-katreport mkat_fpga_tests/test_cbf.py:test_CBF.test__generic_time_sync 2>&1 | tee $TestDir/test__generic_time_sync_$(date +"%H.%M.%S").txt;

        timeout 30m nosetests  -v --logging-level=ERROR --with-katreport mkat_fpga_tests/test_cbf.py:test_CBF.test__generic_small_voltage_buffer 2>&1 | tee $TestDir/test__generic_small_voltage_buffer_$(date +"%H.%M.%S").txt;

        timeout 30m nosetests  -v --logging-level=ERROR --with-katreport mkat_fpga_tests/test_cbf.py:test_CBF.test_${Instrument}_accumulation_length 2>&1 | tee $TestDir/test_${Instrument}_accumulation_length_$(date +"%H.%M.%S").txt;

        if ssh -X ${Username}@${Server_name} 'timeout 3 bash instrument_activate_'${Instrument} >&/dev/null; then
            echo "Running instrument: %s" ${Instrument};  fi

        timeout 30m nosetests  -v --logging-level=ERROR --with-katreport mkat_fpga_tests/test_cbf.py:test_CBF.test_${Instrument}_baseline_correlation_product 2>&1 | tee $TestDir/test_${Instrument}_baseline_correlation_product_$(date +"%H.%M.%S").txt;

        timeout 50m nosetests  -v --logging-level=ERROR --with-katreport mkat_fpga_tests/test_cbf.py:test_CBF.test_${Instrument}_channelisation 2>&1 | tee $TestDir/test_${Instrument}_channelisation_$(date +"%H.%M.%S").txt;

        timeout 30m nosetests  -v --logging-level=ERROR --with-katreport mkat_fpga_tests/test_cbf.py:test_CBF.test_${Instrument}_data_product 2>&1 | tee $TestDir/test_${Instrument}_data_product_$(date +"%H.%M.%S").txt;

        if ssh -X ${Username}@${Server_name} 'timeout 3 bash instrument_activate_'${Instrument} >&/dev/null; then
            echo "Running instrument: %s" ${Instrument};  fi

        timeout 30m nosetests  -v --logging-level=ERROR --with-katreport mkat_fpga_tests/test_cbf.py:test_CBF.test_${Instrument}_delay_inputs 2>&1 | tee $TestDir/test_${Instrument}_delay_inputs_$(date +"%H.%M.%S").txt;

        timeout 30m nosetests  -v --logging-level=ERROR --with-katreport mkat_fpga_tests/test_cbf.py:test_CBF.test_${Instrument}_delay_rate 2>&1 | tee $TestDir/test_${Instrument}_delay_rate_$(date +"%H.%M.%S").txt

        timeout 30m nosetests  -v --logging-level=ERROR --with-katreport mkat_fpga_tests/test_cbf.py:test_CBF.test_${Instrument}_delay_tracking 2>&1 | tee $TestDir/test_${Instrument}_delay_tracking_$(date +"%H.%M.%S").txt;

        timeout 30m nosetests  -v --logging-level=ERROR --with-katreport mkat_fpga_tests/test_cbf.py:test_CBF.test_${Instrument}_fringe_offset 2>&1 | tee $TestDir/test_${Instrument}_fringe_offset_$(date +"%H.%M.%S").txt

        timeout 30m nosetests  -v --logging-level=ERROR --with-katreport mkat_fpga_tests/test_cbf.py:test_CBF.test_${Instrument}_fringe_rate 2>&1 | tee $TestDir/test_${Instrument}_fringe_rate_$(date +"%H.%M.%S").txt;

        if ssh -X ${Username}@${Server_name} 'timeout 3 bash instrument_activate_'${Instrument} >&/dev/null; then
            echo "Running instrument: %s" ${Instrument};  fi

        timeout 30m nosetests  -v --logging-level=ERROR --with-katreport mkat_fpga_tests/test_cbf.py:test_CBF.test_${Instrument}_gain_correction 2>&1 | tee $TestDir/test_${Instrument}_gain_correction_$(date +"%H.%M.%S").txt;

        timeout 30m nosetests  -v --logging-level=ERROR --with-katreport mkat_fpga_tests/test_cbf.py:test_CBF.test_${Instrument}_product_switch 2>&1 | tee $TestDir/test_${Instrument}_product_switch_$(date +"%H.%M.%S").txt;

        if ssh -X ${Username}@${Server_name} 'timeout 3 bash instrument_activate_'${Instrument} >&/dev/null; then
            echo "Running instrument: %s" ${Instrument};  fi

        if [ ${ModeInst} == ${ModAlwd} ]; then
            printf '\n**************************************************\n'
            printf '\n************Running 4K Beamforming Tests**********\n'
            printf '\n**************************************************\n'

            timeout 30m nosetests  -v --logging-level=ERROR --with-katreport mkat_fpga_tests/test_cbf.py:test_CBF.test_${Instrument}_beamforming 2>&1 | tee $TestDir/test_${Instrument}_beamforming_$(date +"%H.%M.%S").txt;

            timeout 30m nosetests  -v --logging-level=ERROR --with-katreport mkat_fpga_tests/test_cbf.py:test_CBF.test_${Instrument}_bf_efficiency 2>&1 | tee $TestDir/test_${Instrument}_bf_efficiency_$(date +"%H.%M.%S").txt;

            nosetests  -v --logging-level=ERROR --with-katreport mkat_fpga_tests/test_cbf.py:test_CBF.test_${Instrument}_channelisation_sfdr_peaks 2>&1 | tee $TestDir/test_${Instrument}_channelisation_sfdr_peak_$(date +"%H.%M.%S").txt;
        else
            nosetests  -v --logging-level=ERROR --with-katreport mkat_fpga_tests/test_cbf.py:test_CBF.test_${Instrument}_channelisation_sfdr_peaks_fast 2>&1 | tee $TestDir/test_${Instrument}_channelisation_sfdr_peaks_fast_$(date +"%H.%M.%S").txt;
        fi
    fi
    echo 'Deprogramming ROACHES associated with instrument %s.' ${Instrument}
    corr2_deprogram.py --hosts /etc/corr/array0-${Instrument}
fi

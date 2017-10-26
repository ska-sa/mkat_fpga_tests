class TestProcedure:
    """Test Procedures"""

    @property
    def TempFaultDetection(self):
        _description = """
        **Host Temperature Fault Detection**

        1. Configure a digitiser simulator to generate correlated input noise signal.
        2. Set a predetermined accumulation period
            - and confirm it has been set via CAM interface.
        3. Initiate SPEAD receiver, enable data to flow and confirm CBF output product
        4. Connect to a random host.
        5. For both Air Inlet and Outlet Temperature Warning.
            - Retrive current air outlet temperature
            - Confirm that the over-temp alarm is Not triggered.
            - Confirm that the under-temp alarm is Not triggered.
            - Confirm that the over-temp alarm is Triggered.
            - Confirm that the under-temp alarm is Triggered.
            - Confirm that the over-temp alarm was set back to default.
            - Confirm that the under-temp alarm was set back to default.
        """
        return _description

    @property
    def MemFaultDetection(self):
        _description = """
        **Memory Error Detection**

        1. Configure a digitiser simulator to generate correlated input noise signal.
        2. Set a predetermined accumulation period
            - and confirm it has been set via CAM interface.
        3. Initiate SPEAD receiver, enable data to flow and confirm CBF output product
        4. Connect to a random F-host via CAM interface
        5. Confirm that the hardware sensor indicates that the QDR is healthy.
        6. Write random data to the QDR memory.
            - Confirm that sensor indicates that the QDR memory is unreadable/corrupted.
            - Confirm that the error counters have incremented, showing that the corner turner experienced faults.
            - Confirm that the error counters have stopped incrementing with last known incremented.
        7. Reset the error counters,
            - Confirm that the error counters have been reset
            - Confirm that sensor indicates that the QDR memory recovered.
        8. Repeat above steps for Xhosts
        """
        return _description

    @property
    def LinkFaultDetection(self):
        _description = """
        **Link Error Detection**

        1. Configure a digitiser simulator to generate correlated input noise signal.
        2. Set a predetermined accumulation period
            - Confirm it has been set via CAM interface.
        3. Initiate SPEAD receiver, enable data to flow and confirm CBF output product
        4. Connect to a random host via CAM interface
        5. Retrieve current multicast destination address
        6. Randomly change the multicast destination address of the host.
            - Confirm that the multicast destination address has been changed
            - Confirm that the X-engine LRU sensor is reporting a failure,
            - Confirm that the SPEAD accumulation are still being produced
        7. Restore the multicast destination address back to original
            - Confirm that the multicast destination address has been changed back to its original
            - Confirm that the X-engine LRU sensor is healthy
            - Confirm that the SPEAD accumulation is being produced and verify that the data is feasible
        """
        return _description

    @property
    def PFBFaultDetection(self):
        _description = """
        **PFB Fault Detection**

        1. Configure a digitiser simulator to generate correlated input noise signal.
        2. Set a predetermined accumulation period,
            - Confirm it has been set via CAM interface.
        3. Initiate SPEAD receiver, enable data to flow and confirm CBF output product
        4. Connect to a random host via CAM interface
        5. Set a sensor polling time
        6. Retrieve FFT-shift on all fhosts
            - Confirm that the sensors indicate that all hosts PFB status is healthy
        7. Set an FFT-shift on all fhosts that would cause a PFB overflow
            - Confirm that the sensors indicated that the fhosts PFB has been set and,
            - Confirm that the sensors indicates that the hosts PFB status has an error
        9. Restore previous FFT Shift values
            - Confirm that the sensors indicated that the fhosts PFB has been set to original and,
            - Confirm that the sensors indicates that the hosts PFB status is healthy
        """
        return _description

    @property
    def ReportSensorStatus(self):
        _description = """
        **Sensor status**

        1. Configure a digitiser simulator to generate correlated input noise signal.
        2. Set a predetermined accumulation period,
            - Confirm it has been set via CAM interface.
        3. Initiate SPEAD receiver, enable data to flow and confirm CBF output product
        4. Check that the number of sensors available on the primary and sub array interface is consistent.
        5. Check that time synchronous is implemented on primary interface
        6. Check that Transient Buffer ready is implemented.
        """
        return _description

    @property
    def ReportHostSensor(self):
        _description = """
        **ROACH2 Sensor (Temp, Voltage, Current, Fan) Status**

        1. This test confirms that each ROACH sensor (Temp, Voltage, Current, Fan) has not Failed.
        2. For all hosts.
            - Verify that the number of hardware sensors are consistent.
            - Confirm that the number of hardware sensors-list are equal to the sensor-values of specific hardware
            - Confirm if hosts contain(s) any errors/faults via CAM interface.
        """
        return _description

    @property
    def ReportHWVersion(self):
        _description = """
        **CBF ROACH version information.**

        1. Configure a digitiser simulator to generate correlated input noise signal.
        2. Set a predetermined accumulation period
            - Confirm it has been set via CAM interface.
        3. Initiate SPEAD receiver, enable data to flow and confirm CBF output product
        4. Determine number of hosts and current configuration(Dsim, X/Fengine)
        5. Retrive hosts firmware infomation
            - UBoot Version
            - ROMFS Version
            - H/W Revision
        """
        return _description

    @property
    def ReportSWVersion(self):
        _description = """
        **CBF Software Version Information**

        1. Configure a digitiser simulator to generate correlated input noise signal.
        2. Set a predetermined accumulation period
            - Confirm it has been set via CAM interface.
        3. Initiate SPEAD receiver, enable data to flow and confirm CBF output product
        4. Request a list of available configuration items using KATCP command "?version-list"
            - F/Xengine firmware information
            - KATCP device, library, protocol
            - CBF server name
        """
        return _description

    @property
    def ReportGitVersion(self):
        _description = """
        **CBF Git Version Information**

        1. Configure a digitiser simulator to generate correlated input noise signal.
        2. Set a predetermined accumulation period
            - Confirm it has been set via CAM interface.
        3. Initiate SPEAD receiver, enable data to flow and confirm CBF output product
        4. Retrieve CORE software information
            - Repo directory
            - Repo branch and status
            - Repo version/tag
        """
        return _description

    @property
    def GainCorr(self):
        _description = """
        **Gain Correction**

        1. Configure a digitiser simulator to generate correlated input noise signal.
        2. Set a predetermined accumulation period
            - Confirm it has been set via CAM interface.
        3. Initiate SPEAD receiver, enable data to flow and confirm CBF output product
        4. Randomly select an input to test.
            - Note: Gains are relative to reference channels, and are increased iteratively until output power is increased by more than 6dB.
        5. Set gain correction on selected input to default
            - Confirm the gain has been set
        6. Iteratively request gain correction on one input on a single channel, single polarisation.
            - Confirm output power increased by less than 1 dB with a random gain increment [Dependent on mode].
            - Until the output power is increased by more than 6 dB
        """
        return _description

    @property
    def BaselineCorrelation(self):
        _description = """
        **Baseline Correlation Products**

        1. Configure a digitiser simulator to be used as input source to F-Engines
        2. Configure a digitiser simulator to generate correlated input noise signal.
        3. Set a predetermined accumulation period
            - Confirm it has been set via CAM interface.
        4. Initiate SPEAD receiver, enable data to flow and confirm CBF output product
        5. Set list for all the correlator input labels as per config file
        6. Capture an initial correlator SPEAD accumulation, and retrieve list of all the correlator input labels via CAM interface.
        7. Get list of all possible baselines (including redundant baselines) present in the correlator output from SPEAD accumulation
        8. Check that each baseline (or its reverse-order counterpart) is present in the correlator output
        9. Expect all baselines and all channels to be non-zero with Digitiser Simulator set to output AWGN.
              - Confirm that no baselines have all-zero visibilities.
              - Confirm that all baseline visibilities are non-zero across all channels
        10. Save initial f-engine equalisations, and ensure they are restored at the end of the test
        11. Set all inputs gains to `Zero`, and confirm that output product is all-zeros
        12. - Confirm that all the inputs equalisations have been set to 'Zero'.
            -  Confirm that all baseline visibilities are `Zero`.
        13. Iterate through input combinations, verifying for each that the correct output appears in the correct baseline product.
            - Set gain/equalisation correction on relevant input
            - Retrieving SPEAD accumulation and,
            - Confirm if gain/equalization correction has been applied.
        14. Check that expected baseline visibilities are nonzero with non-zero inputs and,
            - Confirm that expected baselines visibilities are 'Zeros'
            """
        return _description

    @property
    def BackToBack(self):
        _description = """
        **Baseline Correlation Products Back-to-Back Consistency**

        1. Configure a digitiser simulator to be used as input source to F-Engines
        2. Configure a digitiser simulator to generate periodic wave in order for each FFT to be identical.
        3. Set a predetermined accumulation period
            - Confirm it has been set via CAM interface.
        4. Initiate SPEAD receiver, enable data to flow and confirm CBF output product
        5. Randomly select a channel to test.
        6. Calculate a list of frequencies to test
        7. Sweep the digitiser simulator over the selected/requested frequencies fall within the complete L-band, for each frequency
            - Retrieve channel response for each frequency.
            - Capture SPEAD accumulation and
            - Confirm that the difference between subsequent accumulation is Zero.
            - Check that the maximum difference between the subsequent SPEAD accumulations with the same frequency input is 'Zero' on baseline that baseline

        Note: This test confirms that back-to-back SPEAD accumulations with same frequency input are identical/bit-perfect.
        """
        return _description

    @property
    def FreqConsistency(self):
        _description = """
        **Baseline Correlation Products Frequency Consistency**

        1. Configure a digitiser simulator to be used as input source to F-Engines
        2. Configure a digitiser simulator to generate generate continuous wave.
        3. Set a predetermined accumulation period
            - Confirm it has been set via CAM interface.
        4. Initiate SPEAD receiver, enable data to flow and confirm CBF output product
        5. Randomly select a frequency channel to test.
        6. Sweep the digitiser simulator over the centre frequencies of all the selected frequency channels that fall within the complete L-band
            - Retrieve channel response for each frequency
            - Capture SPEAD accumulations, and
            - Confirm that identical frequency scans between subsequent SPEAD accumulations produce equal results.

        Note: This test confirms if the identical frequency scans produce equal results.
        """
        return _description

    @property
    def SpeadVerify(self):
        _description = """
        **Baseline Correlation Products SPEAD ACC Verification**

        1. Configure a digitiser simulator to be used as input source to F-Engines
        2. Configure a digitiser simulator to generate generate continuous wave on input 0
        3. Set a predetermined accumulation period
            - Confirm it has been set via CAM interface.
        4. Initiate SPEAD receiver, enable data to flow and confirm CBF output product
        5. Capture a correlator SPEAD accumulation.
            - Confirm that auto-correlation in baseline 0 contains Non-Zeros and,
            - Baseline 1 is Zeros, when cw tone is only outputted on input 0.
        6. Reset digitiser simulator to Zeros
        7. Configure digitiser simulator configured to generate cw tone with frequency on input 1
            - Capture a correlator SPEAD accumulation.
            - Confirm that auto-correlation in baseline 1 contains non-Zeros and,
            - Baseline 0 is Zeros, when cw tone is only outputted on input 1.

        Note: This test verifies if a cw tone is only applied to a single input 0, Confirm if VACC is rooted by 1
        """
        return _description

    @property
    def DataProduct(self):
        _description = """
        **Imaging Data Product**

        1. Configure a digitiser simulator to be used as input source to F-Engines
        2. Configure a digitiser simulator to generate correlated noise.
        3. Set a predetermined accumulation period
            - Confirm it has been set via CAM interface.
        4. Initiate SPEAD receiver, enable data to flow and confirm CBF output product
        5. Configure the CBF to simultaneously generate Baseline Correlation Products and Tied-Array Voltage Data Products (If available)
        6. Capture Tied-Array Data and,
            - - Confirm that the tide-array data were captured.
        7. Capture Correlation Data and,
            - Confirm the number of channels in the SPEAD data.
            - Check that data product has the number of frequency channels corresponding to the instrument.
            - Confirm that data products were captured.
        8. Confirm that imaging data product set has been implemented for the instrument
        """
        return _description

    @property
    def Control(self):
        _description = """
        **Control Interface**

        1. Configure a digitiser simulator to be used as input source to F-Engines
        2. Set a predetermined accumulation period
            - Confirm it has been set via CAM interface.
        3. Initiate SPEAD receiver, enable data to flow and confirm CBF output product
        4. List of available CAM commands, current status(executable or abondoned),replies(if any)
        """
        return _description

    @property
    def TimeSync(self):
        _description = """
        **Time synchronisation**

        1. Configure a digitiser simulator to be used as input source to F-Engines
        2. Set a predetermined accumulation period
            - Confirm it has been set via CAM interface.
        3. Initiate SPEAD receiver, enable data to flow and confirm CBF output product
        4. Request NTP pool address used
        4. Confirm that the CBF synchronised time is within 0.005s of UTC time as provided via PTP (NTP server) on the CBF-TRF interface.
        """
        return _description

    @property
    def VoltageBuffer(self):
        _description = """
        **Voltage Buffer Data Product**

        1. Configure a digitiser simulator to be used as input source to F-Engines
        2. Configure a digitiser simulator to generate continuous wave
        3. Set a predetermined accumulation period
            - Confirm it has been set via CAM interface.
        4. Initiate SPEAD receiver, enable data to flow and confirm CBF output product
        5. Check that Transient Buffer ready is implemented via CAM interface.
        6. Capture an ADC snapshot via CAM inteface and,
            - Confirm the FFT length
            - Check that the expected frequency and measured frequency matches to within a channel bandwidth
        """
        return _description

    @property
    def VectorAcc(self):
        _description = """
        **Vector Accumulator**

        1. Configure a digitiser simulator to be used as input source to F-Engines
        2. Configure a digitiser simulator to generate generate continuous wave
        3. Set a predetermined accumulation period
            - Confirm it has been set via CAM interface.
        4. Initiate SPEAD receiver, enable data to flow and confirm CBF output product
        5. Select a test input and frequency channel
        6. Compile a list of accumulation periods to test
        7. Set gain correction on selected input via CAM interface.
        8. Configure a digitiser simulator to generate periodic wave in order for each FFT to be identical.
            - Check that the spectrum is zero except in the test channel
            - Confirm FFT Window samples, Internal Accumulations, VACC accumulation
        9. Retrieve quantiser snapshot of the selected input via CAM Interface
        10. Iteratively set accumulation length and confirm if the righy accumulation is set on the SPEAD accumulation,
            - Confirm that vacc length was set successfully, and equates to a specific accumulation time as per calculation
            - Check that the accumulator actual response is equal to the expected response for the accumulation length
        """
        return _description

    @property
    def ProductSwitching(self):
        _description = """
        1. Configure a digitiser simulator to generate noise.
        2. Configure the CBF to generate a data product, using the noise source. Which specific data product is chosen is irrelevant.
        3. Confirm that SPEAD packets are being produced, with the selected data product(s).
        4. Start timer.
        5. Deprogram CBF and confirm that SPEAD packets are either no longer being produced, or that the data content is at least affected.
        6. Reinitialise the instrument and repeat step 2 and 3.
        7. Confirm that SPEAD packets are being produced, with the selected data product(s).
        8. Stop timer and
            - Confirm data product switching time is less than 60 seconds (Data Product switching time = End time - Start time.)
        9. Repeat for all combinations of available data products, including the case where the "new" data product is the same as the "old" one.
        """
        return _description

    @property
    def Channelisation(self):
        _description = """
        **Channelisation Wideband Coarse/Fine L-band**

        1. Configure a digitiser simulator to be used as input source to F-Engines
        2. Configure a digitiser simulator to generate continuous wave
        3. Set a predetermined accumulation period
            - Confirm it has been set via CAM interface.
        4. Initiate SPEAD receiver, enable data to flow and confirm CBF output product
        5. Calculate number of frequencies to iterate on
        6. Randomly select a frequency channel to test.
        7. Capture an initial correlator SPEAD accumulation and,
            - Determine the number of frequency channels
            - Confirm that the number of channels in the SPEAD accumulation, is equal to the number of frequency channels as calculated
            - Confirm that the Channelise total bandwidth is >= 770000000.0Hz.
            - Confirm the number of calculated channel frequency step is within requirement.
            - Verify that the calculated channel frequency step size is withing requirement
            - Confirm the channelisation spacing and confirm that it is within the maximum tolerance.
        8. Sweep the digitiser simulator over the centre frequencies of at least all the channels that fall within the complete L-band
            - Capture channel response for every frequency channel in the selected frequenies calculated
        Check FFT overflow and QDR errors after channelisation.
        9. Check that the peak channeliser response to input frequencies in central 80% of the test channel frequency band are all in the test channel
        10. Check that VACC output is at < 99% of maximum value, if fails then it is probably overranging.
        11. Check that ripple within 80% of cut-off frequency channel is < 1.5 dB
        12. Check that response at channel-edges are -3 dB relative to the channel centre at selected freq, actual source frequency
        13. Check that relative response at the low band-edge is within the range of -6 +- 1% relative to channel centre response.
        14. Check that relative response at the high band-edge is within the range of -6 +- 1% relative to channel centre response.
         """
        return _description

    @property
    def ChannelisationSFDR(self):
        _description = """
        **Channelisation Spurious Free Dynamic Range**

        1. Configure a digitiser simulator to be used as input source to F-Engines
        2. Configure a digitiser simulator to generate continuous wave
        3. Set a predetermined accumulation period
            - Confirm it has been set via CAM interface.
        4. Initiate SPEAD receiver, enable data to flow and confirm CBF output product
        5. Capture an initial correlator SPEAD accumulation, determine the number of frequency channels.
            - Confirm the number of calculated channel frequency step is within requirement.
            - Determine the number of channels and processing bandwidth
        6. Sweep the digitiser simulator over the all channels that fall within the complete L-band.
            - Capture SPEAD accumulation for every frequency channel and,
        7. Calculate and check that the correct channels have the peak response to each frequency
            - Confirm that the correct channels have the peak response to each frequency
            - Check that no other channels response more than -53 dB.

        Note: This tests confirms that the correct channels have the peak response to each frequency and
        that no other channels have significant relative power, while logging the power usage of the CBF in the background.

         """
        return _description

    @property
    def PowerConsumption(self):
        _description = """
        1. Request power consumption of each PDU via telnet interface at 1 minute intervals.
        2. Repeat for each PDU.
        3. Exercise each of the available CBF data product sets for 10 minutes or more. The total time must be more than 60 minutes.
        4. After >60 minutes, log in to each CBF PDU in turn. Click Logs->Data->log. Copy and paste the data log into a text file. Import that file as a space delimited file into a spreadsheet, with a worksheet per PDU.
        5. Make a column in each worksheet which computes for each entry the percentage instantaneous current drawn per phase i.e. I Ph1/(I Ph1+I Ph2+I Ph3).
        6. Make a column in each worksheet which computes for each entry the max of the percentage instantaneous current drawn per phase divided by the min of the percentage instantaneous current drawn per phase (=MAX(T5:V5)/MIN(T5:V5)).
        7. Compute the maximum of that ratio.
        8. Repeat for each PDU in each worksheet. The load balance test passes if the maximum ratio is less than 1.33 in each worksheet.
            - In each worksheet, make a column which computes total instantaneous power by multiplying current drawn over the three phases, multiplied by 220 i.e. (I Ph1+I Ph2+I Ph3)*220.
            -  In each worksheet, compute the average of the power.
        9. The average power per rack test passes if the average peak power in each of the spreadsheets is <= 6.25kW.
        10. Sum the average power per rack to get a CBF average peak power.
        11. The CBF average peak power test passes if the CBF average power is <= 60kW.
        12. Divide the CBF average power by the number of CBF racks that are actually used, to get a CBF average power per rack.
        13. The CBF maximum heat generation test passes that the CBF average power per rack is <= 5kW.
            - In each worksheet, make a column which computes peak power by multiplying peak current drawn over the three phases, multiplied by 220 i.e. (Imax Ph1+Imax Ph2+Imax Ph3)*220.
            - Sum the peak power of each rack to get a CBF peak power.
            - The CBF peak power test passes if the maximum CBF peak power is <= 60kW.
         """
        return _description

    @property
    def DelayRate(self):
        _description = """
        **Delay Compensation/LO Fringe stopping polynomial: Delay Rate**

        1. Configure a digitiser simulator to be used as input source to F-Engines
        2. Configure a digitiser simulator to generate correlated Gaussian noise
        3. Set a predetermined accumulation period
            - Confirm it has been set via CAM interface.
        4. Initiate SPEAD receiver, enable data to flow and confirm CBF output product
        5. Change CBF input labels and confirm via CAM inteface.
        6. Clear all coarse and fine delays for all inputs before testing
            - Confirm if all previously applied delays have been reset
        7. Retrieve initial SPEAD accumulation, in-order to calculate all relevant parameters.
        8. Get list of all the baselines present in the correlator output
        9. Select input and baseline for testing
        10. Set time to apply delays to 30 integrations/accumulations in the future.
        11. Request Delay(s) Corrections via CAM interface.
            - Confirm delays have been set
            - Calculate the time it takes to load delay/fringe(s), value should be less than 1s as per requirement
        12. Capture SPEAD accumulation containing the change in phase on selected input and discard all irrelevant accumulations.
        13. When final SPEAD accumulation iis received,
            - Capture all subsequent SPEAD accumulation
        14. For all subsequenct SPEAD accumulaions captured,
            - Observe the change in the phase slope, and confirm the phase change is as expected.
            - Check if difference between expected phases and actual phases are 'Almost Equal' within 1.0 degree when a delay rate is applied.
            - Check that the maximum difference between expected phase and actual phase between integrations is less than 1.0 degree.
        """
        return _description

    @property
    def DelayTracking(self):
        _description = """
        **Delay Compensation/LO Fringe stopping polynomial: Delay Tracking**

        1. Configure a digitiser simulator to be used as input source to F-Engines
        2. Configure a digitiser simulator to generate correlated Gaussian noise
        3. Set a predetermined accumulation period
            - Confirm it has been set via CAM interface.
        4. Initiate SPEAD receiver, enable data to flow and confirm CBF output product
        5. Change CBF input labels and confirm via CAM inteface.
        6. Clear all coarse and fine delays for all inputs before testing
            - Confirm if all previously applied delays have been reset
        7. Retrieve initial SPEAD accumulation, in-order to calculate all relevant parameters.
        8. Get list of all the baselines present in the correlator output
        9. Select input and baseline for testing
        10. Set time to apply delays to x integrations/accumulations in the future.
        11. Compile list of delays to be set (iteratively) for testing purposes

        12. Iterate through preselected, execute delays via CAM interface and calculate the amount of time it takes to load the delays
            - Confirm delays have been set
            - Calculate the time it takes to load delay/fringe(s), value should be less than 1s as per requirement
            - Capture SPEAD accumulation (while discarding x dumps) containing the change in delay(s) on selected input
        13. Iteratively, with captured SPEAD accumulations,
            - Check that if difference expected and actual phases are equal at delay 0.00000ns within 1.0 degree.
            - Check that the maximum difference between expected phase and actual phase between integrations is less than 1.0 degree.
            - Check that when a delay of x clock cycle is introduced there is a phase change of x degrees as expected to within 1.0 degree.
            """
        return _description

    @property
    def FringeOffset(self):
        _description = """
        **Delay Compensation/LO Fringe stopping polynomial: Fringe Offset**

        1. Configure a digitiser simulator to be used as input source to F-Engines
        2. Configure a digitiser simulator to generate correlated Gaussian noise
        3. Set a predetermined accumulation period
            - Confirm it has been set via CAM interface.
        4. Initiate SPEAD receiver, enable data to flow and confirm CBF output product
        5. Change CBF input labels and confirm via CAM inteface.
        6. Clear all coarse and fine delays for all inputs before testing
            - Confirm if all previously applied delays have been reset
        7. Retrieve initial SPEAD accumulation, in-order to calculate all relevant parameters.
        8. Get list of all the baselines present in the correlator output
        9. Select input and baseline for testing
        10. Set time to apply delays to x integrations/accumulations in the future.
        11. Request Delay(s) Corrections via CAM interface.
            - Confirm delays have been set
            - Calculate the time it takes to load delay/fringe(s), value should be less than 1s as per requirement

        12. Capture SPEAD accumulation containing the change in phase on selected input and discard all irrelevant accumulations.
        13. When final SPEAD accumulation is received,
            - Capture all subsequent SPEAD accumulation
        14. For all subsequenct SPEAD accumulaions captured,
            - Observe the change in the phase slope, and confirm the phase change is as expected.
            - Check if difference between expected phases and actual phases are 'Almost Equal' within 1.000 degree when fringe offset of 2.548 is applied.
            - Check that the maximum difference between expected phase and actual phase between integrations is less than 1.000 degree
            """
        return _description


    @property
    def FringeRate(self):
        _description = """
        **Delay Compensation/LO Fringe stopping polynomial: Fringe Rate**

        1. Configure a digitiser simulator to be used as input source to F-Engines
        2. Configure a digitiser simulator to generate correlated Gaussian noise
        3. Set a predetermined accumulation period
            - Confirm it has been set via CAM interface.
        4. Initiate SPEAD receiver, enable data to flow and confirm CBF output product
        5. Change CBF input labels and confirm via CAM inteface.
        6. Clear all coarse and fine delays for all inputs before testing
            - Confirm if all previously applied delays have been reset
        7. Retrieve initial SPEAD accumulation, in-order to calculate all relevant parameters.
        8. Get list of all the baselines present in the correlator output
        9. Select input and baseline for testing
        10. Set time to apply delays to x integrations/accumulations in the future.
        11. Request Delay(s) Corrections via CAM interface.
            - Confirm delays have been set
            - Calculate the time it takes to load delay/fringe(s), value should be less than 1s as per requirement

        12. Capture SPEAD accumulation containing the change in phase on selected input and discard all irrelevant accumulations.
        13. When final SPEAD accumulation is received,
            - Capture all subsequent SPEAD accumulation
        14. For all subsequenct SPEAD accumulaions captured,
            - Observe the change in the phase slope, and confirm the phase change is as expected.
            - Check if difference between expected phases and actual phases are 'Almost Equal' within 1.000 degree when fringe offset of 2.548 is applied.
            - Check that the maximum difference between expected phase and actual phase between integrations is less than 1.000 degree
            """
        return _description

    @property
    def DelayInputs(self):
        _description = """
        **Delay Compensation/LO Fringe stopping polynomial: Delayed Input**

        1. Configure a digitiser simulator to be used as input source to F-Engines
        2. Configure a digitiser simulator to generate correlated Gaussian noise
        3. Set a predetermined accumulation period
            - Confirm it has been set via CAM interface.
        4. Initiate SPEAD receiver, enable data to flow and confirm CBF output product
        5. Change CBF input labels and confirm via CAM inteface.
        6. Clear all coarse and fine delays for all inputs before testing
            - Confirm if all previously applied delays have been reset
        7. Retrieve initial SPEAD accumulation, in-order to calculate all relevant parameters.
        8. Get list of all the baselines present in the correlator output
        9. Select input and baseline for testing
        10. Set time to apply delays to x integrations/accumulations in the future.
        11. Request Delay(s) Corrections via CAM interface.
            - Confirm delays have been set
            - Calculate the time it takes to load delay/fringe(s), value should be less than 1s as per requirement
        12. ...
        13. ...
        """
        return _description

    @property
    def Beamformer(self):
        _description = """
        **Beamformer functionality**

        1. Configure a digitiser simulator to be used as input source to F-Engines
        2. Configure a digitiser simulator to generate correlated Gaussian noise
        3. Set a predetermined accumulation period
            - Confirm it has been set via CAM interface.
        4. Initiate SPEAD receiver, enable data to flow and confirm CBF output product
        5. Retrieve current instrument parameters.
        6. Request beamformer level adjust gain
        7. Configure beam tied-array-channelised-voltage.0y passband and set to desired center frequency
            - Confirm that beam tied-array-channelised-voltage.0y passband has been set
        8. Set inputs to desired weights and,
            - Confirm that the input weight has been set to the desired weight.
        9. Capture beam tied-array data and,
            - Read and extract beam data from h5py data file in /ramdisk/epoch_time
            - List missed heaps on partitions
            - Confirm the data type of the beamforming data for one channel.
        11. Expected value is calculated by taking the reference input level and multiplying by the channel weights and quantiser gain.
            - Capture reference level measured by setting the gain for one antenna to 1 and the rest to 0.
            - Capture reference level averaged over x channels. Channel averages determined over x samples.
            - Confirm that the expected voltage level is within 0.2dB of the measured mean value
        Repeat above for different beam weights        """
        return _description




TestProcedure = TestProcedure()


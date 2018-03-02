class TestProcedure:
    """Test Procedures"""

    @property
    def TBD(self):
        _description = """
        **TBD**
        """
        return _description

    @property
    def LinkFaultDetection(self):
        _description = """
        **Link Error Detection**

        1. Connect to a random host via CAM interface
        2. Retrieve current multicast destination address
        3. Randomly change the multicast destination address of the host.
            - Confirm that the multicast destination address has been changed
            - Confirm that the X-engine LRU sensor is reporting a failure,
            - Confirm that the SPEAD accumulation are still being produced
        4. Restore the multicast destination address back to original
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
    def MonitorSensors(self):
        _description = """
        **Sensor status**

        1. Confirm that the number of sensors available on the primary and sub array interface is consistent.
        2. Confirm that time synchronous is implemented on primary interface
        3. Confirm that Transient Buffer ready is implemented.

        **Processing Node's Sensor (Temp, Voltage, Current, Fan) Status**

        1. This test confirms that each Processing Node's sensor (Temp, Voltage, Current, Fan) has not Failed.
        2. For all hosts.
            - Verify that the number of hardware sensors are consistent.
            - Confirm that the number of hardware sensors-list are equal to the sensor-values of specific hardware
            - Confirm if hosts contain(s) any errors/faults via CAM interface.
        """
        return _description

    @property
    def ReportConfiguration(self):
        _description = """
        **CBF processing node version information.**

        1. Determine number of hosts and current configuration(DEngine, X/F/B-Engine)
        2. Retrieve hosts firmware information

        **CBF Software Version Information**

        1. Request a list of available configuration items using KATCP command "?version-list"
            - F/Xengine firmware information
            - KATCP device, library, protocol
            - CBF server name

        **CBF Git Version Information**

        1. Retrieve CORE software information
            - Repository directory
            - Repository branch and status
            - Repository version/tag
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
        1. Configure a digitiser simulator to be used as input source to F-Engines
        2. Configure a digitiser simulator to generate correlated input noise signal.
        3. Set a predetermined accumulation period
            - Confirm it has been set via CAM interface.
        4. Initiate SPEAD receiver, enable data to flow and confirm CBF output product

        **Baseline Correlation Products**

        1. Set list for all the correlator input labels as per config file
        2. Capture an initial correlator SPEAD accumulation, and retrieve list of all the correlator input labels via CAM interface.
        3. Get list of all possible baselines (including redundant baselines) present in the correlator output from SPEAD accumulation
        4. Check that each baseline (or its reverse-order counterpart) is present in the correlator output
        5. Expect all baselines and all channels to be non-zero with Digitiser Simulator set to output AWGN.
              - Confirm that no baselines have all-zero visibilities.
              - Confirm that all baseline visibilities are non-zero across all channels
        6. Save initial f-engine equalisations, and ensure they are restored at the end of the test
        7. Set all inputs gains to `Zero`, and confirm that output product is all-zeros
        8. - Confirm that all the inputs equalisations have been set to 'Zero'.
            -  Confirm that all baseline visibilities are `Zero`.
        9. Iterate through input combinations, verifying for each that the correct output appears in the correct baseline product.
            - Set gain/equalisation correction on relevant input
            - Retrieving SPEAD accumulation and,
            - Confirm if gain/equalization correction has been applied.
        10. Check that expected baseline visibilities are non-zero with non-zero inputs and,
            - Confirm that expected baselines visibilities are 'Zeros'

        **Spead Accumulation Back-to-Back Consistency**

        Note: This test confirms that back-to-back SPEAD accumulations with same frequency input are identical/bit-perfect.

        1. Randomly select a channel to test.
        2. Calculate a list of frequencies to test
        3. Sweep the digitiser simulator over the selected/requested frequencies fall within the complete L-band, for each frequency
            - Retrieve channel response for each frequency.
            - Capture SPEAD accumulation and
            - Confirm that the difference between subsequent accumulation is Zero.
            - Check that the maximum difference between the subsequent SPEAD accumulations with the same frequency input is 'Zero' on baseline that baseline


        **Spead Accumulation Frequency Consistency**

        Note: This test confirms if the identical frequency scans produce equal results.

        1. Randomly select a frequency channel to test.
        2. Sweep the digitiser simulator over the centre frequencies of all the selected frequency channels that fall within the complete L-band
            - Retrieve channel response for each frequency
            - Capture SPEAD accumulations, and
            - Confirm that identical frequency scans between subsequent SPEAD accumulations produce equal results.


        **SPEAD Accumulation Verification**

        Note: This test verifies if a cw tone is only applied to a single input 0, Confirm if VACC is rooted by 1

        1. Configure a digitiser simulator to be used as input source to F-Engines
        2. Configure a digitiser simulator to generate continuous wave, on input 0
        3. Set a predetermined accumulation period
            - Confirm it has been set via CAM interface.
        4. Initiate SPEAD receiver, enable data to flow and confirm CBF output product
        5. Capture SPEAD accumulation, and
            - Confirm that auto-correlation in baseline 0 contains Non-Zeros and,
            - Baseline 1 is Zeros, when cw tone is only outputted on input 0.
        6. Reset digitiser simulator to Zeros
        7. Configure digitiser simulator configured to generate cw tone with frequency on input 1
            - Capture a correlator SPEAD accumulation.
            - Confirm that auto-correlation in baseline 1 contains non-Zeros and
            - Baseline 0 is Zeros, when cw tone is only outputted on input 1.

        **Route Digitisers Raw Data Verification**

        1. The antennas interface to the same core 40Gb/s Ethernet switch as the CBF components. This same switch also provides the interfaces to all data subscribers. The switch is designed to offer a full crossbar interconnect, and so any port is able to access data from any other port at full linerate. All data products, CBF and DIG included, multicast their data into this switch. Any port may subscribe to any combination of these streams using industry-standard IGMPv2 signalling up to the full linerate capacity of the local port.
        2. The baseline correlation test proves that the CBF ingests raw digitiser data. If the baseline correlation test and the analysis in point 1 verifies this requirement.

        """
        return _description


    @property
    def ImagingDataProductSet(self):
        _description = """

        1. Configure a digitiser simulator to be used as input source to F-Engines and generate correlated noise.
        2. Set a predetermined accumulation period
            - Confirm it has been set via CAM interface.
        3. Initiate SPEAD receiver, enable data to flow and confirm CBF output product
        4. Configure the CBF to generate Baseline Correlation Products (If available)
        5. Capture Correlation Data and,
            - Confirm the number of channels in the SPEAD data.
            - Check that data product has the number of frequency channels corresponding to the instrument.
            - Confirm that data products were captured.
        """
        return _description

    @property
    def TiedArrayAuxBaselineCorrelationProducts(self):
        _description = """

        1. Configure a digitiser simulator to be used as input source to F-Engines and generate correlated noise.
        2. Set a predetermined accumulation period
            - Confirm it has been set via CAM interface.
        3. Initiate SPEAD receiver, enable data to flow and confirm CBF output product
        4. Configure the CBF to simultaneously generate Baseline Correlation Products and Tied-Array Voltage Data Products (If available)
        5. Capture Tied-Array Data and,
            - Confirm that the tide-array data were captured.
        6. Capture Correlation Data and,
            - Confirm the number of channels in the SPEAD data.
            - Check that data product has the number of frequency channels corresponding to the instrument.
            - Confirm that data products were captured.
        """
        return _description

    @property
    def TiedArrayVoltageDataProductSet(self):
        _description = """

        1. Configure a digitiser simulator to be used as input source to F-Engines and generate correlated noise.
        2. Set a predetermined accumulation period
            - Confirm it has been set via CAM interface.
        3. Initiate SPEAD receiver, enable data to flow and confirm CBF output product
        4. Configure the CBF to generate Tied-Array Voltage Data Products (If available)
        5. Capture Tied-Array Data and,
            - Confirm that the tide-array data were captured.
        """
        return _description


    @property
    def Control(self):
        _description = """
        **Control VR**

        1. The CBF shall, on request via the CAM interface, set the following parameters:
            a) Downconversion frequency
            b) Channelisation configuration
            c) Accumulation interval
            d) Re-quantiser settings (Gain)
            e) Complex gain correction
            f) Polarisation correction.
        2. The CBF shall, on request via the CAM interface, report the requested setting of each control parameter.
        """
        return _description

    @property
    def TimeSync(self):
        _description = """
        **Time synchronisation**

        1. Request NTP pool address used,
            - Confirm that the CBF synchronised time is within 0.005s of UTC time as provided via PTP (NTP server) on the CBF-TRF interface.
        """
        return _description

    @property
    def VoltageBuffer(self):
        _description = """

        1. Configure a digitiser simulator to be used as input source to F-Engines
        2. Configure a digitiser simulator to generate continuous wave
        3. Set a predetermined accumulation period
            - Confirm it has been set via CAM interface.
        4. Initiate SPEAD receiver, enable data to flow and confirm CBF output product
        5. Check that Transient Buffer ready is implemented via CAM interface.
        6. Capture an ADC snapshot via CAM interface and,
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
        10. Iteratively set accumulation length and confirm if the right accumulation is set on the SPEAD accumulation,

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
        5. Halt the CBF and confirm that SPEAD packets are either no longer being produced, or that the data content is at least affected.
        6. Re-initialise the CBF and,
            - Confirm that SPEAD packets are being produced, with the selected data product(s).
        7. Stop timer and
            - Confirm data product switching time is less than 60 seconds (Data Product switching time = End time - Start time.)
        8. Repeat for all combinations of available data products, including the case where the "new" data product is the same as the "old" one.
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
            - Verify that the calculated channel frequency step size is within requirement
            - Confirm the channelisation spacing and confirm that it is within the maximum tolerance.
        8. Sweep the digitiser simulator over the centre frequencies of at least all the channels that fall within the complete L-band
            - Capture channel response for every frequency channel in the selected frequencies calculated
        9. Check FFT overflow and QDR errors after channelisation.
        10. Check that the peak channeliser response to input frequencies in central 80% of the test
            channel frequency band are all in the test channel.
        11. Check that VACC output is at < 99% of maximum value, if fails then it is probably over-ranging.
        12. Check that ripple within 80% of cut-off frequency channel is < 1.5 dB
        13. Measure the power difference between the middle of the center and the middle of the next adjacent bins and confirm that is > -53dB
        14. Check that response at channel-edges are -3 dB relative to the channel centre at selected freq, actual source frequency
        15. Check that relative response at the low band-edge is within the range of -6 +- 1% relative to channel centre response.
        16. Check that relative response at the high band-edge is within the range of -6 +- 1% relative to channel centre response.
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
        **CBF Power Consumption**

        1. Configure the CBF to produce the imaging data product.
        2. Record power consumption for all relevant PDUs over a period of more than 60 minutes.
        3. Check that the difference in current per phase is less than 5A.
        4. If the difference is more than 5A check that all phase currents are within 15% of each other.
        5. The average power per rack test passes if the average peak power is <= 6.25kW.
        6. Sum the average power per rack to get a CBF average peak power.
        7. The CBF peak and average power test passes if the CBF power is <= 60kW.
        8. Divide the CBF average power by the number of CBF racks that are actually used, to get a CBF average power per rack.
        9. The CBF maximum heat generation test passes that the CBF average power per rack is <= 5kW.
        """
        return _description

    @property
    def CBF_Delay_Phase_Compensation_Control(self):
        _description = """

        1. Configure a digitiser simulator to be used as input source to F-Engines.
        2. Configure a digitiser simulator to generate correlated Gaussian noise.
        3. Set a predetermined accumulation period, and
            - Confirm it has been set via CAM interface.
        4. Initiate SPEAD receiver, enable data to flow and confirm the CBF output product.
        5. Confirm that the user can disable and/or enable Delays and/or Phase changes via CAM interface.
        6. Set delays/phase changes via CAM interface, and
            - Confirm that the time it takes to set the delays/phases is below 1 seconds.
        """
        return _description

    @property
    def CBF_Delay_Phase_Compensation(self):
        _description = """

        1. Configure a digitiser simulator to be used as input source to F-Engines.
        2. Configure a digitiser simulator to generate correlated Gaussian noise.
        3. Set a predetermined accumulation period, and
            - confirm it has been set via CAM interface.
        4. Initiate SPEAD receiver, enable data to flow and,
            - confirm CBF output product
        5. Change current CBF input labels and confirm via CAM interface.
        6. Clear all coarse and fine delays for all inputs, and
            - confirm if all previously applied delays have been reset
        7. Retrieve initial SPEAD accumulation, in-order to calculate all relevant parameters.
        8. Get list of all the baselines present in the correlator output
        9. Select random input and baseline for testing

        **Delay Tracking**

        1. Set time to apply delays to x integrations/accumulations in the future.
        2. Compile list of delays to be set (iteratively) for testing purposes
        3. Iterate through preselected, execute delays via CAM interface and calculate the amount of time it takes to load the delays
            - Confirm delays have been set
            - Calculate the time it takes to load delay/fringe(s), value should be less than 1s as per requirement
            - Capture SPEAD accumulation (while discarding x dumps) containing the change in delay(s) on selected input
        4. Iteratively, with captured SPEAD accumulations,
            - Check that if difference expected and actual phases are equal at delay 0.0ns within 1.0 degree.
            - Check that the maximum difference between expected phase and actual phase between integrations is less than 1.0 degree.
            - Check that when a delay of x clock cycle is introduced there is a phase change of x degrees as expected to within 1.0 degree.

        **Delay Rate**

        1. Set time to apply delays to x integrations/accumulations in the future.
        2. Request Delay(s) Corrections via CAM interface.
            - Confirm delays have been set
            - Calculate the time it takes to load delay/fringe(s), value should be less than 1s as per requirement
        3. Capture SPEAD accumulation containing the change in phase on selected input and discard all irrelevant accumulations.
        4. For all subsequent SPEAD accumulations captured,
            - Observe the change in the phase slope, and confirm the phase change is as expected.
            - Check if difference between expected phases and actual phases are 'Almost Equal', within 1 degree when a delay rate is applied.
            - Check that the maximum difference between expected phase and actual phase between integrations is less than 1 degree.

        **Fringe Offset**

        1. Set time to apply delays to x integrations/accumulations in the future.
        2. Request delay(s) corrections via CAM interface.
            - Confirm delays have been set,
            - Calculate the time it takes to load delay/fringe(s), value should be less than 1s as per requirement
        3. Capture SPEAD accumulation containing the change in phase on selected input and discard all irrelevant accumulations.
        4. For all subsequent SPEAD accumulations captured,
            - Observe the change in the phase slope, and confirm the phase change is as expected.
            - Check if difference between expected phases and actual phases are 'Almost Equal' within 1 degree when fringe offset is applied.
            - Check that the maximum difference between expected phase and actual phase between integrations is less than 1 degree

        **Fringe Rate**

        1. Set time to apply delays to x integrations/accumulations in the future.
        2. Request Delay(s) Corrections via CAM interface.
            - Confirm delays have been set
            - Calculate the time it takes to load delay/fringe(s), value should be less than 1s as per requirement
        3. Capture SPEAD accumulation containing the change in phase on selected input and discard all irrelevant accumulations.
        4. For all subsequent SPEAD accumulations captured,
            - Observe the change in the phase slope, and confirm the phase change is as expected.
            - Check if difference between expected phases and actual phases are 'Almost Equal' within 1 degree when fringe rate is applied.
            - Check that the maximum difference between expected phase and actual phase between integrations is less than 1 degree

        **Delayed Input**

        1. Set time to apply delays to x integrations/accumulations in the future.
        2. Calculate the maximum expected delay on the baseline
        3. Request Delay(s) Corrections via CAM interface, and
            - Confirm delays have been set
            - Calculate the time it takes to load delay/fringe(s), value should be less than 1s as per requirement
        4. Confirm is delay is being applied to the correct baseline

        **Delay Resolution**

        1. Set time to apply delays to x integrations/accumulations in the future.
        2. Calculate the maximum/minimum delays that can be set.
        3. Request Maximum and Minimum delay(s) corrections via CAM interface, and
            - Confirm that the maximum/minimum delays have been set
        4. TBD, For all subsequent SPEAD accumulations captured observe the change in the phase slope, and confirm the phase change is as expected.
        """
        return _description

    @property
    def BeamformerEfficiency(self):
        _description = """
        **CBF Beamformer Efficiency**

        1. Configure the beam former with zero delays and uniform taper (i.e. straight sum of N inputs per polarisation)
        2. Configure a digitiser simulator to be used as input source to F-Engines
        3. Configure a digitiser simulator to generate continuous wave (inject a frequency-swept tone)
        4. Set a predetermined accumulation period
            - Confirm it has been set via CAM interface.
        5. Initiate SPEAD receiver, enable data to flow and confirm CBF output product
        6. Calculate number of frequencies to iterate on
        7. Randomly select a frequency channel to test.
        8. Capture an initial correlator SPEAD accumulation and,
            - Determine the number of frequency channels
            - Confirm that the number of channels in the SPEAD accumulation, is equal to the number of frequency channels as calculated
            - Confirm that the Channelise total bandwidth is >= 770000000.0Hz.
            - Confirm the number of calculated channel frequency step is within requirement.
            - Verify that the calculated channel frequency step size is within requirement
            - Confirm the channelisation spacing and confirm that it is within the maximum tolerance.
        9. Sweep the digitiser simulator over the centre frequencies of at least all the channels that fall within the complete L-band
            - Capture channel response for every frequency channel in the selected frequencies calculated
        10. Measure/record the filter-bank spectral response from a channel from the output of the beamformer
        11. Determine the Half Power Bandwidth as well as the Noise Equivalent Bandwidth for each swept channel
        12. Compute the efficiency as the ratio of Half Power Bandwidth to the Noise Equivalent Bandwidth: efficiency = HPBW/NEBW
        """
        return _description

    @property
    def LBandEfficiency(self):
        _description = """
        **CBF L-band Correlator Efficiency**

        1. Configure a digitiser simulator to be used as input source to F-Engines
        2. Configure a digitiser simulator to generate continuous wave (inject a frequency-swept tone)
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
            - Verify that the calculated channel frequency step size is within requirement
            - Confirm the channelisation spacing and confirm that it is within the maximum tolerance.
        8. Sweep the digitiser simulator over the centre frequencies of at least all the channels that fall within the complete L-band
            - Capture channel response for every frequency channel in the selected frequencies calculated
        9. Measure/record the filter-bank spectral response from a channel
        10. Determine the Half Power Bandwidth as well as the Noise Equivalent Bandwidth for each swept channel
        11. Compute the efficiency as the ratio of Half Power Bandwidth to the Noise Equivalent Bandwidth: efficiency = HPBW/NEBW
        """
        return _description

    @property
    def Beamformer(self):
        _description = """
        **Beamformer Functionality**

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
            - Confirm that the expected voltage level is within 0.2dB of the measured mean value.

        12. Repeat above for different beam weights
        """
        return _description

    @property
    def StateAndModes(self):
        _description = """
        State and Modes

        1. Start with the CBF off. Turn the CBF on by turning on power to the PDU(s). Start a stopwatch.
        2. Observe that each of the LRUs powers up, loads any boot firmware, software bootloaders and operating systems, and launches the operational state software applications. This must be done completely automatically, manual intervention is not permitted.
        3. During this boot process, send a KATCP status request message every few seconds. It is permissible for the CBF to be unresponsive while initialising.
        4. Continue sending status requests every few seconds, until the CBF responds. This is considered to be an indication that the CBF has entered operational state. Record the time from when power was applied to when the CBF first responses to a status request.
        5. Confirm that each of the processing nodes has been allocated an IP address, by inspecting the leases in dnsmasq.
        6. If not, continue checking until this is the case, and record the time that it occurred.
        7. Restart the CBF by issuing the appropriate command on the CAM interface.
        8. Send a status request, and confirm that the CBF does not respond.
        9. Send a status request message every few seconds. It is permissible for the CBF to be unresponsive while initialising.
        10. Continue sending status requests every few seconds, until the CBF responds. Record in the Observations section below the time taken from reboot command to CBF responding.

        Verify Transition from Initialisation state to Fault state

        1. Turn the CBF CMC on.
        2. After five seconds, remove the CAM network cable from the CMC.
        3. Verify that the CMC remains unresponsive, indicating a Fault state.
        4. Reconnect the CAM network cable to the port on the CMC.
        5. Repeatedly issue KATCP system-info request messages every few seconds.
        6. Continue sending system-info requests every few seconds, until the CBF responds. This is considered to be an indication that the CBF has entered operational state.
        7. Verify whether all tests performed in CBF OPERATIONAL mode demonstrated full functionality.
        """
        return _description

    @property
    def PowerSupply(self):
        _description = """
        1. Confirm by inspection of the data sheets of each of the LRUs used in the CBF that the LRUs are specified to work over an input voltage range of at least 209Vrms to 231Vrms, frequency range at least 49.5Hz to 50.5Hz.
        2. Record the item part numbers and supply references to the item data sheets and/or specification used.

        NOTE: Total harmonic distortion is not tested. This is considered low enough risk to be acceptable.
        """
        return _description

    @property
    def ProcuredItemsEMCCert(self):
        _description = """
        1. Inspect the data sheets and/or specification of each of the COTS items making up the CBF ie PDU, CMC, Data Switch LRU.
        2. Confirm whether each has been EMC/RFI certified according to CISPR 22 standard for Class B devices.
        3. For each item where this is true, record the item part number and reference the item data sheets and/or specification used.
        4. For each item where this is not true, perform an analysis of the available EMC/RFI certification versus CISPR 22 standard for Class B devices. If equivalence can be argued, present this for approval to the SKA system engineer. If approved, attach the motivation and record the approval.
        5. The test is considered a pass if each item either has been EMC/RFI certified according to CISPR 22 standard for Class B devices, or has been approved as equivalent by the SKA system engineer.
        """
        return _description

    @property
    def AccumulatorDynamicRange(self):
        _description = """
        1. Provide analysis that the accumulator dynamic range requirement criteria are met.
        """
        return _description

    @property
    def ChannelisedVoltageDataTransfer(self):
        _description = """
        1. Analyse switch infrastructure to ensure users can subscribe to channelised voltage data.
        2. Confirm that the switch infrastructure provides enough bandwidth to transport channelised voltage data.
        """
        return _description

    @property
    def CoolingMethod(self):
        _description = """
        1. Confirm that the front and rear doors of the CBF racks are perforated to allow natural convection of air moving from the cold aisle to the hot aisle.
        2. Confirm that the CBF has no external cooling besides that provided by natural convection of air moving from the cold aisle to the hot aisle, and optionally active cooling (eg fans) which are mounted inside the CBF and/or inside of CBF LRUs.
        """
        return _description

    @property
    def COTSLRUStatusandDisplay(self):
        _description = """
        1. With the CBF installed in the racks and turned on, confirm whether the COTS LRUs indicate via LEDs, visible with the LRUs installed, the state of the LRU as OK.
        2. Record for each COTS LRU, whether a status LED is present or not, and the LRU part number.

        Note that this test is a desirable only, not mandatory. The test status must be marked as Passed once each of the LRUs have been checked, regardless of the result.
        """
        return _description

    @property
    def DataProductsAvailableforAllReceivers(self):
        _description = """
        1. Provide analysis that CBF data products are available for all receivers.
        """
        return _description

    @property
    def DataSubscribersLink(self):
        _description = """
        1. Verify by inspection that the CBF provides the correct number of 40GbE QSFP+ ports on the CBF Data Switch to all data subscriber groups.
        2. Record the number of ports and their locations.
        """
        return _description

    @property
    def DesigntoEMCSANSStandard(self):
        _description = """
        1. Verify by inspection that each CBF rack has a ground connection point.
        2. Verify by inspection that each CBF cabinet has at least one ESD earthing point.
        3. Measure the resistance between the chassis, rack connection points and input power chassis pin on one of each type of LRU used on the CBF. Confirm that this is less than 0.1 ohm. Record the resistance in the Observations section below.
        4. Verify by inspection that the CBF processing node has an integrated power filter on the input power line.
        5. Measure the resistance between the housing of the processing node input power filter and the processing node chassis. Confirm this is less than 0.1 ohm. Record the resistance.
        """
        return _description

    @property
    def DesigntoNRS083Standards(self):
        _description = """
        1. Verify by inspection that each CBF rack has a ground connection point.
        2. Verify by inspection that each CBF cabinet has at least one ESD earthing point.
        3. Measure the resistance between the chassis, rack connection points and input power chassis pin on one of each type of LRU used on the CBF. Confirm that this is less than 0.1 ohm. Record the resistance in the Observations section below.
        4. Verify by inspection that the CBF processing node has an integrated power filter on the input power line.
        5. Measure the resistance between the housing of the processing node input power filter and the processing node chassis. Confirm this is less than 0.1 ohm. Record the resistance.
        """
        return _description

    @property
    def DigitiserCAMData(self):
        _description = """
        1. Connect to a live digitiser from the DMC host.
        2. Request sensor information.
        3. Confirm sensor information is valid.
        """
        return _description

    @property
    def ExternalInterfaces(self):
        _description = """
        1. With the CBF installed in the KAPB, confirm that no more than 16 racks have been used to house the complete CBF.
        2. Record the number of racks used.
        """
        return _description

    @property
    def FailSafe(self):
        _description = """
        1. Confirm by inspection of the datasheets of each of the COTS LRUs making up the CBF, that each LRU has an overtemperature protection mechanism.
            - NOTE: if this information is not contained in the datasheets, direct confirmation from the supplier is acceptable.
        2. Record the item part number and reference relevant page of the datasheet used, or a printout of the supplier confirmation.
        3. Confirm by inspection of the CBF processing node verification results that the overtemperature shutdown function passed. Provide a reference to the processing node test result document.
        """
        return _description

    @property
    def FullFunctionalMode(self):
        _description = """
        1. Refer to CBF functional tests and confirm that the tests were run in the OPERATIONAL state.
        """
        return _description

    @property
    def Humidity(self):
        _description = """
        The August-Roche-Magnus approximation related temperature (T), dew point (TD) and relative humidity (RH) with the equations:

            - RH: = 100 * (EXP((17.625*TD) / (243.04+TD)) / EXP((17.625*T) / (243.04+T)))
            - TD: = 243.04 * (LN(RH/100)+((17.625*T)/(243.04+T))) /(17.625-LN(RH/100)-((17.625*T)/(243.04+T)))
            - T: = 243.04 * (((17.625*TD)/(243.04+TD))-LN(RH/100)) /(17.625+LN(RH/100)-((17.625*TD)/(243.04+TD)))

        Applying these equations to the CBF humidity requirements, over the CBF temperature range, the requirements can be satisfied by the CBF having operational humidity range between 25% and 82%, non condensing.

        1. Inspect the data sheets and/or specification of each of the COTS items making up the CBF ie PDU, CMC, Data Switch LRU.
        2. Confirm whether each has an operational humidity specification better than 25% to 82%, non condensing.
        3. Record the item part numbers in the Observations section below, and attach the item data sheets and/or specification used.

        The humidity environment for the CBF processing node is not specified. However, the specified transport environment is harsher than the KAPB operational environment, and the result of the transport environment verification is used as verification of the humidity environment requirement.

        1. Inspect the verification results of the processing node.
        2. Confirm that the transport environment verification passed.
        """
        return _description

    @property
    def Interchangeability(self):
        _description = """

        Confirm by inspection of the CBF physical item structure, as captured in the latest approved Correlator-Beamformer Design Document, that:

        1. The architecture enables Data Switch LRUs with the same part number and version to be interchangeable with no calibration, tuning or alignment, after configuration files and/or software and/or firmware to the switch have been loaded.
        2. The architecture enables processing node LRUs with the same part number and version to be interchangeable with no calibration, tuning or alignment, after configuration files and/or software and/or firmware to the processing node have been loaded.
        3. The architecture enables PDU and/or CMC LRUs with the same part number and version to be interchangeable with no calibration, tuning or alignment, after configuration files and/or software and/or firmware to the PDU and/or CMC have been loaded.
        """
        return _description

    @property
    def InternalInterfaces(self):
        _description = """
        1. Verify by inspection that the CBF does provide a 19 rack mounting space for the DMC, with height at least 3U, full rack width, and depth at least 1000mm.
        2. Verify by inspection that the output power connectors between the CBF and the DMC are each located in one of the CBF racks, on flyleads with connector specification:
            - Connector type: IEC 60320-2 Standard, type C13 (kettle cord connector)
            - Quantity: Four (4)
        3. Verify by inspection that each power pair is either sourced from a different phase bank of a single PDU, or from different phase bank of separate PDUs.
        4. Verify by inspection that the physical link between the DMC and the CBF Data Switch is via one 10GBASE-CR STP cable, QSPF+ to 4xSPF+, length =3m.
        5. Verify by inspection that the physical link between the DMC and the CBF CAM switch is via one 1000BASE-T STP cable, Cat6a, RJ45 to RJ45, length =2m.
        6. Verify by inspection that the CBF does provide a 19 rack mounting space for M&C switches, with height at least 3U, full rack width, and depth at least 1000mm.
        7. Verify by inspection that the output power connectors between the CBF and the M&C switches are each be located in one of the CBF racks, on fly-leads with connector specification:
           - Connector type: IEC 60320-2 Standard, type C13 (kettle cord connector)
           - Quantity: Six (6)
        8. Verify by inspection that each power pair is either sourced from a different phase bank of a single PDU, or from different phase bank of separate PDUs.
        9. Verify by inspection that the physical link between the M&C switches and the DMC is one 1000BASE-T STP cable, Cat6a, RJ45 to RJ45, length =2m.

        """
        return _description

    @property
    def ItemHandling(self):
        _description = """
        1. Confirm by inspection of the CBF physical item structure that all items weighing between 15kg and 40kg have handles and that those weighing over 40kg have suitable lifting arrangements.
        """
        return _description

    @property
    def ItemMarkingandLabelling(self):
        _description = """
        1. Confirm by inspection that each type of COTS LRU used in the CBF is labelled with the following information
            - Product Supplier Name
            - Product Serial Number.
        2. Confirm by inspection that the above information is visible with the LRU installed in a rack. It is permissible that either the front or rear door of the CBF rack be opened to read the information.
        3. Confirm by inspection that Processing Node LRUs are labelled with the following information:
            - Product Supplier Name
            - Product Name
            - Product Part Number
            - Product Version
            - Product Serial Number.
        4. Confirm by inspection that the above information is visible with the processing node installed. It is permissible that either the front or rear door of the CBF rack be opened to read the information.
        5. Remove the top lid of a processing node.
        6. Confirm by inspection that each type of Mezzanine SRUs are labelled with the following information:
            - Product Supplier Name.
            - Product Name
            - Product Part Number
            - Product Version
            - Product Serial Number.
        7. Confirm by inspection that the above information is visible with the only the processing node top lid removed. It is not permissible to have to unplug the card to read the information.
        8. Confirm by inspection that each port on the LRU making up the CBF Data Switch is labelled with at least the port number.
        9. Determine whether each CBF LRU contains any hazardous material. This would typically be lead and/or heatsink paste.
        10. For each identified LRU containing hazardous material, confirm by inspection that the LRU has a Hazardous Waste label.
        11. Confirm by inspection that each CBF internal cable (ie cables between CBF LRUs) is labelled.

        """
        return _description

    @property
    def Logging(self):
        _description = """
        1. Turn the CBF on and wait for it to boot.
        2. ssh into the CMC.
        3. Watch the current log file using command tail -f <filename>.katlog
        4. Via the CAM interface, request the current log level using KATCP command ?log-level
        5. Confirm the CBF replies with #log-level ok <>, where <> contains one of off, fatal, error, warn, info, debug, trace, all.
        6. Set the log level using KATCP command ?log-level info
        7. Confirm the CBF replies with #log-level ok info
        8. Send an invalid KATCP command eg ?guesswhatthisis
        9. Confirm that a message is logged to the log file that an unknown command was received.
        10. Set the log level using KATCP command ?log-level debug
        11. Confirm the CBF replies with #log-level ok debug
        12. Confirm that messages are being logged periodically, as the CBF is being used.
        13. Set the log level using KATCP command ?log-level fatal
        14. Confirm the CBF replies with #log-level ok fatal
        15. Set the log level back to the setting which it was at the start of the test.
        16. Peruse archived log files and confirm that the log file contains more than 1000 lines.
        """
        return _description

    @property
    def LRUReplacement(self):
        _description = r"""
        The CBF Mean Time To Repair test is accomplished by means of demonstration.
        This must be carried out by a maintenance person rather than a development person, in order to be realistic.

        1. Select the most difficult CBF LRU to remove. The choice is left to the person conducting the test.
        2. Start a timer.
        3. Remove the selected LRU.
        4. Once the LRU has been completely removed, replace the LRU, including refitted all connections.
        5. Run tests to confirm that the CBF is operating correctly.
        6. Stop the timer once the CBF has been confirmed operational.
        7. Confirm that the elapsed time is less than 8 hours. Record the achieved MTTR in the Observations section below.
        8. Repeat this procedure for a processing node and with the processing node removed confirm that the CBF is still operational by starting an instrument.
        """
        return _description

    @property
    def LRUStatusandDisplay(self):
        _description = """
        1. Inspect CBF processing node physical hardware and documentation to confirm that the processing nodes will suitably display faults.
        """
        return _description

    @property
    def LRUStorage(self):
        _description = r"""
        Confirm by inspection that:

        1. The transport packaging containers defined in the CBF structure are CLIP-LOCK containers.
        2. A label, or set of labels, for these containers have been defined, each containing at least the following information:
            - "Fragile" label
            - "Careful transportation"
            - "This side up"
            - Packaged Item Identification label with place for filling in of identification data
        3. A Packaged Item Identification label has been defined when use to carry ROACHs, containing at least the following information:
            - Identification "MeerKAT System Component"
            - LRU Name
            - LRU/SRU Part Number and Version
            - Packaged weight
            - Container stack-ability (if applicable)
        4. A Packaged Item Identification plate has been defined when use to carry COTS LRUs containing at least the following information:
            - LRU Name
            - LRU/SRU Part Number and Version
            - Packaged weight
            - Container stack-ability (if applicable)
        5. Record the part number of the Clip-Lock container(s).
        6. Record the part number of the label, or labels if multiple are defined.

        Note: A single Packaged Item Identification label catering for both options is permissible.
        A single label with space for all of the above information is also permissible.
        """
        return _description

    @property
    def MTBF(self):
        _description = """
        1. Determine the MTBF by inspection of the data sheets of each of the LRUs used in the CBF.
        2. Collate all the MTBF figures and confirm that the full system adheres to the specification.
        3. Record all LRU MTBF figures.
        """
        return _description

    @property
    def PeriodicMaintenanceLRUStorage(self):
        _description = """
        Confirm by inspection of the CBF physical item structure, as captured in the latest approved M1200-0000-003 Correlator-Beamformer Design Document, that the CBF does not contain items which require periodic maintenance.

        - If this is the case, the test can be marked as Passed, since the requirement is no longer applicable.
        - If this is not the case, the test must be marked as Failed until a suitable test has been defined.
        """
        return _description

    @property
    def ProductMarkingEnvironmentals(self):
        _description = """
        The test for the label environmental is to inspect the labels for damage at least one year after the CBF has been deployed to the KAPB.
        That is considered sufficient time for each of the labels to have been exposed to the storage, transport and operational environment to demonstrate the robustness of the labels.
        Between one year and two years after the first CBF has been deployed to the KAPB:

        1. Inspect the labels on each of the items making up the CBF ie PDU, CMC, Processing Nodes, Data Switch LRU.
        2. Confirm that the equipment labels are still attached.
        3. Confirm that the information on the equipment labels is still legible.
        4. Inspect the labels on at least one of each of the cable types in the CBF racks.
        5. Confirm that the cable labels are still attached.
        6. Confirm that the information on the cable labels is still legible.
        7. Record the item part numbers and versions.
        8. It is desirable to attach photographs of the labels.
        """
        return _description

    @property
    def RouteBasicSpectrometerData(self):
        _description = """
        The generation of basic Spectrometer data and Digitisers raw data are functions of the digitiser, not the CBF.
        The CBF merely has to route this data to subscribed users.
        The test thus consists of checking that data from ports assigned to Digitiser inputs can be routed to ports assigned to SP/USE and that the added traffic load will not saturate the network links.

        1. Determine the bandwidth required for Basic Spectrometer Data.
        2. Determine the maximum bandwidth per Digitiser link during normal operation.
        3. Verify the the link will not be saturated when routing Basic Spectrometer Data.
        """
        return _description

    @property

    def SafeDesign(self):
        _description = """
        1. Inspect the Bill-of-Materials of the processing nodes used in the CBF.
            - Confirm that each part used is ROHS compliant, either with or without exception 7b. ROHS 6 and/or ROHS 5 is compliant with this.
        2. Inspect the data sheets of the COTS LRUs used in the CBF.
            - Confirm that LRU is ROHS compliant, either with or without exception 7b. ROHS 6 and/or ROHS 5 is compliant with this.
        3. Record the item part numbers and supply references to the item data sheets or BOM used.
        """
        return _description

    @property
    def SafePhysicalDesign(self):
        _description = """
        With the CBF installed and running in the KAPB:

        1. Connect to a CBF rack using an anti-static wrist strap.
        2. Run your hands over the accessible external edges of the rack, confirming that fingers/hands can be run lightly over all accessible edges without causing cuts.
        3. Confirm that accessible external edges of the rack can be touched comfortably i.e. the temperature is below 80 Degrees Celsius.
        4. Open the front door of each rack. Repeat the sharp edges and surface temperature test on the installed equipment on rack internal surfaces and edges which are now accessible.
        5. Open the back door of each rack. Repeat the sharp edges and surface temperature test on rack internal surfaces and edges which are now accessible.
        6. Close the rack doors.
        7. Repeat for each CBF rack which is utilised.
        """
        return _description

    @property
    def StorageEnvironment(self):
        _description = """
        ETSI EN 300 019-1-1, Class 1.1 specifies a storage environment of:
            - humidity between 5% to 95%
            - temperature between 5 Degrees Celsius and 45 Degrees Celsius
            - condensation
            - no precipitation
            - no icing

        CBF items will be stored in protective boxes, which will protect against condensation. The test therefore checks just for on-operational temperature and humidity ranges of CBF LRUs.

        1. Inspect the data sheets and/or specification of each of the COTS items making up the CBF ie PDU, CMC, Data Switch LRU.
        2. Confirm whether each has storage temperature between 5 Degrees Celsius and 45 Degrees Celsius or better, and non-operational humidity between 5% to 95%or better.
        3. For each item where this is true, record the item part number and reference the item data sheets and/or specification used.

        The storage environment for the CBF processing node is not specified. However, the specified transport environment is harsher than the storage environment, and the result of the transport environment verification is used as verification of the storage environment requirement.

        1. Inspect the verification results of the CBF processing node.
        2. Confirm that the transport environment verification passed.
        """
        return _description

    @property
    def SubArrayDataProductSet(self):
        _description = """
        1. Analyse test results of each data product listed and verify that these are produced.
        """
        return _description

    @property
    def TemperatureRange(self):
        _description = """
        1. Inspect the data sheets and/or specification of each of the COTS items making up the CBF ie PDU, CMC, Data Switch LRU.
        2. Confirm whether each has an operational temperature specification better than 18 Degrees Celsius to 35 Degrees Celsius
        3. Record the item part number and reference the item data sheets and/or specification used.
        """
        return _description

    @property
    def TransportationofComponents(self):
        _description = """
        After deployment of a number of (at least 12) processing nodes to the KAPB and other LRU infrastructure:

        1. If any processing nodes or LRUs failed after transportation, analyse the failure to determine whether vibration was the cause of failure.
        2. The test passes if no processing nodes failed, or if vibration was not the cause of any failures.
        """
        return _description

    @property
    def UseofCOTSEquipment(self):
        _description = """
        Confirm by asking the opinion of the CBF Technical lead, that:

        1. The CBF design utilises largely commercial off the shelf (COTS) equipment
        2. Where possible, COTS components with long support life expectancy have been chosen.

        Note that this test is highly subjective. The opinion of the CBF Technical Lead is sufficient to pass the test.
        """
        return _description



TestProcedure = TestProcedure()

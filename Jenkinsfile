node {
    ansiColor('xterm') {
        // Just some echoes to show the ANSI color.
        stage "\u001B[31mI'm Red\u001B[0m Now not"
    stage "Install Dependencies"
    sh  "#!/bin/bash \n" +
        "bash scripts/Jenkins_Scripts/install_dep_1.sh"

    stage "Verify Dependencies"
    sh  "#!/bin/bash \n" +
        "bash scripts/Jenkins_Scripts/check_dep_2.sh"

    stage "Instrument Initialisation"
    sh "#!/bin/bash \n" +
       "bash scripts/Jenkins_Scripts/init_instrument_3.sh"

    stage "Correlator Rx Verification"
    sh "#!/bin/bash \n" +
       "bash scripts/Jenkins_Scripts/rx_test_4.sh"

    stage "Tests"
    sh "#!/bin/bash \n" +
       "bash scripts/Jenkins_Scripts/run_test_5.sh"
    }

}
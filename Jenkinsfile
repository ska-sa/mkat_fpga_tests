node ("cmc") {
    ansiColor('xterm') {
    stage "Install Dependencies"
    sh  "#!/bin/bash \n" +
        "source .bash_configs \n" +
        "bash scripts/Jenkins_Scripts/1.install_dependencies.sh"

    stage "Verify Dependencies"
    sh  "#!/bin/bash \n" +
        "bash scripts/Jenkins_Scripts/2.verify_dependencies.sh"

    stage "Instrument Initialisation"
    sh "#!/bin/bash \n" +
       "bash scripts/Jenkins_Scripts/3.initialise_instrument.sh"

    stage "Correlator Rx Verification"
    sh "#!/bin/bash \n" +
       "bash scripts/Jenkins_Scripts/4.corr2rx_test.sh"

    stage "Tests"
    sh "#!/bin/bash \n" +
       "bash scripts/Jenkins_Scripts/5.run_tests.sh"
    }

}


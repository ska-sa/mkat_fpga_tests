#!/bin/bash
# Automated python dependencies retrieval and installation,
# This script will install all CBF-Testing dependencies namely:-
# 		spead2, corr2, casperfpga and katcp-python
# These packages will be cloned from their git repositories and installed in /usr/local/src which
# is a dependency default directory for CBF-Testing, and assuming that the latest CMC Software is installed and functional
# Mpho Mphego <mmphego@ska.ac.za>

# Abort on any errors
set -e
. venv/bin/activate
# Special spead2 hacksauce for dbelab04 with old GCC. This will be
# fixed properly by upgrading the test system GCC
export PATH=/opt/gcc4.9.3/bin:"${PATH}"

declare -a PYTHON_SRC_PACKAGES=(spead2 casperfpga corr2 katcp-python )
PYTHON_SRC_DIR=/usr/local/src
SPEAD2_URL=https://pypi.python.org/packages/a1/0f/9cf4ab8923a14ff349d5e85c89ec218ab7a790adfdcbd11877393d0c5bba/spead2-1.1.1.tar.gz
PYTHON_SETUP_FILE=setup.py

function pip_installer {
	if [ "${pkg}" = 'katcp-python' ];then
		pkg=katcp
	fi
	export PYTHON_PKG="${pkg}"
	if python -c "import os; pypkg = os.environ['PYTHON_PKG']; __import__(pypkg)" &> /dev/null; then
		printf 'Package already installed\n';
	else
        printf "Installing %s \n" "${pkg}"
		echo "${INSTALL_DIR}"
		cd "${INSTALL_DIR}"
		if [ ! -f "${PYTHON_SETUP_FILE}" ]; then
	    	printf "Python %s file not found!\n" "${PYTHON_SETUP_FILE}"
			continue
		else
            pip install -e .
            # NO SUDOing when automating
			# sudo python setup.py install --force
			printf "Successfully installed %s in %s\n" "${pkg}" "${INSTALL_DIR}"
		fi
    fi
}

function spead2_installer {
	cd "${INSTALL_DIR}"
	printf "Installing %s\n" "${pkg}"
    # NO SUDOing when automating
	# env PATH=$PATH sudo pip install -v .
    env PATH=$PATH pip install -e .
	printf "Successfully installed %s in %s\n" "${pkg}" "${INSTALL_DIR}"
}

for pkg in "${PYTHON_SRC_PACKAGES[@]}"; do
	INSTALL_DIR="${PYTHON_SRC_DIR}"/"${pkg}"
	if [ "${pkg}" = 'spead2' ]; then
		if [ -d "${INSTALL_DIR}" ]; then
			printf "%s directory exists.\n" "${pkg}"
            export PYTHON_PKG="${pkg}"
            if python -c "import os; pypkg = os.environ['PYTHON_PKG']; __import__(pypkg)" &> /dev/null; then
                printf 'Package already installed\n';
            else
                spead2_installer
            fi
		else
			printf "%s directory doesnt exist cloning.\n" "${pkg}"
			mkdir -p "${INSTALL_DIR}" && cd "$_"
			curl -s "${SPEAD2_URL}" | tar zx
			mv "${pkg}"* "${pkg}"
            export PYTHON_PKG="${pkg}"
            if python -c "import os; pypkg = os.environ['PYTHON_PKG']; __import__(pypkg)" &> /dev/null; then
                printf 'Package already installed\n';
            else
                spead2_installer
            fi
		fi
	elif [ -d "${INSTALL_DIR}" ]; then
		printf "%s directory exists.\n" "${pkg}"
		pip_installer
	else
		printf "%s directory doesnt exist cloning.\n" "${pkg}"
		$(which git) clone git@github.com:ska-sa/"${pkg}".git "${PYTHON_SRC_DIR}"/"${pkg}" && cd "$_"
		pip_installer
	fi
	printf "\n"
done

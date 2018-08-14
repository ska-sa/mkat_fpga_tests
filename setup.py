#!/usr/bin/env python
# https://stackoverflow.com/a/44077346
import atexit
import os
import subprocess

from contextlib import contextmanager
from distutils.command.install import install
from glob import glob

# https://stackoverflow.com/a/49867265
try:
    from pip.download import PipSession
except ImportError:
    from pip._internal.download import PipSession

try:
    from pip.req import parse_requirements
except ImportError:
    from pip._internal.req import parse_requirements

from setuptools import setup
from setuptools import find_packages
from warnings import filterwarnings


# I do not condone it, but we suppressing all warnings
filterwarnings('ignore')

# Latest tagged stable version
try:
    __version__ = subprocess.check_output(
        ["git", "describe", "--tags"]).rstrip().split('-')[-1]
except subprocess.CalledProcessError:
    __version__ = '0.5'

# Install dependencies with failover/failsafe
try:
    install_reqs = parse_requirements("pip-requirements.txt",  session=PipSession())
    __install_requires__ = [str(ir.req) for ir in install_reqs]
    try:
        __install_requires__ = filter('None'.__ne__, __install_requires__)
    except ValueError:
        pass
except Exception:
    __install_requires__ = [
        'cryptography',
        'matplotlib',
        'memory_profiler',
        'ntplib',
        'numpy',
        'sphinx',
        'h5py',
        'Nose',
        'ansicolor',
        'traceback2']


@contextmanager
def cwd(path):
    oldpwd = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(oldpwd)


class cmdatexit_install(install):
    """"""

    def run(self):
        path = ''
        install.run(self)
        if os.path.exists(path):
            with cwd(path):
                print 'Installing %s from %s' % (path, os.getcwd())


setup(name='mkat_fpga_tests',
      version=__version__,
      description='Tests for MeerKAT signal processing FPGAs ',
      long_description=open('README.md').read(),
      license='GPL',
      author='SKA SA DBE Team',
      author_email='mmphego@ska.ac.za',
      url='https://github.com/ska-sa/mkat_fpga_tests',
      classifiers=[
          'Development Status :: 3 - Alpha',
          'Intended Audience :: Radio Telescope correlator builders',
          'Operating System :: OS Independent',
          'License :: OSI Approved :: GNU General Public License (GPL)',
          'Topic :: Scientific/Engineering :: Astronomy',
      ],
      install_requires=__install_requires__,
      dependency_links=['https://github.com/ska-sa/nosekatreport'],
      provides=['mkat_fpga_tests'],
      packages=find_packages(),
      scripts=glob('scripts/*'),
      # cmdclass={"install": cmdatexit_install, },
      )

# end

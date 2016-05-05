###############################################################################
# SKA South Africa (http://ska.ac.za/)                                        #
# Author: cam@ska.ac.za                                                       #
# Copyright @ 2013 SKA SA. All rights reserved.                               #
#                                                                             #
# THIS SOFTWARE MAY NOT BE COPIED OR DISTRIBUTED IN ANY FORM WITHOUT THE      #
# WRITTEN PERMISSION OF SKA SA.                                               #
###############################################################################
import sys

from setuptools import setup, find_packages


setup(
    name='nose-kat-report',
    version='0.2',
    description='Nose plugin for writing an annoted test report',
    long_description=open('README.rst').read(),
    author='Neilen Marais',
    author_email='nmarais@ska.ac.za',
    license='GPL',
    packages=find_packages(exclude=['ez_setup']),
    install_requires=['Nose>=0.11.0', 'traceback2', 'ansicolors'],
    url='',
    include_package_data=True,
    entry_points="""
        [nose.plugins.0.10]
        nosekatreport = nosekatreport:KatReportPlugin
        """,
    classifiers = [
        'Development Status :: 2 - Pre-Alpha',
        'Environment :: Console',
        'Intended Audience :: Developers',
        'Natural Language :: English',
        'License :: OSI Approved :: GNU General Public License (GPL)',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.5',
        'Programming Language :: Python :: 2.6',
        'Programming Language :: Python :: 2.7',
        'Topic :: Software Development :: Testing',
        'Topic :: Software Development :: Documentation'
        ],
)

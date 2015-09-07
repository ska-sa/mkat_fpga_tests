import unittest
import logging
import os

import matplotlib
import matplotlib.pyplot as plt

from nosekatreport import Aqf
from nosekatreport import aqf_vr

from mkat_fpga_tests import cls_end_aqf

@cls_end_aqf
class test_Something(unittest.TestCase):

    @aqf_vr('TP.C.1.19')
    def test_channelisation(self):
        """CBF Channelisation test"""
        Aqf.step("Frobbing the Nitz all the way to see that it pootles")
        Aqf.equals('frobway', 'froball', 'Verify that frob is froball')
        Aqf.step('Check boo baz')
        Aqf.equals('boo baz', 'boo baz', 'Check that the boo baz has the right value')


    @aqf_vr('TP.C.1.19234')
    def test_blah(self):
        """Blah test"""
        Aqf.step('Blap blap')
        Aqf.is_true(True, 'Confirm the Blah is Blapping')
        os.system('touch blah.png')
        Aqf.image('blah.png', 'the caption', 'an alternative')
        plt.plot(range(4))
        plt.title('The title')
        plt.xlabel('x axis')
        plt.ylabel('y axis')
        Aqf.matplotlib_fig('blieh.svg', caption='A caption', alt='hiehiehie')

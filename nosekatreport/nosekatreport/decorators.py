###############################################################################
# SKA South Africa (http://ska.ac.za/)                                        #
# Author: cam@ska.ac.za                                                       #
# Copyright @ 2013 SKA SA. All rights reserved.                               #
#                                                                             #
# THIS SOFTWARE MAY NOT BE COPIED OR DISTRIBUTED IN ANY FORM WITHOUT THE      #
# WRITTEN PERMISSION OF SKA SA.                                               #
###############################################################################
__all__ = ['satisfies_requirement', 'satisfies_vr',
           'site_only', 'site_acceptance',
           'system', 'aqf_requirements', 'aqf_vr', 'intrusive', 'slow']


def grand_decorator(name, gd_func, *gd_args):

    def decorator(func, *args, **kwargs):
        # end in here if decorator was assigned like @deco()
        setattr(func, "aqf_%s" % name, 1)
        return func

    if gd_func:
        # end in here if decorator was assigned like @deco
        setattr(gd_func, "aqf_%s" % name, 1)
        return gd_func
    else:
        return decorator


def satisfies_requirement(requirement):
    """Single requirement"""
    def decorator(func):
        try:
            func.katreport_requirements.append(requirement)
        except AttributeError:
            func.katreport_requirements = [requirement,]
        return func
    return decorator

def satisfies_vr(ver_requirement):
    """Single verification requirement"""
    def decorator(func):
        try:
            func.katreport_ver_requirements.append(ver_requirement)
        except AttributeError:
            func.katreport_ver_requirements = [ver_requirement,]
        return func
    return decorator


def slow(*args):
    """Tag the test method or object as a slow tests.

    Usage example on test method: ::

        @slow()
        def test_01_one(self):

    Usage example on test class: ::

        @slow()
        class TestOne(unittest.TestCase):

    This decorator can be assigned by @slow or @slow()

    :return: Function.

    """
    return grand_decorator('slow', *args)


def intrusive(*args):
    """Tag the test method or object as intrusive.

    Intrusive tests are typicaly tests that do things like change values
    on simulators, delete files.
    This test will **NOT** run onsite(in the karoo).

    Usage example on test method: ::

        @intrusive()
        def test_01_one(self):

    Usage example on test class: ::

        @intrusive()
        class TestOne(unittest.TestCase):

    This decorator can be assigned by @intrusive or @intrusive()

    :return: Function.

    """
    return grand_decorator('intrusive', *args)
    #def decorator(func):
    #    setattr(func, 'aqf_intrusive', 1)
    #    setattr(func, "aqf_system_karoo", 0)
    #    return func
    #return decorator

def site_acceptance(*args):
    """Tag the test method or object as part of the site acceptance test.

    This test is part of the site acceptance tests.

    Usage example on test method: ::

        @site_acceptance()
        def test_01_one(self):

    Usage example on test class: ::

        @site_acceptance()
        class TestOne(unittest.TestCase):

    This decorator can be assigned by @site_acceptance or @site_acceptance()

    :return: Function.

    """
    return grand_decorator('site_acceptance', *args)


def site_only(*args):
    """Tag the test method or object as site only.

    This test will only run onsite (in the karoo).

    Usage example on test method: ::

        @site_only()
        def test_01_one(self):

    Usage example on test class: ::

        @site_only()
        class TestOne(unittest.TestCase):

    This decorator can be assigned by @site_only or @site_only()

    :return: Function.

    """
    return grand_decorator('site_only', *args)


def system(*systems, **system_kwargs):
    """Decorator to indicate to which systems this test applies.

    Method and object decorator.

    Usage example on method: ::

        @system('mkat', 'mkat_rts', 'kat7')
        def test_wind_speed_received_form_anc(self):

    Usage example on object: ::

        @system('all')
        class TestOne(unittest.TestCase):

    Advanced usage: (get out of jail)
        Eg. Disable the all selection and enable mkat and kat7 ::

        @system('mkat', all=False, kat7=True)

    :params: String. Systems where this test can be run. eg. All, Mkat
    :return: Function.

    """
    def decorator(func):
        if not hasattr(func, 'aqf_systems'):
            func.katreport_systems = []
        for system_name in systems:
            setattr(func, "aqf_system_%s" % str(system_name).lower(),
                    True)
        for system_name in system_kwargs:
            setattr(func, "aqf_system_%s" % str(system_name).lower(),
                    system_kwargs[system_name] and True)
        return func
    return decorator


def aqf_requirements(*requirements):
    """Decorator to add requirements to a test method.

    Eg. aqf_requirements("VR.CM.EXAMPLE.1")

    :param requirements: The positional arguments will be unpacked as a
                         list of arguments
    :return: function

    """
    return aqf_vr(*requirements)


def aqf_vr(*requirements):
    """Decorator to add verification requirements to a test method.

    Eg. aqf_vr("VR.CM.SITE.XX.1", "VR.CM.SITE.YY.2")
    Eg. aqf_vr("VR.CM.AUTO.XX.21")
    Eg. aqf_vr("VR.CM.DEMO.ZZ.33")

    :param requirements: The positional arguments will be unpacked as a
    list of arguments

    :return: function

    """
    def decorator(func):
        try:
            for requirement in requirements:
                if requirement.startswith("VR.") and ".SITE." in requirement:
                    setattr(func, "aqf_site_test", True)
                    setattr(func, "aqf_site_acceptance", True)
                elif requirement.startswith("VR.") and ".DEMO." in requirement:
                    setattr(func, "aqf_demo_test", True)
                else:
                    # All other tests are AUTO tests
                    setattr(func, "aqf_auto_test", True)
                func.katreport_requirements.append(requirement)
        except AttributeError:
            func.katreport_requirements = list(set(requirements))
        return func
    return decorator
#

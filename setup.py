#!/usr/bin/env python
###################################################################
#  Zorro Dose-fractionated Image Registration Software
#
#      License: BSD
#      Author:  Robert A. McLeod
#
#  See LICENSE.txt for details about copyright and
#  rights to use.
####################################################################

# Deployment on PyPI:
# https://hynek.me/articles/sharing-your-labor-of-love-pypi-quick-and-dirty/
# Note that PyPI does not permit binary wheels for Linux, only MS Windows and OS-X.  So just the source distribution.

# Building on Intel C Compiler:
# https://gehrcke.de/2014/02/building-numpy-and-scipy-with-intel-compilers-and-intel-mkl-on-a-64-bit-machine/

# Some samples:
#     https://github.com/pypa/sampleproject/blob/master/setup.py
#     https://github.com/django/django/blob/master/setup.py
#     http://python-packaging-user-guide.readthedocs.org/en/latest/distributing/
#
#  ManyLinux wheels:
#     https://www.python.org/dev/peps/pep-0513/

"""
 NOTES FOR WINDOWS:
 For Python 2.7, you must have the x64 environment variables set correctly...

from:
    http://scikit-learn.org/stable/developers/advanced_installation.html#windows
for the 64-bit architecture, you either need the full visual studio or the free windows sdks that 
can be downloaded from the links below.
the windows sdks include the msvc compilers both for 32 and 64-bit architectures. they come as a 
grmsdkx_en_dvd.iso file that can be mounted as a new drive with a setup.exe installer in it.

for python 2 you need sdk v7.0: ms windows sdk for windows 7 and .net framework 3.5 sp1

for python 3 you need sdk v7.1: ms windows sdk for windows 7 and .net framework 4

both sdks can be installed in parallel on the same host. to use the windows sdks, you need to setup 
the environment of a cmd console launched with the following flags (at least for sdk v7.0):
cmd /e:on /v:on /k
then configure the build environment with (FOR PYTHON 2.7):

[Python 2.7]
set distutils_use_sdk=1
set mssdk=1
"c:\program files\microsoft sdks\windows\v7.0\setup\windowssdkver.exe" -q -version:v7.0
"c:\program files\microsoft sdks\windows\v7.0\bin\setenv.cmd" /x64 /release
[activate py27]
python setup.py install

[Python 3.4]
set distutils_use_sdk=1
set mssdk=1
"c:\program files\microsoft sdks\windows\v7.0\setup\windowssdkver.exe" -q -version:v7.1
"c:\program files\microsoft sdks\windows\v7.0\bin\setenv.cmd" /x64 /release
[activate py34]
python setup.py install


replace /x64 by /x86 to build for 32-bit python instead of 64-bit python.

Ignore the "Missing compiler_cxx fix for MSVCCompiler" error message.

You also need the .NET Framework 3.5 SP1 installed for Python 2.7
"""
import shutil
import os
import sys
from distutils.command.clean import clean
# To use a consistent encoding
from codecs import open
from os import path
import time


major_ver = 0
minor_ver = 7
nano_ver = 3
branch = 'b1'

version = "%d.%d.%d%s" % (major_ver, minor_ver, nano_ver, branch)

# Write version.py to Automator and Zorro
with open( "zorro/__version__.py", 'w' ) as fh:
    fh.write( "__version__ = '" + version + "'\n" )
with open( "automator/__version__.py", 'w' ) as fh:
    fh.write( "__version__ = '" + version + "'\n" )


INSTALL_AUTOMATOR = True
if sys.version_info < (2, 7):
    raise RuntimeError( "ZorroAutomator requires python 2.7 or greater" )
if sys.version_info > (3, 4):
    print( "WARNING: PySide is not compatible with Python > 3.5, Automator will not be installed" )
    INSTALL_AUTOMATOR = False
    
# Always prefer setuptools over distutils
try:
    import setuptools
except ImportError:
    setuptools = None

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
try:
    with open(path.join(here, 'README.rst'), encoding='utf-8') as f:
        readmerst = f.read()
except: # No long description
    long_description = ""
    pass

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

def configuration(parent_package=None,top_path=None):
    from numpy.distutils.misc_util import Configuration
    
    config = Configuration( None, parent_package, top_path )
    config.set_options(ignore_setup_xxx_py=False,
               assume_default_configuration=True,
               delegate_options_to_subpackages=True,
               quiet=False)

    config.add_subpackage('zorro')
    os.chdir( here )
    config.add_subpackage('numexprz')
    os.chdir( here )
    if INSTALL_AUTOMATOR:
        config.add_subpackage('automator')
        os.chdir( here )

    config.add_data_files( (".", 'README.rst') )
    config.add_data_files( (".", "LICENSE.txt") )
    config.add_data_files( (".", "MANIFEST.in") )
    config.add_data_files( (".", "CHANGELOG.txt") )
    config.add_data_files( (".", "setup.py") )
    config.add_data_files( (".", "cxfreeze_setup.py") )
    config.add_data_files( (".", "requirements.txt") )

    # RAM: This root-level config isn't especially needed, except for making source distributions.
    config.make_config_py()
    print( "CONFIGURATION DICT" )
    print( config )
    
    return config

def setup_zorro():
    metadata = dict(
                      name = "zorroautomator",
                      version = version,
                      description='Zorro Automator - dose-fractionated image registrator',
                      long_description = readmerst,
                      author='Robert A. McLeod',
                      author_email='robert.mcleod@unibas.ch',
                      url='https://github.com/C-CINA/Zorro',
                      license='BSD',
                      packages=[], # DON'T INLCUDE subpackages here in setuptools...
                      install_requires=requirements,
                      setup_requires=requirements,
                      entry_points={
                          "console_scripts": [ "zorro = zorro.__main__:main", 
                                               ],
                          "gui_scripts" : [ "automator = automator.__main__:main",
                                            "ims = zorro.zorro_plotting:main" ],
                          },
                      # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
                      extras_require={
                            ':python_version <= "3.4"': [
                                'pyside',
                            ],
                        },
                      classifiers=[
                            # How mature is this project? Common values are
                            #   3 - Alpha
                            #   4 - Beta
                            #   5 - Production/Stable
                            'Development Status :: 4 - Beta',
                    
                            # Indicate who your project is intended for
                            'Topic :: Scientific/Engineering :: Bio-Informatics',
                            'Topic :: Scientific/Engineering :: Image Recognition',
                    
                            # Pick your license as you wish (should match "license" above)
                            'License :: OSI Approved :: BSD License',
                    
                            # Specify the Python versions you support here. In particular, ensure
                            # that you indicate whether you support Python 2, Python 3 or both.
                            'Programming Language :: Python :: 2',
                            'Programming Language :: Python :: 2.7',
                            'Programming Language :: Python :: 3',
                            'Programming Language :: Python :: 3.4',
                            'Programming Language :: Python :: 3.5',
                            
                            # OS
                            'Operating System :: POSIX',
                            'Operating System :: Microsoft :: Windows',
                            'Operating System :: MacOS :: MacOS X',
                        ],
                    keywords=['template matching', 'image registration'],
                    zip_safe=False, # interpreter DLL cannot be zipped
    )
    

        
    if (len(sys.argv) >= 2 and ('--help' in sys.argv[1:] or sys.argv[1]
    in ('--help-commands', 'egg_info', '--version', 'clean'))):

        # For these actions, NumPy is not required.
        #
        # They are required to succeed without Numpy for example when
        # pip is used to install Numexpr when Numpy is not yet present in
        # the system.
        # (via https://github.com/abhirk/scikit-learn/blob/master/setup.py)
        try:
            from setuptools import setup
        except ImportError:
            from distutils.core import setup

#        metadata['name']    = 'numexprz'
#        metadata['version'] = version
    else: # RAM: monkey patch numpy distutils
        from numpy.distutils.core import setup
        #from numpy.distutils.command.build_ext import build_ext as numpy_build_ext

        try:  # Python 3
            # Code taken form numpy/distutils/command/build_py.py
            from distutils.command.build_py import build_py_2to3 as old_build_py
            from numpy.distutils.misc_util import is_string

            class build_py(old_build_py):

                def run(self):
                    build_src = self.get_finalized_command('build_src')
                    if build_src.py_modules_dict and self.packages is None:
                        self.packages = list(build_src.py_modules_dict.keys())
                    old_build_py.run(self)

                def find_package_modules(self, package, package_dir):
                    modules = old_build_py.find_package_modules(self, package, package_dir)

                    # Find build_src generated *.py files.
                    build_src = self.get_finalized_command('build_src')
                    modules += build_src.py_modules_dict.get(package, [])

                    return modules

                def find_modules(self):
                    old_py_modules = self.py_modules[:]
                    new_py_modules = list(filter(is_string, self.py_modules))
                    self.py_modules[:] = new_py_modules
                    modules = old_build_py.find_modules(self)
                    self.py_modules[:] = old_py_modules

                    return modules

        except ImportError:  # Python 2
            from numpy.distutils.command.build_py import build_py

        def localpath(*args):
            return path.abspath(path.join(*((path.dirname(__file__),) + args)))

        class cleaner(clean):

            def run(self):
                # Recursive deletion of build/ directory
                path = localpath("build")
                try:
                    shutil.rmtree(path)
                except Exception:
                    print("Error: Failed to remove directory %s" % path)
                else:
                    print("Success: Cleaned up %s" % path)

                # Now, the extension and other files
                try:
                    import imp
                except ImportError:
                    if os.name == 'posix':
                        paths = [localpath("zorro/numexprz/interpreter.so")]
                    else:
                        paths = [localpath("zorro/numexprz/interpreter.pyd")]
                else:
                    paths = []
                    for suffix, _, _ in imp.get_suffixes():
                        if suffix == '.py':
                            continue
                        paths.append(localpath("zorro/numexprz", "interpreter" + suffix))
                paths.append(localpath("zorro/numexprz/__config__.py"))
                paths.append(localpath("zorro/numexprz/__config__.pyc"))
                for path in paths:
                    try:
                        os.remove(path)
                    except Exception:
                        print("Error: Failed to clean up file %s" % path)
                    else:
                        print("Cleaning up %s" % path)

                clean.run(self)


        metadata['configuration'] = configuration

    setup(**metadata)


if __name__ == '__main__':
    # RAM: don't call setuptools anymore with numpy.distutils, it generates some bugs
    #setuptools.setup()
    
    t0 = time.time()
    setup_zorro()
    t1 = time.time()
    print( "Completed: Zorro build/install time (s): %.3f" % (t1-t0) )


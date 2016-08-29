import cx_Freeze
import sys, glob, os, os.path
import PySide
#import traceback 


##############################################################################################
# Basic way to build this: run it, run the build exe (./zorro) from another terminal,
# include, exclude as needed.
# ALWAYS CALL AS ./zorro in Linux because the exe is not on the system path
# ALWAYS CALL AS ./zorro in Linux because the exe is not on the system path
# ALWAYS CALL AS ./zorro in Linux because the exe is not on the system path
# ALWAYS CALL AS ./zorro in Linux because the exe is not on the system path
#############################################################################################

#############################################################################################
# MUST MODIFY hooks.py IN cx_Freeze AS FOLLOWS:
#def load_scipy(finder, module):
#    """the scipy module loads items within itself in a way that causes
#       problems without the entire package and a number of other subpackages
#       being present."""
#    # finder.IncludePackage("scipy.lib")
#    finder.IncludePackage("scipy._lib")
#    finder.IncludePackage("scipy.misc")

# Monkey-patch cx_Freeze.hooks.load_scipy()
def load_scipy_monkeypatched(finder, module):
    """the scipy module loads items within itself in a way that causes
       problems without the entire package and a number of other subpackages
       being present."""
    finder.IncludePackage("scipy._lib")
    finder.IncludePackage("scipy.misc")

cx_Freeze.hooks.load_scipy = load_scipy_monkeypatched
#############################################################################################


# Need to exclude some dependencies that are definitely not needed.
# I would like to exlcude PyQt4 but Matplotlib crashes
if sys.version_info.major == 2:
    excludes = [ 'collections.abc', 'Tkinter', 'pandas',  ]
else:
    excludes = [  'Tkinter', 'pandas',  ]


# Need to add some things scipy has imported in a funny way.
includes = ['scipy.sparse.csgraph._validation', 
'scipy.integrate.lsoda',
'scipy.integrate.vode', 
'scipy.special._ufuncs_cxx', 
'scipy.special._ufuncs', 
'scipy.special._ellip_harm_2',
'scipy.sparse.csgraph._validation', 
'atexit',
'PySide.QtGui',
'PySide.QtCore',
 ]
# Ok now we have trouble with the matplotlib.backends
packages = [
'matplotlib.backends.backend_qt4agg',
'PySide.QtGui',
'PySide.QtCore',
'atexit',
]

# Inlcude graphics files.  Could also add Readme and such.
include_files = [ "automator/icons/" ]

# Include Qt libs, because by default they are not present.
# Ok, so this does not fix the Qt library mis-match problem.
"""
QtDir = os.path.dirname( PySide.__file__  ) 
# TODO: Windows support for dll
QtFiles = [ (os.path.join( QtDir, "shiboken.so"), "shiboken.so" ) ]
QtLibGlob = glob.glob( os.path.join( QtDir, "Qt*" ) )
for globItem in QtLibGlob:
    QtFiles.append( (globItem, os.path.basename(globItem)) )
include_files += QtFiles
"""

icon = "automator/icons/CINAlogo.png"

buildOptions = dict(
    packages = packages, 
    excludes = excludes, 
    includes = includes,
    include_files = include_files,
    icon=icon )


base = 'Console'
if sys.platform == 'win32':
    gui_base = "Win32GUI"
else:
    gui_base = 'Console'

executables = [
    cx_Freeze.Executable( script='zorro/__main__.py', 
        base=base, 
        targetName='zorro', 
        copyDependentFiles=True),
    cx_Freeze.Executable( script='automator/__main__.py', 
        base=gui_base, 
        targetName='automator',
        copyDependentFiles=True )
]



cx_Freeze.setup(name='Zorro',
    version = '0.5',
    description = 'Zorro image registration software',
    options = dict(build_exe = buildOptions),
    executables = executables,
)



###################################################################
#  Zorro Dose-fractionated Image Registration Software
#
#      License: MIT
#      Author:  Robert A. McLeod
#      Website: https://github.com/C-CINA/zorro-dev
#
#  See LICENSE.txt for details about copyright and
#  rights to use.
####################################################################

from __future__ import (division, absolute_import, print_function)

from .zorro import (ImageRegistrator,)
from .__main__ import main
from . import zorro_util as util
from . import zorro_plotting as plot

import os.path

from .__version__ import __version__

from . import extract

# Call Zorro as a blocking subprocess call on a provided config/log file.
def call( configName ):
    import subprocess
    execname = os.path.join( os.path.split( __file__ )[0], "__main__.py" )
    subprocess.call( "python " + execname + " -c " + configName, shell=True )
    # subprocess.call( "zorro -c " + configName, shell=True )

#__workingScript = os.path.join( os.path.dirname(__file__), "zorroWorkingScript.py" )
#if os.path.isfile( __workingScript ):
#    print( "Zorro directory processing script may be found in : " + __workingScript )



#!/usr/bin/env python
###################################################################
#  Automator for Zorro and 2dx
#
#      License: LGPL
#      Author:  Robert A. McLeod
#      Website: https://github.com/C-CINA/zorro-dev
#
#  See LICENSE.txt for details about copyright and
#  rights to use.
####################################################################

from __future__ import (division, absolute_import, print_function)

import os, os.path
os.environ.setdefault('QT_API','pyside')

dirname = os.path.dirname(__file__)

from .__version__ import __version__


from .zorroSkulkManager import (skulkHost, skulkHeap, skulkPaths, skulkManager, zorroState)
from .Automator import (Automator, main)

#try:
#    from PySide import QtGui, QtCore
#    os.environ.setdefault('QT_API','pyside')
#except:
#    # Import PyQt4 as backup?  I suspect this still causes license issues
#    ImportError( "Automator.py: PySide not found, I am going to crash now. Bye." )
# os.chdir( origdir )
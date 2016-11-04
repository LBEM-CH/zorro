# -*- coding: utf-8 -*-
"""
Created on Fri Sep 30 09:08:28 2016

@author: Robert A. McLeod
"""

import automator
import numpy.testing as npt
import os, os.path, sys
import subprocess
import tempfile
import unittest
from PySide.QtTest import QTest
from PySide.QtGui import QApplication

if __name__ == "__main__":
    
    # RAM: Here we need some helper class 
    _instance = None

        
    # Generate a sample GUI object
    dasAuto = automator.Automator( testing=True )
    dasAuto.skulk.setDEBUG( True )


#==============================================================================
# Check functionality of updateDict on strings getattr functionality
# def updateDict( self, dictHandle, key, funchandle, funcarg = None )
#==============================================================================
    if os.name != "nt":
        def targetDir():
            return "input"
        
        dasAuto.updateDict( u"skulk.paths", u"input_dir", targetDir )
        npt.assert_equal( dasAuto.skulk.paths[u"input_dir"], targetDir() )
    else:
        print( "TODO: check relative pathing" )
        
    dasAuto.updateDict( u"cfgCluster", u"n_threads", dasAuto.sbNThreads.value )
    npt.assert_equal( dasAuto.cfgCluster[u"n_threads"], dasAuto.sbNThreads.value() )
    
    
    

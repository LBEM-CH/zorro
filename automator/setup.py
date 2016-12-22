# -*- coding: utf-8 -*-
"""
Created on Wed Mar  2 15:32:31 2016

@author: Robert A. McLeod
"""
import glob
import os, os.path
import subprocess


def configuration( parent_package='', top_path=None ):
    from numpy.distutils.misc_util import Configuration
    config = Configuration( 'automator', parent_package, top_path )
    
    my_path = os.path.dirname(__file__)
#    uiList = glob.glob( os.path.join( my_path, "*.ui" ) )
#    try:
#        for ui in uiList:
#            uiPy = os.path.splitext( ui )[0] + ".py"
#            subprocess.call( "pyside-uic %s -o %s" % ( ui, uiPy ), shell=True )
#    except:
#        print( "Automator failed to compile QtDesigner UI files, is pyside-uic in system path?" )
    
#    icon_files = glob.glob( path.join( path.realpath('./icons'), '*.png' ) )
#    print( "Found icons for Automator: " + str(icon_files) )
    
    config.add_data_files( ('icons', 'automator/icons/*.png' ) )
    config.add_data_files( ('', 'automator/*.ini') )
    config.add_data_files( ("", "automator/*.ui" ) )
    config.add_data_files( ("", "automator/*.txt" ) )    
    
    config.make_config_py()
    return config
    
if __name__=="__main__":
    print( "Run the setup.py in the base package directory" )

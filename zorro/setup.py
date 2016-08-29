# -*- coding: utf-8 -*-
"""
Created on Wed Mar  2 15:32:31 2016

@author: Robert A. McLeod
"""

def configuration( parent_package='', top_path=None ):
    from numpy.distutils.misc_util import Configuration
    config = Configuration( 'zorro', parent_package, top_path )
    
    
    config.add_subpackage( 'scripts' )
    
    config.add_data_files( ("", "zorro/*.zor") )
    
    config.make_config_py()
    return config
    
if __name__=="__main__":
    print( "Run the setup.py in the base package directory" )
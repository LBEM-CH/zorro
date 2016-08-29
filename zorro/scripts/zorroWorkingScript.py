# -*- coding: utf-8 -*-
"""
Created on Mon Sep  7 10:25:34 2015
@author: Robert A. McLeod
@email: robbmcleod@gmail.com OR robert.mcleod@unibas.ch
"""
import numpy as np
import zorro
import os, os.path, glob

compress_ext = '.bz2'
zorroDefault = zorro.ImageRegistrator()
##### IMPORTANT VARIABLES ####
zorroDefault.diagWidth = 5
zorroDefault.n_threads = 16
zorroDefault.suppressOrigin = True
zorroDefault.shapePadded = [4096,4096]
zorroDefault.doCompression = False
zorroDefault.doDoseFilter = True
zorroDefault.doLazyFRC = True
zorroDefault.savePNG = True
zorroDefault.doCTF = True

##### TYPICAL DEFAULT VARIABLES #####
zorroDefault.maxShift = 60 # This is only within diagWidth frames difference
zorroDefault.weightMode = 'autologistic' 
zorroDefault.Bmode = 'opti'
# zorroDefault.peaksigThres = 5.0
zorroDefault.subPixReg = 16
zorroDefault.triMode = 'diag'
zorroDefault.shiftMethod = 'lanczos'
zorroDefault.fftw_effort = 'FFTW_MEASURE'
zorroDefault.originMode = 'centroid'
# zorroDefault.preShift = True
zorroDefault.saveC = False
zorroDefault.plotDict['Transparent'] = False # Not helpful if Automator isn't used.
zorroDefault.files['figurePath'] = 'figures/'

# Find all compressed .dm4 files
compressfiles = glob.glob( "*.dm4" + compress_ext )
#for J, cfile in enumerate(compressfiles):
#    compressfiles[J] = os.path.splitext( cfile )[0]
# Find all .dm4 files in a directory that don't have a log and process them.
filenames = glob.glob( "*.dm4" )
filenames = np.sort( filenames + compressfiles )
#lognames = glob.glob( "*.dm4.log" )
## Strip the .log format and see if filename is in lognames
#for (J,log) in enumerate(lognames):
#    lognames[J] = os.path.splitext(log)[0]

print( "Found " + str(len(filenames)) + " raw stacks total in directory" )
#filenames = list( set(filenames) - set(lognames) )
#print( "Processing " + str(len(filenames)) + " remaining stacks" )
# Now add back the compressed extension if it exists...
#for J, filename in enumerate( filenames ):
#    if filename in compressfiles:
#        filenames[J] = filename + compress_ext


for filename in filenames:
    cfg_name = filename + ".log"
    if os.path.isfile( cfg_name ):
        zorroDefault.loadConfig( cfg_name )
        if zorroDefault.METAstatus == 'fini':
            continue 
        
    # Else process the stack        
    zorroDefault.initDefaultFiles( filename )
    zorroDefault.saveConfig( cfg_name )
    zorro.call( cfg_name )

print( "%%%%% Zorro processing complete %%%%%" )
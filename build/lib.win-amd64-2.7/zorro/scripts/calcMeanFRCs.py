# -*- coding: utf-8 -*-
"""
Created on Tue Jun  7 16:12:33 2016

@author: rmcleod
"""

import numpy as np
import matplotlib.pyplot as plt
import os, os.path, glob

mcFRCFiles = glob.glob( "FRC/*mcFRC.npy" )
zorroFRCFiles = glob.glob( "FRC/*zorroFRC.npy" )

zorroFRCs = [None] * len( zorroFRCFiles)


for J in np.arange( len(zorroFRCFiles) ):
    zorroFRCs[J] = np.load( zorroFRCFiles[J] )
    
mcFRCs = [None] * len( mcFRCFiles)    
for J in np.arange( len(mcFRCFiles) ):
    mcFRCs[J] = np.load( mcFRCFiles[J] )
    
zorroMeanFRC = np.mean( np.array(zorroFRCs), axis=0 )
mcMeanFRC = np.mean( np.array(mcFRCs), axis=0 )

plt.figure()
plt.plot( mcMeanFRC, '.-', color='firebrick', label='MotionCorr' )
plt.plot( zorroMeanFRC, '.-', color='black', label='Zorro' )
plt.title( "Mean FRC Re-aligned from MotionCorr" )
plt.legend()
plt.xlim( [0,len(mcMeanFRC)] )
plt.savefig( "Dataset_mean_MC_vs_Zorro.png" )
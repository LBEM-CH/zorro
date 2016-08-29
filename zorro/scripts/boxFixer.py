# -*- coding: utf-8 -*-
"""
Pre-filter boxes for being too close to the edge

Created on Thu Apr 21 10:52:53 2016

@author: rmcleod
"""


import numpy as np
import glob, os, os.path
imageShape = [3710,3838]


boxNames = glob.glob( "*.box" )

if not os.path.isdir( "./goodBoxes" ):
    os.mkdir( "./goodBoxes" )

for J, boxName in enumerate(boxNames):
    print( "Loading: " + boxName )
    boxes = np.loadtxt( boxName, dtype='int' )

    if not np.any( boxes ):
        # Remove the box-file, it's empty
        print( "Delete empty: " + boxName )
        continue
    
    boxWidth = np.int( boxes[0,2] )
    
    keepBoxes = np.logical_and( boxes[:,0] > 0, boxes[:,1] > 0 )
    keepBoxes = np.logical_and( keepBoxes,  boxes[:,0] < (imageShape[0]-boxWidth) )
    keepBoxes = np.logical_and( keepBoxes, boxes[:,1] < (imageShape[1]-boxWidth) )
    
    goodBoxes = boxes[keepBoxes,:]
    np.savetxt( os.path.join( "./goodBoxes", boxName ), goodBoxes, fmt='%8d' )
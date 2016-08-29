"""
Converts all .box files in a directory to the associated .star files.
"""

import numpy as np
import os, os.path, glob

boxList = glob.glob("*.box")

starHeader = """
data_

loop_
_rlnCoordinateX #1
_rlnCoordinateY #2
"""

shapeImage = [3838,3710]
for boxFile in boxList:
    print( "Loading %s" % boxFile )
    boxData = np.loadtxt(boxFile)
    
    xCoord = boxData[:,0]
    yCoord = boxData[:,1]
    boxX = boxData[:,2]/2
    boxY = boxData[:,3]/2
    
    keepElements = ~((xCoord < boxX)|(yCoord < boxY)|(xCoord > shapeImage[1]-boxX)|(yCoord> shapeImage[0]-boxY))
    xCoord = xCoord[keepElements]
    yCoord = yCoord[keepElements]
    boxX = boxX[keepElements]
    boxY = boxY[keepElements]
    
    starFilename = os.path.splitext( boxFile )[0] + ".star"
    with open( starFilename, 'wb' ) as sh:
        sh.writelines( starHeader )
        for J in np.arange(0,len(xCoord)):
            sh.write( "%.1f %.1f\n" % (xCoord[J]+boxX[J], yCoord[J]+boxY[J] ) )
           
        sh.write( "\n" )
    sh.close()

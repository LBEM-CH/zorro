#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  7 11:52:11 2016

@author: rmcleod

Better GCTF log parser

Template: 
    
**************************************   LAST CYCLE    ************************************************************ *

   Defocus_U   Defocus_V       Angle        CCC
    22195.93    22722.04       47.05    0.039181  Final Values

Resolution limit estimated by EPA: RES_LIMIT 3.972 
Estimated Bfactor: B_FACTOR  5.09 

......................................  VALIDATION  ......................................
Differences from Original Values:
  RESOLUTION   Defocus_U   Defocus_V     Angle       CCC      CONVERGENCE  
  20-08A       -50.77       11.77        5.63        0.01     VALIDATION_SCORE: 5        
  15-06A       -30.03       30.16        3.43        0.01     VALIDATION_SCORE: 5        
  12-05A       -10.21       -1.73       -0.38        0.00     VALIDATION_SCORE: 5        
  10-04A       -12.08       -0.40        0.48        0.00     VALIDATION_SCORE: 5        
  08-03A        17.42      -33.01        2.31        0.00     VALIDATION_SCORE: 5    
  Processing done successfully.

"""

import numpy as np
import re, os, os.path, glob
import matplotlib.pyplot as plt


def parseGCTFLog( logName ):
    """
    Parse the GCTF log, tested on GCTF v1.06.
    """
    validIndex = -1
    CTFInfo = dict()
        
    with open( logName, 'r' ) as fh:
        logLines = fh.readlines()
        
        # Find "LAST CYCLE"
        for J in np.arange( len( logLines )-1, 0, -1 ):
            # Search backwards
            if "VALIDATION" in logLines[J]:
                # print( "Found VALIDATION at line %d" % J )
                validIndex = J
                
            if "LAST CYCLE" in logLines[J]:
                # print( "Found LAST CYCLE at line %d" % J )
                if validIndex > 0:
                    validLines = logLines[validIndex+2:-1]
                resultsLines = logLines[J:validIndex]
                break
            
        ctfList = resultsLines[3].split()
        CTFInfo[u'DefocusU'] = float( ctfList[0] )
        CTFInfo[u'DefocusV'] = float( ctfList[1] )
        CTFInfo[u'DefocusAngle'] = float( ctfList[2] )
        CTFInfo[u'CtfFigureOfMerit'] = float( ctfList[3] )
        CTFInfo[u'FinalResolution'] = float( resultsLines[5].split()[6] )
        CTFInfo[u'Bfactor'] = float( resultsLines[6].split()[3] )
        
        # Would be kind of nice to use pandas for the validation, but let's stick to a dict
        for valid in validLines:
            valid = valid.split()
            try:
                CTFInfo[ valid[0] ] = [ float(valid[1]), float(valid[2]), float(valid[3]), float(valid[4]), float(valid[6]) ] 
            except ValueError:
                CTFInfo[ valid[0] ] = [ valid[1], valid[2], valid[3], valid[4], valid[5] ] 
    
        return CTFInfo


        
if __name__ == "__main__":
    logNames = glob.glob( "*gctf.log" )
    N = len(logNames)
    
    
    ctfData = np.zeros( [N, 6], dtype='float32' )
    ctfValid = np.zeros( [N, 5], dtype='float32' )
    
    logDicts = [None] * N
    for J, logName in enumerate( logNames ):
        ctfDict = parseGCTFLog( logName )
        logDicts[J]
        ctfData[J,0] = 0.5*( ctfDict[u'Defocus_U'] + ctfDict[u'Defocus_V'] )
        ctfData[J,1] = np.abs( ctfDict[u'Defocus_U'] - ctfDict[u'Defocus_V'] )
        ctfData[J,2] = ctfDict[u'CtfFigureOfMerit']
        ctfData[J,3] = ctfDict[u'FinalResolution']
        ctfData[J,4] = ctfDict[u'Bfactor']
        ctfValid[J,0] = ctfDict['20-08A'][-1]
        ctfValid[J,1] = ctfDict['15-06A'][-1]
        ctfValid[J,2] = ctfDict['12-05A'][-1]   
        ctfValid[J,3] = ctfDict['10-04A'][-1]  
        ctfValid[J,4] = ctfDict['08-03A'][-1]  

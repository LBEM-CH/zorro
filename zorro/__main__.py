# -*- coding: utf-8 -*-
"""
Created on Fri Jul 29 10:32:17 2016

@author: Robert A. McLeod

Discussion on how to use entry points:

https://chriswarrick.com/blog/2014/09/15/python-apps-the-right-way-entry_points-and-scripts/
"""


def main():
    # Get command line arguments
    from matplotlib import rc
    rc('backend', qt4="PySide")
    
    import sys, os
    import zorro
    import numpy as np
    import time

    
    # Usage: 
    # python `which zorro.py` -i Test.dm4 -c default.ini -o test.mrc
    stackReg = zorro.ImageRegistrator()
    configFile = None
    inputFile = None
    outputFile = None
    
    try: print( "****Running Zorro-command-line on hostname: %s****"%os.uname()[1] )
    except: pass
    
    for J in np.arange(0,len(sys.argv)):
        # First argument is zorro.py
        # Then we expect flag pairs
        # sys.argv is a Python list
        if sys.argv[J] == '-c':
            configFile = sys.argv[J+1]
            J += 1
        elif sys.argv[J] == '-i':
            inputFile = sys.argv[J+1]
            J += 1
        elif sys.argv[J] == '-o':
            outputFile = sys.argv[J+1]
            J += 1
            pass
    
    if inputFile == None and configFile == None:
        print( "No input files, outputing template.zor" )
        stackReg.saveConfig( configNameIn = "template.zor")
        sys.exit()
    if inputFile == None and not configFile == None:
        stackReg.loadConfig( configNameIn=configFile, loadData = True )
        stackReg.bench['total0'] = time.time() 
        
    if not inputFile == None and not configFile == None:
        stackReg.loadConfig( configNameIn=configFile, loadData = False )
        stackReg.bench['total0'] = time.time() 
        
        stackReg.files['stack'] = inputFile
        stackReg.loadData()
        
    if not outputFile == None:
        stackReg.files['sum'] = outputFile    
        
    # Force use of 'Agg' for matplotlib.  It's slower than Qt4Agg but doesn't crash on the cluster
    stackReg.plotDict['backend'] = 'Agg'
    
    if stackReg.triMode == 'refine':
        # In the case of 'refine' we have to call 'diag' first if it hasn't already 
        # been performde.
        if not bool(stackReg.errorDictList[-1]) and np.any( stackReg.imageSum != None ):
            # Assume that 
            print( "Zorro refine assuming that initial alignment has already been performed." )
            pass
        else:
            print( "Zorro refine performing initial alignment." )
            stackReg.triMode = 'diag'
            stackReg.alignImageStack()
            stackReg.loadData() # Only re-loads the stack
        stackReg.triMode = 'refine'
        # TODO: enable subZorro refinment.
        stackReg.alignImageStack()

    else:
        # Execute the alignment as called for Zorro/UnBlur/etc.
        stackReg.alignImageStack()
    
    # Save everthing and do rounding/compression operations
    stackReg.saveData() # Can be None 
    
    # Save plots
    if stackReg.savePNG:
        stackReg.plot()
    
    stackReg.printProfileTimes()
    
    stackReg.METAstatus = 'fini'
    stackReg.saveConfig()
    print( "Zorro exiting" )
    
    sys.exit()
    
if __name__ == "__main__":
    main()
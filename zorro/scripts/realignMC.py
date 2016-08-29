# -*- coding: utf-8 -*-
"""
Created on Tue May 31 12:04:10 2016

@author: rmcleod
"""

if __name__ == "__main__":
    import zorro
    import numpy as np
    #from zorro import ims
    import matplotlib.pyplot as plt
    import os, os.path
    import sys

    filename = sys.argv[1]

    print( "Operating on: " + filename )    
    plt.switch_backend( 'Agg' )

    stackFront = os.path.splitext( filename )[0]
    zreg = zorro.ImageRegistrator()
    zreg.loadData( filename  )
    mcSum = np.sum( zreg.images, axis=0 )
    
    zreg.lazyFouRingCorr()
    mcFRC = zreg.FRC.copy()
    
    #ims( zreg.imageSum )
    #zreg.alignImageStack()
    zreg.n_threads = 16
    zreg.maxShift = 30 # Realistically if we are realigning data it cannot be excessively off
    zreg.doLazyFRC = True
    zreg.saveC = True
    zreg.suppressOrigin = False
    zreg.plotDict['multiprocess'] = False
    zreg.plotDict['Transparent'] = False
    zreg.plotDict['backend'] = 'Agg'
    
    # Build a custom mask that's smaller than normal
    zreg.masks = zorro.util.edge_mask( maskShape=[3838,3710], edges=[128,128,128,128] )
    zreg.pixelsize = 1.326
    zreg.voltage = 300.0
    zreg.detectorPixelSize = 5.0
    zreg.C3 = 2.7

    zreg.filterMode = 'dose,background'
    zreg.doseFiltParam[5] = 2 # Missing two frames, so add them to the dose filtering
    zreg.diagWidth = 5 # Pretty rapid drop-off in this data set.
    
    zreg.alignImageStack()
    
    plt.figure()
    plt.plot( mcFRC, '.-', color='firebrick', label='MotionCorr' )
    plt.plot( zreg.FRC, '.-', color='black', label='Zorro' )
    plt.title( "FRCs Re-aligned from MotionCorr (could be dangerous)" )
    plt.legend()
    plt.xlim( [0,len(mcFRC)] )
    plt.savefig( os.path.join( "FRC", stackFront + "_MC_vs_Zorro.png" ) )
    
    np.save( os.path.join( "FRC", stackFront + "_zorroFRC.npy"),  zreg.FRC )
    np.save( os.path.join( "FRC/", stackFront + "_mcFRC.npy"),  mcFRC )
    
    #ims( zreg.C )
    zorro.ioMRC.MRCExport( zreg.imageSum, os.path.join( "realign", stackFront + "_rezorro.mrc" ) )
    zorro.ioMRC.MRCExport( zreg.filtSum, os.path.join( "realign", stackFront + "_rezorro_filt.mrc" ) )
    
    zreg.savePNG = True
    zreg.plot()
    plt.close('all')
    
    zreg.METAstatus = 'fini'
    zreg.saveConfig()
# TO DO: load all the .npy files from disk and add the averages over the whole set




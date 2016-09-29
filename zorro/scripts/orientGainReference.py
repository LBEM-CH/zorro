# -*- coding: utf-8 -*-
"""
Gain reference orientation analysis

Created on Sun Sep 25 09:03:27 2016

@author: Robert A. McLeod
"""

import zorro
import zorro.zorro_util as util
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as col
import numexprz as nz



def orientGainRef( stackName, gainRefName,
                   stackIsInAHole=True, applyHotPixFilter = True, doNoiseCorrelation=False,
                   relax=0.95, n_threads = None ):
    """
     USAGE

     Applies the gain reference over all possible orientations to determine which 
     is the best, quantitatively.  Calculates the image standard deviation after 
     gain normalization (less is better) and the number of outlier pixels (less is 
     better) and the degree of correlated noise (less is better)
    
     User should select a short flat-field stack (20 frames) in a hole.  If no flat-field
     image is available, pick a stack with no carbon or large amounts of ice, that 
     exhibited a large amount of drive.
    """
    # ANALYSIS SCRIPT
    zGain = zorro.ImageRegistrator()
    zGain.loadData( gainRefName, target='sum' )
    
    zStack = zorro.ImageRegistrator()
    zStack.loadData( stackName )
    rawStack = np.copy( zStack.images )
    
    #print( zGain.imageSum.shape )
    #print( zStack.images.shape )
    
    
    # Check if stack and gain is transposed
    if zGain.imageSum.shape[0] == zGain.imageSum.shape[1]:
        print( "Square images" )
        orientList = [ 
             [0,False,False], [0,False,False], 
             [0,True,False], [0,True,True],
             [1,False,False], [1,False,False], 
             [1,True,False], [1,True,True], 
            ]
    elif (zGain.imageSum.shape[0] != zStack.images.shape[1] 
            and zGain.imageSum.shape[0] == zStack.images.shape[2] ):
        print( "Rectangular image, rot90=1" )
        orientList = [ 
             [1,False,False], [1,False,False], 
             [1,True,False], [1,True,True], 
            ]
    else:
        print( "Rectangular image, rot90=0" )
        orientList = [ 
             [0,False,False], [0,False,False], 
             [0,True,False], [0,True,True], 
            ]
        
    # If the images are rectangular our life is easier as it cuts the number of 
    # possible orientations in half.
    
    N = len(orientList)
    
    stdArray = np.zeros( N )
    outlierPixCount = np.zeros( N )
    corrNoiseCoeff = np.zeros( N )
    hotCutoff = np.zeros( N )
    deadCutoff = np.zeros( N )
    binnedSum = [None] * N
    
    if bool(doNoiseCorrelation):
        FFTMage = np.empty( zStack.images.shape[1:], dtype='complex64' )
        FFTConj = np.empty( zStack.images.shape[1:], dtype='complex64' )
        IFFTCorr = np.empty( zStack.images.shape[1:], dtype='complex64' )
        FFT2, IFFT2 = zorro.zorro_util.pyFFTWPlanner( FFTMage, FFTConj, n_threads=24 )
        normConst2 = np.float32( 1.0  / np.size( FFTMage )**2 )
        
    for I, orient in enumerate(orientList):
        
        gainRef = np.copy( zGain.imageSum )
        if orient[0] > 0:
            print( "Rotating gain refernce by 90 degrees" )
            gainRef = np.rot90( gainRef, k=orient[0] )
        if orient[1] and orient[2]:
            print( "Rotating gain reference by 180 degrees" )
            gainRef = np.rot90( gainRef, k=2 )
        elif orient[1]:
            print( "Mirroring gain reference vertically" )
            gainRef = np.flipud( gainRef )
        elif orient[2]:
            print( "Mirroring gain reference horizontally" )
            gainRef = np.fliplr( gainRef )
            
        zStack.images = zStack.images * gainRef
        
        if applyHotPixFilter:
            zStack.hotpixInfo['relax'] = relax
            zStack.hotpixFilter()
            outlierPixCount[I] = zStack.hotpixInfo['guessDeadpix'] + zStack.hotpixInfo['guessHotpix']
            hotCutoff[I] = zStack.hotpixInfo[u'cutoffUpper']
            deadCutoff[I] = zStack.hotpixInfo[u'cutoffLower']
    
        binnedSum[I] = util.squarekernel( np.sum(zStack.images,axis=0), k=3 )
        # zorro.zorro_plotting.ims( binnedSum[I] )
        
        stdArray[I] = np.std( np.sum( zStack.images, axis=0 ) )
    
    
        if bool(stackIsInAHole) and bool(doNoiseCorrelation) :
            # Go through even-odd series
            for J in np.arange(1,zStack.images.shape[0]):
                print( "(Orientation %d of %d) Compute Fourier correlation  %d" % (I,N,J) )
                if np.mod(J,2) == 1:
                    FFT2.update_arrays( zStack.images[J,:,:].astype('complex64'), FFTConj ); FFT2.execute()
                else:
                    FFT2.update_arrays( zStack.images[J,:,:].astype('complex64'), FFTMage ); FFT2.execute()
                
                IFFT2.update_arrays( nz.evaluate( "normConst2*FFTMage*conj(FFTConj)"), IFFTCorr ); IFFT2.execute() 
                corrNoiseCoeff[I] += np.abs( IFFTCorr[0,0] )
        elif bool(doNoiseCorrelation): 
            # Calculate phase correlations with a frame seperation of 6 frames to 
            # avoid signal correlation
            frameSep = 6
            for J in np.arange(0,zStack.images.shape[0] - frameSep):
                print( "(Orientation %d of %d) Compute Fourier correlation  %d" % (I,N,J) )
                FFT2.update_arrays( zStack.images[J,:,:].astype('complex64'), FFTConj ); FFT2.execute()
                FFT2.update_arrays( zStack.images[J+frameSep,:,:].astype('complex64'), FFTMage ); FFT2.execute()
                
                IFFT2.update_arrays( nz.evaluate( "normConst2*FFTMage*conj(FFTConj)"), IFFTCorr ); IFFT2.execute() 
                corrNoiseCoeff[I] += np.real( IFFTCorr[0,0] )
            pass
        
            corrNoiseCoeff[I] /= normConst2
        zStack.images = np.copy( rawStack )
      
      
    #corrNoiseCoeff /= np.min( corrNoiseCoeff )
    #stdArray /= np.min(stdArray)  
    
    bestIndex = np.argmin( stdArray )
    
    nrows = 2
    ncols = np.floor_divide( N+1, 2 )
    plt.figure( figsize=(16,9))
    for I in np.arange(N):
        plt.subplot( nrows*100 + ncols*10 + (I+1)  )
        clim = util.histClim( binnedSum[I], cutoff=1E-3 )
        plt.imshow( binnedSum[I], cmap='gray', norm=col.LogNorm(), vmin=clim[0], vmax=clim[1] )
        plt.axis('off')
        if I == bestIndex:
            textcolor = 'purple'
        else:
            textcolor= 'black'
        title = "kRot: %d, VertFlip: %s, HorzFlip: %s \n" % (orientList[I][0], orientList[I][1], orientList[I][2])
                
        title += r"$\sigma: %.5g$" % stdArray[I]
        if bool(applyHotPixFilter):
            title += r"$, outliers: %d$" % outlierPixCount[I]
        if bool(doNoiseCorrelation):
            title += r"$, R_{noise}: %.5g$" % corrNoiseCoeff[I]
            
        plt.title( title, fontdict={'fontsize':14, 'color':textcolor} )
        plt.show(block=False)
    
    return orientList[bestIndex]


# USER PARAMETERS
#==============================================================================
# if __name__ == "__main__":
#     gainRefName = "/Projects/FourByte/raw/gainref/Gain-ref_x1m2_2016-08-30_0900AM_till_2016-08-30_1100PM.dm4"
#     stackName = "/Projects/FourByte/raw/Aug31_02.41.10.mrc"
#     orientGainRef( gainRefName, stackName, stackIsInAHole=False, applyHotPixFilter=True,
#                   relax=0.95, n_threads = nz.detect_number_of_cores() )
#==============================================================================

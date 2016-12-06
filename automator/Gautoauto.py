# -*- coding: utf-8 -*-
"""
--==%% Automating Gautomatch %%==--

Because it's not really that automatic.

Created on Tue Apr 12 10:25:42 2016
@author: Robert A. McLeod
@email: robert.mcleod@unibas.ch
"""
from zorro import util
import mrcz
import numpy as np
import matplotlib.pyplot as plt
import os, os.path, glob
import subprocess as sp
import multiprocessing as mp
import time
import skimage.io

###### USER OPTIONS FOR Gautomatch ######
# Benchmarking with bs-dw36: 24 physical cores Xeon(R) CPU E5-2680 v3 @ 2.50GHz, one GTX980 GPU
# cina-haf02 (white computer) is probably of similar-speed
# 16 proc = broke
# 8 proc = 23.0 s (breaks sometimes)
# 4 proc = 23.7 s
# Bencmarking with bs-gpu01: 4 physical CPU cores Xeon CPU E5-2603 0 @ 1.80GHz, four Tesla C2075 GPUs
# 4 proc = 143 s
# 2 proc = 263 s
# Bencmarking with bs-gpu04: 4 physical CPU cores Xeon CPU E5-2603 0 @ 1.80GHz, four Tesla C2075 GPUs
# 8 proc = 90 s
# 4 proc = 42 s
# 2 proc = 100 s



def runGauto( params ):
    """
    params = [mrcName, mode, optInit, optPlot, optRefine]
    mode can be 'batch' or 'inspect'
    """

    mrcName = params[0]
    pngFrontName = params[1]
    GautoMode = params[2]
    optInit = params[3]
    optPlot = params[4]
    optRefine = params[5]
    if optRefine == None:
        optRefine = { 'doRefine':False }
        
    if GautoMode == 'batch':
        initRunGauto( mrcName, optInit, optPlot, optRefine )
        
        if bool(optRefine['doRefine']):
            optRefine['thresCC'] = refineRunGauto( mrcName )
        else:
            if 'cc_cutoff' in optInit:
                optRefine['thresCC'] = optInit['cc_cutoff']
            else:
                optRefine['thresCC'] = 0.1 # Default value
                
            if not bool(optRefine['thresCC']): optRefine['thresCC'] = 0.0
        pass
    
        # saveDiagnosticImages( mrcName, pngFrontName, optInit, optPlot, optRefine )

    elif GautoMode == 'inspect':
        optPlot['write_pref_mic'] = True; optPlot['write_ccmax_mic'] = True; optPlot['write_bg_mic'] = True; 
        optPlot['write_bgfree_mic'] = True; optPlot['write_lsigma_mic'] = True; optPlot['write_mic_mask'] = True;
        
        initRunGauto( mrcName, optInit, optPlot, optRefine )
        
        saveDiagnosticImages( mrcName, pngFrontName, optInit, optPlot, optRefine )
    else:
        print( "Unknown mode: %s"%GautoMode )


def initRunGauto( mrcName, optInit, optPlot, optRefine ):
    try:
        Gauto_exec = util.which( "gautomatch" )
    except:
        try:
            Gauto_exec = util.which( "Gautomatch-v0.53_sm_20_cu7.5_x86_64" )
        except:
            raise SystemError( "Gautomatch not found in system path" )
    devnull = open(os.devnull, 'w' )
    
    extra_options = ""
    if 'do_pre_filter' in optPlot and bool(optPlot['do_pre_filter']):
        extra_options += " --do_pre_filter"
    if 'write_pref_mic' in optPlot and bool(optPlot['write_pref_mic']):
        extra_options += " --write_pref_mic"
    if ('write_ccmax_mic' in optPlot and bool(optPlot['write_ccmax_mic'])) or bool(optRefine['doRefine']):
        extra_options += " --write_ccmax_mic"
    if 'write_bg_mic' in optPlot and bool(optPlot['write_bg_mic']):
        extra_options += " --write_bg_mic"
    if 'write_bgfree_mic' in optPlot and bool(optPlot['write_bgfree_mic']):
        extra_options += " --write_bgfree_mic"
    if 'write_lsigma_mic' in optPlot and bool(optPlot['write_lsigma_mic']):
        extra_options += " --write_lsigma_mic"
    if 'write_mic_mask' in optPlot and bool(optPlot['write_mic_mask']):
        extra_options += " --write_mic_mask"

    optGauto = " "        
    for (key,val) in optInit.items():
        if bool(val):
            optGauto += " --%s"%key + " " + str(val)
    
    print( "********** GAUTOMATCH *****************" )
    print( Gauto_exec + " " + mrcName + optGauto + extra_options )
    sp.call( Gauto_exec + " " + mrcName + optGauto + extra_options, shell=True, stdout=devnull, stderr=devnull )
    print( "***************************************" )
    pass

def refineRunGauto( mrcName, optInit, optPlot, optRefine ):
    try:
        Gauto_exec = util.which( "Gautomatch" )
    except:
        try:
            Gauto_exec = util.which( "Gautomatch-v0.53_sm_20_cu7.0_x86_64" )
        except:
            raise SystemError( "Gautomatch not found in system path" )
    devnull = open(os.devnull, 'w' )
    
    ccMaxName = os.path.splitext( mrcName )[0] + "_ccmax.mrc"
    ccMax, _ = mrcz.readMRC( ccMaxName )
        
    pdfCC, hX = np.histogram( ccMax[optPlot['edge']:-optPlot['edge'],optPlot['edge']:-optPlot['edge']], bins=512 )
    pdfCC = pdfCC.astype('float32');    hX = hX[:-1]
        
    cdfCC = np.cumsum( pdfCC )
    cdfCC /= cdfCC[-1] # Normalize the cumulative sum to a CDF
    # Find the threshold value for the cross-correlation cutoff
    thresCC = hX[ np.argwhere( cdfCC > optRefine['cdfThres'] )[0][0] ]
    
    extra_options = "" # No extra options for refine at present
    
    optGauto = " "
    copyOptInit = optInit.copy()
    copyOptInit['cc_cutoff'] = thresCC
    for (key,val) in copyOptInit.items():
        if bool(val):
            optGauto += " --%s"%key + " " + str(val)
    
    # print( "Refinement Command: " + Gauto_exec + " " + mrcName + optGauto + extra_options )
    sp.call( Gauto_exec + " " + mrcName + optGauto + extra_options, shell=True, stdout=devnull, stderr=devnull )
    return thresCC
    
def saveDiagnosticImages( mrcName, pngFrontName, optInit, optPlot, optRefine ):
        
    mrcFront = os.path.splitext( mrcName )[0]
    
    print( "saveDiag looking for : " + mrcFront + "_automatch.box" )
    
    boxName = mrcFront + "_automatch.star"
    autoBoxes = np.loadtxt( boxName, comments="_", skiprows=4 )
    if autoBoxes.size == 0: # Empty box
        return 
        
    goodnessMetric = util.normalize( autoBoxes[:,4] )
    boxes = (autoBoxes[:,0:2] ).astype( 'int' )
        
    if 'thresCC' in optRefine:
        print( "For %s picked %d boxes with CC_threshold of %.3f" %(mrcName, boxes.shape[0], np.float32(optRefine['thresCC'])) )
    else:
        print( "For %s picked %d boxes" %(mrcName, boxes.shape[0]) )
        
    
    if not 'boxsize' in optInit:
        if not 'diameter' in optInit:
            # Ok, so we have no idea on the box size, so diameter is default of 400
            optInit['diameter'] = 400.0
        optInit['boxsize'] = optInit['diameter'] / optInit['apixM']
    
    maskName = pngFrontName + "_boxMask.png"
    generateAlphaMask( maskName, boxes.copy(), optInit['boxsize'], goodnessMetric.copy(), optPlot )
    
    #print( optPlot )
    
    if 'write_pref_min' in optPlot and bool(optPlot['write_pref_mic']):
        diagName = mrcFront + "_pref.mrc"
        pngName = pngFrontName + "_pref.png"
        generatePNG( diagName, pngName, boxes.copy(), optInit['boxsize'], goodnessMetric.copy(), optPlot )
    if 'write_ccmax_mic' in optPlot and bool(optPlot['write_ccmax_mic']):
        diagName = mrcFront + "_ccmax.mrc"
        pngName = pngFrontName + "_ccmax.png"
        generatePNG( diagName, pngName, boxes.copy(), optInit['boxsize'], goodnessMetric.copy(), optPlot)
    if 'write_bg_mic' in optPlot and bool(optPlot['write_bg_mic']):
        diagName = mrcFront + "_bg.mrc"
        pngName = pngFrontName + "_bg.png"
        generatePNG( diagName, pngName, boxes.copy(), optInit['boxsize'], goodnessMetric.copy(), optPlot)
    if 'write_bgfree_mic' in optPlot and bool(optPlot['write_bgfree_mic']):
        diagName = mrcFront + "_bgfree.mrc"
        pngName = pngFrontName + "_bgfree.png"
        generatePNG( diagName, pngName, boxes.copy(), optInit['boxsize'], goodnessMetric.copy(), optPlot )
    if 'write_lsigma_mic' in optPlot and bool(optPlot['write_lsigma_mic']):
        diagName = mrcFront + "_lsigma.mrc"
        pngName = pngFrontName + "_lsigma.png"
        generatePNG( diagName, pngName, boxes.copy(), optInit['boxsize'], goodnessMetric.copy(),  optPlot)
    if 'write_mic_mask' in optPlot and bool(optPlot['write_mic_mask']):
        diagName = mrcFront + "_mask.mrc"
        pngName = pngFrontName + "_mask.png"
        generatePNG( diagName, pngName, boxes.copy(), optInit['boxsize'], goodnessMetric.copy(), optPlot )
        
def generateAlphaMask( maskName, boxes, boxWidth, goodnessMetric, optPlot ):
    """
    Generate a PNG that is mostly transparent except for the boxes which are also high alpha.  To be 
    plotted on top of diagnostic images.
    """
    # print( "optPlot['binning'] = %s, type: %s" % (str(optPlot['binning']),type(optPlot['binning']) ))
    binShape = np.array( optPlot['shapeOriginal']  ) / optPlot['binning']
    
    # print( "boxes: %s, type: %s" % (boxes, type(boxes)))
    boxes = np.floor_divide( boxes, optPlot['binning'] )
    
    boxMask = np.zeros( [binShape[0], binShape[1], 4], dtype='float32' )
    # boxAlpha = 255*optPlot['boxAlpha']
    boxAlpha = optPlot['boxAlpha']
    colorMap = plt.get_cmap( optPlot['colorMap'] )
    
    boxWidth2 = np.int( boxWidth/2 )
    print( "DEBUG: writing box alpha mask for %d boxes" % boxes.shape[0] )
    if boxes.ndim > 1:
        for J in np.arange( boxes.shape[0] ):
            color = np.array( colorMap( goodnessMetric[J] ) )
            #color[:3] *=  (255 * color[:3] )
            color[3] = boxAlpha
            print( "Box at %s has color %s" %(boxes[J,:],color))
            
            # X-Y coordinates
            try:
                boxMask[ boxes[J,1]-boxWidth2:boxes[J,1]+boxWidth2, boxes[J,0]-boxWidth2:boxes[J,0]+boxWidth2,:3] += color[:3]
                # Force even alpha even with overlapping boxes
                boxMask[ boxes[J,1]-boxWidth2:boxes[J,1]+boxWidth2, boxes[J,0]-boxWidth2:boxes[J,0]+boxWidth2,:3] = color[3]
            except:
                pass # Don't draw anything
        
    # Save diagnostic image
    # We don't flip this because it's being plotted by matplotlib on top of our other diagnostic images.

    plt.figure()
    plt.imshow( boxMask )
    plt.title( "boxMask before clipping" )
    boxMask = (255* np.clip( boxMask, 0.0, 1.0) ).astype('uint8')

    plt.figure()
    plt.imshow( boxMask )
    plt.title( "boxMask after clipping" )
    plt.show( block=True )
    # Maybe the default skimage plugin can't handle alpha?
    skimage.io.imsave( maskName, boxMask  )
            
            
    
       
def generatePNG( diagName, pngName, boxes, boxWidth, goodnessMetric, optPlot ):
    ###############################################
    
    boxWidth2 = np.int( boxWidth/2 )
    if not os.path.isfile( diagName ):
        print( "File not found: %s"%diagName )
        return
          
    Mage, _ = mrcz.readMRC( diagName )
    if Mage.shape[0] <= 512:
        binning = 8
    elif Mage.shape[0] <= 682:
        binning = 6
    elif Mage.shape[0] <= 1024:
        binning = 4
    elif Mage.shape[0] <= 2048:
        binning = 2
    else:
        binning = 1

    boxes = np.floor_divide( boxes, binning )
    print( "DEBUG: binning = " + str(binning) )
    # Cut off the edges where the images may be uneven
    edge = np.floor_divide( optPlot['edge'], binning )
    # Gautomatch ouputs 927 x 927, which is 3708 x 3708 binned by 4
    x_special = 0
    boxes[:,0] -= (edge + x_special)
    # TODO: why is Y-axis origin of boxes wierd???  Ah-ha!  Gautomatch is making every image rectangular!  
    # How to know this special value without loading original image?
    y_special = np.floor_divide( (3838-3708) , (4*binning) )
    boxes[:,1] += (edge + y_special)
    
    cropMage = Mage[edge:-edge,edge:-edge]
    # Stretch contrast limits
    cutoff = 1E-3
    clim = util.histClim( cropMage, cutoff=cutoff )
    
    cropMage = ( 255 * util.normalize( np.clip( cropMage, clim[0], clim[1] ) ) ).astype('uint8')
    # Make into grayscale RGB
    cropMage = np.dstack( [cropMage, cropMage, cropMage] )
    
    # Make a origin box for debugging
    # cropMage[:edge,:edge,:] = np.zeros( 3, dtype='uint8' )
    # Write in colored boxes for particle positions
    colorMap = plt.get_cmap( optPlot['colorMap'] )
    if boxes.ndim > 1:
        for J in np.arange( boxes.shape[0] ):
            color = np.array( colorMap( goodnessMetric[J] )[:-1] )
            color /= np.sum( color )
            
            # X-Y coordinates
            # Did these somehow become box centers?
            # boxElem = cropMage[ boxes[J,0]:boxes[J,0]+boxes[J,2], boxes[J,1]:boxes[J,1]+boxes[J,3], : ]
            # boxElem = (1.0-optPlot['boxAlpha'])*boxElem + optPlot['boxAlpha']*color*boxElem
            # cropMage[ boxes[J,0]:boxes[J,0]+boxes[J,2], boxes[J,1]:boxes[J,1]+boxes[J,3], : ] = boxElem.astype('uint8')

            # Y-X coordinates
            try:
                boxElem = cropMage[ boxes[J,1]-boxWidth2:boxes[J,1]+boxWidth2, boxes[J,0]-boxWidth2:boxes[J,0]+boxWidth2,:]
                boxElem = (1.0-optPlot['boxAlpha'])*boxElem + optPlot['boxAlpha']*color*boxElem
                cropMage[ boxes[J,1]-boxWidth2:boxes[J,1]+boxWidth2, boxes[J,0]-boxWidth2:boxes[J,0]+boxWidth2,:] = boxElem.astype('uint8')
                # Draw origin
                # cropMage[ boxes[J,1],boxes[J,0],:] = np.array( [255, 0, 0], dtype='uint8' )
            except:
                pass # Don't draw anything
        
    # Save diagnostic image
    cropMage = np.flipud(cropMage)
    skimage.io.imsave( pngName, cropMage  )
    # Remove the MRC
    os.remove( diagName )
    
    pass

def batchProcess( mrcNames, pngFronts, optInit, optPlot, optRefine=None, n_processes=4 ):
    #outQueue = mp.Queue()
    pool = mp.Pool( processes=n_processes )
    
    
    GautoParam = [None] * len(mrcNames)
    for J, mrcName in enumerate(mrcNames):
        GautoParam[J] = [mrcNames[J],pngFronts[J],'batch', optInit, optPlot, optRefine]
        
    pool.map( runGauto, GautoParam )
    pool.close()
    pool.join()
#    try: 
#        pool.map( runGauto, GautoParam )
#        pool.close()
#    except Exception, e:
#        print( "Gautoauto Error: un-handled exception: " + str(e) )
#        pool.terminate()
#    finally:
#        pool.join()

if __name__ == "__main__":
    import sys

    # None for any entry in the dictionary will not be passed to Gautomatch
    optInit = {}
    optInit['apixM'] = 1.326
    optInit['diameter'] = 180
    optInit['boxsize']= 224
    optInit['min_dist'] = 240
    
    optInit['T'] = None # Templates.mrc
    optInit['apixT'] = None
    optInit['ang_step'] = None
    optInit['speed'] = None
    optInit['cc_cutoff'] = 0.1
    optInit['lsigma_D'] = 200
    optInit['lsigma_cutoff'] = None
    optInit['lave_D'] = 300
    optInit['lave_max'] = None
    optInit['lave_min'] = None
    optInit['hp'] = 400
    optInit['lp'] = None
    optInit['pre_lp'] = None
    optInit['pre_hp'] = None
    
    ##### OPTIONS FOR Gautomatch REFINEMENT ######
    optRefine = {}
    optRefine['doRefine'] = False
    
    optRefine['cdfThres'] = 0.90 # varies in range [0,1], for example 0.95 is 95th percentile
    
    ##### Plotting options #####
    optPlot = {}
    optPlot['edge'] = 64 # edge in pixels to chop off ccmax to avoid edge artifacts
    optPlot['boxAlpha'] = 0.25
    optPlot['colorMap'] = plt.cm.gist_rainbow
    optPlot['pref'] = False
    optPlot['ccmax'] = False
    optPlot['lsigma'] = False
    optPlot['bg'] = False
    optPlot['bgfree'] = True
    optPlot['mask'] = False
    
    mrcGlob = "*filt.mrc" # Try not to use *.mrc as it will match *ccmax.mrc for example

    t0 = time.time()
    
    mrcList = glob.glob( mrcGlob )
    
    if len( sys.argv ) > 1 and (sys.argv[1] == "--inspect" or sys.argv[1] == "-i"):    
        import random
        
        if len( sys.argv ) >= 3 and bool( sys.argv[2] ):
            icnt = int( sys.argv[2] )
        else:
            icnt = 3

        indices = np.array( random.sample( np.arange(0,len(mrcList) ), icnt) ).flatten()
        
        print( "Inspecting %d micrographs : %s" % (icnt,indices) )
        mrcList = np.array(mrcList)[ indices ]
        GautoParam = [None] * len(mrcList)
        for J, mrcName in enumerate(mrcList):
            GautoParam[J] = [mrcName,'inspect']
            
    else: #default behaviour is batch processing of entire directory.
        batchProcess( mrcList )
        
    
    # print( output )
    t1 = time.time()
    print( "Finished auto-Gautomatch in %.2f s" %(t1-t0) )
    
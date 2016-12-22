# -*- coding: utf-8 -*-
"""
Created on Mon May 23 11:00:35 2016

@author: Robert A. McLeod
"""

# TODO: move all this into ReliablePy?
from __future__ import division, print_function, absolute_import

import numpy as np
from . import  (zorro_util, zorro)
import time
import os, os.path, glob
import scipy.ndimage
import re
import collections
import mrcz
from mrcz import ReliablePy

# For a multiprocessing, maybe I should use subprocess pool and a master process?  That would avoid having to 
# any of the Python MPI libraries.  Then again maybe I should just learn mpi4py, likely it would be a 
# much more robust substitute for multiprocessing going forward.nan
try:
    from mpi4py import MPI
except:
    #print( "WARNING, zorro.extract: mpi4py module not found, use of multi-processed components will generate errors" )
    pass
    
def readCTF3Log( logName ):
    ctfDict = {}
    if os.path.isfile( logName ):
        ctfInfo = np.loadtxt( logName, skiprows=1, usecols=(0,1,2,3,4), dtype='str' )

    ctfDict['DefocusU'] = np.float32( ctfInfo[1,0] )
    ctfDict['DefocusV'] = np.float32( ctfInfo[1,1] )
    ctfDict['DefocusAngle'] = np.float32( ctfInfo[1,2] )
    ctfDict['Voltage'] = np.float32( ctfInfo[0,1] )
    ctfDict['SphericalAberration'] = np.float32( ctfInfo[0,0] )
    ctfDict['AmplitudeContrast'] = np.float32( ctfInfo[0,2] )
    ctfDict['Magnification'] = np.float32( ctfInfo[0,3] )
    ctfDict['DetectorPixelSize'] = np.float32( ctfInfo[0,4] )
    ctfDict['CtfFigureOfMerit'] = np.float32( ctfInfo[1,3] )
    
    return ctfDict
    
def readGCTFLog( logName ):
    ctfDict = {}
    
    with open( logName, 'rb' ) as lh:
        logLines = lh.read()
        
    pixelsize = np.float32( re.findall( "--apix\s+\d+\.\d+", logLines )[0].split()[1] )
    ctfDict['DetectorPixelSize'] = np.float32( re.findall( "--dstep\s+\d+\.\d+", logLines )[0].split()[1] )
    ctfDict['Magnification'] = 1E4 * ctfDict['DetectorPixelSize'] / pixelsize
    ctfDict['Voltage'] = np.float32( re.findall( "--kv\s+\d+\.\d+", logLines )[0].split()[1] )
    ctfDict['SphericalAberration'] = np.float32( re.findall( "--cs\s+\d+\.\d+", logLines )[0].split()[1] )
    ctfDict['AmplitudeContrast'] = np.float32( re.findall( "--ac\s+\d+\.\d+", logLines )[0].split()[1] )
    
    FinalString = re.findall( "\s+\d+\.\d+\s+\d+\.\d+\s+\d+\.\d+\s+\d+\.\d+\s+Final\sValues", logLines )
    FinalSplit = FinalString[0].split()
    ctfDict['DefocusU'] = np.float32( FinalSplit[0] )
    ctfDict['DefocusV'] = np.float32( FinalSplit[1] )
    ctfDict['DefocusAngle'] = np.float32( FinalSplit[2] )
    ctfDict['CtfFigureOfMerit'] = np.float32( FinalSplit[3] )
    return ctfDict
    
    

    

def partExtract( globPath, boxShape, boxExt=".star", 
                binShape = None, binKernel = 'lanczos2',
                rootName="part", sigmaFilt=-1.0, 
                invertContrast=True, normalize=True, fitBackground=True,
                movieMode=False, startFrame=None, endFrame=None, doseFilter=False ):
    """
    Extracts particles from aligned and summed micrographs (generally <micrograph>.mrc).  
    cd .
    globPath = "align\*.mrc" for example, will process all such mrc files.  
        *.log will use Zorro logs
        Can also just pass a list of files
        
    TODO: CHANGE TO LOAD FROM A ZORRO LOG FILE?  OR JUST MAKE IT THE PREFERRED OPTION?
    
    Expects to find <micrograph>.star in the directory which has all the box centers in Relion .star format
    
    binShape = [y,x] is the particle box size to resample to.  If == None no resampling is done.  
        For binning, binKernel = 'lanczos2' or 'gauss'  reflects the anti-aliasing filter used.
          
    rootName affects the suffix appended to each extracted particle stack (<micrograph>_<rootName>.mrcs)
    
    sigmaFilt is the standard deviation applied for removal of x-rays and hot pixels, 2.5 - 3.0 is the recommended range.
    It uses the sigmaFilt value to compute a confidence interval to filter the intensity value of only outlier 
    pixels (typically ~ 1 %)
    
    invertContrast = True, inverts the contrast, as required for Relion/Frealign.
    
    normalize = True changes the particles to have 0.0 mean and 1.0 standard deviation, as required for Relion/Frealign.
    
    fitBackground = True removes a 2D Gaussian from the image. In general it's better to perform this prior to 
    particle picking using the Zorro dose filtering+background subtraction mechanism.  
    
    TODO: GRAB THE CTF INFORMATION AS WELL AND MAKE A MERGED .STAR FILE
    
    TODO: add a movie mode that outputs a substack (dose-filtered) average.
    """
    
    t0 = time.time()
    if isinstance( globPath, list ) or isinstance( globPath, tuple ):
        mrcFiles = globPath
    else:
        mrcFiles = glob.glob( globPath )
        
    try:
        os.mkdir( "Particles" )
    except:
        pass
        
    particleStarList = [None]*len(mrcFiles)
    
    for K, mrcFileName in enumerate(mrcFiles):
        
        boxFileName = os.path.splitext( mrcFileName )[0] + boxExt
        if not os.path.isfile( boxFileName ):
            print( "Could not find .box/.star file: " + boxFileName )
            continue
        
        rlnBox = ReliablePy.ReliablePy()
        rlnBox.load( boxFileName )
    
        xCoord = rlnBox.star['data_']['CoordinateX']
        yCoord = rlnBox.star['data_']['CoordinateY']
    
        mrcMage = mrcz.readMRC( mrcFileName )[0]
        
        ###### Remove background from whole image #####
        if bool( fitBackground ):
            mrcMage -= zorro_util.backgroundEstimate( mrcMage )
            
        ###### Check for particles too close to the edge and remove those coordinates. #####
        keepElements = ~( (xCoord < boxShape[1]/2) | 
                        ( yCoord < boxShape[0]/2) | 
                        ( xCoord > mrcMage.shape[1]-boxShape[1]/2) | 
                        ( yCoord > mrcMage.shape[0]-boxShape[0]/2) )
        xCoord = xCoord[keepElements];  yCoord = yCoord[keepElements]
        
        ##### Extract particles #####
        particles = np.zeros( [len(xCoord), boxShape[0], boxShape[1]], dtype='float32' )
        for J in np.arange( len(xCoord) ):
            
            
            partMat = mrcMage[ yCoord[J]-boxShape[0]/2:yCoord[J]+boxShape[0]/2, xCoord[J]-boxShape[1]/2:xCoord[J]+boxShape[1]/2 ]
            
            ###### Apply confidence-interval gaussian filter #####
            if sigmaFilt > 0.0:
                partMean = np.mean( partMat )
                partStd = np.std( partMat )
                partHotpix = np.abs(partMat - partMean) > sigmaFilt*partStd
                # Clip before applying the median filter to better limit multiple pixel hot spots
                partMat = np.clip( partMat, partMean - sigmaFilt*partStd, partMean + sigmaFilt*partStd )
            
                # Let's stick to a gaussian_filter, it's much faster than a median filter and seems equivalent if we pre-clip
                # boxFilt[J,:,:] = scipy.ndimage.median_filter( boxMat[J,:,:], [5,5] )
                partFilt = scipy.ndimage.gaussian_filter( partMat, 4.0 )
                particles[J,:,:] = partHotpix * partFilt + (~ partHotpix) * partMat
            else:
                particles[J,:,:] = partMat
                
        #ims( -particles )
        
        ##### Re-scale particles #####
        if np.any( binShape ):
            binFact = np.array( boxShape ) / np.array( binShape )
            # Force binFact to power of 2 for now.
            
            import math
            binFact[0] = np.floor( math.log( binFact[0], 2 ) ) ** 2
            binFact[1] = np.floor( math.log( binFact[1], 2 ) ) ** 2
            
            [xSample,ySample] = np.meshgrid( np.arange( 0,binShape[1] )/binFact[1], np.arange( 0,binShape[0] )/binFact[0] )

                
            if binKernel == 'lanczos2':
                # 2nd order Lanczos kernel
                lOrder = 2
                xWin = np.arange( -lOrder, lOrder + 1.0/binFact[1], 1.0/binFact[1] )
                yWin = np.arange( -lOrder, lOrder + 1.0/binFact[0], 1.0/binFact[0] )
                xWinMesh, yWinMesh = np.meshgrid( xWin, yWin )
                rmesh = np.sqrt( xWinMesh*xWinMesh + yWinMesh*yWinMesh )
            
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    windowKernel = (lOrder/(np.pi*np.pi*rmesh*rmesh)) * np.sin( np.pi / lOrder * rmesh ) * np.sin( np.pi * rmesh ) 
                windowKernel[ yWin==0, xWin==0 ] = 1.0
                
            elif binKernel == 'gauss':
                xWin = np.arange( -2.0*binFact[1],2.0*binFact[1]+1.0 )
                yWin = np.arange( -2.0*binFact[0],2.0*binFact[0]+1.0 )
                print( "binFact = " + str(binFact) )
                xWinMesh, yWinMesh = np.meshgrid( xWin, yWin )
                # rmesh = np.sqrt( xWinMesh*xWinMesh + yWinMesh*yWinMesh )
                windowKernel = np.exp( -xWinMesh**2/(0.36788*binFact[1]) - yWinMesh**2/(0.36788*binFact[0]) )
                pass
            
            partRescaled = np.zeros( [len(xCoord), binShape[0], binShape[1]], dtype='float32' )
            for J in np.arange( len(xCoord) ):
                # TODO: switch from squarekernel to an interpolator so we can use non-powers of 2
                partRescaled[J,:,:] = zorro_util.squarekernel( scipy.ndimage.convolve( particles[J,:,:], windowKernel ), 
                    k= binFact[0] )
            
            particles = partRescaled
        pass
        # ims( windowKernel ) # DEBUG
    
        print( " particles.dtype = " + str(particles.dtype) )
        ###### Normalize particles after binning and background subtraction #####
        if bool(normalize):
            particles -= np.mean( particles, axis=0  )
            particles *= 1.0 / np.std( particles, axis=0  )
        
        ##### Invert contrast #####
        if bool(invertContrast):
            particles = -particles
            

        

        

        ###### Save particles to disk ######
        # Relion always saves particles within ./Particles/<fileext> but we could use anything if we want to 
        # make changes and save them in the star file.
        particleFileName = os.path.join( "Particles", os.path.splitext( mrcFileName )[0] +"_" + rootName + ".mrcs" )
        # TODO: add pixel size to particles file
        mrcz.writeMRC( particles, particleFileName ) # TODO: pixelsize
            
        ##### Output a star file with CTF and particle info. #####
        print( "Particle file: " + particleFileName )
        particleStarList[K] = os.path.splitext( particleFileName )[0] + ".star"
        print( "star file: " + particleStarList[K] )
        
        headerDict = { "ImageName":1, "CoordinateX":2, "CoordinateY":3, "MicrographName": 4 }
        lookupDict = dict( zip( headerDict.values(), headerDict.keys() ) )
        # 
        with open( particleStarList[K], 'wb' ) as fh:
            fh.write( "\ndata_images\n\nloop_\n")
        
            for J in np.sort(lookupDict.keys()):
                fh.write( "_rln" + lookupDict[J] + " #" + str(J) + "\n")

            for I in np.arange(0, len(xCoord) ):
                mrcsPartName = os.path.splitext( particleStarList[K] )[0] + ".mrcs" 
                fh.write( "%06d@%s  %.1f  %.1f  %s\n" % ( I+1, mrcsPartName, xCoord[I], yCoord[I], mrcFileName ) )

        
    # TODO: join all star files, for multiprocessing this should just return the list of star files
    # TODO: add CTF Info
    t1 = time.time()
    print( "Particle extraction finished in (s): %.2f" % (t1-t0) )
    return particleStarList
    
def joinParticleStars( outputName = "zbin2.star", starGlob="Particles/align/*.star", ctfExt="_gctf.log" ):
    """
    Take all the star files generated above, load the CTF information, and write a complete data.star
    file for Relion processing.
    """
    
    masterRln = ReliablePy.ReliablePy()
    masterRln.star['data_'] = collections.OrderedDict()
    masterRln.star['data_']['MicrographName'] = []
    masterRln.star['data_']['CoordinateX'] = []
    masterRln.star['data_']['CoordinateY'] = []
    masterRln.star['data_']['ImageName'] = []
    masterRln.star['data_']['DefocusU'] = []
    masterRln.star['data_']['DefocusV'] = []
    masterRln.star['data_']['DefocusAngle'] = []
    masterRln.star['data_']['Voltage'] = []
    masterRln.star['data_']['SphericalAberration'] = []
    masterRln.star['data_']['AmplitudeContrast'] = []
    masterRln.star['data_']['Magnification'] = []
    masterRln.star['data_']['DetectorPixelSize'] = []
    masterRln.star['data_']['CtfFigureOfMerit'] = []
    
    fh = open( outputName, 'wb' )
    fh.write( "\ndata_\n\nloop_\n")
    
    headerKeys = masterRln.star['data_'].keys()
    for J, key in enumerate(headerKeys):
        fh.write( "_rln" + key + " #" + str(J) + "\n")
        
        
    starList = glob.glob( starGlob )
    for starFile in starList:
        print( "Joining  " + starFile )
        
        stackRln = ReliablePy.ReliablePy()
        stackRln.load( starFile )
        
        # First check for and load a ctf log
        micrographName = stackRln.star['data_images']['MicrographName'][0]
        
        ##### Find the CTF info.  #####
        # 1st, look for a ctffind3.log file?
        logName = os.path.splitext( micrographName )[0] + ctfExt
        
        
        if not os.path.isfile( logName ):
            logName = os.path.splitext( micrographName )[0].rstrip("_filt") + ctfExt
            if not os.path.isfile( logName ):
                print( "WARNING: CTF results not found for : " + micrographName )
            else: 
                foundLog = True
        else: 
            foundLog = True
          
        try:
            if ctfExt == "_gctf.log" and foundLog:
                ctfDict = readGCTFLog( logName )
            elif ctfExt == "_ctffind3.log" and foundLog:
                ctfDict = readCTF3Log( logName )
            elif ctfExt == ".mrc.zor" and foundLog:
                zReg = zorro.ImageRegistrator()
                zReg.loadConfig( logName )
                ctfDict = zReg.CTFInfo
                ctfDict['Voltage'] = zReg.voltage
                ctfDict['SphericalAberration'] = zReg.C3
                ctfDict['Magnification'] = 1E4 * zReg.detectorPixelSize / zReg.pixelsize
                ctfDict['DetectorPixelSize'] = zReg.detectorPixelSize
        except:
            print( "Error: Could not load CTF log for %s" % micrographName )
            continue
        
        # If the log exists, add the star file
        n_part = len( stackRln.star['data_images']['MicrographName'] )
        
        # Build the dictionary up more
        stackRln.star['data_images']['DefocusU'] = [ctfDict['DefocusU']] * n_part   
        stackRln.star['data_images']['DefocusV'] = [ctfDict['DefocusV']] * n_part  
        stackRln.star['data_images']['DefocusAngle'] = [ctfDict['DefocusAngle']] * n_part  
        stackRln.star['data_images']['Voltage'] = [ctfDict['Voltage']] * n_part
        stackRln.star['data_images']['SphericalAberration'] = [ctfDict['SphericalAberration']] * n_part  
        stackRln.star['data_images']['AmplitudeContrast'] = [ctfDict['AmplitudeContrast']] * n_part   
        stackRln.star['data_images']['Magnification'] = [ctfDict['Magnification']] * n_part   
        stackRln.star['data_images']['DetectorPixelSize'] = [ctfDict['DetectorPixelSize']] * n_part   
        stackRln.star['data_images']['CtfFigureOfMerit'] = [ctfDict['CtfFigureOfMerit']] * n_part  
        
        # TODO: add extra columns from relion-2?
        
        for I in np.arange(n_part):
            fh.write( "    ")
            for J, key in enumerate(headerKeys):
                fh.write( str( stackRln.star['data_images'][key][I] ) )
                fh.write( "   " )
            fh.write( "\n" )
        
    fh.close()
    
if __name__ == "__main__":
#    bigList = glob.glob( "/Projects/BTV_GFP/filt/*.mrc" )
#    # TODO: split the bigList into N_worker processes
#    bigList = bigList[:8]
#    
#    rln = ReliablePy.ReliablePy()
#    partExtract( rln, bigList, boxShape=[512,512], 
#                boxExt=".star", binShape = [128,128], binKernel = 'lanczos2', rootName="zbin2", sigmaFilt=2.5, 
#                invertContrast=True, normalize=True, fitBackground=True )

    pass
    
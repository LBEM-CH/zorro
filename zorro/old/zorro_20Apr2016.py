#!/opt/anaconda/bin/python
# -*- coding: utf-8 -*-
# Unfortunately the `which` way of calling python can't accept command-line arguments.
"""
Created on Mon Nov 03 16:13:48 2014

@author: Robert A. McLeod
@email: robbmcleod@gmail.com OR robert.mcleod@unibas.ch

A selection of alignment routines designed for registering and summing stacks 
of images or diffraction patterns in the field of electron microscopy.
"""
from __future__ import division, print_function

import numpy as np
import numexprz as ne

try:
    print( "NumExpr initialized with: " + str( ne.nthreads ) + " threads" )
    # Now see which numexpr we have, by the dtype of float (whether it casts or not)
    tdata = np.complex64( 1.0 + 2.0j )
    fftw_dtype = ne.evaluate( 'tdata + tdata' ).dtype
    float_dtype = ne.evaluate( 'real(tdata+tdata)' ).dtype
    print( "NumExpr running with dtype = " + str(float_dtype) )
except: pass

import scipy.optimize
import scipy.ndimage
import time
import zorro_util as util
import zorro_plotting as plot
import ioMRC
import ioDM
import os, os.path
import subprocess

# Multiprocessing disabled on Windows due to general bugginess in the module?
import multiprocessing as mp

try:
    import pyfftw
except:
    print( "Zorro did not find pyFFTW package: get it at https://pypi.python.org/pypi/pyFFTW" )
try:
    import tables
except:
    print( "Zorro did not find pyTables installation for HDF5 file support" )
import matplotlib.pyplot as plt




#### OBJECT-ORIENTED INTERFACE ####
class ImageRegistrator(object):
# Should be able to handle differences in translation, rotation, and scaling
# between images
    
    def __init__( self ):
        # Declare class members
        self.verbose = 0
        self.saveC = False
        
        # Meta-information for processing, not saved in configuration files.
        self.METApriority = 0.0
        self.METAstatus = 'new'
        self.METAmtime = 0.0
        self.METAsize = 0
        
        # FFTW_PATIENT is bugged for powers of 2, so use FFTW_MEASURE as default
        self.fftw_effort = "FFTW_MEASURE"
        # TODO: change this to drop into cachePath
        
        self.n_threads = None # Number of cores to limit FFTW to, if None uses all cores 
        if os.name == 'nt':
            self.cachePath = "C:\\Temp\\"
        else:
            self.cachePath = "/scratch/"
        # Pass in to use a fixed wisdom file when planning FFTW
        # self.wisdom_file = os.path.join( self.cachePath, "fftw_wisdom.pkl" ) 
          
        # CALIBRATIONS
        self.pixelsize = None # Typically we use nanometers, the same units as Digital Micrograph
        self.voltage = None # Accelerating voltage, kV
        self.C3 = None # Spherical aberration of objective, mm

        # DEBUG
        # Timings
        self.t = np.zeros( [128] ) # This is an array so I can have fixed times in the code, rather than appending a list
        self.saveC = False
        self.saveCsub = False # Only saves the 16 x 16 about the maximum.
        
        # INFORMATION REDUCTION
        # The SNR at high spatial frequencies tends to be lower due to how information transfer works, so 
        # removing/filtering those frequencies can improve stability of the registration.  YMMV, IMHO, etc.
        self.Brad = 512 # Gaussian low-pass applied to data before registration, units are radius in Fourier space, or equivalent point-spread function in real-space
        self.Bmode = 'opti' # can be a real-space Gaussian convolution, 'conv' or Fourier filter, 'fourier', or 'opti' for automatic Brad
        # For Bmode = 'fourier', a range of available filters can be used: gaussian, gauss_trunc, butterworth.order (order is an int), hann, hamming
        self.BfiltType = 'gaussian'
        self.fouCrop = [3072,3072] # Size of FFT in frequency-space to crop to (e.g. [2048,2048])
        self.reloadData = True
        
        # Data
        self.images = None
        self.imageSum = None
        self.filtSum = None # Dose-filtered, Wiener-filtered, etc. representations go here
        self.doDoseFilter = False
        self.FFTSum = None
        # If you want to use one mask, it should have dims [1,N_Y,N_X]. This is 
        # to ensure Cythonized code can interact safely with Numpy
        self.incohFouMag = None # Incoherent Fourier magnitude, for CTF determination, resolution checks
        self.masks = None
        self.maskSum = None
        
        # Results
        self.translations = None
        self.transEven = None # For even-odd tiled FRC, the half-stack translations
        self.transOdd = None # For even-odd tiled FRC, the half-stack translations
        self.velocities = None # pixel velocity, in pix/frame, to find frames that suffer from excessive drift
        self.rotations = None # rotations, for polar-transformed data
        self.scales = None # scaling, for polar-transformed data
        self.errorDictList = [] # A list of dictionaries of errors and such from different runs on the same data.
        self.trackCorrStats = False
        self.corrStats = None
        self.doCTF = False
        self.CTF4Results = None
        self.CTF4Diag = None
        self.doLazyFRC = True
        self.FRC = None # A Fourier ring correlation
        
        # Registration parameters
        self.shapePadded = [4096,4096]
        self.shapeOriginal = None
        self.shapeBinned = None 
        self.subPixReg = 16 # fraction of a pixel to REGISTER image shift to
        # Subpixel alignment method: None (shifts still registered subpixally), lanczos, or fourier
        # lanczos is cheaper but can have strong artifacts, fourier is more expensive but may have less artifacts
        self.shiftMethod = 'lanczos' 
        self.maxShift = 80 # Generally should be 1/2 distance to next lattice spacing
        # Pre-shift every image by that of the previous frame, useful for high-resolution where one can jump a lattice
        # i.e. should be used with small values for maxShift
        self.preShift = False
        # Solver weighting can be raw max correlation coeffs (None), normalized to [0,1] by the 
        # min and max correlations ('norm'), or 'logistic' function weighted which
        # requires corrThres to be set.
        self.peakLocMode = 'interpolated' # interpolated (oversampled), or a RMS-best fit like fitlaplacian
        self.weightMode = 'autologistic' # autologistic, normalized, unweighted, logistic, or corr
        self.peaksigThres = 6.0
        self.logisticK = 5.0
        self.logisticNu = 0.15
        self.originMode = 'centroid' # 'centroid' or None
        self.suppressOrigin = True # Delete the XC pixel at (0,0).  Only necessary if gain reference is bad, but defaults to on.
        
        # Triangle-matrix indexing parameters
        self.triMode = 'diag' # Can be: tri, diag, auto, first
        self.startFrame = 0
        self.endFrame = 0
        self.diagWidth = 6
        self.autoMax = 10
        # TODO: these pixel thresholds aren't used anymore and should be removed before release
        self.sigmaThres = 4.0
        self.pixErrThres = 2.5
        # corrThres should not be less than 0.0013 for 3710x3838 arrays, as that is the Poisson limit
        self.corrThres = None # Use with 'auto' mode to stop doing cross-correlations if the values drop below the threshold
        
        self.velocityThres = None # Pixel velocity threshold (pix/frame), above which to throw-out frames with too much motion blur.
        
        #### INPUT/OUTPUT ####
        self.files = { "config":None, "stack":None, "mask":None, "sum":None, "align":None, "figurePath":None,
                      "xc":None, "moveRawPath":None, "original":None }

        #self.savePDF = False
        self.savePNG = False
        self.saveMovie = True
        self.doCompression = False
        self.compress_ext = ".bz2"

        #### PLOTTING ####
        self.plotDict = { "imageSum":True, "imageFirst":False, "FFTSum":False, "polarFFTSum":True, 
                         "filtSum":True, 'stats': False,
                         "corrTriMat":False, "peaksigTriMat": True, 
                         "translations":True, "pixRegError":True, 
                         "CTF4Diag":True, "logisticWeights": True, "lazyFRC": True, 
                         'Transparent': True, 'dpi':144, 'image_cmap':'gray', 'graph_cmap':'gnuplot', 
                         'fontsize':12, 'fontstyle': 'sans-serif',
                         'show':False }
        pass
    
    def initDefaultFiles( self, stackName ):
        self.files['stack'] = stackName
        self.files['config'] = stackName + ".log"

        stackPath, stackFront = os.path.split( stackName )
        stackFront = os.path.splitext( stackFront )[0]
        
        self.files['align'] = os.path.relpath( os.path.join( "./align", stackFront + "_zorro_movie.mrcs" ), start=stackPath )
        self.files['sum'] = os.path.relpath( stackPath, os.path.join( "./sum", stackFront + "_zorro.mrc" ), start=stackPath ) 
        self.files['figurePath'] = os.path.relpath( os.path.join(stackPath, "./figs"), start=stackPath  )

            
    def xcorr2_mc( self, gpu_id = 0, loadResult=True ):
        """
        This makes an external operating system call to the Cheng's lab GPU-based 
        B-factor multireference executable. It and CUDA libraries must be on the system 
        path and libary path respectively.
        
        NOTE: Spyder looks loads PATH and LD_LIBRARY_PATH from .profile, not .bashrc
        """
            
        import re, os
        # import uuid
        
        dosef_cmd = util.which("dosefgpu_driftcorr")
        if dosef_cmd is None:
            print( "Error: dosefgpu_driftcorr not found in system path." )
            return
        
        #tempFileHash = str(uuid.uuid4() ) # Key let's us multiprocess safely
        stackBase = os.path.basename( os.path.splitext( self.files['stack'] )[0] )
        
        if self.cachePath is None:
            self.cachePath = "."
            
        InName = os.path.join( self.cachePath, stackBase + "_mcIn.mrc" )
        # Unfortunately these files may as well be in the working directory.    
        OutAvName = os.path.join( self.cachePath, stackBase + "_mcOutAv.mrc" )
        OutStackName = os.path.join( self.cachePath, stackBase + "_mcOut.mrc" )
        LogName = os.path.join( self.cachePath, stackBase + "_mc.log" )
        ioMRC.MRCExport( self.images.astype('float32'), InName )

        # Force binning to 1, as performance with binning is poor
        binning = 1
        if self.Brad is not None:
            # Li masking is in MkPosList() in cufunc.cu (line 413)
            # Their r2 is normalized and mine isn't
            # Li has mask = exp( -0.5 * bfactor * r_norm**2 )
            # r_norm**2 = x*x/Nx*Nx + y*y/Ny*Ny = r**2 / (Nx**2 + Ny**2)
            # For non-square arrays they have a non-square (but constant frequency) filter 
            # RAM has mask = exp( -(r/brad)**2 )
            # We can only get Bfactor approximately then but it's close enough for 3710x3838
            bfac = 2.0 * (self.images.shape[1]**2 + self.images.shape[2]**2) / (self.Brad**2) 
            print( "Using B-factor of " + str(bfac) + " for dosefgpu_driftcorr" )
        else:
            bfac = 1000 # dosef default 'safe' bfactor for mediocre gain reference
        # Consider: Dosef suffers at the ends of the sequence, so make the middle frame zero drift?
        # align_to = np.floor( self.images.shape[0]/2 )
        # This seems to cause more problems then it's worth.
        align_to = 0
        if self.diagWidth != None:
            fod = self.diagWidth
        else:
            fod = 0
        # Dosef can limit search to a certain box size    
        if self.maxShift == None:
            maxshift = 96
        else:
            maxshift = self.maxShift * 2
        if self.startFrame == None:
            self.startFrame = 0
        if self.endFrame == None:
            self.endFrame = 0

        motion_flags = (  " " + InName 
                + " -gpu " + str(gpu_id)
                + " -nss " + str(self.startFrame) 
                + " -nes " + str(self.endFrame) 
                + " -fod " + str(fod) 
                + " -bin " + str(binning) 
                + " -bft " + str(bfac) 
                + " -atm -" + str(align_to) 
                + " -pbx " + str(maxshift)
                + " -ssc 1 -fct " + OutStackName 
                + " -fcs " + OutAvName 
                + " -flg " + LogName )

        sub = subprocess.Popen( dosef_cmd + motion_flags, shell=True )
        sub.wait()
        # os.system(dosef_cmd + motion_flags)
        
        # Parse to get the translations
        fhMC = open( LogName )
        MClog = fhMC.readlines()
        fhMC.close()
        
        # Number of footer lines changes with the options you use.
        # I would rather find Sum Frame #000
        for linenumber, line in enumerate(MClog):
            try: 
                test = re.findall( "Sum Frame #000", line)
                if bool(test): 
                    frameCount = np.int( re.findall( "\d\d\d", line )[1] ) + 1
                    break
            except: pass
        
        MClog_crop = MClog[linenumber+1:linenumber+frameCount+1]
        MCdrifts = np.zeros( [frameCount,2] )
        for J in np.arange(0,frameCount):
            MCdrifts[J,:] = re.findall( r"([+-]?\d+.\d+)", MClog_crop[J] )[1:]
        # Zorro saves translations, motioncorr saves shifts.
        self.translations = -np.fliplr( MCdrifts )
        
        if self.originMode == 'centroid':
            centroid = np.mean( self.translations, axis=0 )
            self.translations -= centroid

        
        if bool(loadResult):
            print( "Loading MotionCorr aligned frames into ImageRegistrator.images" )
            self.images = ioMRC.MRCImport( OutStackName )
            # Calclulate sum
            self.imageSum = np.sum( self.images, axis=0 )
            

        time.sleep(0.5)
        try: os.remove(InName)
        except: pass
        try: os.remove(OutStackName)
        except: pass
        try: os.remove(OutAvName)
        except: pass
        try: os.remove(LogName)
        except: pass
        
    def xcorr2_unblur( self, dosePerFrame = None, restoreNoise = True, minShift = 2.0, terminationThres = 0.1, 
                      maxIteration=10, verbose=False, loadResult=True   ):
        """
        Calls UnBlur by Grant and Rohou using the Zorro interface.
        """
        unblur_exename = "unblur_openmp_7_17_15.exe"
        if util.which( unblur_exename ) is None:
            print( "UnBlur not found in system path" )
            return
        
        print( "Calling UnBlur for " + self.files['stack'] )
        print( "   written by Timothy Grant and Alexis Rohou: http://grigoriefflab.janelia.org/unblur" )
        print( "   http://grigoriefflab.janelia.org/node/4900" )
        
        import os
        
        try: os.umask( 0002 ) # Why is Python not using default umask from OS?
        except: pass
        
        if self.cachePath is None:
            self.cachePath = "."
            
        # Force trailing slashes onto cachePatch

        
        stackBase = os.path.basename( os.path.splitext( self.files['stack'] )[0] )
        frcOutName = os.path.join( self.cachePath, stackBase + "_unblur_frc.txt" )
        shiftsOutName = os.path.join( self.cachePath, stackBase + "_unblur_shifts.txt" )
        outputAvName = os.path.join( self.cachePath, stackBase + "_unblur.mrc" )
        outputStackName = os.path.join( self.cachePath, stackBase + "_unblur_movie.mrc" )
        

        ps = self.pixelsize * 10.0
        if bool( self.doDoseFilter ):
            if dosePerFrame == None:
                # We have to guesstimate the dose per frame in e/A^2 if it's not provided
                dosePerFrame = np.mean( self.images ) / (ps*ps)
            preExposure = 0.0
        if self.Brad is not None:
            # Li masking is in MkPosList() in cufunc.cu (line 413)
            # Their r2 is normalized and mine isn't
            # Li has mask = exp( -0.5 * bfactor * r_norm**2 )
            # r_norm**2 = x*x/Nx*Nx + y*y/Ny*Ny = r**2 / (Nx**2 + Ny**2)
            # For non-square arrays they have a non-square (but constant frequency) filter 
            # RAM has mask = exp( -(r/brad)**2 )
            # We can only get Bfactor approximately then but it's close enough for 3710x3838
            bfac = 2.0 * (self.images.shape[1]**2 + self.images.shape[2]**2) / (self.Brad**2) 
            print( "Using B-factor of " + str(bfac) + " for UnBlur" )
        else:
            bfac = 1000 # dosef default 'safe' bfactor for mediocre gain reference
        outerShift = self.maxShift * ps
        # RAM: I see no reason to let people change the Fourier masking
        vertFouMaskHW = 1
        horzFouMaskHW = 1
        
        try: 
            mrcname = os.path.join( self.cachePath, stackBase + "_unblurIN.mrc" )
            ioMRC.MRCExport( self.images.astype('float32'), mrcname )
        except:
            print( "Error in exporting MRC file to UnBlur" )
            return
         
        # Are there flags for unblur?  Check the source code.
        flags = "" # Not using any flags
         
        unblurexec = ( unblur_exename + " " + flags + " << STOP_PARSING \n" + mrcname )
        
        unblurexec = (unblurexec + "\n" + str(self.images.shape[0]) + "\n" +
            outputAvName + "\n" + shiftsOutName + "\n" + str(ps) + "\n" +
            str(self.doDoseFilter) )
            
        if bool(self.doDoseFilter):
            unblurexec += "\n" + str(dosePerFrame) + "\n" + str(self.voltage) + "\n" + str(preExposure)
            
        unblurexec += ("\n yes \n" + outputStackName + "\n yes \n" + 
            frcOutName + "\n" + str(minShift) + "\n" + str(outerShift) + "\n" +
            str(bfac) + "\n" + str( np.int(vertFouMaskHW) ) + "\n" + str( np.int(horzFouMaskHW) ) + "\n" +
            str(terminationThres) + "\n" + str(maxIteration) )
            
        if bool(self.doDoseFilter):
            print( "Warning: restoreNoise is not implemented in UnBlur's source as of version 7_17_15" )
            unblurexec += "\n" + str(restoreNoise)
            
        unblurexec += "\n" + str(verbose) 
              
        unblurexec = unblurexec + "\nSTOP_PARSING"
        
        print( unblurexec )
        sub = subprocess.Popen( unblurexec, shell=True )
        sub.wait()
        
        # os.system( unblurexec )
        
        try:
            self.FRC = np.loadtxt(frcOutName, comments='#', skiprows=0 )
            self.translations = np.loadtxt( shiftsOutName, comments='#', skiprows=0 ).transpose()
            # UnBlur uses Fortran ordering, so we need to swap y and x for Zorro C-ordering
            self.translations = np.fliplr( self.translations )
            # UnBlur returns drift in Angstroms
            self.translations /= ps
            # UnBlur registers to middle frame
            # print( "WARNING: UnBlur shift normalization modified for testing" )
            self.translations -= self.translations[0,:]
            
            if bool( loadResult ):
                print( "Loading UnBlur aligned frames into ImageRegistrator.images" )
                self.imageSum = ioMRC.MRCImport( outputAvName )
                self.images = ioMRC.MRCImport( outputStackName )
        except IOError:
            print( "UnBlur likely core-dumped, try different input parameters?" )
        finally:
            time.sleep(0.5) # DEBUG: try and see if temporary files are deleteable now.
        
            frcOutName = os.path.join( self.cachePath, stackBase + "_unblur_frc.txt" )
            shiftsOutName = os.path.join( self.cachePath, stackBase + "_unblur_shifts.txt" )
            outputAvName = os.path.join( self.cachePath, stackBase + "_unblur.mrc" )
            outputStackName = os.path.join( self.cachePath, stackBase + "_unblur_movie.mrc" )
        pass
    
        if self.originMode == 'centroid':
            centroid = np.mean( self.translations, axis=0 )
            self.translations -= centroid
    
        time.sleep(0.5)
        try: os.remove( mrcname )
        except: print( "Could not remove Unblur MRC input file" )
        try: os.remove( frcOutName )
        except: print( "Could not remove Unblur FRC file" )
        try: os.remove( shiftsOutName )
        except: print( "Could not remove Unblur Shifts file" )
        try: os.remove( outputAvName )
        except: print( "Could not remove Unblur MRC average" )
        try: os.remove( outputStackName )
        except: print( "Could not remove Unblur MRC stack" )
        
    def xcorrnm2_tri( self, triIndices=None ):
        """
        Robert A. McLeod
        robbmcleod@gmail.com
        April 16, 2015
        
        triIndices is the index locations to correlate to.  If None, self.triMode 
        is used to build one.  Normally you should use self.triMode for the first iteration, 
        and pass in a triIndice from the errorDict if you want to repeat.
        
        returns : [shiftsTriMat, corrTriMat]
        
        This is an evolution of the Padfield cross-correlation algorithm  to take 
        advantage of the Cheng multi-reference approach for cross-correlation 
        alignment of movies. 
            Padfield, "Masked object registration in the Fourier domain," IEEE
            Transactions on Image Processing 21(5) (2012): 3706-2718.
            
            Li et al. Nature Methods, 10 (2013): 584-590.
            
        It cross-correlates every frame to every other frame to build a triangular
        matrix of shifts and then does a functional minimization over the set of 
        equations. This means the computational cost grows with a power law with
        the number of frames but it is more noise resistant.  
        

        triIndices can be an arbitrary boolean N x N matrix of frames to correlate
        Alternatively it can be a string which will generate an appropriate matrix:
             'tri' (default) correlates all frames to eachother
             'first' is correlate to the first frame as a template
             'diag' correlates to the next frame (i.e. a diagonal )
             'auto' is like diag but automatically determines when to stop based on corrcoeffThes
        diagWidth is for 'diag' and the number of frames to correlate each frame to, 
            default is None, which does the entire triangular matrix
            diagWidth = 1 correlates to each preceding frame
        
        NOTE: only calculates FFTs up to Nyquist/2.
        """
        shapeImage = np.array( [self.images.shape[1], self.images.shape[2]] )
#        N = np.asarray( self.images.shape )[0] - 1
        N = np.asarray( self.images.shape )[0]
            
        self.t[0] = time.time() 
        if self.preShift:
            print( "Warning: Preshift will break if there are skipped frames in a triIndices row." )

        # Test to see if triIndices is a np.array or use self.triMode
        if hasattr( triIndices, "__array__" ): # np.array
            # Ensure triIndices is a square array of the right size
            if triIndices.shape[0] != N or triIndices.shape[1] != N:
                raise IndexError("triIndices is wrong size, should be of length: " + str(N) )

        elif triIndices is None:
            [xmesh, ymesh] = np.meshgrid( np.arange(0,N), np.arange(0,N) )
            trimesh = xmesh - ymesh
            # Build the triMat if it wasn't passed in as an array
            if( self.triMode == 'first' ):
                print( "Correlating in template mode to first image" )
                triIndices = np.ones( [1,N], dtype='bool' )
                triIndices[0,0] = False # Don't autocorrelate the first frame.
            elif( self.triMode == 'diag' ):
                if (self.diagWidth is None) or (self.diagWidth < 0):
                    # For negative numbers, align the entire triangular matrix
                    self.diagWidth = N
                    
                triIndices = (trimesh <= self.diagWidth ) * (trimesh > 0 )
                print( "Correlating in diagonal mode with width " + str(self.diagWidth) )
            elif( self.triMode == 'autocorr' ):
                triIndices = (trimesh == 0)
            elif( self.triMode == 'refine' ):
                triIndices = trimesh == 0
            else: # 'tri' or 'auto' ; default is an upper triangular matrix
                triIndices = trimesh >= 1
            pass
        else:
            raise TypeError( "Error: triIndices not recognized as valid: " + str(triIndices) )
            

        if self.masks is None or self.masks == []:
            print( "Warning: No mask not recommened with MNXC-style correlation" )
            self.masks = np.ones( [1,shapeImage[0],shapeImage[1]], dtype = self.images.dtype )
            
        if( self.masks.ndim == 2 ):
            print( "Forcing masks to have ndim == 3" )
            self.masks = np.reshape( self.masks.astype(self.images.dtype), [1,shapeImage[0],shapeImage[1]] )
             
        # Pre-loop allocation
        self.__shiftsTriMat = np.zeros( [N,N,2], dtype=float_dtype ) # Triagonal matrix of shifts in [I,J,(y,x)]
        self.__corrTriMat = np.zeros( [N,N], dtype=float_dtype ) # Triagonal matrix of maximum correlation coefficient in [I,J]
        self.__peaksigTriMat = np.zeros( [N,N], dtype=float_dtype ) # Triagonal matrix of correlation peak contrast level
        
        # Make pyFFTW objects
        if self.fouCrop is None:
            tempFullframe = pyfftw.n_byte_align_empty( shapeImage, fftw_dtype.itemsize, dtype=fftw_dtype )
            self.__FFT2, self.__IFFT2 = util.pyFFTWPlanner( tempFullframe, wisdomFile=os.path.join( self.cachePath, "fftw_wisdom.pkl" ), effort = self.fftw_effort, n_threads=self.n_threads )
            shapeCropped = shapeImage
            self.__tempComplex = pyfftw.n_byte_align_empty( shapeCropped, fftw_dtype.itemsize, dtype=fftw_dtype )
        else:
            tempFullframe = pyfftw.n_byte_align_empty( shapeImage, fftw_dtype.itemsize, dtype=fftw_dtype )
            self.__FFT2, _ = util.pyFFTWPlanner( tempFullframe, wisdomFile=os.path.join( self.cachePath, "fftw_wisdom.pkl" ) , effort = self.fftw_effort, n_threads=self.n_threads, doReverse=False )
            # Force fouCrop to multiple of 2
            shapeCropped = 2 * np.floor( np.array( self.fouCrop ) / 2.0 ).astype('int')
            self.__tempComplex = pyfftw.n_byte_align_empty( shapeCropped, fftw_dtype.itemsize, dtype=fftw_dtype )
            _, self.__IFFT2 = util.pyFFTWPlanner( self.__tempComplex, wisdomFile=os.path.join( self.cachePath, "fftw_wisdom.pkl" ) , effort = self.fftw_effort, n_threads=self.n_threads, doForward=False )
        
        self.__templateImageFFT = pyfftw.n_byte_align_empty( shapeCropped, fftw_dtype.itemsize, dtype=fftw_dtype )
        self.__templateSquaredFFT = pyfftw.n_byte_align_empty( shapeCropped, fftw_dtype.itemsize, dtype=fftw_dtype )
        self.__templateMaskFFT = pyfftw.n_byte_align_empty( shapeCropped, fftw_dtype.itemsize, dtype=fftw_dtype )
        self.__tempComplex2 = pyfftw.n_byte_align_empty( shapeCropped, fftw_dtype.itemsize, dtype=fftw_dtype )
        
        # Subpixel initialization
        # Ideally subPix should be a power of 2 (i.e. 2,4,8,16,32)
        self.__subR = 8 # Sampling range around peak of +/- subR
        if self.subPixReg is None: self.subPixReg = 1;
        if self.subPixReg > 1.0:  
            # hannfilt = np.fft.fftshift( ram.apodization( name='hann', size=[subR*2,subR*2], radius=[subR,subR] ) ).astype( fftw_dtype )
            # Need a forward transform that is [subR*2,subR*2] 
            self.__Csub = pyfftw.n_byte_align_empty( [self.__subR*2,self.__subR*2], fftw_dtype.itemsize, dtype=fftw_dtype )
            self.__CsubFFT = pyfftw.n_byte_align_empty( [self.__subR*2,self.__subR*2], fftw_dtype.itemsize, dtype=fftw_dtype )
            self.__subFFT2, _ = util.pyFFTWPlanner( self.__Csub, fouMage=self.__CsubFFT, wisdomFile=os.path.join( self.cachePath, "fftw_wisdom.pkl" ) , effort = self.fftw_effort, n_threads=self.n_threads, doReverse = False )
            # and reverse transform that is [subR*2*subPix, subR*2*subPix]
            self.__CpadFFT = pyfftw.n_byte_align_empty( [self.__subR*2*self.subPixReg,self.__subR*2*self.subPixReg], fftw_dtype.itemsize, dtype=fftw_dtype )
            self.__Csub_over = pyfftw.n_byte_align_empty( [self.__subR*2*self.subPixReg,self.__subR*2*self.subPixReg], fftw_dtype.itemsize, dtype=fftw_dtype )
            _, self.__subIFFT2 = util.pyFFTWPlanner( self.__CpadFFT, fouMage=self.__Csub_over, wisdomFile=os.path.join( self.cachePath, "fftw_wisdom.pkl" ) , effort = self.fftw_effort, n_threads=self.n_threads, doForward = False )
        

        self.__maskProduct = np.zeros( shapeCropped )
        normConst2 = np.float32( 1.0 / ( np.float64(shapeCropped[0])*np.float64(shapeCropped[1]))**2.0 )
        
        if self.masks.shape[0] == 1 :
            # tempComplex = self.masks[0,:,:].astype( fftw_dtype ) 
            self.__baseMaskFFT = pyfftw.n_byte_align_empty( shapeCropped, fftw_dtype.itemsize, dtype=fftw_dtype )

            self.__FFT2.update_arrays( self.masks[0,:,:].squeeze().astype( fftw_dtype ), tempFullframe ); self.__FFT2.execute()
            # FFTCrop
            self.__baseMaskFFT[0:shapeCropped[0]/2,0:shapeCropped[1]/2] = tempFullframe[0:shapeCropped[0]/2,0:shapeCropped[1]/2]
            self.__baseMaskFFT[0:shapeCropped[0]/2,-shapeCropped[1]/2:] = tempFullframe[0:shapeCropped[0]/2,-shapeCropped[1]/2:] 
            self.__baseMaskFFT[-shapeCropped[0]/2:,0:shapeCropped[1]/2] = tempFullframe[-shapeCropped[0]/2:,0:shapeCropped[1]/2] 
            self.__baseMaskFFT[-shapeCropped[0]/2:,-shapeCropped[1]/2:] = tempFullframe[-shapeCropped[0]/2:,-shapeCropped[1]/2:] 
            
            self.__templateMaskFFT = np.conj( self.__baseMaskFFT )
            
            # maskProduct term is M1^* .* M2
            templateMaskFFT = self.__templateMaskFFT; 
            baseMaskFFT = self.__baseMaskFFT # Pointer assignment
            self.__tempComplex2 = ne.evaluate( "templateMaskFFT * baseMaskFFT" )
            self.__IFFT2.update_arrays( self.__tempComplex2, self.__tempComplex ); self.__IFFT2.execute()
            tempComplex = self.__tempComplex
            self.__maskProduct = ne.evaluate( "normConst2*real(tempComplex)" )
        else:
            # Pre-allocate only
            self.__baseMaskFFT = np.zeros( [N, shapeCropped[0], shapeCropped[1]], dtype=fftw_dtype )
        
            
        if bool( self.maxShift ) or self.Bmode is 'fourier':
            if self.maxShift is None or self.preShift is True:
                [xmesh,ymesh] = np.meshgrid( np.arange(-shapeCropped[0]/2, shapeCropped[0]/2), np.arange(-shapeCropped[1]/2, shapeCropped[1]/2)  )
            else:
                [xmesh,ymesh] = np.meshgrid( np.arange(-self.maxShift, self.maxShift), np.arange(-self.maxShift, self.maxShift)  )
            
            rmesh2 = ne.evaluate( "xmesh*xmesh + ymesh*ymesh" )
            # rmesh2 = xmesh*xmesh + ymesh*ymesh
            if bool( self.maxShift ): 
                self.__mask_maxShift = ( rmesh2 < self.maxShift**2.0 )
            if self.Bmode is 'fourier':
                self.__Bfilter = np.fft.fftshift( util.apodization( name=self.BfiltType, size=shapeCropped, radius=[self.Brad,self.Brad] ) )

        self.t[1] = time.time() 
        # Pre-compute forward FFTs (template will just be copied conjugate Fourier spectra)
        self.__imageFFT = np.zeros( [N, self.shapePadded[0], self.shapePadded[1]], dtype=fftw_dtype )
        self.__baseImageFFT = np.zeros( [N, shapeCropped[0], shapeCropped[1]], dtype=fftw_dtype )
        self.__baseSquaredFFT = np.zeros( [N, shapeCropped[0], shapeCropped[1]], dtype=fftw_dtype )
        
        # Looping for triagonal matrix
        # For auto this is wrong, so make these lists instead
        currIndex = 0
        if self.saveC: self.C = []
        if self.saveCsub: self.Csub = []
        
        print( "Pre-computing forward Fourier transforms" )
        # For even-odd and noise estimates, we often skip many rows
        # precompIndices = np.unique( np.vstack( [np.argwhere( np.sum( triIndices, axis=1 ) > 0 ), np.argwhere( np.sum( triIndices, axis=0 ) > 0 ) ] ) )
        precompIndices = np.unique( np.vstack( [np.argwhere( np.sum( triIndices, axis=1 ) >= 0 ), 
                                                np.argwhere( np.sum( triIndices, axis=0 ) >= 0 ) ] ) )
        for I in precompIndices:
            if self.verbose >= 2: 
                print( "Precomputing forward FFT frame: " + str(I) )
                
            # Apply masks to images
            if self.masks.shape[0] == 1:
                masks_block = self.masks[0,:,:]
                images_block = self.images[I,:,:]
            else:
                masks_block = self.masks[I,:,:]
                images_block = self.images[I,:,:]
                
            tempReal = ne.evaluate( "masks_block * images_block" ).astype( fftw_dtype )
            self.__FFT2.update_arrays( tempReal, tempFullframe ); self.__FFT2.execute()
            if self.shiftMethod == "fourier":
                self.__imageFFT[I,:,:] = tempFullframe.copy(order='C')
                # FFTCrop
                self.__baseImageFFT[I,0:shapeCropped[0]/2,0:shapeCropped[1]/2] = self.__imageFFT[I,0:shapeCropped[0]/2,0:shapeCropped[1]/2]
                self.__baseImageFFT[I,0:shapeCropped[0]/2,-shapeCropped[1]/2:] = self.__imageFFT[I,0:shapeCropped[0]/2,-shapeCropped[1]/2:] 
                self.__baseImageFFT[I,-shapeCropped[0]/2:,0:shapeCropped[1]/2] = self.__imageFFT[I,-shapeCropped[0]/2:,0:shapeCropped[1]/2] 
                self.__baseImageFFT[I,-shapeCropped[0]/2:,-shapeCropped[1]/2:] = self.__imageFFT[I,-shapeCropped[0]/2:,-shapeCropped[1]/2:] 
                print( "TODO: check memory consumption" )
            else:
                # FFTCrop
                self.__baseImageFFT[I,0:shapeCropped[0]/2,0:shapeCropped[1]/2] = tempFullframe[0:shapeCropped[0]/2,0:shapeCropped[1]/2]
                self.__baseImageFFT[I,0:shapeCropped[0]/2,-shapeCropped[1]/2:] = tempFullframe[0:shapeCropped[0]/2,-shapeCropped[1]/2:] 
                self.__baseImageFFT[I,-shapeCropped[0]/2:,0:shapeCropped[1]/2] = tempFullframe[-shapeCropped[0]/2:,0:shapeCropped[1]/2] 
                self.__baseImageFFT[I,-shapeCropped[0]/2:,-shapeCropped[1]/2:] = tempFullframe[-shapeCropped[0]/2:,-shapeCropped[1]/2:] 
            

            
            self.__FFT2.update_arrays( ne.evaluate( "tempReal*tempReal" ).astype( fftw_dtype ), tempFullframe ); self.__FFT2.execute()
            # FFTCrop
            self.__baseSquaredFFT[I,0:shapeCropped[0]/2,0:shapeCropped[1]/2] = tempFullframe[0:shapeCropped[0]/2,0:shapeCropped[1]/2]
            self.__baseSquaredFFT[I,0:shapeCropped[0]/2,-shapeCropped[1]/2:] = tempFullframe[0:shapeCropped[0]/2,-shapeCropped[1]/2:] 
            self.__baseSquaredFFT[I,-shapeCropped[0]/2:,0:shapeCropped[1]/2] = tempFullframe[-shapeCropped[0]/2:,0:shapeCropped[1]/2] 
            self.__baseSquaredFFT[I,-shapeCropped[0]/2:,-shapeCropped[1]/2:] = tempFullframe[-shapeCropped[0]/2:,-shapeCropped[1]/2:] 
            
            if not self.masks.shape[0] == 1:
                self.__FFT2.update_arrays( self.masks[I,:,:].squeeze().astype( fftw_dtype), tempFullframe ); self.__FFT2.execute()
                # FFTCrop
                self.__baseMaskFFT[I,0:shapeCropped[0]/2,0:shapeCropped[1]/2] = tempFullframe[0:shapeCropped[0]/2,0:shapeCropped[1]/2]
                self.__baseMaskFFT[I,0:shapeCropped[0]/2,-shapeCropped[1]/2:] = tempFullframe[0:shapeCropped[0]/2,-shapeCropped[1]/2:] 
                self.__baseMaskFFT[I,-shapeCropped[0]/2:,0:shapeCropped[1]/2] = tempFullframe[-shapeCropped[0]/2:,0:shapeCropped[1]/2] 
                self.__baseMaskFFT[I,-shapeCropped[0]/2:,-shapeCropped[1]/2:] = tempFullframe[-shapeCropped[0]/2:,-shapeCropped[1]/2:] 

            pass
        del masks_block, images_block
        self.t[2] = time.time() 
    
        print( "Starting correlation calculations, mode: " + self.triMode )
        if self.triMode == 'refine':
            
            # Find FFT sum (it must be reduced by the current frame later)
            # FIXME: Is there some reason this might not be linear after FFT?
            # FIXME: is it the complex conjugate operation below???
            self.__sumFFT = np.sum( self.__baseImageFFT, axis = 0 )
            self.__sumSquaredFFT = np.sum( self.__baseSquaredFFT, axis = 0 )
            
            print( "In refine" )
            for I in np.arange(self.images.shape[0] - 1):
                # In refine mode we have to build the template on the fly from imageSum - currentImage
                self.__templateImageFFT = np.conj( self.__sumFFT - self.__baseImageFFT[I,:,:]  ) / self.images.shape[0]
                self.__templateSquaredFFT = np.conj( self.__sumSquaredFFT - self.__baseSquaredFFT[I,:,:] ) / self.images.shape[0]
                tempComplex2 = None
                
                self.mnxc2( I, I, shapeCropped, refine=True )
                #### Find maximum positions ####    
                self.locatePeak( I, I )
                if self.verbose: 
                    print( "Refine # " + str(I) + " shift: [%.2f"%self.__shiftsTriMat[I,I,0] 
                            + ", %.2f"%self.__shiftsTriMat[I,I,1]
                            + "], cc: %.6f"%self.__corrTriMat[I,I] 
                            + ", peak sig: %.3f"%self.__peaksigTriMat[I,I] )    
        else:
            # For even-odd and noise estimates, we often skip many rows
            rowIndices = np.unique( np.argwhere( np.sum( triIndices, axis=1 ) > 0 ) )
            #print( "rowIndices: " + str(rowIndices) )
            for I in rowIndices:
                # I is the index of the template image
                tempComplex = self.__baseImageFFT[I,:,:]
                self.__templateImageFFT = ne.evaluate( "conj(tempComplex)")
                    
                tempComplex2 =  self.__baseSquaredFFT[I,:,:]
                self.__templateSquaredFFT = ne.evaluate( "conj(tempComplex2)")
        
                if not self.masks.shape[0] == 1:
                    tempComplex = baseMaskFFT[I,:,:]
                    self.__templateMaskFFT = ne.evaluate( "conj(tempComplex)")
        
                # Now we can start looping through base images
                columnIndices = np.unique( np.argwhere( triIndices[I,:] ) )
                #print( "columnIndices: " + str(columnIndices) )
                for J in columnIndices:
                    
                    ####### MNXC2 revisement with private variable to make the code more manageable.
                    self.mnxc2( I, J, shapeCropped )
                    
                    #### Find maximum positions ####    
                    self.locatePeak( I, J )
                    
                    if self.verbose: 
                        print( "# " + str(I) + "->" + str(J) + " shift: [%.2f"%self.__shiftsTriMat[I,J,0] 
                            + ", %.2f"%self.__shiftsTriMat[I,J,1]
                            + "], cc: %.6f"%self.__corrTriMat[I,J] 
                            + ", peak sig: %.3f"%self.__peaksigTriMat[I,J] )    
                        
                    # Correlation stats is for establishing correlation scores for fixed-pattern noise.
                    if bool( self.trackCorrStats ):
                        # Track the various statistics about the correlation map, mean, std, max, skewness
                        if self.corrStats is None:
                            # Mean, std, max, maxposx, maxposy, (val at 0,0), imageI mean, imageI std, imageJ mean, imageJ std =  10 columns
                            K = np.sum(triIndices)
                            self.corrStats = {}
                            self.corrStats['K'] = K
                            self.corrStats['meanC'] = np.zeros([K])
                            self.corrStats['varC'] = np.zeros([K])
                            self.corrStats['maxC'] = np.zeros([K])
                            self.corrStats['maxPos'] = np.zeros([K,2])
                            self.corrStats['originC'] = np.zeros([K])
                            print( "Computing stack mean" )
                            self.corrStats['stackMean'] = np.mean( self.images )
                            print( "Computing stack variance" )
                            self.corrStats['stackVar'] = np.var( self.images )
                            
                        self.corrStats['meanC'][currIndex] = np.mean(self.__C_filt)
                        self.corrStats['varC'][currIndex] = np.var(self.__C_filt)
                        self.corrStats['maxC'][currIndex] = np.max(self.__C_filt)
                        self.corrStats['maxPos'][currIndex,:] = np.unravel_index( np.argmax(self.__C_filt), shapeCropped ) - np.array([self.__C_filt.shape[0]/2, self.__C_filt.shape[1]/2])
                        self.corrStats['originC'][currIndex] = self.__C_filt[self.__C.shape[0]/2, self.__C.shape[1]/2]   
                    
                    # triMode 'auto' diagonal mode    
                    if self.triMode == 'auto' and (self.__peaksigTriMat[I,J] <= self.peaksigThres or J-I >= self.autoMax):
                        if self.verbose: print( "triMode 'auto' stopping at frame: " + str(J) )
                        break
                    currIndex += 1
                pass # C max position location
        self.t[3] = time.time()
        

        
        if self.fouCrop is not None:
            self.__shiftsTriMat[:,:,0] *= self.shapePadded[0] / shapeCropped[0]
            self.__shiftsTriMat[:,:,1] *= self.shapePadded[1] / shapeCropped[1]
        
        # Pointer reference house-keeping
        del templateMaskFFT, normConst2, tempComplex, tempComplex2 # Pointer
        return

    """
    def mnxc2( self, I, J, shapeCropped, refine=False ):
        # 2-D Masked, Intensity Normalized, Cross-correlation

        tempComplex = self.__tempComplex # Pointer re-assignment
        tempComplex2 = self.__tempComplex2 # Pointer re-assignment
        maskProduct = self.__maskProduct
        normConst2 =  np.float32( 1.0 / ( np.float64(shapeCropped[0])*np.float64(shapeCropped[1]))**2.0 )
        
        if not self.masks.shape[0] == 1:
            # Compute maskProduct, term is M1^* .* M2
            baseMask_block = self.__baseMaskFFT[J,:,:]; templateMaskFFT = self.__templateMaskFFT # Pointer re-assignment
            tempComplex2 = ne.evaluate( "templateMaskFFT * baseMask_block" )
            self.__IFFT2.update_arrays( tempComplex2, tempComplex ); self.__IFFT2.execute()
            # maskProduct = np.clip( np.round( np.real( tempComplex ) ), eps, np.Inf )
            self.__maskProduct = ne.evaluate( "real(tempComplex)*normConst2" )
            
        # Compute mask correlation terms
        if self.masks.shape[0] == 1:
            templateImageFFT = self.__templateImageFFT; baseMask_block = self.__baseMaskFFT # Pointer re-assignment
        self.__IFFT2.update_arrays( ne.evaluate( "baseMask_block * templateImageFFT"), tempComplex ); self.__IFFT2.execute()
        

        Corr_templateMask = ne.evaluate( "real(tempComplex)*normConst2" ) # Normalization
        
        templateMaskFFT = self.__templateMaskFFT
        baseImageFFT_block = self.__baseImageFFT[J,:,:]
        
        self.__IFFT2.update_arrays( ne.evaluate( "templateMaskFFT * baseImageFFT_block"), tempComplex ); self.__IFFT2.execute()

        #ims( (np.abs(tempComplex),np.abs(templateMaskFFT), np.abs(baseImageFFT_block)), titles=("temp","templateMask","baseImage",) )
        # These haven't been normalized, so let's do so.  They are FFT squared, so N*N
        # This reduces the strain on single-precision range.
        Corr_baseMask =  ne.evaluate( "real(tempComplex)*normConst2" ) # Normalization

        # Compute the intensity normalzaiton for the template
        if self.masks.shape[0] == 1:
            baseMaskFFT = self.__baseMaskFFT; templateSquaredFFT = self.__templateSquaredFFT
            self.__IFFT2.update_arrays( ne.evaluate( "baseMaskFFT * templateSquaredFFT"), tempComplex ); self.__IFFT2.execute()
        else:
            self.__IFFT2.update_arrays( ne.evaluate( "baseMaskFFT_block * templateSquaredFFT"), tempComplex ); self.__IFFT2.execute()

        # DenomTemplate = ne.evaluate( "real(tempComplex)*normConst2 - real( Corr_templateMask * (Corr_templateMask / maskProduct) )" )
        
        # Compute the intensity normalzaiton for the base Image
        baseSquared_block = self.__baseSquaredFFT[J,:,:]
        self.__IFFT2.update_arrays( ne.evaluate( "templateMaskFFT * baseSquared_block"), tempComplex2 ); self.__IFFT2.execute()
        
        #ims( (np.abs(tempComplex2),np.abs(Corr_baseMask), np.abs(Corr_templateMask)), titles=("temp2","Corr_base","Corr_temp",) )
        
        # Compute Denominator intensity normalization
        # DenomBase = ne.evaluate( "real(tempComplex2)*normConst2- real( Corr_baseMask * (Corr_baseMask / maskProduct) )" )
        print( "DEBUG: something not evaluating correctly in Windows numexprz" )
        # Denom = ne.evaluate( "sqrt( (real(tempComplex2)*normConst2 - real( Corr_baseMask * (Corr_baseMask / maskProduct))) * (real(tempComplex)*normConst2 - real( Corr_templateMask * (Corr_templateMask / maskProduct)) ) )" )
        
        Term1_NE = ne.evaluate( "(real(tempComplex2)*normConst2) - real( Corr_baseMask * (Corr_baseMask / maskProduct))" )
        Term1 = np.real(tempComplex2)*normConst2 - np.real( Corr_baseMask * (Corr_baseMask / maskProduct)) 
        Term2_NE = ne.evaluate( "real(tempComplex)*normConst2 - real( Corr_templateMask * (Corr_templateMask / maskProduct))" ) 
        Term2 = np.real(tempComplex)*normConst2 - np.real( Corr_templateMask * (Corr_templateMask / maskProduct))
        
        # Ok so Term2 is very negative if we have the sum...
        # ims( (Term1, Term1_NE, Term2, Term2_NE), titles=("T1","T1_NE", "T2", "T2_NE",) )
        

        Denom = ne.evaluate( "sqrt(Term1_NE*Term2_NE)" )
        
        
        # What happened to numexpr clip?
        Denom = np.clip( Denom, 1, np.Inf )
        # print( "Number of small Denominator values: " + str(np.sum(DenomTemplate < 1.0)) )
        
        # Compute Numerator (the phase correlation)
        tempComplex2 = ne.evaluate( "baseImageFFT_block * templateImageFFT" )
        self.__IFFT2.update_arrays( tempComplex2, tempComplex ); self.__IFFT2.execute()
        # Numerator = ne.evaluate( "real(tempComplex)*normConst2 - real( Corr_templateMask * Corr_baseMask / maskProduct)" ) 
        
        # Compute final correlation
        self.__C = ne.evaluate( "(real(tempComplex)*normConst2 - real( Corr_templateMask * Corr_baseMask / maskProduct)) / Denom" )
        Denom_NP = np.sqrt(Term1 * Term2)
        

        
                print( "%%%%%%%%%%%%%%%%%%%" )
        print( "ImageRegistrator.images.dtype = " + str(self.) )
        print( "%%%%%%%%%%%%%%%%%%%" )
        if bool(self.suppressOrigin):
            # If gain reference is quite old we can still get some cross-artifacts.
            # TODO: better methodology?  Median filter over a small area?
            # Fit to the Fourier window?
            self.__C[0,0] = 0.25 * ( self.__C[1,1] + self.__C[-1,1] + self.__C[-1,1] + self.__C[-1,-1] )
            
        # We have everything in normal FFT order until here; Some speed-up could be found by its removal.
        # Pratically we don't have to do this fftshift, but it makes plotting easier to understand
        self.__C = np.fft.ifftshift( self.__C )
        
        ims( (Denom_NP, Denom, np.real(tempComplex), self.__C), titles=("Denom-NP","Denom-NE", "PXC", "C") )

        # We can crop C if maxShift is not None and preShift is False
        if self.maxShift is not None and self.preShift is False:
            self.__C = self.__C[shapeCropped[0]/2-self.maxShift:shapeCropped[0]/2+self.maxShift, shapeCropped[1]/2-self.maxShift:shapeCropped[1]/2+self.maxShift]

     
        del normConst2, baseMask_block, templateMaskFFT, templateImageFFT, Corr_templateMask, baseImageFFT_block
        del Corr_baseMask, baseSquared_block, baseMaskFFT, templateSquaredFFT, maskProduct
        del tempComplex, tempComplex2
    """
        
    def mnxc2( self, I, J, shapeCropped, refine=False ):
        """
        2-D Masked, Intensity Normalized, Cross-correlation
        """
        tempComplex = self.__tempComplex # Pointer re-assignment
        tempComplex2 = self.__tempComplex2 # Pointer re-assignment
        maskProduct = self.__maskProduct
        normConst2 = np.float32( 1.0 / (shapeCropped[0]*shapeCropped[1])**2 )
        
        if not self.masks.shape[0] == 1:
            # Compute maskProduct, term is M1^* .* M2
            baseMask_block = self.__baseMaskFFT[J,:,:]; templateMaskFFT = self.__templateMaskFFT # Pointer re-assignment
            tempComplex2 = ne.evaluate( "templateMaskFFT * baseMask_block" )
            self.__IFFT2.update_arrays( tempComplex2, tempComplex ); self.__IFFT2.execute()
            # maskProduct = np.clip( np.round( np.real( tempComplex ) ), eps, np.Inf )
            self.__maskProduct = ne.evaluate( "real(tempComplex)*normConst2" )
            
        # Compute mask correlation terms
        if self.masks.shape[0] == 1:
            templateImageFFT = self.__templateImageFFT; baseMask_block = self.__baseMaskFFT # Pointer re-assignment
        self.__IFFT2.update_arrays( ne.evaluate( "baseMask_block * templateImageFFT"), tempComplex ); self.__IFFT2.execute()
        

        Corr_templateMask = ne.evaluate( "real(tempComplex)*normConst2" ) # Normalization
        
        baseImageFFT_block = self.__baseImageFFT[J,:,:]; templateMaskFFT = self.__templateMaskFFT
        self.__IFFT2.update_arrays( ne.evaluate( "templateMaskFFT * baseImageFFT_block"), tempComplex ); self.__IFFT2.execute()

        # These haven't been normalized, so let's do so.  They are FFT squared, so N*N
        # This reduces the strain on single-precision range.
        Corr_baseMask =  ne.evaluate( "real(tempComplex)*normConst2" ) # Normalization

        # Compute the intensity normalzaiton for the template
        if self.masks.shape[0] == 1:
            baseMaskFFT = self.__baseMaskFFT; templateSquaredFFT = self.__templateSquaredFFT
            self.__IFFT2.update_arrays( ne.evaluate( "baseMaskFFT * templateSquaredFFT"), tempComplex ); self.__IFFT2.execute()
        else:
            self.__IFFT2.update_arrays( ne.evaluate( "baseMaskFFT_block * templateSquaredFFT"), tempComplex ); self.__IFFT2.execute()

        # DenomTemplate = ne.evaluate( "real(tempComplex)*normConst2 - real( Corr_templateMask * (Corr_templateMask / maskProduct) )" )
        
        # Compute the intensity normalzaiton for the base Image
        baseSquared_block = self.__baseSquaredFFT[J,:,:]
        self.__IFFT2.update_arrays( ne.evaluate( "templateMaskFFT * baseSquared_block"), tempComplex2 ); self.__IFFT2.execute()
        
        # Compute Denominator intensity normalization
        # DenomBase = ne.evaluate( "real(tempComplex2)*normConst2- real( Corr_baseMask * (Corr_baseMask / maskProduct) )" )
        Denom = ne.evaluate( "sqrt( (real(tempComplex2)*normConst2- real( Corr_baseMask * (Corr_baseMask / maskProduct)))" + 
            "* (real(tempComplex)*normConst2 - real( Corr_templateMask * (Corr_templateMask / maskProduct)) ) )" )
        # What happened to numexpr clip?
        Denom = np.clip( Denom, 1, np.Inf )
        # print( "Number of small Denominator values: " + str(np.sum(DenomTemplate < 1.0)) )
        
        # Compute Numerator (the phase correlation)
        tempComplex2 = ne.evaluate( "baseImageFFT_block * templateImageFFT" )
        self.__IFFT2.update_arrays( tempComplex2, tempComplex ); self.__IFFT2.execute()
        # Numerator = ne.evaluate( "real(tempComplex)*normConst2 - real( Corr_templateMask * Corr_baseMask / maskProduct)" ) 
        
        # Compute final correlation
        self.__C = ne.evaluate( "(real(tempComplex)*normConst2 - real( Corr_templateMask * Corr_baseMask / maskProduct)) / Denom" )
        
        # print( "%%%% mnxc2.Denom.dtype = " + str(Denom.dtype) )

        if bool(self.suppressOrigin):
            # If gain reference is quite old we can still get some cross-artifacts.
            # TODO: better methodology?  Median filter over a small area?
            # Fit to the Fourier window?
            self.__C[0,0] = 0.25 * ( self.__C[1,1] + self.__C[-1,1] + self.__C[-1,1] + self.__C[-1,-1] )
            
        # We have everything in normal FFT order until here; Some speed-up could be found by its removal.
        # Pratically we don't have to do this fftshift, but it makes plotting easier to understand
        self.__C = np.fft.ifftshift( self.__C )

        # We can crop C if maxShift is not None and preShift is False
        if self.maxShift is not None and self.preShift is False:
            self.__C = self.__C[shapeCropped[0]/2-self.maxShift:shapeCropped[0]/2+self.maxShift, shapeCropped[1]/2-self.maxShift:shapeCropped[1]/2+self.maxShift]

     
        del normConst2, baseMask_block, templateMaskFFT, templateImageFFT, Corr_templateMask, baseImageFFT_block
        del Corr_baseMask, baseSquared_block, baseMaskFFT, templateSquaredFFT, maskProduct
        del tempComplex, tempComplex2
    
    def locatePeak( self, I, J ):
        """
        Subpixel peak location by Fourier interpolation.
        """
        tempComplex = self.__tempComplex;  tempComplex2 = self.__tempComplex2
        # Apply B-factor low-pass filter to correlation function
        if self.Bmode == 'opti':
            self.t[20] = time.time()
            # Want to define this locally so it inherits scope.
            def inversePeakContrast( Bsigma ):
                self.__C_filt = scipy.ndimage.gaussian_filter( self.__C, Bsigma )
                return  np.std(self.__C_filt ) / (np.max(self.__C_filt ) - np.mean(self.__C_filt ) )
                    
            # B_opti= scipy.optimize.fminbound( inversePeakContrast, 0.0, 10.0, xtol=1E-3 )
            sigmaOptiMax = 7.0
            sigmaOptiMin = 0.0
            maxIter = 15 # Let's apply some more constraints to speed this up
            tolerance = 0.01
            result = scipy.optimize.minimize_scalar( inversePeakContrast, 
                                    bounds=[sigmaOptiMin,sigmaOptiMax], method="bounded", 
                                    tol=tolerance, options={'maxiter':maxIter}  )
                                    
            self.__C_filt = scipy.ndimage.gaussian_filter( self.__C, result.x )
            self.t[21] = time.time()
            if self.verbose >= 2:
                print( "Found optimum B-sigma: %.3f"%result.x + ", with peak sig: %.3f"%(1.0/result.fun)+" in %.2f"%(self.t[21]-self.t[20])+" s" ) 
        elif bool(self.Brad) and self.Bmode =='fourier':
            tempComplex = self.__C.astype(fftw_dtype)
            self.__FFT2.update_arrays( tempComplex, tempComplex2 ); self.__FFT2.execute()
            Bfilter = self.__Bfilter
            self.__IFFT2.update_arrays( ne.evaluate( "tempComplex2*Bfilter" ), tempComplex ); self.__IFFT2.execute()
            # Conservation of counts with Fourier filtering is not 
            # very straight-forward.
            C_filt = ne.evaluate( "real( tempComplex )/sqrt(normConst)" )
        elif bool(self.Brad) and self.Bmode == 'conv' or self.Bmode == 'convolution':
            # Convert self.Brad as an MTF to an equivalent sigma for a PSF
            # TODO: Check that Bsigma is correct with Fourier cropping"
            Bsigma = self.shapePadded / (np.sqrt(2) * np.pi * self.Brad)
            # Scipy's gaussian filter conserves total counts
            self.__C_filt = scipy.ndimage.gaussian_filter( self.__C, Bsigma )
        else: # No filtering
            self.__C_filt = self.__C
        
   
        # Apply maximum shift max mask, if present
        if bool( self.maxShift ):
            
            # for previous frame alignment compensation, we need to shift the mask around...
            C_filt = self.__C_filt
            if bool( self.preShift ):
                rolledMask = np.roll( np.roll( self.__mask_maxShift, 
                    np.round(self.__shiftsTriMat[I,J-1,0]).astype('int'), axis=0 ), 
                    np.round(self.__shiftsTriMat[I,J-1,1]).astype('int'), axis=1 )
                C_masked = ne.evaluate("C_filt*rolledMask")
                cmaxpos = np.unravel_index( np.argmax( C_masked ), C_masked.shape )
                self.__peaksigTriMat[I,J] = (C_masked[cmaxpos] - np.mean(C_filt[rolledMask]))/ np.std(C_filt[rolledMask])
            else:
                mask_maxShift = self.__mask_maxShift
                C_masked = ne.evaluate("C_filt*mask_maxShift")
                cmaxpos = np.unravel_index( np.argmax( C_masked ), C_filt.shape )
                self.__peaksigTriMat[I,J] = (C_masked[cmaxpos] - np.mean(C_filt[self.__mask_maxShift]))/ np.std(C_filt[self.__mask_maxShift])
        else: # No maxshift
            cmaxpos = np.unravel_index( np.argmax(C_filt), C_filt.shape )
            self.__peaksigTriMat[I,J] = (self.__corrTriMat[I,J] - np.mean(C_filt))/ np.std(C_filt)
        
        if self.saveC:
            # Maybe save in a pyTable if it's really needed.peaksig
            if self.preShift:
                self.C.append(self.__C_filt*rolledMask)
            else:
                self.C.append(self.__C_filt)
                
        if self.subPixReg > 1.0: # Subpixel peak estimation by Fourier interpolation

            Csub = C_filt[cmaxpos[0]-self.__subR:cmaxpos[0]+self.__subR, cmaxpos[1]-self.__subR:cmaxpos[1]+self.__subR ]
                
            # Csub is shape [2*subR, 2*subR]
            if Csub.shape[0] == 2*self.__subR and Csub.shape[1] == 2*self.__subR:
                self.__subFFT2.update_arrays( Csub.astype( fftw_dtype ), self.__CsubFFT ); self.__subFFT2.execute()
                # padding has to be done from the middle
                CpadFFT = np.pad( np.fft.fftshift(self.__CsubFFT), ((self.subPixReg-1)*self.__subR,), mode='constant', constant_values=(0.0,)  )
                CpadFFT = np.fft.ifftshift( CpadFFT )
                self.__subIFFT2.update_arrays( CpadFFT, self.__Csub_over ); self.__subIFFT2.execute()
                # Csub_overAbs = ne.evaluate( "abs( Csub_over )") # This is still complex
                Csub_overAbs = np.abs( self.__Csub_over )
                
                if self.saveCsub:
                    self.Csub.append(Csub_overAbs)
                
                Csub_maxpos = np.unravel_index( np.argmax( Csub_overAbs ), Csub_overAbs.shape )

                round_pos = cmaxpos - np.array(self.__C.shape)/2.0
                # Csub_max is being shifted 1 sub-pixel in the negative direction compared to the integer shift
                # because of array centering, hence the np.sign(round_pos)
                remainder_pos = Csub_maxpos - np.array(self.__Csub_over.shape)/2.0 + np.sign( round_pos )
                remainder_pos /= self.subPixReg
                
                # shiftsTriMat[I,J-1,:] = cmaxpos + np.array( Csub_maxpos, dtype='float' )/ np.float(self.subPixReg) - np.array( [subR, subR] ).astype('float')
                self.__shiftsTriMat[I,J,:] = round_pos + remainder_pos
                # Switching from FFTpack to pyFFTW has messed up the scaling of the correlation coefficients, so
                # scale by (subR*2.0)**2.0
                self.__corrTriMat[I,J] = Csub_overAbs[ Csub_maxpos[0], Csub_maxpos[1] ] / (self.__subR*2.0)**2.0
            else:
                print( "Correlation sub-area too close to maxShift!  Subpixel location broken." )
                self.__shiftsTriMat[I,J,:] = cmaxpos - np.array(self.__C.shape)/2.0
                self.__corrTriMat[I,J] = self.__C[ cmaxpos[0], cmaxpos[1] ]   
        else: # Do integer pixel registration
            self.__shiftsTriMat[I,J,:] = cmaxpos - np.array(self.__C.shape)/2.0
            self.__corrTriMat[I,J] = self.__C[ cmaxpos[0], cmaxpos[1] ] 
            
        del tempComplex, tempComplex2
        try: 
            del mask_maxShift, Bfilter 
        except: pass
        pass
                        
    def shiftsSolver( self, shiftsTriMat_in, corrTriMat_in, peaksigTriMat_in, acceptedEqns=None, mode='basin', Niter=100 ):
        """
        Functional minimization optimization of the triangular correlation matrix
        
        Minimizes the RMS error for the individual frame position equations, and 
        outputs an error dictionary.
        
            acceptedEqns is 'good' equations as determined by a previous run.  
                Should always be None for the first iteration.
            
            mode can be 'basin' for the global optimizer or 'local' for the local optimizer.
                In general the performance penalty for the global optimizer is trivial.
                
            Niter is the number of iterations for the 
        """
        # Change to allow the autocorrelations to be present, but we never want them in the solver
        shiftsTriMat = shiftsTriMat_in[:-1,1:,:]
        corrTriMat = corrTriMat_in[:-1,1:]
        peaksigTriMat = peaksigTriMat_in[:-1,1:]
        triIndices = corrTriMat.astype( 'bool' )
        
        # Build a dictionary of all the feedback parameters 
        errorDict = {}
        # Append the dictionary to the list of dicts and return it as well
        self.errorDictList.append( errorDict )
        errorDict['corrTriMat'] = corrTriMat_in
        errorDict['peaksigTriMat'] = peaksigTriMat_in

        
        self.t[4] = time.time()
        shapeImage = np.array( [self.images.shape[1], self.images.shape[2]] )
        N = np.asarray( self.images.shape )[0] - 1
        last_col = np.zeros( N )
            
        #### BUILD VECTORIZED SHIFTS b_x, b_y AND EQUATION COEFFICIENT MATRIX Acoeff
        M = 0
        for I in np.arange(0,N):
            # Find the last non-zero element in the tri-matrix for each row
            # This determines the sub-sampled view for each equation set.
            if triIndices[I,:].any():
                last_col[I] = np.argwhere(triIndices[I,:])[-1] + 1
                M += last_col[I] - I
        
        Acoeff = np.zeros( [M,N] )
        Arow_pos = 0
        for I in np.arange(0,N):
            rotview = np.rot90( triIndices[I:last_col[I],I:last_col[I]], k=2 )
            Acoeff[ Arow_pos:Arow_pos+rotview.shape[0], I:I+rotview.shape[1] ] = rotview
            Arow_pos += rotview.shape[0]
            
        #  triIndices = corrTriMat.astype( 'bool' )
        # Now we can ravel triIndices and get the indices from that
        vectorIndices = np.arange(0,triIndices.size)[np.ravel( triIndices )]
        # And this is to go backwards from a vector to an upper-triangular matrix
        unravelIndices = np.unravel_index( vectorIndices, [N,N] )        
        b_x = np.ravel( shiftsTriMat[triIndices,1] )
        b_y = np.ravel( shiftsTriMat[triIndices,0] )
        
        #### REMOVE UNACCEPTED EQUATIONS FROM THE SOLVER ####
        if acceptedEqns is None:
            Maccepted = M
            acceptedEqns = np.ones_like( b_x, dtype='bool' )
        else:
            Maccepted = np.sum( acceptedEqns )
        print( "Optimization of shifts over M = " + str(Maccepted) + " accepted equations." )
        
        #### WEIGHTS FOR OPTIMIZATION ####
        # There's only 2.5 % difference between the weighted and un-weighted versions for the STEM test cases.
        # CryoEM would be expected to be higher as the CC's are about 0.001 compared to 0.3
        if self.weightMode is None or self.weightMode == 'corr': # use raw correlation scores or peaksig
            weights =  np.ravel( peaksigTriMat[triIndices] )
        elif self.weightMode is 'unweighted': # don't weight peaks
            weights = np.ones_like( np.ravel( peaksigTriMat[triIndices] ) )
        elif self.weightMode == 'norm' or self.weightMode == 'normalized':
            ### Scale the weights so that lower correlations count for next-to-nothing
            weights = util.normalize( np.ravel( peaksigTriMat[triIndices] ) )
        elif self.weightMode == 'autologistic':
            # Calculate a logistic from the CDF of the peaksig values
            self.cdfLogisticCurve() # Sets peaksigThres, logisticK, and logisticNu
            
            peakSig = np.ravel( peaksigTriMat[triIndices] ).astype( 'float64' )
            weights = 1.0 - 1.0 / (1.0 + np.exp( -self.logisticK*(-peakSig + self.peaksigThres) ) )**self.logisticNu
        elif self.weightMode == 'logistic':
            # Use a fixed 
            peakSig = np.ravel( peaksigTriMat[triIndices] ).astype( 'float64' )
            weights = 1.0 - 1.0 / (1.0 + np.exp( -self.logisticK*(-peakSig + self.peaksigThres) ) )**self.logisticNu
        else:   
            print( "UNKNOWN WEIGHTING METHOD, REVERTING TO CORRELATION SCORES" )
            weights =  np.ravel( peaksigTriMat[triIndices] )
#            logisticCutoff = 0.01 # Value of logistic weight at the cutoff Correlation threshold. Should never, ever be below 0.5
#            C_cutoff = (1/self.weightK)* np.log( 1.0 / logisticCutoff - 1 )
#            if self.corrThres is None:
#                raise AssertionError("Zorro.shiftsSolver requires a correlation threshold for logistical weighting")
#            weights = 1.0 / ( 1.0 + self.weightK * np.exp(np.ravel( peaksigTriMat[triIndices] ) - self.corrThres - C_cutoff) ) 
            

        
        #### REMOVE UNACCEPTED EQUATIONS FROM THE SOLVER ####
        if acceptedEqns is None:
            Maccepted = M
            acceptedEqns = np.ones_like( b_x, dtype='bool' )
        else:
            Maccepted = np.sum( acceptedEqns )
        
        #### SETUP THE FUNCTIONAL OPTIMIZER ####
        pix_tol = 1.0E-5 # The fraction of a pixel we try to solve to (so one 10'000th of a pixel)
        relativeEst = np.zeros( [N, 2] )
        drift_guess = np.zeros( N )
        bounds = np.ones( [N,2] )
        bounds[:,0] = -1.0
        # Bounds scales by self.maxShift * number of frames
        if self.maxShift is None:
            bounds *= np.min( [shapeImage[0]/2.0, shapeImage[1]/2.0] )
        else:
            bounds *= np.min( [shapeImage[0]/2.0, shapeImage[1]/2.0, N*self.maxShift] )
            
        if mode == 'local':
            #### LOCAL MINIMIZATION X, Y SOLUTION ####
            # Is there any value for a simultaneous X-Y solution?  No, because the A-coefficient 
            # matrix would be:
            #     Acoeff2 = np.hstack( (np.vstack( (Acoeff, zeroA) ), np.vstack( (zeroA, Acoeff) )) )
            # So the two sets of equations are completely independent
            try:
                outX = scipy.optimize.minimize( util.weightedErrorNorm, drift_guess, method="L-BFGS-B", 
                    args=(Acoeff, b_x, weights*acceptedEqns), 
                    bounds=bounds, tol=pix_tol  )
#                outX = scipy.optimize.minimize( weightedErrorNorm, drift_guess, method="L-BFGS-B", 
#                    args=(Acoeff[acceptedEqns,:], b_x[acceptedEqns], weights[acceptedEqns]), 
#                    bounds=bounds, tol=pix_tol  )
                relativeEst[:,1] = outX.x
            except:
                raise RuntimeError( "Error: caught exception on X-minimizer" )

            try:   
                outY = scipy.optimize.minimize( util.weightedErrorNorm, drift_guess, method="L-BFGS-B", 
                    args=(Acoeff, b_y, weights*acceptedEqns), 
                    bounds=bounds, tol=pix_tol  )
                relativeEst[:,0] = outY.x
            except:
                raise RuntimeError( "Error: caught exception on Y-minimizer" )
                
        elif mode == 'basin':
            #### GLOBAL MINIMIZATION X, Y SOLUTION ####
            basinArgs = {}
            basinArgs["bounds"] = bounds
            basinArgs["tol"] = pix_tol
            basinArgs["method"] =  "L-BFGS-B"
            basinArgs["args"] = (Acoeff, b_x, weights*acceptedEqns)
            try:
                outX = scipy.optimize.basinhopping( util.weightedErrorNorm, drift_guess, niter=Niter, minimizer_kwargs=basinArgs )
                relativeEst[:,1] = outX.x
            except:
                raise RuntimeError( "Error: caught exception on X-minimizer" )   
            # basinArgs["args"] = (Acoeff[acceptedEqns], b_y[acceptedEqns], weights[acceptedEqns])
            basinArgs["args"] = (Acoeff, b_y, weights*acceptedEqns)
            try:
                outY = scipy.optimize.basinhopping( util.weightedErrorNorm, drift_guess, niter=Niter, minimizer_kwargs=basinArgs )
                relativeEst[:,0] = outY.x
            except:
                raise RuntimeError( "Error: caught exception on Y-minimizer" )   
                
        else:
            print( "Error: mode not understood by shiftsMinimizer: " + mode )
            return
        
        #### ERROR ANALYSIS (for precision of estimated shifts) ####
        acceptedEqnsUnraveled = np.zeros( [N,N] )
        acceptedEqnsUnraveled[unravelIndices[0], unravelIndices[1]] = acceptedEqns
        acceptedEqnsUnraveled = np.pad( acceptedEqnsUnraveled, ((0,1),(1,0)), mode='constant' )
        
        # Ok so how big is relativeEst?  Can we add in zeros?
        # Or maybe I should just give weights as weights*acceptedEqnsUnr
        errorXY = np.zeros( [M,2] )
        errorXY[:,1] = np.dot( Acoeff, relativeEst[:,1] ) - b_x
        errorXY[:,0] = np.dot( Acoeff, relativeEst[:,0] ) - b_y
        errorNorm = np.sqrt( errorXY[:,0]*errorXY[:,0] + errorXY[:,1]*errorXY[:,1] )

        mean_errorNorm = np.mean( errorNorm[acceptedEqns] )
        std_errorNorm = np.std( errorNorm[acceptedEqns] )
    
        # Error unraveled (i.e. back to the upper triangular matrix form)
        errorUnraveled = np.zeros( [N,N] )
        errorXun = np.zeros( [N,N] )
        errorYun = np.zeros( [N,N] )
        errorUnraveled[unravelIndices[0], unravelIndices[1]] = errorNorm
        errorXun[unravelIndices[0], unravelIndices[1]] = np.abs( errorXY[:,1] )
        errorYun[unravelIndices[0], unravelIndices[1]] = np.abs( errorXY[:,0] )
        
        errorXun = np.pad( errorXun, ((0,1),(1,0)), mode='constant' )
        errorYun = np.pad( errorYun, ((0,1),(1,0)), mode='constant' )
        triPadded = np.pad( triIndices, ((0,1),(1,0)), mode='constant' )
        
        # Mask out un-used equations from error numbers
        errorYun = errorYun * acceptedEqnsUnraveled
        errorXun = errorXun * acceptedEqnsUnraveled
        triPadded = triPadded * acceptedEqnsUnraveled    

        # errorX and Y are per-frame error estimates
        errorX = np.zeros( N+1 )
        errorY = np.zeros( N+1 )
        # Sum horizontally and vertically, keeping in mind diagonal is actually at x-1
        for J in np.arange(0,N+1):
            # Here I often get run-time warnings, which suggests a divide-by-zero or similar.
            try:
                errorX[J] = ( np.sum( errorXun[J,:]) + np.sum(errorXun[:,J-1]) ) / ( np.sum( triPadded[J,:]) + np.sum(triPadded[:,J-1]) )
            except:
                pass
            try:
                errorY[J] = ( np.sum( errorYun[J,:]) + np.sum(errorYun[:,J-1]) ) / ( np.sum( triPadded[J,:]) + np.sum(triPadded[:,J-1]) )
            except:
                pass
        self.t[5] = time.time() 
        

        
        # translations (to apply) are the negative of the found shifts
        errorDict['translations'] = -np.vstack( (np.zeros([1,2]), np.cumsum( relativeEst, axis=0 ) ) )
        errorDict['relativeEst'] = relativeEst
        errorDict['acceptedEqns'] = acceptedEqns
        # Not necessary to save triIndices, it's the non-zero elements of corrTriMat
        # errorDict['triIndices'] = triIndices
        errorDict['errorXY'] = errorXY
        errorDict['shiftsTriMat'] = shiftsTriMat_in
        errorDict['errorX'] = errorX 
        errorDict['errorY'] = errorY 
        errorDict['errorUnraveled'] = errorUnraveled
        errorDict['mean_errorNorm'] = mean_errorNorm
        errorDict['std_errorNorm'] = std_errorNorm 
        errorDict['M'] = M
        errorDict['Maccepted'] = Maccepted
        
        
        return errorDict
        
    def alignImageStack( self ):
        """
        alignImageStack does a masked cross-correlation on a set of images.  
        masks can be a single mask, in which case it is re-used for each image, or 
        individual for each corresponding image.  
        
        Subpixel shifting is usually done with a large, shifted Lanczos resampling kernel. 
        This was found to be faster than with a phase gradient in Fourier space.
        """
        
        # Setup
        if self.n_threads is None:
            self.n_threads = ne.detect_number_of_cores()
        # pyFFTW is set elsewhere
        ne.set_num_threads( self.n_threads )
        print( "NumExpr using: " + str(self.n_threads) + " threads" )

        #Baseline un-aligned stack, useful for see gain reference problems
        # self.unalignedSum = np.sum( self.images, axis=0 )
        if np.any( self.shapeBinned ):
            self.binStack()
            
        # Do CTF measurement first, in-case we want to use the result for Wiener filtering afterward
        # Plus CTFFIND4 crashes a lot, so we save processing if it can't fit the CTF
        if bool( self.doCTF ):
            self.execCTFFind4( movieMode=True )
            
        """
        Application of binning and padding.
        """
        if np.any(self.shapePadded):
            self.padStack()
            
        """
        Registration, first run: Call xcorrnm2_tri to do the heavy lifting
        """
        self.t[8] = time.time() 
        self.xcorrnm2_tri()
        self.t[9] = time.time() 
        
        """
        Functional minimization over system of equations
        """
        self.t[11] = time.time()
        if self.triMode == 'first':
            self.translations = -self.__shiftsTriMat[0,:]
            self.errorDictList.append({})
            self.errorDictList[-1]['shiftsTriMat'] = self.__shiftsTriMat
            self.errorDictList[-1]['corrTriMat'] = self.__corrTriMat
            self.errorDictList[-1]['peaksigTriMat'] = self.__peaksigTriMat
            self.errorDictList[-1]['translations'] = self.translations.copy()
        elif self.triMode == 'refine':
            self.errorDictList.append({})
            self.errorDictList[-1]['shiftsTriMat'] = self.__shiftsTriMat
            self.errorDictList[-1]['corrTriMat'] = self.__corrTriMat
            self.errorDictList[-1]['peaksigTriMat'] = self.__peaksigTriMat
            
            m = self.images.shape[0]
            self.translations = np.zeros( [m,2], dtype='float32' )
            print( "Refine shifts:" )
            print( self.__shiftsTriMat[:,:,0] )
            print( "========================" )
            print( self.__shiftsTriMat[:,:,1] )
            for K in np.arange(m): 
                self.translations[K,:] = -self.__shiftsTriMat[K,K,:]
            self.errorDictList[-1]['translations'] = self.translations.copy()
            
        else:
            # Every round of shiftsSolver makes an error dictionary
            self.shiftsSolver( self.__shiftsTriMat, self.__corrTriMat, self.__peaksigTriMat )
            self.translations = self.errorDictList[-1]['translations'].copy( order='C' )
            
        """
        Alignment and projection through Z-axis (averaging)
        """
        if np.any(self.shapePadded): # CROP back to original size
            self.cropStack()
        self.applyShifts()
        

        
        
        self.t[10] = time.time() 
        if bool(self.doLazyFRC):
            self.lazyFouRingCorr()
            
        self.t[30] = time.time()
        
        if bool(self.doDoseFilter):
            print( "Generating dose-filtered sum" )
            self.doseFilter()
        
        self.cleanPrivateVariables()
        
        pass # End of alignImageStack
        
    def cleanPrivateVariables(self):
        """
        Remove all private ("__") variables so the memory they occupy is released.
        """
        print( "TODO: finish cleanPrivateVariables" )
        try: del self.__FFT2, self.__IFFT2
        except: pass
        try:  del self.__subFFT2, self.__subIFFT2
        except: pass
        try: del self.__imageFFT
        except: pass
        try: del self.__Bfilter
        except: pass
    
        del self.__baseImageFFT, self.__baseMaskFFT, self.__baseSquaredFFT, self.__C, 
        
    def applyShifts( self ):
        # Apply centroid origin, or origin at frame #0 position?
        if self.originMode == 'centroid':
            centroid = np.mean( self.translations, axis=0 )
            self.translations -= centroid
        # if self.originMode == None do nothing
        
        shifts_round = np.round( self.translations ).astype('int')
        shifts_remainder = self.translations - shifts_round
        
        # Use RAMutil.imageShiftAndCrop to do a non-circular shift of the images to 
        # integer pixel shifts, then subpixel with Lanczos
        m = self.images.shape[0] # image count
        if self.subPixReg > 1.0 and self.shiftMethod == 'fourier':

            # Setup FFTs for shifting.
            FFTImage = pyfftw.n_byte_align_empty( self.shapePadded, fftw_dtype.itemsize, dtype=fftw_dtype )
            RealImage = pyfftw.n_byte_align_empty( self.shapePadded, fftw_dtype.itemsize, dtype=fftw_dtype )
            normConst = 1.0 / (self.shapePadded[0]*self.shapePadded[1])
            # Make pyFFTW objects
            _, IFFT2 = util.pyFFTWPlanner( FFTImage, wisdomFile=os.path.join( self.cachePath, "fftw_wisdom.pkl" ), effort = self.fftw_effort, n_threads=self.n_threads, doForward=False )
            [xmesh, ymesh] = np.meshgrid( np.arange(-RealImage.shape[1]/2,RealImage.shape[1]/2) / np.float(RealImage.shape[1] ), 
                np.arange(-RealImage.shape[0]/2,RealImage.shape[0]/2)/np.float(RealImage.shape[0]) )
            twoj_pi = np.complex64( -2.0j * np.pi )
        
        for J in np.arange(0,m):
            if self.subPixReg > 1.0 and self.shiftMethod == 'lanczos':
                # Lanczos realspace shifting
                # self.images[J,:,:] = util.imageShiftAndCrop( self.images[J,:,:], shifts_round[J,:] )
                #Roll the image instead to preserve information in the stack, in case someone deletes the original
                self.images[J,:,:] = np.roll( np.roll( self.images[J,:,:], shifts_round[J,0], axis=0 ), shifts_round[J,1], axis=1 )
                
                self.images[J,:,:] = util.lanczosSubPixShift( self.images[J,:,:], subPixShift=shifts_remainder[J,:], kernelShape=5, lobes=3 )
                
                if self.verbose: print( "Correction (lanczos) "+ str(np.around(self.translations[J,:],decimals=4))+" applied to image: " + str(J) )
            elif self.subPixReg > 1.0 and self.shiftMethod == 'fourier':
                # Fourier gradient subpixel shift
                
#                RealImage = self.images[J,:,:].astype( fftw_dtype )
#                FFT2.update_arrays( RealImage, FFTImage ); FFT2.execute()
#                FFTImage *= np.fft.fftshift( np.exp( -2.0j * np.pi * (xmesh*self.translations[J,1] + ymesh*self.translations[J,0]) )  )
#                IFFT2.update_arrays( FFTImage, RealImage ); IFFT2.execute()
#                # Normalize and reduce to float32
#                self.images[J,:,:] = np.real( RealImage ).astype(self.images.dtype) / RealImage.size
                tX = self.translations[J,1]; tY = ymesh*self.translations[J,0]
                FFTImage = self.__imageFFT[J,:,:] * np.fft.fftshift( ne.evaluate( "exp(twoj_pi * (xmesh*tX + ymesh*tY))")  )
#                FFTImage = self.__imageFFT[J,:,:] * np.fft.fftshift( np.exp( twoj_pi * (xmesh*self.translations[J,1] + 
#                                                                                            ymesh*self.translations[J,0]) )  )
                             
                IFFT2.update_arrays( FFTImage, RealImage ); IFFT2.execute()
                # Normalize and reduce to float32
                if self.images.shape[1] < RealImage.shape[0] or self.images.shape[2] < RealImage.shape[1]:
                    self.images[J,:,:] = np.real( ne.evaluate( "normConst * real(RealImage)" ) ).astype(self.images.dtype)[:self.images.shape[1],:self.images.shape[2]]
                else:
                    self.images[J,:,:] = np.real( ne.evaluate( "normConst * real(RealImage)" ) ).astype(self.images.dtype)
                
                if self.verbose: print( "Correction (fourier) "+ str(np.around(self.translations[J,:],decimals=4))+" applied to image: " + str(J) )
            else:
                # self.images[J,:,:] = util.imageShiftAndCrop( self.images[J,:,:], shifts_round[J,:] )
                #Roll the image instead to preserve information in the stack, in case someone deletes the original
                self.images[J,:,:] = np.roll( np.roll( self.images[J,:,:], shifts_round[J,0], axis=0 ), shifts_round[J,1], axis=1 )

                if self.verbose: print( "Correction (integer) "+ str(shifts_round[J,:])+" applied to image: " + str(J) )
                
            # Also do masks (single-pixel precision only) if seperate for each image
            if not self.masks is None and self.masks.shape[0] > 1:
                self.masks[J,:,:] = util.imageShiftAndCrop( self.masks[J,:,:], shifts_round[J,:] )
        # Build sum
        self.imageSum = np.sum( self.images, axis=0 )
        # Clean up numexpr pointers
        try: del normConst, tX, tY, twoj_pi
        except: pass
    
    def binStack( self ):
        # This is for the moment highly inefficient. But then again super-resolution mode rarely makes sense
        # over just using a higher magnification for the MTF of the Gatan K-2
        bShape2 = np.array( self.shapeBinned ) / 2
        scaleFactor = np.float32(self.images.shape[1]) / np.float32(self.shapeBinned[0])
        self.pixelsize *= scaleFactor
        
        tempFull = np.zeros( [ self.images.shape[1], self.images.shape[2] ], dtype=fftw_dtype)
        tempBin = np.zeros( self.shapeBinned, dtype=fftw_dtype )
        tempImage = np.zeros( self.shapeBinned, dtype=fftw_dtype )
        FFT2, _ = util.pyFFTWPlanner( tempFull, tempFull, 
                        wisdomFile=os.path.join( self.cachePath, "fftw_wisdom.pkl" ), 
                        effort = self.fftw_effort, n_threads=self.n_threads, doReverse=False )
        _, IFFT2bin = util.pyFFTWPlanner( tempBin, tempBin, 
                        wisdomFile=os.path.join( self.cachePath, "fftw_wisdom.pkl" ), 
                        effort = self.fftw_effort, n_threads=self.n_threads, doForward=False )   
                        
        self.images = self.images.tolist()
        normConst = 1.0 / self.shapeBinned[0]*self.shapeBinned[1]
        for J in np.arange( len( self.images ) ):
            self.images[J] = np.array( self.images[J] ).astype(fftw_dtype)

            FFT2.update_arrays( self.images[J], tempFull ); FFT2.execute()
            # Crop
            tempBin[0:bShape2[0],0:bShape2[1]] = tempFull[0:bShape2[0],0:bShape2[1]]
            tempBin[0:bShape2[0],-bShape2[1]:] = tempFull[0:bShape2[0],-bShape2[1]:] 
            tempBin[-bShape2[0]:,0:bShape2[1]] = tempFull[-bShape2[0]:,0:bShape2[1]] 
            tempBin[-bShape2[0]:,-bShape2[1]:] = tempFull[-bShape2[0]:,-bShape2[1]:]
            tempBin *= normConst
            IFFT2bin.update_arrays( tempBin, tempImage ); IFFT2bin.execute()
            self.images[J] = np.real(tempImage).astype( float_dtype ) 
        pass
        self.images = np.asarray( self.images )
        self.shapeOriginal = self.shapeBinned 

        
    def padStack( self, padSize=None ):
        """
        This function is used to zero-pad both the images and masks.  This breaks
        the circular shift issues.
        
        Defaults to self.shapePadded
        
        It can also improve performance as FFTW is fastest for dimensions that are powers of 2, 
        and still fast for powers of 2,3, and 5.  Wierd dimensions then should be padded 
        to an optimized size, which the helper function FindValidFFTDim can provide good
        guesses for.
        
        In general try to have 20 % of your total number of pixels within the mask to reduce
        floating-point round-off error in the masked cross-correlation.
        """
        # Take the stack and zero-pad it 
        # Unfortunately this step is memory intensive as we need to make a new array
        # to copy the values of the old one into.
        
        if padSize is None:
            padSize = self.shapePadded 
            
        if not np.any(padSize):
            print( "Cannot pad to: " + str(padSize) )
            return
        
            
        m = self.images.shape[0]
        self.shapeOriginal = [ self.images.shape[1], self.images.shape[2] ]
        self.shapePadded = padSize # This needs to be recorded for undoing the padding operation
        
        print( "Padding images and masks to shape: " + str(padSize) )
        paddedImages = np.zeros( [m, padSize[0], padSize[1]], dtype=self.images.dtype )
        paddedImages[:,:self.shapeOriginal[0],:self.shapeOriginal[1]] = self.images
        self.images = paddedImages
        # Then make or pad the mask appropriately.
        if self.masks is None:
            self.masks = np.zeros( [1,padSize[0],padSize[1]], dtype='bool', order='C' )
            self.masks[0,:self.shapeOriginal[0],:self.shapeOriginal[1]] = 1.0
        else:
            if self.masks.shape[1] != self.shapePadded[0] and self.masks.shape[2] != self.shapePadded[1]:
                mmask = self.masks.shape[0]
                paddedMasks = np.zeros( [mmask, padSize[0], padSize[1]], dtype=self.masks.dtype )
                paddedMasks[:,:self.shapeOriginal[0],:self.shapeOriginal[1]] = self.masks
                self.masks = paddedMasks
            pass # else do nothing
        pass
    
    def cropStack( self, cropSize=None ):
        """
        Undos the operation from ImageRegistrator.padStack()
        
        Defaults to self.shapeOriginal.
        """
        if cropSize is None:
            cropSize = self.shapeOriginal
            
        if not bool(cropSize):
            print( "Cannot crop to: " + str(cropSize) )
            return
        
        print( "Cropping auto-applied mask pixels back to shape: " + str(self.shapeOriginal) )
        self.images = self.images[ :, :cropSize[0], :cropSize[1] ]
        # Crop masks too
        self.masks = self.masks[ :, :cropSize[0], :cropSize[1] ]
        # And sum if present
        if self.imageSum is not None:
            self.imageSum = self.imageSum[ :cropSize[0], :cropSize[1] ]
            
    def cdfLogisticCurve( self, errIndex = -1, bins = None ):
        """
        Calculate the cumulative distribution function of the peak significance scores, and fit a logistic 
        curve to them, for deriving a weighting function.
        """
        # The error dict list doesn't normally exist here.
        
        peaksigTriMat = self.errorDictList[errIndex]['peaksigTriMat']
        peaksigs = peaksigTriMat[ peaksigTriMat > 0.0 ]
        if bins == None:
            bins = np.int( peaksigs.size/7.0 )
            
        [pdfPeaks, hSigma ] = np.histogram( peaksigs, bins=bins )
        hSigma = hSigma[:-1]
        pdfPeaks = pdfPeaks.astype( 'float32' )
        cdfPeaks = np.cumsum( pdfPeaks )
        cdfPeaks /= cdfPeaks[-1]
    
        # BASIN-HOPPING
        basinArgs = {}
        bounds = ( (np.min(peaksigs), np.max(peaksigs)), (0.1, 20.0), (0.05, 5.0) )
        basinArgs["bounds"] = bounds
        basinArgs["tol"] = 1E-6
        basinArgs["method"] =  "L-BFGS-B"
        basinArgs["args"] = ( hSigma, cdfPeaks )
        # x is [SigmaThres, K, Nu, background]
        x0 = [np.mean(peaksigs), 5.0, 1.0]
        outBasin = scipy.optimize.basinhopping( util.minLogistic, x0, niter=50, minimizer_kwargs=basinArgs )
        
        # Set the logistics curve appropriately.
        self.peaksigThres = outBasin.x[0]
        self.logisticK = outBasin.x[1]
        self.logisticNu = outBasin.x[2]
        
        # Diagnostics (for plotting)
        self.errorDictList[errIndex]['pdfPeaks'] = pdfPeaks
        self.errorDictList[errIndex]['cdfPeaks'] = cdfPeaks
        self.errorDictList[errIndex]['hSigma'] = hSigma
        self.errorDictList[errIndex]['logisticNu'] = self.logisticNu
        self.errorDictList[errIndex]['logisticK'] = self.logisticK
        self.errorDictList[errIndex]['peaksigThres'] = self.peaksigThres
        pass
        
    def velocityCull( self, velocityThres=None ):
        """
        Computes the pixel velocities, using a 5-point numerical differentiation on the 
        translations.  Note that a 5-point formula inherently has some low-pass filtering
        built-in.
        
        TODO: this would be better of using a spline interpolation (def smoothTrajectory() ) to 
        estimate the local velocity than numerical differentiation.
        
        if velocityThres == None, self.velocityThres is used.
        if velocityThres < 0.0, no thresholding is applied (i.e. good for just 
        computing the velocity to produce plots)
        """

        velo_diff2 = np.diff( self.translations, axis=0 )
        speed_diff2 = np.sqrt( np.sum( velo_diff2**2.0, axis=1 ))
        self.velocities = np.zeros( [self.translations.shape[0]] )
        self.velocities[0] = speed_diff2[0]
        self.velocities[1:-1] = 0.5*(speed_diff2[:-1] + speed_diff2[1:])
        self.velocities[-1] = speed_diff2[-1]
        
        # Establish what velocities we should crop?
        plt.figure()
        plt.plot( np.arange(0,self.velocities.shape[0]), self.velocities, 'o-k' )
        plt.xlabel( 'Frame number, m' )
        plt.ylabel( 'Pixel velocity, v (pix/frame)' )
        
        # TODO: this is fairly useless due to noise, properly minimum-acceleration splines fits would work 
        # much better I suspect
        print( "Velocity culling still under development, useful only for diagnostics at present." )
        pass
    
    def smoothTrajectory( self, dampen = 0.5 ):
        """
        Fit a dampened spline to the translations. This seems to be most useful for refinement as it has been 
        shown in UnBlur to help break correlated noise systems.  It reminds me a bit of simulated annealing 
        but the jumps aren't random.
        
        dampen should be std of position estimates, so about 0.25 - 1.0 pixels.  If generating smoothing for 
        velocity estimation use a higher dampening factor.  
        """
        
        if np.any( self.translations ) == None:
            print( "smoothTrajectory requires an estimate for translations" )
            return
        import scipy.interpolate
        
        
        frames = np.arange( self.translations.shape[0] )
        ySplineObj = scipy.interpolate.UnivariateSpline( frames, self.translations[:,0], k=5, s=dampen )
        xSplineObj = scipy.interpolate.UnivariateSpline( frames, self.translations[:,1], k=5, s=dampen )
        
        smoothedTrans = np.zeros( self.translations.shape )
        smoothedTrans[:,0] = ySplineObj(frames); smoothedTrans[:,1] = xSplineObj(frames)
        return smoothedTrans
    
    def calcIncoherentFourierMag( self ):
        """
        Compute the Fourier transform of each frame in the movie and average
        the Fourier-space magnitudes.  This gives a baseline to compare how 
        well the alignment did vesus the  spatial information content of the 
        individual images.
        
        This is the square root of the power spectrum.  
        """
        itemsize = 8
        frameFFT = pyfftw.n_byte_align_empty( self.images.shape[1:], itemsize, dtype=fftw_dtype )
        self.incohFouMag = np.zeros( self.images.shape[1:], dtype=float_dtype )
        FFT2, _ = util.pyFFTWPlanner( frameFFT, wisdomFile=os.path.join( self.cachePath, "fftw_wisdom.pkl" ), n_threads = self.n_threads, doReverse=False )
        
        for J in np.arange(0,self.images.shape[0]):
            FFT2.update_arrays( np.squeeze( self.images[J,:,:]).astype(fftw_dtype), frameFFT ); FFT2.execute()
            self.incohFouMag += np.abs( frameFFT )
        pass
        self.incohFouMag = np.fft.fftshift( self.incohFouMag / self.images.shape[0] )
        
    def evenOddFouRingCorr( self, xcorr = 'tri', box=[512,512], overlap=0.5, debug=False ):
        """ 
        Seperates the frames into even and odd frames and tries to calculate a 
        Fourier Ring Correlation (FRC) from the two sets.  Oscillations in the 
        FRC are normal for this application because of the objective contrast 
        transfer function. Note: this function is not well-optimized. It reloads 
        the data from disk several times to conserve memory.
        THIS FUNCTION DESTROYS THE DATA IN THE OBJECT.
        
            xcorr = 'tri'  uses the zorro approach.
            xcorr = 'mc' tries to use dosefgpu_driftcorr (Motioncorr)
            xcorr = 'unblur' uses UnBlur
            
            box is the shape of the moving window, and limits the maximum 
            resolution the FRC is calculated to.
        If you plan to run both, use 'mc' first.  
        """
        
        m = self.images.shape[0]
        evenIndices = np.arange(0, m, 2)
        oddIndices = np.arange(1, m, 2)
        
        import uuid
        logName = str(uuid.uuid4() ) + ".log"
        self.saveConfig( logName ) 
        
        print( evenIndices )
        evenReg = ImageRegistrator()
        evenReg.loadConfig( logName )
        evenReg.images = self.images[evenIndices,:,:].copy(order='C')
        
        print( oddIndices )
        oddReg = ImageRegistrator()
        oddReg.loadConfig( logName )
        oddReg.images = self.images[oddIndices,:,:].copy(order='C')
        
        if xcorr == 'tri' or xcorr is None:
            if self.masks is None:
                evenReg.masks = util.edge_mask( maskShape=[ self.images.shape[1], self.images.shape[2] ] )
                oddReg.masks = evenReg.masks
            elif self.masks.shape[0] > 1:
                evenReg.masks = self.masks[evenIndices,:,:]
                oddReg.masks = self.masks[oddIndices,:,:]
            elif self.masks.shape[0] == 1:
                evenReg.masks = self.masks
                oddReg.masks = self.mask
            
            print( "#####  Zorro even frames alignment  #####" )
            evenReg.alignImageStack()
            self.transEven = evenReg.translations.copy( order='C' )

            print( "#####  Zorro odd frames alignment  #####" )
            oddReg.alignImageStack()
            self.transOdd = oddReg.translations.copy( order='C' )
            
        
            
        elif xcorr == 'mc':
            print( "#####  Motioncorr even frames alignment  #####" )
            evenReg.xcorr2_mc( loadResult = False )
            evenReg.applyShifts()
            self.transEven = evenReg.translations.copy( order='C' )

            print( "#####  Motioncorr odd frames alignment  #####" )
            oddReg.xcorr2_mc( loadResult = False )
            oddReg.applyShifts()
            self.transOdd = oddReg.translations.copy( order='C' )
            
        elif xcorr == 'unblur':
            print( "#####  UnBlur even frames alignment  #####" )
            evenReg.xcorr2_unblur( loadResult=False )
            evenReg.applyShifts()
            self.transEven = evenReg.translations.copy( order='C' )

            print( "#####  UnBlur odd frames alignment  #####" )
            oddReg.xcorr2_unblur( loadResult=False )
            oddReg.applyShifts()
            self.transOdd = oddReg.translations.copy( order='C' )
            
        else:
            print( "Unknown xcorr method for even-odd FRC: " + str(xcorr) )
            

        print( "#####  Computing even-odd Fourier ring correlation  #####" )
        eoReg = ImageRegistrator()
        eoReg.loadConfig( logName )
        eoReg.images = np.empty( [2, evenReg.imageSum.shape[0], evenReg.imageSum.shape[1] ], dtype=float_dtype)
        eoReg.images[0,:,:] = evenReg.imageSum; eoReg.images[1,:,:] = oddReg.imageSum
        eoReg.triMode = 'first'
        
        try: os.remove( logName )
        except: print( "Could not remove temporary log file: " + logName )

        # This actually aligns the two phase images
        # We use Zorro for this for all methods because we have more trust in the masked, normalized
        # cross correlation
        eoReg.alignImageStack()
        
        eoReg.tiledFRC( eoReg.images[0,:,:], eoReg.images[1,:,:], 
                       trans=np.hstack( [self.transEven, self.transOdd] ), box=box, overlap=overlap )
        
        self.FRC2D = eoReg.FRC2D
        self.FRC = eoReg.FRC
        if self.saveC:
            self.evenC = evenReg.C
            self.oddC = oddReg.C
        if self.saveCsub:
            self.evenCsub = evenReg.Csub
            self.oddCsub = oddReg.Csub
        return evenReg, oddReg
        
    def lazyFouRingCorr( self, box=[512,512], overlap=0.5, debug=False ):
        # Computes the FRC from the full stack, taking even and odd frames for the half-sums
        # These are not independant half-sets! ... but it still gives us a very good impression 
        # of alignment success or failure.
        m = self.images.shape[0]
        evenIndices = np.arange(0, m, 2)
        oddIndices = np.arange(1, m, 2)                     
        
        evenSum = np.sum( self.images[evenIndices,:,:], axis=0 )
        oddSum = np.sum( self.images[oddIndices,:,:], axis=0 )
        
        self.tiledFRC( evenSum, oddSum, box=box, overlap=overlap )
        # Force the length to be box/2 because the corners are poorly sampled
        self.FRC = self.FRC[:box[0]/2]
        pass
    
    def tiledFRC( self, Image1, Image2, trans=None, box=[512,512], overlap=0.5 ):
        """
        Pass in two images, which are ideally averages from two independantly processed half-sets. 
        Compute the FRC in many tiles of shape 'box', and average the FRC over all tiles.
        
        Overlap controls how much tiles overlap by, with 0.5 being half-tiles and 0.0 being no overlap,
        i.e. they are directly adjacent.  Negative overlaps may be used for sparser samping.  
        
        Produced both a 2D FRC, which is generally of better quality than a power-spectrum, and 
        """
        FFT2, _ = util.pyFFTWPlanner( np.zeros(box, dtype=fftw_dtype), 
                             wisdomFile=os.path.join( self.cachePath, "fftw_wisdom.pkl" ) , n_threads = self.n_threads, 
                             effort=self.fftw_effort, doReverse=False )
        if overlap > 0.8:
            print("tiledFRC takes huge amounts of time as overlap->1.0" )
            overlap = 0.8                
            
        if trans is None:
            trans = self.translations
            
        minCrop = 5
        if not np.any(trans):
            cropLim = np.array( [minCrop,minCrop,minCrop,minCrop] ) # Keep away from any edge artifacts
        else:
            yendcrop = -np.minimum( np.floor( trans[:,0].min() ), minCrop )
            xendcrop = -np.minimum( np.floor( trans[:,1].min() ), minCrop )
            ystartcrop = np.maximum( np.ceil( trans[:,0].max() ), minCrop )
            xstartcrop = np.maximum( np.ceil( trans[:,1].max() ), minCrop )
            cropLim =  np.array( [ystartcrop, xstartcrop, yendcrop, xendcrop] )

                         
        print( cropLim )
        print( Image1.shape )
        hann = util.apodization( name='hann', shape=box ).astype(float_dtype)
        tilesX = np.floor( np.float( Image1.shape[1] - cropLim[1] - cropLim[3] - box[1])/ box[1] / (1.0-overlap) ).astype('int')
        tilesY = np.floor( np.float( Image1.shape[0] - cropLim[0] - cropLim[2] - box[0])/ box[0] / (1.0-overlap) ).astype('int')
        print( "Tiles for FRC: " + str( tilesX) + ":" + str(tilesY))
        FFTEven = np.zeros( box, dtype=fftw_dtype )
        FFTOdd = np.zeros( box, dtype=fftw_dtype )
        normConstBox = np.float32( 1.0 / FFTEven.size**2 )
        FRC2D = np.zeros( box, dtype=float_dtype )
        for I in np.arange(0,tilesY):
            for J in np.arange(0,tilesX):
                offset = np.array( [ I*box[0]*(1.0-overlap)+cropLim[0], J*box[1]*(1.0-overlap)+cropLim[1] ])
                
                tileEven = (hann*Image1[offset[0]:offset[0]+box[0], offset[1]:offset[1]+box[1] ]).astype(fftw_dtype)
                FFT2.update_arrays( tileEven, FFTEven ); FFT2.execute()
                tileOdd = (hann*Image2[offset[0]:offset[0]+box[0], offset[1]:offset[1]+box[1] ]).astype(fftw_dtype)
                FFT2.update_arrays( tileOdd, FFTOdd ); FFT2.execute()
    
                FFTOdd *= normConstBox
                FFTEven *= normConstBox
                
                # Calculate the normalized FRC in 2-dimensions
                # FRC2D += ne.evaluate( "real(FFTEven*conj(FFTOdd)) / sqrt(real(abs(FFTOdd)**2) * real(abs(FFTEven)**2) )" )
                # Some issues with normalization?
                FRC2D += ne.evaluate( "real(FFTEven*conj(FFTOdd)) / sqrt(real(FFTOdd*conj(FFTOdd)) * real(FFTEven*conj(FFTEven)) )" )
              
        # Normalize
        FRC2D /= FRC2D[0,0]
        FRC2D = np.fft.fftshift( FRC2D )
        
        rotFRC, _ = util.rotmean( FRC2D )
        self.FRC = rotFRC
        self.FRC2D = FRC2D

    def localFRC( self, box=[256,256], overlap=0.5 ):
        # Only work on the even and odd frames?
        m = self.images.shape[0]
        evenIndices = np.arange(0, m, 2)
        oddIndices = np.arange(1, m, 2)  
        
        center = 2048
        evenBox = np.sum( self.images[evenIndices, center-box[0]/2:center+box[0]/2, center-box[1]/2:center+box[1]/2 ], axis=0 )
        oddBox = np.sum( self.images[oddIndices, center-box[0]/2:center+box[0]/2, center-box[1]/2:center+box[1]/2 ], axis=0 )
        FFTEven = np.zeros( box, dtype=fftw_dtype )
        FFTOdd = np.zeros( box, dtype=fftw_dtype )
        
        normConstBox = np.float32( 1.0 / FFTEven.size**2 )
        
        FFT2, _ = util.pyFFTWPlanner( np.zeros(box, dtype=fftw_dtype), 
                             wisdomFile=os.path.join( self.cachePath, "fftw_wisdom.pkl" ) , n_threads = self.n_threads, 
                             effort=self.fftw_effort, doReverse=False )
        FFT2.update_arrays( evenBox, FFTEven ); FFT2.execute()
        FFT2.update_arrays( oddBox, FFTOdd ); FFT2.execute()
        
        FFTOdd *= normConstBox
        FFTEven *= normConstBox
        
        FRC2D = ne.evaluate( "real(FFTEven*conj(FFTOdd)) / sqrt(real(FFTOdd*conj(FFTOdd)) * real(FFTEven*conj(FFTEven)) )" )
        FRC2D /= FRC2D[0,0]
        FRC2D = np.fft.fftshift( FRC2D )
        
        rotFRC, _ = util.rotmean( FRC2D )
        
        plt.figure()
        plt.plot( rotFRC )
        plt.title( "Local FRC over box = " + str(box) )
        
    def doseFilter( self, dosePerFrame=None ):
        """
        This is a port from Grant's electron_dose.f90 from UnBlur.  It uses fixed critical dose factors
        to apply filters to each image based on their accumulated dose.  
        
        dosePerFrame by default is estimated from the data. Currently we have no gain factor, so we assume 
        the input numbers are in electrons.  Optionally one may provide the dosePerFrame, which becomes 
        helpful on a thicker specimen, or one that has been pre-processed (i.e. binned from 8k to 4k).  
        """
        
        critDoseA = np.float32( 0.24499 )
        critDoseB = np.float32( -1.6649 )
        critDoseC = np.float32( 2.8141 )
        voltageScaling = np.float32( np.sqrt( self.voltage / 300.0  ) ) # increase in radiolysis at lower values.
        cutoffOrder = 16
        
        # It looks like they build some mesh that is sqrt(qxmesh + qymesh) / pixelsize
        # I think this is probably just qmesh in inverse Angstroms (keeping in mind Zorro's internal
        # pixelsize is nm)
        m = self.images.shape[0]
        N = self.shapePadded[0]
        M = self.shapePadded[1]
        invPSx = 1.0 / (M*(self.pixelsize*10))
        invPSy = 1.0 / (N*(self.pixelsize*10))
        
        xmesh, ymesh = np.meshgrid( np.arange(-M/2,M/2), np.arange(-N/2,N/2))
        qmesh = ne.evaluate( "sqrt(xmesh*xmesh*(invPSx**2) + ymesh*ymesh*(invPSy**2))" )
        qmesh = np.fft.fftshift( qmesh )
        
        # Since there's a lot of hand waving, let's assume dosePerFrame is constant
        # What about on a GIF where the observed dose is lower due to the filter?
        if dosePerFrame == None:
            totalDose = np.mean( self.imageSum )
            dosePerFrame = totalDose / m
        accumDose = np.zeros( m + 1 )
        accumDose[1:] = np.cumsum( np.ones(m) * dosePerFrame )
        # optimalDose = 2.51284 * critDose
        
        critDoseMesh = ne.evaluate( "voltageScaling*(critDoseA * qmesh**critDoseB + critDoseC)" )
        #critDoseMesh[N/2,M/2] = 0.001 * np.finfo( 'float32' ).max
        critDoseMesh[N/2,M/2] = critDoseMesh[N/2,M/2-1]**2

        # We probably don't need an entire mesh here...
        qvect = np.arange(0,self.shapePadded[0]/2) * np.sqrt( invPSx*invPSy )
        optiDoseVect = np.zeros( self.shapePadded[0]/2 )
        optiDoseVect[1:] = 2.51284*voltageScaling*(critDoseA * qvect[1:]**critDoseB + critDoseC)
        optiDoseVect[0] = optiDoseVect[1]**2
        
        padWidth = np.array(self.shapePadded) - np.array(self.imageSum.shape)
        doseFilter = np.zeros( self.shapePadded, dtype=fftw_dtype )
        FFTimage = np.zeros( self.shapePadded, dtype=fftw_dtype )
        # zorroReg.filtSum = np.zeros_like( zorroReg.imageSum )
        FFT2, _ = util.pyFFTWPlanner( doseFilter, wisdomFile=os.path.join( self.cachePath, "fftw_wisdom.pkl" ) , effort = self.fftw_effort, n_threads=self.n_threads, doReverse=False )
                
        for J in np.arange(0,m):
            print( "Filtering for dose: %.2f"%accumDose[J+1] )
            doseFinish = accumDose[J+1] # Dose at end of frame period
            doseStart = accumDose[J] # Dose at start of frame period
            # qmesh is in reciprocal angstroms, so maybe I can ignore how they build the mesh and 
            # use a matrix meshgrid
            
            # Try adding Fourier magnitudes, if that has artifacts we can do an FFT on each image
            filt = ne.evaluate( "exp( (-0.5*doseFinish)/critDoseMesh)")
            thresQ = qvect[ np.argwhere( np.abs(doseFinish - optiDoseVect) < np.abs(doseStart - optiDoseVect) )[-1] ]
            
            # thres = ne.evaluate( "abs(doseFinish - optiDoseMesh) < abs(doseStart - optiDoseMesh)" )
            # This filter step is slow, try to do this analytically?  Can we find the radius from the above equation?
            # thres = scipy.ndimage.gaussian_filter( thres.astype(zorro.float_dtype), cutoffSigma )
            thres = ne.evaluate( "exp( -(qmesh/thresQ)**cutoffOrder )" )
            
            # Numpy's pad is also quite slow
            paddedImage = np.pad( self.images[J,:,:].astype(fftw_dtype),
                                     ((0,padWidth[0]),(0,padWidth[1])), mode='symmetric'   )
                                     
            FFT2.update_arrays( paddedImage, FFTimage ); FFT2.execute()
            doseFilter += ne.evaluate( "FFTimage * thres * filt" )
        pass
        self.filtSum = np.abs( np.fft.ifft2( doseFilter ) )[:self.imageSum.shape[0],:self.imageSum.shape[1]]

        del invPSx, invPSy, qmesh, optiDoseVect, doseFinish, doseStart, critDoseA, critDoseB, critDoseC, 
        del voltageScaling, filt, thres, thresQ, cutoffOrder
        

    def setBfiltCutoff( self, cutoffSpacing ):
        """
        stackReg.bBfiltCutoff( cutoffSpacing )
        
        Expects stackReg.pixelsize to be set, and stackReg.images to be loaded.  
        Units of pixelsize from DM4 is nm, so the cutoff spacing should also be 
        nm.  E.g. cutoffspacing = 0.3 [nm] is 3.0 Angstroms. 
        
        For a gaussian B-filter, the cutoff is where the filter ampitude drops 
        to 1/exp(1)
        """
        shapeImage = np.array( self.images.shape[1:] )
        psInv = 1.0 / (self.pixelsize*shapeImage)
        
        cutoffInv = 1.0 / cutoffSpacing
        
        self.Brad = cutoffInv / psInv
        print( "Setting Brad to: " + str(self.Brad) )
        pass
    
    def getCropLimits( self, trans = None ):
        if trans is None:
            trans = self.translations

        yendcrop = np.minimum( np.floor( trans[:,0].min() ), 0 )
        if yendcrop == 0:
            yendcrop = None
        xendcrop = np.minimum( np.floor( trans[:,1].min() ), 0 )
        if xendcrop == 0:
            xendcrop = None
        ystartcrop = np.maximum( np.ceil( trans[:,0].max() ), 0 )
        xstartcrop = np.maximum( np.ceil( trans[:,1].max() ), 0 )
        return np.array( [ystartcrop, xstartcrop, yendcrop, xendcrop] )
        
    def getSumCropToLimits( self ):
        """
        Gets imageSum cropped so that no pixels with partial dose are kept.

        """
        cropLim = self.getCropLimits()
        return self.imageSum[cropLim[0]:cropLim[2], cropLim[1]:cropLim[3]]
        
    def getFiltSumCropToLimits( self ):
        """
        Gets filtSum cropped so that no pixels with partial dose are kept.
        """
        cropLim = self.getCropLimits()
        return self.filtSum[cropLim[0]:cropLim[2], cropLim[1]:cropLim[3]]
        
    def getImagesCropToLimits( self ):
        """
        Gets images stack cropped so that no pixels with partial dose are kept.
        """
        cropLim = self.getCropLimits()
        return self.images[:,cropLim[0]:cropLim[2], cropLim[1]:cropLim[3]]
        
    def getMaskCropLimited( self ):
        """
        Get a mask that crops the portion of the image that moved, for refinement.
        """
        cropLim = self.getCropLimits()
        if cropLim[2] == None: cropLim[2] = 0;
        if cropLim[3] == None: cropLim[3] = 0;
            
        if np.any( self.shapeOriginal ):
            newMask = np.zeros( [1,self.shapeOriginal[0],self.shapeOriginal[1]], dtype=float_dtype )
            newMask[:,cropLim[0]:self.shapeOriginal[0]+cropLim[2], cropLim[1]:self.shapeOriginal[1]+cropLim[3]] = 1.0
        else:
            newMask = np.zeros( [1,self.images.shape[1],self.images.shape[2]], dtype=float_dtype )
            newMask[:,cropLim[0]:self.images.shape[1]+cropLim[2], cropLim[1]:self.images.shape[2]+cropLim[3]] = 1.0
        return newMask
        
                                     
    def execCTFFind4( self, movieMode=False, box_size = 512, contrast=0.067, 
                     min_res=50.0, max_res=4.0, 
                     min_C1=5000.0, max_C1=45000.0, C1_step = 500.0, 
                     A1_tol = 100.0, displayDiag=False ):
        """
        Calls CTFFind4, must be on the system path.
             movieMode = True does not require an aligned image (works on Fourier magnitudes)
             box_size = CTFFind parameter, box size to FFT
             contrast = estimated phase contrast in images
             min_res = minimum resolution to fit, in Angstroms
             max_res = maximum resolution to fit, in Angstroms.  Water ice is around 3.4 Angstroms
             min_C1 = minimum defocus search range, in Angstroms
             max_C1 = maximum defocus search range, in Angstroms
             C1_step = defocus search step size, in Angstroms
             A1_tol = 2-fold astigmatism tolerance, in Angstroms
             displayDiag = True plots the diagnostic output image
        """
        if util.which( 'ctffind' ) is None:
            print( "Error: CTFFIND not found!" )
            return
        if self.pixelsize is None:
            print( "Set pixelsize (in nm) before calling execCTFFind4" )
            return
        elif self.voltage is None:
            print( "Set voltage (in kV) before calling execCTFFind4" )
            return
        elif self.C3 is None:
            print( "Set C3 (in mm) before calling execCTFFind4" )
            return
        
        print( "Calling CTFFIND4 for " + self.files['stack'] )
        print( "   written by Alexis Rohou: http://grigoriefflab.janelia.org/ctffind4" )
        print( "   http://biorxiv.org/content/early/2015/06/16/020917" )
        
        ps = self.pixelsize * 10.0
        min_res = np.min( [min_res, 50.0] )
        
        try: os.umask( 0002 ) # Why is Python not using default umask from OS?
        except: pass
        
        if self.cachePath is None:
            self.cachePath = "."
            
        # Force trailing slashes onto cachePatch
        stackBase = os.path.splitext( os.path.basename( self.files['stack'] ) )[0]
            
        diagOutName = os.path.join( self.cachePath, stackBase + ".ctf" )
         
        try: 
            mrcname = os.path.join( self.cachePath, stackBase + "_ctf4.mrc" )
            if movieMode:
                input_is_a_movie = 'true'
                ioMRC.MRCExport( self.images.astype('float32'), mrcname )
                number_of_frames_to_average = 1
            else:
                input_is_a_movie = 'false'
                ioMRC.MRCExport( self.imageSum.astype('float32'), mrcname )
        except:
            print( "Error in exporting MRC file to CTFFind4" )
            return
         
        # flags = "--amplitude-spectrum-input --filtered-amplitude-spectrum-input"
        flags = "" # Not using any flags
        find_additional_phase_shift = 'false'
        
        ctfexec = ( "ctffind " + flags + " << STOP_PARSING \n" + mrcname )
        if input_is_a_movie == 'true' or input_is_a_movie == 'yes':
             ctfexec = ctfexec + "\n" + input_is_a_movie + "\n" + str(number_of_frames_to_average)
         
        ctfexec = (ctfexec + "\n" + diagOutName + "\n" + str(ps) + "\n" + str(self.voltage) + "\n" +
            str(self.C3) + "\n" + str(contrast) + "\n" + str(box_size) + "\n" +
            str(min_res) + "\n" + str(max_res) + "\n" + str(min_C1) + "\n" + 
            str(max_C1) + "\n" + str(C1_step) + "\n" + str(A1_tol) + "\n" + 
            find_additional_phase_shift )    
        ctfexec = ctfexec + "\nSTOP_PARSING"

        print( ctfexec )
        sub = subprocess.Popen( ctfexec, shell=True )
        sub.wait()
        # os.system( ctfexec )
        
        #print( "CTFFIND4 execution time (s): " + str(t1-t0))    
        try:
            logName = os.path.join( self.cachePath, stackBase + ".txt" )
            print( "Trying to load from: " + logName )
            # Log has 5 comment lines, then 1 header, and
            # Micrograph number, DF1, DF2, Azimuth, Additional Phase shift, CC, and max spacing fit-to
            self.CTF4Results = np.loadtxt(logName, comments='#', skiprows=1 )
            self.CTF4Diag = ioMRC.MRCImport( diagOutName )
            
        except IOError:
            print( "CTFFIND4 likely core-dumped, try different input parameters?" )
        pass
        # Write a RELION-style _ctffind3.log file, with 5 um pixel size...
        self.saveRelionCTF3( amp_contrast=contrast, dstep = 5.0 )
            
        # TODO: having trouble with files not being deletable, here.  Is CTFFIND4 holding them open?  Should 
        # I just pause for a short time?
        time.sleep(0.5) # DEBUG: try and see if temporary files are deletable now.
        try: os.remove( mrcname )
        except IOError: 
            print( "Could not remove temporary file: " + str(IOError.message) )
        try: os.remove( diagOutName )
        except: pass
        # Delete CTF4 logs
        try: os.remove( os.path.join( self.cachePath, stackBase + "_avrot.txt") )
        except: pass
        try: os.remove( logName )
        except: pass
        try: os.remove( os.path.join( self.cachePath, stackBase + ".ctf" ) )
        except: pass
       
    def saveRelionCTF3(self, amp_contrast=0.07, dstep=5.0 ):
        # Saves the results from CTF4 in a pseudo-CTF3 log that RELION 1.3/1.4 can handle
        # Relevant code is in ctffind_runner.cpp, in the function getCtffindResults() (line 248)
        # Relion searchs for: 
        #   "CS[mm], HT[kV], AmpCnst, XMAG, DStep[um]"
        # and
        #      DFMID1      DFMID2      ANGAST          CC
        #
        #    15876.71    16396.97       52.86     0.10179  Final Values
    
        # Mag goes from micrometers of detector pixel size, to specimen pixel size (in nm)
        mag = (dstep*1E-6) / (self.pixelsize*1E-9)
        
        if self.files['sum'] != None:
            sumFront = os.path.splitext( self.files['sum'] )[0]
        else:
            sumFront = os.path.splitext( self.files['stack'] )[0]
        # Check to see if the sum directory exists already or not
        sumDir = os.path.split( sumFront )[0]
        if bool(sumDir) and not os.path.isdir( sumDir ):
            os.mkdir( sumDir )         
        
        self.files['ctflog'] = sumFront + "_ctffind3.log"
        logh = open( self.files['ctflog'], "w" )
        
        logh.write( "CS[mm], HT[kV], AmpCnst, XMAG, DStep[um]\n" )
        logh.write( "%.2f"%self.C3 + " %.1f"%self.voltage + " " + 
            str(amp_contrast) + " %.1f" %mag + " %.2f"%dstep + "\n" )

        logh.write( "%.1f"%self.CTF4Results[1]+ " %.1f"%self.CTF4Results[2] 
            + " %.4f"%self.CTF4Results[3]+ " %.4f"%self.CTF4Results[4] + " Final Values\n ")
        logh.close()
        pass
    
    def loadData( self, stackNameIn = None, target="stack", leading_zeros=0, useMemmap=False, endian='le' ):
        """
        Import either a sequence of DM3 files, a MRCS stack, a DM4 stack, or an HDF5 file.
        
        Target is a string representation of the member name, i.e. 'images', 'imageSum', 'C0'
        
        Files can be compressed with 'lbzip2' (preferred) or 'pigz' with extension '.bz2' or '.gz'
        
        On Windows machines you must have 7-zip in the path to manage compression, and 
        only .bz2 is supported
        
        filename can be an absolute path name or relative pathname.  Automatically 
        assumes file format based on extension.
        """
        self.t[6] = time.time() 
        # import os
        from os.path import splitext
        
        if stackNameIn != None:
            self.files[target] = stackNameIn
            
        #### DECOMPRESS FILE ####
        # This will move the file to the cachePath, so potentially could result in some confusion
        self.files[target] = util.decompressFile( self.files[target], outputDir = self.cachePath )
        
        # Check for each if it's a sequence or not
        [file_front, file_ext] = splitext( self.files[target] )
        
        #### IMAGE FILES ####
        if file_ext == ".dm3" :
            print( "Loading DM3 files in sequence" )
            try:
                import DM3lib as dm3
                from glob import glob
            except:
                raise ImportError( "Error: DM3lib not found, download at: http://imagejdocu.tudor.lu/doku.php?id=plugin:utilities:python_dm3_reader:start" )
                return
             
            file_seq = file_front.rstrip( '1234567890' )
            filelist = glob( file_seq + "*" + file_ext )
            
            file_nums = []
            for I in range(0, len(filelist) ):
                # Get all the file_nums
                [file_front, fit_ext] = splitext( filelist[I] )
                file_strip = file_front.rstrip( '1234567890' ) # Strip off numbers
                file_nums.append( file_front[len(file_strip):]  )
            file_nums = np.sort( np.array(file_nums,dtype='int' ) )
            
            filecount = len(filelist)
            
            # TO DO: handle things that aren't sequential lists of DM3 files
            # Note, ideally we append to images rather than overwriting
            dm3struct = dm3.DM3( self.files[target] )
            tempData = np.empty( [ filecount, dm3struct.imagedata.shape[0], dm3struct.imagedata.shape[1]]  )
            tempData[0,:,:] = dm3struct.imagedata

            for I in np.arange( 1, filecount ):
                filenameDM3 = file_strip + str(file_nums[I]).zfill(leading_zeros) + self.file_ext
                print( "Importing: " + filenameDM3 )
                dm3struct = dm3.DM3( filenameDM3 )
                tempData[I,:,:] = dm3struct.imagedata
        elif file_ext == '.tif' or file_ext == '.tiff':
            try:
                import skimage.io
            except:
                print( "Error: scikit-image or glob not found!" )
                return  
                
            print( "Importing: " + self.files[target] )
            tempData = skimage.io.imread( self.files[target] )
                
            """
            # Sequence mode
            print( "Loading TIFF files in sequence" )
            try:
                import skimage.io
                from glob import glob
            except:
                print( "Error: scikit-image or glob not found!" )
                return

            file_seq = file_front.rstrip( '1234567890' )
            filelist = glob( file_seq + "*" + self.file_ext )
            
            file_nums = []
            for I in range(0, len(filelist) ):
                # Get all the file_nums
                [file_front, fit_ext] = splitext( filelist[I] )
                file_strip = file_front.rstrip( '1234567890' ) # Strip off numbers
                file_nums.append( file_front[len(file_strip):]  )
            file_nums = np.sort( np.array(file_nums,dtype='int' ) )
            filecount = len(filelist)
            
            # see if freeimage is available
            try:
                skimage.io.use_plugin( 'freeimage' )
            except:
                print( "FreeImage library not found, it is recommended for TIFF input." )
                skimage.io.use_plugin( 'tifffile' )
                
            mage1 = skimage.io.imread( self.files[target] )
            tempData = np.empty( [ filecount, mage1.shape[0], mage1.shape[1]]  )
            tempData[0,:,:] = mage1

            for I in np.arange( 1, filecount ):
                filenameTIFF = file_strip + str(file_nums[I]).zfill(leading_zeros) + self.file_ext
                print( "Importing: " + filenameTIFF )
                tempData[I,:,:] = skimage.io.imread( filenameTIFF )
            """
        elif file_ext == ".dm4":
            # Expects a DM4 image stack
            print( "Open as DM4: " + self.files[target] )
            dm4obj = ioDM.DM4Import( self.files[target], verbose=False, useMemmap = useMemmap )
            tempData = np.copy( dm4obj.im[1].imageData.astype( float_dtype ), order='C' )
            # Load pixelsize from file
            try:
                if bool( dm4obj.im[1].imageInfo['DimXScale'] ):
                    self.pixelsize = dm4obj.im[1].imageInfo['DimXScale'] # DM uses units of nm
            except KeyError: pass
            try: 
                if bool(dm4obj.im[1].imageInfo['Voltage'] ):
                    self.voltage = dm4obj.im[1].imageInfo['Voltage'] / 1000.0 # in kV
            except KeyError: pass
            try:
                if bool(dm4obj.im[1].imageInfo['C3']):
                    self.C3 = dm4obj.im[1].imageInfo['C3'] # in mm
            except KeyError: pass
            
            del dm4obj
        elif file_ext == ".mrc" or file_ext == '.mrcs':
            # Expects a MRC image stack
            tempData, header = ioMRC.MRCImport( self.files[target], endian=endian, returnHeader=True )
            self.pixelsize = np.float32( header['pixelsize'][0] / 10.0 ) # Convert from Angstroms to nm
            # Should try writing C3 and voltage somewhere 
        elif file_ext == ".hdf5" or file_ext == ".h5":
            # import tables
            try:
                h5file = tables.open_file( self.files[target], mode='r' )
            except:
                print( "Could not open HDF5 file: " + self.files[target] )
            print( h5file )
            try: tempData = np.copy( h5file.get_node( '/', "images" ), order='C' )
            except: print( "HDF5 file import did not find /images" )
            # TODO: load other nodes
            try: self.pixelsize = np.copy( h5file.get_node( '/', "pixelsize" ), order='C' )
            except: print( "HDF5 file import did not find /pixelsize" )
            try: self.knownDrift = np.copy( h5file.get_node( '/', "drift" ), order='C' )
            except: print( "HDF5 file import did not find /drift" )
            
            try:
                h5file.close()
            except:
                pass
            pass
        
        # Finally, assign to target
        if target == "stack" or target == 'align':
            self.images = tempData
        elif target == "sum":
            self.imageSum = tempData
            print( "TODO: set filename for imageSum in loadData" )
        elif target == "filt":
            self.filtSum = tempData
        elif target == "xc":
            self.C = tempData
            print( "TODO: set filename for C in loadData" )
        elif target == "mask":
            self.masks = tempData
            
        self.t[7] = time.time() 
        
    def saveData( self ):
        """
        Save files to disk.  
        
        Do compression of stack if requested, self.compression = '.bz2' for example
        uses lbzip2 or 7-zip. '.gz' is also supported by not recommended.
        
        TODO: add dtype options, including a sloppy float for uint16 and uint8
        """
        import os
        try: os.umask( 0002 ) # Why is Python not using default umask from OS?
        except: pass

        # If self.files['config'] exists we save relative to it.  Otherwise we default to the place of 
        # self.files['stack']
        if bool( self.files['config'] ): 
            baseDir = os.path.split( self.files['config'] )[0]
        else:
            baseDir = os.path.split( self.files['stack'] )[0]
        stackFront, stackExt = os.path.splitext( os.path.basename( self.files['stack'] ) )
        
        # Change the current directory to make relative pathing sensible
        os.chdir( baseDir )
        
        if stackExt == ".bz2" or stackExt == ".gz" or stackExt == ".7z":
            # compressExt = stackExt
            stackFront, stackExt = os.path.splitext( stackFront )
            
        if self.files['sum'] is None: # Default sum name
            self.files['sum'] = os.path.join( "sum", stackFront + "_zorro.mrc" )

        # Does the directory exist?  Often this will be a relative path to file.config
        sumPath, sumFile = os.path.split( self.files['sum'] )
        if not os.path.isabs( sumPath ):
            sumPath = os.path.realpath( sumPath ) # sumPath is always real
        if bool(sumPath) and not os.path.isdir( sumPath ):
            os.mkdir( sumPath )
        relativeSumPath = os.path.relpath( sumPath )
        
        #### SAVE ALIGNED SUM ####
        if self.verbose >= 1:
            print( "Saving: " + os.path.join(sumPath,sumFile) )
        ioMRC.MRCExport( self.imageSum.astype('float32'), os.path.join(sumPath,sumFile) )

        # Compress sum
        if bool(self.doCompression):
            util.compressFile( os.path.join(sumPath,sumFile), self.compress_ext, n_threads=self.n_threads )

        #### SAVE ALIGNED STACK ####
        if bool(self.saveMovie):
            if self.files['align'] is None: # Default filename for aligned movie
                self.files['align'] = os.path.join( "align", stackFront + "_zorro_movie.mrcs" )
                
            # Does the directory exist?
            alignPath, alignFile = os.path.split( self.files['align'] )
            if not os.path.isabs( sumPath ):
                alignPath = os.path.realpath( alignPath )
            if bool(alignPath) and not os.path.isdir( alignPath ):
                os.mkdir( alignPath )
            
            if self.verbose >= 1:
                print( "Saving: " + os.path.join(alignPath,alignFile) )
            ioMRC.MRCExport( self.images.astype('float32'), os.path.join(alignPath,alignFile) ) 

            # Compress stack
            if bool(self.doCompression):
                util.compressFile( os.path.join(alignPath,alignFile), self.compress_ext, n_threads=self.n_threads )
                
        if bool(self.doDoseFilter): # This will be in the same place as sum
            if self.files['filt'] is None: # Default filename for filtered sum
                self.files['filt'] = os.path.join( relativeSumPath, os.path.splitext(sumFile)[0] + "_filt" +  os.path.splitext(sumFile)[1] )
            
            filtPath, filtFile = os.path.split( self.files['filt'] )
            if not os.path.isabs( filtPath ):
                filtPath = os.path.realpath( filtPath ) # sumPath is always real
                
            if self.verbose >= 1:
                print( "Saving: " + os.path.join(filtPath, filtFile) )
            ioMRC.MRCExport( self.filtSum.astype('float32'), os.path.join(filtPath, filtFile) ) 

        #### SAVE CROSS-CORRELATIONS FOR FUTURE PROCESSING OR DISPLAY ####
        if self.saveC:
            self.files['xc'] = os.path.join( self.files['figurePath'], os.path.splitext(sumFile)[0] + "_xc.mrc" )
            if self.verbose >= 1:
                print( "Saving: " + self.files['xc'] )
                
            ioMRC.MRCExport( np.asarray( self.C, dtype='float32'), self.files['xc'] )
            if bool(self.doCompression):
                util.compressFile( self.files['xc'], self.compress_ext, n_threads=self.n_threads )
            
        #### SAVE OTHER INFORMATION IN A LOG FILE ####
        # Log file is saved seperately...  Calling it here could lead to confusing behaviour.

        if bool( self.files['moveRawPath'] ) and not os.path.isdir( self.files['moveRawPath'] ):
            os.mkdir( self.files['moveRawPath'] )
                
        if bool( self.doCompression ): # does compression and move in one op
            self.files['stack'] = util.compressFile( self.files['stack'], outputDir=self.files['moveRawPath'], 
                                               n_threads=self.n_threads, compress_ext=self.compress_ext )
        elif bool( self.files['moveRawPath'] ):
            newStackName = os.path.join( self.files['moveRawPath'], os.path.split( self.files['stack'])[1] )
            print( "Moving " +self.files['stack'] + " to " + newStackName )
            os.rename( self.files['stack'], newStackName )
            self.files['stack'] = newStackName
        pass
    


    def loadConfig( self, configNameIn = None, loadData=False ):
        """
        Initialize the ImageRegistrator class from a config file
        
        loadData = True will load data from the given filenames.
        """
        import ConfigParser
        import json
        
        if not bool(configNameIn):
            if not bool( self.files['config'] ):
                pass # Do nothing
            else:
                print( "Cannot find configuration file: " + self.files['config'] )
        else:
            self.files['config'] = configNameIn

        print( "Loading config file: " + self.files['config'] )
        config = ConfigParser.ConfigParser(allow_no_value = True)
        config.optionxform = str
        
        ##### Paths #####
        config.read( self.files['config'] )
        # I'd prefer to pop an error here if configName doesn't exist
        
        
        
        # Initialization
        try: self.verbose = config.getint( 'initialization', 'verbose' )
        except: pass

        try: self.fftw_effort = config.get( 'initialization', 'fftw_effort' ).upper()
        except: pass
        try: self.n_threads = config.getint( 'initialization', 'n_threads' )
        except: pass
        try: self.saveC = config.getboolean( 'initialization', 'saveC' )
        except: pass
        try: self.saveCsub = config.getint( 'initialization', 'saveCsub' )
        except: pass
        try: self.METAstatus = config.get( 'initialization', 'METAstatus' )
        except: pass
    
        # Calibrations
        try: self.pixelsize = config.getfloat('calibration','pixelsize')
        except: pass
        try: self.voltage = config.getfloat('calibration','voltage')
        except: pass
        try: self.C3 = config.getfloat('calibration','C3')
        except: pass
    
        # Data
        try: self.trackCorrStats = config.getboolean( 'data', 'trackCorrStats' )
        except: pass
        try: 
            keyList = json.loads( config.get( 'corrstats', 'keylist' ) )
            corrStats = {}
            for key in keyList:
                corrStats[key] = np.array( json.loads( config.get( 'corrstats', key ) ) )
                # convert singular values from arrays
                if corrStats[key].size == 1:
                    corrStats[key] = corrStats[key].item(0)
            self.corrStats = corrStats
        except: pass
        try: self.CTF4Results = json.loads( config.get( 'data', 'CTF4Results' ) )
        except: pass
        

        # Results 
        # Load arrays with json
        try: self.translations = np.array( json.loads( config.get( 'results', 'translations' ) ) )
        except: pass
        try: self.transEven = np.array( json.loads( config.get( 'results', 'transEven' ) ) )
        except: pass
        try: self.transOdd = np.array( json.loads( config.get( 'results', 'transOdd' ) ) )
        except: pass
        try: self.velocities = np.array( json.loads( config.get( 'results', 'velocities' ) ) )
        except: pass
        try: self.rotations = np.array( json.loads( config.get( 'results', 'rotations' ) ) )
        except: pass
        try: self.scales = np.array( json.loads( config.get( 'results', 'scales' ) ) )
        except: pass
        try: self.FRC = np.array( json.loads( config.get( 'results', 'FRC' ) ) )
        except: pass
        try: self.doCTF = config.getboolean( 'results', 'doCTF' )
        except: pass

        errorDictsExist=True
        errCnt = 0
        while errorDictsExist:
            try:
                newErrorDict = {}
                dictName = 'errorDict'+str(errCnt)
                # Load the list of keys and then load them element-by-element
                keyList = json.loads( config.get( dictName, 'keylist' ) )
                for key in keyList:
                    newErrorDict[key] = np.array( json.loads( config.get( dictName, key ) ) )
                    # convert singular values from arrays
                    if newErrorDict[key].size == 1:
                        newErrorDict[key] = newErrorDict[key].item(0)
                self.errorDictList.append(newErrorDict)
            except: # This stops loading dicts on more or less any error at present
                errorDictsExist=False
                break
            errCnt += 1
            
        
        # Registration parameters
        try: self.triMode = config.get('registration', 'triMode' )
        except: pass
    
        try: self.startFrame = config.getint('registration', 'startFrame' )
        except: pass
        try: self.endFrame = config.getint('registration', 'endFrame' )
        except: pass
    
        try: self.shapePadded = np.array( json.loads( config.get( 'registration', 'shapePadded' ) ) )
        except: pass
    
        try: self.shapeOriginal = np.array( json.loads( config.get( 'registration', 'shapeOriginal' ) ) )
        except: pass
        try: self.shapeBinned = np.array( json.loads( config.get( 'registration', 'shapeBinned' ) ) )
        except: pass
        try: self.fouCrop = np.array( json.loads( config.get( 'registration', 'fouCrop' ) ) )
        except: pass
        try: self.subPixReg = config.getint('registration', 'subPixReg' )
        except: pass
        try: self.shiftMethod = config.get('registration', 'shiftMethod' )
        except: pass
        try: self.maxShift = config.getint('registration', 'maxShift' )
        except: pass
        try: self.preShift = config.getboolean( 'registration', 'preShift' )
        except: pass
        try: self.triMode = config.get('registration', 'triMode' )
        except: pass
        try: self.diagWidth = config.getint('registration', 'diagWidth' )
        except: pass
        try: self.autoMax = config.getint('registration', 'autoMax' )
        except: pass
        try: self.peaksigThres = config.getfloat( 'registration', 'peaksigThres' )
        except: pass
        try: self.pixErrThres = config.getfloat( 'registration', 'pixErrThres' )
        except: pass
        try: self.sigmaThres = config.getfloat('registration', 'sigmaThres' )
        except: pass
        try: self.corrThres = config.getfloat('registration', 'corrThres' )
        except: pass
        try: self.velocityThres = config.getfloat('registration', 'velocityThres' )
        except: pass
        try: self.Brad = config.getfloat('registration', 'Brad' )
        except: pass
        try: self.Bmode = config.get('registration', 'Bmode' )
        except: pass
        try: self.BfiltType = config.get('registration', 'BfiltType' )
        except: pass
        try: self.originMode = config.get('registration', 'originMode' )
        except: pass
        try: self.suppressOrigin = config.getboolean( 'registration', 'suppressOrigin' )
        except: pass
        try: self.weightMode = config.get('registration', 'weightMode' )
        except: pass
        try: self.logisticK = config.getfloat('registration', 'logisticK' )
        except: pass
        try: self.logisticNu = config.getfloat('registration', 'logisticNu' )
        except: pass
        try: self.doDoseFilter = config.getboolean( 'registration', 'doDoseFilter' )
        except: pass
        try: self.doFRC = config.getboolean( 'registration', 'doLazyFRC' )
        except: pass
    
        # IO 
        try:
            filesKeys = json.loads( config.get( 'io', 'fileskeys' ) )
            #self.files = {}
            for fileKey in filesKeys:
                self.files[fileKey] = config.get( 'io', fileKey )
        except: pass
        #try: self.savePDF = config.getboolean('io', 'savePDF' )
        #except: pass
        try: self.savePNG = config.getboolean('io', 'savePNG' )
        except: pass
        try: self.compress_ext = config.get('io', 'compress_ext' )
        except: pass
        try: self.saveMovie = config.getboolean( 'io', 'saveMovie' )
        except: pass
        try: self.doCompression = config.getboolean( 'io', 'doCompression' )
        except: pass
    
        # Plot
        try:
            plotDictKeys = json.loads( config.get( 'plot', 'plotkeys' ) )
            for plotKey in plotDictKeys:
                self.plotDict[plotKey] = util.guessCfgType( config.get( 'plot', plotKey ) )
        except: pass    

        
        if bool(loadData) and self.files.has_key('stack') and self.files['stack'] != None:
            self.loadData()
        pass
    
    def saveConfig( self, configNameIn=None ):
        """
        Write the state of the ImageRegistrator class from a config file
        """
        import ConfigParser
        import json
        import os
        try: os.umask( 0002 ) # Why is Python not using default umask from OS?
        except: pass        
        
        if not bool( configNameIn ):
            if self.files['config'] is None:
                self.files['config'] = self.files['stack'] + ".log"
        else:
            self.files['config'] = configNameIn
        # Does the directory exist?
        configPath = os.path.realpath( os.path.split( self.files['config'] )[0] )
        if bool(configPath) and not os.path.isdir( configPath ):
            os.mkdir( configPath )
        
        # Write config
        config = ConfigParser.ConfigParser(allow_no_value = True)
        config.optionxform = str
        
        # Initialization
        config.add_section( 'initialization' )
        config.set( 'initialization', '# For detailed use instructions: github.com/C-CINA/zorro/wiki/Advice-for-Choice-of-Registration-Options', None )
        config.set( 'initialization', 'verbose', self.verbose )
        config.set( 'initialization', 'fftw_effort', self.fftw_effort )
        # Any time we cast variables we need handle errors from numpy
        config.set( 'initialization', '# n_threads is usually best if set to the number of physical cores (CPUs)' )
        try: config.set( 'initialization', 'n_threads', np.int(self.n_threads) )
        except: pass
        config.set( 'initialization', 'saveC', self.saveC )
        config.set( 'initialization', 'saveCsub', self.saveCsub )
        config.set( 'initialization', 'METAstatus', self.METAstatus )
        
        # Calibrations
        config.add_section( 'calibration' )
        config.set( 'calibration', "# Zorro can strip this information from .DM4 files if its is present in tags" )
        config.set( 'calibration' , "# Pixel size in nanometers" )
        config.set( 'calibration','pixelsize', self.pixelsize )
        config.set( 'calibration' , "# Accelerating voltage in kV" )
        config.set( 'calibration','voltage', self.voltage )
        config.set( 'calibration' , "# Spherical aberration in mm" )
        config.set( 'calibration','C3', self.C3 )
        
        # Registration parameters
        config.add_section( 'registration' )
        config.set( 'registration' , "# tri, diag, first, auto, or autocorr" )
        config.set( 'registration', 'triMode', self.triMode )
        
        
        if self.shapePadded is not None:
            if type(self.shapePadded) == type(np.array(1)):
                self.shapePadded = self.shapePadded.tolist()
            config.set( 'registration', "# Use a padding 10 % bigger than the original image, select an efficient size with zorro_util.findValidFFTWDim()" )   
            config.set( 'registration', 'shapePadded', json.dumps( self.shapePadded) )
            
        if self.shapeOriginal is not None:
            if type(self.shapeOriginal) == type(np.array(1)):
                self.shapeOriginal = self.shapeOriginal.tolist()
            config.set( 'registration', 'shapeOriginal', json.dumps( self.shapeOriginal ) )
        if self.shapeBinned is not None:
            if type(self.shapeBinned) == type(np.array(1)):
                self.shapeBinned = self.shapeBinned.tolist()
            config.set( 'registration', 'shapeBinned', json.dumps( self.shapeBinned ) )
            
        if self.fouCrop is not None:
            if type(self.fouCrop) == type(np.array(1)):
                self.fouCrop = self.fouCrop.tolist()
            config.set( 'registration', 'fouCrop', json.dumps( self.fouCrop ) )
        
        try: config.set( 'registration', 'subPixReg', np.int(self.subPixReg) )
        except: pass
        config.set( 'registration', 'shiftMethod', self.shiftMethod )
        config.set( 'registration' , "# Maximum shift in pixels within diagWidth/autoMax frames" )
        try: config.set( 'registration', 'maxShift', np.int(self.maxShift) )
        except: pass
        config.set( 'registration' , "# preShift = True is useful for crystalline specimens where you want maxShift to follow the previous frame position" )
        config.set( 'registration', 'preShift', self.preShift )
        
        try: config.set( 'registration', 'diagWidth', np.int(self.diagWidth) )
        except: pass
        try: config.set( 'registration', 'autoMax', np.int(self.autoMax) )
        except: pass
        try: config.set( 'registration', 'startFrame', np.int(self.startFrame) )
        except: pass
        try: config.set( 'registration', 'endFrame', np.int(self.endFrame) )
        except: pass
        
        config.set( 'registration' , "# peakSigThres changes with dose but usually is uniform for a dataset" )
        config.set( 'registration', 'peaksigThres', self.peaksigThres )
        config.set( 'registration' , "# DEPRECATED: pixErrThres and sigmaThres throw away equations which do not improve the solution" )
        config.set( 'registration', 'pixErrThres', self.pixErrThres )
        config.set( 'registration', 'sigmaThres', self.sigmaThres )
        config.set( 'registration' , "# corrThres is DEPRECATED" )
        config.set( 'registration', 'corrThres', self.corrThres )
        config.set( 'registration', 'velocityThres', self.velocityThres )
        config.set( 'registration' , "# Brad is radius of B-filter in Fourier pixels" )
        config.set( 'registration', 'Brad', self.Brad )
        config.set( 'registration' , "# Bmode = conv, opti, or fourier" )
        config.set( 'registration', 'Bmode', self.Bmode )
        config.set( 'registration', 'BFiltType', self.BfiltType )
        config.set( 'registration' , "# originMode is centroid, or (empty), empty sets frame 1 to (0,0)" )
        config.set( 'registration', 'originMode', self.originMode )
        config.set( 'registration' , "# weightMode is one of logistic, corr, norm, unweighted" )
        config.set( 'registration', 'weightMode', self.weightMode )
        config.set( 'registration', 'logisticK', self.logisticK )
        config.set( 'registration', 'logisticNu', self.logisticNu )
        config.set( 'registration' , "# Set suppressOrigin = True if gain reference artifacts are excessive" )
        config.set( 'registration', 'suppressOrigin', self.suppressOrigin )
        config.set( 'registration', 'doDoseFilter', self.doDoseFilter )
        config.set( 'registration', 'doLazyFRC', self.doLazyFRC )
        
        # IO
        config.add_section('io')
        #config.set( 'io', 'savePDF', self.savePDF )
        config.set( 'io', 'savePNG', self.savePNG )
        config.set( 'io', 'compress_ext', self.compress_ext )
        config.set( 'io', 'saveMovie', self.saveMovie )
        config.set( 'io', 'doCompression', self.doCompression )
        fileKeys = self.files.keys()
        config.set( 'io' , "# Note: all paths are relative to the config/log file" )
        config.set( 'io', 'fileskeys', json.dumps( fileKeys ) )
        for fileKey in fileKeys:
            config.set( 'io', fileKey, self.files[fileKey] )

        # Plot
        config.add_section( 'plot' )
        plotDictKeys = self.plotDict.keys()
        config.set( 'plot', 'plotkeys', json.dumps( plotDictKeys ) )
        for plotKey in plotDictKeys:
            config.set( 'plot', plotKey, self.plotDict[plotKey] )
        
        # Results 
        # Seems Json does a nice job of handling numpy arrays if converted to lists
        config.add_section( 'results' )
        if self.translations is not None:
            config.set( 'results', 'translations', json.dumps( self.translations.tolist() ) )
        if self.transEven is not None:
            config.set( 'results', 'transEven', json.dumps( self.transEven.tolist() ) )
        if self.transOdd is not None:
            config.set( 'results', 'transOdd', json.dumps( self.transOdd.tolist() ) )
        if self.rotations is not None:    
            config.set( 'results', 'rotations', json.dumps( self.rotations.tolist() ) )
        if self.scales is not None:
            config.set( 'results', 'scales', json.dumps( self.scales.tolist() ) )
        if self.velocities is not None:
            config.set( 'results', 'velocities', json.dumps( self.velocities.tolist() ) )
        if self.FRC is not None:
            config.set( 'results', 'FRC', json.dumps( self.FRC.tolist() ) )
        config.set( 'results', 'doCTF', self.doCTF )
    
        # Data
        config.add_section( 'data' )
        try: config.set( 'data', 'CTF4Results', json.dumps( self.CTF4Results.tolist() ) )
        except: pass
            
        config.set( 'data', 'trackCorrStats', self.trackCorrStats )
        if self.corrStats is not None:
            config.add_section( 'corrstats' )
            keyList = self.corrStats.keys()
            config.set( 'corrstats', 'keylist', json.dumps( keyList ) )
            for key in keyList:
                if( hasattr( self.corrStats[key], "__array__" ) ):
                    config.set( 'corrstats', key, json.dumps( self.corrStats[key].tolist() ) )
                else:
                    config.set( 'corrstats', key, json.dumps( self.corrStats[key] ) )
                    
        # Error dicts
        for errCnt, errorDict in enumerate(self.errorDictList):
            dictName = 'errorDict'+str(errCnt)
            config.add_section( dictName )
            keyList = errorDict.keys()
            config.set( dictName, 'keylist', json.dumps( keyList ) )
            for key in keyList:
                if( hasattr( errorDict[key], "__array__" ) ):
                    config.set( dictName, key, json.dumps( errorDict[key].tolist() ) )
                else:
                    config.set( dictName, key, json.dumps( errorDict[key] ) )
        
        try:
            # Would be nice to have some error handling if cfgFH already exists
            # Could try and open it with a try: open( 'r' )
            cfgFH = open( self.files['config'] , 'w+' )
            if self.verbose >= 1:
                print( "Saving config file: " + self.files['config'] )
            config.write( cfgFH )
            cfgFH.close()
        except:
            print( "Error in loading config file: " + self.files['config'] )


    def plot( self, title = "" ):
        """
        Multiprocessed matplotlib diagnostic plots. 
        
        For each plot, make a list that contains the name of the plot, and a dictionary that contains all the 
        information necessary to render the plot.
        """
        
        if not bool(title):
            # Remove any pathing from default name as figurePath overrides this.
            if bool( self.files['stack'] ):
                self.plotDict['title'] = os.path.split( self.files['stack'] )[1]
            else:
                self.plotDict['title'] = "default"
        else:
            self.plotDict['title'] = title
            
        # figurePath needs to be relative to the config directory, which may not be the current directory.
        if bool(self.savePNG ):
            os.chdir( os.path.split(self.files['config'])[0] )
            
        plotArgs = []
        # IF IMAGESUM
        if np.any(self.imageSum) and self.plotDict.has_key('imageSum') and ( self.plotDict['imageSum'] ):
            print( "zorro.plot.imageSum" )
            plotDict = self.plotDict.copy()
            
            # Unfortunately binning only saves time if we do it before pickling the data off to multiprocess.
            # TODO: http://stackoverflow.com/questions/7894791/use-numpy-array-in-shared-memory-for-multiprocessing
            binning = 2
            plotDict['pixelsize'] = self.pixelsize * binning
            imageSumBinned = util.magickernel( self.getSumCropToLimits(), k=1 )
            plotDict['image'] = imageSumBinned

            if bool(self.savePNG):
                plotDict['plotFile'] = os.path.join( self.files['figurePath'], self.plotDict['title'] + "_imageSum.png")
                self.files['figImageSum'] = plotDict['plotFile']
            plotArgs.append( ['image', plotDict] )
            
        # IF FILTSUM
        if np.any(self.filtSum) and self.plotDict.has_key('filtSum') and bool( self.plotDict['filtSum'] ):
            print( "zorro.plot.filtSum" )
            plotDict = self.plotDict.copy()
            
            # Unfortunately binning only saves time if we do it before pickling the data off to multiprocess.
            # TODO: http://stackoverflow.com/questions/7894791/use-numpy-array-in-shared-memory-for-multiprocessing
            binning = 2
            plotDict['pixelsize'] = self.pixelsize * binning
            filtSumBinned = util.magickernel( self.getFiltSumCropToLimits(), k=1 )
            plotDict['image'] = filtSumBinned        

            if bool(self.savePNG):
                plotDict['plotFile'] = os.path.join( self.files['figurePath'], self.plotDict['title'] + "_filtSum.png")
                self.files['figFiltSum'] = plotDict['plotFile']
            plotArgs.append( ['image', plotDict] )
        
        # IF FFTSUM
        if np.any(self.imageSum) and self.plotDict.has_key('FFTSum') and bool( self.plotDict['FFTSum'] ):
            print( "zorro.plot.FFTSum" )
            plotDict = self.plotDict.copy()
            try: 
                plotDict['pixelsize'] = self.pixelsize * binning
                plotDict['image'] = imageSumBinned
            except:
                binning = 2
                plotDict['pixelsize'] = self.pixelsize * binning
                imageSumBinned = util.magickernel( self.getSumCropToLimits(), k=1 )
                plotDict['image'] = imageSumBinned
                
            if bool(self.savePNG):
                plotDict['plotFile'] = os.path.join( self.files['figurePath'], self.plotDict['title'] + "_FFTSum.png")
                self.files['figFFTSum'] = plotDict['plotFile']
            plotArgs.append( ['FFT', plotDict] )
            pass
        
        # IF POLARFFTSUM
        if np.any(self.imageSum) and self.plotDict.has_key('polarFFTSum') and bool( self.plotDict['polarFFTSum'] ):
            print( "zorro.plot.PolarFFTSum" )
            plotDict = self.plotDict.copy()
            try:
                plotDict['pixelsize'] = self.pixelsize * binning
                plotDict['image'] = imageSumBinned
            except:
                binning = 2
                plotDict['pixelsize'] = self.pixelsize * binning
                imageSumBinned = util.magickernel( self.getSumCropToLimits(), k=1 )
                plotDict['image'] = imageSumBinned
                
            if bool(self.savePNG):
                plotDict['plotFile'] = os.path.join( self.files['figurePath'], self.plotDict['title'] + "_polarFFTSum.png")
                self.files['figPolarFFTSum'] = plotDict['plotFile']
            plotArgs.append( ['polarFFT', plotDict] )
            pass
        
        # IF TRANSLATIONS
        if np.any(self.translations) and self.plotDict.has_key('translations') and bool( self.plotDict['translations'] ):
            print( "zorro.plot.Translations" )
            plotDict = self.plotDict.copy()
            if np.any( self.translations ):
                plotDict['translations'] = self.translations
                try:
                    plotDict['errorX'] = self.errorDictList[0]['errorX']
                    plotDict['errorY'] = self.errorDictList[0]['errorY']
                except: pass
                if bool(self.savePNG):
                    plotDict['plotFile'] = os.path.join( self.files['figurePath'], self.plotDict['title'] + "_translations.png")
                    self.files['figTranslations'] = plotDict['plotFile']
                plotArgs.append( ['translations', plotDict] )  
                
        # IF PIXEL REGISTRATION ERROR
        if len(self.errorDictList) > 0 and self.plotDict.has_key('pixRegError') and bool( self.plotDict['pixRegError'] ):
            print( "zorro.plot.PixRegError" )
            plotDict = self.plotDict.copy()
            plotDict['errorXY'] = self.errorDictList[0]['errorXY']
            if bool(self.savePNG):
                plotDict['plotFile'] = os.path.join( self.files['figurePath'], self.plotDict['title'] + "_pixRegError.png")
                self.files['figPixRegError'] = plotDict['plotFile']
            plotArgs.append( ['pixRegError', plotDict] )  
            
        # IF CORRTRIMAT
        if len(self.errorDictList) > 0 and self.plotDict.has_key('corrTriMat') and bool( self.plotDict['corrTriMat'] ):
            print( "zorro.plot.coor" )
            plotDict = self.plotDict.copy()
            plotDict['corrTriMat'] = self.errorDictList[-1]['corrTriMat']
            if bool(self.savePNG):
                plotDict['plotFile'] = os.path.join( self.files['figurePath'], self.plotDict['title'] + "_corrTriMat.png")
                self.files['figCorrTriMat'] = plotDict['plotFile']
            plotArgs.append( ['corrTriMat', plotDict] )  
            
        # IF PEAKSIGTRIMAT
        if len(self.errorDictList) > 0 and self.plotDict.has_key('peaksigTriMat') and bool( self.plotDict['peaksigTriMat'] ):
            print( "zorro.plot.peaksig" )
            plotDict = self.plotDict.copy()
            plotDict['peaksigTriMat'] = self.errorDictList[-1]['peaksigTriMat']
            if bool(self.savePNG):
                plotDict['plotFile'] = os.path.join( self.files['figurePath'], self.plotDict['title'] + "_peaksigTriMat.png")
                self.files['figPeaksigTriMat'] = plotDict['plotFile']
            plotArgs.append( ['peaksigTriMat', plotDict] )  
            
        # IF LOGISTICS CURVE        
        if len(self.errorDictList) > 0 and self.plotDict.has_key('logisticWeights') and bool( self.plotDict['logisticWeights'] ):
            print( "zorro.plot.logist" )
            plotDict = self.plotDict.copy()
            if self.weightMode == 'autologistic' or self.weightMode == 'logistic':
                plotDict['peaksigThres'] = self.peaksigThres
                plotDict['logisticK'] = self.logisticK
                plotDict['logisticNu'] = self.logisticNu
            plotDict['errorXY'] = self.errorDictList[0]["errorXY"]
            plotDict['peaksigVect'] = self.errorDictList[0]["peaksigTriMat"][ self.errorDictList[0]["peaksigTriMat"] > 0.0  ]
            
            if self.errorDictList[0].has_key( 'cdfPeaks' ):
               plotDict['cdfPeaks'] = self.errorDictList[0]['cdfPeaks']
               plotDict['hSigma'] = self.errorDictList[0]['hSigma']
            if bool(self.savePNG):
                plotDict['plotFile'] = os.path.join( self.files['figurePath'], self.plotDict['title'] + "_logisticWeights.png")
                self.files['figLogisticWeights'] = plotDict['plotFile']
            plotArgs.append( ['logisticWeights', plotDict] )
             
        # IF LAZY FRC PLOT
        if np.any(self.FRC) and self.plotDict.has_key('lazyFRC') and bool( self.plotDict['lazyFRC'] ):
            print( "zorro.plot.FRC" )
            plotDict = self.plotDict.copy()
            plotDict['FRC'] = self.FRC
            plotDict['pixelsize'] = self.pixelsize
            if bool(self.savePNG):
                plotDict['plotFile'] = os.path.join( self.files['figurePath'], self.plotDict['title'] + "_lazyFRC.png")
                self.files['figLazyFRC'] = plotDict['plotFile']
            plotArgs.append( ['lazyFRC', plotDict] )
            
        # IF CTF4DIAG PLT
        if np.any(self.CTF4Diag) and self.plotDict.has_key('CTF4Diag') and bool( self.plotDict['CTF4Diag'] ):
            print( "zorro.plot.CTF4" )
            plotDict = self.plotDict.copy()
            
            plotDict['CTF4Diag'] = self.CTF4Diag
            plotDict['CTF4Results'] = self.CTF4Results
            plotDict['pixelsize'] = self.pixelsize
            if bool(self.savePNG):
                plotDict['plotFile'] = os.path.join( self.files['figurePath'], self.plotDict['title'] + "_CTF4Diag.png")
                print( "CTF4.plotFile = " + str(plotDict['plotFile']) )
                self.files['figCTF4Diag'] = plotDict['plotFile']
            plotArgs.append( ['CTF4Diag', plotDict] )
            
        # IF STATS PLOT
        if self.plotDict.has_key('stats') and bool( self.plotDict['stats'] ):
            print( "zorro.plot.stats" )
            plotDict = self.plotDict.copy()
            plotDict['pixelsize'] = self.pixelsize
            plotDict['voltage'] = self.voltage
            plotDict['C3'] = self.C3
            if len( self.errorDictList ) > 0 and self.errorDictList[-1].has_key('peaksigTriMat'):
                peaksig = self.errorDictList[-1]['peaksigTriMat']
                peaksig = peaksig[ peaksig > 0.0 ]
                plotDict['meanPeaksig'] = np.mean( peaksig )
                plotDict['stdPeaksig'] = np.std( peaksig )
            if np.any( self.CTF4Results ):
                plotDict['CTF4Results'] = self.CTF4Results
        
            if bool(self.savePNG):
                plotDict['plotFile'] = os.path.join( self.files['figurePath'], self.plotDict['title'] + "_Stats.png")
                self.files['figStats'] = plotDict['plotFile']
            plotArgs.append( ['stats', plotDict] )
                
        ######
        #Multiprocessing pool (to speed up matplotlib's slow rendering and hopefully remove polling loop problems)
        #####
        # I could just disable multiprocessing on NT os?  if os.name != 'nt':
        #for J, item in enumerate(plotArgs):
        #    print( "################  %d  ############"%J )
        #    print( "Plot type: " + item[0] )
        #    print( item[1] )
        
        print( "TODO: PROTECT WINDOWS FROM MULTIPROCESSING MODULE" )
        figPool = mp.Pool( processes=self.n_threads )
        output = figPool.map( plot.generate, plotArgs )
        figPool.close()

        print( output )
        
            
    
    # TODO: change each plot to be a subfunction, and can return a figure, which we can then send back to MplCanvas
    def plotOLD( self, errIndex = -1,  title=None,
                imFilt = scipy.ndimage.gaussian_filter, imFiltArg = 1.0, 
                interpolation='nearest', 
                graph_cm = 'gnuplot', image_cm = 'gray', 
                PDFHandle = None, backend='Qt4Agg' ):
        """
        Plots a report of all relevant parameters for the given errorDictionary index 
        (the default of -1 is the last-most error dictionary in the list).
        
        Which plots are generated is controlled by plotDict.  Build one with zorro.buildPlotDict
        
        Can also produce a PDF report, with the given filename, if self.savePDF = True.  
        More common is to produce individual PNGs.
        """
        import os
        try: os.umask( 0002 ) # Why is Python not using default umask from OS?
        except: pass
    

        if self.files['figurePath'] == None:
            self.files['figurePath'] = '.'
        if not os.path.isdir( os.path.realpath(self.files['figurePath']) ):
            os.mkdir( os.path.realpath(self.files['figurePath']) )
            
        plt.switch_backend( backend )
    
        try: # For Linux, use FreeSerif
            plt.rc('font', family='FreeSerif', size=16)
        except:
            try: 
                plt.rc( 'font', family='serif', size=16)
            except: pass
        
        if self.savePDF:
            from matplotlib.backends.backend_pdf import PdfPages
            # PDFHandle should be a PdfPages object
            if PDFHandle is not None:
                pp = PDFHandle
            else:
                pp = PdfPages( os.path.join( self.files['figurePath'], self.files['stack'] + ".pdf") )
                
        if title is None:
            title = os.path.splitext( os.path.basename( self.files['stack'] ) )[0]
            
        
        if self.plotDict["imageSum"] and bool( np.any(self.imageSum) ):
            summage = self.getSumCropToLimits()
            if imFilt is not None:
                summage = imFilt( summage, imFiltArg )
            clim = util.histClim( summage, cutoff=1E-4 )
            fig1 = plt.figure()
            if self.pixelsize is None:
                plt.imshow( summage, vmin=clim[0], vmax=clim[1], interpolation=interpolation )
            else:
                pltmage = plt.imshow( summage, vmin=clim[0], vmax=clim[1],  
                            interpolation=interpolation  )
                util.plotScalebar( pltmage, self.pixelsize )       
            plt.set_cmap( image_cm )
            plt.title( "Image sum: " + title )
            plt.axis('off')
            
            if self.savePNG: 
                self.files["figImageSum"] = os.path.join( self.files['figurePath'], title + "_imageSum.png")
                plt.savefig( self.files["figImageSum"], transparent=self.plotDict['Transparent'], bbox_inches='tight', dpi=self.plotDict['dpi'] )
            if self.savePDF: pp.savefig(dpi=self.plotDict['dpi'])
            if self.savePNG or self.savePDF: plt.close(fig1)
                
            if bool( np.any(self.filtSum) ):
                filtmage = self.getFiltSumCropToLimits()
                # NO additional filtering of filtSum allowed
                clim = util.histClim( filtmage, cutoff=1E-4 )
                fig1A = plt.figure()
                if self.pixelsize is None:
                    plt.imshow( filtmage, vmin=clim[0], vmax=clim[1], interpolation=interpolation )
                else:
                    pltmage = plt.imshow( filtmage, vmin=clim[0], vmax=clim[1],  
                                interpolation=interpolation  )
                    util.plotScalebar( pltmage, self.pixelsize )       
                plt.set_cmap( image_cm )
                plt.title( "Dose filtered: " + title )
                plt.axis('off')
                
                if self.savePNG: 
                    self.files["figFiltSum"] = os.path.join( self.files['figurePath'], title + "_filtSum.png")
                    plt.savefig( self.files["figFiltSum"], transparent=self.plotDict['Transparent'], bbox_inches='tight', dpi=self.plotDict['dpi'] )
                if self.savePDF: pp.savefig(dpi=self.plotDict['dpi'])
                if self.savePNG or self.savePDF: plt.close(fig1A)
                
        if self.plotDict["FFTSum"] and bool( np.any(self.imageSum) ):
            # Rather than use pyFFTW here I'll just use numpy fft
            self.FFTSum = np.abs( np.fft.fftshift( np.fft.fft2(self.imageSum) ) )
            
            figFFT = plt.figure()
            if self.pixelsize is None:
                plt.imshow( np.log10(self.FFTSum + 1.0), interpolation=interpolation )
            else:
                pixelsize_inv = 1.0 / (self.imageSum.shape[0] * self.pixelsize)
                pltmage = plt.imshow( np.log10(self.FFTSum + 1.0),  
                            interpolation=interpolation  )
                util.plotScalebar( pltmage, pixelsize_inv, units="nm^{-1}" ) 
            plt.set_cmap( image_cm )
            plt.title( "FFT: " + title )
            plt.axis('off')
            
            if self.savePNG: 
                self.files["figFFTSum"] = os.path.join( self.files['figurePath'], title + "_FFTSum.png")
                plt.savefig( self.files["figFFTSum"], transparent=self.plotDict['Transparent'], bbox_inches='tight', dpi=self.plotDict['dpi'] )
            if self.savePDF: pp.savefig(dpi=self.plotDict['dpi'])
            if self.savePNG or self.savePDF: plt.close(figFFT)
            pass
        if self.plotDict["polarFFTSum"] and bool( np.any(self.imageSum) ):
            if self.FFTSum is None:
                self.FFTSum = np.abs( np.fft.fftshift( np.fft.fft2(self.imageSum) ) )
            polarFFTSum = util.img2polar( self.FFTSum )
            
            figPolarFFT = plt.figure()
            if self.pixelsize is None:
                plt.imshow( np.log10(polarFFTSum + 1.0),  interpolation=interpolation )
            else:
                pixelsize_inv = 1.0 / (self.imageSum.shape[0] * self.pixelsize)
                pltmage = plt.imshow( np.log10(polarFFTSum + 1.0), 
                            interpolation=interpolation  )
                util.plotScalebar( pltmage, pixelsize_inv, units="nm^{-1}" ) 
            plt.set_cmap( image_cm )
            plt.title( "Polar FFT: " + title )
            plt.axis('off')

            if self.savePNG: 
                self.files["figPolarFFTSum"] = os.path.join( self.files['figurePath'], title + "_polarFFTSum.png")
                plt.savefig( self.files["figPolarFFTSum"], transparent=self.plotDict['Transparent'], bbox_inches='tight', dpi=self.plotDict['dpi'] )
            if self.savePDF: pp.savefig(dpi=self.plotDict['dpi'])
            if self.savePNG or self.savePDF: plt.close(figPolarFFT)
            pass
        if self.plotDict["imageFirst"] and bool( np.any(self.images) ):     
            firstmage = self.images[0,:,:]
            dose_mean = np.mean(firstmage)
            dose_std = np.std(firstmage)
                
            clim = util.histClim( firstmage, cutoff=1E-4 )
            fig2 = plt.figure()
            if self.pixelsize is None:
                pltmage = plt.imshow( firstmage, vmin=clim[0], vmax=clim[1], interpolation=interpolation  )
            else:
                pltmage = plt.imshow( firstmage, vmin=clim[0], vmax=clim[1],
                            interpolation=interpolation )
                util.plotScalebar( pltmage, self.pixelsize ) 
            plt.set_cmap( image_cm )
            plt.title( "First image dose: %.2f"%dose_mean + " +/- %.2f"%dose_std )
            plt.axis('off')

            if self.savePNG: 
                self.files["figImageFirst"] = os.path.join( self.files['figurePath'], title + "_imageFirst.png")
                plt.savefig( self.files["figImageFirst"], transparent=self.plotDict['Transparent'], bbox_inches='tight', dpi=self.plotDict['dpi'] )
            if self.savePDF: pp.savefig(dpi=self.plotDict['dpi'])
            if self.savePNG or self.savePDF: plt.close(fig2)
                
        try:
            corrmat = self.errorDictList[-1][ 'corrTriMat' ]
        except: pass
    
        if self.plotDict["corrTriMat"]:
            try:
                # plot from the last-most error dictionary
                fig3 = plt.figure()
                corrmat = self.errorDictList[-1][ 'corrTriMat' ]
                clim = [np.min(corrmat[corrmat>0.0]) * 0.75, np.max(corrmat[corrmat>0.0])]
                plt.imshow( corrmat, interpolation="nearest", vmin=clim[0], vmax=clim[1] )
                plt.xlabel( "Base image" )
                plt.ylabel( "Template image" )
                plt.colorbar()
                plt.title( "Maximum correlation upper-tri matrix" )
                plt.set_cmap( graph_cm )

                if self.savePNG: 
                    self.files["figCorrTriMat"] = os.path.join( self.files['figurePath'], title + "_corrTriMat.png")
                    plt.savefig( self.files["figCorrTriMat"], transparent=self.plotDict['Transparent'], bbox_inches='tight', dpi=self.plotDict['dpi'] )
                if self.savePDF: pp.savefig()
                if self.savePNG or self.savePDF: plt.close(fig3)
            except:
                plt.close(fig3)
        if self.plotDict["peaksigTriMat"]:
            try:
                # plot from the last-most error dictionary
                # 
                fig3B = plt.figure()
                peaksig = self.errorDictList[-1][ 'peaksigTriMat' ]
                clim = [np.min(peaksig[peaksig>0.0])*0.75, np.max(peaksig[peaksig>0.0])]
                # plt.matshow( peaksig, interpolation="nearest", norm=LogNorm(vmin=clim[0], vmax=clim[1]) )
                plt.imshow( peaksig, interpolation="nearest", vmin=clim[0], vmax=clim[1] )
                plt.xlabel( "Base image" )
                plt.ylabel( "Template image" )
                plt.colorbar()
                plt.title( "Peak significance upper-tri matrix" )
                plt.set_cmap( graph_cm )

                if self.savePNG: 
                    self.files["figPeaksigTriMat"] = os.path.join( self.files['figurePath'], title + "_peaksigTriMat.png")
                    plt.savefig( self.files["figPeaksigTriMat"], transparent=self.plotDict['Transparent'], bbox_inches='tight', dpi=self.plotDict['dpi'] )
                if self.savePDF: pp.savefig()
                if self.savePNG or self.savePDF: plt.close(fig3B)
            except:
                plt.close(fig3B)
        if self.plotDict["shiftsTriMat"]:  
            try:
                from matplotlib.colors import SymLogNorm
                [fig4, (ax1, ax2)] = plt.subplots( ncols = 2 )
                
                from mpl_toolkits.axes_grid1 import make_axes_locatable 
                shiftsX = self.errorDictList[errIndex][ 'shiftsTriMat' ][:,:,1]
                shiftsY = self.errorDictList[errIndex][ 'shiftsTriMat' ][:,:,0]
                
                climX = util.histClim( shiftsX[corrmat>0.0], cutoff=1E-3 )
                im1 = ax1.imshow( shiftsX, interpolation="nearest", norm=SymLogNorm( 1.0, vmin=climX[0], vmax=climX[1])  )
                ax1.axis('image')
                ax1.set_xlabel( "Base image" )
                ax1.set_ylabel( "Template image" )
                divider1 = make_axes_locatable(ax1)
                cax1 = divider1.append_axes("right", size="20%", pad=0.05)
                plt.colorbar( im1, cax=cax1 )
                plt.set_cmap( graph_cm )
                ax1.set_title( "X-axis drift matrix" )
                
                climY = util.histClim( shiftsY[corrmat>0.0], cutoff=1E-3 )
                im2 = ax2.imshow( shiftsY, interpolation="nearest", norm=SymLogNorm( 1.0, vmin=climY[0], vmax=climY[1])  )
                ax2.axis('image')
                ax2.set_xlabel( "Base image" )
                ax2.set_ylabel( "Template image" )
                divider2 = make_axes_locatable(ax2)
                cax2 = divider2.append_axes("right", size="20%", pad=0.05)
                plt.colorbar( im2, cax=cax2 )
                plt.set_cmap( graph_cm )
                ax2.set_title( "Y-axis drift matrix" )

                if self.savePNG: 
                    self.files["figShiftsTriMat"] = os.path.join( self.files['figurePath'], title + "_shiftsTriMat.png")
                    plt.savefig( self.files["figShiftsTriMat"], transparent=self.plotDict['Transparent'], bbox_inches='tight', dpi=self.plotDict['dpi'] )
                if self.savePDF: pp.savefig()
                if self.savePNG or self.savePDF: plt.close(fig4)
            except:
                plt.close(fig4)
        if self.plotDict["errorTriMat"]:
            try:
                fig5 = plt.figure()
                errorUnraveled = self.errorDictList[-1][ 'errorUnraveled' ]
                clim = util.histClim( errorUnraveled[corrmat>0.0], cutoff=1E-2 )
                plt.imshow( errorUnraveled, interpolation="nearest", vmin=clim[0], vmax=clim[1] )
                plt.xlabel( "Base image" )
                plt.ylabel( "Template image" )
                plt.colorbar()
                plt.set_cmap( graph_cm )
                plt.title( "Pixel registration error matrix" )

                if self.savePNG: 
                    self.files["figPixRegError"] = os.path.join( self.files['figurePath'], title + "_pixRegError.png")
                    plt.savefig( self.files["figPixRegError"], transparent=self.plotDict['Transparent'], bbox_inches='tight', dpi=self.plotDict['dpi'] )
                if self.savePDF: pp.savefig()
                if self.savePNG or self.savePDF: plt.close(fig5)
            except:
                plt.close(fig5)
        
        if self.plotDict["translations"] and bool( np.any(self.translations) ):
            fig6 = plt.figure()
            plt.rc('lines', linewidth=2.0, markersize=16.0 )
            if self.triMode == 'tri':
                try:
                    firstTranslations = -self.errorDictList[errIndex][ 'shiftsTriMat' ][0,:,:].squeeze()
                    plt.plot( firstTranslations[:,1], firstTranslations[:,0], 'r.-' )
                except (KeyError,IndexError):
                    pass
            try:
                plt.errorbar( self.translations[:,1], self.translations[:,0], fmt='k-', 
                             xerr = self.errorDictList[errIndex]['errorX'], yerr=self.errorDictList[errIndex]['errorY']  )
            except :
                plt.plot( self.translations[:,1], self.translations[:,0], 'k.-' )
            if self.triMode == 'tri':
                plt.legend( ( "Align to first frame", "Align to corrTriMat" ), loc='lower right' )
            else:
                plt.legend( ( "Align to corrTriMat", ), loc='lower right' )
            plt.set_cmap( graph_cm )
            plt.xlabel( 'X-axis drift (pix)' )
            plt.ylabel( 'Y-axis drift (pix)' )
            plt.title( "Drift estimate" )

            if self.savePNG: 
                self.files["figTranslations"] = os.path.join( self.files['figurePath'], title + "_translations.png")
                plt.savefig( self.files["figTranslations"], transparent=self.plotDict['Transparent'], bbox_inches='tight', dpi=self.plotDict['dpi'] )
            if self.savePDF: pp.savefig()
            if self.savePNG or self.savePDF: plt.close(fig6)
        if self.plotDict["pixRegError"]:
            try:
                fig7 = plt.figure()
                errorXY = np.abs( self.errorDictList[errIndex]['errorXY'] )
                
                acceptedEqns = self.errorDictList[errIndex]['acceptedEqns']
                meanErrorY = np.mean( errorXY[acceptedEqns,0] )
                stdErrorY = np.std( errorXY[acceptedEqns,0] )
                meanErrorX = np.mean( errorXY[acceptedEqns,1] )
                stdErrorX = np.std( errorXY[acceptedEqns,1] )
                M = self.errorDictList[errIndex]['M']
                
                plt.rc('lines', linewidth=1.5, markersize=6.0 )
                plt.subplot( '211' )
                try:
                    plt.semilogy( np.arange(0,M), self.errorDictList[errIndex]['stdThreshold'][0]*np.ones_like(np.arange(0,M)), '-r' )
                except (KeyError,IndexError):
                    pass
                try: 
                    plt.semilogy( np.arange(0,M), self.pixErrThres*np.ones_like(np.arange(0,M)), '-g' )
                except (KeyError,IndexError):
                    pass

                plt.semilogy( np.arange(0,M), errorXY[:,0], 'ok:' )
                plt.xlabel( 'Equation number' )
                plt.ylabel( 'Drift error estimate (pix) ' )
                plt.title( "RMS Y-error estimate: %.2f"%meanErrorY + " +/- %.2f"%stdErrorY  + " pixels" )
                
                plt.subplot( '212' )
                try:
                    plt.semilogy( np.arange(0,M), self.errorDictList[errIndex]['stdThreshold'][1]*np.ones_like(np.arange(0,M)), '-b' )
                except (KeyError,IndexError):
                    pass
                try: 
                    plt.semilogy( np.arange(0,M), self.pixErrThres*np.ones_like(np.arange(0,M)), '-g' )
                except (KeyError,IndexError):
                    pass

                plt.semilogy( np.arange(0,M), errorXY[:,1], 'ok:' )
                plt.xlabel( 'Equation number' )
                plt.ylabel( 'Drift error estimate (pix) ' )
                plt.title( "RMS X-error estimate: %.2f"%meanErrorX + " +/- %.2f"%stdErrorX  + " pixels" )
                plt.set_cmap( graph_cm )
                
                if self.savePNG: 
                    self.files["figErrorPlots"] = os.path.join( self.files['figurePath'], title + "_errorPlots.png")
                    plt.savefig(  self.files["figErrorPlots"], transparent=self.plotDict['Transparent'], bbox_inches='tight', dpi=self.plotDict['dpi'] )
                if self.savePDF: pp.savefig()
                if self.savePNG or self.savePDF: plt.close(fig7)
            except:
                plt.close(fig7)
        

        if self.plotDict["CTF4Diag"] and bool( np.any(self.CTF4Diag) ):    
            fig8 = plt.figure()
            try: # Sav
                plt.imshow( self.CTF4Diag )
                plt.set_cmap( image_cm )
                plt.title(r"CTFFIND4: " + title +"\n $DF_1 = %.f"%self.CTF4Results[1] 
                    + " \AA,  DF_2 = %.f"%self.CTF4Results[2] 
                    +" \AA, R = %.4f"%self.CTF4Results[5] + ", Max fit: %.1f"%self.CTF4Results[6] + " \AA$"  )
                
                if self.savePNG: 
                    self.files["figCTF4Diag"] = os.path.join( self.files['figurePath'], title + "_CTF4Diag.png")
                    plt.savefig( self.files["figCTF4Diag"], transparent=self.plotDict['Transparent'], bbox_inches='tight', dpi=self.plotDict['dpi'] )
                if self.savePDF: pp.savefig()
                if self.savePNG or self.savePDF: plt.close(fig8)
            except:
                plt.close(fig8)
        if self.plotDict["logisticWeights"]:    
            try: # peak sig versus pixel error for logistics weighting
                (fig9, ax1) = plt.subplots()
                errorXY = self.errorDictList[0]["errorXY"]
                pixError = np.sqrt( errorXY[:,0]**2 + errorXY[:,1]**2 )
                peaksigVect = self.errorDictList[0]["peaksigTriMat"][ self.errorDictList[0]["peaksigTriMat"] > 0.0  ]
                
                # Mixing a log-plot with a linear-plot in a plotyy style.
                ax1.semilogy( peaksigVect, pixError, 'k.' )
                # ax1.plot( peaksigVect, pixError, 'k.' )
                ax1.set_xlabel( 'Correlation peak significance, $\sigma$' )
                ax1.set_ylabel( 'Pixel registration error' )
                ax1.set_ylim( [0,1] )
                ax1.set_ylim( [1E-2, 1E2] )
                ax1.set_xlim( peaksigVect.min(), peaksigVect.max() )
                
                ax2 = ax1.twinx()
                if self.weightMode == 'logistic' or self.weightMode == 'autologistic':
                    # Plot threshold sigma value
                    ax2.plot( [self.peaksigThres, self.peaksigThres], [0.0, 1.0], '--', 
                             color='firebrick', label=r'$\sigma_{thres} = %.2f$'%self.peaksigThres )
                    
                    # Plot the logistics curve
                    peakSig = np.arange( np.min(peaksigVect), np.max(peaksigVect), 0.05 )
                    
                    weights = util.logistic( peakSig, self.peaksigThres, self.logisticK, self.logisticNu )
                    # ax2.semilogy( peakSig, weights, label=r"Weights $K=%.2f$, $\nu=%.3f$"%( self.logisticK, self.logisticNu), color='royalblue' )
                    ax2.plot( peakSig, weights, label=r"Weights $K=%.2f$, $\nu=%.3f$"%( self.logisticK, self.logisticNu), color='royalblue' )
                    
                    # Add CDF as scatter plot if it was calculated, also on the second axes
                    if self.errorDictList[0].has_key( 'cdfPeaks' ):
                        #ax2.semilogy( self.errorDictList[0]['hSigma'], self.errorDictList[0]['cdfPeaks'], '.', label = r'$\sigma-$CDF', color='slateblue' )
                        ax2.plot( self.errorDictList[0]['hSigma'], self.errorDictList[0]['cdfPeaks'], '+', label = r'$\sigma-$CDF', color='slateblue' )
                        
                    ax2.set_ylim( [0,1] )
                    # ax2.set_ylim( [1E-2, 1E2] )
                    ax2.set_ylabel( "Weights / CDF", color='royalblue' )
                    
                    for t1 in ax2.get_yticklabels():
                        t1.set_color( 'royalblue' )
                else:
                    pass
                plt.legend( loc='best', fontsize=14 )
                
                if self.savePNG: 
                    self.files["figLogisticWeights"] = os.path.join( self.files['figurePath'], title + "_logisticWeights.png") 
                    plt.savefig( self.files["figLogisticWeights"], transparent=self.plotDict['Transparent'], bbox_inches='tight', dpi=self.plotDict['dpi'] )
                if self.savePDF: pp.savefig()
                if self.savePNG or self.savePDF: plt.close(fig9)
            except:
                plt.close(fig9)
                
        if self.plotDict["lazyFRC"] and bool( np.any(self.FRC) ):
            try:
                (fig10, ax1) = plt.subplots()
                inv_ps = 1.0 / (2.0* self.FRC.size* self.pixelsize)
                freqAxis = np.arange( self.FRC.size ) * inv_ps
                
                plt.plot( freqAxis, self.FRC, 'k.-', label="Non-independant FRC is \n not a resolution estimate" )
                ax1.set_xlabel( r"Spatial frequency, $q$ ($nm^{-1}$)" )
                ax1.set_xlim( [freqAxis.min(), freqAxis.max()] )
                ax1.set_ylabel( "Fourier ring correlation" )
                ax1.legend( loc='best' )

                if self.savePNG: 
                        self.files["figLazyFRC"] = os.path.join( self.files['figurePath'], title + "_lazyFRC.png") 
                        plt.savefig( self.files["figLazyFRC"], transparent=self.plotDict['Transparent'], bbox_inches='tight', dpi=self.plotDict['dpi'] )
                if self.savePDF: pp.savefig()
                if self.savePNG or self.savePDF: plt.close(fig9)
            except:
                plt.close(fig10)
        ##### END OF PLOT #####
        # Show all plots, not needed in Qt matplotlib backends    
        plt.show( block = False )
        
        if self.savePDF: 
            if PDFHandle is None: pp.close()
            plt.close("all")
        elif self.savePNG:
            plt.close("all")    
            
            
        
    def makeMovie( self, movieName = None, clim = None, frameRate=3, graph_cm = 'gnuplot' ):
        """
        Use FFMPEG to generate movies showing the correlations.  C0 must not be None.
        
        The ffmpeg executable must be in the system path.
        """
        import os

        fex = '.png'
        print( "makeMovie must be able to find FFMPEG on the system path" )
        print( "Strongly recommended to use .mp4 extension" )
        if movieName is None:
            movieName = self.files['stack'] + ".mp4"
        
        m = self.C0.shape[0]
        
        # Turn off display of matplotlib temporarily
        originalBackend = plt.get_backend()
        plt.switch_backend('agg')
        plt.rc('font', family='FreeSerif', size=16)
        corrmat = self.errorDictList[-1][ 'corrTriMat' ]
        climCM = [np.min(corrmat[corrmat>0.0]) * 0.75, np.max(corrmat[corrmat>0.0])]
        # Get non-zero indices from corrmat
        
        # Note that FFMPEG starts counting at 0.  
        for J in np.arange(0,m):
            corrMap = self.C0[J,:,:].copy(order='C')
            
            figCM = plt.figure()
            plt.subplot( '121' )
            # corrTriMat
            plt.imshow( corrmat, interpolation="nearest", vmin=climCM[0], vmax=climCM[1] )
            plt.xlabel( "Base image" )
            plt.ylabel( "Template image" )
            plt.colorbar( orientation='horizontal' )
            plt.title( "Maximum correlation upper-tri matrix" )
            plt.set_cmap( graph_cm )
            # Draw lines (How to unravel J???)
            plt.plot(  )
            plt.plot(  )
            # Reset xlim and ylim
            plt.xlim( [0, corrMap.shape[2]-1] )
            plt.ylim( [0, corrMap.shape[1]-1] )
            
            
            # C0
            plt.subplot( '122' )
            if clim is None:
                plt.imshow( corrMap, interpolation='none' )
            else:
                plt.imshow( corrMap, interpolation='none', vmin=clim[0], vmax=clim[1] )
            plt.set_cmap( graph_cm )
            plt.colorbar( orientation='horizontal' )
            
            # Render and save
            plt.tight_layout()
            plt.pause(0.05)
            plt.savefig( "corrMap_%05d"%J + fex, dpi=self.plotDict['dpi'] )
            plt.close( figCM )
            # corrMap = ( 255.0 * util.normalize(corrMap) ).astype('uint8')
            # Convert to colormap as follows: Image.fromarray( np.uint8( cm.ocean_r(stddesk)*255))
            # skimage.io.imsave( "corrMap_%05d"%J + fex, mage, plugin='freeimage' )
            # skimage.io.imsave( "corrMap_%05d"%J + fex, corrMap )
            pass
        time.sleep(0.5)
        
        # Remove the old movie if it's there
        try: 
            os.remove( movieName )
        except:
            pass
        
        # Make a movie with lossless H.264
        # One problem is that H.264 isn't compatible with PowerPoint.  Can use Handbrake to make it so...
        # Framerate command isn't working...
        comstring = "ffmpeg -r "+str(frameRate)+ " -f image2 -i \"corrMap_%05d"+fex+"\" -c:v libx264 -preset veryslow -qp 0 -r "+str(frameRate)+ " "+movieName
        print( comstring )
        sub = subprocess.Popen( comstring, shell=True )
        sub.wait()
        # os.system( comstring )
        # Clean up
        for J in np.arange(0,m):
            os.remove( "corrMap_%05d"%J + fex )
        pass
        plt.switch_backend(originalBackend)

    def printProfileTimes( self ):
        """ Go through and print out all the profile times in self.t """
        print( "----PROFILING TIMES----" )
        print( "    Load files (s): %.3f"%(self.t[7] - self.t[6]) )
        print( "    Image/mask binning (s): %.3f"%(self.t[14] - self.t[13]) ) 
        print( "    Enter xcorrnm2_tri (s): %.3f"%(self.t[0] - self.t[8]) ) 
        print( "    X-correlation initialization (s): %.3f"%(self.t[1] - self.t[0]) )
        print( "    X-correlation forward FFTs (s): %.3f"%(self.t[2] - self.t[1]) )
        print( "    X-correlation main computation (s): %.3f"%(self.t[3] - self.t[2]) )
        print( "    Complete (entry-to-exit) xcorrnm2_tri (s): %.3f"%(self.t[9] - self.t[8]) ) 
        print( "    Shifts solver (last iteration, s): %.3f"%(self.t[5] - self.t[4]) )
        print( "    Total shifts solver (all iterations, s): %.3f"%(self.t[12] - self.t[11]) )
        print( "    Subpixel alignment (s): %.3f"%(self.t[10] - self.t[9]) ) 
        print( "    lazy Fourier Ring Correlation (s): %.3f"%(self.t[30] - self.t[10]))
        print( "###############################" )
        print( "    Total execution time (s): %.3f"%(self.t[10] - self.t[0]) )
        pass


        
#### COMMAND-LINE INTERFACE ####
if __name__ == '__main__':
    # Get command line arguments
    import sys
    
    # Usage: 
    # python `which zorro.py` -i Test.dm4 -c default.ini -o test.mrc
    stackReg = ImageRegistrator()
    configFile = None
    inputFile = None
    outputFile = None
    
    try: print( "****Running Zorro-command-line on hostname: %s****"%os.uname()[1] )
    except: pass
    
    for J in np.arange(0,len(sys.argv)):
        # First argument is mnxc_solver.py
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
        print( "No input files, outputing template.cfg" )
        stackReg.saveConfig( configNameIn = "template.cfg")
        sys.exit()
    if inputFile == None and not configFile == None:
        stackReg.loadConfig( configNameIn=configFile, loadData = True )
        
    if not inputFile == None and not configFile == None:
        stackReg.loadConfig( configNameIn=configFile, loadData = False )

        stackReg.files['stack'] = inputFile
        stackReg.loadData()
        
    if not outputFile == None:
        stackReg.files['sum'] = outputFile    
        
    # Execute the alignment
    stackReg.alignImageStack()
    
    # Save plots
    if stackReg.savePNG:
        stackReg.plot()
    # Save everthing and do rounding/compression operations
    stackReg.saveData() # Can be None 
    stackReg.METAstatus = 'fini'
    stackReg.saveConfig()
    
    print( "Zorro exiting" )
    sys.exit()

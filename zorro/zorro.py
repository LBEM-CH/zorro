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
from __future__ import division, print_function, absolute_import, unicode_literals

import numpy as np
if np.version.version.split('.')[1] == 7:
    print( "WARNING: NUMPY VERSION 1.7 DETECTED, ZORRO IS DESIGNED FOR >1.10" )
    print( "CHECK YOUR ENVIRONMENT VARIABLES TO SEE IF EMAN2 HAS HIJACKED YOUR PYTHON DISTRIBUTION" )
    
import numexprz as nz
# Now see which numexpr we have, by the dtype of float (whether it casts or not)
try:
    # Now see which numexpr we have, by the dtype of float (whether it casts or not)
    tdata = np.complex64( 1.0 + 2.0j )
    fftw_dtype = nz.evaluate( 'tdata + tdata' ).dtype
    float_dtype = nz.evaluate( 'real(tdata+tdata)' ).dtype
except: 
    fftw_dtype = 'complex128'
    float_dtype = 'float64'


import scipy.optimize
import scipy.ndimage
import scipy.stats
import time
try:
    import ConfigParser as configparser
except:
    import configparser # Python 3

# Here we have to play some games depending on where the file was called from
# with the use of absolute_import
# print( "__name__ of zorro: " + str(__name__) )
try:
    import zorro_util as util
    import zorro_plotting as plot
except ImportError:
    from . import zorro_util as util
    from . import zorro_plotting as plot
    
import mrcz
    
import os, os.path, tempfile, sys
import subprocess

# Should we disable Multiprocessing on Windows due to general bugginess in the module?
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

# Numpy.pad is bad at dealing with interpreted strings
if sys.version_info >= (3,0):
    symmetricPad = u'symmetric'
    constantPad = u'constant'
else: 
    symmetricPad = b'symmetric'
    constantPad = b'constant'

#### OBJECT-ORIENTED INTERFACE ####
class ImageRegistrator(object):
# Should be able to handle differences in translation, rotation, and scaling
# between images
    
    def __init__( self ):
        # Declare class members
        self.verbose = 0
        self.umask = 2
        
        # Meta-information for processing, not saved in configuration files.
        self.METApriority = 0.0
        self.METAstatus = u'new'
        self.METAmtime = 0.0
        self.METAsize = 0
        
        self.xcorrMode = 'zorro' # 'zorro', 'unblur v1.02', 'motioncorr v2.1'
        # FFTW_PATIENT is bugged for powers of 2, so use FFTW_MEASURE as default
        self.fftw_effort = u"FFTW_MEASURE"
        # TODO: change this to drop into cachePath
        
        self.n_threads = nz.nthreads # Number of cores to limit FFTW to, if None uses all cores 
        self.cachePath = tempfile.gettempdir()
          
        # CALIBRATIONS
        self.pixelsize = None # Typically we use nanometers, the same units as Digital Micrograph
        self.voltage = 300.0 # Accelerating voltage, kV
        self.C3 = 2.7 # Spherical aberration of objective, mm
        self.gain = None
        self.detectorPixelSize = None # Physical dimensions of detector pixel (5 um for K2)

        # Timings
        self.bench = {} # Dict holds various benchmark times for the code
        self.saveC = False # Save the cross-correlation within +/- maxShift
        
        # INFORMATION REDUCTION
        # The SNR at high spatial frequencies tends to be lower due to how information transfer works, so 
        # removing/filtering those frequencies can improve stability of the registration.  YMMV, IMHO, etc.
        
        self.Brad = 512 # Gaussian low-pass applied to data before registration, units are radius in Fourier space, or equivalent point-spread function in real-space
        self.Bmode = u'opti' # can be a real-space Gaussian convolution, 'conv' or Fourier filter, 'fourier', or 'opti' for automatic Brad
        # For Bmode = 'fourier', a range of available filters can be used: gaussian, gauss_trunc, butterworth.order (order is an int), hann, hamming
        self.BfiltType = u'gaussian'
        self.fouCrop = [3072,3072] # Size of FFT in frequency-space to crop to (e.g. [2048,2048])
        self.reloadData = True
        
        # Data
        self.images = None
        self.imageSum = None
        self.filtSum = None # Dose-filtered, Wiener-filtered, etc. representations go here
        self.gainRef = None # For application of gain reference in Zorro rather than Digital Micrograph/TIA/etc.
        self.gainInfo = { 
            "Horizontal": True, "Vertical": True, "Diagonal":False,
            "GammaParams": [ 0.12035633, -1.04171635, -0.03363192,  1.03902726],
        }
        
        # One of None,  'dose', 'dose,background', 'dosenorm', 'gaussLP', 'gaussLP,background'
        # also 'hot' can be in the comma-seperated list for pre-filtering of hot pixels
        self.filterMode = None 
        # Dose filt param = [dosePerFrame, critDoseA, critDoseB, critDoseC, cutoffOrder, missingStartFrame]
        self.doseFiltParam = [None, 0.24499, -1.6649, 2.8141, 32, 0]
        # for 'hot' in filterMode
        self.hotpixInfo = { u"logisticK":6.0, u"relax":0.925, u"maxSigma":8.0, u"psf": u"K2",
                           u"guessHotpix":0, u"guessDeadpix":0, u"decorrOutliers":False,
                           u"cutoffLower":-4.0, u"cutoffUpper":3.25, u"neighborPix":0 }
        
        
        self.FFTSum = None
        # If you want to use one mask, it should have dims [1,N_Y,N_X]. This is 
        # to ensure Cythonized code can interact safely with Numpy
        self.incohFouMag = None # Incoherent Fourier magnitude, for CTF determination, resolution checks
        self.masks = None
        self.maskSum = None
        self.C = None
        
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
        
        
        self.doLazyFRC = True
        self.doEvenOddFRC = False
        self.FRC = None # A Fourier ring correlation

        # Filtering
        # TODO: add more fine control over filtering options        
        
        # CTF currently supports CTFFIND4.1 or GCTF
        self.CTFProgram = None # None, "ctffind4.1", or "gctf", 'ctffind4.1,sum' works on (aligned) sum, same for 'gctf,sum'
        self.CTFInfo = { u'DefocusU':None, u'DefocusV': None, u'DefocusAngle':None, u'CtfFigureOfMerit':None,
                         u'FinalResolution': None, u'AmplitudeContrast':0.07, u'AdditionalPhaseShift':None,
                         }
        self.CTFDiag = None # Diagnostic image from CTFFIND4.1 or GCTF
        
        # DEPRICATED ctf stuff
        #self.doCTF = False
        #self.CTF4Results = None # Micrograph number, DF1, DF2, Azimuth, Additional Phase shift, CC, and max spacing fit-to
        #self.CTF4Diag = None
        
        # Registration parameters
        self.shapePadded = [4096,4096]
        self.shapeOriginal = None
        self.shapeBinned = None 
        self.subPixReg = 16 # fraction of a pixel to REGISTER image shift to
        # Subpixel alignment method: None (shifts still registered subpixally), lanczos, or fourier
        # lanczos is cheaper computationally and has fewer edge artifacts
        self.shiftMethod = u'lanczos' 
        self.maxShift = 100 # Generally should be 1/2 distance to next lattice spacing
        # Pre-shift every image by that of the previous frame, useful for high-resolution where one can jump a lattice
        # i.e. should be used with small values for maxShift
        self.preShift = False
        # Solver weighting can be raw max correlation coeffs (None), normalized to [0,1] by the 
        # min and max correlations ('norm'), or 'logistic' function weighted which
        # requires corrThres to be set.
        self.peakLocMode = u'interpolated' # interpolated (oversampled), or a RMS-best fit like fitlaplacian
        self.weightMode = u'autologistic' # autologistic, normalized, unweighted, logistic, or corr
        self.peaksigThres = 6.0
        self.logisticK = 5.0
        self.logisticNu = 0.15
        self.originMode = u'centroid' # 'centroid' or None
        self.suppressOrigin = True # Delete the XC pixel at (0,0).  Only necessary if gain reference is bad, but defaults to on.
        
        # Triangle-matrix indexing parameters
        self.triMode = u'diag' # Can be: tri, diag, auto, first
        self.startFrame = 0
        self.endFrame = 0
        self.diagStart = 0 # XC to neighbour frame on 0, next-nearest neighbour on +1, etc.
        self.diagWidth = 5
        self.autoMax = 10

        self.corrThres = None # Use with 'auto' mode to stop doing cross-correlations if the values drop below the threshold
        
        self.velocityThres = None # Pixel velocity threshold (pix/frame), above which to throw-out frames with too much motion blur.
        
        #### INPUT/OUTPUT ####
        self.files = { u"config":None, u"stack":None, u"mask":None, u"sum":None, 
                       u"align":None, u"figurePath":None, u"xc":None, 
                       u"moveRawPath":None, u"original":None, u"gainRef":None,
                       u"stdout": None, u"automatch":None, u"rejected":None,
                       u"compressor": None, u"clevel": 1 }

        #self.savePDF = False
        self.savePNG = True
        self.saveMovie = True
        self.doCompression = False
        
        self.compress_ext = ".bz2"

        #### PLOTTING ####
        self.plotDict = { u"imageSum":True, u"imageFirst":False, u"FFTSum":True, u"polarFFTSum":True, 
                         u"filtSum":True, u'stats': False,
                         u"corrTriMat":False, u"peaksigTriMat": True, 
                         u"translations":True, u"pixRegError":True, 
                         u"CTFDiag":True, u"logisticWeights": True, u"FRC": True, 
                         u'Transparent': True, u'plot_dpi':144, u'image_dpi':250,
                         u'image_cmap':u'gray', u'graph_cmap':u'gnuplot', 
                         u'fontsize':12, u'fontstyle': u'serif', u'colorbar': True,
                         u'backend': u'Qt4Agg', u'multiprocess':True,
                         u'show':False }
        pass
    
    def initDefaultFiles( self, stackName ):
        self.files[u'stack'] = stackName
        self.files[u'config'] = stackName + u".zor"

        stackPath, stackFront = os.path.split( stackName )
        stackFront = os.path.splitext( stackFront )[0]
        
        if not 'compressor' in self.files or not bool(self.files['compressor']):
            mrcExt = ".mrc"
            mrcsExt = ".mrcs"
        else:
            mrcExt = ".mrcz"
            mrcsExt = ".mrcsz" 
            
        self.files[u'align'] = os.path.relpath( 
                    os.path.join( u"./align", "%s_zorro_movie%s" %(stackFront, mrcsExt)  ), 
                    start=stackPath )
        self.files[u'sum'] = os.path.relpath( stackPath, 
                    os.path.join( u"./sum", "%s_zorro%s" %(stackFront, mrcExt) ), 
                    start=stackPath ) 
        self.files[u'figurePath'] = os.path.relpath( 
                os.path.join(stackPath, u"./figs"), start=stackPath  )

            
    def xcorr2_mc2_1( self, gpu_id = 0, loadResult=True, clean=True ):
        """
        This makes an external operating system call to the Cheng's lab GPU-based 
        B-factor multireference executable. It and CUDA libraries must be on the system 
        path and libary path respectively.
        
        NOTE: Spyder looks loads PATH and LD_LIBRARY_PATH from .profile, not .bashrc
        """
            
        
        
        dosef_cmd = util.which("dosefgpu_driftcorr")
        if dosef_cmd is None:
            print( "Error: dosefgpu_driftcorr not found in system path." )
            return
        
        #tempFileHash = str(uuid.uuid4() ) # Key let's us multiprocess safely
        stackBase = os.path.basename( os.path.splitext( self.files['stack'] )[0] )
        
        if self.cachePath is None:
            self.cachePath = "."
            
        InName = os.path.join( self.cachePath, stackBase + u"_mcIn.mrc" )
        # Unfortunately these files may as well be in the working directory.    
        OutAvName = os.path.join( self.cachePath, stackBase + u"_mcOutAv.mrc" )
        OutStackName = os.path.join( self.cachePath, stackBase + u"_mcOut.mrc" )
        logName = os.path.join( self.cachePath, stackBase + u"_mc.zor" )
        mrcz.writeMRC( self.images, InName )

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
                + " -flg " + logName )

        sub = subprocess.Popen( dosef_cmd + motion_flags, shell=True )
        sub.wait()
        
        
        self.loadMCLog( logName )
            

        time.sleep(0.5)
        if bool(clean):
            try: os.remove(InName)
            except: pass
            try: os.remove(OutStackName)
            except: pass
            try: os.remove(OutAvName)
            except: pass
            try: os.remove(logName)
            except: pass
        
    def loadMCLog( self, logName ):
        """
        Load and part a MotionCorr log from disk using regular expressions.
        """
        import re
        
        # Parse to get the translations
        fhMC = open( logName )
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
        
        if self.originMode == u'centroid':
            centroid = np.mean( self.translations, axis=0 )
            self.translations -= centroid
            
    def xcorr2_unblur1_02( self, dosePerFrame = None, minShift = 2.0, terminationThres = 0.1, 
                      maxIteration=10, verbose=False, loadResult=True, clean=True   ):
        """
        Calls UnBlur by Grant and Rohou using the Zorro interface.
        """
        self.bench['unblur0']  = time.time()
        unblur_exename = "unblur_openmp_7_17_15.exe"
        if util.which( unblur_exename ) is None:
            print( "UnBlur not found in system path" )
            return
        
        print( "Calling UnBlur for " + self.files['stack'] )
        print( "   written by Timothy Grant and Alexis Rohou: http://grigoriefflab.janelia.org/unblur" )
        print( "   http://grigoriefflab.janelia.org/node/4900" )
        
        import os
        
        try: os.umask( self.umask ) # Why is Python not using default umask from OS?
        except: pass
        
        if self.cachePath is None:
            self.cachePath = "."
            
        # Force trailing slashes onto cachePatch
        stackBase = os.path.basename( os.path.splitext( self.files[u'stack'] )[0] )
        frcOutName = os.path.join( self.cachePath, stackBase + u"_unblur_frc.txt" )
        shiftsOutName = os.path.join( self.cachePath, stackBase + u"_unblur_shifts.txt" )
        outputAvName = os.path.join( self.cachePath, stackBase + u"_unblur.mrc" )
        outputStackName = os.path.join( self.cachePath, stackBase + u"_unblur_movie.mrc" )
        

        ps = self.pixelsize * 10.0
        if 'dose' in self.filterMode:
            doDoseFilter = True
            if dosePerFrame == None:
                # We have to guesstimate the dose per frame in e/A^2 if it's not provided
                dosePerFrame = np.mean( self.images ) / (ps*ps)
            preExposure = 0.0
            if 'dosenorm' in self.filterMode:
                restoreNoise=True
            else:
                restoreNoise=False
        else:
            doDoseFilter = False
            
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
            bfac = 1500 # dosef default 'safe' bfactor for mediocre gain reference
        outerShift = self.maxShift * ps
        # RAM: I see no reason to let people change the Fourier cross masking
        vertFouMaskHW = 1
        horzFouMaskHW = 1
        
        try: 
            mrcName = os.path.join( self.cachePath, stackBase + "_unblurIN.mrc" )
            mrcz.writeMRC( self.images, mrcName )
        except:
            print( "Error in exporting MRC file to UnBlur" )
            return
         
        # Are there flags for unblur?  Check the source code.
        flags = "" # Not using any flags
         
        unblurexec = ( unblur_exename + " " + flags + " << STOP_PARSING \n" + mrcName )
        
        unblurexec = (unblurexec + "\n" + str(self.images.shape[0]) + "\n" +
            outputAvName + "\n" + shiftsOutName + "\n" + str(ps) + "\n" +
            str(doDoseFilter) )
            
        if bool(doDoseFilter):
            unblurexec += "\n" + str(dosePerFrame) + "\n" + str(self.voltage) + "\n" + str(preExposure)
            
        unblurexec += ("\n yes \n" + outputStackName + "\n yes \n" + 
            frcOutName + "\n" + str(minShift) + "\n" + str(outerShift) + "\n" +
            str(bfac) + "\n" + str( np.int(vertFouMaskHW) ) + "\n" + str( np.int(horzFouMaskHW) ) + "\n" +
            str(terminationThres) + "\n" + str(maxIteration) )
            
        if bool(doDoseFilter):
            unblurexec += "\n" + str(restoreNoise)
            
        unblurexec += "\n" + str(verbose) 
              
        unblurexec = unblurexec + "\nSTOP_PARSING"
        
        print( unblurexec )
        sub = subprocess.Popen( unblurexec, shell=True )
        sub.wait()
        
        try:
            # Their FRC is significantly different from mine.
            self.FRC = np.loadtxt(frcOutName, comments='#', skiprows=0 )
            self.translations = np.loadtxt( shiftsOutName, comments='#', skiprows=0 ).transpose()
            # UnBlur uses Fortran ordering, so we need to swap y and x for Zorro C-ordering
            self.translations = np.fliplr( self.translations )
            # UnBlur returns drift in Angstroms
            self.translations /= ps
            # UnBlur registers to middle frame
            self.translations -= self.translations[0,:]
            
            if bool( loadResult ):
                print( "Loading UnBlur aligned frames into ImageRegistrator.images" )
                if 'dose' in self.filterMode:
                    # TODO: WHow to get both filtered images and unfiltered?
                    self.imageSum = mrcz.readMRC( outputAvName )[0]
                else:
                    self.imageSum = mrcz.readMRC( outputAvName )[0]
                # TODO: We have a bit of an issue, this UnBlur movie is dose filtered...
                self.images = mrcz.readMRC( outputStackName )[0]
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
        if bool(clean):
            try: os.remove( mrcName )
            except: print( "Could not remove Unblur MRC input file" )
            try: os.remove( frcOutName )
            except: print( "Could not remove Unblur FRC file" )
            try: os.remove( shiftsOutName )
            except: print( "Could not remove Unblur Shifts file" )
            try: os.remove( outputAvName )
            except: print( "Could not remove Unblur MRC average" )
            try: os.remove( outputStackName )
            except: print( "Could not remove Unblur MRC stack" )
        self.bench['unblur1']  = time.time()
        
        
    def __init_xcorrnm2( self, triIndices=None ):
        """
        
        """
        self.bench['xcorr0'] = time.time() 
        
        shapeImage = np.array( [self.images.shape[1], self.images.shape[2]] )
        self.__N = np.asarray( self.images.shape )[0]
            
        if self.preShift:
            print( "Warning: Preshift will break if there are skipped frames in a triIndices row." )

        # Test to see if triIndices is a np.array or use self.triMode
        if hasattr( triIndices, "__array__" ): # np.array
            # Ensure triIndices is a square array of the right size
            if triIndices.shape[0] != self.__N or triIndices.shape[1] != self.__N:
                raise IndexError("triIndices is wrong size, should be of length: " + str(self.__N) )

        elif triIndices is None:
            [xmesh, ymesh] = np.meshgrid( np.arange(0,self.__N), np.arange(0,self.__N) )
            trimesh = xmesh - ymesh
            # Build the triMat if it wasn't passed in as an array
            if( self.triMode == 'first' ):
                print( "Correlating in template mode to first image" )
                triIndices = np.ones( [1,self.__N], dtype='bool' )
                triIndices[0,0] = False # Don't autocorrelate the first frame.
            elif( self.triMode == u'diag' ):
                if (self.diagWidth is None) or (self.diagWidth < 0):
                    # For negative numbers, align the entire triangular matrix
                    self.diagWidth = self.__N
                    
                triIndices = (trimesh <= self.diagWidth + self.diagStart ) * (trimesh > self.diagStart )
                print( "Correlating in diagonal mode with width " + str(self.diagWidth) )
            elif( self.triMode == u'autocorr' ):
                triIndices = (trimesh == 0)
            elif( self.triMode == u'refine' ):
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
            self.masks = np.reshape( self.masks.astype(self.images.dtype), [1,shapeImage[0],shapeImage[1]] )
             
        # Pre-loop allocation
        self.__shiftsTriMat = np.zeros( [self.__N,self.__N,2], dtype=float_dtype ) # Triagonal matrix of shifts in [I,J,(y,x)]
        self.__corrTriMat = np.zeros( [self.__N,self.__N], dtype=float_dtype ) # Triagonal matrix of maximum correlation coefficient in [I,J]
        self.__peaksigTriMat = np.zeros( [self.__N,self.__N], dtype=float_dtype ) # Triagonal matrix of correlation peak contrast level
        self.__originTriMat= np.zeros( [self.__N,self.__N], dtype=float_dtype ) # Triagonal matrix of origin correlation coefficient in [I,J]
        
        # Make pyFFTW objects
        if not bool( np.any( self.fouCrop ) ):
            self.__tempFullframe = np.empty( shapeImage, dtype=fftw_dtype )
            self.__FFT2, self.__IFFT2 = util.pyFFTWPlanner( self.__tempFullframe, wisdomFile=os.path.join( self.cachePath, "fftw_wisdom.pkl" ), effort = self.fftw_effort, n_threads=self.n_threads )
            self.__shapeCropped = shapeImage
            self.__tempComplex = np.empty( self.__shapeCropped, dtype=fftw_dtype )
        else:
            self.__tempFullframe = np.empty( shapeImage,  dtype=fftw_dtype )
            self.__FFT2, _ = util.pyFFTWPlanner( self.__tempFullframe, wisdomFile=os.path.join( self.cachePath, "fftw_wisdom.pkl" ) , effort = self.fftw_effort, n_threads=self.n_threads, doReverse=False )
            # Force fouCrop to multiple of 2
            self.__shapeCropped = 2 * np.floor( np.array( self.fouCrop ) / 2.0 ).astype('int')
            self.__tempComplex = np.empty( self.__shapeCropped, dtype=fftw_dtype )
            _, self.__IFFT2 = util.pyFFTWPlanner( self.__tempComplex, wisdomFile=os.path.join( self.cachePath, "fftw_wisdom.pkl" ) , effort = self.fftw_effort, n_threads=self.n_threads, doForward=False )
        
        self.__shapeCropped2 = (np.array( self.__shapeCropped) / 2.0).astype('int')
        self.__templateImageFFT = np.empty( self.__shapeCropped, dtype=fftw_dtype )
        self.__templateSquaredFFT = np.empty( self.__shapeCropped, dtype=fftw_dtype )
        self.__templateMaskFFT = np.empty( self.__shapeCropped, dtype=fftw_dtype )
        self.__tempComplex2 = np.empty( self.__shapeCropped, dtype=fftw_dtype )
        
        # Subpixel initialization
        # Ideally subPix should be a power of 2 (i.e. 2,4,8,16,32)
        self.__subR = 8 # Sampling range around peak of +/- subR
        if self.subPixReg is None: self.subPixReg = 1;
        if self.subPixReg > 1.0:  
            # hannfilt = np.fft.fftshift( ram.apodization( name='hann', size=[subR*2,subR*2], radius=[subR,subR] ) ).astype( fftw_dtype )
            # Need a forward transform that is [subR*2,subR*2] 
            self.__Csub = np.empty( [self.__subR*2,self.__subR*2], dtype=fftw_dtype )
            self.__CsubFFT = np.empty( [self.__subR*2,self.__subR*2], dtype=fftw_dtype )
            self.__subFFT2, _ = util.pyFFTWPlanner( self.__Csub, fouMage=self.__CsubFFT, wisdomFile=os.path.join( self.cachePath, "fftw_wisdom.pkl" ) , effort = self.fftw_effort, n_threads=self.n_threads, doReverse = False )
            # and reverse transform that is [subR*2*subPix, subR*2*subPix]
            self.__CpadFFT = np.empty( [self.__subR*2*self.subPixReg,self.__subR*2*self.subPixReg], dtype=fftw_dtype )
            self.__Csub_over = np.empty( [self.__subR*2*self.subPixReg,self.__subR*2*self.subPixReg], dtype=fftw_dtype )
            _, self.__subIFFT2 = util.pyFFTWPlanner( self.__CpadFFT, fouMage=self.__Csub_over, wisdomFile=os.path.join( self.cachePath, "fftw_wisdom.pkl" ) , effort = self.fftw_effort, n_threads=self.n_threads, doForward = False )
        
        
        self.__maskProduct = np.zeros( self.__shapeCropped, dtype=float_dtype )
        self.__normConst2 = np.float32( 1.0 / ( np.float64(self.__shapeCropped[0])*np.float64(self.__shapeCropped[1]))**2.0 )
        self.bench['xcorr1'] = time.time() 
        
        return triIndices 
        
    def xcorrnm2_speckle( self, triIndices=None ):
        """
        Robert A. McLeod
        robbmcleod@gmail.com
        October 1, 2016
        
        With data recorded automatically from SerialEM, we no long have access to the gain reference
        normalization step provided by Gatan.  With the K2 detector, gain normalization is no 
        longer a simple multiplication.  Therefore we see additional, multiplicative (or speckle) 
        noise in the images compared to those recorded by Gatan Microscopy Suite.  Here we want 
        to use a different approach from the Padfield algorithm, which is useful for suppressing 
        additive noise, and 
        
        In general Poisson noise should be speckle noise, especially at the dose rates commonly 
        seen in cryo-EM.
        
        """
        triIndices = self.__init_xcorrnm2( triIndices = triIndices)
        
        # Pre-compute forward FFTs (template will just be copied conjugate Fourier spectra)
        self.__imageFFT = np.empty( [self.__N, self.shapePadded[0], self.shapePadded[1]], dtype=fftw_dtype )
        
        self.__autocorrHalfs = np.empty( [self.__N, self.__shapeCropped[0], self.__shapeCropped[1]], dtype=float_dtype )
        
        currIndex = 0
        self.__originC = []; self.C = []
        
        print( "Pre-computing forward Fourier transforms and autocorrelations" )
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
                
            self.__tempComplex = nz.evaluate( "masks_block * images_block" ).astype( fftw_dtype )    
            self.__FFT2.update_arrays( self.__tempComplex, self.__imageFFT[I,:,:]); self.__FFT2.execute()
            
            print( "TODO: FOURIER CROPPING" )
            
            # Compute autocorrelation
            imageFFT = self.__imageFFT[I,:,:]
            # Not sure if numexpr is useful for such a simple operation?
            self.__tempComplex = nz.evaluate( "imageFFT * conj(imageFFT)" )
            self.__IFFT2.update_arrays( self.__tempComplex, self.__tempComplex2 )
            tempComplex2 = self.__tempComplex2
            
            nz.evaluate( "0.5*abs(tempComplex2)", out=self.__autocorrHalfs[I,:,:] )
        self.bench['xcorr2'] = time.time() 
        
        
    
        ########### COMPUTE PHASE CORRELATIONS #############
        print( "Starting correlation calculations, mode: " + self.triMode )
        if self.triMode == u'refine':
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
                
                self.mnxc2_SPECKLE( I, I, self.__shapeCropped, refine=True )
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
                self.__templateImageFFT = nz.evaluate( "conj(tempComplex)")

        
                # Now we can start looping through base images
                columnIndices = np.unique( np.argwhere( triIndices[I,:] ) )
                #print( "columnIndices: " + str(columnIndices) )
                for J in columnIndices:
                    
                    ####### MNXC2 revisement with private variable to make the code more manageable.
                    self.mnxc2_speckle( I, J, self.__shapeCropped )
                    
                    #### Find maximum positions ####    
                    self.locatePeak( I, J )
                    
                    if self.verbose: 
                        print( "# " + str(I) + "->" + str(J) + " shift: [%.2f"%self.__shiftsTriMat[I,J,0] 
                            + ", %.2f"%self.__shiftsTriMat[I,J,1]
                            + "], cc: %.6f"%self.__corrTriMat[I,J] 
                            + ", peak sig: %.3f"%self.__peaksigTriMat[I,J] )    
                        
                    # Correlation stats is for establishing correlation scores for fixed-pattern noise.
                    if bool( self.trackCorrStats ):
                        self.calcCorrStats( currIndex, triIndices )
                        
                    # triMode 'auto' diagonal mode    
                    if self.triMode == u'auto' and (self.__peaksigTriMat[I,J] <= self.peaksigThres or J-I >= self.autoMax):
                        if self.verbose: print( "triMode 'auto' stopping at frame: " + str(J) )
                        break
                    currIndex += 1
                pass # C max position location
        
        

        if bool( np.any( self.fouCrop ) ):
            self.__shiftsTriMat[:,:,0] *= self.shapePadded[0] / self.__shapeCropped[0]
            self.__shiftsTriMat[:,:,1] *= self.shapePadded[1] / self.__shapeCropped[1]
        
        self.bench['xcorr3'] = time.time()
        # Pointer reference house-keeping
        del images_block, masks_block, imageFFT, tempComplex2
        
            
    def xcorrnm2_tri( self, triIndices=None ):
        """
        Robert A. McLeod
        robbmcleod@gmail.com
        April 16, 2015
        
        triIndices is the index locations to correlate to.  If None, self.triMode 
        is used to build one.  Normally you should use self.triMode for the first iteration, 
        and pass in a triIndice from the errorDict if you want to repeat.
        
        returns : [shiftsTriMat, corrTriMat, peaksTriMat]
        
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
        
        triIndices = self.__init_xcorrnm2( triIndices = triIndices)
        
        if self.masks.shape[0] == 1 :
            # tempComplex = self.masks[0,:,:].astype( fftw_dtype ) 
            self.__baseMaskFFT = np.empty( self.__shapeCropped, dtype=fftw_dtype )

            self.__FFT2.update_arrays( self.masks[0,:,:].squeeze().astype( fftw_dtype ), self.__tempFullframe ); self.__FFT2.execute()
            # FFTCrop
            sC2 = self.__shapeCropped2
            self.__baseMaskFFT[0:sC2[0],0:sC2[1]] = self.__tempFullframe[0:sC2[0],0:sC2[1]]
            self.__baseMaskFFT[0:sC2[0],-sC2[1]:] = self.__tempFullframe[0:sC2[0],-sC2[1]:] 
            self.__baseMaskFFT[-sC2[0]:,0:sC2[1]] = self.__tempFullframe[-sC2[0]:,0:sC2[1]] 
            self.__baseMaskFFT[-sC2[0]:,-sC2[1]:] = self.__tempFullframe[-sC2[0]:,-sC2[1]:] 
            
            self.__templateMaskFFT = np.conj( self.__baseMaskFFT )
            
            # maskProduct term is M1^* .* M2
            templateMaskFFT = self.__templateMaskFFT; 
            baseMaskFFT = self.__baseMaskFFT # Pointer assignment
            self.__tempComplex2 = nz.evaluate( "templateMaskFFT * baseMaskFFT" )
            self.__IFFT2.update_arrays( self.__tempComplex2, self.__tempComplex ); self.__IFFT2.execute()
            tempComplex = self.__tempComplex
            normConst2 = self.__normConst2
            self.__maskProduct = nz.evaluate( "normConst2*real(tempComplex)" )
        else:
            # Pre-allocate only
            self.__baseMaskFFT = np.zeros( [self.__N, self.__shapeCropped[0], self.__shapeCropped[1]], dtype=fftw_dtype )
        
            
        if bool( self.maxShift ) or self.Bmode is u'fourier':
            if self.maxShift is None or self.preShift is True:
                [xmesh,ymesh] = np.meshgrid( np.arange(-self.__shapeCropped2[0], self.__shapeCropped2[0]), 
                                            np.arange(-self.__shapeCropped2[1], self.__shapeCropped2[1])  )
            else:
                [xmesh,ymesh] = np.meshgrid( np.arange(-self.maxShift, self.maxShift), np.arange(-self.maxShift, self.maxShift)  )
            
            rmesh2 = nz.evaluate( "xmesh*xmesh + ymesh*ymesh" )
            # rmesh2 = xmesh*xmesh + ymesh*ymesh
            if bool( self.maxShift ): 
                self.__mask_maxShift = ( rmesh2 < self.maxShift**2.0 )
            if self.Bmode is u'fourier':
                self.__Bfilter = np.fft.fftshift( util.apodization( name=self.BfiltType, 
                                                                   size=self.__shapeCropped, 
                                                                   radius=[self.Brad,self.Brad] ) )

        self.bench['xcorr1'] = time.time() 
        # Pre-compute forward FFTs (template will just be copied conjugate Fourier spectra)
        self.__imageFFT = np.empty( [self.__N, self.shapePadded[0], self.shapePadded[1]], dtype=fftw_dtype )
        self.__baseImageFFT = np.empty( [self.__N, self.__shapeCropped[0], self.__shapeCropped[1]], dtype=fftw_dtype )
        self.__baseSquaredFFT = np.empty( [self.__N, self.__shapeCropped[0], self.__shapeCropped[1]], dtype=fftw_dtype )
        
        # Looping for triagonal matrix
        # For auto this is wrong, so make these lists instead
        currIndex = 0
        self.__originC = []; self.C = []


            
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
                
            tempReal = nz.evaluate( "masks_block * images_block" ).astype( fftw_dtype )

                    
            self.__FFT2.update_arrays( tempReal, self.__tempFullframe ); self.__FFT2.execute()
            if self.shiftMethod == u"fourier":
                self.__imageFFT[I,:,:] = self.__tempFullframe.copy(order='C')
                # FFTCrop
                self.__baseImageFFT[I,0:sC2[0],0:sC2[1]] = self.__imageFFT[I,0:sC2[0],0:sC2[1]]
                self.__baseImageFFT[I,0:sC2[0],-sC2[1]:] = self.__imageFFT[I,0:sC2[0],-self.__sC2[1]:] 
                self.__baseImageFFT[I,-sC2[0]:,0:sC2[1]] = self.__imageFFT[I,-sC2[0]:,0:self.__sC2[1]] 
                self.__baseImageFFT[I,-sC2[0]:,-sC2[1]:] = self.__imageFFT[I,-sC2[0]:,-sC2[1]:] 
                print( "TODO: check memory consumption" )
            else:
                # FFTCrop
                self.__baseImageFFT[I,0:sC2[0],0:sC2[1]] = self.__tempFullframe[0:sC2[0],0:sC2[1]]
                self.__baseImageFFT[I,0:sC2[0],-sC2[1]:] = self.__tempFullframe[0:sC2[0],-sC2[1]:] 
                self.__baseImageFFT[I,-sC2[0]:,0:sC2[1]] = self.__tempFullframe[-sC2[0]:,0:sC2[1]] 
                self.__baseImageFFT[I,-sC2[0]:,-sC2[1]:] = self.__tempFullframe[-sC2[0]:,-sC2[1]:] 
            


            
            self.__FFT2.update_arrays( nz.evaluate( "tempReal*tempReal" ).astype( fftw_dtype ), self.__tempFullframe ); self.__FFT2.execute()
            # FFTCrop
            self.__baseSquaredFFT[I,0:sC2[0],0:sC2[1]] = self.__tempFullframe[0:sC2[0],0:sC2[1]]
            self.__baseSquaredFFT[I,0:sC2[0],-sC2[1]:] = self.__tempFullframe[0:sC2[0],-sC2[1]:] 
            self.__baseSquaredFFT[I,-sC2[0]:,0:sC2[1]] = self.__tempFullframe[-sC2[0]:,0:sC2[1]] 
            self.__baseSquaredFFT[I,-sC2[0]:,-sC2[1]:] = self.__tempFullframe[-sC2[0]:,-sC2[1]:] 
            
            
            
            if not self.masks.shape[0] == 1:
                self.__FFT2.update_arrays( self.masks[I,:,:].squeeze().astype( fftw_dtype), self.__tempFullframe ); self.__FFT2.execute()
                # FFTCrop
                self.__baseMaskFFT[I,0:sC2[0],0:sC2[1]] = self.__tempFullframe[0:sC2[0],0:sC2[1]]
                self.__baseMaskFFT[I,0:sC2[0],-sC2[1]:] = self.__tempFullframe[0:sC2[0],-sC2[1]:] 
                self.__baseMaskFFT[I,-sC2[0]:,0:sC2[1]] = self.__tempFullframe[-sC2[0]:,0:sC2[1]] 
                self.__baseMaskFFT[I,-sC2[0]:,-sC2[1]:] = self.__tempFullframe[-sC2[0]:,-sC2[1]:] 

            pass
        del masks_block, images_block
        

                
        self.bench['xcorr2'] = time.time() 
    
        print( "Starting correlation calculations, mode: " + self.triMode )
        if self.triMode == u'refine':
            
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
                
                self.mnxc2( I, I, self.__shapeCropped, refine=True )
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
                self.__templateImageFFT = nz.evaluate( "conj(tempComplex)")
                    
                tempComplex2 =  self.__baseSquaredFFT[I,:,:]
                self.__templateSquaredFFT = nz.evaluate( "conj(tempComplex2)")
        
                if not self.masks.shape[0] == 1:
                    tempComplex = baseMaskFFT[I,:,:]
                    self.__templateMaskFFT = nz.evaluate( "conj(tempComplex)")
        
                # Now we can start looping through base images
                columnIndices = np.unique( np.argwhere( triIndices[I,:] ) )
                #print( "columnIndices: " + str(columnIndices) )
                for J in columnIndices:
                    
                    ####### MNXC2 revisement with private variable to make the code more manageable.
                    self.mnxc2( I, J, self.__shapeCropped )
                    
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
                        self.calcCorrStats( currIndex, triIndices )
                        
                    # triMode 'auto' diagonal mode    
                    if self.triMode == u'auto' and (self.__peaksigTriMat[I,J] <= self.peaksigThres or J-I >= self.autoMax):
                        if self.verbose: print( "triMode 'auto' stopping at frame: " + str(J) )
                        break
                    currIndex += 1
                pass # C max position location
        
        

        if bool( np.any( self.fouCrop ) ):
            self.__shiftsTriMat[:,:,0] *= self.shapePadded[0] / self.__shapeCropped[0]
            self.__shiftsTriMat[:,:,1] *= self.shapePadded[1] / self.__shapeCropped[1]
        
        self.bench['xcorr3'] = time.time()
        # Pointer reference house-keeping
        del templateMaskFFT, tempComplex, tempComplex2 # Pointer
        return

        
    def mnxc2( self, I, J, shapeCropped, refine=False ):
        """
        2-D Masked, Intensity Normalized, Cross-correlation
        """
        tempComplex = self.__tempComplex # Pointer re-assignment
        tempComplex2 = self.__tempComplex2 # Pointer re-assignment
        maskProduct = self.__maskProduct
        normConst2 = self.__normConst2
        
        if not self.masks.shape[0] == 1:
            # Compute maskProduct, term is M1^* .* M2
            baseMask_block = self.__baseMaskFFT[J,:,:]; templateMaskFFT = self.__templateMaskFFT # Pointer re-assignment
            tempComplex2 = nz.evaluate( "templateMaskFFT * baseMask_block" )
            self.__IFFT2.update_arrays( tempComplex2, tempComplex ); self.__IFFT2.execute()
            # maskProduct = np.clip( np.round( np.real( tempComplex ) ), eps, np.Inf )
            self.__maskProduct = nz.evaluate( "real(tempComplex)*normConst2" )
            
        # Compute mask correlation terms
        if self.masks.shape[0] == 1:
            templateImageFFT = self.__templateImageFFT; baseMask_block = self.__baseMaskFFT # Pointer re-assignment
        self.__IFFT2.update_arrays( nz.evaluate( "baseMask_block * templateImageFFT"), tempComplex ); self.__IFFT2.execute()
        

        Corr_templateMask = nz.evaluate( "real(tempComplex)*normConst2" ) # Normalization
        
        baseImageFFT_block = self.__baseImageFFT[J,:,:]; templateMaskFFT = self.__templateMaskFFT
        self.__IFFT2.update_arrays( nz.evaluate( "templateMaskFFT * baseImageFFT_block"), tempComplex ); self.__IFFT2.execute()

        # These haven't been normalized, so let's do so.  They are FFT squared, so N*N
        # This reduces the strain on single-precision range.
        Corr_baseMask =  nz.evaluate( "real(tempComplex)*normConst2" ) # Normalization

        # Compute the intensity normalzaiton for the template
        if self.masks.shape[0] == 1:
            baseMaskFFT = self.__baseMaskFFT; templateSquaredFFT = self.__templateSquaredFFT
            self.__IFFT2.update_arrays( nz.evaluate( "baseMaskFFT * templateSquaredFFT"), tempComplex ); self.__IFFT2.execute()
        else:
            self.__IFFT2.update_arrays( nz.evaluate( "baseMaskFFT_block * templateSquaredFFT"), tempComplex ); self.__IFFT2.execute()

        # DenomTemplate = nz.evaluate( "real(tempComplex)*normConst2 - real( Corr_templateMask * (Corr_templateMask / maskProduct) )" )
        
        # Compute the intensity normalzaiton for the base Image
        baseSquared_block = self.__baseSquaredFFT[J,:,:]
        self.__IFFT2.update_arrays( nz.evaluate( "templateMaskFFT * baseSquared_block"), tempComplex2 ); self.__IFFT2.execute()
        
        # Compute Denominator intensity normalization
        # DenomBase = nz.evaluate( "real(tempComplex2)*normConst2- real( Corr_baseMask * (Corr_baseMask / maskProduct) )" )
        Denom = nz.evaluate( "sqrt( (real(tempComplex2)*normConst2- real( Corr_baseMask * (Corr_baseMask / maskProduct)))" + 
            "* (real(tempComplex)*normConst2 - real( Corr_templateMask * (Corr_templateMask / maskProduct)) ) )" )
            
        # What happened to numexpr clip?
        Denom = np.clip( Denom, 1, np.Inf )
        # print( "Number of small Denominator values: " + str(np.sum(DenomTemplate < 1.0)) )
        
        # Compute Numerator (the phase correlation)
        tempComplex2 = nz.evaluate( "baseImageFFT_block * templateImageFFT" )
        self.__IFFT2.update_arrays( tempComplex2, tempComplex ); self.__IFFT2.execute()
        # Numerator = nz.evaluate( "real(tempComplex)*normConst2 - real( Corr_templateMask * Corr_baseMask / maskProduct)" ) 
        
        # Compute final correlation
        self.__C = nz.evaluate( "(real(tempComplex)*normConst2 - real( Corr_templateMask * Corr_baseMask / maskProduct)) / Denom" )
        

        # print( "%%%% mnxc2.Denom.dtype = " + str(Denom.dtype) )
        self.__originTriMat[I,J] = self.__C[0,0]
        if bool(self.suppressOrigin):
            # If gain reference is quite old we can still get one bright pixel at the center.
            # The hot pixel filter has mitigated this but it's still a minor source of bias.
            self.__C[0,0] = 0.125 * ( self.__C[1,0] + self.__C[0,1] + self.__C[-1,0] + self.__C[-1,0] +
                self.__C[1,1] + self.__C[-1,1] + self.__C[-1,1] + self.__C[-1,-1] )
            
        # We have everything in normal FFT order until here; Some speed-up could be found by its removal.
        # Pratically we don't have to do this fftshift, but it makes plotting easier to understand
        self.__C = np.fft.ifftshift( self.__C )

        # We can crop C if maxShift is not None and preShift is False
        if self.maxShift is not None and self.preShift is False:
            shapeCropped2 = (np.array(shapeCropped)/2.0).astype('int')
            self.__C = self.__C[shapeCropped2[0]-self.maxShift:shapeCropped2[0]+self.maxShift, shapeCropped2[1]-self.maxShift:shapeCropped2[1]+self.maxShift]

     
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
            self.bench['opti0'] = time.time()
            # Want to define this locally so it inherits scope variables.
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
                                    options={'maxiter':maxIter, 'xatol':tolerance }  )
                                    
            self.__C_filt = scipy.ndimage.gaussian_filter( self.__C, result.x )
            self.bench['opti0'] = time.time()
            if self.verbose >= 2:
                print( "Found optimum B-sigma: %.3f"%result.x + ", with peak sig: %.3f"%(1.0/result.fun)+" in %.1f"%(1E3*(self.bench['opti1']-self.bench['opti0']))+" ms" ) 
        elif bool(self.Brad) and self.Bmode =='fourier':
            tempComplex = self.__C.astype(fftw_dtype)
            self.__FFT2.update_arrays( tempComplex, tempComplex2 ); self.__FFT2.execute()
            Bfilter = self.__Bfilter
            self.__IFFT2.update_arrays( nz.evaluate( "tempComplex2*Bfilter" ), tempComplex ); self.__IFFT2.execute()
            # Conservation of counts with Fourier filtering is not 
            # very straight-forward.
            C_filt = nz.evaluate( "real( tempComplex )/sqrt(normConst)" )
        elif bool(self.Brad) and self.Bmode == u'conv' or self.Bmode == u'convolution':
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
                # print( "In pre-shift" )
                # This isn't working with 'refine'
                if self.triMode != u'refine':
                    rolledMask = np.roll( np.roll( self.__mask_maxShift, 
                        np.round(self.__shiftsTriMat[I,J-1,0]).astype('int'), axis=0 ), 
                        np.round(self.__shiftsTriMat[I,J-1,1]).astype('int'), axis=1 )
                elif self.triMode == u'refine':
                    # With refine the matrix is populated like an autocorrelation function.
                    rolledMask = np.roll( np.roll( self.__mask_maxShift, 
                        np.round(self.__shiftsTriMat[I-1,I-1,0]).astype('int'), axis=0 ), 
                        np.round(self.__shiftsTriMat[I-1,I-1,1]).astype('int'), axis=1 )
                    pass
                C_masked = nz.evaluate("C_filt*rolledMask")
                cmaxpos = np.unravel_index( np.argmax( C_masked ), C_masked.shape )
                self.__peaksigTriMat[I,J] = (C_masked[cmaxpos] - np.mean(C_filt[rolledMask]))/ np.std(C_filt[rolledMask])
            else:
                mask_maxShift = self.__mask_maxShift
                C_masked = nz.evaluate("C_filt*mask_maxShift")
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
                # TODO: I think pad has issues with complex numbers?
                #CpadFFT = np.pad( np.fft.fftshift(self.__CsubFFT), ((self.subPixReg-1)*self.__subR,), mode=b'constant', constant_values=(0.0,)  )
                

                self.__CpadFFT = np.zeros( [self.subPixReg*self.__subR*2,self.subPixReg*self.__subR*2], dtype=fftw_dtype )
                
                # NUMPY BUG: mode has to be a byte string
                
                self.__CpadFFT.real = np.pad( np.fft.fftshift(self.__CsubFFT.real), ((self.subPixReg-1)*self.__subR,), mode=constantPad, constant_values=(0.0,)  )
                self.__CpadFFT.imag = np.pad( np.fft.fftshift(self.__CsubFFT.imag), ((self.subPixReg-1)*self.__subR,), mode=constantPad, constant_values=(0.0,)  )
                self.__CpadFFT = np.fft.ifftshift( self.__CpadFFT )
                self.__subIFFT2.update_arrays( self.__CpadFFT, self.__Csub_over ); self.__subIFFT2.execute()
                # Csub_overAbs = nz.evaluate( "abs( Csub_over )") # This is still complex
                Csub_overAbs = np.abs( self.__Csub_over )
                
                
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
                print( "Correlation sub-area too close to maxShift!  Subpixel location broken.  Consider increasing maxShift." )
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
    

    def calcCorrStats( self, currIndex, triIndices ):
        # Track the various statistics about the correlation map, mean, std, max, skewness
        if currIndex == 0 or self.corrStats is None:
            # Mean, std, max, maxposx, maxposy, (val at 0,0), imageI mean, imageI std, imageJ mean, imageJ std =  10 columns
            K = np.sum(triIndices)
            self.corrStats = {}
            self.corrStats[u'K'] = K
            self.corrStats[u'meanC'] = np.zeros([K])
            self.corrStats[u'varC'] = np.zeros([K])
            self.corrStats[u'maxC'] = np.zeros([K])
            self.corrStats[u'maxPos'] = np.zeros([K,2])
            self.corrStats[u'originC'] = np.zeros([K])
            print( "Computing stack mean" )
            self.corrStats[u'stackMean'] = np.mean( self.images )
            print( "Computing stack variance" )
            self.corrStats[u'stackVar'] = np.var( self.images )
            
        self.corrStats[u'meanC'][currIndex] = np.mean(self.__C_filt)
        self.corrStats[u'varC'][currIndex] = np.var(self.__C_filt)
        self.corrStats[u'maxC'][currIndex] = np.max(self.__C_filt)
        self.corrStats[u'maxPos'][currIndex,:] = np.unravel_index( np.argmax(self.__C_filt), \
                                                self.__shapeCropped ) - \
                                                np.array([self.__C_filt.shape[0]/2, self.__C_filt.shape[1]/2])
        self.corrStats[u'originC'][currIndex] = self.__C_filt[self.__C.shape[0]/2, self.__C.shape[1]/2]   
    
                        
    def shiftsSolver( self, shiftsTriMat_in, corrTriMat_in, peaksigTriMat_in, 
                     acceptedEqns=None, mode='basin', Niter=100 ):
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

        shapeImage = np.array( [self.images.shape[1], self.images.shape[2]] )
        N = np.asarray( self.images.shape )[0] - 1
        last_col = np.zeros( N, dtype='int' )
            
        #### BUILD VECTORIZED SHIFTS b_x, b_y AND EQUATION COEFFICIENT MATRIX Acoeff
        M = 0
        for I in np.arange(0,N, dtype='int'):
            # Find the last non-zero element in the tri-matrix for each row
            # This determines the sub-sampled view for each equation set.
            if triIndices[I,:].any():
                last_col[I] = np.argwhere(triIndices[I,:])[-1] + 1
                M += last_col[I] - I
        
        # For some reason this becomes -1 if we make last_col not float.
        M = np.int(M)
        Acoeff = np.zeros( [M,N] )
        Arow_pos = 0
        for I in np.arange(0,N, dtype='int'):
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
        # This was a cornerstone of MotionCorr but it often leads to problems, so let's avoid it completely
        # in favour of deweighting bad equations.
        if acceptedEqns is None:
            Maccepted = M
            acceptedEqns = np.ones_like( b_x, dtype='bool' )
        else:
            Maccepted = np.sum( acceptedEqns )
        print( "Optimization of shifts over M = " + str(Maccepted) + " equations." )
        
        #### WEIGHTS FOR OPTIMIZATION ####
        # There's only 2.5 % difference between the weighted and un-weighted versions for the STEM test cases.
        # CryoEM would be expected to be higher as the CC's are about 0.001 compared to 0.3
        if self.weightMode is None or self.weightMode == u'corr': # use raw correlation scores or peaksig
            weights =  np.ravel( peaksigTriMat[triIndices] )
        elif self.weightMode is u'unweighted': # don't weight peaks
            weights = np.ones_like( np.ravel( peaksigTriMat[triIndices] ) )
        elif self.weightMode == u'norm' or self.weightMode == u'normalized':
            ### Scale the weights so that lower correlations count for next-to-nothing
            weights = util.normalize( np.ravel( peaksigTriMat[triIndices] ) )
        elif self.weightMode == u'autologistic':
            # Calculate a logistic from the CDF of the peaksig values
            self.cdfLogisticCurve() # Sets peaksigThres, logisticK, and logisticNu
            
            peakSig = np.ravel( peaksigTriMat[triIndices] ).astype( 'float64' )
            weights = 1.0 - 1.0 / (1.0 + np.exp( -self.logisticK*(-peakSig + self.peaksigThres) ) )**self.logisticNu
        elif self.weightMode == u'logistic':
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
            
        if mode == u'local':
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
                
        elif mode == u'basin':
            #### GLOBAL MINIMIZATION X, Y SOLUTION ####
            basinArgs = {}
            basinArgs[u"bounds"] = bounds
            basinArgs[u"tol"] = pix_tol
            basinArgs[u"method"] =  u"L-BFGS-B"
            basinArgs[u"args"] = (Acoeff, b_x, weights*acceptedEqns)
            try:
                outX = scipy.optimize.basinhopping( util.weightedErrorNorm, drift_guess, niter=Niter, minimizer_kwargs=basinArgs )
                relativeEst[:,1] = outX.x
            except:
                raise RuntimeError( "Error: caught exception on X-minimizer" )   
            # basinArgs["args"] = (Acoeff[acceptedEqns], b_y[acceptedEqns], weights[acceptedEqns])
            basinArgs[u"args"] = (Acoeff, b_y, weights*acceptedEqns)
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
        acceptedEqnsUnraveled = np.pad( acceptedEqnsUnraveled, ((0,1),(1,0)), mode=constantPad )
        
        # Ok so how big is relativeEst?  Can we add in zeros?
        # Or maybe I should just give weights as weights*acceptedEqnsUnr
        errorXY = np.zeros( [M,2] )
        
        ############# Unweighted error ################
        """
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
        
        errorXun = np.pad( errorXun, ((0,1),(1,0)), mode=constantPad )
        errorYun = np.pad( errorYun, ((0,1),(1,0)), mode=constantPad )
        triPadded = np.pad( triIndices, ((0,1),(1,0)), mode=constantPad )
        
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
        """

        ################## Weighted error ######################
        # Make any zero weight just very small
        weights = np.clip( weights, 1E-6, np.Inf )
        
        errorXY[:,1] = np.dot( Acoeff, relativeEst[:,1] ) - b_x
        errorXY[:,0] = np.dot( Acoeff, relativeEst[:,0] ) - b_y
        
        errorNorm = np.sqrt( errorXY[:,0]*errorXY[:,0] + errorXY[:,1]*errorXY[:,1] )

        acceptedErrorNorm = errorNorm[acceptedEqns]
        
        mean_errorNorm = np.sum( weights * acceptedErrorNorm ) / np.sum(weights)
        mean_unweighted = np.mean( errorNorm[acceptedEqns] )
        
        # print( "RMS: " + str(np.sum( weights * acceptedErrorNorm**2 )) )
        # print( "Normed RMS: " + str(np.sum( weights * acceptedErrorNorm**2 ) / np.sum(weights)))
        # print( "mean_errorNorm**2 + " + str(  mean_errorNorm**2 ))
        std_errorNorm = np.sqrt( np.sum( weights * acceptedErrorNorm**2 ) 
                / np.sum(weights) - mean_errorNorm**2 )
        # np.sqrt( np.sum( unalignedHist * unalignedCounts**2 )/ sumFromHist - meanFromHist*meanFromHist  )

        std_unweighted = np.std( acceptedErrorNorm )
        # print( "sum(acceptedErrorNorm): %f" % np.sum(acceptedErrorNorm) )
        print( "MEAN ERROR (weighted: %f | unweighted: %f )" % (mean_errorNorm, mean_unweighted)  )
        print( "STD ERROR (weighted: %f | unweighted: %f )" % (std_errorNorm, std_unweighted)  )
        
        # Error unraveled (i.e. back to the upper triangular matrix form)
        errorUnraveled = np.zeros( [N,N] )
        errorXun = np.zeros( [N,N] )
        errorYun = np.zeros( [N,N] )
        weightsUn = np.zeros( [N,N] )
        errorUnraveled[unravelIndices[0], unravelIndices[1]] = errorNorm
        weightsUn[unravelIndices[0], unravelIndices[1]] = weights
        errorXun[unravelIndices[0], unravelIndices[1]] = np.abs( errorXY[:,1] )
        errorYun[unravelIndices[0], unravelIndices[1]] = np.abs( errorXY[:,0] )
        
        errorXun = np.pad( errorXun, ((0,1),(1,0)), mode=constantPad )
        errorYun = np.pad( errorYun, ((0,1),(1,0)), mode=constantPad )
        triPadded = np.pad( triIndices, ((0,1),(1,0)), mode=constantPad )
        weightsUn = np.pad( weightsUn, ((0,1),(1,0)), mode=constantPad )
        
        # DEBUG: weighted error trimats
        # plot.ims( (errorXun, weightsUn, errorYun, acceptedEqnsUnraveled), titles=( "errorXun","weightsUn","errorYun", "AcceptedUnraveled") )
        
        # Mask out un-used equations from error numbers
        errorYun = errorYun * acceptedEqnsUnraveled
        errorXun = errorXun * acceptedEqnsUnraveled
        triPadded = triPadded * acceptedEqnsUnraveled    

        # errorX and Y are per-frame error estimates
        errorX = np.zeros( N+1 )
        errorY = np.zeros( N+1 )
        # Sum horizontally and vertically, keeping in mind diagonal is actually at x-1
        for J in np.arange(0,N+1):
            try:
                errorX[J] = ( ( np.sum( errorXun[J,:]*weightsUn[J,:]) + np.sum(errorXun[:,J-1]*weightsUn[:,J-1]) ) / 
                        ( np.sum( weightsUn[J,:]) + np.sum(weightsUn[:,J-1]) ) )
            except:
                print( "Warning: per-frame error estimation failed, possibly due to zero-weight in solution solver" )
            try:
                errorY[J] = ( ( np.sum( errorYun[J,:]*weightsUn[J,:]) + np.sum(errorYun[:,J-1]*weightsUn[:,J-1]) ) / 
                        ( np.sum( weightsUn[J,:]) + np.sum(weightsUn[:,J-1]) ) )
            except:
                print( "Warning: per-frame error estimation failed, possibly due to zero-weight in solution solver" )
        #### END WEIGHTED ERROR ############
        
        # translations (to apply) are the negative of the found shifts
        errorDict[u'translations'] = -np.vstack( (np.zeros([1,2]), np.cumsum( relativeEst, axis=0 ) ) )
        errorDict[u'relativeEst'] = relativeEst
        errorDict[u'acceptedEqns'] = acceptedEqns
        # Not necessary to save triIndices, it's the non-zero elements of corrTriMat
        # errorDict['triIndices'] = triIndices
        errorDict[u'weights'] = weights
        errorDict[u'errorXY'] = errorXY
        errorDict[u'shiftsTriMat'] = shiftsTriMat_in
        errorDict[u'errorX'] = errorX 
        errorDict[u'errorY'] = errorY 
        errorDict[u'errorUnraveled'] = errorUnraveled
        errorDict[u'mean_errorNorm'] = mean_errorNorm
        errorDict[u'std_errorNorm'] = std_errorNorm 
        errorDict[u'M'] = M
        errorDict[u'Maccepted'] = Maccepted
        
        
        return errorDict
        
    def alignImageStack( self ):
        """
        alignImageStack does a masked cross-correlation on a set of images.  
        masks can be a single mask, in which case it is re-used for each image, or 
        individual for each corresponding image.  
        
        Subpixel shifting is usually done with a large, shifted Lanczos resampling kernel. 
        This was found to be faster than with a phase gradient in Fourier space.
        """
        
        # Setup threading, pyFFTW is set elsewhere in planning
        if self.n_threads is None:
            self.n_threads = nz.detect_number_of_cores()
        else:
            nz.set_num_threads( self.n_threads )
        print( "Numexprz using %d threads and float dtype: %s" % (nz.nthreads, float_dtype) )


            
        #Baseline un-aligned stack, useful for see gain reference problems
        # self.unalignedSum = np.sum( self.images, axis=0 )
        if np.any( self.shapeBinned ):
            self.binStack()
            
        # It's generally more robust to do the hot pixel filtering after binning 
        # from SuperRes.
        if self.filterMode != None and 'hot' in self.filterMode.lower():
            self.hotpixFilter()
            
        # Do CTF measurement first, so we save processing if it can't fit the CTF
        # Alternatively if CTFProgram == 'ctffind,sum' this is performed after alignment. 
        if bool(self.CTFProgram): 
            splitCTF = self.CTFProgram.lower().replace(' ','').split(',')
            if len(splitCTF) == 1 and ( splitCTF[0] == u'ctffind' or splitCTF[0] == u'ctffind4.1'):
                self.execCTFFind41( movieMode=True )
            elif len(splitCTF) == 1 and ( splitCTF[0] == u'ctffind4' ):
                self.execCTFFind4( movieMode=True )    
            elif len(splitCTF) == 1 and (splitCTF[0] == u'gctf'): # Requires CUDA and GPU
                self.execGCTF( movieMode=True )
            

            
        """
        Registration, first run: Call xcorrnm2_tri to do the heavy lifting
        """
        if self.xcorrMode.lower() == 'zorro':
            """
            Application of padding.
            """
            if np.any(self.shapePadded):
                self.padStack()
            
            self.xcorrnm2_tri()

            """
            Functional minimization over system of equations
            """
            self.bench['solve0'] = time.time()
            if self.triMode == u'first':
                self.translations = -self.__shiftsTriMat[0,:]
                self.errorDictList.append({})
                self.errorDictList[-1][u'shiftsTriMat'] = self.__shiftsTriMat
                self.errorDictList[-1][u'corrTriMat'] = self.__corrTriMat
                self.errorDictList[-1][u'originTriMat'] = self.__originTriMat
                self.errorDictList[-1][u'peaksigTriMat'] = self.__peaksigTriMat
                self.errorDictList[-1][u'translations'] = self.translations.copy()
            elif self.triMode == u'refine':
                self.errorDictList.append({})
                self.errorDictList[-1][u'shiftsTriMat'] = self.__shiftsTriMat
                self.errorDictList[-1][u'corrTriMat'] = self.__corrTriMat
                self.errorDictList[-1][u'originTriMat'] = self.__originTriMat
                self.errorDictList[-1][u'peaksigTriMat'] = self.__peaksigTriMat
                
                m = self.images.shape[0]
                self.translations = np.zeros( [m,2], dtype='float32' )
    
                for K in np.arange(m): 
                    self.translations[K,:] = -self.__shiftsTriMat[K,K,:]
                self.errorDictList[-1][u'translations'] = self.translations.copy()
                
            else:
                # Every round of shiftsSolver makes an error dictionary
                self.shiftsSolver( self.__shiftsTriMat, self.__corrTriMat, self.__peaksigTriMat )
                self.errorDictList[-1][u'originTriMat'] = self.__originTriMat
                self.translations = self.errorDictList[-1][u'translations'].copy( order='C' )
            self.bench['solve1'] = time.time()
            
            """
            Alignment and projection through Z-axis (averaging)
            """
            if np.any(self.shapePadded): # CROP back to original size
                self.cropStack()
            self.applyShifts()
        elif self.xcorrMode.lower() == 'unblur v1.02':
            self.xcorr2_unblur1_02()
        elif self.xcorrMode.lower() == 'motioncorr v2.1':
            self.xcorr2_mc2_1()
        elif self.xcorrMode.lower() == 'move only':
            pass
        else:
            raise ValueError( "Zorro.alignImageStack: Unknown alignment tool %s" % self.xcorrMode )
        

        # Calculate CTF on aligned sum if requested
        if bool(self.CTFProgram) and len(splitCTF) >= 2 and splitCTF[1]== u'sum':
            if splitCTF[0] == u'ctffind' or splitCTF[0] == u'ctffind4.1':
                self.execCTFFind41( movieMode=False )
            elif splitCTF[0] == u'ctffind4':
                self.execCTFFind4( movieMode=False )
            elif splitCTF[0] == u'gctf': # Requires CUDA
                self.execGCTF( movieMode=False )
        
        if bool(self.doEvenOddFRC):
            self.evenOddFouRingCorr()
        elif bool(self.doLazyFRC): # Even-odd FRC has priority
            self.lazyFouRingCorr()

        
        
        # Apply filters as a comma-seperated list.  Whitespace is ignored.
        if bool( self.filterMode ):
            splitFilter = self.filterMode.lower().replace(' ','').split(',')
            if len(splitFilter) > 0:
                self.bench['dose0'] = time.time()
                for filt in splitFilter:
                    if filt == u"dose" and not "unblur" in self.xcorrMode.lower():
                        print( "Generating dose-filtered sum" )
                        # Dose filter will ALWAYS overwrite self.filtSum because it has to work with individual frames
                        self.doseFilter( normalize=False )
                    elif filt == u"dosenorm" and not "unblur" in self.xcorrMode.lower():
                        print( "Generating Fourier-magnitude normalized dose-filtered sum" )
                        # Dose filter will ALWAYS overwrite self.filtSum because it has to work with individual frames
                        self.doseFilter( normalize=True )
                    elif filt == u"background":
                        print( "Removing 2D Gaussian background from micrograph" )
                        if not np.any(self.filtSum):
                            self.filtSum = self.imageSum.copy()
                        self.filtSum -= util.backgroundEstimate( self.filtSum )
                    elif filt == u"gausslp":
                        print( "TODO: implement parameters for gauss filter" )
                        if not np.any(self.filtSum):
                            self.filtSum = self.imageSum.copy()
                        self.filtSum = scipy.ndimage.gaussian_filter( self.filtSum, 3.0 )
                self.bench['dose1'] = time.time()
        
        
        self.cleanPrivateVariables()
        pass # End of alignImageStack
        
    def cleanPrivateVariables(self):
        """
        Remove all private ("__") variables so the memory they occupy is released.
        """
        # TODO: go through the code and see if there's anything large leftover.
        try: del self.__FFT2, self.__IFFT2
        except: pass
        try:  del self.__subFFT2, self.__subIFFT2
        except: pass
        try: del self.__imageFFT
        except: pass
        try: del self.__Bfilter
        except: pass
        try: del self.__baseImageFFT, self.__baseMaskFFT, self.__baseSquaredFFT, self.__C
        except: pass
        
    def applyShifts( self ):
        self.bench['shifts0'] = time.time()
        # Apply centroid origin, or origin at frame #0 position?
        if self.originMode == u'centroid':
            centroid = np.mean( self.translations, axis=0 )
            self.translations -= centroid
        # if self.originMode == None do nothing
        
        shifts_round = np.round( self.translations ).astype('int')
        #shifts_remainder = self.translations - shifts_round
        
        # Use RAMutil.imageShiftAndCrop to do a non-circular shift of the images to 
        # integer pixel shifts, then subpixel with Lanczos
        m = self.images.shape[0] # image count
        if self.subPixReg > 1.0 and self.shiftMethod == u'fourier':
            # Fourier gradient subpixel shift
            # Setup FFTs for shifting.
            FFTImage = np.empty( self.shapePadded, dtype=fftw_dtype )
            RealImage = np.empty( self.shapePadded, dtype=fftw_dtype )
            normConst = 1.0 / (self.shapePadded[0]*self.shapePadded[1])
            # Make pyFFTW objects
            _, IFFT2 = util.pyFFTWPlanner( FFTImage, wisdomFile=os.path.join( self.cachePath, "fftw_wisdom.pkl" ), effort = self.fftw_effort, n_threads=self.n_threads, doForward=False )
            [xmesh, ymesh] = np.meshgrid( np.arange(-RealImage.shape[1]/2,RealImage.shape[1]/2) / np.float(RealImage.shape[1] ), 
                np.arange(-RealImage.shape[0]/2,RealImage.shape[0]/2)/np.float(RealImage.shape[0]) )
            twoj_pi = np.complex64( -2.0j * np.pi )
            
            for J in np.arange(0,m):
                # Normalize and reduce to float32
                tX = self.translations[J,1]; tY = ymesh*self.translations[J,0]
                FFTImage = self.__imageFFT[J,:,:] * np.fft.fftshift( nz.evaluate( "exp(twoj_pi * (xmesh*tX + ymesh*tY))")  )
                             
                IFFT2.update_arrays( FFTImage, RealImage ); IFFT2.execute()
                # Normalize and reduce to float32
                if self.images.shape[1] < RealImage.shape[0] or self.images.shape[2] < RealImage.shape[1]:
                    self.images[J,:,:] = np.real( nz.evaluate( "normConst * real(RealImage)" ) ).astype(self.images.dtype)[:self.images.shape[1],:self.images.shape[2]]
                else:
                    self.images[J,:,:] = np.real( nz.evaluate( "normConst * real(RealImage)" ) ).astype(self.images.dtype)
                
                if self.verbose: print( "Correction (fourier) "+ str(np.around(self.translations[J,:],decimals=4))+" applied to image: " + str(J) )
        
        elif self.subPixReg > 1.0 and self.shiftMethod == u'lanczos':
            # Lanczos realspace shifting
            util.lanczosSubPixShiftStack( self.images, self.translations, n_threads=self.n_threads )

            # Original unparallelized version
#            shifts_remainder = self.translations - shifts_round
#            for J in np.arange(0,m):
#                # self.images[J,:,:] = util.imageShiftAndCrop( self.images[J,:,:], shifts_round[J,:] )
#                #Roll the image instead to preserve information in the stack, in case someone deletes the original
#                self.images[J,:,:] = np.roll( np.roll( self.images[J,:,:], shifts_round[J,0], axis=0 ), shifts_round[J,1], axis=1 )
#                
#                self.images[J,:,:] = util.lanczosSubPixShift( self.images[J,:,:], subPixShift=shifts_remainder[J,:], kernelShape=5, lobes=3 )
#                
#                if self.verbose: print( "Correction (lanczos) "+ str(np.around(self.translations[J,:],decimals=4))+" applied to image: " + str(J) )


        else:
            for J in np.arange(0,m):
                # self.images[J,:,:] = util.imageShiftAndCrop( self.images[J,:,:], shifts_round[J,:] )
                #Roll the image instead to preserve information in the stack, in case someone deletes the original
                self.images[J,:,:] = np.roll( np.roll( self.images[J,:,:], shifts_round[J,0], axis=0 ), shifts_round[J,1], axis=1 )

                if self.verbose: print( "Correction (integer) "+ str(shifts_round[J,:])+" applied to image: " + str(J) )
                
        # Also do masks (single-pixel precision only) if seperate for each image
        if not self.masks is None and self.masks.shape[0] > 1:
            for J in np.arange(0,m):
                self.masks[J,:,:] = util.imageShiftAndCrop( self.masks[J,:,:], shifts_round[J,:] )
                
        # Build sum
        self.imageSum = np.sum( self.images, axis=0 )
        # Clean up numexpr pointers
        try: del normConst, tX, tY, twoj_pi
        except: pass
        self.bench['shifts1'] = time.time()
    
    def __lanczosSubPixShiftStack( self ):
        
        tPool = mp.ThreadPool( self.n_threads )

        slices = self.images.shape[0]
        # Build parameters list for the threaded processeses, consisting of index
        tArgs = [None] * slices
        for J in np.arange(slices):
            tArgs[J] = (J, self.images, self.translations)
        
        # All operations are done 'in-place' 
        tPool.map( util.lanczosIndexedShift, tArgs )
        tPool.close()
        tPool.join()
        pass

    def binStack( self, binKernel = 'fourier' ):
        """
        binKernel can be 'lanczos2' or 'fourier', which does a Lanczos resampling or Fourier cropping, 
        respectively.  Lanczos kernel can only resample by powers of 2 at present.  
        
        The Lanczos kernel has some aliasing problems at present so it's use isn't advised yet.
        """
        self.bench['bin0'] = time.time()
        bShape2 = (np.array( self.shapeBinned ) / 2).astype('int')
        binScale = np.array( [self.images.shape[1], self.images.shape[2]] ) / np.array( self.shapeBinned )
        self.pixelsize *= np.mean( binScale )
        print( "Binning stack from %s to %s" % (str(self.images.shape[1:]),str(self.shapeBinned)))
        
        if binKernel == u'lanczos2':
            import math
            binFact = [ np.floor( math.log( binScale[0], 2 ) ) ** 2, np.floor( math.log( binScale[1], 2 ) ) ** 2]
            # Add some error checking if binShape isn't precisely the right size.
            
            print( "binFact = " + str(binFact) )
            
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
            print( windowKernel.shape )
            
            binArray = np.zeros( [self.images.shape[0], self.shapeBinned[0], self.shapeBinned[1]], dtype='float32' )
            for J in np.arange( self.images.shape[0] ):
                # TODO: switch from squarekernel to an interpolator so we can use non-powers of 2
                binArray[J,:,:] = util.squarekernel( scipy.ndimage.convolve( self.images[J,:,:], windowKernel ), 
                    k= binFact[0] )

        elif binKernel == u'fourier':

    
            binArray = np.zeros( [self.images.shape[0], self.shapeBinned[0], self.shapeBinned[1]], dtype='float32' )
            FFTImage = np.zeros( [ self.images.shape[1], self.images.shape[2] ], dtype=fftw_dtype)
            FFTBinned = np.zeros( self.shapeBinned, dtype=fftw_dtype )
            FFT2, _ = util.pyFFTWPlanner( FFTImage, FFTImage, 
                            wisdomFile=os.path.join( self.cachePath, "fftw_wisdom.pkl" ), 
                            effort = self.fftw_effort, n_threads=self.n_threads, doReverse=False )
            _, IFFT2bin = util.pyFFTWPlanner( FFTBinned, FFTBinned, 
                            wisdomFile=os.path.join( self.cachePath, "fftw_wisdom.pkl" ), 
                            effort = self.fftw_effort, n_threads=self.n_threads, doForward=False ) 
    
            
            ImageBinned = np.zeros( self.shapeBinned, dtype=fftw_dtype )
            normConst = 1.0 / (self.shapeBinned[0]*self.shapeBinned[1])
            for J in np.arange( self.images.shape[0] ):
                FFT2.update_arrays( self.images[J,:,:].astype( fftw_dtype ), FFTImage ); FFT2.execute()
                # Crop
                FFTBinned[:bShape2[0],:bShape2[1]] = FFTImage[:bShape2[0],:bShape2[1]]
                FFTBinned[:bShape2[0], -bShape2[1]:] = FFTImage[:bShape2[0], -bShape2[1]:]
                FFTBinned[-bShape2[0]:,:bShape2[1]] = FFTImage[-bShape2[0]:,:bShape2[1]]
                FFTBinned[-bShape2[0]:,-bShape2[1]:] = FFTImage[-bShape2[0]:,-bShape2[1]:]
                
                # Normalize
                FFTBinned *= normConst
                
                # Invert
                IFFT2bin.update_arrays( FFTBinned, ImageBinned ); IFFT2bin.execute()
                
                # De-complexify
                binArray[J,:,:] = np.real( ImageBinned )
                pass
            
        pass
    
        del self.images
        self.images = binArray
        self.bench['bin1'] = time.time()


        
    def padStack( self, padSize=None, interiorPad=0 ):
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
        self.bench['pad0'] = time.time()
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
            if interiorPad > 0:
                self.masks[0, interiorPad:self.shapeOriginal[0]-interiorPad,
                           interiorPad:self.shapeOriginal[1]-interiorPad] = 1.0
            else:
                self.masks[0,:self.shapeOriginal[0], :self.shapeOriginal[1] ] = 1.0
        else:
            if self.masks.shape[1] != self.shapePadded[0] and self.masks.shape[2] != self.shapePadded[1]:
                mmask = self.masks.shape[0]
                paddedMasks = np.zeros( [mmask, padSize[0], padSize[1]], dtype=self.masks.dtype )
                paddedMasks[:,:self.shapeOriginal[0],:self.shapeOriginal[1]] = self.masks
                self.masks = paddedMasks
            pass # else do nothing
        pass
        self.bench['pad1'] = time.time()
    
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
        basinArgs[u"bounds"] = bounds
        basinArgs[u"tol"] = 1E-6
        basinArgs[u"method"] =  u"L-BFGS-B"
        basinArgs[u"args"] = ( hSigma, cdfPeaks )
        # x is [SigmaThres, K, Nu, background]
        x0 = [np.mean(peaksigs), 5.0, 1.0]
        outBasin = scipy.optimize.basinhopping( util.minLogistic, x0, niter=50, minimizer_kwargs=basinArgs )
        
        # Set the logistics curve appropriately.
        self.peaksigThres = outBasin.x[0]
        self.logisticK = outBasin.x[1]
        self.logisticNu = outBasin.x[2]
        
        # Diagnostics (for plotting)
        self.errorDictList[errIndex][u'pdfPeaks'] = pdfPeaks
        self.errorDictList[errIndex][u'cdfPeaks'] = cdfPeaks
        self.errorDictList[errIndex][u'hSigma'] = hSigma
        self.errorDictList[errIndex][u'logisticNu'] = self.logisticNu
        self.errorDictList[errIndex][u'logisticK'] = self.logisticK
        self.errorDictList[errIndex][u'peaksigThres'] = self.peaksigThres
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
        frameFFT = np.empty( self.images.shape[1:], dtype=fftw_dtype )
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
        self.bench['frc0'] = time.time()
        m = self.images.shape[0]
        evenIndices = np.arange(0, m, 2)
        oddIndices = np.arange(1, m, 2)
        
        original_configName = self.files[u'config']
        import uuid
        tempLogName = str(uuid.uuid4() ) + u".zor"
        self.saveConfig( tempLogName ) 
        self.files[u'config'] = original_configName # Restore original configuration file.
        
        evenReg = ImageRegistrator()
        evenReg.loadConfig( tempLogName )
        evenReg.images = self.images[evenIndices,:,:].copy(order='C')
        
        oddReg = ImageRegistrator()
        oddReg.loadConfig( tempLogName )
        oddReg.images = self.images[oddIndices,:,:].copy(order='C')
        
        
        
        if xcorr == u'tri' or xcorr is None:
            if self.masks is None:
                evenReg.masks = util.edge_mask( maskShape=[ self.images.shape[1], self.images.shape[2] ] )
                oddReg.masks = evenReg.masks
            elif self.masks.shape[0] > 1:
                evenReg.masks = self.masks[evenIndices,:,:]
                oddReg.masks = self.masks[oddIndices,:,:]
            elif self.masks.shape[0] == 1:
                evenReg.masks = self.masks
                oddReg.masks = self.masks
            
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
        eoReg.loadConfig( tempLogName )
        eoReg.images = np.empty( [2, evenReg.imageSum.shape[0], evenReg.imageSum.shape[1] ], dtype=float_dtype)
        eoReg.images[0,:,:] = evenReg.imageSum; eoReg.images[1,:,:] = oddReg.imageSum
        eoReg.triMode = u'first'
        
        
        
        try: os.remove( tempLogName )
        except: print( "Could not remove temporary log file: " + tempLogName )

        # This actually aligns the two phase images
        # We use Zorro for this for all methods because we have more trust in the masked, normalized
        # cross correlation
        eoReg.alignImageStack()
        
        # Save the aligned eoReg images for subZorro use
        stackFront = os.path.splitext( self.files[u'sum'] )[0]
        if not 'compressor' in self.files or not bool(self.files['compressor']):
            mrcExt = ".mrc"
        else:
            mrcExt = ".mrcz"
            
        mrcz.writeMRC( evenReg.imageSum, u"%s_even%s" % (stackFront, mrcExt ),
                        compressor=self.files[u'compressor'], clevel=self.files[u'clevel'], n_threads=self.n_threads)
        mrcz.writeMRC( oddReg.imageSum, u"%s_odd%s" % (stackFront, mrcExt ),
                        compressor=self.files[u'compressor'], clevel=self.files[u'clevel'], n_threads=self.n_threads) 
        
        eoReg.tiledFRC( eoReg.images[0,:,:], eoReg.images[1,:,:], 
                       trans=np.hstack( [self.transEven, self.transOdd] ), box=box, overlap=overlap )
        
        self.FRC2D = eoReg.FRC2D
        self.FRC = eoReg.FRC
        
        if self.saveC:
            self.evenC = evenReg.C
            self.oddC = oddReg.C

        self.bench['frc1'] = time.time()
        return evenReg, oddReg
        
    def lazyFouRingCorr( self, box=[512,512], overlap=0.5, debug=False ):
        """
        Computes the FRC from the full stack, taking even and odd frames for the half-sums
        These are not independent half-sets! ... but it still gives us a decent impression 
        of alignment success or failure, and it's very fast.
        """
        self.bench['frc0'] = time.time()
        m = self.images.shape[0]
        evenIndices = np.arange(0, m, 2)
        oddIndices = np.arange(1, m, 2)                     
        
        evenSum = np.sum( self.images[evenIndices,:,:], axis=0 )
        oddSum = np.sum( self.images[oddIndices,:,:], axis=0 )
        
        self.tiledFRC( evenSum, oddSum, box=box, overlap=overlap )
        # Force the length to be box/2 because the corners are poorly sampled
        self.FRC = self.FRC[: np.int(box[0]/2)]
        self.bench['frc1'] = time.time()

    
    def tiledFRC( self, Image1, Image2, trans=None, box=[512,512], overlap=0.5 ):
        """
        Pass in two images, which are ideally averages from two independently processed half-sets. 
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

                         
        hann = util.apodization( name=u'hann', shape=box ).astype(float_dtype)
        tilesX = np.floor( np.float( Image1.shape[1] - cropLim[1] - cropLim[3] - box[1])/ box[1] / (1.0-overlap) ).astype('int')
        tilesY = np.floor( np.float( Image1.shape[0] - cropLim[0] - cropLim[2] - box[0])/ box[0] / (1.0-overlap) ).astype('int')
        if self.verbose >= 2:
            print( "Tiles for FRC: " + str( tilesX) + ":" + str(tilesY))
            
        FFTEven = np.zeros( box, dtype=fftw_dtype )
        FFTOdd = np.zeros( box, dtype=fftw_dtype )
        normConstBox = np.float32( 1.0 / FFTEven.size**2 )
        FRC2D = np.zeros( box, dtype=float_dtype )
        for I in np.arange(0,tilesY):
            for J in np.arange(0,tilesX):
                offset = np.array( [ I*box[0]*(1.0-overlap)+cropLim[0], J*box[1]*(1.0-overlap)+cropLim[1] ]).astype('int')
                
                tileEven = (hann*Image1[offset[0]:offset[0]+box[0], offset[1]:offset[1]+box[1] ]).astype(fftw_dtype)
                FFT2.update_arrays( tileEven, FFTEven ); FFT2.execute()
                tileOdd = (hann*Image2[offset[0]:offset[0]+box[0], offset[1]:offset[1]+box[1] ]).astype(fftw_dtype)
                FFT2.update_arrays( tileOdd, FFTOdd ); FFT2.execute()
    
                FFTOdd *= normConstBox
                FFTEven *= normConstBox
                
                # Calculate the normalized FRC in 2-dimensions
                # FRC2D += nz.evaluate( "real(FFTEven*conj(FFTOdd)) / sqrt(real(abs(FFTOdd)**2) * real(abs(FFTEven)**2) )" )
                # Some issues with normalization?
                FRC2D += nz.evaluate( "real(FFTEven*conj(FFTOdd)) / sqrt(real(FFTOdd*conj(FFTOdd)) * real(FFTEven*conj(FFTEven)) )" )
              
        # Normalize
        FRC2D /= FRC2D[0,0]
        FRC2D = np.fft.fftshift( FRC2D )
        
        rotFRC, _ = util.rotmean( FRC2D )
        self.FRC = rotFRC
        self.FRC2D = FRC2D

    def localFRC( self, box=[256,256], overlap=0.5 ):
        # Only work on the even and odd frames?
        m = self.images.shape[0]
        box2 = (np.array(box)/2).astype('int')
        evenIndices = np.arange(0, m, 2)
        oddIndices = np.arange(1, m, 2)  
        
        center = 2048
        
        evenBox = np.sum( self.images[evenIndices, center-box2[0]:center+box2[0], center-box2[1]:center+box2[1] ], axis=0 )
        oddBox = np.sum( self.images[oddIndices, center-box2[0]:center+box2[0], center-box2[1]:center+box2[1] ], axis=0 )
        FFTEven = np.zeros( box, dtype=fftw_dtype )
        FFTOdd = np.zeros( box, dtype=fftw_dtype )
        
        normConstBox = np.float32( 1.0 / FFTEven.size**2 )
        
        FFT2, _ = util.pyFFTWPlanner( np.zeros(box, dtype=fftw_dtype), 
                             wisdomFile=os.path.join( self.cachePath, u"fftw_wisdom.pkl" ) , n_threads = self.n_threads, 
                             effort=self.fftw_effort, doReverse=False )
        FFT2.update_arrays( evenBox, FFTEven ); FFT2.execute()
        FFT2.update_arrays( oddBox, FFTOdd ); FFT2.execute()
        
        FFTOdd *= normConstBox
        FFTEven *= normConstBox
        
        FRC2D = nz.evaluate( "real(FFTEven*conj(FFTOdd)) / sqrt(real(FFTOdd*conj(FFTOdd)) * real(FFTEven*conj(FFTEven)) )" )
        FRC2D /= FRC2D[0,0]
        FRC2D = np.fft.fftshift( FRC2D )
        
        rotFRC, _ = util.rotmean( FRC2D )
        
        plt.figure()
        plt.plot( rotFRC )
        plt.title( "Local FRC over box = " + str(box) )
      
     
        
    def doseFilter( self, normalize=False ):
        """
        This is a port from Grant's electron_dose.f90 from UnBlur.  It uses fixed critical dose factors
        to apply filters to each image based on their accumulated dose.  We can potentially use 
        high-dose detector like the Medipix to determine these dose factors in advance, on a per-protein 
        basis.  However in that case the assumption is that radiation damage measured from diffraction of crystals 
        results accurately contrast, which is perhaps not true for single particle.
        
        dosePerFrame by default is estimated from the data. If zorroReg.gain = None, we assume 
        the input numbers are in electrons.  
        
        missingStartFrames is for data that has the starting x frames removed. It will guess (based on the gain if 
        present) the missing total dose.
        
        Paramaters are set as follows:
        
            zorroReg.doseFiltParam = [dosePerFrame, critDoseA, critDoseB, critDoseC, cutoffOrder, missingStartFrame]
        
        When using a tight objective aperture and a GIF and thicker ice it's best to record the dose 
        rate in a hole and set self.doseFiltParam[0] appropriately, in terms of electrons per pixel per frame
        
        Also fits a 2D gaussian to the image background and subtracts it.  This improves performance of particle 
        picking tools such as Gauto match, and keeps all the intensities uniform for Relion's group scale correction.
        This can be used with Zorro's particle extraction routines.  
        """
        # print( "DEBUG 1: doseFilter: imageSum # nans %d" % np.sum(np.isnan(self.imageSum) ) )
        critDoseA = np.float32( self.doseFiltParam[1] )
        critDoseB = np.float32( self.doseFiltParam[2] )
        critDoseC = np.float32( self.doseFiltParam[3] )
        cutoffOrder = np.float32( self.doseFiltParam[4] )
        
        if not bool( self.voltage ):
            self.METAstatus = u"error"
            self.saveConfig()
            raise ValueError( "Accelerating voltage not set in advance for dose filtering" )
        voltageScaling = np.float32( np.sqrt( self.voltage / 300.0  ) ) # increase in radiolysis at lower values.
        
        
        # It looks like they build some mesh that is sqrt(qxmesh + qymesh) / pixelsize
        # I think this is probably just qmesh in inverse Angstroms (keeping in mind Zorro's internal
        # pixelsize is nm)
        m = self.images.shape[0]
        N = self.shapePadded[0]
        M = self.shapePadded[1]
        invPSx = np.float32( 1.0 / (M*(self.pixelsize*10)) )
        invPSy = np.float32( 1.0 / (N*(self.pixelsize*10)) )
        
        xmesh, ymesh = np.meshgrid( np.arange(-M/2,M/2), np.arange(-N/2,N/2))
        xmesh = xmesh.astype(float_dtype);  ymesh = ymesh.astype(float_dtype)
        #print( "xmesh.dtype: %s" % xmesh.dtype )
        qmesh = nz.evaluate( "sqrt(xmesh*xmesh*(invPSx**2) + ymesh*ymesh*(invPSy**2))" )
        #print( "qmesh.dtype: %s" % qmesh.dtype )
        qmesh = np.fft.fftshift( qmesh )
        
        #print( "qmesh.dtype: %s" % qmesh.dtype )
        
        # Since there's a lot of hand waving, let's assume dosePerFrame is constant
        # What about on a GIF where the observed dose is lower due to the filter?  That can be incorporated 
        # with a gain estimator.
        if self.doseFiltParam[0] == None:
            totalDose = np.mean( self.imageSum ) 
            dosePerFrame = totalDose / m
            missingDose = dosePerFrame * np.float32( self.doseFiltParam[5] )
        else:
            dosePerFrame = self.doseFiltParam[0]
            
        accumDose = np.zeros( m + 1, dtype=float_dtype ) 
        accumDose[1:] = np.cumsum( np.ones(m) * dosePerFrame )
        accumDose += missingDose
        # optimalDose = 2.51284 * critDose
        
        critDoseMesh = nz.evaluate( "voltageScaling*(critDoseA * qmesh**critDoseB + critDoseC)" )
        #critDoseMesh[N/2,M/2] = 0.001 * np.finfo( 'float32' ).max
        critDoseMesh[ np.int(N/2), np.int(M/2)] = critDoseMesh[ np.int(N/2), np.int(M/2)-1]**2
        #print( "critDoseMesh.dtype: %s" % critDoseMesh.dtype )

        # We probably don't need an entire mesh here...
        qvect = (np.arange(0,self.shapePadded[0]/2) * np.sqrt( invPSx*invPSy ) ).astype( float_dtype )
        optiDoseVect = np.zeros( int(self.shapePadded[0]/2), dtype=float_dtype )
        optiDoseVect[1:] = np.float32(2.51284)*voltageScaling*(critDoseA * qvect[1:]**critDoseB + critDoseC)
        optiDoseVect[0] = optiDoseVect[1]**2
        #print( "optiDoseVect.dtype: %s" % optiDoseVect.dtype )
        
        
        padWidth = np.array(self.shapePadded) - np.array(self.imageSum.shape)
        doseFilteredSum = np.zeros( self.shapePadded, dtype=fftw_dtype )
        filterMag = np.zeros( self.shapePadded, dtype=float_dtype )
        FFTimage = np.empty( self.shapePadded, dtype=fftw_dtype )
        # zorroReg.filtSum = np.zeros_like( zorroReg.imageSum )
        FFT2, IFFT2 = util.pyFFTWPlanner( doseFilteredSum, wisdomFile=os.path.join( self.cachePath, "fftw_wisdom.pkl" ) , 
                                     effort = self.fftw_effort, n_threads=self.n_threads )
        

        for J in np.arange(0,m):
            print( "Filtering for dose: %.2f e/A^2"% (accumDose[J+1]/(self.pixelsize*10)**2) )
            doseFinish = accumDose[J+1] # Dose at end of frame period
            doseStart = accumDose[J] # Dose at start of frame period
            # qmesh is in reciprocal angstroms, so maybe I can ignore how they build the mesh and 
            # use a matrix meshgrid
            
            minusHalfDose = np.float32( -0.5*doseFinish )
            filt = nz.evaluate( "exp( minusHalfDose/critDoseMesh)")
            #print( "filt.dtype: %s" % filt.dtype )
            thresQ = qvect[ np.argwhere( np.abs(doseFinish - optiDoseVect) < np.abs(doseStart - optiDoseVect) )[-1] ]
            
            # thres = nz.evaluate( "abs(doseFinish - optiDoseMesh) < abs(doseStart - optiDoseMesh)" )
            # This filter step is slow, try to do this analytically?  Can we find the radius from the above equation?
            # thres = scipy.ndimage.gaussian_filter( thres.astype(zorro.float_dtype), cutoffSigma )
            thres = nz.evaluate( "exp( -(qmesh/thresQ)**cutoffOrder )" )
            #print( "thres.dtype: %s" % thres.dtype )
            #print( "qmesh.dtype: %s" % qmesh.dtype )
            #print( "thresQ.dtype: %s" % thresQ.dtype )
            #print( "cutoffOrder.dtype: %s" % cutoffOrder.dtype )
            
            # Numpy's pad is also quite slow
            paddedImage = np.pad( self.images[J,:,:].astype(fftw_dtype),
                                     ((0,padWidth[0]),(0,padWidth[1])), mode=symmetricPad   )
                                     
            FFT2.update_arrays( paddedImage, FFTimage ); FFT2.execute()
            # print( "FFTimage.dtype: %s" % FFTimage.dtype )
            # Adding Fourier complex magntiude works fine
            if bool(normalize):
                currentFilter = nz.evaluate( "thres*filt" )
                filterMag += currentFilter
                doseFilteredSum += nz.evaluate( "FFTimage * currentFilter" )
                
            else:
                doseFilteredSum += nz.evaluate( "FFTimage * thres * filt" )
        pass
        
        # print( "doseFilteredSum.dtype: %s" % doseFilteredSum.dtype )
        if bool( normalize ):
            alpha = np.float32(1.0) # Prevent divide by zero errors by adding a fixed factor of unity before normalizing.
            filterMag = np.float32(1.0) / ( filterMag + alpha )
            # Using FFTimage as a temporary array 
            IFFT2.update_arrays( doseFilteredSum*filterMag, FFTimage ); IFFT2.execute()
        else:
            # Using FFTimage as a temporary array 
            IFFT2.update_arrays( doseFilteredSum, FFTimage ); IFFT2.execute()
        self.filtSum = np.abs( FFTimage[:self.imageSum.shape[0],:self.imageSum.shape[1]] )
        # print( "filtSum.dtype: %s" % self.filtSum.dtype )


        
        del invPSx, invPSy, qmesh, optiDoseVect, doseFinish, doseStart, critDoseA, critDoseB, critDoseC, 
        del voltageScaling, filt, thres, thresQ, cutoffOrder, minusHalfDose
     
        
    def hotpixFilter( self, cutoffLower=None, cutoffUpper=None, neighbourThres = 0.01 ):
        """
        Identifies and removes hot pixels using a stocastic weighted approach.
        replaced with a Gaussian filter.  Hot pixels do not affect Zorro too much 
        due to the intensity-normalized cross-correlation but the tracks of the 
        hot pixels do upset other software packages.
        
        PSF is used to provide a camera-specific PSF to filter hot pixels.  If 
        you have an MTF curve for a detector we can provide a psf tailored to that
        particular device, otherwise use None for a uniform filter.
        """
        self.bench['hot0'] = time.time()

        # 3 x 3 kernels
        if self.hotpixInfo[u"psf"] == u"K2":
            psf = np.array( [0.0, 0.173235968], dtype=float_dtype )
        else: # default to uniform filter
            psf = np.array( [0.0, 1.0], dtype=float_dtype )
        
        psfKernel = np.array( [  [psf[1]*psf[1], psf[1], psf[1]*psf[1] ],
               [psf[1], 0.0, psf[1] ],
               [psf[1]*psf[1], psf[1], psf[1]*psf[1] ]], dtype=float_dtype  )
        psfKernel /= np.sum( psfKernel )
        
        if self.images.ndim == 2: 
            # Mostly used when processing flatfields for gain reference normalization
            self.images = np.reshape( self.images, [1, self.images.shape[0], self.images.shape[1]])
            MADE_3D = True
        else:
            MADE_3D = False
        
        
        unalignedSum = np.sum( self.images, axis=0 )
        sumMean = np.mean( unalignedSum )
        poissonStd = np.sqrt( sumMean )
        
        histBins = np.arange( np.floor( sumMean - self.hotpixInfo[u"maxSigma"]*poissonStd)-0.5, np.ceil(sumMean+self.hotpixInfo[u"maxSigma"]*poissonStd)+0.5, 1 )
        unalignedHist, unalignedCounts = np.histogram( unalignedSum, histBins )
        unalignedHist = unalignedHist.astype(float_dtype); 
        
        # Make unalignedCounts bin centers rather than edges
        unalignedCounts = unalignedCounts[:-1].astype(float_dtype) 
        unalignedCounts += 0.5* (unalignedCounts[1]-unalignedCounts[0])
        
        
        # Here we get sigma values from the CDF, which is smoother than the PDF due 
        # to the integration applied.
        cdfHist = np.cumsum( unalignedHist )
        cdfHist /= cdfHist[-1]
        
        ###################################
        # Optimization of mean and standard deviation
        # TODO: add these stats to the object
        
        def errorNormCDF( params ):
            return np.sum( np.abs( cdfHist - 
                scipy.stats.norm.cdf( unalignedCounts, loc=params[0], scale=params[1] ) ) )
        
        bestNorm = scipy.optimize.minimize( errorNormCDF, (sumMean,poissonStd),
                            method="L-BFGS-B", 
                            bounds=((sumMean-0.5*poissonStd,  sumMean+0.5*poissonStd),
                                    (0.7*poissonStd, 1.3*poissonStd) ) )
        #####################################
        
        sigmaFromCDF = np.sqrt(2) * scipy.special.erfinv( 2.0 * cdfHist - 1 ) 
        
        normalSigma = (unalignedCounts - bestNorm.x[0]) / bestNorm.x[1] 
        
        errorNormToCDF = normalSigma - sigmaFromCDF
        keepIndices = ~np.isinf( errorNormToCDF )
        errorNormToCDF = errorNormToCDF[keepIndices]
        normalSigmaKeep = normalSigma[keepIndices]
        

        # Try for linear fits, resort to defaults if it fails
        if not bool(cutoffLower):
            try:
                lowerIndex = np.where( errorNormToCDF > -0.5 )[0][0]
                lowerA = np.array( [normalSigmaKeep[:lowerIndex], np.ones(lowerIndex )] )
                lowerFit = np.linalg.lstsq( lowerA.T, errorNormToCDF[:lowerIndex] )[0]
                cutoffLower = np.float32( -lowerFit[1]/lowerFit[0] )
                self.hotpixInfo[u'cutoffLower'] = float( cutoffLower )
            except:
                print( "zorro.hotpixFilter failed to estimate bound for dead pixels, defaulting to -4.0" )
                cutoffLower = np.float32( self.hotpixInfo['cutoffLower'] )
                
        if not bool(cutoffUpper):
            try:
                upperIndex = np.where( errorNormToCDF < 0.5 )[0][-1]
                upperA = np.array( [normalSigmaKeep[upperIndex:], np.ones( len(normalSigmaKeep) - upperIndex )] )
                upperFit = np.linalg.lstsq( upperA.T, errorNormToCDF[upperIndex:] )[0]
                cutoffUpper = np.float32( -upperFit[1]/upperFit[0] )
                self.hotpixInfo[u'cutoffUpper'] = float( cutoffUpper )
            except:
                print( "zorro.hotpixFilter failed to estimate bound for hot pixels, defaulting to +3.25" )
                cutoffUpper = np.float32( self.hotpixInfo['cutoffUpper'] )
            
            
        unalignedSigma = (unalignedSum - bestNorm.x[0]) / bestNorm.x[1]
        
        
        # JSON isn't serializing numpy types anymore, so we have to explicitely cast them
        self.hotpixInfo[u"guessDeadpix"] = int( np.sum( unalignedSigma < cutoffLower ) )
        self.hotpixInfo[u"guessHotpix"] = int( np.sum( unalignedSigma > cutoffUpper  ) )
        self.hotpixInfo[u"frameMean"] = float( bestNorm.x[0]/self.images.shape[0] )
        self.hotpixInfo[u"frameStd"] = float( bestNorm.x[1]/np.sqrt(self.images.shape[0]) )
        
        print( "Applying outlier pixel filter with sigma limits (%.2f,%.2f), n=(dead:%d,hot:%d)" \
              % (cutoffLower, cutoffUpper, self.hotpixInfo[u"guessDeadpix"],self.hotpixInfo[u"guessHotpix"] ) )
        # Some casting problems here with Python float up-casting to np.float64...
        UnityFloat32 = np.float32( 1.0 )
        logK = np.float32( self.hotpixInfo[u'logisticK'] )
        relax = np.float32( self.hotpixInfo[u'relax'] )
        logisticMask = nz.evaluate( "1.0 - 1.0 / ( (1.0 + exp(logK*(unalignedSigma-cutoffLower*relax)) ) )" )
        logisticMask = nz.evaluate( "logisticMask / ( (1.0 + exp(logK*(unalignedSigma-cutoffUpper*relax)) ) )" ).astype(float_dtype)
        
        convLogisticMask = nz.evaluate( "UnityFloat32 - logisticMask" )
        # So we need 2 masks, one for pixels that have no outlier-neighbours, and 
        # another for joined/neighbourly outlier pixels.
        # I can probably make the PSF kernel smaller... to speed things up.
        neighbourlyOutlierMask = (UnityFloat32 - logisticMask) * scipy.ndimage.convolve( np.float32(1.0) - logisticMask, psfKernel )
        
        """
        Singleton outliers have no neighbours that are also outliers, so we substitute their values 
        with the expected value based on the point-spread function of the detector.
        """
        singletonOutlierMask = nz.evaluate( "convLogisticMask * (neighbourlyOutlierMask <= neighbourThres)" )
        m = self.images.shape[0]
        unalignedMean = nz.evaluate( "unalignedSum/m" )
        psfFiltMean = scipy.ndimage.convolve( unalignedMean, psfKernel ).astype(float_dtype)
        
        
        """
        The neighbourFilt deals with outliers that have near neihbours that are also 
        outliers. This isn't uncommon due to defects in the camera.
        """
        neighbourlyOutlierMask = nz.evaluate( "neighbourlyOutlierMask > neighbourThres" )
        neighbourlyIndices = np.where( nz.evaluate( "neighbourlyOutlierMask > neighbourThres" ) )
        bestMean = bestNorm.x[0] / m
        print( "Number of neighborly outlier pixels: %d" % len(neighbourlyIndices[0]) )
        self.hotpixInfo[u'neighborPix'] = len(neighbourlyIndices[0])
        neighbourFilt = np.zeros_like( psfFiltMean )
        for (nY, nX) in zip( neighbourlyIndices[0], neighbourlyIndices[1] ):
            # We'll use 5x5 here, substituting the bestMean if it's all garbage
            neighbourhood = neighbourlyOutlierMask[nY-1:nY+2,nX-1:nX+2]
            nRatio = np.sum( neighbourhood ) / neighbourhood.size
            if nRatio > 0.66 or nRatio <= 0.001 or np.isnan(nRatio):
                neighbourFilt[nY,nX] = bestMean
            else:
                neighbourFilt[nY,nX] = convLogisticMask[nY,nX]*np.mean(unalignedMean[nY-1:nY+2,nX-1:nX+2][~neighbourhood])
        
        stack = self.images
        self.images = nz.evaluate( "logisticMask*stack + singletonOutlierMask*psfFiltMean + neighbourFilt" )
        
        if u"decorrOutliers" in self.hotpixInfo and self.hotpixInfo[ u"decorrOutliers" ]:
            """
            This adds a bit of random noise to pixels that have been heavily filtered 
            to a uniform value, so they aren't correlated noise.  This should only 
            affect Zorro and Relion movie processing.
            """
            decorrStd = np.sqrt( bestNorm.x[1]**2 / m ) / 2.0
            N_images = self.images.shape[0]
            filtPosY, filtPosX = np.where( logisticMask < 0.5 )
        
            # I don't see a nice way to vectorize this loop.  With a ufunc?
            for J in np.arange( filtPosY.size ):
                self.images[ :, filtPosY[J], filtPosX[J] ] += np.random.normal( \
                            scale=decorrStd*convLogisticMask[filtPosY[J],filtPosX[J]], size=N_images )
                
        
        if MADE_3D:
            self.images = np.squeeze( self.images )
            
        self.bench['hot1'] = time.time()
        del logK, relax, logisticMask, psfFiltMean, stack, UnityFloat32, singletonOutlierMask
        pass
    
    def hotpixFilter_SINGLETON( self, cutoffLower=None, cutoffUpper=None  ):
        """
        Identifies and removes hot pixels using a stocastic weighted approach.
        replaced with a Gaussian filter.  Hot pixels do not affect Zorro too much 
        due to the intensity-normalized cross-correlation but the tracks of the 
        hot pixels do upset other software packages.
        
        PSF is used to provide a camera-specific PSF to filter hot pixels.  If 
        you have an MTF curve for a detector we can provide a psf tailored to that
        particular device, otherwise use None for a uniform filter.
        """
        self.bench['hot0'] = time.time()
        
        if self.hotpixInfo[u"psf"] == u"K2":
            psf = np.array( [0.0, 0.173235968, 0.016518], dtype='float32' )
        else: # default to uniform filter
            psf = np.array( [0.0, 1.0, 1.0], dtype='float32' )
        
        psfKernel = np.array( [  [psf[2]*psf[2], psf[2]*psf[1], psf[2], psf[2]*psf[1], psf[2]*psf[2] ],
               [psf[2]*psf[1], psf[1]*psf[1], psf[1], psf[1]*psf[1], psf[1]*psf[2] ],
               [psf[2], psf[1], 0.0, psf[1], psf[2] ],
               [psf[2]*psf[1], psf[1]*psf[1], psf[1], psf[1]*psf[1], psf[1]*psf[2] ],
               [ psf[2]*psf[2], psf[2]*psf[1], psf[2], psf[2]*psf[1], psf[2]*psf[2] ] ], dtype='float32'  )
        psfKernel /= np.sum( psfKernel )
        
        if self.images.ndim == 2: 
            # Mostly used when processing flatfields for gain reference normalization
            self.images = np.reshape( self.images, [1, self.images.shape[0], self.images.shape[1]])
            MADE_3D = True
        else:
            MADE_3D = False
        
        
        unalignedSum = np.sum( self.images, axis=0 )
        sumMean = np.mean( unalignedSum )
        poissonStd = np.sqrt( sumMean )
        
        
        
        histBins = np.arange( np.floor( sumMean - self.hotpixInfo[u"maxSigma"]*poissonStd)-0.5, np.ceil(sumMean+self.hotpixInfo[u"maxSigma"]*poissonStd)+0.5, 1 )
        unalignedHist, unalignedCounts = np.histogram( unalignedSum, histBins )
        unalignedHist = unalignedHist.astype('float32'); 
        
        # Make unalignedCounts bin centers rather than edges
        unalignedCounts = unalignedCounts[:-1].astype('float32') 
        unalignedCounts += 0.5* (unalignedCounts[1]-unalignedCounts[0])


        # Here we get sigma values from the CDF, which is smoother than the PDF due 
        # to the integration applied.
        cdfHist = np.cumsum( unalignedHist )
        cdfHist /= cdfHist[-1]
        
        ###################################
        # Optimization of mean and standard deviation
        # TODO: add these stats to the object
        
        def errorNormCDF( params ):
            return np.sum( np.abs( cdfHist - 
                scipy.stats.norm.cdf( unalignedCounts, loc=params[0], scale=params[1] ) ) )
        
        bestNorm = scipy.optimize.minimize( errorNormCDF, (sumMean,poissonStd),
                            method="L-BFGS-B", 
                            bounds=((sumMean-0.5*poissonStd,  sumMean+0.5*poissonStd),
                                    (0.7*poissonStd, 1.3*poissonStd) ) )
        # normCDF = scipy.stats.norm.cdf( unalignedCounts, loc=bestNorm.x[0], scale=bestNorm.x[1] )
        #####################################
    
        sigmaFromCDF = np.sqrt(2) * scipy.special.erfinv( 2.0 * cdfHist - 1 ) 
        
        #sumFromHist = np.sum( unalignedHist )
        #meanFromHist = np.float32( np.sum( unalignedHist * unalignedCounts ) / sumFromHist )
        #stdFromHist = np.float32( np.sqrt( np.sum( unalignedHist * unalignedCounts**2 )/ sumFromHist - meanFromHist*meanFromHist  ) )
        #invStdFromHist = np.float32(1.0 / stdFromHist )
        
        normalSigma = (unalignedCounts - bestNorm.x[0]) / bestNorm.x[1] 
        
        # TODO: try to keep these infs from being generated in the first place
        errorNormToCDF = normalSigma - sigmaFromCDF
        keepIndices = ~np.isinf( errorNormToCDF )
        errorNormToCDF = errorNormToCDF[keepIndices]
        # unalignedCountsKeep = unalignedCounts[keepIndices]
        normalSigmaKeep = normalSigma[keepIndices]
    
        # TODO: add try-except, using a fixed error difference if the fitting fails
        if not bool(cutoffLower):
            try:
                lowerIndex = np.where( errorNormToCDF > -0.5 )[0][0]
                lowerA = np.array( [normalSigmaKeep[:lowerIndex], np.ones(lowerIndex )] )
                lowerFit = np.linalg.lstsq( lowerA.T, errorNormToCDF[:lowerIndex] )[0]
                cutoffLower = np.float32( -lowerFit[1]/lowerFit[0] )
            except:
                print( "zorro.hotpixFilter failed to estimate bound for dead pixels, defaulting to -4.0" )
                cutoffLower = np.float32( -4.0 )
                
        if not bool(cutoffUpper):
            try:
                upperIndex = np.where( errorNormToCDF < 0.5 )[0][-1]
                upperA = np.array( [normalSigmaKeep[upperIndex:], np.ones( len(normalSigmaKeep) - upperIndex )] )
                upperFit = np.linalg.lstsq( upperA.T, errorNormToCDF[upperIndex:] )[0]
                cutoffUpper = np.float32( -upperFit[1]/upperFit[0] )
            except:
                print( "zorro.hotpixFilter failed to estimate bound for hot pixels, defaulting to +3.25" )
                cutoffLower = np.float32( 3.25 )
            
        unalignedSigma = (unalignedSum - bestNorm.x[0]) / bestNorm.x[1]

        print( "Applying progressive outlier pixel filter with sigma limits (%.2f,%.2f)" % (cutoffLower, cutoffUpper) )
        # JSON isn't serializing numpy types anymore, so we have to explicitely cast them
        self.hotpixInfo[u'cutoffLower'] = float( cutoffLower )
        self.hotpixInfo[u'cutoffUpper'] = float( cutoffUpper )
        self.hotpixInfo[u"guessDeadpix"] = int( np.sum( unalignedSigma < cutoffLower ) )
        self.hotpixInfo[u"guessHotpix"] = int( np.sum( unalignedSigma > cutoffUpper  ) )
        self.hotpixInfo[u"frameMean"] = float( bestNorm.x[0]/self.images.shape[0] )
        self.hotpixInfo[u"frameStd"] = float( bestNorm.x[1]/np.sqrt(self.images.shape[0]) )
        
        logK = np.float32( self.hotpixInfo[u'logisticK'] )
        relax = np.float32( self.hotpixInfo[u'relax'] )
        logisticMask = nz.evaluate( "1.0 - 1.0 / ( (1.0 + exp(logK*(unalignedSigma-cutoffLower*relax)) ) )" )
        
        logisticMask = nz.evaluate( "logisticMask / ( (1.0 + exp(logK*(unalignedSigma-cutoffUpper*relax)) ) )" ).astype('float32')
        
        # So we need 2 masks, one for pixels that have no outlier-neighbours, and 
        # another for joined/neighbourly outlier pixels.
        singletonOutlierMask = scipy.ndimage.convolve( logisticMask, np.ones_like(psfKernel) )
        
        
        # Some casting problems here with Python float up-casting to np.float64...
        UnityFloat32 = np.float32( 1.0 )
        
        psfFiltMean = scipy.ndimage.convolve( unalignedSum/self.images.shape[0], psfKernel ).astype('float32')
        
        
        stack = self.images
        nz.evaluate( "(UnityFloat32-logisticMask) *stack + logisticMask*psfFiltMean" )
        
        
        if u"decorrOutliers" in self.hotpixInfo and self.hotpixInfo[ u"decorrOutliers" ]:
            """
            This adds a bit of random noise to pixels that have been heavily filtered 
            to a uniform value, so they aren't correlated noise.  This should only 
            affect Zorro and Relion movie processing.
            """
            decorrStd = np.std( self.images[0,:,:] )
            N_images = self.images.shape[0]
            filtPosY, filtPosX = np.where( logisticMask < 0.98 )

            # I don't see a nice way to vectorize this loop.  With a ufunc?
            for J in np.arange( filtPosY.size ):
                self.images[ :, filtPosY[J], filtPosX[J] ] += np.random.normal( scale=decorrStd, size=N_images )
            

        if MADE_3D:
            self.images = np.squeeze( self.images )
            
        self.bench['hot1'] = time.time()
        del logK, relax, logisticMask, psfFiltMean, stack, UnityFloat32, singletonOutlierMask

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

        yendcrop = np.minimum( np.floor( trans[:,0].min() ), 0 ).astype('int')
        if yendcrop == 0:
            yendcrop = None
        xendcrop = np.minimum( np.floor( trans[:,1].min() ), 0 ).astype('int')
        if xendcrop == 0:
            xendcrop = None
        ystartcrop = np.maximum( np.ceil( trans[:,0].max() ), 0 ).astype('int')
        xstartcrop = np.maximum( np.ceil( trans[:,1].max() ), 0 ).astype('int')
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
        
    def execGCTF( self, movieMode=False, movieFrameToAverage=8, movieFit=0, movieType=1  ):
        """
        Calls GCTF.  
        
        I.e. movieMode=True
        # Movie options to calculate defocuses of each frame:  
        # --mdef_aveN         8                 Average number of moive frames for movie or particle stack CTF refinement
        # --mdef_fit          0                 0: no fitting; 1: linear fitting defocus changes in Z-direction
        # --mdef_ave_type     0                 0: coherent average, average FFT with phase information(suggested for movies); 1:incoherent average, only average amplitude(suggested for particle stack); 
        """
        self.bench['ctf0'] = time.time()
        print( "   Kai Zhang, 'Gctf: real-time CTF determination and correction',  J. Struct. Biol., 193(1): 1-12, (2016)" )
        print( "   http://www.sciencedirect.com/science/article/pii/S1047847715301003" )

        if self.cachePath is None:
            self.cachePath = "."
        try: os.umask( self.umask ) # Why is Python not using default umask from OS?
        except: pass
            
        stackBase = os.path.splitext( os.path.basename( self.files[u'stack'] ) )[0]
        
        mrcName = os.path.join( self.cachePath, stackBase + u"_gctf.mrc" )
        mrcFront = os.path.splitext( mrcName )[0]
        diagOutName = mrcFront + u".ctf"
        logName = mrcFront + u"_ctffind3.log"
        epaName = mrcFront + u"_EPA.log"
        
        if bool( movieMode ):
            # Write an MRCS
            mrcz.writeMRC( self.images, mrcName )
            # Call GCTF

            gctf_exec = "gctf %s --apix %f --kV %f --cs %f --do_EPA 1 --mdef_ave_type 1 --logsuffix  _ctffind3.log " % (mrcName, self.pixelsize*10, self.voltage, self.C3 )
            gctf_exec += " --mdef_aveN %d --mdef_fit %d --mdef_ave_type %d" %( movieFrameToAverage, movieFit, movieType )
        else: # No movieMode
            if not np.any( self.imageSum ):
                raise AttributeError( "Error in execGCTF: No image sum found" )
            mrcz.writeMRC( self.imageSum, mrcName )
            # Call GCTF
            gctf_exec = "gctf %s --apix %f --kV %f --cs %f --do_EPA 1 --logsuffix  _ctffind3.log " % (mrcName, self.pixelsize*10, self.voltage, self.C3 )

        # Need to redirect GCTF output to null because it's formatted with UTF-16 and this causes Python 2.7 problems.
        devnull = open(os.devnull, 'w' )
        subprocess.call( gctf_exec, shell=True, stdout=devnull, stderr=devnull )
        # sub = subprocess.Popen( gctf_exec, shell=True )
        #sub.wait() 

        # Diagnostic image ends in .ctf
        self.CTFDiag = mrcz.readMRC( diagOutName )[0]

        # Parse the output _ctffind3.log for the results
        with open( logName, 'r' ) as fh:
            logCTF = fh.readlines()

        ctf = logCTF[-5].split()
        self.CTFInfo[u'DefocusU'] = float( ctf[0] )
        self.CTFInfo[u'DefocusV'] = float( ctf[1] )
        self.CTFInfo[u'DefocusAngle'] = float( ctf[2] )
        self.CTFInfo[u'CtfFigureOfMerit'] = float( ctf[3] )
        self.CTFInfo[u'FinalResolution'] = float( logCTF[-3].split()[-1] )
        self.CTFInfo[u'Bfactor'] = float( logCTF[-2].split()[-1] )

        # Output compact _ctffind3.log
        self.saveRelionCTF3( )
        
        # Remove temporary files and log file
        try: os.remove( diagOutName )
        except: pass
        try: os.remove( logName ) # Technically we could keep this.
        except: pass
        try: os.remove( mrcName )
        except: pass
        try: os.remove( epaName )
        except: pass
        self.bench['ctf1'] = time.time()
     
    def execCTFFind41( self, movieMode=False, box_size = 1024, contrast=0.067, 
                     min_res=50.0, max_res=4.0, 
                     min_C1=5000.0, max_C1=45000.0, C1_step = 500.0, 
                     A1_tol = 500.0 ):
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
        """
        self.bench['ctf0'] = time.time()
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
        
        print( "Calling CTFFIND4.1 for %s" % self.files['stack']  )
        print( "   written by Alexis Rohou: http://grigoriefflab.janelia.org/ctffind4" )
        print( "   http://biorxiv.org/content/early/2015/06/16/020917" )
        
        ps = self.pixelsize * 10.0
        min_res = np.min( [min_res, 50.0] )
        
        try: os.umask( self.umask ) # Why is Python not using default umask from OS?
        except: pass
        
        if self.cachePath is None:
            self.cachePath = "."
            
        # Force trailing slashes onto cachePatch
        stackBase = os.path.splitext( os.path.basename( self.files[u'stack'] ) )[0]
            
        diagOutName = os.path.join( self.cachePath, stackBase + u".ctf" )
         
        try: 
            mrcName = os.path.join( self.cachePath, stackBase + u"_ctf4.mrc" )
            if bool(movieMode):
                mrcz.writeMRC( self.images, mrcName )
                number_of_frames_to_average = 1
            else:
                mrcz.writeMRC( self.imageSum, mrcName )
        except:
            print( "Error in exporting MRC file to CTFFind4.1" )
            return
         
        # flags = "--amplitude-spectrum-input --filtered-amplitude-spectrum-input"
        flags = "" # Not using any flags
        find_additional_phase_shift = "no"
        knownAstig = "no"
        largeAstig = "no"
        restrainAstig = "yes"
        expertOptions = "no"
        
        ctfexec = ( "ctffind " + flags + " << STOP_PARSING \n" + mrcName + "\n" )
        if bool(movieMode):
             ctfexec = ctfexec + "yes\n" + str(number_of_frames_to_average + "\n" )
         
        ctfexec = (ctfexec + diagOutName + "\n" + str(ps) + "\n" + str(self.voltage) + "\n" +
            str(self.C3) + "\n" + str(contrast) + "\n" + str(box_size) + "\n" +
            str(min_res) + "\n" + str(max_res) + "\n" + str(min_C1) + "\n" + 
            str(max_C1) + "\n" + str(C1_step) + "\n" + str(knownAstig) + "\n" + 
            str(largeAstig) + "\n" + str(restrainAstig) + "\n" +
            str(A1_tol) + "\n" + find_additional_phase_shift + "\n" +
            str(expertOptions) )    
        ctfexec = ctfexec + "\nSTOP_PARSING"

        subprocess.call( ctfexec, shell=True )

        
        try:
            logName = os.path.join( self.cachePath, stackBase + ".txt" )
            print( "Trying to load from: " + logName )
            # Log has 5 comment lines, then 1 header, and
            # Micrograph number, DF1, DF2, Azimuth, Additional Phase shift, CC, and max spacing fit-to
            CTF4Results = np.loadtxt(logName, comments='#', skiprows=1 )
            self.CTFInfo[u'DefocusU'] = float( CTF4Results[1] )
            self.CTFInfo[u'DefocusV'] = float( CTF4Results[2] )
            self.CTFInfo[u'DefocusAngle'] = float( CTF4Results[3] )
            self.CTFInfo[u'AdditionalPhaseShift'] = float( CTF4Results[4] )
            self.CTFInfo[u'CtfFigureOfMerit'] = float( CTF4Results[5] )
            self.CTFInfo[u'FinalResolution'] = float( CTF4Results[6] )
            
            self.CTFDiag = mrcz.readMRC( diagOutName )[0]
            
        except:
            print( "CTFFIND4 likely core-dumped, try different input parameters?" )
        pass
        # Write a RELION-style _ctffind3.log file, with 5 um pixel size...
        self.saveRelionCTF3()
            
        # TODO: having trouble with files not being deletable, here.  Is CTFFIND4 holding them open?  Should 
        # I just pause for a short time?
        time.sleep(0.5) # DEBUG: try and see if temporary files are deletable now.
        try: os.remove( mrcName )
        except IOError: 
            print( "Could not remove temporary file: " + str(IOError) )
        try: os.remove( diagOutName )
        except IOError: 
            print( "Could not remove temporary file: " + str(IOError) )
        # Delete CTF4 logs
        try: os.remove( os.path.join( self.cachePath, stackBase + "_avrot.txt") )
        except: pass
        try: os.remove( logName )
        except: pass
        try: os.remove( os.path.join( self.cachePath, stackBase + ".ctf" ) )
        except: pass
        self.bench['ctf1'] = time.time()
                                
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
        self.bench['ctf0'] = time.time()
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
        
        try: os.umask( self.umask ) # Why is Python not using default umask from OS?
        except: pass
        
        if self.cachePath is None:
            self.cachePath = "."
            
        # Force trailing slashes onto cachePatch
        stackBase = os.path.splitext( os.path.basename( self.files[u'stack'] ) )[0]
            
        diagOutName = os.path.join( self.cachePath, stackBase + u".ctf" )
         
        try: 
            mrcName = os.path.join( self.cachePath, stackBase + u"_ctf4.mrc" )
            if movieMode:
                input_is_a_movie = 'true'
                mrcz.writeMRC( self.images, mrcName )
                number_of_frames_to_average = 1
            else:
                input_is_a_movie = 'false'
                mrcz.writeMRC( self.imageSum, mrcName )
        except:
            print( "Error in exporting MRC file to CTFFind4" )
            return
         
        # flags = "--amplitude-spectrum-input --filtered-amplitude-spectrum-input"
        flags = "" # Not using any flags
        find_additional_phase_shift = 'false'
        
        ctfexec = ( "ctffind " + flags + " << STOP_PARSING \n" + mrcName )
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
            CTF4Results = np.loadtxt(logName, comments='#', skiprows=1 )
            self.CTFInfo[u'DefocusU'] = float( CTF4Results[1] )
            self.CTFInfo[u'DefocusV'] = float( CTF4Results[2] )
            self.CTFInfo[u'DefocusAngle'] = float( CTF4Results[3] )
            self.CTFInfo[u'AdditionalPhaseShift'] = float( CTF4Results[4] )
            self.CTFInfo[u'CtfFigureOfMerit'] = float( CTF4Results[5] )
            self.CTFInfo[u'FinalResolution'] = float( CTF4Results[6] )
            
            self.CTFDiag = mrcz.readMRC( diagOutName )[0]
            
        except IOError:
            print( "CTFFIND4 likely core-dumped, try different input parameters?" )
        pass
        # Write a RELION-style _ctffind3.log file, with 5 um pixel size...
        self.saveRelionCTF3()
            
        # TODO: having trouble with files not being deletable, here.  Is CTFFIND4 holding them open?  Should 
        # I just pause for a short time?
        time.sleep(0.5) # DEBUG: try and see if temporary files are deletable now.
        try: os.remove( mrcName )
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
        self.bench['ctf1'] = time.time()
       
    def saveRelionCTF3( self ):
        # Saves the results from CTF4 in a pseudo-CTF3 log that RELION 1.3/1.4 can handle
        # Relevant code is in ctffind_runner.cpp, in the function getCtffindResults() (line 248)
        # Relion searchs for: 
        #   "CS[mm], HT[kV], AmpCnst, XMAG, DStep[um]"
        # and
        #      DFMID1      DFMID2      ANGAST          CC
        #
        #    15876.71    16396.97       52.86     0.10179  Final Values
    
        # Mag goes from micrometers of detector pixel size, to specimen pixel size (in nm)
        amp_contrast = self.CTFInfo[u'AmplitudeContrast']
        
        if bool(self.detectorPixelSize):
            dstep = self.detectorPixelSize # Assumed to be in microns
        else:
            dstep = 5.0 # default value of 5.0 microns, Relion-2 doesn't use it anyway...
        
        mag = (dstep*1E-6) / (self.pixelsize*1E-9)
        
        if self.files[u'sum'] != None:
            sumFront = os.path.splitext( self.files[u'sum'] )[0]
        else:
            sumFront = os.path.splitext( self.files[u'stack'] )[0]
        # Check to see if the sum directory exists already or not
        sumDir = os.path.split( sumFront )[0]
        if bool(sumDir) and not os.path.isdir( sumDir ):
            os.mkdir( sumDir )         
        
        self.files[u'ctflog'] = sumFront + u"_ctffind3.log"
        logh = open( self.files[u'ctflog'], "w" )
        
        logh.write( u"CS[mm], HT[kV], AmpCnst, XMAG, DStep[um]\n" )
        logh.write( u"%.2f"%self.C3 + u" %.1f"%self.voltage + u" " + 
            str(amp_contrast) + u" %.1f" %mag + u" %.2f"%dstep + u"\n" )
        
        try:
            logh.write( u"%.1f"%self.CTFInfo['DefocusU']+ u" %.1f"%self.CTFInfo['DefocusV'] 
                + u" %.4f"%self.CTFInfo['DefocusAngle']+ u" %.4f"%self.CTFInfo['CtfFigureOfMerit'] 
                + u" Final Values\n ")
        except:
            print( "Warning: Could not write CTFInfo to ctf3-style log, probably CTF estimation failed" )
        logh.close()
        pass
    
    def loadData( self, stackNameIn = None, target=u"stack", leading_zeros=0, useMemmap=False ):
        """
        Import either a sequence of DM3 files, a MRCS stack, a DM4 stack, or an HDF5 file.
        
        Target is a string representation of the member name, i.e. 'images', 'imageSum', 'C0'
        
        Files can be compressed with 'lbzip2' (preferred) or 'pigz' with extension '.bz2' or '.gz'
        
        On Windows machines you must have 7-zip in the path to manage compression, and 
        only .bz2 is supported
        
        filename can be an absolute path name or relative pathname.  Automatically 
        assumes file format based on extension.
        """
        self.bench['loaddata0'] = time.time() 
        # import os
        from os.path import splitext
        
        if stackNameIn != None:
            self.files[target] = stackNameIn
            
        #### DECOMPRESS FILE ####
        # This will move the file to the cachePath, so potentially could result in some confusion
        self.files[target] = util.decompressFile( self.files[target], outputDir = self.cachePath )
        
        
        [file_front, file_ext] = splitext( self.files[target] )
        
        #### IMAGE FILES ####
        if file_ext == u".dm3" :
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
        elif file_ext == u'.tif' or file_ext == u'.tiff':
            # For compressed TIFFs we should use PIL, as it's the fastest.  Freeimage
            # is actually the fastest but it only imports the first frame in a stack...
            try:
                import skimage.io
            except:
                print( "Error: scikit-image or glob not found!" )
                return  
                
            print( "Importing: " + self.files[target] )
            try:
                tempData = skimage.io.imread( self.files[target], plugin='pil' ).astype( 'float32' )
            except:
                print( "Error: PILlow image library not found, reverting to (slow) TIFFFile" )
                tempData = skimage.io.imread( self.files[target], plugin='tifffile' ).astype( 'float32' )
                
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
        elif file_ext == u".dm4":
            # Expects a DM4 image stack
            print( "Open as DM4: " + self.files[target] )
            dm4obj = mrcz.readDM4( self.files[target], verbose=False, useMemmap = useMemmap )
            tempData = np.copy( dm4obj.im[1].imageData.astype( float_dtype ), order='C' )
            # Load pixelsize from file
            try:
                if bool( dm4obj.im[1].imageInfo['DimXScale'] ):
                    if dm4obj.im[1].imageInfo[u'DimXUnits'] == u'\x14\x00': # This is what we get with no value set.
                        print( "DM4 pixels have no units, keeping previously set pixelsize" )
                        if self.pixelsize == None:
                            self.pixelsize
                        #else do nothing
                    else:
                        self.pixelsize = dm4obj.im[1].imageInfo['DimXScale'] # DM uses units of nm, we assume we don't have rectangular pixels because that's evil
            except KeyError: pass
            try: 
                if bool(dm4obj.im[1].imageInfo['Voltage'] ):
                    self.voltage = dm4obj.im[1].imageInfo['Voltage'] / 1000.0 # in kV
            except KeyError: pass
            try:
                if bool(dm4obj.im[1].imageInfo['C3']):
                    self.C3 = dm4obj.im[1].imageInfo['C3'] # in mm
            except KeyError: pass
            try:
                if bool(dm4obj.im[1].imageInfo['DetectorPixelSize']):
                    self.detectorPixelSize = dm4obj.im[1].imageInfo['DetectorPixelSize'][0] # in um
            except KeyError: pass    
            
            del dm4obj
        elif file_ext == u".mrc" or file_ext == u'.mrcs' or file_ext == u".mrcz" or file_ext == u".mrczs":
            # Expects a MRC image stack
            tempData, header = mrcz.readMRC( self.files[target], pixelunits=u'nm' )
            # Force data to 32-bit float if it uint8 or uint16
            if tempData.dtype.itemsize < 4:
                tempData = tempData.astype('float32')
                
            # As old MotionCorr data has no pixelsize in the header, only accept if the MRC file has non-zero
            # This allows a pre-set of ImageRegistrator.pixelsize
            if not np.isclose( header[u'pixelsize'][0] , 0.0 ):
                # Convert from Angstroms to nm performed internally
                self.pixelsize = np.float32( header[u'pixelsize'][0]  ) 
            # Should try writing C3 and voltage somewhere 
        elif file_ext == u".hdf5" or file_ext == u".h5":
            
            try:
                h5file = tables.open_file( self.files[target], mode='r' )
            except:
                print( "Could not open HDF5 file: " + self.files[target] )
            print( h5file )
            try: tempData = np.copy( h5file.get_node( '/', "images" ), order='C' ).astype('float32')
            except: print( "HDF5 file import did not find /images" )
            # TODO: load other nodes
            try: self.pixelsize = np.copy( h5file.get_node( '/', "pixelsize" ), order='C' )
            except: print( "HDF5 file import did not find /pixelsize" )
            try: self.voltage = np.copy( h5file.get_node( '/', "voltage" ), order='C' )
            except: print( "HDF5 file import did not find /voltage" )
            try: self.detectorPixelSize = np.copy( h5file.get_node( '/', "detectorPixelSize" ), order='C' )
            except: print( "HDF5 file import did not find /detectorPixelSize" )
            try: self.C3 = np.copy( h5file.get_node( '/', "C3" ), order='C' )
            except: print( "HDF5 file import did not find /C3" )
            
            try:
                h5file.close()
            except:
                pass
            pass
        else:
            print( "Unknown file extesion: " + stackNameIn )
            return
        
        #### GAIN REFERENCE MANAGEMENT ####
        if target != u'gainRef' and u'gainRef' in self.files and bool(self.files[u'gainRef']):
        
            # The Gatan gain reference is always a multiplication operation.  What of FEI and DE detectors?  
            if not np.any( self.gainRef ):
                self.loadData( self.files[u'gainRef'], target=u'gainRef' )
            
            gainRef = self.gainRef

            # Apply gain reference to each tempData, this should broadcast with numexpr?
            print( "Applying gain reference: %s" % self.files[u'gainRef'] )
            tempData = nz.evaluate( "gainRef * tempData" )
            pass
        
        # Finally, assign to target
        # TODO: set self.files[] dict values?
        if target == u"stack" or target == u'align' or target == u'images':
            if tempData.ndim != 3: # Probably the user saved a 2D image by mistake
                self.METAstatus = u"error"
                self.saveConfig()
                raise ValueError( "zorro.loadData: stacks must be 3D data" )
                
            if bool(self.gain) and not np.isclose( self.gain, 1.0 ):
                self.images = tempData / self.gain
            else:
                self.images = tempData
                
        elif target == u"sum" or target == u'imageSum':
            self.imageSum = tempData
            
            
        elif target == u"gainRef":
            # Apply flips and rotations
            if 'Diagonal' in self.gainInfo and self.gainInfo['Diagonal']:
                print( "Rotating gain reference by 90 degrees" )
                tempData = np.rot90( tempData, k = 1 )
                
            if 'Horizontal' in self.gainInfo and self.gainInfo['Horizontal'] and \
                'Vertical' in self.gainInfo and  self.gainInfo['Vertical']:
                # This is an image mirror, usually.
                print( "Rotating gain reference by 180 degrees (mirror)" )
                tempData = np.rot90( tempData, k =2 )
            elif 'Horizontal' in self.gainInfo and self.gainInfo['Horizontal']:
                print( "Flipping gain reference horizontally (mirror)" )
                tempData = np.fliplr( tempData )
            elif 'Vertical' in self.gainInfo and  self.gainInfo['Vertical']:
                print( "Flipping gain reference vertically (mirror)" )
                tempData = np.flipud( tempData )
            # TODO: see if any other labs have some wierd configuration of flips and rotations.
                
            # The Gatan gain reference has a lot of hot pixel artifacts, that we'll clip away for the moment
            # Perhaps we should explicitely use the same algorithm as the hot pixel mask.
                
            #gainCutoff = 1E-4
            #gainLim = util.histClim( tempData, cutoff=gainCutoff )
            #hotpix = ( tempData <= gainLim[0] ) | ( tempData >= gainLim[1] )
            # Possibly we could skip the uniform filter and just force hot pixels to 
            # 1.0?  I might get in trouble from a non-Gatan detector?
            
            # self.gainRef = ~hotpix*tempData + hotpix*scipy.ndimage.uniform_filter( tempData, size=5 )
            #self.gainRef = ~hotpix*tempData + hotpix
            self.gainRef = tempData
            
        elif target == u"filt" or target == u'filtSum':
            self.filtSum = tempData
        elif target == u"xc":
            self.C = tempData
            print( "TODO: set filename for C in loadData" )
        elif target == u"mask":
            self.masks = tempData
            
        self.bench['loaddata1'] = time.time() 
        
    def saveData( self ):
        """
        Save files to disk.  
        
        Do compression of stack if requested, self.compression = '.bz2' for example
        uses lbzip2 or 7-zip. '.gz' is also supported by not recommended.
        
        TODO: add dtype options, including a sloppy float for uint16 and uint8
        """
        self.bench['savedata0'] = time.time()
        import os, shutil
        try: os.umask( self.umask ) # Why is Python not using default umask from OS?
        except: pass

        # If self.files['config'] exists we save relative to it.  Otherwise we default to the place of 
        # self.files['stack']
#        if bool( self.files['config'] ): 
#            baseDir = os.path.dirname( self.files['config'] )
#        else:
#            baseDir = os.path.dirname( self.files['stack'] )
        stackFront, stackExt = os.path.splitext( os.path.basename( self.files[u'stack'] ) )
        
        if not 'compressor' in self.files or not bool(self.files['compressor']):
            mrcExt = ".mrc"
            mrcsExt = ".mrcs"
            self.files['compressor'] = None
            self.files['clevel'] = 0
        else:
            mrcExt = ".mrcz"
            mrcsExt = ".mrcsz" 
        
        # Change the current directory to make relative pathing sensible
#        try:
#            os.chdir( baseDir )
#        except: 
#            baseDir = "."# Usually baseDir is "" which is "."
        
        if stackExt == ".bz2" or stackExt == ".gz" or stackExt == ".7z":
            # compressExt = stackExt
            stackFront, stackExt = os.path.splitext( stackFront )
            
        if self.files[u'sum'] is None: # Default sum name
            self.files[u'sum'] = os.path.join( u"sum", u"%s_zorro%s" %(stackFront, mrcExt) )

        # Does the directory exist?  Often this will be a relative path to file.config
        sumPath, sumFile = os.path.split( self.files[u'sum'] )
        if not os.path.isabs( sumPath ):
            sumPath = os.path.realpath( sumPath ) # sumPath is always real
        if bool(sumPath) and not os.path.isdir( sumPath ):
            os.mkdir( sumPath )
        relativeSumPath = os.path.relpath( sumPath )
        
        #### SAVE ALIGNED SUM ####
        if self.verbose >= 1:
            print( "Saving: " + os.path.join(sumPath,sumFile) )
        mrcz.writeMRC( self.imageSum, os.path.join(sumPath,sumFile), 
                        pixelsize=self.pixelsize, pixelunits=u'nm',
                        voltage = self.voltage, C3 = self.C3, gain = self.gain,
                        compressor=self.files[u'compressor'], 
                        clevel=self.files[u'clevel'], 
                        n_threads=self.n_threads) 

        # Compress sum
        if bool(self.doCompression):
            util.compressFile( os.path.join(sumPath,sumFile), self.compress_ext, n_threads=self.n_threads )

        #### SAVE ALIGNED STACK ####
        if bool(self.saveMovie):
            if self.files[u'align'] is None: # Default filename for aligned movie
                self.files[u'align'] = os.path.join( u"align", u"%s_zorro_movie%s" % (stackFront, mrcsExt) )
                
            # Does the directory exist?
            alignPath, alignFile = os.path.split( self.files[u'align'] )
            if not os.path.isabs( sumPath ):
                alignPath = os.path.realpath( alignPath )
            if bool(alignPath) and not os.path.isdir( alignPath ):
                os.mkdir( alignPath )
            
            if self.verbose >= 1:
                print( "Saving: " + os.path.join(alignPath,alignFile) )
            mrcz.writeMRC( self.images, os.path.join(alignPath,alignFile), 
                            pixelsize=self.pixelsize, pixelunits=u'nm',
                            voltage = self.voltage, C3 = self.C3, gain = self.gain,
                            compressor=self.files[u'compressor'], 
                            clevel=self.files[u'clevel'], 
                            n_threads=self.n_threads) 

            # Compress stack
            if bool(self.doCompression):
                util.compressFile( os.path.join(alignPath,alignFile), self.compress_ext, n_threads=self.n_threads )
                
        if bool(self.filterMode) and np.any(self.filtSum): # This will be in the same place as sum
            if not u'filt' in self.files or self.files[u'filt'] is None: # Default filename for filtered sum
                self.files[u'filt'] = os.path.join( relativeSumPath, u"%s_filt%s" %(os.path.splitext(sumFile)[0], mrcExt) )
            
            filtPath, filtFile = os.path.split( self.files[u'filt'] )
            if not os.path.isabs( filtPath ):
                filtPath = os.path.realpath( filtPath ) 
                
            if self.verbose >= 1:
                print( "Saving: " + os.path.join(filtPath, filtFile) )
            mrcz.writeMRC( self.filtSum, os.path.join(filtPath, filtFile), 
                            pixelsize=self.pixelsize, pixelunits=u'nm',
                            voltage = self.voltage, C3 = self.C3, gain = self.gain,
                            compressor=self.files[u'compressor'], 
                            clevel=self.files[u'clevel'], 
                            n_threads=self.n_threads) 

        #### SAVE CROSS-CORRELATIONS FOR FUTURE PROCESSING OR DISPLAY ####
        if self.saveC and self.C != None:
            self.files[u'xc'] = os.path.join( sumPath, u"%s_xc%s" % (os.path.splitext(sumFile)[0],mrcsExt) )
            if self.verbose >= 1:
                print( "Saving: " + self.files[u'xc'] )
                
            mrcz.writeMRC( np.asarray( self.C, dtype='float32'), self.files[u'xc'], 
                            pixelsize=self.pixelsize, pixelunits=u'nm',
                            voltage = self.voltage, C3 = self.C3, gain = self.gain,
                            compressor=self.files[u'compressor'], 
                            clevel=self.files[u'clevel'], 
                            n_threads=self.n_threads) 
                            
            if bool(self.doCompression):
                util.compressFile( self.files[u'xc'], self.compress_ext, n_threads=self.n_threads )
            
        #### SAVE OTHER INFORMATION IN A LOG FILE ####
        # Log file is saved seperately...  Calling it here could lead to confusing behaviour.

        if u'moveRawPath' in self.files and bool( self.files[u'moveRawPath'] ) and not os.path.isdir( self.files[u'moveRawPath'] ):
            os.mkdir( self.files[u'moveRawPath'] )
                
        if bool( self.doCompression ): # does compression and move in one op
            self.files[u'stack'] = util.compressFile( self.files[u'stack'], outputDir=self.files[u'moveRawPath'], 
                                               n_threads=self.n_threads, compress_ext=self.compress_ext )
        elif u'moveRawPath' in self.files and bool( self.files[u'moveRawPath'] ):
            newStackName = os.path.join( self.files[u'moveRawPath'], os.path.split( self.files[u'stack'])[1] )
            print( "Moving " +self.files[u'stack'] + " to " + newStackName )
            
            try:
                os.rename( self.files[u'stack'], newStackName )
            except:
                # Often we can't rename between file systems so we need to copy and delete instead
                shutil.copyfile( self.files[u'stack'], newStackName )
                # if os.path.isfile( newStackName) and filecmp.cmp( self.files['stack'], newStackName ):
                # filecmp is very, very slow...  we need a better trick, maybe just compare sizes
                if os.path.isfile( newStackName):
                    os.remove( self.files[u'stack'] )
                else:
                    print( "Error in copying raw stack, original will not be deleted from input directory" )

            self.files[u'stack'] = newStackName
        pass
        self.bench['savedata1'] = time.time()


    def loadConfig( self, configNameIn = None, loadData=False ):
        """
        Initialize the ImageRegistrator class from a config file
        
        loadData = True will load data from the given filenames.
        """
        
        import json
        if not bool(configNameIn):
            if not bool( self.files['config'] ):
                pass # Do nothing
            else:
                print( "Cannot find configuration file: " + self.files[u'config'] )
        else:
            self.files[u'config'] = configNameIn

        print( "Loading config file: " + self.files[u'config'] )
        config = configparser.RawConfigParser(allow_no_value = True)
        try:
            config.optionxform = unicode # Python 2
        except:
            config.optionxform = str # Python 3
        
        ##### Paths #####
        # I'd prefer to pop an error here if configName doesn't exist
        if not os.path.isfile( self.files[u'config'] ):
            raise IOError( "zorro.loadConfig: Could not load config file %s" % self.files[u'config'] )
        config.read( self.files[u'config'] )
        
        
         
        # Initialization
        try: self.verbose = config.getint( u'initialization', u'verbose' )
        except: pass
        try: self.umask = config.getint( u'initialization', u'umask' )
        except: pass
        try: self.fftw_effort = config.get( u'initialization', u'fftw_effort' ).upper()
        except: pass
        try: self.n_threads = config.getint( u'initialization', u'n_threads' )
        except: pass
        try: self.saveC = config.getboolean( u'initialization', u'saveC' )
        except: pass
        try: self.METAstatus = config.get( u'initialization', u'METAstatus' )
        except: pass
        try: self.cachePath = config.get( u'initialization', u'cachePath' )
        except: pass
    
        # Calibrations
        try: self.pixelsize = config.getfloat(u'calibration',u'pixelsize')
        except: pass
        try: self.voltage = config.getfloat(u'calibration',u'voltage')
        except: pass
        try: self.C3 = config.getfloat(u'calibration',u'C3')
        except: pass
        try: self.gain = config.getfloat(u'calibration',u'gain')
        except: pass
        try: self.detectorPixelSize = config.getfloat(u'calibration',u'detectorPixelSize')
        except: pass
        try: self.gainInfo = json.loads( config.get( u'calibration', u'gainInfo' ))   
        except: pass
    
        # Data
        try: self.trackCorrStats = config.getboolean( u'data', u'trackCorrStats' )
        except: pass
        try: self.corrStats = json.loads( config.get(u'data', u'corrStats') )
        except: pass
        try: self.bench = json.loads( config.get(u'data', u'bench') )
        except: pass
        try: self.hotpixInfo = json.loads( config.get(u'data', u'hotpixInfo') )
        except: pass
        

        # Results 
        # Load arrays with json
        try: self.translations = np.array( json.loads( config.get( u'results', u'translations' ) ) )
        except: pass
        try: self.transEven = np.array( json.loads( config.get( u'results', u'transEven' ) ) )
        except: pass
        try: self.transOdd = np.array( json.loads( config.get( u'results', u'transOdd' ) ) )
        except: pass
        try: self.velocities = np.array( json.loads( config.get( u'results', u'velocities' ) ) )
        except: pass
        try: self.rotations = np.array( json.loads( config.get( u'results', u'rotations' ) ) )
        except: pass
        try: self.scales = np.array( json.loads( config.get( u'results', u'scales' ) ) )
        except: pass
        try: self.FRC = np.array( json.loads( config.get( u'results', u'FRC' ) ) )
        except: pass
    
        try: self.CTFProgram = config.get( u'ctf', u'CTFProgram' )
        except: pass
        # CTF dict
        try: self.ctfInfo = json.loads( config.get( u'ctf', u'CTFInfo' ) )
        except: pass


        errorDictsExist=True
        errCnt = 0
        while errorDictsExist:
            try:
                newErrorDict = {}
                dictName = u'errorDict%d' % errCnt
                # Load the list of keys and then load them element-by-element
                # newErrorDict = json.loads( config.get( 'data', dictName ) )
                keyList = json.loads( config.get( dictName, u'keyList' ) )
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
        try: self.xcorrMode = config.get( u'registration', u'xcorrMode' )
        except: pass
        try: self.triMode = config.get( u'registration', u'triMode' )
        except: pass
    
        try: self.startFrame = config.getint( u'registration', u'startFrame' )
        except: pass
        try: self.endFrame = config.getint( u'registration', u'endFrame' )
        except: pass
    
        try: self.shapePadded = np.array( json.loads( config.get( u'registration', u'shapePadded' ) ) )
        except: pass
    
        try: self.shapeOriginal = np.array( json.loads( config.get( u'registration', u'shapeOriginal' ) ) )
        except: pass
        try: self.shapeBinned = np.array( json.loads( config.get( u'registration', u'shapeBinned' ) ) )
        except: pass
        try: self.fouCrop = np.array( json.loads( config.get( u'registration', u'fouCrop' ) ) )
        except: pass
        try: self.subPixReg = config.getint( u'registration', u'subPixReg' )
        except: pass
        try: self.shiftMethod = config.get( u'registration', u'shiftMethod' )
        except: pass
        try: self.maxShift = config.getint( u'registration', u'maxShift' )
        except: pass
        try: self.preShift = config.getboolean( u'registration', u'preShift' )
        except: pass
        try: self.triMode = config.get( u'registration', u'triMode' )
        except: pass
        try: self.diagWidth = config.getint( u'registration', u'diagWidth' )
        except: pass
        try: self.diagStart = config.getint( u'registration', u'diagStart' )
        except: pass
        try: self.autoMax = config.getint( u'registration', u'autoMax' )
        except: pass
        try: self.peaksigThres = config.getfloat( u'registration', u'peaksigThres' )
        except: pass

        try: self.corrThres = config.getfloat( u'registration', u'corrThres' )
        except: pass
        try: self.velocityThres = config.getfloat( u'registration', u'velocityThres' )
        except: pass
        try: self.Brad = config.getfloat( u'registration', u'Brad' )
        except: pass
        try: self.Bmode = config.get( u'registration', u'Bmode' )
        except: pass
        try: self.BfiltType = config.get( u'registration', u'BfiltType' )
        except: pass
        try: self.originMode = config.get( u'registration', u'originMode' )
        except: pass
        try: self.suppressOrigin = config.getboolean( u'registration', u'suppressOrigin' )
        except: pass
        try: self.weightMode = config.get( u'registration', u'weightMode' )
        except: pass
        try: self.logisticK = config.getfloat( u'registration', u'logisticK' )
        except: pass
        try: self.logisticNu = config.getfloat( u'registration', u'logisticNu' )
        except: pass
        try: self.filterMode = config.get( u'registration', u'filterMode' )
        except: pass
        try: self.doFRC = config.getboolean( u'registration', u'doLazyFRC' )
        except: pass
        try: self.doEvenOddFRC = config.getboolean( u'registration', u'doEvenOddFRC' )
        except: pass
        try: self.doseFiltParam = json.loads( config.get( u'registration', u'doseFiltParam' ) ) # This one stays a list
        except: pass

    
        # IO 
        try: self.files = json.loads( config.get( u'io', u'files' ) )
        except: pass
        try: self.savePNG = config.getboolean( u'io', u'savePNG' )
        except: pass
        try: self.compress_ext = config.get( u'io', u'compress_ext' )
        except: pass
        try: self.saveMovie = config.getboolean( u'io', u'saveMovie' )
        except: pass
        try: self.doCompression = config.getboolean( u'io', u'doCompression' )
        except: pass
    
        # Plot 
        try: self.plotDict = json.loads( config.get( u'plot', u'plotDict' ) )
        except: pass

        
        if bool(loadData) and u'stack' in self.files and self.files[u'stack'] != None:
            self.loadData()
        pass
    
    def saveConfig( self, configNameIn=None ):
        """
        Write the state of the ImageRegistrator class from a config file
        """

        import json
        import os
        try: os.umask( self.umask ) # Why is Python not using default umask from OS?
        except: pass        
        
        if not bool( configNameIn ):
            if self.files[u'config'] is None:
                self.files[u'config'] = self.files[u'stack'] + u".zor"
        else:
            self.files['config'] = configNameIn
        # Does the directory exist?
        configPath = os.path.realpath( os.path.dirname( self.files[u'config'] ) )
        if bool(configPath) and not os.path.isdir( configPath ):
            os.mkdir( configPath )
        
        # Write config
        config = configparser.RawConfigParser(allow_no_value = True)
        try:
            config.optionxform = unicode # Python 2
        except:
            config.optionxform = str # Python 3
        
        # Initialization
        config.add_section( u'initialization' )
        config.set( u'initialization', u'METAstatus', self.METAstatus )
        config.set( u'initialization', u'# METAstatus _MUST_ appear as second line in file' )
        config.set( u'initialization', u'# For detailed use instructions: github.com/C-CINA/zorro/wiki', None )
        config.set( u'initialization', u'verbose', self.verbose )
        config.set( u'initialization', u'umask', self.umask )
        config.set( u'initialization', u'fftw_effort', self.fftw_effort )
        # Any time we cast variables we need handle errors from numpy
        config.set( u'initialization', u'# n_threads is usually best if set to the number of physical cores (CPUs)' )
        try: config.set( u'initialization', u'n_threads', np.int(self.n_threads) )
        except: pass
        config.set( u'initialization', u'saveC', self.saveC )
        config.set( u'initialization', u'cachePath', self.cachePath )
        
        
        # Calibrations
        config.add_section( u'calibration' )
        config.set( u'calibration', u"# Zorro can strip this information from .DM4 files if its is present in tags" )
        config.set( u'calibration' , u"# Pixel size in nanometers" )
        config.set( u'calibration',u'pixelsize', self.pixelsize )
        config.set( u'calibration' , u"# Accelerating voltage in kV" )
        config.set( u'calibration',u'voltage', self.voltage )
        config.set( u'calibration' , u"# Spherical aberration in mm" )
        config.set( u'calibration',u'C3', self.C3 )
        config.set( u'calibration' , u"# Gain in electrons/count" )
        config.set( u'calibration',u'gain', self.gain )
        config.set( u'calibration',u'detectorPixelSize', self.detectorPixelSize )
        config.set( u'calibration', u'gainInfo', json.dumps( self.gainInfo ) )
        
        # Registration parameters
        config.add_section( u'registration' )
        config.set( u'registration', u'xcorrMode', self.xcorrMode )
        config.set( u'registration' , u"# tri, diag, first, auto, or autocorr" )
        config.set( u'registration', u'triMode', self.triMode )
        
        
        if self.shapePadded is not None:
            if type(self.shapePadded) == type(np.array(1)):
                self.shapePadded = self.shapePadded.tolist()
            config.set( u'registration', u"# Use a padding 10 % bigger than the original image, select an efficient size with zorro_util.findValidFFTWDim()" )   
            config.set( u'registration', u'shapePadded', json.dumps( self.shapePadded) )
            
        if self.shapeOriginal is not None:
            if type(self.shapeOriginal) == type(np.array(1)):
                self.shapeOriginal = self.shapeOriginal.tolist()
            config.set( u'registration', u'shapeOriginal', json.dumps( self.shapeOriginal ) )
        if self.shapeBinned is not None:
            if type(self.shapeBinned) == type(np.array(1)):
                self.shapeBinned = self.shapeBinned.tolist()
            config.set( u'registration', u'shapeBinned', json.dumps( self.shapeBinned ) )
            
        if self.fouCrop is not None:
            if type(self.fouCrop) == type(np.array(1)):
                self.fouCrop = self.fouCrop.tolist()
            config.set( u'registration', u'fouCrop', json.dumps( self.fouCrop ) )
        
        try: config.set( u'registration', u'subPixReg', np.int(self.subPixReg) )
        except: pass
        config.set( u'registration', u'shiftMethod', self.shiftMethod )
        config.set( u'registration' , u"# Maximum shift in pixels within diagWidth/autoMax frames" )
        try: config.set( u'registration', u'maxShift', np.int(self.maxShift) )
        except: pass
        config.set( u'registration' ,u"# preShift = True is useful for crystalline specimens where you want maxShift to follow the previous frame position" )
        config.set( u'registration', u'preShift', self.preShift )
        
        try: config.set( u'registration', u'diagStart', np.int(self.diagStart) )
        except: pass
        try: config.set( u'registration', u'diagWidth', np.int(self.diagWidth) )
        except: pass
        try: config.set( u'registration', u'autoMax', np.int(self.autoMax) )
        except: pass
        try: config.set( u'registration', u'startFrame', np.int(self.startFrame) )
        except: pass
        try: config.set( u'registration', u'endFrame', np.int(self.endFrame) )
        except: pass
        
        config.set( u'registration' , u"# peakSigThres changes with dose but usually is uniform for a dataset" )
        config.set( u'registration', u'peaksigThres', self.peaksigThres )
        config.set( u'registration' , u"# corrThres is DEPRECATED" )
        config.set( u'registration', u'corrThres', self.corrThres )
        config.set( u'registration', u'velocityThres', self.velocityThres )
        config.set( u'registration' , u"# Brad is radius of B-filter in Fourier pixels" )
        config.set( u'registration', u'Brad', self.Brad )
        config.set( u'registration' , u"# Bmode = conv, opti, or fourier" )
        config.set( u'registration', u'Bmode', self.Bmode )
        config.set( u'registration', u'BFiltType', self.BfiltType )
        config.set( u'registration' , u"# originMode is centroid, or (empty), empty sets frame 1 to (0,0)" )
        config.set( u'registration', u'originMode', self.originMode )
        config.set( u'registration' , u"# weightMode is one of logistic, corr, norm, unweighted" )
        config.set( u'registration', u'weightMode', self.weightMode )
        config.set( u'registration', u'logisticK', self.logisticK )
        config.set( u'registration', u'logisticNu', self.logisticNu )
        config.set( u'registration' , u"# Set suppressOrigin = True if gain reference artifacts are excessive" )
        config.set( u'registration', u'suppressOrigin', self.suppressOrigin )
        config.set( u'registration', u'filterMode', self.filterMode )
        config.set( u'registration', u'doLazyFRC', self.doLazyFRC )
        config.set( u'registration', u'doEvenOddFRC', self.doEvenOddFRC )
        if np.any( self.doseFiltParam ) and bool( self.filterMode ):
            config.set( u'registration', u'doseFiltParam', json.dumps( self.doseFiltParam ) )

        
        # CTF
        config.add_section( u'ctf' )
        config.set( u'ctf', u'CTFProgram', self.CTFProgram )
        config.set( u'ctf', u'CTFInfo', json.dumps( self.CTFInfo ) )
        
        # IO
        config.add_section(u'io')
        config.set( u'io', u'savePNG', self.savePNG )
        config.set( u'io', u'compress_ext', self.compress_ext )
        config.set( u'io', u'saveMovie', self.saveMovie )
        config.set( u'io', u'doCompression', self.doCompression )

        config.set( u'io' , u"# Note: all paths are relative to the current working directory." )
        config.set( u'io', u'files', json.dumps( self.files ) )


        # Plot
        config.add_section( u'plot' )
        config.set( u'plot', u'plotDict', json.dumps( self.plotDict ) )

        
        # Results 
        # Seems Json does a nice job of handling numpy arrays if converted to lists
        config.add_section( u'results' )
        if self.translations is not None:
            config.set( u'results', u'translations', json.dumps( self.translations.tolist() ) )
        if self.transEven is not None:
            config.set( u'results', u'transEven', json.dumps( self.transEven.tolist() ) )
        if self.transOdd is not None:
            config.set( u'results', u'transOdd', json.dumps( self.transOdd.tolist() ) )
        if self.rotations is not None:    
            config.set( u'results', u'rotations', json.dumps( self.rotations.tolist() ) )
        if self.scales is not None:
            config.set( u'results', u'scales', json.dumps( self.scales.tolist() ) )
        if self.velocities is not None:
            config.set( u'results', u'velocities', json.dumps( self.velocities.tolist() ) )
        if self.FRC is not None:
            config.set( u'results', u'FRC', json.dumps( self.FRC.tolist() ) )

    
        # Data
        config.add_section( u'data' )
        config.set( u'data', u'hotpixInfo', json.dumps( self.hotpixInfo) )
        config.set( u'data', u'trackCorrStats', self.trackCorrStats )
        config.set( u'data', u'corrStats', json.dumps( self.corrStats) )
        config.set( u'data', u'bench', json.dumps( self.bench ) )
                    
        # Error dicts
        for errCnt, errorDict in enumerate(self.errorDictList):
            # For serialization, the errorDict arrays have to be lists.)
            dictName = u'errorDict%d'%errCnt
            config.add_section( dictName )
            keyList = list( errorDict.keys() )
            config.set( dictName, u'keyList', json.dumps( keyList ) )
            for key in keyList:
                if( hasattr( errorDict[key], "__array__" ) ):
                    config.set( dictName, key, json.dumps( errorDict[key].tolist() ) )
                else:
                    config.set( dictName, key, json.dumps( errorDict[key] ) )
        
        try:
            # Would be nice to have some error handling if cfgFH already exists
            # Could try and open it with a try: open( 'r' )
            cfgFH = open( self.files[u'config'] , 'w+' )
            if self.verbose >= 1:
                print( "Saving config file: " + self.files[u'config'] )
            config.write( cfgFH )
            cfgFH.close()
        except:
            print( "Error in loading config file: " + self.files[u'config'] )


    def plot( self, title = "" ):
        """
        Multiprocessed matplotlib diagnostic plots. 
        
        For each plot, make a list that contains the name of the plot, and a dictionary that contains all the 
        information necessary to render the plot.
        """
        self.bench['plot0'] = time.time()
        if not bool(title):
            # Remove any pathing from default name as figurePath overrides this.
            if bool( self.files[u'stack'] ):
                self.plotDict[u'title'] = os.path.split( self.files[u'stack'] )[1]
            else:
                self.plotDict[u'title'] = u"default"
        else:
            self.plotDict[u'title'] = title
            
        # figurePath needs to be relative to the config directory, which may not be the current directory.
#        if bool(self.savePNG ) and bool(self.files['config']):
#            try: # Sometimes this is empty
#                os.chdir( os.path.split(self.files['config'])[0] )
#            except: pass
        
        # Error checks on figurePath
        if not bool( self.files[u'figurePath'] ):
            self.files[u'figurePath'] = u"./fig"
        if not os.path.isdir( self.files[u'figurePath'] ):
            os.mkdir( self.files[u'figurePath'] )
            
        plotArgs = []
        # IF IMAGESUM
        if np.any(self.imageSum) and u'imageSum' in self.plotDict and ( self.plotDict[u'imageSum'] ):
            #print( "zorro.plot.imageSum" )
            plotDict = self.plotDict.copy()
            
            # Unfortunately binning only saves time if we do it before pickling the data off to multiprocess.
            # TODO: http://stackoverflow.com/questions/7894791/use-numpy-array-in-shared-memory-for-multiprocessing
            binning = 2
            plotDict[u'pixelsize'] = self.pixelsize * binning
            imageSumBinned = util.magickernel( self.getSumCropToLimits(), k=1 )
            plotDict[u'image'] = imageSumBinned
            # RAM: temporary expidient of filtering FFTs of large images to increase contrast
            if self.imageSum.shape[0]*binning > 3072 and self.imageSum.shape[1]*binning > 3072:
                plotDict[u'lowPass'] = 0.75

            if bool(self.savePNG):
                plotDict[u'plotFile'] = os.path.join( self.files[u'figurePath'], self.plotDict[u'title'] + "_imageSum.png")
                self.files[u'figImageSum'] = plotDict[u'plotFile']
            plotArgs.append( [u'image', plotDict] )
            
        # IF FILTSUM
        if np.any(self.filtSum) and u'filtSum' in self.plotDict and bool( self.plotDict[u'filtSum'] ):
            #print( "zorro.plot.filtSum" )
            plotDict = self.plotDict.copy()
            
            # Unfortunately binning only saves time if we do it before pickling the data off to multiprocess.
            # TODO: http://stackoverflow.com/questions/7894791/use-numpy-array-in-shared-memory-for-multiprocessing
            binning = 2
            plotDict[u'pixelsize'] = self.pixelsize * binning
            filtSumBinned = util.magickernel( self.getFiltSumCropToLimits(), k=1 )
            plotDict[u'image'] = filtSumBinned        
            # RAM: temporary expidient of filtering FFTs of large images to increase contrast
            if self.imageSum.shape[0]*binning > 3072 and self.imageSum.shape[1]*binning > 3072:
                plotDict[u'lowPass'] = 0.75

            if bool(self.savePNG):
                plotDict[u'plotFile'] = os.path.join( self.files[u'figurePath'], self.plotDict[u'title'] + "_filtSum.png")
                self.files[u'figFiltSum'] = plotDict[u'plotFile']
            plotArgs.append( [u'image', plotDict] )
        
        # IF FFTSUM
        if np.any(self.imageSum) and u'FFTSum' in self.plotDict and bool( self.plotDict[u'FFTSum'] ):
            #print( "zorro.plot.FFTSum" )
            plotDict = self.plotDict.copy()

            
            # No FFT binning please
            plotDict[u'pixelsize'] = self.pixelsize
            # We would like the cropped sum but that can be a wierd size that is slow for the FFT
            plotDict[u'image'] = self.imageSum 
            
            # RAM: temporary expidient of filtering FFTs of large images to increase contrast
            if self.imageSum.shape[0] > 3072 and self.imageSum.shape[1] > 3072:
                plotDict[u'lowPass'] = 3.0
            
            if bool(self.savePNG):
                plotDict[u'plotFile'] = os.path.join( self.files[u'figurePath'], self.plotDict[u'title'] + "_FFTSum.png")
                self.files[u'figFFTSum'] = plotDict[u'plotFile']
            plotArgs.append( [u'FFT', plotDict] )
            pass
        
        # IF POLARFFTSUM
        if np.any(self.imageSum) and u'polarFFTSum' in self.plotDict and bool( self.plotDict[u'polarFFTSum'] ):
            #print( "zorro.plot.PolarFFTSum" )
            plotDict = self.plotDict.copy()

            # No FFT binning please
            plotDict[u'pixelsize'] = self.pixelsize
            # We would like the cropped sum but that can be a wierd size that is slow for the FFT
            plotDict[u'image'] = self.imageSum     
            
            # RAM: temporary expidient of filtering FFTs of large images to increase contrast
            if self.imageSum.shape[0] > 3072 and self.imageSum.shape[1] > 3072:
                plotDict[u'lowPass'] = 1.5
                
            if bool(self.savePNG):
                plotDict[u'plotFile'] = os.path.join( self.files[u'figurePath'], self.plotDict[u'title'] + "_polarFFTSum.png")
                self.files[u'figPolarFFTSum'] = plotDict[u'plotFile']
            plotArgs.append( [u'polarFFT', plotDict] )
            pass
        
        # IF TRANSLATIONS
        if np.any(self.translations) and u'translations' in self.plotDict and bool( self.plotDict[u'translations'] ):
            #print( "zorro.plot.Translations" )
            plotDict = self.plotDict.copy()
            if np.any( self.translations ):
                plotDict[u'translations'] = self.translations
                try:
                    plotDict[u'errorX'] = self.errorDictList[0][u'errorX']
                    plotDict[u'errorY'] = self.errorDictList[0][u'errorY']
                except: pass
                if bool(self.savePNG):
                    plotDict[u'plotFile'] = os.path.join( self.files[u'figurePath'], self.plotDict[u'title'] + "_translations.png")
                    self.files[u'figTranslations'] = plotDict[u'plotFile']
                plotArgs.append( [u'translations', plotDict] )  
                
        # IF PIXEL REGISTRATION ERROR
        if len(self.errorDictList) > 0 and u'pixRegError' in self.plotDict and bool( self.plotDict[u'pixRegError'] ):
            #print( "zorro.plot.PixRegError" )
            plotDict = self.plotDict.copy()
            plotDict[u'errorX'] = self.errorDictList[0][u'errorX']
            plotDict[u'errorY'] = self.errorDictList[0][u'errorY']
            plotDict[u'errorXY'] = self.errorDictList[0][u'errorXY']
            if bool(self.savePNG):
                plotDict[u'plotFile'] = os.path.join( self.files[u'figurePath'], self.plotDict[u'title'] + "_pixRegError.png")
                self.files[u'figPixRegError'] = plotDict[u'plotFile']
            plotArgs.append( [u'pixRegError', plotDict] )  
            
        # IF CORRTRIMAT
        if len(self.errorDictList) > 0 and u'corrTriMat' in self.plotDict and bool( self.plotDict[u'corrTriMat'] ):
            #print( "zorro.plot.coor" )
            plotDict = self.plotDict.copy()
            plotDict[u'corrTriMat'] = self.errorDictList[-1][u'corrTriMat']
            if bool(self.savePNG):
                plotDict[u'plotFile'] = os.path.join( self.files[u'figurePath'], self.plotDict[u'title'] + "_corrTriMat.png")
                self.files[u'figCorrTriMat'] = plotDict[u'plotFile']
            plotArgs.append( [u'corrTriMat', plotDict] )  
            
        # IF PEAKSIGTRIMAT
        if len(self.errorDictList) > 0 and u'peaksigTriMat' in self.plotDict and bool( self.plotDict[u'peaksigTriMat'] ):
            #print( "zorro.plot.peaksig" )
            plotDict = self.plotDict.copy()
            plotDict[u'peaksigTriMat'] = self.errorDictList[-1][u'peaksigTriMat']
            if bool(self.savePNG):
                plotDict[u'plotFile'] = os.path.join( self.files[u'figurePath'], self.plotDict[u'title'] + "_peaksigTriMat.png")
                self.files[u'figPeaksigTriMat'] = plotDict[u'plotFile']
            plotArgs.append( [u'peaksigTriMat', plotDict] )  
            
        # IF LOGISTICS CURVE        
        if len(self.errorDictList) > 0 and u'logisticWeights' in self.plotDict and bool( self.plotDict[u'logisticWeights'] ):
            #print( "zorro.plot.logist" )
            plotDict = self.plotDict.copy()
            if self.weightMode == u'autologistic' or self.weightMode == u'logistic':
                plotDict[u'peaksigThres'] = self.peaksigThres
                plotDict[u'logisticK'] = self.logisticK
                plotDict[u'logisticNu'] = self.logisticNu
            plotDict[u'errorXY'] = self.errorDictList[0][u"errorXY"]
            plotDict[u'peaksigVect'] = self.errorDictList[0][u"peaksigTriMat"][ self.errorDictList[0]["peaksigTriMat"] > 0.0  ]
            
            if u'cdfPeaks' in self.errorDictList[0]:
               plotDict[u'cdfPeaks'] = self.errorDictList[0][u'cdfPeaks']
               plotDict[u'hSigma'] = self.errorDictList[0][u'hSigma']
            if bool(self.savePNG):
                plotDict[u'plotFile'] = os.path.join( self.files[u'figurePath'], self.plotDict[u'title'] + "_logisticWeights.png")
                self.files[u'figLogisticWeights'] = plotDict[u'plotFile']
            plotArgs.append( [u'logisticWeights', plotDict] )
             
        # IF FRC PLOT
        if np.any(self.FRC) and u'FRC' in self.plotDict and bool( self.plotDict[u'FRC'] ):
            #print( "zorro.plot.FRC" )
            plotDict = self.plotDict.copy()
            plotDict[u'FRC'] = self.FRC
            plotDict[u'pixelsize'] = self.pixelsize

                
            if bool( self.doEvenOddFRC ):
                plotDict[u'labelText'] = u"Even-odd frame independent FRC"
            else:
                plotDict[u'labelText'] = u"Non-independent FRC is not a resolution estimate"
                
            if bool(self.savePNG):
                plotDict[u'plotFile'] = os.path.join( self.files[u'figurePath'], self.plotDict[u'title'] + "_FRC.png")
                self.files[u'figLazyFRC'] = plotDict[u'plotFile']
            plotArgs.append( [u'lazyFRC', plotDict] )
            
        # IF CTFDIAG PLT
        if np.any(self.CTFDiag) and u'CTFDiag' in self.plotDict and bool( self.plotDict[u'CTFDiag'] ):
            plotDict = self.plotDict.copy()
            
            plotDict[u'CTFDiag'] = self.CTFDiag
            plotDict[u'CTFInfo'] = self.CTFInfo
            plotDict[u'pixelsize'] = self.pixelsize
            if bool(self.savePNG):
                plotDict[u'plotFile'] = os.path.join( self.files[u'figurePath'], self.plotDict[u'title'] + "_CTFDiag.png")
                self.files[u'figCTFDiag'] = plotDict[u'plotFile']
            plotArgs.append( [u'CTFDiag', plotDict] )
            
        # IF STATS PLOT
        if u'stats' in self.plotDict and bool( self.plotDict[u'stats'] ):
            #print( "zorro.plot.stats" )
            plotDict = self.plotDict.copy()
            plotDict[u'pixelsize'] = self.pixelsize
            plotDict[u'voltage'] = self.voltage
            plotDict[u'C3'] = self.C3
            if len( self.errorDictList ) > 0 and u'peaksigTriMat' in self.errorDictList[-1]:
                peaksig = self.errorDictList[-1][u'peaksigTriMat']
                peaksig = peaksig[ peaksig > 0.0 ]
                plotDict[u'meanPeaksig'] = np.mean( peaksig )
                plotDict[u'stdPeaksig'] = np.std( peaksig )
                plotDict[u'CTFInfo'] = self.CTFInfo
        
            if bool(self.savePNG):
                plotDict[u'plotFile'] = os.path.join( self.files[u'figurePath'], self.plotDict[u'title'] + "_Stats.png")
                self.files[u'figStats'] = plotDict[u'plotFile']
            plotArgs.append( [u'stats', plotDict] )
                
        ######
        #Multiprocessing pool (to speed up matplotlib's slow rendering and hopefully remove polling loop problems)
        #####      
        if os.name != u'nt' and bool( self.plotDict[u'multiprocess'] ):
            figPool = mp.Pool( processes=self.n_threads )
            print( " n_threads = %d, plotArgs length = %d" %( self.n_threads, len(plotArgs) ) )
            figPool.map( plot.generate, plotArgs )
            
            figPool.close()
            figPool.terminate()
            # Wait for everyone to finish, otherwise on the infinityband cluster we have problems with partially rendered files.
            figPool.join() 
        else: # Windows mode, can also be used for debugging when plot goes haywire
            # Don't multiprocess the plots, but execute serially.
            for plotArg in plotArgs:
                plot.generate( plotArg )
        self.bench['plot1'] = time.time()
        
    def makeMovie( self, movieName = None, clim = None, frameRate=3, graph_cm = u'gnuplot' ):
        """
        Use FFMPEG to generate movies showing the correlations.  C0 must not be Nonz.
        
        The ffmpeg executable must be in the system path.
        """
        import os

        fex = '.png'
        print( "makeMovie must be able to find FFMPEG on the system path" )
        print( "Strongly recommended to use .mp4 extension" )
        if movieName is None:
            movieName = self.files[u'stack'] + u".mp4"
        
        m = self.C0.shape[0]
        
        # Turn off display of matplotlib temporarily
        originalBackend = plt.get_backend()
        plt.switch_backend(u'agg')
        plt.rc(u'font', family=self.plotDict[u'fontstyle'], size=self.plotDict[u'fontsize'])
        corrmat = self.errorDictList[-1][ u'corrTriMat' ]
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
            plt.savefig( "corrMap_%05d"%J + fex, dpi=self.plotDict['image_dpi'] )
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
        print( "    dtypes: float: %s, complex: %s" %(float_dtype, fftw_dtype) )
        if bool( np.any(self.filtSum) ):
            print( "    images.dtype: %s, filtSum.dtype: %s" % (self.images.dtype, self.filtSum.dtype) )
        else:
            print( "    images.dtype: %s" % (self.images.dtype) )
            
        if str(self.images.dtype) == 'float64':
            print( "    WARNING: running in double-precision (may be slow)" )
            
        try: print( "    Loading files (s): %.3f"%(self.bench['loaddata1'] - self.bench['loaddata0']) )
        except: pass
        try: print( "    Image/mask binning (s): %.3f"%(self.bench['bin1'] - self.bench['bin0']) ) 
        except: pass
        try: print( "    X-correlation initialization (s): %.3f"%(self.bench['xcorr1'] - self.bench['xcorr0']) )
        except: pass
        try: print( "    X-correlation forward FFTs (s): %.3f"%(self.bench['xcorr2'] - self.bench['xcorr1']) )
        except: pass
        try: print( "    X-correlation main computation (s): %.3f"%(self.bench['xcorr3'] - self.bench['xcorr2']) )
        except: pass
        try: print( "    Complete (entry-to-exit) xcorrnm2_tri (s): %.3f"%(self.bench['xcorr3'] - self.bench['xcorr0']) ) 
        except: pass
        try: print( "    Complete Unblur (s): %.3f" % (self.bench['unblur1'] - self.bench['unblur0']) )
        except: pass
        try: print( "    Shifts solver (last iteration, s): %.3f"%(self.bench['solve1'] - self.bench['solve0']) )
        except: pass
        try: print( "    Subpixel alignment (s): %.3f"%(self.bench['shifts1'] - self.bench['shifts0']) )
        except: pass
        try: print( "    Fourier Ring Correlation (s): %.3f"%(self.bench['frc1'] - self.bench['frc0']))
        except: pass
        try: print( "    Post-process filtering (s): %.3f"%(self.bench['dose1'] - self.bench['dose0']))
        except: pass
        try: print( "    Hotpixel mask (s): %.3f" % (self.bench['hot1'] - self.bench['hot0']))
        except: pass
        try: print( "    CTF estimation with %s (s): %.3f" %( self.CTFProgram, self.bench['ctf1']-self.bench['ctf0'] ) )
        except: pass
        try: print( "    Plot rendering (s): %.3f"%(self.bench['plot1'] - self.bench['plot0']))
        except: pass
        try:  print( "    Save files (s): %.3f"%(self.bench['savedata1'] - self.bench['savedata0']))
        except: pass
        print( "###############################" )
        try: print( "    Total execution time (s): %.3f"%(time.time() - self.bench['total0']) )
        except: pass
        pass



    
##### COMMAND-LINE INTERFACE ####
#if __name__ == '__main__':
#    main()

#!/opt/anaconda/bin/python
# -*- coding: utf-8 -*-
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
import ioMRC
import ioDM
import os
try:
    import pyfftw
except:
    print( "Zorro did not find pyFFTW package: get it at https://pypi.python.org/pypi/pyFFTW" )
try:
    import tables
except:
    print( "Zorro did not find pyTables installation for HDF5 file support" )
# import subprocess # This seems very buggy in Python 2.7

# DEBUG imports
import matplotlib.pyplot as plt
from plotting import ims

#### OBJECT-ORIENTED INTERFACE ####
class ImageRegistrator(object):
# Should be able to handle differences in translation, rotation, and scaling
# between images
    
    def __init__( self ):
        # Declare class members
        self.verbose = True
        self.saveC = False
        
        # FFTW_PATIENT is bugged for powers of 2, so use FFTW_MEASURE as default
        self.fftw_effort = "FFTW_MEASURE"
        self.wisdom_file = None # Pass in to use a fixed wisdom file when planning FFTW
        self.n_threads = None # Number of cores to limit FFTW to, if None uses all cores 
        if os.name == 'nt':
            self.cachePath = "C:\\Temp\\"
        else:
            self.cachePath = "/scratch/"
            
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
        self.Brad = None # Gaussian low-pass applied to data before registration, units are radius in Fourier space, or equivalent point-spread function in real-space
        self.Bmode = 'opti' # can be a real-space Gaussian convolution, 'conv' or Fourier filter, 'fourier', or 'opti' for automatic Brad
        # For Bmode = 'fourier', a range of available filters can be used: gaussian, gauss_trunc, butterworth.order (order is an int), hann, hamming
        self.BfiltType = 'gaussian'
#        self.rebinFact = None # Binning, generally powers of 2 are required at present
#        self.rebinMode = 'square' # Available binning types: square, magic, fourier
        self.fouCrop = None # Size of FFT in frequency-space to crop to (e.g. [2048,2048])
        self.reloadData = True
        
        # Data
        self.images = None
        self.imageSum = None
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
        self.CTF4Results = None
        self.FRC = None # A Fourier ring correlation
        
        # Registration parameters
        self.shapePadded = [4096,4096]
        self.shapeOriginal = None
        self.subPixReg = 16 # fraction of a pixel to REGISTER image shift to
        # Subpixel alignment method: None (shifts still registered subpixally), lanczos, or fourier
        # lanczos is cheaper but can have strong artifacts, fourier is more expensive but may have less artifacts
        self.subPixMethod = 'lanczos' 
        self.maxShift = 80 # Generally should be 1/2 distance to next lattice spacing
        # Pre-shift every image by that of the previous frame, useful for high-resolution where one can jump a lattice
        # i.e. should be used with small values for maxShift
        self.preShift = False
        # Solver weighting can be raw max correlation coeffs (None), normalized to [0,1] by the 
        # min and max correlations ('norm'), or 'logistic' function weighted which
        # requires corrThres to be set.
        self.weightMode = 'logistic' # normalized, unweighted, logistic, or corr
        self.logisticK = 5.0
        self.logisticNu = 0.15
        self.originMode = 'centroid' # 'centroid' or None
        self.suppressOrigin = True # Delete the XC pixel at (0,0).  Only necessary if gain reference is bad, but defaults to on.
        
        # Triangle-matrix indexing parameters
        self.triMode = 'auto' # Can be: tri, diag, auto, first
        self.diagWidth = 6
        self.autoMax = 10
        self.sigmaThres = 4.0
        self.pixErrThres = 2.5
        # corrThres should not be less than 0.0013 for 3710x3838 arrays, as that is the Poisson limit
        self.corrThres = None # Use with 'auto' mode to stop doing cross-correlations if the values drop below the threshold
        self.peaksigThres = 5.0
        self.velocityThres = None # Pixel velocity threshold (pix/frame), above which to throw-out frames with too much motion blur.
        
        #### INPUT/OUTPUT ####
        # cachePath _must_ have a trailing slash
        self.configName = None
        self.stackName = None
        self.maskName = None
        self.saveName = None # if None auto-generates + "_zorro.mrc"
        self.savePDF = False
        self.savePNG = False
        self.doCompression = False
        self.saveMovie = True
        #self.stack_ext = None
        #self.save_ext = '.mrc'
        # self.saveAlignOnly = False # Do you want to save the alignedStack?
        self.compress_ext = '.bz2'
        # Do you want to delete the raw input file after you save the aligned stack?
        # If False the raw stack will be compressed if compress_ext is not None but not deleted
        # self.deleteRaw = False 
        
        pass
    
    def xcorr2_mc( self, first_frame = 0, last_frame = 0, gpu_id = 0 ):
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
        stackFront, _ = os.path.splitext( self.stackName )
        
        if self.cachePath is None:
            self.cachePath = "."
            
        # Force trailing slashes onto cachePatch
        self.cachePath = os.path.join( self.cachePath, "" )    
            
        InName = os.path.join( self.cachePath, stackFront + "_mcIn.mrc" )
        # Unfortunately these files may as well be in the working directory.    
        OutAvName = stackFront + "_mcOutAv.mrc"
        OutStackName = os.path.join( self.cachePath, stackFront + "_mcOut.mrc" )
        LogName = stackFront + "_mc.log"
        ioMRC.MRCExport( self.images.astype('float32'), InName )

        first_frame = 0
        last_frame = 0

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
        if self.diagWidth is not None:
            fod = self.diagWidth
        else:
            fod = 0
        # Dosef can limit search to a certain box size    
        if self.maxShift is None:
            maxshift = 96
        else:
            maxshift = self.maxShift * 2

        motion_flags = (  " " + InName 
                + " -gpu " + str(gpu_id)
                + " -nss " + str(first_frame) 
                + " -nes " + str(last_frame) 
                + " -fod " + str(fod) 
                + " -bin " + str(binning) 
                + " -bft " + str(bfac) 
                + " -atm -" + str(align_to) 
                + " -pbx " + str(maxshift)
                + " -ssc 1 -fct " + OutStackName 
                + " -fcs " + OutAvName 
                + " -flg " + LogName )

        os.system(dosef_cmd + motion_flags)
        
        print( "Loading dosef aligned frames into self.images" )
        self.images = ioMRC.MRCImport( OutStackName )

        
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
        for J in xrange(0,frameCount):
            MCdrifts[J,:] = re.findall( r"([+-]?\d+.\d+)", MClog_crop[J] )[1:]
        # Zorro saves translations, motioncorr saves shifts.
        self.translations = -np.fliplr( MCdrifts )
        
        if self.originMode == 'centroid':
            centroid = np.mean( self.translations, axis=0 )
            self.translations -= centroid

        # Calclulate sum
        self.imageSum = np.sum( self.images, axis=0 )
        # Seems like motioncorr is not count-conserving?
        croppedSum = self.getSumCropToLimits()
        self.sumMean = np.mean( croppedSum  ) # Total dose
        self.sumVariance = np.var( croppedSum ) # Noise level
        self.sumContrast = np.sqrt( 2.0 * (self.sumVariance - self.sumMean) / self.sumMean**2 )
        
        time.sleep(0.5)
        try: os.remove(InName)
        except: pass
        try: os.remove(OutStackName)
        except: pass
        try: os.remove(OutAvName)
        except: pass
        try: os.remove(LogName)
        except: pass
        

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
            else: # 'tri' or 'auto' ; default is an upper triangular matrix
                triIndices = trimesh >= 1
            pass
        else:
            raise TypeError( "Error: triIndices not recognized as valid: " + str(triIndices) )
            
#        print( "#####DEBUG TRI-INDICES#######" )
#        print( "No eqns = " + str(np.sum(triIndices)) )
#        print( "" ); print( triIndices ); print( "" )
#        ims( triIndices )
        
        # Enforce image positivity
#        globalMin = np.min( self.images )
#        if( globalMin < 0.0 ):
#            self.images -= globalMin
            
        # Make images complex for FFTW
        # bitdepth = self.images.dtype.itemsize*8
        bitdepth = float_dtype.itemsize*8
        if not self.images.dtype.name.startswith( 'complex' ):
            bitdepth *= 2
            fftw_dtype = np.dtype( 'complex'+str(bitdepth) )
            # imagesMasked is a complex-version of images, that later
            # it should be deleted later.
        
        if self.masks is None or self.masks == []:
            print( "Warning: No mask not recommened with MNXC-style correlation" )
            self.masks = np.ones( [1,shapeImage[0],shapeImage[1]], dtype = self.images.dtype )
            
        if( self.masks.ndim == 2 ):
            print( "Forcing masks to have ndim == 3" )
            self.masks = np.reshape( self.masks.astype(self.images.dtype), [1,shapeImage[0],shapeImage[1]] )
             
        # Pre-loop allocation
        shiftsTriMat = np.zeros( [N,N,2], dtype=float_dtype ) # Triagonal matrix of shifts in [I,J,(y,x)]
        corrTriMat = np.zeros( [N,N], dtype=float_dtype ) # Triagonal matrix of maximum correlation coefficient in [I,J]
        peaksigTriMat = np.zeros( [N,N], dtype=float_dtype ) # Triagonal matrix of correlation peak contrast level
        peaksigTriMat_filt = np.zeros( [N,N], dtype=float_dtype ) # Triagonal matrix of correlation peak contrast level, before filtering
        
        # Make pyFFTW objects
        if self.fouCrop is None:
            tempFullframe = pyfftw.n_byte_align_empty( shapeImage, fftw_dtype.itemsize, dtype=fftw_dtype )
            FFT2, IFFT2 = util.pyFFTWPlanner( tempFullframe, wisdomFile=self.wisdom_file, effort = self.fftw_effort, n_threads=self.n_threads )
            shapeCropped = shapeImage
            tempComplex = pyfftw.n_byte_align_empty( shapeCropped, fftw_dtype.itemsize, dtype=fftw_dtype )
        else:
            tempFullframe = pyfftw.n_byte_align_empty( shapeImage, fftw_dtype.itemsize, dtype=fftw_dtype )
            FFT2, _ = util.pyFFTWPlanner( tempFullframe, wisdomFile=self.wisdom_file, effort = self.fftw_effort, n_threads=self.n_threads, doReverse=False )
            # Force fouCrop to multiple of 2
            shapeCropped = 2 * np.floor( np.array( self.fouCrop ) / 2.0 ).astype('int')
            tempComplex = pyfftw.n_byte_align_empty( shapeCropped, fftw_dtype.itemsize, dtype=fftw_dtype )
            _, IFFT2 = util.pyFFTWPlanner( tempComplex, wisdomFile=self.wisdom_file, effort = self.fftw_effort, n_threads=self.n_threads, doForward=False )
        
        templateImageFFT = pyfftw.n_byte_align_empty( shapeCropped, fftw_dtype.itemsize, dtype=fftw_dtype )
        templateSquaredFFT = pyfftw.n_byte_align_empty( shapeCropped, fftw_dtype.itemsize, dtype=fftw_dtype )
        templateMaskFFT = pyfftw.n_byte_align_empty( shapeCropped, fftw_dtype.itemsize, dtype=fftw_dtype )
        tempComplex2 = pyfftw.n_byte_align_empty( shapeCropped, fftw_dtype.itemsize, dtype=fftw_dtype )
        
        # Subpixel initialization
        # Ideally subPix should be a power of 2 (i.e. 2,4,8,16,32)
        subR = 8 # Sampling range around peak of +/- subR
        if self.subPixReg is None: self.subPixReg = 1;
        if self.subPixReg > 1.0:  
            # hannfilt = np.fft.fftshift( ram.apodization( name='hann', size=[subR*2,subR*2], radius=[subR,subR] ) ).astype( fftw_dtype )
            # Need a forward transform that is [subR*2,subR*2] 
            Csub = pyfftw.n_byte_align_empty( [subR*2,subR*2], fftw_dtype.itemsize, dtype=fftw_dtype )
            CsubFFT = pyfftw.n_byte_align_empty( [subR*2,subR*2], fftw_dtype.itemsize, dtype=fftw_dtype )
            subFFT2, _ = util.pyFFTWPlanner( Csub, fouMage=CsubFFT, wisdomFile=self.wisdom_file, effort = self.fftw_effort, n_threads=self.n_threads, doReverse = False )
            # and reverse transform that is [subR*2*subPix, subR*2*subPix]
            CpadFFT = pyfftw.n_byte_align_empty( [subR*2*self.subPixReg,subR*2*self.subPixReg], fftw_dtype.itemsize, dtype=fftw_dtype )
            Csub_over = pyfftw.n_byte_align_empty( [subR*2*self.subPixReg,subR*2*self.subPixReg], fftw_dtype.itemsize, dtype=fftw_dtype )
            _, subIFFT2 = util.pyFFTWPlanner( CpadFFT, fouMage=Csub_over, wisdomFile=self.wisdom_file, effort = self.fftw_effort, n_threads=self.n_threads, doForward = False )
        
        Corr_templateMask = np.zeros( shapeCropped )
        Corr_baseMask = np.zeros( shapeCropped )
        Numerator = np.zeros( shapeCropped )
        DenomTemplate = np.zeros( shapeCropped )
        DenomBase = np.zeros( shapeCropped )
        maskProduct = np.zeros( shapeCropped )
        # normConst = np.float32( 1.0 / Numerator.size )
        normConst2 = np.float32( 1.0 / Numerator.size**2 )
        if self.masks.shape[0] == 1 :
            # tempComplex = self.masks[0,:,:].astype( fftw_dtype ) 
            baseMaskFFT = pyfftw.n_byte_align_empty( shapeCropped, fftw_dtype.itemsize, dtype=fftw_dtype )

            FFT2.update_arrays( self.masks[0,:,:].squeeze().astype( fftw_dtype ), tempFullframe ); FFT2.execute()
            # FFTCrop
            baseMaskFFT[0:shapeCropped[0]/2,0:shapeCropped[1]/2] = tempFullframe[0:shapeCropped[0]/2,0:shapeCropped[1]/2]
            baseMaskFFT[0:shapeCropped[0]/2,-shapeCropped[1]/2:] = tempFullframe[0:shapeCropped[0]/2,-shapeCropped[1]/2:] 
            baseMaskFFT[-shapeCropped[0]/2:,0:shapeCropped[1]/2] = tempFullframe[-shapeCropped[0]/2:,0:shapeCropped[1]/2] 
            baseMaskFFT[-shapeCropped[0]/2:,-shapeCropped[1]/2:] = tempFullframe[-shapeCropped[0]/2:,-shapeCropped[1]/2:] 
            
            templateMaskFFT = baseMaskFFT.conj()
            
            # maskProduct term is M1^* .* M2
            tempComplex2 = ne.evaluate( "templateMaskFFT * baseMaskFFT" )
            IFFT2.update_arrays( tempComplex2, tempComplex ); IFFT2.execute()
            maskProduct = ne.evaluate( "normConst2*real(tempComplex)" )
        else:
            # Pre-allocate only
            baseMaskFFT = np.zeros( [N+1, shapeCropped[0], shapeCropped[1]], dtype=fftw_dtype )

        if bool( self.maxShift ) or self.Bmode is 'fourier':
            if self.maxShift is None or self.preShift is True:
                [xmesh,ymesh] = np.meshgrid( np.arange(-shapeCropped[0]/2, shapeCropped[0]/2), np.arange(-shapeCropped[1]/2, shapeCropped[1]/2)  )
            else:
                [xmesh,ymesh] = np.meshgrid( np.arange(-self.maxShift, self.maxShift), np.arange(-self.maxShift, self.maxShift)  )
            
            rmesh2 = ne.evaluate( "xmesh*xmesh + ymesh*ymesh" )
            # rmesh2 = xmesh*xmesh + ymesh*ymesh
            if bool( self.maxShift ): 
                mask_maxShift = ( rmesh2 < self.maxShift**2.0 )
            if self.Bmode is 'fourier':
                Bfilter = np.fft.fftshift( util.apodization( name=self.BfiltType, size=shapeCropped, radius=[self.Brad,self.Brad] ) )

        self.t[1] = time.time() 
        # Pre-compute forward FFTs (template will just be copied conjugate Fourier spectra)
        baseImageFFT = np.zeros( [N+1, shapeCropped[0], shapeCropped[1]], dtype=fftw_dtype )
        baseSquaredFFT = np.zeros( [N+1, shapeCropped[0], shapeCropped[1]], dtype=fftw_dtype )
        
        # Looping for triagonal matrix
        # For auto this is wrong, so make these lists instead
        currIndex = 0
        if self.saveC: self.C = []
        if self.saveCsub: self.Csub = []
        
        print( "Pre-computing forward Fourier transforms" )
        # For even-odd and noise estimates, we often skip many rows
        precompIndices = np.unique( np.vstack( [np.argwhere( np.sum( triIndices, axis=1 ) > 0 ), np.argwhere( np.sum( triIndices, axis=0 ) > 0 ) ] ) )
        for I in precompIndices:
            print( "Precomputing forward FFT frame: " + str(I) )
            # Apply masks to images
            if self.masks.shape[0] == 1:
                masks_block = self.masks[0,:,:]
                images_block = self.images[I,:,:]
            else:
                masks_block = self.masks[I,:,:]
                images_block = self.images[I,:,:]
                
            tempReal = ne.evaluate( "masks_block * images_block" ).astype( fftw_dtype )
            FFT2.update_arrays( tempReal, tempFullframe ); FFT2.execute()
            # FFTCrop
            baseImageFFT[I,0:shapeCropped[0]/2,0:shapeCropped[1]/2] = tempFullframe[0:shapeCropped[0]/2,0:shapeCropped[1]/2]
            baseImageFFT[I,0:shapeCropped[0]/2,-shapeCropped[1]/2:] = tempFullframe[0:shapeCropped[0]/2,-shapeCropped[1]/2:] 
            baseImageFFT[I,-shapeCropped[0]/2:,0:shapeCropped[1]/2] = tempFullframe[-shapeCropped[0]/2:,0:shapeCropped[1]/2] 
            baseImageFFT[I,-shapeCropped[0]/2:,-shapeCropped[1]/2:] = tempFullframe[-shapeCropped[0]/2:,-shapeCropped[1]/2:] 
                
            FFT2.update_arrays( ne.evaluate( "tempReal*tempReal" ).astype( fftw_dtype ), tempFullframe ); FFT2.execute()
            # FFTCrop
            baseSquaredFFT[I,0:shapeCropped[0]/2,0:shapeCropped[1]/2] = tempFullframe[0:shapeCropped[0]/2,0:shapeCropped[1]/2]
            baseSquaredFFT[I,0:shapeCropped[0]/2,-shapeCropped[1]/2:] = tempFullframe[0:shapeCropped[0]/2,-shapeCropped[1]/2:] 
            baseSquaredFFT[I,-shapeCropped[0]/2:,0:shapeCropped[1]/2] = tempFullframe[-shapeCropped[0]/2:,0:shapeCropped[1]/2] 
            baseSquaredFFT[I,-shapeCropped[0]/2:,-shapeCropped[1]/2:] = tempFullframe[-shapeCropped[0]/2:,-shapeCropped[1]/2:] 

            
            if not self.masks.shape[0] == 1:
                FFT2.update_arrays( self.masks[I,:,:].squeeze().astype( fftw_dtype), tempFullframe ); FFT2.execute()
                # FFTCrop
                baseMaskFFT[I,0:shapeCropped[0]/2,0:shapeCropped[1]/2] = tempFullframe[0:shapeCropped[0]/2,0:shapeCropped[1]/2]
                baseMaskFFT[I,0:shapeCropped[0]/2,-shapeCropped[1]/2:] = tempFullframe[0:shapeCropped[0]/2,-shapeCropped[1]/2:] 
                baseMaskFFT[I,-shapeCropped[0]/2:,0:shapeCropped[1]/2] = tempFullframe[-shapeCropped[0]/2:,0:shapeCropped[1]/2] 
                baseMaskFFT[I,-shapeCropped[0]/2:,-shapeCropped[1]/2:] = tempFullframe[-shapeCropped[0]/2:,-shapeCropped[1]/2:] 

            pass
        del masks_block, images_block
        self.t[2] = time.time() 
    
            
        print( "Starting correlation calculations" )
        # For even-odd and noise estimates, we often skip many rows
        rowIndices = np.unique( np.argwhere( np.sum( triIndices, axis=1 ) > 0 ) )
        for I in rowIndices:
            # I is the index of the template image
            templateImageFFT[:] = baseImageFFT[I,:,:]
            templateImageFFT = ne.evaluate( "conj(templateImageFFT)")
                
            templateSquaredFFT[:] =  baseSquaredFFT[I,:,:]
            templateSquaredFFT = ne.evaluate( "conj(templateSquaredFFT)")
    
            if not self.masks.shape[0] == 1:
                templateMaskFFT[:] = baseMaskFFT[I,:,:]
                templateMaskFFT = ne.evaluate( "conj(templateMaskFFT)")
    
            # This is the matrix of cross-correlations.  It's doesn't need to be a matrix 
            # if we don't want to introduce parallelization here.
            C = np.zeros( [shapeCropped[0], shapeCropped[1] ] )
    
            # Now we can start looping through base images
            # TODO: Multi-threading of each stage could happen here if we Cythonized the code
            columnIndices = np.unique( np.argwhere( triIndices[I,:] ) )
            for J in columnIndices:
                
                if not self.masks.shape[0] == 1:
                    # Compute maskProduct, term is M1^* .* M2
                    baseMask_block = baseMaskFFT[J,:,:]
                    tempComplex2 = ne.evaluate( "templateMaskFFT * baseMask_block" )
                    IFFT2.update_arrays( tempComplex2, tempComplex ); IFFT2.execute()
                    # maskProduct = np.clip( np.round( np.real( tempComplex ) ), eps, np.Inf )
                    maskProduct = ne.evaluate( "real(tempComplex)*normConst2" )
                    
                # Compute mask correlation terms
                if self.masks.shape[0] == 1:
                    IFFT2.update_arrays(  ne.evaluate( "baseMaskFFT * templateImageFFT"), tempComplex ); IFFT2.execute()
                else:
                    IFFT2.update_arrays( ne.evaluate( "baseMask_block * templateImageFFT"), tempComplex ); IFFT2.execute()
                #Corr_templateMask = tempComplex.copy( order='C' )
                Corr_templateMask = ne.evaluate( "real(tempComplex)*normConst2" ) # Normalization
                
                baseImageFFT_block = baseImageFFT[J,:,:]
                IFFT2.update_arrays( ne.evaluate( "templateMaskFFT * baseImageFFT_block"), tempComplex ); IFFT2.execute()
                #Corr_baseMask = tempComplex.copy( order='C' )

                # These haven't been normalized, so let's do so.  They are FFT squared, so N*N
                # This reduces the strain on single-precision range.
                Corr_baseMask =  ne.evaluate( "real(tempComplex)*normConst2" ) # Normalization

                # Compute Numerator (the phase correlation)
                tempComplex2 = ne.evaluate( "baseImageFFT_block * templateImageFFT" )
                IFFT2.update_arrays( tempComplex2, tempComplex ); IFFT2.execute()

                Numerator = ne.evaluate( "real(tempComplex)*normConst2" ) # Normalization
                Numerator = ne.evaluate( "Numerator - real( Corr_templateMask * Corr_baseMask / maskProduct)" ) 
                # Compute the intensity normalzaiton for the template
                if self.masks.shape[0] == 1:
                    IFFT2.update_arrays( ne.evaluate( "baseMaskFFT * templateSquaredFFT"), tempComplex ); IFFT2.execute()
                else:
                    IFFT2.update_arrays( ne.evaluate( "baseMaskFFT_blcok * templateSquaredFFT"), tempComplex ); IFFT2.execute()

                DenomTemplate = ne.evaluate( "real(tempComplex)*normConst2") # Normalization
                DenomTemplate = ne.evaluate( "DenomTemplate - real( Corr_templateMask * (Corr_templateMask / maskProduct) )" )
                
                # Compute the intensity normalzaiton for the base Image
                baseSquared_block = baseSquaredFFT[J,:,:]
                IFFT2.update_arrays( ne.evaluate( "templateMaskFFT * baseSquared_block"), tempComplex ); IFFT2.execute()
                DenomBase = ne.evaluate( "real(tempComplex)*normConst2") # Normalization
                DenomBase = ne.evaluate( "DenomBase - real( Corr_baseMask * (Corr_baseMask / maskProduct) )" )
                
    
                ne.evaluate( "sqrt( DenomBase * DenomTemplate )", out=DenomTemplate )
                # What happened to numexpr clip?
                DenomTemplate = np.clip( DenomTemplate, 1, np.Inf )
                # print( "Number of small Denominator values: " + str(np.sum(DenomTemplate < 1.0)) )
                
                C = ne.evaluate( "Numerator / DenomTemplate" )

                # C = np.clip( C, -1, 1 ) 
                if bool(self.suppressOrigin):
                    # If gain reference is quite old we can still get some cross-artifacts.
                    # TODO: better methodology?  Median filter over a small area?
                    # Fit to the Fourier window?
                    C[0,0] = 0.25 * ( C[1,1] + C[-1,1] + C[-1,1] + C[-1,-1] )
                    
                # We have everything in normal FFT order until here; Some speed-up could be found by its removal.
                # Pratically we don't have to do this fftshift, but it makes plotting easier to understand
                C = np.fft.ifftshift( C )

                # We can crop C if maxShift is not None and preShift is False
                if self.maxShift is not None and self.preShift is False:
                    C = C[shapeCropped[0]/2-self.maxShift:shapeCropped[0]/2+self.maxShift, shapeCropped[1]/2-self.maxShift:shapeCropped[1]/2+self.maxShift]
                
                # if self.verbose:
                #     ims( (C,), titles=("I: " + str(I)+ ", J: " + str(J) ), cutoff=1E-4 )
                #     time.sleep(0.5)

                #### Find maximum positions ####    
                # Apply B-factor low-pass filter to correlation function
                if self.Bmode == 'opti':
                    self.t[20] = time.time()
                    # Want to define this locally so it inherits scope.
                    def inversePeakContrast( Bsigma ):
                        C_filt = scipy.ndimage.gaussian_filter( C, Bsigma )
                        return  np.std(C_filt ) / (np.max(C_filt ) - np.mean(C_filt ) )
                            
                    # B_opti= scipy.optimize.fminbound( inversePeakContrast, 0.0, 10.0, xtol=1E-3 )
                    sigmaOptiMax = 7.0
                    sigmaOptiMin = 0.0
                    result = scipy.optimize.minimize_scalar( inversePeakContrast, bounds=[sigmaOptiMin,sigmaOptiMax], method="bounded"  )
                    C_filt = scipy.ndimage.gaussian_filter( C, result.x )
                    self.t[21] = time.time()
                    if self.verbose >= 2:
                        print( "Found optimum B-sigma: %.3f"%result.x + ", with peak sig: %.3f"%(1.0/result.fun)+" in %.2f"%(self.t[21]-self.t[20])+" s" ) 
                elif bool(self.Brad) and self.Bmode =='fourier':
                    tempComplex = C.astype(fftw_dtype)
                    FFT2.update_arrays( tempComplex, tempComplex2 ); FFT2.execute()
                    IFFT2.update_arrays( ne.evaluate( "tempComplex2*Bfilter" ), tempComplex ); IFFT2.execute()
                    # Conservation of counts with Fourier filtering is not 
                    # very straight-forward.
                    C_filt = ne.evaluate( "real( tempComplex )/sqrt(normConst)" )
                elif bool(self.Brad) and self.Bmode == 'conv' or self.Bmode == 'convolution':
                    # Convert self.Brad as an MTF to an equivalent sigma for a PSF
                    # TODO: Check that Bsigma is correct with Fourier cropping"
                    Bsigma = shapeImage / (np.sqrt(2) * np.pi * self.Brad)
                    # Scipy's gaussian filter conserves total counts
                    C_filt = scipy.ndimage.gaussian_filter( C, Bsigma )
                else: # No filtering
                    C_filt = C
                
                
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
                        
                    self.corrStats['meanC'][currIndex] = np.mean(C_filt)
                    self.corrStats['varC'][currIndex] = np.var(C_filt)
                    self.corrStats['maxC'][currIndex] = np.max(C_filt)
                    self.corrStats['maxPos'][currIndex,:] = np.unravel_index( np.argmax(C_filt), shapeCropped ) - np.array([C_filt.shape[0]/2, C_filt.shape[1]/2])
                    self.corrStats['originC'][currIndex] = C_filt[C.shape[0]/2, C.shape[1]/2]   
                    
                # Apply maximum shift max mask, if present
                if bool( self.maxShift ):
                    # for previous frame alignment compensation, we need to shift the mask around...
                    if bool( self.preShift ):
                        rolledMask = np.roll( np.roll( mask_maxShift, 
                            np.round(shiftsTriMat[I,J-1,0]).astype('int'), axis=0 ), 
                            np.round(shiftsTriMat[I,J-1,1]).astype('int'), axis=1 )
                        C_masked = ne.evaluate("C_filt*rolledMask")
                        cmaxpos = np.unravel_index( np.argmax( C_masked ), C_masked.shape )
                        peaksigTriMat[I,J] = (C_masked[cmaxpos] - np.mean(C_filt[rolledMask]))/ np.std(C_filt[rolledMask])
                    else:
                        C_masked = ne.evaluate("C_filt*mask_maxShift")
                        cmaxpos = np.unravel_index( np.argmax( C_masked ), C_filt.shape )
                        peaksigTriMat[I,J] = (C_masked[cmaxpos] - np.mean(C_filt[mask_maxShift]))/ np.std(C_filt[mask_maxShift])
                else: # No maxshift
                    cmaxpos = np.unravel_index( np.argmax(C_filt), C_filt.shape )
                    peaksigTriMat[I,J] = (corrTriMat[I,J] - np.mean(C_filt))/ np.std(C_filt)
                    


                    
#                if self.verbose:
#                    clim = util.histClim( C, cutoff=1E-3 )
#                    plt.subplot( '121' )
#                    plt.imshow( C, vmin=clim[0], vmax=clim[1], interpolation='none' )
#                    # plt.imshow( C )
#                    plt.title( "I: " + str(I) + ", J: " + str(J) )
#                    clim2 = util.histClim( C_filt, cutoff=1E-3 )
#                    plt.subplot( '122' )
#                    plt.imshow( C_filt, vmin=clim2[0], vmax=clim2[1], interpolation='none' )
#                    plt.title("Filtered C")
#                    plt.pause(0.05)
  
                
                if self.saveC:
                    # Maybe save in a pyTable if it's really needed.
                    if self.preShift:
                        self.C.append(C_filt*rolledMask)
                    else:
                        self.C.append(C_filt)

                if self.saveCsub:
                    self.Csub0.append(C_filt[cmaxpos[0]-subR:cmaxpos[0]+subR, cmaxpos[1]-subR:cmaxpos[1]+subR ])
                            
                if self.subPixReg > 1.0: # Subpixel peak estimation by Fourier interpolation
                    # print( "Interpolating unfiltered correlation!" )
                    # TODO: use filtered or unfiltered correlation for sub-pixel interpolation?
                    # Also change peaksigTriMat below!
                    # Csub = C_filt[cmaxpos[0]-subR:cmaxpos[0]+subR, cmaxpos[1]-subR:cmaxpos[1]+subR ]
                    Csub = C_filt[cmaxpos[0]-subR:cmaxpos[0]+subR, cmaxpos[1]-subR:cmaxpos[1]+subR ]
                        
                    # Csub is shape [2*subR, 2*subR]
                    if Csub.shape[0] == 2*subR and Csub.shape[1] == 2*subR:
                        subFFT2.update_arrays( Csub.astype( fftw_dtype ), CsubFFT ); subFFT2.execute()
                        # padding has to be done from the middle
                        # Try removing the hannfilt?
                        CpadFFT = np.pad( np.fft.fftshift(CsubFFT), ((self.subPixReg-1)*subR,), mode='constant', constant_values=(0.0,)  )
                        CpadFFT = np.fft.ifftshift( CpadFFT )
                        subIFFT2.update_arrays( CpadFFT, Csub_over ); subIFFT2.execute()
                        # Csub_overAbs = ne.evaluate( "abs( Csub_over )") # This is still complex
                        Csub_overAbs = np.abs( Csub_over )
                        
                        Csub_maxpos = np.unravel_index( np.argmax( Csub_overAbs ), Csub_overAbs.shape )

                        round_pos = cmaxpos - np.array(C.shape)/2.0
                        # Csub_max is being shifted 1 sub-pixel in the negative direction compared to the integer shift
                        # because of array centering, hence the np.sign(round_pos)
                        remainder_pos = Csub_maxpos - np.array(Csub_over.shape)/2.0 + np.sign( round_pos )
                        remainder_pos /= self.subPixReg
                        
                        # shiftsTriMat[I,J-1,:] = cmaxpos + np.array( Csub_maxpos, dtype='float' )/ np.float(self.subPixReg) - np.array( [subR, subR] ).astype('float')
                        shiftsTriMat[I,J,:] = round_pos + remainder_pos
                        # Switching from FFTpack to pyFFTW has messed up the scaling of the correlation coefficients, so
                        # scale by (subR*2.0)**2.0
                        corrTriMat[I,J] = Csub_overAbs[ Csub_maxpos[0], Csub_maxpos[1] ] / (subR*2.0)**2.0
                    else:
                        print( "Correlation sub-area too close to maxShift!  Subpixel location broken." )
                        shiftsTriMat[I,J,:] = cmaxpos - np.array(C.shape)/2.0
                        corrTriMat[I,J] = C[ cmaxpos[0], cmaxpos[1] ]   
                else: # Do integer pixel registration
                    shiftsTriMat[I,J,:] = cmaxpos - np.array(C.shape)/2.0
                    corrTriMat[I,J] = C[ cmaxpos[0], cmaxpos[1] ] 
                
#                # See what the peakSig is after filtering.
#                if self.maxShift is None:
#                    peaksigTriMat[I,J] = (corrTriMat[I,J] - np.mean(C) )/ np.std(C)
#                else:
#                    if bool( self.preShift ):
#                        peaksigTriMat[I,J] = (corrTriMat[I,J] - np.mean(C[rolledMask]))/ np.std(C[rolledMask])
#                    else:
#                        # TODO only calculate C values over the maxshift mask indices
#                        peaksigTriMat[I,J] = (corrTriMat[I,J] - np.mean(C) )/ np.std(C)
                    
                    
                if self.verbose: 
                    print( "# " + str(I) + "->" + str(J) + " shift: [%.2f"%shiftsTriMat[I,J,0] 
                        + ", %.2f"%shiftsTriMat[I,J,1]
                        + "], cc: %.6f"%corrTriMat[I,J] 
                        + ", peak sig: %.3f"%peaksigTriMat[I,J] )
#                        + ", filtered sig: %.3f"%peaksigTriMat_filt[I,J] )          
                    
                # triMode 'auto' diagonal mode    
                if self.triMode == 'auto' and (peaksigTriMat[I,J] <= self.peaksigThres or J-I >= self.autoMax):
                    if self.verbose: print( "triMode 'auto' stopping at frame: " + str(J) )
                    break
                currIndex += 1
            pass # C max position location
        self.t[3] = time.time()
        
        if self.fouCrop is not None:
            shiftsTriMat[:,:,0] *= self.shapePadded[0] / shapeCropped[0]
            shiftsTriMat[:,:,1] *= self.shapePadded[1] / shapeCropped[1]
        
        # Explicite housekeeping for NumExpr pointers
        del baseImageFFT_block, maskProduct, Corr_templateMask, 
        del Corr_baseMask, Numerator, DenomBase, DenomTemplate, baseSquared_block
        try: del baseMask_block
        except: pass
        try: del Bfilter
        except: pass
        try: del rolledMask
        except: pass
        # return the shifts and correlation matrices
        # They are members of the class but this is a light-weight way to not over-write the base
        # class values if the programmer chooses to test with multiple runs of this function.
        return shiftsTriMat, corrTriMat, peaksigTriMat, peaksigTriMat_filt
    
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
        elif self.weightMode == 'logistic':
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

        # errorX and Y are upper triangular version of errorXY
        errorX = np.zeros( N )
        errorY = np.zeros( N )
        # Sum horizontally and vertically, keeping in mind diagonal is actually at x-1
        for J in np.arange(0,N):
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
        
        # Build a dictionary of all the feedback parameters 
        errorDict = {}
        errorDict['relativeEst'] = relativeEst
        errorDict['acceptedEqns'] = acceptedEqns
        # Not necessary to save triIndices, it's the non-zero elements of corrTriMat
        # errorDict['triIndices'] = triIndices
        errorDict['errorXY'] = errorXY
        errorDict['shiftsTriMat'] = shiftsTriMat_in
        errorDict['corrTriMat'] = corrTriMat_in
        errorDict['peaksigTriMat'] = peaksigTriMat_in
        errorDict['errorX'] = errorX 
        errorDict['errorY'] = errorY 
        errorDict['errorUnraveled'] = errorUnraveled
        errorDict['mean_errorNorm'] = mean_errorNorm
        errorDict['std_errorNorm'] = std_errorNorm 
        errorDict['M'] = M
        errorDict['Maccepted'] = Maccepted
        # Append the dictionary to the list of dicts and return it as well
        self.errorDictList.append( errorDict )
        
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
            
        """
        Application of binning and padding.
        """
        # We will RELOAD the data from disk after alignment.
        
#        if self.rebinFact > 1.0:
#            self.binData()
            
        if np.any(self.shapePadded):
            self.padStack( self.shapePadded )
            
        """
        Registration, first run: Call xcorrnm2_tri to do the heavy lifting
        """
        self.t[8] = time.time() 
        [shiftsTriMat1st, corrTriMat1st, peaksigTriMat1st, peaksigTriMat1st_filt] = self.xcorrnm2_tri()
        #DEBUG
        self.peaksigTriMat_filt = peaksigTriMat1st_filt
        self.t[9] = time.time() 
        
        """
        Functional minimization over system of equations
        """
        self.t[11] = time.time()
        if self.triMode == 'first':
            self.translations = -np.vstack( (np.zeros([1,2]), shiftsTriMat1st[0,:]) )
            self.errorDictList.append({})
            self.errorDictList[-1]['shiftsTriMat'] = shiftsTriMat1st
            self.errorDictList[-1]['corrTriMat'] = corrTriMat1st
            self.errorDictList[-1]['peaksigTriMat'] = peaksigTriMat1st
        else:
            # I should probably save a list of dictionaries
            errorDict1st = self.shiftsSolver( shiftsTriMat1st, corrTriMat1st, peaksigTriMat1st )
            
            # Now do a second iteration, throwing out outlier equations.
            # We apply thresholding seperately on each axis because flags 
            # are typically worse than skips for STEM
            # First apply pixel error threshold, then the sigma standard deviation threshold
            errorXY = np.abs( errorDict1st['errorXY'] )
            pixErrAccepted = errorXY < self.pixErrThres
        
            stdThreshold = np.zeros( 2 )
            if self.sigmaThres < 1.0: 
                print( "WARNING: excessive sigma thresholding not recommend as it increases bias" )
            stdThreshold[0] = np.mean( errorXY[pixErrAccepted[:,0],0] ) + self.sigmaThres*np.std( errorXY[pixErrAccepted[:,0],0] )
            print( "Thresholding Y-equations with sigma = %.2f"%self.sigmaThres + " giving threshold limit of %.2f"%stdThreshold[0] + " pixels error.")
            stdThreshold[1] = np.mean( errorXY[pixErrAccepted[:,1],1] ) + self.sigmaThres*np.std( errorXY[pixErrAccepted[:,1],1])
            print( "Thresholding X-equations with sigma = %.2f"%self.sigmaThres + " giving threshold limit of %.2f"%stdThreshold[1] + " pixels error.")        
            goodEqns = (pixErrAccepted[:,0] * pixErrAccepted[:,1] 
                * (errorXY[:,0] < stdThreshold[0]) * (errorXY[:,1] < stdThreshold[1]) )# This is raveled
                
            self.shiftsSolver( shiftsTriMat1st, corrTriMat1st, peaksigTriMat1st, acceptedEqns=goodEqns )
            # I want stdThreshold to be in the dict for plotting
            self.errorDictList[-1]['stdThreshold'] = stdThreshold
            self.t[12] = time.time()
            
            # translations (to apply) are the negative of the found shifts
            self.translations = -np.vstack( (np.zeros([1,2]), np.cumsum( errorDict1st['relativeEst'], axis=0 ) ) )
            
            
        """
        Alignment and projection through Z-axis (averaging)
        """
#        # RELOAD DATA IF IT WAS CHANGED BY BINNING
#        if self.reloadData and self.rebinFact >= 2.0:
#            print( "Reloading data from disk due to binning" )
#            self.loadData( self.stackName )
#            # scale translations if data was binned
#            self.translations *= self.rebinFact
#            # TODO: Error dicts are all off by a factor of binning, including the errors...
        if np.any(self.shapePadded): # CROP back to original size
            print( "Cropping auto-applied mask pixels" )
            self.cropStack( self.shapeOriginal )
            
        self.applyShifts()
        
        self.t[10] = time.time() 
        

        pass # End of alignImageStack
        

    
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
        if self.subPixReg > 1.0 and self.subPixMethod == 'fourier':
            bitdepth = self.images.dtype.itemsize*8
            if not self.images.dtype.name.startswith( 'complex' ):
                bitdepth *= 2
                fftw_dtype = np.dtype( 'complex'+str(bitdepth) )
            # Setup FFTs for shifting.
            FFTImage = pyfftw.n_byte_align_empty( self.images.shape[1:], fftw_dtype.itemsize, dtype=fftw_dtype )
            RealImage = pyfftw.n_byte_align_empty( self.images.shape[1:], fftw_dtype.itemsize, dtype=fftw_dtype )
            # Make pyFFTW objects
            FFT2, IFFT2 = util.pyFFTWPlanner( FFTImage, effort = self.fftw_effort, n_threads=self.n_threads )
            [xmesh, ymesh] = np.meshgrid( np.arange(-RealImage.shape[1]/2,RealImage.shape[1]/2) / np.float(RealImage.shape[1] ), 
                np.arange(-RealImage.shape[0]/2,RealImage.shape[0]/2)/np.float(RealImage.shape[0]) )
        
        for J in xrange(1,m): # No need to shift first image
            if self.subPixReg > 1.0 and self.subPixMethod == 'lanczos':
                # Lanczos realspace shifting
                # self.images[J,:,:] = util.imageShiftAndCrop( self.images[J,:,:], shifts_round[J,:] )
                #Roll the image instead to preserve information in the stack, in case someone deletes the original
                self.images[J,:,:] = np.roll( np.roll( self.images[J,:,:], shifts_round[J,0], axis=0 ), shifts_round[J,1], axis=1 )
                
                self.images[J,:,:] = util.lanczosSubPixShift( self.images[J,:,:], subPixShift=shifts_remainder[J,:], kernelShape=5, lobes=3 )
                
                if self.verbose: print( "Correction (lanczos) "+ str(np.around(self.translations[J,:],decimals=4))+" applied to image: " + str(J) )
            elif self.subPixReg > 1.0 and self.subPixMethod == 'fourier':
                # Fourier gradient subpixel shift
                # TODO: do we want to crop out the circular-shifted part?  getSumCropToLimits already does this...
                RealImage = self.images[J,:,:].astype( fftw_dtype )
                FFT2.update_arrays( RealImage, FFTImage ); FFT2.execute()
                FFTImage *= np.fft.fftshift( np.exp( -2.0j * np.pi * (xmesh*self.translations[J,1] + ymesh*self.translations[J,0]) )  )
                IFFT2.update_arrays( FFTImage, RealImage ); IFFT2.execute()
                # Normalize and reduce to float32
                self.images[J,:,:] = np.real( RealImage ).astype(self.images.dtype) / RealImage.size
                
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

    
#    def binData( self ):
#        self.t[13] = time.time()
#        print( "Binning images and masks by: " + str(self.rebinFact) + " using method: " + self.rebinMode )
#        # Force binning to be one of 2,4,8,16, ... for square and magic modes
#        if self.rebinMode is 'square' or self.rebinMode is 'magic':
#            numbins = np.int( np.floor( np.log2( self.rebinFact ) ) )
#            self.rebinFact = 2**numbins
#            
#        # Check to see if binning results in fractional image shapes
#        if np.mod( self.images.shape[1]/self.rebinFact, 1.0 ) !=  0.0 or np.mod( self.images.shape[0]/self.rebinFact, 1.0 ) !=  0.0:
#            print( "Warning: binning is resulting in fractional images, some errors in registration will result." )
#            
#        binnedImages = np.zeros( [self.images.shape[0], 
#                                  self.images.shape[1]/self.rebinFact,
#                                    self.images.shape[2]/self.rebinFact ],
#                                    dtype=self.images.dtype )
#        if self.masks is not None:
#            binnedMasks = np.zeros( [self.masks.shape[0],
#                                     self.masks.shape[1]/self.rebinFact,
#                                     self.masks.shape[2]/self.rebinFact ],
#                                     dtype=self.masks.dtype )
#        if self.rebinMode is 'square':
#            # TODO: re-write squarekernel to have 
#            for J in np.arange(0,self.images.shape[0]):
#                binnedImages[J,:,:] = util.squarekernel( self.images[J,:,:], k=numbins )
#            if self.masks is not None:    
#                for J in np.arange(0,self.masks.shape[0]):
#                    binnedMasks[J,:,:] = util.squarekernel( self.masks[J,:,:], k=numbins )
#        elif self.rebinMode is 'magic':
#            for J in np.arange(0,self.images.shape[0]):
#                binnedImages[J,:,:] = util.magickernel( self.images[J,:,:], k=numbins )
#            if self.masks is not None:
#                for J in np.arange(0,self.masks.shape[0]):
#                    binnedMasks[J,:,:] = util.magickernel( self.masks[J,:,:], k=numbins )
#
#        elif self.rebinMode is 'fourier':
#            bitdepth = self.images.dtype.itemsize*8
#            if not self.images.dtype.name.startswith( 'complex' ):
#                bitdepth *= 2
#                fftw_dtype = np.dtype( 'complex'+str(bitdepth) )
#            
#            FFTImage = pyfftw.n_byte_align_empty( self.images.shape[1:], fftw_dtype.itemsize, dtype=fftw_dtype )
#            FFTCrop = pyfftw.n_byte_align_empty( binnedImages.shape[1:], fftw_dtype.itemsize, dtype=fftw_dtype )
#            RealCrop = pyfftw.n_byte_align_empty( binnedImages.shape[1:], fftw_dtype.itemsize, dtype=fftw_dtype )
#            print( FFTImage.shape )
#            print( FFTCrop.shape )
#            print( RealCrop.shape )
#            binFFT2, _ = util.pyFFTWPlanner( FFTImage, effort = self.fftw_effort, n_threads=self.n_threads, doReverse=False )
#            _, binIFFT2 = util.pyFFTWPlanner( FFTCrop, effort = self.fftw_effort, n_threads=self.n_threads, doForward=False )
#            for J in np.arange(0,self.images.shape[0]):
#                binFFT2.update_arrays( self.images[J,:,:].squeeze().astype(fftw_dtype), FFTImage ); binFFT2.execute()
#                FFTCrop = np.fft.fftshift( FFTImage )[self.images.shape[1]/2-binnedImages.shape[1]/2:self.images.shape[1]/2+binnedImages.shape[1]/2, 
#                    self.images.shape[2]/2-binnedImages.shape[2]/2:self.images.shape[2]/2+binnedImages.shape[2]/2]
#                FFTCrop = np.fft.ifftshift( FFTCrop )
#                binIFFT2.update_arrays( FFTCrop, RealCrop  ); binIFFT2.execute()
#                binnedImages[J,:,:] = np.real( RealCrop )
#            if self.masks is not None:
#                # BUG: masks are already the right shape here.
#                for J in np.arange(0,self.masks.shape[0]):
#                    binFFT2.update_arrays( self.masks[J,:,:].squeeze().astype(fftw_dtype), FFTImage ); binFFT2.execute()
#                    FFTCrop = np.fft.fftshift( FFTImage )[self.masks.shape[1]/2-binnedMasks.shape[1]/2:self.masks.shape[1]/2+binnedMasks.shape[1]/2, 
#                        self.masks.shape[2]/2-binnedMasks.shape[2]/2:self.masks.shape[2]/2+binnedMasks.shape[2]/2]
#                    FFTCrop = np.fft.ifftshift( FFTCrop )
#                    print( FFTCrop.shape )
#                    binIFFT2.update_arrays( FFTCrop, RealCrop  ); binIFFT2.execute()
#                    binnedMasks[J,:,:] = np.real( RealCrop )
#            pass    
#        self.images = binnedImages
#        if self.masks is not None:
#            self.masks = binnedMasks
#        self.t[14] = time.time()
            
    def velocityCull( self, velocityThres=None ):
        """
        Computes the pixel velocities, using a 5-point numerical differentiation on the 
        translations.  Note that a 5-point formula inherently has some low-pass filtering
        built-in.
        
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
        print( "Padding images and masks to shape: " + str(padSize) )
            
        m = self.images.shape[0]
        shapeImage = [ self.images.shape[1], self.images.shape[2] ]
        self.shapePadded = padSize
        self.shapeOriginal = shapeImage
        
        paddedImages = np.zeros( [m, padSize[0], padSize[1]], dtype=self.images.dtype )
        paddedImages[:,:shapeImage[0],:shapeImage[1]] = self.images
        self.images = paddedImages
        # Then make or pad the mask appropriately.
        if self.masks is None:
            self.masks = np.zeros( [1,padSize[0],padSize[1]], dtype='bool', order='C' )
            self.masks[0,:shapeImage[0],:shapeImage[1]] = 1.0
        else:
            mmask = self.masks.shape[0]
            paddedMasks = np.zeros( [mmask, padSize[0], padSize[1]], dtype=self.masks.dtype )
            paddedMasks[:,:shapeImage[0],:shapeImage[1]] = self.masks
            self.masks = paddedMasks
            pass
        pass
    
    def cropStack( self, cropSize ):
        """
        Undos the operation from ImageRegistrator.padStack()
        
        Defaults to self.shapeOriginal.
        """
        if cropSize is None:
            cropSize = self.shapeOriginal
            
        if not bool(cropSize):
            print( "Cannot crop to: " + str(cropSize) )
            return
            
        self.images = self.images[ :, :cropSize[0], :cropSize[1] ]
        # Crop masks too
        self.masks = self.masks[ :, :cropSize[0], :cropSize[1] ]
    
    def calcIncoherentFourierMag( self ):
        """
        Compute the Fourier transform of each frame in the movie and average
        the Fourier-space magnitudes.  This gives a baseline to compare how 
        well the alignment did vesus the  spatial information content of the 
        individual images.
        
        This is the square root of the power spectrum.  
        """
        itemsize = 8
        frameFFT = pyfftw.n_byte_align_empty( self.images.shape[1:], itemsize, dtype='complex64' )
        self.incohFouMag = np.zeros( self.images.shape[1:], dtype=float_dtype )
        FFT2, _ = util.pyFFTWPlanner( frameFFT, n_threads = self.n_threads, doReverse=False )
        
        for J in xrange(0,self.images.shape[0]):
            FFT2.update_arrays( np.squeeze( self.images[J,:,:]).astype('complex64'), frameFFT ); 
            FFT2.execute()
            self.incohFouMag += np.abs( frameFFT )
        pass
        self.incohFouMag = np.fft.fftshift( self.incohFouMag / self.images.shape[0] )
        
    def evenOddFouRingCorr( self, xcorr = 'tri', box=[256,256], overlap=0.5, debug=False ):
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
                     
        evenReg = ImageRegistrator()
        evenReg.loadConfig( logName )
        evenReg.images = self.images[evenIndices,:,:].copy(order='C')
        oddReg = ImageRegistrator()
        oddReg.loadConfig( logName )
        oddReg.images = self.images[oddIndices,:,:].copy(order='C')
        
        if self.masks is None:
            evenReg.masks = util.edge_mask( maskShape=[ self.images.shape[1], self.images.shape[2] ] )
            oddReg.masks = evenReg.masks
        elif self.masks.shape[0] > 1:
            evenReg.masks = self.masks[evenIndices,:,:]
            oddReg.masks = self.masks[oddIndices,:,:]
        elif self.masks.shape[0] == 1:
            evenReg.masks = self.masks
            oddReg.masks = self.mask
                
        if xcorr == 'tri' or xcorr is None:
            print( "#####  Zorro even frames alignment  #####" )
            evenReg.alignImageStack()
            self.transEven = evenReg.translations.copy( order='C' )

            print( "#####  Zorro odd frames alignment  #####" )
            oddReg.alignImageStack()
            self.transOdd = oddReg.translations.copy( order='C' )
            
        elif xcorr == 'mc':
            print( "#####  Motioncorr even frames alignment  #####" )
            evenReg.xcorr2_mc()
            self.transEven = evenReg.translations.copy( order='C' )

            print( "#####  Motioncorr odd frames alignment  #####" )
            oddReg.xcorr2_mc()
            self.transOdd = oddReg.translations.copy( order='C' )
        elif xcorr == 'unblur':
            print( "#####  UnBlur even frames alignment  #####" )
            evenReg.xcorr2_unblur()
            self.transEven = evenReg.translations.copy( order='C' )

            print( "#####  UnBlur odd frames alignment  #####" )
            oddReg.xcorr2_unblur()
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
                       translations=np.hstack( [self.transEven, self.transOdd] ), box=box, overlap=overlap )
        
        self.FRC2D = eoReg.FRC2D
        self.FRC = eoReg.FRC
        if self.saveC:
            self.evenC = evenReg.C
            self.oddC = oddReg.C
        if self.saveCsub:
            self.evenCsub = evenReg.Csub
            self.oddCsub = oddReg.Csub
        return evenReg, oddReg
        
    def lazyFouRingCorr( self, box=[512,512], overlap=0.0, debug=False ):
        # Computes the FRC from the full stack, taking even and odd frames for the half-sums
        # These are not independant half-sets! ... but it still gives us a very good impression 
        # of alignment success or failure.
        m = self.images.shape[0]
        evenIndices = np.arange(0, m, 2)
        oddIndices = np.arange(1, m, 2)                     
        
        evenSum = np.sum( self.images[evenIndices,:,:] )
        oddSum = np.sum( self.images[oddIndices,:,:] )
        
        self.tiledFRC( evenSum, oddSum, box=box, overlap=overlap )
        pass
    
    def tiledFRC( self, Image1, Image2, translations=None, box=[512,512], overlap=0.5 ):
        """
        Pass in two images, which are ideally averages from two independantly processed half-sets. 
        Compute the FRC in many tiles of shape 'box', and average the FRC over all tiles.
        
        Overlap controls how much tiles overlap by, with 0.5 being half-tiles and 0.0 being no overlap,
        i.e. they are directly adjacent.  Negative overlaps may be used for sparser samping.  
        
        Produced both a 2D FRC, which is generally of better quality than a power-spectrum, and 
        """
        FFT2, _ = util.pyFFTWPlanner( np.zeros(box, dtype=fftw_dtype), 
                             wisdomFile=self.wisdom_file, n_threads = self.n_threads, 
                             effort=self.fftw_effort, doReverse=False )
        if overlap > 0.8:
            print("tiledFRC takes huge amounts of time as overlap->1.0" )
            overlap = 0.8                
            
        if translations is None:
            translations = self.translations
        if not np.any(translations):
            transLim = np.array( [5,5,5,5] ) # Keep away from any edge artifacts
        else:
            transLim = np.array( [ np.maximum( np.ceil( translations[:,0].max() ), 0 ),
                        -np.minimum( np.floor( translations[:,0].min() ), 0 ),
                         np.maximum( np.ceil( translations[:,1].max() ), 0 ),
                        -np.minimum( np.floor( translations[:,1].min() ), 0 ) ] )
            transLim += 5
                         
        print( transLim )
        print( Image1.shape )
        hann = util.apodization( name='hann', shape=box ).astype(float_dtype)
        tilesX = np.floor( np.float( Image1.shape[0] - transLim[2] - transLim[3] - box[1])/ box[1] / (1.0-overlap) ).astype('int')
        tilesY = np.floor( np.float( Image1.shape[1] - transLim[0] - transLim[1] - box[0])/ box[0] / (1.0-overlap) ).astype('int')
        print( str( tilesX) + ":" + str(tilesY))
        FFTEven = np.zeros( box, dtype=fftw_dtype )
        FFTOdd = np.zeros( box, dtype=fftw_dtype )
        normConstBox = np.float32( 1.0 / FFTEven.size**2 )
        FRC2D = np.zeros( box, dtype=float_dtype )
        for I in xrange(0,tilesY):
            for J in xrange(0,tilesX):
                offset = np.array( [ I*box[0]*(1.0-overlap)+transLim[0], J*box[1]*(1.0-overlap)+transLim[2] ])
                
                tileEven = (hann*Image1[offset[0]:offset[0]+box[0], offset[1]:offset[1]+box[1] ]).astype(fftw_dtype)
                FFT2.update_arrays( tileEven, FFTEven ); FFT2.execute()
                tileOdd = (hann*Image2[offset[0]:offset[0]+box[0], offset[1]:offset[1]+box[1] ]).astype(fftw_dtype)
                FFT2.update_arrays( tileOdd, FFTOdd ); FFT2.execute()
    
                FFTOdd *= normConstBox
                FFTEven *= normConstBox
                
                # Calculate the normalized FRC in 2-dimensions
                FRC2D += ne.evaluate( "real(FFTEven*conj(FFTOdd)) / sqrt(real(abs(FFTOdd)**2) * real(abs(FFTEven)**2) )" )
                
        FRC2D = np.fft.fftshift( FRC2D )        
        rotFRC, _ = util.rotmean( FRC2D )
        self.FRC = rotFRC
        self.FRC2D = FRC2D


        
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
    

    
    def getSumCropToLimits( self, mode=None ):
        """
        Gets imageSum cropped so that no pixels with partial dose are kept.
        
        If mode = 'mc' it assumes that dosefgpu_driftcorr was used to do the alignment.
        """
        if mode == None:
            trans = self.translations
        elif mode == 'mc' :
            trans = -self.translations
            
        yendcrop = np.minimum( np.floor( trans[:,0].min() ), 0 )
        if yendcrop == 0:
            yendcrop = None
        xendcrop = np.minimum( np.floor( trans[:,1].min() ), 0 )
        if xendcrop == 0:
            xendcrop = None
        ystartcrop = np.maximum( np.ceil( trans[:,0].max() ), 0 )
        xstartcrop = np.maximum( np.ceil( trans[:,1].max() ), 0 )
        cropped_image = self.imageSum[ ystartcrop:yendcrop, xstartcrop:xendcrop ]
        return cropped_image
        
    def getImagesCropToLimits( self, mode=None ):
        """
        Gets images stack cropped so that no pixels with partial dose are kept.
        
        If mode = 'mc' it assumes that dosefgpu_driftcorr was used to do the alignment.
        """
        if mode == None:
            trans = self.translations
        elif mode == 'mc' :
            trans = -self.translations
            
        yendcrop = np.minimum( np.floor( trans[:,0].min() ), 0 )
        xendcrop = np.minimum( np.floor( trans[:,1].min() ), 0 )
        ystartcrop = np.maximum( np.ceil( trans[:,0].max() ), 0 )
        xstartcrop = np.maximum( np.ceil( trans[:,1].max() ), 0 )
        
        cropped_images = np.zeros( [ self.images.shape[0], 
                                    self.images.shape[1] - ystartcrop + yendcrop, 
                                    self.images.shape[2] - xstartcrop + xendcrop ])
                                    
        if xendcrop == 0:
            xendcrop = None
        if yendcrop == 0:
            yendcrop = None

        for I in np.arange(0,self.images.shape[0]):
            cropped_images[I,:,:] = self.images[ I, ystartcrop:yendcrop, xstartcrop:xendcrop ]
        return cropped_images
                    
#    def normalizeStack( self, mu = 0.0, sigma = 1.0 ):
#        """
#        Normalizes each image so that it has mean = mean and standard deviation = sigma.
#        This can help with round-off problems. Stores the means and stds so that 
#        they can be retrieved by the user later for quantitative analysis on the stack.
#        
#        At present the user is responsible for un-normalizing the stack after alignment,
#        if it is desired.
#        """
#        for J in np.arange( 0, self.images.shape[0] ):
#            self.imageMean[J] = np.mean( self.images[J,:,:] )
#            self.images[J,:,:] -= self.imageMean[J] + mu
#            self.imageStd[J] = np.std( self.images[J,:,:] )
#            self.images[J,:,:] *= sigma / self.imageStd[J]                            
    
    def execCTFFind4( self, movieMode=False, box_size = 512, contrast=0.1, 
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
        
        print( "Calling CTFFIND4 for " + self.stackName )
        print( "   written by Alexis Rohou: http://grigoriefflab.janelia.org/ctffind4" )
        print( "   http://biorxiv.org/content/early/2015/06/16/020917" )
        
        ps = self.pixelsize * 10.0
        min_res = np.min( [min_res, 50.0] )
        
        import os
        from os.path import splitext
        
        try: os.umask( 002 ) # Why is Python not using default umask from OS?
        except: pass
        
        if self.cachePath is None:
            self.cachePath = "."
            
        # Force trailing slashes onto cachePatch
        self.cachePath = os.path.join( self.cachePath, "" )
        
            
        stackFront = splitext( self.stackName )[0]
        diagOutName = os.path.join( self.cachePath, stackFront + ".ctf" )
         
        try: 
            mrcname = os.path.join( self.cachePath, stackFront + "_ctf4.mrc" )
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
        # subprocess.all( ctfexec )
        print( ctfexec )
        os.system( ctfexec )
        
        #print( "CTFFIND4 execution time (s): " + str(t1-t0))    
        try:
            logName = self.cachePath + stackFront + ".txt"
            print( "Trying to load from: " + logName )
            # Log has 5 comment lines, then 1 header, and
            # Micrograph number, DF1, DF2, Azimuth, Additional Phase shift, CC, and max spacing fit-to
            self.CTF4Results = np.loadtxt(logName, comments='#', skiprows=1 )
            
            self.CTF4Diag = ioMRC.MRCImport( diagOutName )
        
        except IOError:
            print( "CTFFIND4 likely core-dumped, try different input parameters?" )
        finally:
            # Write a RELION-style _ctffind3.log file, with 5 um pixel size...
            self.saveRelionCTF3( amp_contrast=contrast, dstep = 5.0 )
        pass
            
        # TODO: having trouble with files not being deletable, here.  Is CTFFIND4 holding them open?  Should 
        # I just pause for a short time?
        time.sleep(0.5) # DEBUG: try and see if temporary files are deletable now.
        try: os.remove( mrcname )
        except IOError: 
            print( "Could not remove temporary file: " + str(IOError.message) )
        try: os.remove( diagOutName )
        except: pass
        # Delete CTF4 logs
        try: os.remove( os.path.join( self.cachePath, stackFront + "_avrot.txt") )
        except: pass
        try: os.remove( os.path.join( self.cachePath, stackFront + ".txt") )
        except: pass
        try: os.remove( os.path.join( self.cachePath, stackFront + ".ctf" ) )
        except: pass
    
    def xcorr2_unblur( self, doseFilter = False, minShift = 2.0, terminationThres = 0.1, maxIteration=10, verbose=False   ):
        """
        Calls UnBlur by Grant and Rohou using the Zorro interface.
        """
        unblur_exename = "unblur_openmp_7_17_15.exe"
        if util.which( unblur_exename ) is None:
            print( "UnBlur not found in system path" )
            return
        
        print( "Calling UnBlur for " + self.stackName )
        print( "   written by Timothy Grant and Alexis Rohou: http://grigoriefflab.janelia.org/unblur" )
        print( "   http://grigoriefflab.janelia.org/node/4900" )
        
        import os
        from os.path import splitext
        
        try: os.umask( 002 ) # Why is Python not using default umask from OS?
        except: pass
        
        if self.cachePath is None:
            self.cachePath = "."
            
        # Force trailing slashes onto cachePatch
        self.cachePath = os.path.join( self.cachePath, "" )
        
        stackFront = splitext( self.stackName )[0]
        frcOutName = os.path.join( self.cachePath, stackFront + "_unblur_frc.txt" )
        shiftsOutName = os.path.join( self.cachePath, stackFront + "_unblur_shifts.txt" )
        outputAvName = os.path.join( self.cachePath, stackFront + "_unblur.mrc" )
        outputStackName = os.path.join( self.cachePath, stackFront + "_unblur_movie.mrc" )
        

        ps = self.pixelsize * 10.0
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
            mrcname = os.path.join( self.cachePath, stackFront + "_unblurIN.mrc" )
            ioMRC.MRCExport( self.images.astype('float32'), mrcname )
        except:
            print( "Error in exporting MRC file to UnBlur" )
            return
         
        # Are there flags for unblur?  Check the source code.
        flags = "" # Not using any flags
         
        unblurexec = ( unblur_exename + " " + flags + " << STOP_PARSING \n" + mrcname )
        
        unblurexec = (unblurexec + "\n" + str(self.images.shape[0]) + "\n" +
            outputAvName + "\n" + shiftsOutName + "\n" + str(ps) + "\n" +
            str(doseFilter) + "\n yes \n" + outputStackName + "\n yes \n" + 
            frcOutName + "\n" + str(minShift) + "\n" + str(outerShift) + "\n" +
            str(bfac) + "\n" + str( np.int(vertFouMaskHW) ) + "\n" + str( np.int(horzFouMaskHW) ) + "\n" +
            str(terminationThres) + "\n" + str(maxIteration) + "\n" + str(verbose) )
              
        unblurexec = unblurexec + "\nSTOP_PARSING"
        
        print( unblurexec )
        os.system( unblurexec )
        
        try:
            self.FRC = np.loadtxt(frcOutName, comments='#', skiprows=0 )
            self.translations = np.loadtxt( shiftsOutName, comments='#', skiprows=0 ).transpose()
            # UnBlur uses Fortran ordering, so we need to swap y and x for Zorro C-ordering
            self.translations = np.fliplr( self.translations )
            self.imageSum = ioMRC.MRCImport( outputAvName )
            self.images = ioMRC.MRCImport( outputStackName )
        except IOError:
            print( "UnBlur likely core-dumped, try different input parameters?" )
        finally:
            time.sleep(0.5) # DEBUG: try and see if temporary files are deleteable now.
        
            frcOutName = os.path.join( self.cachePath, stackFront + "_unblur_frc.txt" )
            shiftsOutName = os.path.join( self.cachePath, stackFront + "_unblur_shifts.txt" )
            outputAvName = os.path.join( self.cachePath, stackFront + "_unblur.mrc" )
            outputStackName = os.path.join( self.cachePath, stackFront + "_unblur_movie.mrc" )
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
        mag = dstep*1E3 / self.pixelsize
        
        stackFront = os.path.splitext( self.stackName )[0]
        logh = open( stackFront + "_zorro_ctffind3.log", "w" )
        
        logh.write( "CS[mm], HT[kV], AmpCnst, XMAG, DStep[um]\n" )
        logh.write( "%.2f"%self.C3 + " %.1f"%self.voltage + " " + 
            str(amp_contrast) + " %.1f" %mag + " %.2f"%dstep + "\n" )

        logh.write( "%.1f"%self.CTF4Results[1]+ " %.1f"%self.CTF4Results[2] 
            + " %.4f"%self.CTF4Results[3]+ " %.4f"%self.CTF4Results[4] + " Final Values\n ")
        logh.close()
        pass
    
    def loadData( self, stackNameIn = None, target="images", leading_zeros=0, useMemmap=False, endian='le' ):
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
        
        if stackNameIn is None:
            if self.stackName is None:
                print( "No stack name provided, cannot load image files" )
                return
            else:
                stackNameIn = self.stackName 
      
        #### DECOMPRESS FILE ####
        stackNameIn = util.decompressFile( stackNameIn )
        
        # Check for each if it's a sequence or not
        [file_front, file_ext] = splitext( stackNameIn )
        
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
            dm3struct = dm3.DM3( stackNameIn )
            tempData = np.empty( [ filecount, dm3struct.imagedata.shape[0], dm3struct.imagedata.shape[1]]  )
            tempData[0,:,:] = dm3struct.imagedata

            for I in np.arange( 1, filecount ):
                filenameDM3 = file_strip + str(file_nums[I]).zfill(leading_zeros) + self.file_ext
                print( "Importing: " + filenameDM3 )
                dm3struct = dm3.DM3( filenameDM3 )
                tempData[I,:,:] = dm3struct.imagedata
        elif file_ext == '.tif' or file_ext == '.tiff':
            print( "Loading TIFF files in sequence" )
            try:
                import skimage.io
                from glob import glob
            except:
                print( "Error: scikit-image or glob not found!" )
                return
             
            file_seq = self.file_front.rstrip( '1234567890' )
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
                
            mage1 = skimage.io.imread( stackNameIn )
            tempData = np.empty( [ filecount, mage1.shape[0], mage1.shape[1]]  )
            tempData[0,:,:] = mage1

            for I in np.arange( 1, filecount ):
                filenameTIFF = file_strip + str(file_nums[I]).zfill(leading_zeros) + self.file_ext
                print( "Importing: " + filenameTIFF )
                tempData[I,:,:] = skimage.io.imread( filenameTIFF )
        elif file_ext == ".dm4":
            # Expects a DM4 image stack
            dm4obj = ioDM.DM4Import( stackNameIn, verbose=False, useMemmap = useMemmap )
            tempData = np.copy( dm4obj.im[1].imageData, order='C' )
            # Load pixelsize from file
            try:
                self.pixelsize = dm4obj.im[1].imageInfo['DimXScale'] # DM uses units of nm
            except KeyError: pass
            try: 
                self.voltage = dm4obj.im[1].imageInfo['Voltage'] / 1000.0 # in kV
            except KeyError: pass
            try:
                self.C3 = dm4obj.im[1].imageInfo['Cs'] # in mm
            except KeyError: pass
            
        elif file_ext == ".mrc" or file_ext == '.mrcs':
            # Expects a MRC image stack
            tempData = ioMRC.MRCImport( stackNameIn, endian=endian )
        elif file_ext == ".hdf5" or file_ext == ".h5":
            # import tables
            try:
                h5file = tables.open_file( stackNameIn, mode='r' )
            except:
                print( "Could not open HDF5 file: " + stackNameIn )
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
        if target == "images":
            self.images = tempData
            if self.stackName is None:
                self.stackName = stackNameIn
        elif target == "imageSum":
            self.imageSum = tempData
        elif target == "C0":
            self.C0 = tempData
        elif target == "FFTSum":
            self.FFTSum = tempData
        elif target == "incohFouMag":
            self.incohFouMag = tempData
        elif target == "masks":
            self.masks = tempData
            
        self.t[7] = time.time() 
        
    def saveData( self, saveNameIn = None ):
        """
        Save files to disk.  
        
        Do compression of stack if requested, self.compression = '.bz2' for example
        uses lbzip2 or 7-zip. '.gz' is also supported by not recommended.
        
        TODO: add dtype options, including a sloppy float for uint16 and uint8
        """
        import os
        try: os.umask( 002 ) # Why is Python not using default umask from OS?
        except: pass

        
        if not bool(saveNameIn):
            if self.saveName is None:
                stack_front, _ = os.path.splitext( self.stackName )
                saveName = stack_front + "_zorro.mrc"
            else:
                saveName = self.saveName
        else:
            self.saveName = saveNameIn
            saveName = saveNameIn
        
        save_front, save_ext = os.path.splitext( saveName )
        #### SAVE ALIGNED SUM ####
        alignSumName = save_front + save_ext
        print( "Saving: " + alignSumName )
        ioMRC.MRCExport( self.imageSum.astype('float32'), alignSumName )

        # Compress sum
        if self.doCompression:
            util.compressFile( alignSumName, self.compress_ext )

        
        #### SAVE ALIGNED STACK ####
        if self.saveMovie:
            if save_ext == '.mrc':
                alignStackName = save_front + "_movie.mrcs"
            else:
                alignStackName = save_front + "_movie" + save_ext
            
            print( "Saving: " + alignStackName )
            ioMRC.MRCExport( self.images.astype('float32'), alignStackName ) 

            # Compress stack
            if self.doCompression:
                util.compressFile( alignStackName, self.compress_ext )

            
        #### SAVE CROSS-CORRELATIONS FOR FUTURE PROCESSING ####
        if self.saveC:
            xcorrName = save_front + "_xc" + save_ext
            
            print( "Saving: " + xcorrName )
            ioMRC.MRCExport( np.asarray( self.C0, dtype='float32'), xcorrName )
            if self.doCompression:
                util.compressFile( xcorrName, self.compress_ext )
            
        #### SAVE OTHER INFORMATION IN A LOG FILE ####
        # TODO: how to handle this exactly?  Just call saveConfig?
            
        
        #### DELETE OR COMPRESS ORIGINAL INPUT FILE ####
        # This should compress file series in TIFF or DM3 into a single archive
        #if bool(self.deleteRaw):
            # print( "deleteRaw has been disabled until it can be robustly tested" )
            # print( "Deleting: " + self.stackName )
            # os.remove( self.stackName )
        if self.doCompression:
            util.compressFile( self.stackName, self.compress_ext )
        pass
    


    def loadConfig( self, configNameIn = None, loadData=False ):
        """
        Initialize the ImageRegistrator class from a config file
        
        loadData = True will load data from the given filenames.
        """
        import ConfigParser
        import json
        
        if not bool(configNameIn):
            if not bool( self.configName ):
                pass # Do nothing
            else:
                print( "Cannot find configuration file: " + self.configName )
        else:
            self.configName = configNameIn

        print( "Loading config file: " + self.configName )
        config = ConfigParser.ConfigParser(allow_no_value = True)
        config.optionxform = str
        
        ##### Paths #####
        config.read( self.configName )
        # I'd prefer to pop an error here if configName doesn't exist
        
        
        
        # Initialization
        try: self.verbose = config.getint( 'initialization', 'verbose' )
        except: pass
        # TODO: wisdom_file currently has to be lower case
        try: self.wisdom_file = config.get( 'initialization', 'wisdom_file' )
        except: pass
        try: self.fftw_effort = config.get( 'initialization', 'fftw_effort' ).upper()
        except: pass
        try: self.n_threads = config.getint( 'initialization', 'n_threads' )
        except: pass
        try: self.saveC = config.getboolean( 'initialization', 'saveC' )
        except: pass
        try: self.saveCsub = config.getint( 'initialization', 'saveCsub' )
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

        errorDictsExist = True
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
                if 'key' in locals() and bool(key):
                    print( "DEBUG: Stop loading error dicts for key: " + key + ", errCnt = " + str(errCnt) )
                errorDictsExist=False
                break
            errCnt += 1
            
        
        # Registration parameters
        try: self.triMode = config.get('registration', 'triMode' )
        except: pass
        try: self.shapePadded = np.array( json.loads( config.get( 'registration', 'shapePadded' ) ) )
        except: pass
        try: self.shapeOriginal = np.array( json.loads( config.get( 'registration', 'shapeOriginal' ) ) )
        except: pass
        try: self.fouCrop = np.array( json.loads( config.get( 'registration', 'fouCrop' ) ) )
        except: pass
        try: self.subPixReg = config.getint('registration', 'subPixReg' )
        except: pass
        try: self.subPixMethod = config.get('registration', 'subPixMethod' )
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
#        try: self.rebinFact = config.getfloat('registration', 'rebinFact' )
#        except: pass
#        try: self.rebinMode = config.get('registration', 'rebinMode' )
#        except: pass
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
    
        # IO (will be over-written by command-line parameters)
        try: self.stackName = config.get('io', 'stackName' )
        except: pass
        # try: self.stack_ext = config.get('io', 'stack_ext' )
        # except: pass
        try: self.saveName = config.get('io', 'saveName' )
        except: pass
        # try: self.save_ext = config.get('io', 'save_ext' )
        # except: pass
        try: self.maskName = config.get('io', 'maskName' )
        except: pass
        try: self.savePDF = config.getboolean('io', 'savePDF' )
        except: pass
        try: self.savePNG = config.getboolean('io', 'savePNGs' )
        except: pass
        try: self.compress_ext = config.get('io', 'compress_ext' )
        except: pass
        try: self.saveMovie = config.getboolean( 'io', 'saveMovie' )
        except: pass
        try: self.doCompression = config.getboolean( 'io', 'doCompression' )
        except: pass
        # try: self.deleteRaw = config.getboolean('io', 'deleteraw' )
        # except: pass
        
        if bool(loadData) and self.stackName is not None:
            self.loadData()
        pass
    
    def saveConfig( self, configNameIn=None ):
        """
        Write the state of the ImageRegistrator class from a config file
        """
        import ConfigParser
        import json
        import os
        try: os.umask( 002 ) # Why is Python not using default umask from OS?
        except: pass        
        
        if not bool( configNameIn ):
            configName = self.configName
        else:
            configName = configNameIn
        
        # Write config
        config = ConfigParser.ConfigParser(allow_no_value = True)
        config.optionxform = str
        
        # Initialization
        config.add_section( 'initialization' )
        config.set( 'initialization', '# For detailed use instructions: github.com/C-CINA/zorro/wiki/Advice-for-Choice-of-Registration-Options', None )
        config.set( 'initialization', 'verbose', self.verbose )
        config.set( 'initialization', 'wisdom_file', self.wisdom_file )
        config.set( 'initialization', 'fftw_effort', self.fftw_effort )
        # Any time we cast variables we need handle errors from numpy
        config.set( 'initialization', '# n_threads is usually best if set to the number of physical cores (CPUs)' )
        try: config.set( 'initialization', 'n_threads', np.int(self.n_threads) )
        except: pass
        config.set( 'initialization', 'saveC', self.saveC )
        config.set( 'initialization', 'saveCsub', self.saveCsub )
        
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
            
        if self.fouCrop is not None:
            if type(self.fouCrop) == type(np.array(1)):
                self.fouCrop = self.fouCrop.tolist()
            config.set( 'registration', 'fouCrop', json.dumps( self.fouCrop ) )
        
        try: config.set( 'registration', 'subPixReg', np.int(self.subPixReg) )
        except: pass
        config.set( 'registration', 'subPixMethod', self.subPixMethod )
        config.set( 'registration' , "# Maximum shift in pixels within diagWidth/autoMax frames" )
        try: config.set( 'registration', 'maxShift', np.int(self.maxShift) )
        except: pass
        config.set( 'registration' , "# preShift = True is useful for crystalline specimens where you want maxShift to follow the previous frame position" )
        config.set( 'registration', 'preShift', self.preShift )
        
        try: config.set( 'registration', 'diagWidth', np.int(self.diagWidth) )
        except: pass
        try: config.set( 'registration', 'autoMax', np.int(self.autoMax) )
        except: pass
        
        config.set( 'registration' , "# peakSigThres changes with dose but usually is uniform for a dataset" )
        config.set( 'registration', 'peaksigThres', self.peaksigThres )
        config.set( 'registration' , "# pixErrThres and sigmaThres through out equations which don't fix the solution" )
        config.set( 'registration', 'pixErrThres', self.pixErrThres )
        config.set( 'registration', 'sigmaThres', self.sigmaThres )
        config.set( 'registration' , "# corrThres is deprecated" )
        config.set( 'registration', 'corrThres', self.corrThres )
        config.set( 'registration', 'velocityThres', self.velocityThres )
        config.set( 'registration' , "# Brad is radius of B-filter in Fourier pixels" )
        config.set( 'registration', 'Brad', self.Brad )
        config.set( 'registration' , "# Bmode = conv, opti, or fourier" )
        config.set( 'registration', 'Bmode', self.Bmode )
        config.set( 'registration', 'BFiltType', self.BfiltType )
#        config.set( 'registration' , "# rebin not recommended, use rebinFact = 2 or 4, rebinMode can be fourier" )
#        config.set( 'registration', 'rebinFact', self.rebinFact )
#        config.set( 'registration', 'rebinMode', self.rebinMode )
        config.set( 'registration' , "# originMode is centroid, or (empty), empty sets frame 1 to (0,0)" )
        config.set( 'registration', 'originMode', self.originMode )
        config.set( 'registration' , "# weightMode is one of logistic, corr, norm, unweighted" )
        config.set( 'registration', 'weightMode', self.weightMode )
        config.set( 'registration', 'logisticK', self.logisticK )
        config.set( 'registration', 'logisticNu', self.logisticNu )
        config.set( 'registration' , "# Set suppressOrigin = True if gain reference artifacts are excessive" )
        config.set( 'registration', 'suppressOrigin', self.suppressOrigin )
        
        # IO
        config.add_section('io')
        config.set( 'io', 'stackName', self.stackName )
        config.set( 'io', 'saveName', self.saveName )
        config.set( 'io', 'maskName', self.maskName )
        config.set( 'io', 'savePDF', self.savePDF )
        config.set( 'io', 'savePNGs', self.savePNG )
        # config.set( 'io', '# savename extension overrides save_ext ' )
        # config.set( 'io', 'save_ext', self.save_ext )
        config.set( 'io', 'compress_ext', self.compress_ext )
        config.set( 'io', 'saveMovie', self.saveMovie )
        config.set( 'io', 'doCompression', self.doCompression )
        # config.set( 'io', 'deleteraw', self.deleteRaw )
        
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
                config.set( dictName, key, json.dumps( errorDict[key].tolist() ) )
        
        try:
            # Would be nice to have some error handling if cfgFH already exists
            # Could try and open it with a try: open( 'r' )
            cfgFH = open( configName , 'w+' )
            print( "Saving config file: " + configName )
            config.write( cfgFH )
            cfgFH.close()
        except:
            print( "Error in loading config file: " + configName )

    def plot( self, errIndex = -1,  title=None,
                plotDict = {},
                imFilt = scipy.ndimage.gaussian_filter, imFiltArg = 1.0, 
                interpolation='nearest', dpi=300,
                graph_cm = 'gnuplot', image_cm = 'gray',
                toPDF = False, PDFHandle = None, toPNG = False ):
        """
        Plots a report of all relevant parameters for the given errorDictionary index 
        (the default of -1 is the last-most error dictionary in the list).
        
        Which plots are generated is controlled by plotDict.  Build one with zorro.buildPlotDict
        
        Can also produce a PDF report, with the given filename, if toPDF = True.  
        More common is to produce individual PNGs.
        """
        import os
        try: os.umask( 002 ) # Why is Python not using default umask from OS?
        except: pass
    
        if not plotDict:
            plotDict = buildPlotDict()
            
    
        try: # For Linux, use FreeSerif
            plt.rc('font', family='FreeSerif', size=16)
        except:
            try: 
                plt.rc( 'font', family='serif', size=16)
            except: pass
        
        if toPDF:
            from matplotlib.backends.backend_pdf import PdfPages
            # PDFHandle should be a PdfPages object
            if PDFHandle is not None:
                pp = PDFHandle
            else:
                pp = PdfPages( self.stackName + ".pdf" )
                
        if title is None:
            title = self.stackName
            
        
        if plotDict["imageSum"] and self.imageSum is not None:
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
            plt.tight_layout()
            if toPNG: plt.savefig( title + "_imageSum.png" , dpi=dpi )
            if toPDF: pp.savefig(dpi=dpi)
            if toPNG or toPDF: plt.close(fig1)
        if plotDict["FFTSum"] and self.imageSum is not None:
            # Rather than use pyFFTW here I'll just use numpy fft
            self.FFTSum = np.abs( np.fft.fftshift( np.fft.fft2(self.imageSum) ) )
            
            figFFT = plt.figure()
            if self.pixelsize is None:
                plt.imshow( np.log10(self.FFTSum + 1.0), interpolation=interpolation )
            else:
                pixelsize_inv = 1.0 / (self.images.shape[-1] * self.pixelsize)
                pltmage = plt.imshow( np.log10(self.FFTSum + 1.0),  
                            interpolation=interpolation  )
                util.plotScalebar( pltmage, pixelsize_inv, units="nm^{-1}" ) 
            plt.set_cmap( image_cm )
            plt.title( "FFT: " + title )
            plt.axis('off')
            plt.tight_layout()
            if toPNG: plt.savefig( title + "_FFTSum.png" , dpi=dpi )
            if toPDF: pp.savefig(dpi=dpi)
            if toPNG or toPDF: plt.close(figFFT)
            pass
        if plotDict["polarFFTSum"] and self.imageSum is not None:
            if self.FFTSum is None:
                self.FFTSum = np.abs( np.fft.fftshift( np.fft.fft2(self.imageSum) ) )
            polarFFTSum = util.img2polar( self.FFTSum )
            
            figPolarFFT = plt.figure()
            if self.pixelsize is None:
                plt.imshow( np.log10(polarFFTSum + 1.0),  interpolation=interpolation )
            else:
                pixelsize_inv = 1.0 / (self.images.shape[-1] * self.pixelsize)
                pltmage = plt.imshow( np.log10(polarFFTSum + 1.0), 
                            interpolation=interpolation  )
                util.plotScalebar( pltmage, pixelsize_inv, units="nm^{-1}" ) 
            plt.set_cmap( image_cm )
            plt.title( "Polar FFT: " + title )
            plt.axis('off')
            plt.tight_layout()
            if toPNG: plt.savefig( title + "_polarFFTSum.png" , dpi=dpi )
            if toPDF: pp.savefig(dpi=dpi)
            if toPNG or toPDF: plt.close(figPolarFFT)
            pass
        if plotDict["imageFirst"] and self.images is not None:     
            firstmage = self.images[0,:,:]
            dose_mean = np.mean(firstmage)
            dose_std = np.std(firstmage)
#            if imFilt is not None:
#                firstmage = imFilt( firstmage, imFiltArg )
                
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
            plt.tight_layout()
            if toPNG: plt.savefig( title + "_imageFirst.png" , dpi=dpi )
            if toPDF: pp.savefig(dpi=400)
            if toPNG or toPDF: plt.close(fig2)
        if plotDict["corrTriMat"]:
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
                plt.tight_layout()
                if toPNG: plt.savefig( title + "_corrTriMat.png" , dpi=dpi )
                if toPDF: pp.savefig()
                if toPNG or toPDF: plt.close(fig3)
            except:
                plt.close(fig3)
        if plotDict["peaksigTriMat"]:
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
                plt.tight_layout()
                if toPNG: plt.savefig( title + "_peaksigTriMat.png" , dpi=dpi )
                if toPDF: pp.savefig()
                if toPNG or toPDF: plt.close(fig3B)
            except:
                plt.close(fig3B)
        if plotDict["shiftsTriMat"]:  
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
                plt.tight_layout()
                if toPNG: plt.savefig( title + "_shiftsTriMat.png" , dpi=dpi )
                if toPDF: pp.savefig()
                if toPNG or toPDF: plt.close(fig4)
            except:
                plt.close(fig4)
        if plotDict["errorTriMat"]:
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
                plt.tight_layout()
                if toPNG: plt.savefig( title + "_pixRegError.png" , dpi=dpi )
                if toPDF: pp.savefig()
                if toPNG or toPDF: plt.close(fig5)
            except:
                plt.close(fig5)
        
        if plotDict["translations"] and self.translations is not None:
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
            plt.tight_layout()
            if toPNG: plt.savefig( title + "_translations.png" , dpi=dpi )
            if toPDF: pp.savefig()
            if toPNG or toPDF: plt.close(fig6)
        if plotDict["pixRegError"]:
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
    #            if self.sigmaThres is not None and self.pixErrThres is not None:
    #                plt.legend( ("Std threshold", "Abs Threshold") )
    #            elif self.sigmaThres is not None:
    #                plt.legend( ("Std Threshold",) )
    #            elif self.pixErrThres is not None:
    #                plt.legend( ("Abs Threshold",) )
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
    #            if self.sigmaThres is not None and self.pixErrThres is not None:
    #                plt.legend( ("Std threshold", "Abs Threshold") )
    #            elif self.sigmaThres is not None:
    #                plt.legend( ("Std Threshold",) )
    #            elif self.pixErrThres is not None:
    #                plt.legend( ("Abs Threshold",) )
                plt.semilogy( np.arange(0,M), errorXY[:,1], 'ok:' )
                plt.xlabel( 'Equation number' )
                plt.ylabel( 'Drift error estimate (pix) ' )
                plt.title( "RMS X-error estimate: %.2f"%meanErrorX + " +/- %.2f"%stdErrorX  + " pixels" )
                plt.set_cmap( graph_cm )
                
                plt.tight_layout()
                if toPNG: plt.savefig( title + "_errorPlotsThres.png" , dpi=dpi )
                if toPDF: pp.savefig()
                if toPNG or toPDF: plt.close(fig7)
            except:
                plt.close(fig7)
            
        if plotDict["CTF4Diag"]:    
            try: # Sav
                fig8 = plt.figure()
                plt.imshow( self.CTF4Diag )
                plt.set_cmap( image_cm )
                plt.title(r"CTFFIND4: " + title +"\n $DF_1 = %.f"%self.CTF4Results[1] 
                    + " \AA,  DF_2 = %.f"%self.CTF4Results[2] 
                    +" \AA,  {\gamma} = %.2f"%self.CTF4Results[3] 
                    + " ^{\circ}, R = %.4f"%self.CTF4Results[4] + "$"  )
                
                plt.tight_layout()
                if toPNG: plt.savefig( title + "_CTFFIND4.png" , dpi=dpi )
                if toPDF: pp.savefig()
                if toPNG or toPDF: plt.close(fig8)
            except:
                plt.close(fig8)
        if plotDict["logisticsCurve"]:    
            try: # peak sig versus pixel error for logistics weighting
                (fig9, ax1) = plt.subplots()
                errorXY = self.errorDictList[0]["errorXY"]
                pixError = np.sqrt( errorXY[:,0]**2 + errorXY[:,1]**2 )
                peaksigVect = self.errorDictList[0]["peaksigTriMat"][ self.errorDictList[0]["corrTriMat"] > 0.0  ]
                
                plt.semilogy( peaksigVect, pixError, 'k.' )
                plt.plot( [self.peaksigThres, self.peaksigThres], [np.min(pixError), np.max(pixError)], 'r--' )
                ax1.set_xlabel( 'Correlation peak significance, $\sigma$' )
                ax1.set_ylabel( 'Pixel registration error' )
    
                if self.weightMode == 'logistic':
                    peakSig = np.arange( np.min(peaksigVect), np.max(peaksigVect), 0.05 )
                    logWeights = 1.0 - 1.0 / (1.0 + np.exp( -self.logisticK*(-peakSig + self.peaksigThres) ) )**self.logisticNu 
                    plt.plot( peakSig, logWeights )
                    plt.legend( ("Correlations", "Threshold", "Weighting") )
                else:
                    plt.legend( ("Correlations", "Threshold") )
                plt.ylim( [1E-2, 1E2] )
                
                if toPNG: plt.savefig( title + "_PeaksigVsPixError.png" , dpi=dpi )
                if toPDF: pp.savefig()
                if toPNG or toPDF: plt.close(fig8)
            except:
                plt.close(fig9)

        # Show all plots, not needed in Qt matplotlib backends    
        plt.show( block = False )
        
        if toPDF: 
            if PDFHandle is None: pp.close()
            plt.close("all")
        elif toPNG:
            plt.close("all")
        
    def makeMovie( self, movieName = None, clim = None, frameRate=3, dpi=250, graph_cm = 'gnuplot' ):
        """
        Use FFMPEG to generate movies showing the correlations.  C0 must not be None.
        
        The ffmpeg executable must be in the system path.
        """
        import os

        fex = '.png'
        print( "makeMovie must be able to find FFMPEG on the system path" )
        print( "Strongly recommended to use .mp4 extension" )
        if movieName is None:
            movieName = self.saveName + ".mp4"
        
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
            plt.savefig( "corrMap_%05d"%J + fex, dpi=dpi )
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
        os.system( comstring )
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
        print( "###############################" )
        print( "    Total execution time (s): %.3f"%(self.t[10] - self.t[0]) )
        pass

def buildPlotDict(): # Notice this is a static function
    plotDict = {}
    plotDict["imageSum"] = True
    plotDict["imageFirst"] = True
    plotDict["FFTSum"] = True
    plotDict["polarFFTSum"] = False
    plotDict["corrTriMat"] = False
    plotDict["shiftsTriMat"] = False
    plotDict["peaksigTriMat"] = True
    plotDict["errorTriMat"] = True
    plotDict["translations"] = True
    plotDict["pixRegError"] = True
    plotDict["CTF4Diag"] = True
    plotDict["logisticsCurve"] = True
    return plotDict
        
#### COMMAND-LINE INTERFACE ####
if __name__ == '__main__':
    # Get command line arguments
    import sys
    # Usage: 
    # /mnt/ssd/anaconda/bin/python zorro.py -i Test.dm4 -c default.ini -o test.mrc
    stackReg = ImageRegistrator()
    configFile = None
    inputFile = None
    outputFile = None
    
    for J in np.arange(0,len(sys.argv)):
        # First argument is mnxc_solver.py
        # Then we expect flag pairs
        # sys.argv is a Python list
        if sys.argv[J] == '-c':
            configFile = sys.argv[J+1]
            J += 1
        elif sys.argv[J] == '-i':
            inputFile = sys.argv[J+1]
            stackReg.loadData()
            J += 1
        elif sys.argv[J] == '-o':
            outputFile = sys.argv[J+1]
            J += 1
            pass
    
    if inputFile is None and configFile is None:
        print( "No input files, outputing template.cfg" )
        stackReg.saveConfig( configNameIn = "template.cfg")
        sys.exit()
    if inputFile is None and configFile is not None:
        stackReg.loadConfig( configNameIn=configFile, loadData = True )
    if inputFile is not None and configFile is not None:
        stackReg.loadConfig( configNameIn=configFile, loadData = False )
        stackReg.stackName = inputFile
        stackReg.loadData()
        
    if outputFile is not None:
        stackReg.saveName = outputFile    
    elif stackReg.saveName is None: # Default output behaviour
        pass
        
    saveFront, _ = os.path.splitext( stackReg.saveName )
    # Execute the alignment
    # TODO: test to see if translations already exists
    stackReg.alignImageStack()
    # Save PDF report and/or individual PNGs if desired
    if stackReg.savePDF or stackReg.savePNGs:
        plt.switch_backend( u'agg' )
        stackReg.plot( toPDF=stackReg.savePDF, toPNG=stackReg.savePNGs )
    # Save everthing and do rounding/compression operations
    stackReg.saveConfig( saveFront + ".log" )
    stackReg.saveData()   
    sys.exit()

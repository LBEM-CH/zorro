# -*- coding: utf-8 -*-
"""
MNXC utility (static-)functions 
Created on Sun Jul 05 11:26:39 2015
@author: Robert A. McLeod

This is for general 'helper' functions that don't fit well into the MNXC object.
"""
import numpy as np
import math
import scipy.ndimage
import matplotlib.pyplot as plt
# from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import mpl_toolkits.axes_grid1.anchored_artists
import os, os.path
import subprocess
from multiprocessing.pool import ThreadPool

#### STATIC HANDLE ####
# Not sure if this is used here, useful for storing data in the function as if 
# it is a Python object, for repeated operations
def static_var(varname, value):
    def decorate(func):
        setattr(func, varname, value)
        return func
    return decorate
    

############### IMAGE UTILITY FUNCTIONS ###############

def ravel_trimat( trimat ):
    """
    Return's a 1-D representation of non-zero elements of trimat.
    
    raveled, unravelIndices = ravel_trimat( triangularMatrix )
    
    The unravelIndices is necessary to unravel the vector back into matrix form.
    """
    [M,N] = trimat.shape
    triIndices = trimat.astype('bool')
    vectorIndices = np.arange(0,triIndices.size)[np.ravel( triIndices )]
    unravelIndices = np.unravel_index( vectorIndices, [M,N] )
    raveled = np.ravel( trimat[triIndices] )
    return raveled, unravelIndices
    
def unravel_trimat( raveled, unravelIndices, shape=None ):
    """
    Undo's a 
    
    Note: if shape = None, M,N are taken to be the maximum values in the matrix, so if there's zero columns 
    on the right, or zero rows on the bottom, they will be cropped in the returned triangular matrix.
    """
    if shape == None:
        M = np.max( unravelIndices[0] )
        N = np.max( unravelIndices[1] )
    else:
        M = shape[0]; N = shape[1]
    unraveled = np.zeros( [M,N] )
    unraveled[unravelIndices[0], unravelIndices[1]] = raveled
    return unraveled

def apodization( name = 'butter.32', shape= [2048,2048], radius=None ):
    """ apodization( name = 'butter.32', size = [2048,2048], radius=None )
    Provides a 2-D filter or apodization window for Fourier filtering or image clamping.
        Radius = None defaults to shape/2
    
    Valid names are: 
        'hann' - von Hann cosine window on radius
        'hann_square' as above but on X-Y
        'hamming' - good for apodization, nonsense as a filter
        'butter.X' Butterworth multi-order filter where X is the order of the Lorentzian
        'butter_square.X' Butterworth in X-Y
        'gauss_trunc' - truncated gaussian, higher performance (smaller PSF) than hann filter
        'gauss' - regular gaussian
    NOTE: There are windows in scipy.signal for 1D-filtering...
    WARNING: doesn't work properly for odd image dimensions
    """
    # Make meshes
    shape = np.asarray( shape )
    if radius is None:
        radius = shape/2.0
    else:
        radius = np.asarray( radius, dtype='float' )
    # DEBUG: Doesn't work right for odd numbers
    [xmesh,ymesh] = np.meshgrid( np.arange(-shape[1]/2,shape[1]/2), np.arange(-shape[0]/2,shape[0]/2) )
    r2mesh = xmesh*xmesh/( np.double(radius[0])**2 ) + ymesh*ymesh/( np.double(radius[1])**2 )
    
    try:
        [name, order] = name.lower().split('.')
        order = np.double(order)
    except ValueError:
        order = 1
        
    if name == 'butter':
        window =  np.sqrt( 1.0 / (1.0 + r2mesh**order ) )
    elif name == 'butter_square':
        window = np.sqrt( 1.0 / (1.0 + (xmesh/radius[1])**order))*np.sqrt(1.0 / (1.0 + (ymesh/radius[0])**order) )
    elif name == 'hann':
        cropwin = ((xmesh/radius[1])**2.0 + (ymesh/radius[0])**2.0) <= 1.0
        window = cropwin.astype('float') * 0.5 * ( 1.0 + np.cos( 1.0*np.pi*np.sqrt( (xmesh/radius[1])**2.0 + (ymesh/radius[0])**2.0  )  ) )
    elif name == 'hann_square':
        window = ( (0.5 + 0.5*np.cos( np.pi*( xmesh/radius[1]) ) ) *
            (0.5 + 0.5*np.cos( np.pi*( ymesh/radius[0] )  ) ) )
    elif name == 'hamming':
        cropwin = ((xmesh/radius[1])**2.0 + (ymesh/radius[0])**2.0) <= 1.0
        window = cropwin.astype('float') *  ( 0.54 + 0.46*np.cos( 1.0*np.pi*np.sqrt( (xmesh/radius[1])**2.0 + (ymesh/radius[0])**2.0  )  ) )
    elif name == 'hamming_square':
        window = ( (0.54 + 0.46*np.cos( np.pi*( xmesh/radius[1]) ) ) *
            (0.54 + 0.46*np.cos( np.pi*( ymesh/radius[0] )  ) ) )
    elif name == 'gauss' or name == 'gaussian':
        window = np.exp( -(xmesh/radius[1])**2.0 - (ymesh/radius[0])**2.0 )
    elif name == 'gauss_trunc':
        cropwin = ((0.5*xmesh/radius[1])**2.0 + (0.5*ymesh/radius[0])**2.0) <= 1.0
        window = cropwin.astype('float') * np.exp( -(xmesh/radius[1])**2.0 - (ymesh/radius[0])**2.0 )
    elif name == 'lanczos':
        print( "TODO: Implement Lanczos window" )
        return
    else:
        print( "Error: unknown filter name passed into apodization" )
        return
    return window
    
def edge_mask( maskShape=[2048,2048], edges=[64,64,64,64] ):
    """
    Generate a mask with edges removed to [y1,y2,x1,x2]
    """
    edgemask = np.ones( maskShape )    
    [xmesh,ymesh] = np.meshgrid( np.arange(0,maskShape[1]), np.arange(0,maskShape[0]) )
    edgemask *= xmesh >= edges[2]
    edgemask *= xmesh <= maskShape[1] - edges[3]
    edgemask *= ymesh >= edges[0] 
    edgemask *= ymesh <= maskShape[0] - edges[1]
    edgemask = np.reshape( edgemask, [1, edgemask.shape[0], edgemask.shape[1]]).astype( 'bool' )
    return edgemask

@static_var( "rfloor", None )
@static_var( "rceil", None )
@static_var( "rmax", 0 )
@static_var( "remain", 0 )
@static_var( "remain_n", 0 )
@static_var( "weights", 0 )
@static_var( "raxis", 0 )
@static_var( "prevN", 0 )
@static_var( "prevM", 0 )
@static_var( "weights", 0 )
@static_var( "raxis", 0 )
def rotmean( mage ):
    """
    Computes the rotational mean about the center of the image.  Generally used 
    on the magnitude of Fourier transforms. Uses static variables that accelerates 
    the precomputation of the meshes if you call it repeatedly on the same 
    dimension arrays. 
    
    NOTE: returns both rmean, raxis so you must handle the raxis part.
    
    Mage should be a power of two.  If it's not, it's padded automatically
    """
    if np.mod( mage.shape[1],2 ) == 1 and np.mod( mage.shape[0],2) == 1:
        mage = np.pad( mage, ((0,1),(0,1)), 'edge' )
    elif np.mod( mage.shape[1],2 ) == 1:
        mage = np.pad( mage, ((0,0),(0,1)), 'edge' )
    elif np.mod( mage.shape[0],2 ) == 1:
        mage = np.pad( mage, ((0,1),(0,0)), 'edge' )    
        
    N = int( np.floor( mage.shape[1]/2.0 ) )
    M = int( np.floor( mage.shape[0]/2.0 ) )
    
    if N != rotmean.prevN or M != rotmean.prevM:
        # Initialize everything
        rotmean.prevN = N
        rotmean.prevM = M
        
        rotmean.rmax = np.int( np.ceil( np.sqrt( N**2 + M**2 ) ) + 1 )
        [xmesh, ymesh] = np.meshgrid( np.arange(-N, N), np.arange(-M, M) )
        rmesh = np.sqrt( xmesh**2 + ymesh**2 )
        rotmean.rfloor = np.floor( rmesh )
        
        rotmean.remain = rmesh - rotmean.rfloor
        # Make rfloor into an index look-up table
        rotmean.rfloor = rotmean.rfloor.ravel().astype('int')
        rotmean.rceil = (rotmean.rfloor+1).astype('int')
        
        # Ravel
        rotmean.remain = rotmean.remain.ravel()
        rotmean.remain_n = 1.0 - rotmean.remain
        
#        rotmean.weights = np.zeros( [rotmean.rmax] )
#        weights_n = np.zeros( [rotmean.rmax] )
#        
#        weights_n[rotmean.rfloor] += rotmean.remain_n
#        rotmean.weights[ (rotmean.rfloor+1) ] = rotmean.remain
#        rotmean.weights += weights_n
        
        rotmean.weights = np.bincount( rotmean.rceil, rotmean.remain ) + np.bincount( rotmean.rfloor, rotmean.remain_n, minlength=rotmean.rmax  )
        rotmean.raxis = np.arange(0,rotmean.weights.size)
    else:
        # Same size image as previous time
        # Excellent now only 150 ms in here for 2k x 2k...
        # Rotmean_old was 430 ms on the desktop
        pass
    
    # I can flatten remain and mage
    mage = mage.ravel()
    mage_p = mage * rotmean.remain
    mage_n = mage * rotmean.remain_n

    # rmean = np.zeros( np.size(rotmean.weights) )
    # rmean_n = np.zeros( np.size(rotmean.weights) )

    # Find lower ("negative") remainders
    #rmean_n = np.bincount( rotmean.rfloor, mage_n  )
    #rmean_n[rotmean.rfloor] += mage_n
    
    # Add one to indexing array and add positive remainders to next-neighbours in sum
    #rmean[ (rotmean.rfloor+1) ] += mage_p
    
    rmean = np.bincount( rotmean.rceil, mage_p ) + np.bincount( rotmean.rfloor, mage_n, minlength=rotmean.rmax  )
    
    # sum
    # rmean += rmean_n
    # and normalize sum to average
    rmean /= rotmean.weights
    
    return [rmean, rotmean.raxis]
    
def normalize(a):
    """ Normalizes the input to the range [0.0,1.0].
    
    Returns floating point if integer data is passed in."""
    if np.issubdtype( a.dtype, np.integer ):
        a = a.astype( 'float' )
    amin = a.min()
    arange = (a.max() - amin)
    a -= amin
    a /= arange
    return a    
    
def imageShiftAndCrop( mage, shiftby ):
    """ imageShiftAndCrop( mage, shiftby )
    This is a relative shift, integer pixel only, pads with zeros to cropped edges
    
    mage = input image
    shiftby = [y,x] pixel shifts    
    """
    
    # Actually best approach is probably to roll and then zero out the parts we don't want
    # The pad function is expensive in comparison

    shiftby = np.array( shiftby, dtype='int' )
    # Shift X
    if(shiftby[1] < 0 ):
        mage = np.roll( mage, shiftby[1], axis=1 )
        mage[:, shiftby[1]+mage.shape[1]:] = 0.0
    elif shiftby[1] == 0:
        pass
    else: # positive shift
        mage = np.roll( mage, shiftby[1], axis=1 )
        mage[:, :shiftby[1]] = 0.0
    # Shift Y
    if( shiftby[0] < 0 ):
        mage = np.roll( mage, shiftby[0], axis=0 )
        mage[shiftby[0]+mage.shape[0]:,:] = 0.0
    elif shiftby[0] == 0:
        pass
    else:  # positive shift
        mage = np.roll( mage, shiftby[0], axis=0 )
        mage[:shiftby[0],:] = 0.0
    return mage

# Incorporate some static vars for the meshes?
# It's fairly trivial compared to the convolve cost, but if we moved the subPixShift
# outside it's possible.
# Best performance improvement would likely be to put it as a member function in
# ImageRegistrator so that it can work on data in-place.
def lanczosSubPixShift( imageIn, subPixShift, kernelShape=3, lobes=None ):
    """ lanczosSubPixShift( imageIn, subPixShift, kernelShape=3, lobes=None )
        imageIn = input 2D numpy array
        subPixShift = [y,x] shift, recommened not to exceed 1.0, should be float
        
    Random values of kernelShape and lobes gives poor performance.  Generally the 
    lobes has to increase with the kernelShape or you'll get a lowpass filter.
    
    Generally lobes = (kernelShape+1)/2 
    
    kernelShape=3 and lobes=2 is a lanczos2 kernel, it has almost no-lowpass character
    kernelShape=5 and lobes=3 is a lanczos3 kernel, it's the typical choice
    Anything with lobes=1 is a low-pass filter, but next to no ringing artifacts
    """
    
    lanczos_filt = lanczosSubPixKernel( subPixShift, kernelShape=kernelShape, lobes=lobes )
    
    # Accelerate this with a threadPool
    imageOut = scipy.ndimage.convolve( imageIn, lanczos_filt, mode='reflect' )
    return imageOut
    
def lanczosSubPixKernel( subPixShift, kernelShape=3, lobes=None  ):
    """
    Generate a kernel suitable for ni.convolve to subpixally shift an image.
    """
    kernelShape = np.array( [kernelShape], dtype='int' )
    if kernelShape.ndim == 1: # make it 2-D
        kernelShape = np.array( [kernelShape[0], kernelShape[0]], dtype='int' )
        
    if lobes is None:
        lobes = (kernelShape[0]+1)/2
    
    x_range = np.arange(-kernelShape[1]/2,kernelShape[1]/2)+1.0-subPixShift[1]
    x_range = ( 2.0 / kernelShape[1] ) * x_range 
    y_range = np.arange(-kernelShape[1]/2,kernelShape[0]/2)+1.0-subPixShift[0]
    y_range = ( 2.0 /kernelShape[0] ) * y_range
    [xmesh,ymesh] = np.meshgrid( x_range, y_range )
    
    
    lanczos_filt = np.sinc(xmesh * lobes) * np.sinc(xmesh) * np.sinc(ymesh * lobes) * np.sinc(ymesh)
    
    lanczos_filt = lanczos_filt / np.sum(lanczos_filt) # Normalize filter output
    return lanczos_filt
    
def lanczosSubPixShiftStack( imageStack, translations, n_threads=16 ):
    """
    Does subpixel translations shifts for a stack of images using a ThreadPool to distribute the load.
    
    I could make this a general function utility by passing in the function handle.  
    """
    tPool = ThreadPool( n_threads )
    if imageStack.ndim != 3:
        raise ValueError( "lanczosSubPixShiftStack() only works on image stacks with Z-axis as the zero dimension" )        
        
    slices = imageStack.shape[0]
    # Build parameters list for the threaded processeses, consisting of index
    tArgs = [None] * slices
    for J in np.arange(slices):
        tArgs[J] = (J, imageStack, translations)
    
    # All operations are done 'in-place' 
    tPool.map( lanczosIndexedShift, tArgs )
    tPool.close()
    tPool.join()
    

def lanczosIndexedShift( params ):
    """ lanczosIndexedShift( params )
        params = (index, imageStack, translations, kernelShape=3, lobes=None)
        imageStack = input 3D numpy array
        translations = [y,x] shift, recommened not to exceed 1.0, should be float
        
    Random values of kernelShape and lobes gives poor performance.  Generally the 
    lobes has to increase with the kernelShape or you'll get a lowpass filter.
    
    Generally lobes = (kernelShape+1)/2 
    
    kernelShape=3 and lobes=2 is a lanczos2 kernel, it has almost no-lowpass character
    kernelShape=5 and lobes=3 is a lanczos3 kernel, it's the typical choice
    Anything with lobes=1 is a low-pass filter, but next to no ringing artifacts
    
    If you cheat and pass in rounded shifts only the roll will be performed, so this can be used to accelerate
    roll as well in a parallel environment.
    """
    if len( params ) == 3:
        [index, imageStack, translations] = params
        kernelShape = 3 
        lobes = None
    elif len( params ) == 4:
        [index, imageStack, translations, kernelShape] = params
        lobes = None
    elif len( params ) == 5:
        [index, imageStack, translations, kernelShape, lobes] = params

    integer_trans = np.round( translations[index,:] ).astype('int')
    # Integer shift
    imageStack[index,:,:] = np.roll( np.roll( imageStack[index,:,:], 
            integer_trans[0], axis=0 ), 
            integer_trans[1], axis=1 )
    # Subpixel shift
    remain_trans = np.remainder( translations[index,:], 1)
    if not (np.isclose( remain_trans[0], 0.0) and np.isclose( remain_trans[1], 0.0) ):
        kernel = lanczosSubPixKernel( remain_trans, kernelShape=kernelShape, lobes=lobes  )
        # RAM: I tried to use the out= keyword but it's perhaps not thread-safe.
        imageStack[index,:,:] =  scipy.ndimage.convolve( imageStack[index,:,:], kernel, mode='reflect' )   

    
    
def img2polar(img, center=None, final_radius=None, initial_radius = None, phase_width = None, mode='linear',  interpolate='bilinear'):
    """ Convert a Cartesian image into polar coordinates.
    
    Center is where to rotate about, typically [M/2, N/2]
    final_radius is the maximum r value to interpolate out too, typically N/2
    initial radius is where to start (can chop off the center if desired)
    phase_width is the pixel count in angle (x-axis)
    mode is whether to operate on the log of the radius or not
        'linear' = linear radius
        'log' log(radius)
    interpolate is the interpolation method, 
        'nn' = nearest neighbour ( should be smoothed afterward)
        'bilinear' = bilinear (recommended)
        
    Can only pass in 2-D images at present.
    """
    shapeImage = np.array( img.shape )
    
    if center is None:
        center = np.round( shapeImage/2.0 )
    if final_radius is None:
        final_radius = np.min( np.floor(shapeImage/2.0) )
    if initial_radius is None:
        initial_radius = 0
    if phase_width is None:
        phase_width = final_radius - initial_radius

    if mode == 'lin' or mode == 'linear':
        theta , R = np.meshgrid( np.linspace(0, 2*np.pi, phase_width), np.arange(initial_radius, final_radius))
    elif mode == 'log':
        theta , R = np.meshgrid( np.linspace(0, 2*np.pi, phase_width), 
                                np.logspace(np.log10(1.0+initial_radius), np.log10(final_radius), final_radius-initial_radius) )
        

        R = np.exp( R /( (final_radius-initial_radius)*2.)*np.log( phase_width ) )
    
    Xcart = R * np.cos(theta) + center[1]
    Ycart = R * np.sin(theta) + center[0]

    if( interpolate == 'nn'):
        Xcart = Xcart.astype(int)
        Ycart = Ycart.astype(int)

        polar_img = img[Ycart,Xcart]
        polar_img = np.reshape(polar_img,(final_radius-initial_radius,phase_width))
        
    elif( interpolate == 'bilinear' ):
        Xfloor = np.floor( Xcart )
        Yfloor = np.floor( Ycart )
        Xremain = Xcart - Xfloor
        Yremain = Ycart - Yfloor
        Xfloor = Xfloor.astype('int') # Can be used for indexing now
        Yfloor = Yfloor.astype('int')
        
        # Need to pad the input array by one pixel on the far edge
        img = np.pad( img, ((0,1),(0,1)), mode='symmetric' )
        # Index the four points
        polar_img = img[Yfloor+1,Xfloor+1] *Xremain*Yremain
        polar_img += img[Yfloor,Xfloor] * (1.0 - Xremain)*(1.0 - Yremain)
        polar_img += img[Yfloor+1,Xfloor] * (1.0 - Xremain)*Yremain
        polar_img += img[Yfloor,Xfloor+1] * Xremain*(1.0 - Yremain)
        # Crop the far edge, because we interpolated one pixel too far
        # polar_img = polar_img[:-1,:-1]
        polar_img = np.reshape(polar_img,( np.int(final_radius-initial_radius), np.int(phase_width) ))
    return polar_img
    
def interp2_bilinear(im, x, y):
    """
    Ultra-fast interpolation routine for 2-D images.  x and y are meshes.  The 
    coordinates of the image are assumed to be 0,shape[0], 0,shape[1]
    
    BUG: This is sometimes skipping the last row and column
    """
    x = np.asarray(x)
    y = np.asarray(y)

    x0 = np.floor(x).astype(int)
    x1 = x0 + 1
    y0 = np.floor(y).astype(int)
    y1 = y0 + 1

    # RAM: center this cliping with a roll?
    x0 = np.clip(x0, 0, im.shape[1]-1);
    x1 = np.clip(x1, 0, im.shape[1]-1);
    y0 = np.clip(y0, 0, im.shape[0]-1);
    y1 = np.clip(y1, 0, im.shape[0]-1);

    Ia = im[ y0, x0 ]
    Ib = im[ y1, x0 ]
    Ic = im[ y0, x1 ]
    Id = im[ y1, x1 ]

    wa = (x1-x) * (y1-y)
    wb = (x1-x) * (y-y0)
    wc = (x-x0) * (y1-y)
    wd = (x-x0) * (y-y0)

    return wa*Ia + wb*Ib + wc*Ic + wd*Id
    
def interp2_nn( im, x, y ):
    """
    Fast nearest neighbour interpolation, used for more advaced (filtered)
    methods such as Lanczos filtering.  x and y are meshes.  The coordinates of 
    the image are assumed to be 0,shape[0], 0,shape[1]
    """
    # We use floor instead of round because otherwise we end up with a +(0.5,0.5) pixel shift
    px = np.floor(x).astype(int)
    py = np.floor(y).astype(int)
    # Clip checking, could be more efficient because px and py are sorted...
    px = np.clip( px, 0, im.shape[1]-1 )
    py = np.clip( py, 0, im.shape[0]-1 )
    
    return im[py,px]
    
def backgroundEstimate( input_image, fitType='gauss2', binFact=128, lpSigma=4.0 ):
    """
    Fits a 2D gaussian to a micrograph (which is heavily binned and Gaussian filtered) and returns the estimated
    background.  In general this is a very robust means to deal with non-uniform illumination or uneven ice. 
    Uses the basin-hopping algorithm as it's much more robust.
    
    If the background is extremely weak (std < 1.0) then the fitting is ignored and just the mean value is reported.
    """
    # background fitting
    xcrop, ycrop = np.meshgrid( np.arange(0,input_image.shape[1], binFact), np.arange(0,input_image.shape[0], binFact) )
    nn_sum = interp2_nn( scipy.ndimage.gaussian_filter( input_image, lpSigma ), xcrop, ycrop )
    
    # Crop 1 from each zero-edge and 2 pixels from each end-edge to avoid edge artifacts in fitting procedure
    # This is compensated for in the xback, yback meshes below
    xmesh = xcrop.astype('float32') / binFact - xcrop.shape[1]/2.0
    ymesh = ycrop.astype('float32') / binFact - ycrop.shape[1]/2.0
    
    xmesh = xmesh[1:-2, 1:-2]
    ymesh = ymesh[1:-2, 1:-2]
    nn_sum = nn_sum[1:-2, 1:-2]

    # Maybe you need to add an x_c*y_c term?  On experimentation the cross-term doesn't help
    def gauss2( p, x_in, y_in ):
        x_c = x_in - p[1]
        y_c = y_in - p[2]
        return p[0] + p[3]*np.exp(-x_c*x_c/p[4]**2 - y_c*y_c/p[5]**2) 
    
    def errorGauss2( p, c ):
        x_c = xmesh - p[1]
        y_c = ymesh - p[2]
        return np.sum( np.abs( c - (p[0] + p[3]*np.exp(-x_c*x_c/p[4]**2 - y_c*y_c/p[5]**2) ) ) )
        
    paramMat = np.ones( 6, dtype='float64' )          
    paramMat[0] = np.mean( nn_sum )
    paramMat[3] = np.mean( nn_sum )
    paramMat[4] = np.mean( nn_sum ) * np.std( nn_sum )
    paramMat[5] = paramMat[4]
    
    fitGauss2D = scipy.optimize.minimize( errorGauss2, paramMat, method='Powell', args=(nn_sum,)  )
    # fitGauss2D = scipy.optimize.basinhopping( errorGauss2, paramMat, minimizer_kwargs={'args':(nn_sum,), 'method':"Powell"} )
    
    xback, yback = np.meshgrid( np.arange(input_image.shape[1]), np.arange(input_image.shape[0]) )
    xback = xback.astype('float32') - input_image.shape[1]/2.0
    xback /= binFact
    yback = yback.astype('float32') - input_image.shape[0]/2.0
    yback /= binFact

    if fitGauss2D.success:
        back = gauss2( fitGauss2D.x, xback, yback )
        return back
    else: # Failure, have no effect
        print( "Background estimation failed" )
        return np.zeros_like( input_image )
        
    
def magickernel( imageIn, k=1, direction='down' ):
    """ 
    magickernel( imageIn, k=1, direction='down' )
        k = number of binning operations, so k = 3 bins by 8 x 8 
    Implementation of the magickernel for power of 2 image resampling.  Generally 
    should be used to get two images 'close' in size before using a more aggressive resampling 
    method like bilinear.  
    
    direction is either 'up' (make image 2x bigger) or 'down' (make image 2x smaller)
    k is the number of iterations to apply it.
    """
    
    if k > 1:
        imageIn = magickernel( imageIn, k=k-1, direction=direction )
        
    if direction == 'up':
        h = np.array( [[0.25, 0.75, 0.75, 0.25]] )
        h = h* np.transpose(h)
        
        imageOut = np.zeros( [ 2*imageIn.shape[0], 2*imageIn.shape[1] ] )
        # Slice the input image interlaced into the larger output image
        imageOut[1::2,1::2] = imageIn
        # Apply the magic kernel
        imageOut = scipy.ndimage.convolve( imageOut, h )
        
    elif direction == 'down':
        imageIn = np.pad( imageIn, [1,1], 'reflect' )
        
        h = 0.5*np.array( [[0.25, 0.75, 0.75, 0.25]] )
        h = h* np.transpose(h)
        # This is computationally a little expensive, we are only using one in four values afterward
        imageOut = scipy.ndimage.convolve( imageIn, h)
        # Slicing is (start:stop:step)
        imageOut = imageOut[0:-2:2,0:-2:2]
    else:
        return
    return imageOut
    
def squarekernel( imageIn, k=1, direction='down' ):
    """ 
    squarekernel( imageIn, k=1, direction='down' )
        k = number of binning operations, so k = 3 bins by 8 x 8 
    Implementation of a square kernel for power of 2 image resampling, i.e. rebinning
    
    direction is either 'up' (make image 2x bigger) or 'down' (make image 2x smaller)
    k is the number of iterations to apply it.
    """
    
    if k > 3:
        # We can do this for probably bin-factors of 2,4, and 8?
        imageIn = squarekernel( imageIn, k=(k-1), direction=direction )
    
    if k == 1:
        h = np.array( [[1.0, 1.0]] )
        step = 2
    elif k == 2:
        h = np.array( [[1.0,1.0,1.0,1.0]] )
        step = 4
    elif k == 3:
        h = np.array( [[1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0]] )
        step = 8
    h = h * np.transpose(h)    
    
    if direction == 'up':
        imageOut = np.zeros( [ 2*imageIn.shape[0], 2*imageIn.shape[1] ], dtype=imageIn.dtype )
        # Slice the input image interlaced into the larger output image
        imageOut[1::2,1::2] = imageIn
        # Apply the magic kernel
        imageOut = scipy.ndimage.convolve( imageOut, h )
        
    elif direction == 'down':
        # This is computationally a little expensive, we are only using one in four values afterward
        imageOut = scipy.ndimage.convolve( imageIn, h )
        # Slicing is (start:stop:step)
        imageOut = imageOut[0:-1:step,0:-1:step]
    else:
        return
    return imageOut

def imHist(imdata, bins_=256):
    '''Compute image histogram.
        [histIntensity, histX] = imHist( imageData, bins_=256 )
    '''
    im_values =  np.ravel(imdata)
    hh, bins_ = np.histogram( im_values, bins=bins_ )
    # check histogram format
    if len(bins_)==len(hh):
        pass
    else:
        bins_ = bins_[:-1]    # 'bins' == bin_edges
        
    return hh, bins_

def histClim( imData, cutoff = 0.01, bins_ = 512 ):
    '''Compute display range based on a confidence interval-style, from a histogram
    (i.e. ignore the 'cutoff' proportion lowest/highest value pixels)'''
    
    if( cutoff <= 0.0 ):
        return imData.min(), imData.max()
    # compute image histogram
    hh, bins_ = imHist(imData, bins_)
    hh = hh.astype( 'float' )

    # number of pixels
    Npx = np.sum(hh)
    hh_csum = np.cumsum( hh )
    
    # Find indices where hh_csum is < and > Npx*cutoff
    try:
        i_forward = np.argwhere( hh_csum < Npx*(1.0 - cutoff) )[-1][0]
        i_backward = np.argwhere( hh_csum > Npx*cutoff )[0][0]
    except IndexError:
        print( "histClim failed, returning confidence interval instead" )
        from scipy.special import erfinv
        
        sigma = np.sqrt(2) * erfinv( 1.0 - cutoff )
        return ciClim( imData, sigma )
    
    clim =  np.array( [bins_[i_backward], bins_[i_forward]] )
    if clim[0] > clim[1]:
        clim = np.array( [clim[1], clim[0]] )
    return clim
    
def ciClim( imData, sigma = 2.5 ):
    """
    Confidence interval color limits, for images.  Most useful for highly spikey data.
    """
    meanData = np.mean( imData )
    stdData = np.std( imData )
    return np.array( [meanData - sigma*stdData, meanData + sigma*stdData] )
    
def plotScalebar( mage, pixelsize, units='nm', color='r', forceWidth=None ):
    """
    Pass in an image objectand a pixelsize, and function will add a properly scaled 
    scalebar to it.
    
        mage is the return from plt.imshow(), i.e. a matplotlib.
        pixelsize is what it says it is
        units can be any string
        color can be any matplotlib recognize color type (array or string)
        forceWidth sets the width to forceWidth units
        
    Note: auto-sets the image extent, do not use with plt.axis('image')
    """
    # Figure out scalesize from plot extent
    magesize = mage.get_size()
    # This is scaling the image!
    # mage.set_extent( [0.0, magesize[1]*pixelsize, 0.0, magesize[0]*pixelsize] )
    
    if forceWidth == None:
        targetwidth = 0.16
        targetValue = targetwidth * magesize[1] * pixelsize
        pow10 = np.int( np.floor( np.log10( targetValue ) ) )
        scalevalue = np.round( targetValue, decimals=-pow10 )
        scalesize = scalevalue / pixelsize
        
    else:
        scalevalue = forceWidth
        scalesize = forceWidth / pixelsize
    
    # Use %g formatter here.
    textscale = r'$%2g'%scalevalue +'\/' + units + "$"
    
    scalebar1 = mpl_toolkits.axes_grid1.anchored_artists.AnchoredSizeBar( mage.axes.transData, scalesize, textscale,
              pad=0.2, loc=4, sep=7, borderpad=0.4, frameon=True)
    scalebar1.txt_label._text.set_color( color )
    scalebar1.txt_label._text.set_weight( 'bold' )
    scalebar1.txt_label._text.set_size( 18 )
    scalebar1.size_bar.get_children()[0].set( color=color, linewidth=6.0 )
    scalebar1.patch.set(alpha=0.66, boxstyle='round')
    # plt.gca().add_artist(scalebar1)
    mage.axes.add_artist(scalebar1)
    plt.pause(0.05) # Often scalebar isn't rendered before plt.show calls.
    pass

def plotHistClim( mage, cutoff=1E-3, colorbar=False, cbartitle="" ):
    """
    Pass in an image object and a pixelsize, and function will change
    
        mage is the return from plt.imshow(), i.e. a matplotlib.
        cutoff is the histogram cutoff passed into histClim
        colorbar=True will add a colorbar to the plot
    """
    clim = histClim( mage.get_array(), cutoff=cutoff )
    mage.set_clim( vmin=clim[0], vmax=clim[1] )
    if bool(colorbar):
        cbar = plt.colorbar(mage)
        cbar.set_label(cbartitle, rotation=270)
    pass    
    
############### MISCELLANEOUS ###############
def powerpoly1( x, a1, b1, a2, c1 ):
    return a1*(x**b1) + a2*x + c1

def fit( x, y, funchandle='gauss1', estimates=None ):
    """ Returns: fitstruct,  fitY, Rbest """
    from scipy.optimize import curve_fit 
    from scipy.stats.stats import linregress

    if funchandle == 'gauss1':
        def fitfunc( x, a1, b1, c1 ):
            return a1 * np.exp( -( (x-b1)/ c1)**2 )
        # Really arbitrary c1 estimate at basically 25 pixels..
        if estimates is None:
            estimates = np.array( [np.max(y), x[np.argmax(y)], 25.0*(x[1]-x[0]) ] )
        
    elif funchandle == 'poly1':
        def fitfunc( x, a1, b1 ):
            return a1 * x + b1
        if estimates is None:
            slope = (np.max(y)-np.min(y))/(np.max(x)-np.min(x))
            intercept = np.min(y) - slope*x[np.argmin(y)]
            estimates = [slope, intercept]
    elif funchandle == 'poly2':
        def fitfunc( x, a1, b1, c1 ):
            return a1 * x **2.0 + b1 *x + c1
        if estimates is None:
            slope = (np.max(y)-np.min(y))/(np.max(x)-np.min(x))
            intercept = np.min(y) - slope*x[np.argmin(y)]
            estimates = [0.0, slope, intercept]
    elif funchandle == 'poly3':
        def fitfunc( x, a1, b1, c1, d1 ):
            return a1 * x **3.0 + b1 *x**2.0 + c1*x + d1
        if estimates is None:
            slope = (np.max(y)-np.min(y))/(np.max(x)-np.min(x))
            intercept = np.min(y) - slope*x[np.argmin(y)]
            estimates = [0.0, 0.0, slope, intercept]
    elif funchandle == 'poly5':
        def fitfunc( x, a1, b1, c1, d1, e1, f1 ):
            return a1 * x **5.0 + b1 *x**4.0 + c1*x**3.0 + d1*x**2.0 + e1*x + f1
        if estimates is None:
            slope = (np.max(y)-np.min(y))/(np.max(x)-np.min(x))
            intercept = np.min(y) - slope*x[np.argmin(y)]
            estimates = [0.0, 0.0, 0.0, 0.0, slope, intercept]
    elif funchandle == 'abs1':
        def fitfunc( x, a1 ):
            return a1 * np.abs( x )
        if estimates is None:
            estimates = np.array( [ (np.max(y)-np.min(y))/(np.max(x)-np.min(x))])
    elif funchandle == 'exp':
        def fitfunc( x, a1, c1 ):
            return a1 * np.exp( c1*x )
        if estimates is None:
            estimates = np.array( [1.0, -1.0] )
    elif funchandle == 'expc':
        def fitfunc( x, a1, c1, d1 ):
            return a1 * np.exp( c1*x ) + d1
        if estimates is None:
            estimates = np.array( [1.0, -1.0, 1.0] )
    elif funchandle == 'power1':
        def fitfunc( x, a1, b1 ):
            return a1*(x**b1)
        if estimates is None:
            estimates = np.array( [1.0, -2.0] )   
    elif funchandle == 'power2':
        def fitfunc( x, a1, b1, c1 ):
            return a1*(x**b1) + c1
        if estimates is None:
            estimates = np.array( [1.0, -2.0, 1.0] )    
    elif funchandle == 'powerpoly1':
        def fitfunc( x, a1, b1, a2, c1 ):
            return a1*(x**b1) + a2*x + c1
        if estimates == None:
            estimates = np.array( [1.0, -2.0, 0.0,  1.0] )
    else:
        fitfunc = funchandle
        
    try:
        fitstruct, pcov = curve_fit( fitfunc, x, y, p0=estimates )
        perr = np.sqrt(np.diag(pcov))
        print( "Fitting completed with perr = " + str(perr) )
        fitY = fitfunc( x, *fitstruct )
        goodstruct = linregress( x, fitfunc( x, *fitstruct ) )
        Rbest = goodstruct[2]
    except RuntimeError:
        print( "RAM: Curve fitting failed")
        return
    return fitstruct,  fitY, Rbest

def guessCfgType( value ):
    # For guessing the data type (bool, integer, float, or string only) from ConfigParser
    if value.lower() == 'true':
        return True
    if value.lower() == 'false':
        return False
    try:
        value = np.int( value )
        return value
    except: 
        pass
    try: 
        value = np.float32( value )
        return value
    except: 
        pass 
    return value

def weightedErrorNorm( x, A, b, weights ):
    # weighted error to set of shifts
    return np.sum( weights * np.abs(np.dot( A, x) - b) )
    
def errorNorm( x, A, b ):
    # No damping
    return np.sum( np.abs(np.dot( A, x) - b) )
    
# Fit a logistical curve to the sigmoid... and use that as the weighting curve???
def logistic( peaksAxis, SigmaThres, K, Nu):
    return 1.0 - 1.0 / (1.0 + np.exp( -K*(-peaksAxis + SigmaThres) ) )**Nu

# So peaksig_axis and cdf are immutable, only K, SigmaThres, and Nu should be optimized
def minLogistic( x, hSigma, cdfPeaksig ):
    return np.float32( np.sum( np.abs( cdfPeaksig - (1.0 - 1.0 / (1.0 + np.exp( -x[1]*(-hSigma + x[0]) ) )**np.float64(x[2])   )) ) )
            
def which( program ):
    # Tries to locate a program 
    import os
    if os.name == 'nt':
        program_ext = os.path.splitext( program )[1]
        if program_ext == "":
            prog_exe = which( program + ".exe" )
            if prog_exe != None:
                return prog_exe
            return which( program + ".com" )
            
    def is_exe(fpath):
        return os.path.isfile(fpath) and os.access(fpath, os.X_OK)

    fpath, fname = os.path.split(program)
    if fpath:
        if is_exe(program):
            return program
    else:
        for path in os.environ["PATH"].split(os.pathsep):
            path = path.strip('"')
            exe_file = os.path.join(path, program)
            if is_exe(exe_file):
                return exe_file
    return None
    
def compressFile( filename, compress_ext = '.bz2', outputDir = None, n_threads=None ):
    
    if os.path.isfile( filename + compress_ext ):
        os.remove( filename + compress_ext )
    if n_threads == None:
        # print( "Warning: zorro_util.compressFile defaulting to 8 threads" )
        n_threads = 8
        
    relpath, basename = os.path.split( filename )
    if outputDir == None:
        outputDir = relpath
    newFilename = os.path.join( outputDir, basename + compress_ext )
    
    # WTH is going on here...  why is lbzip2 crapping out?
    if compress_ext == '.bz2' and which( 'lbzip2' ) != None:
        sub = subprocess.Popen( "lbzip2 -n %d"%n_threads +" -1 -c " + filename + " > " + newFilename, shell=True )
    elif compress_ext == '.gz' and which( 'pigz' ) != None:
        sub = subprocess.Popen( "pigz -p %d"%n_threads + " -1 -c " + filename + " > " + newFilename, shell=True )  
    elif which( '7z' ) != None:
        if compress_ext == '.bz2':
            sub = subprocess.Popen( "7z a -tbzip2 " + newFilename + " " + filename, shell=True )
        elif compress_ext == '.gz':
            sub = subprocess.Popen( "7z a -tgzip " + newFilename + " "  + filename, shell=True )
    else:
        print( "Warning: cannot compress " + filename + compress_ext )
        return
    while sub.wait(): pass
    #print( "compressFile: Trying to remove: " + filename )
    os.remove( filename )
    return newFilename
    
        
def decompressFile( filename, outputDir = None, n_threads = None ):
    
    relpath, file_front = os.path.split( filename )
    [file_front, file_ext] = os.path.splitext( file_front )
    file_ext = file_ext.lower()
    if n_threads == None:
        # print( "Warning: zorro_util.decompressFile defaulting to 8 threads" )
        n_threads = 8
    
    if outputDir == None:
        outputDir = relpath

    newFilename = os.path.join(outputDir,file_front)
    #### COMPRESSED FILES ####        
    if file_ext == '.bz2':
        if which('lbzip2') != None:
            sub = subprocess.Popen( "lbzip2 -n %d"%n_threads +" -d -c " + filename +" > " + newFilename, shell=True ) # File is now decompressed
        elif which('7z') != None:
            sub = subprocess.Popen( "7z e -o" + outputDir + " " + filename, shell=True )
        else:
            print( "Neither lbzip2 nor 7z found in path, cannot decompress files.")
            return filename
        
    elif file_ext == '.gz':
        if which('pigz') != None:
            sub = subprocess.Popen( "pigz -p %d"%n_threads +" -d -c " + filename + " > " + newFilename, shell=True ) # File is now decompressed
        # Make new filename from filefront
        elif which( '7z' ) != None:
            sub = subprocess.Popen( "7z e -o" + outputDir + " " + filename, shell=True )
        else:
            print( "Neither pigz nor 7z found in path, cannot decompress files.")
            return filename
    else:
        # Do nothing
        return filename
    # We can only get here if 
    while sub.wait(): pass
    #print( "decompressFile: Trying to remove: " + filename )
    os.remove( filename ) # Remove original, because redirects trn -k on.
    return newFilename  # Make new filename from filefront
    
    
############### pyFFTW interface ###############
def pyFFTWPlanner( realMage, fouMage=None, wisdomFile = None, effort = 'FFTW_MEASURE', n_threads = None, doForward = True, doReverse = True ):
    """
    Appends an FFTW plan for the given realMage to a text file stored in the same
    directory as RAMutil, which can then be loaded in the future with pyFFTWLoadWisdom.
    
    NOTE: realMage should be typecast to 'complex64' normally.
    
    NOTE: planning pickle files are hardware dependant, so don't copy them from one 
    machine to another. wisdomFile allows you to specify a .pkl file with the wisdom
    tuple written to it.  The wisdomFile is never updated, whereas the default 
    wisdom _is_ updated with each call. For multiprocessing, it's important to 
    let FFTW generate its plan from an ideal processor state.
    
    TODO: implement real, half-space fourier transforms rfft2 and irfft2 as built
    """
    
    import pyfftw
    import pickle
    import os.path
    from multiprocessing import cpu_count
    
    utilpath = os.path.dirname(os.path.realpath(__file__))
    
    # First import whatever we already have
    if wisdomFile is None:
        wisdomFile = os.path.join( utilpath, "pyFFTW_wisdom.pkl" )


    if os.path.isfile(wisdomFile):
        try:
            fh = open( wisdomFile, 'rb')
        except:
            print( "Util: pyFFTW wisdom plan file: " + str(wisdomFile) + " invalid/unreadable" )
            
        try:
            pyfftw.import_wisdom( pickle.load( fh ) )
        except: 
            # THis is not normally a problem, it might be empty?
            print( "Util: pickle failed to import FFTW wisdom" )
            pass
        try:
            fh.close()
        except: 
            pass

    else:
        # Touch the file
        os.umask(0000) # Everyone should be able to delete scratch files
        with open( wisdomFile, 'wb') as fh:
            pass
    
        # I think the fouMage array has to be smaller to do the real -> complex FFT?
    if fouMage is None:
        if realMage.dtype.name == 'float32':
            print( "pyFFTW is recommended to work on purely complex data" )
            fouShape = realMage.shape
            fouShape.shape[-1] = realMage.shape[-1]//2 + 1
            fouDtype =  'complex64'
            fouMage = np.empty( fouShape, dtype=fouDtype )
        elif realMage.dtype.name == 'float64': 
            print( "pyFFTW is recommended to work on purely complex data" )
            fouShape = realMage.shape
            fouShape.shape[-1] = realMage.shape[-1]//2 + 1
            fouDtype = 'complex128'
            fouMage = np.empty( fouShape, dtype=fouDtype )
        else: # Assume dtype is complexXX
            fouDtype = realMage.dtype.name
            fouMage = np.zeros( realMage.shape, dtype=fouDtype )
            
    if n_threads is None:
        n_threads = cpu_count()
    print( "FFTW using " + str(n_threads) + " threads" )
    
    if bool(doForward):
        #print( "Planning forward pyFFTW for shape: " + str( realMage.shape ) )
        FFT2 = pyfftw.builders.fft2( realMage, planner_effort=effort, 
                                    threads=n_threads, auto_align_input=True )
    else:
        FFT2 = None
    if bool(doReverse):
        #print( "Planning reverse pyFFTW for shape: " + str( realMage.shape ) )
        IFFT2 = pyfftw.builders.ifft2( fouMage, planner_effort=effort, 
                                      threads=n_threads, auto_align_input=True )
    else: 
        IFFT2 = None

    # Setup so that we can call .execute on each one without re-copying arrays
    # if FFT2 is not None and IFFT2 is not None:
    #    FFT2.update_arrays( FFT2.get_input_array(), IFFT2.get_input_array() )
    #    IFFT2.update_arrays( IFFT2.get_input_array(), FFT2.get_input_array() )
    # Something is different in the builders compared to FFTW directly. 
    # Can also repeat this for pyfftw.builders.rfft2 and .irfft2 if desired, but 
    # generally it seems slower.
    # Opening a file for writing is supposed to truncate it
    # if bool(savePlan):
    #if wisdomFile is None:
    # with open( utilpath + "/pyFFTW_wisdom.pkl", 'wb') as fh:
    with open( wisdomFile, 'wb' ) as fh:
        pickle.dump( pyfftw.export_wisdom(), fh )
            
    return FFT2, IFFT2
    
def findValidFFTWDim( inputDims ):
    """
    Finds a valid dimension for which FFTW can optimize its calculations. The 
    return is a shape which is forced to be square, as this gives uniform pixel
    size in x-y in Fourier space.
    
    If you want a minimum padding size, call as findValidFFTWDim( image.shape + 128 ) 
    or similar.
    """
    dim = np.max( np.round( inputDims ) )
    maxPow2 = np.int( np.ceil( math.log( dim, 2 ) ) )
    maxPow3 = np.int( np.ceil( math.log( dim, 3 ) ) )
    maxPow5 = np.int( np.ceil( math.log( dim, 5 ) ) )
    maxPow7 = np.int( np.ceil( math.log( dim, 7 ) ) )   
    
    dimList = np.zeros( [(maxPow2+1)*(maxPow3+1)*(maxPow5+1)*(maxPow7+1)] )
    count = 0
    for I in np.arange(0,maxPow7+1):
        for J in np.arange(0,maxPow5+1):
            for K in np.arange(0,maxPow3+1):
                for L in np.arange(0,maxPow2+1):
                    dimList[count] = 2**L * 3**K * 5**J * 7**I
                    count += 1
    dimList = np.sort( np.unique( dimList ) )
    dimList = dimList[ np.argwhere(dimList < 2*dim)].squeeze()
    dimList = dimList.astype('int64')
    # Throw out odd image shapes, this just causes more problems with many 
    # functions
    dimList = dimList[ np.mod(dimList,2)==0 ]
    
    # Find first dim that equals or exceeds dim
    nextValidDim =  dimList[np.argwhere( dimList >= dim)[0,0]]
    return np.array( [nextValidDim, nextValidDim] )    
    
    
    

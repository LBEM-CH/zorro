# -*- coding: utf-8 -*-
"""
Drift registration test on artificial, analytically generated phantom.

A purely analytical phantom allows us to generate subpixel shifts as desired.

Introduce: shot noise, per frame radiation damage, hot-pixel, background intensity ramp

Created on Thu Jan  7 16:37:51 2016

@author: Robert A. McLeod
"""

import zorro
import RAMutil as ram
import zorro
import numpy as np
import numexprz as ne
from plotting import ims
import matplotlib.pyplot as plt
import scipy.interpolate as intp

np.random.seed( 44 )
M = 40
N = 2048

contrast = 0.5 # Generally around 0.1 or so for energy-filtered images.  
tx = 0.5 # Time between frames in seconds (for drift)
ePerPixel = 2 # electrons per pixel per frame, must be an integer to make Poisson/Binomial statistics feasible computationally
# For a binomial distribution, the dose has to be an integer.  We could simulate the sum and then
# randomly distribute the counts in all the frames?
kV = 300E3
wavelen = ram.ewavelength( kV ) * 1E9 # wavelength in nm
C1 = 1000 # Defocus, nm
A1x = 50 # Astigmatism, x-axis, nm
A1y = 25 # Astigmatism, orthogonal to x-axis (which is actually 45 degrees, not 90), nm
pixelsize = 0.13 # nm per pixel
objRadius = 12 # object size in pixels (will be depricated when I develop a more complicated object)
Bfactor = 1.0 # nm^2 
criticalDose = 2000 # Scaling on Bfactor, electrons per square nm before contrast drops by 1/e
n_threads = 24
D_charge = 10.0
D_environ = 1.0 # Drift in pix**2 / s
k_decay = 0.5
velocity = np.array( [2.5, 0] ) # constant (bias) velocity applied to drift in pix/s
hotpixelRate = 0.005 # hot pixels, probability per pixel
hotpixelSigma = 5.0 # How bright are hot pixels?  Normally distributed with (Mean + sigma*std)

ne.set_num_threads( n_threads )

[xmesh,ymesh] = np.meshgrid( np.arange(-N/2,N/2).astype(zorro.float_dtype), np.arange(-N/2,N/2).astype(zorro.float_dtype) ) 
r2mesh = ne.evaluate( 'xmesh**2 + ymesh**2' )

t_axis = np.arange( 1, M + tx ) * tx
dose_axis = np.arange( 1, M+1 ) * ePerPixel / pixelsize**2

Drift_decay = np.exp( -k_decay * t_axis )
# Looks like we have to simulate each step individually if we want different scales
shifts = np.zeros( [M,2] )
for J in xrange(0,M):
    shifts[J,1] = np.random.normal( loc=velocity[1]*tx, scale=(D_charge * Drift_decay[J] + D_environ)*np.sqrt(tx), size=1 )
    shifts[J,0] = np.random.normal( loc=velocity[0]*tx, scale=(D_charge * Drift_decay[J] + D_environ)*np.sqrt(tx), size=1 )
trans = np.cumsum( shifts, axis=0 )

centroid = np.mean( trans, axis=0 )
trans -= centroid

splineX = intp.UnivariateSpline( t_axis, trans[:,1], s = 0.0 )
splineY = intp.UnivariateSpline( t_axis, trans[:,0], s = 0.0 )

t_display = np.linspace( np.min(t_axis), np.max(t_axis), 2048 )
plt.figure()
plt.plot( trans[:,1], trans[:,0], 'k.', label='trans' )
plt.plot( splineX(t_display), splineY(t_display), label='spline' )
plt.legend( loc='best' )

# motion-blur velocity vectors can be computed from numerical derivative of the splines
print( "TODO: compute instantaneous velocity vectors" )


# Object is a few spheres
# Make a short array of object positions
objectCount = 50
objectPositions = np.random.uniform( low = -N/2 + objRadius, high=N/2-objRadius, size=[objectCount,2] )
# Check for overlap
for J in xrange(0,objectCount):
    # Check against the following points
    for K in xrange(J+1,objectCount):
        euclid = np.sqrt( np.sum( (objectPositions[J,:] - objectPositions[K,:])**2) )
        if euclid <= 2*objRadius:
            print( str(J) + " is to close to " + str(K) )
            objectPositions[K,:] = np.random.uniform( low = -N/2 + objRadius, high=N/2-objRadius, size=[1,2] )
        K -= 1
    pass

phaseObject = np.ones( [M, N, N] )
for J in xrange(0,M):
    # Projected potential of a sphere as a 
    for K in xrange(0,objectCount):
        offsetX = objectPositions[K,1] + trans[J,1]
        offsetY = objectPositions[K,0] + trans[J,0]
        r2_s = ne.evaluate( "(xmesh+offsetX)**2 + (ymesh+offsetY)**2" )
        r_s = ne.evaluate( "sqrt(r2_s)" )
        projSphere = ne.evaluate( "sqrt( objRadius**2 - r2_s)" )
        projSphere[ np.isnan( projSphere ) ] = 0.0
        projSphere = 1.0 - projSphere*contrast/np.max( projSphere )
        if np.sum( np.isnan( projSphere) ) > 0:
            print( "Found Nans in J = " + str(J) + " and K = " + str(K) )
        else:
            phaseObject[J,:,:] *= projSphere

#ims( phaseObject[J,:,:] )    

# TODO: radiation damage, drift MTF, and CTF in Fourier space
[FFT2,IFFT2] = zorro.util.pyFFTWPlanner( r2mesh.astype(zorro.fftw_dtype), n_threads=n_threads )
FFTPhase = np.zeros( [M, N, N], dtype=zorro.fftw_dtype )
inv_ps = 1.0 / (pixelsize * N)
    
# Apply CTF waveplate
# Let's just ignore wavelength?  I don't think we've used it anywhere else?
qxmesh = xmesh * inv_ps * wavelen
qymesh = ymesh * inv_ps * wavelen
q2mesh = ne.evaluate( "qxmesh*qxmesh + qymesh*qymesh" )
# Compute CTF phase gradient
phaseC1 = ne.evaluate( 'C1*0.5*(qxmesh**2 + qymesh**2)' )
phaseA1x = ne.evaluate( 'A1x*0.5*(qxmesh**2 - qymesh**2)' )
phaseA1y = ne.evaluate( 'A1y*0.5*(-qxmesh**2 + qymesh**2)' )
wavenumber = 2.0j * np.pi / wavelen
Gamma = ne.evaluate( 'exp( wavenumber * (phaseC1 + phaseA1x + phaseA1y) )'  )

#ims( np.angle(Gamma), titles=("Gamma aberration waveplate",) )

Gamma = np.fft.ifftshift( Gamma )
realObject = np.zeros_like( phaseObject, dtype=zorro.fftw_dtype )
Bfilter = np.zeros_like( phaseObject, dtype=zorro.fftw_dtype )
for J in xrange(0,M):
    FFT2.update_arrays( phaseObject[J,:,:].astype(zorro.fftw_dtype), FFTPhase[J,:,:] ); FFT2.execute()

    # Build B-factor for this dose
    # Bfactor needs to be modified by the wavelen, because q meshes are unitless (angular spectrum)
    Bfactor_funcDose = Bfactor*dose_axis[J] / criticalDose / 4 / wavelen**2
    
    Bfilter[J,:,:] = np.fft.ifftshift( ne.evaluate( "exp( -Bfactor_funcDose * q2mesh )" ) )
    FFTPhase[J,:,:] *= Bfilter[J,:,:]
    
    # Apply CTF phase plate to complex amplitude
    FFTPhase[J,:,:] *= Gamma 
    
    # TODO: apply drift MTF 
    # MTF_sinc = np.sinc( np.pi * t_x * velocity * q )
    # Define the randomwalk as having a mean square displacement of what?
    # MTF_randomwalk = 1.0 - np.exp( -4.0 * np.pi * D * tx)
    # So the velocity 
    
    
    # Inverse FFT back to real-space
    IFFT2.update_arrays( FFTPhase[J,:,:], realObject[J,:,:] ); IFFT2.execute()
    
realObject = np.real( realObject ).astype( zorro.float_dtype ) / (N*N)
print( "realObject min contrast: %f"%np.min( realObject ) )
print( "realObject max contrast: %f"%np.max( realObject ) )
realContrast = (np.max(realObject) - np.min(realObject))/(np.max(realObject) + np.min(realObject))
print( "realObject contrast: %f"%(realContrast) )
# Force maximum contrast to 1.0
realObject = realObject / np.max( realObject )
# Apply a low-pass filter to the CTF to reflect coherence angle Beta?



#ims( realObject, titles=("Real Object",)  )
#ims( np.abs( np.fft.fftshift( np.fft.fft2( realObject[0,:,:] ))))
#ims( np.abs( np.fft.fftshift( np.fft.fft2( realObject[J,:,:] ))))




# Generate a weak background, that's off-center
#background = ne.evaluate( "1.0 - ((xmesh - N/8)**2 + (ymesh+N/16)**2 ) / (N*4.0)**2" )
#print( "Background minimum: %f"%np.min( background ) )
## We assume the background is in the illumination, but it's not affected by the CTF because it's 
## pure amplitude.
#phaseObject *= background


# So we need Poisson distributed counts, and then we need to apply the phaseObject as a binomial distribution 
# on top of the Poisson counts
shotNoise = np.random.poisson( lam=ePerPixel, size=realObject.shape ) # This generator for poisson is quite slow...
noisyObject = np.random.binomial( shotNoise, realObject, size=phaseObject.shape ).astype( zorro.float_dtype )
# Is this right?  Should it be the other way around?  But then we don't have a good continuous Poisson 

noisyContrast = np.sqrt(2) * np.std( noisyObject, axis=(1,2) ) / np.mean( noisyObject, axis=(1,2))
print( "noisyObject mean contrast: %f"%(np.mean(noisyContrast) ) )


# ims( noisyObject, titles=("Noisy Object",)  )


# Apply detector MTF, ideally this would be DQE
print( "TODO: apply detector MTF" )

# Apply hot pixels
print( "TODO: apply hot pixels" )

hotpixMask = np.random.binomial( 1, hotpixelRate, size=[N,N] )
# NOT WHAT I WANT, need some variation frame-to-frame in hot pixel values...
hotpixImage = np.random.normal( loc=(ePerPixel +hotpixelSigma*np.sqrt(ePerPixel)), 
        scale=hotpixelSigma*np.sqrt(ePerPixel), size=hotpixMask.shape ).astype('float32')
hotpixImage = np.clip( hotpixMask, 0, np.Inf )
hotpixImage *= (hotpixMask.astype( 'float32' ))
ims( hotpixImage )

for J in xrange(0,M):
    print( "Applying hot pixel mask for image " + str(J) )
    noisyObject[J,:,:] = (noisyObject[J,:,:] *(~hotpixMask)) + hotpixImage


zorroReg = zorro.ImageRegistrator()
zorroReg.shapePadded = zorro.util.findValidFFTWDim( [N*1.1,N*1.1] )
zorroReg.images = noisyObject
zorroReg.stackName = 'Sim'
zorroReg.saveC = True
zorroReg.Bmode = 'opti'
zorroReg.triMode = 'diag'
zorroReg.weightMode = 'logistic'
zorroReg.peaksigThres = 5.5
zorroReg.diagWidth = 5
#zorroReg.Brad = 256

zorroReg.alignImageStack()

unblurReg = zorro.ImageRegistrator()
unblurReg.images = noisyObject
unblurReg.stackName = "Sim"
unblurReg.pixelsize = 1.0
unblurReg.xcorr2_unblur()

mcReg = zorro.ImageRegistrator()
mcReg.images = noisyObject
mcReg.stackName = "Sim"
mcReg.Brad = 256
mcReg.xcorr2_mc()

# The features are too low frequency for the maximum to be reliable here...  Need sharper features in the 
# phantom.
ims( zorroReg.C )

plt.figure()
plt.plot( trans[:,1], trans[:,0], '.-', label='Sim' )
plt.plot( zorroReg.translations[:,1], zorroReg.translations[:,0], '.-', label='Zorro' )
plt.plot( unblurReg.translations[:,1], unblurReg.translations[:,0], '.-', label='UnBlur' )
plt.plot( mcReg.translations[:,1], mcReg.translations[:,0], '.-', label='MC' )
plt.title( "Translations from analytic phantom" )
plt.legend( loc='best' )

bias_zorro = np.mean( zorroReg.translations - trans, axis=0 )
rms_zorro = np.std( zorroReg.translations - trans, axis=0 )

bias_unblur = np.mean( unblurReg.translations - trans, axis=0 )
rms_unblur = np.std( unblurReg.translations - trans, axis=0 )

bias_mc = np.mean( mcReg.translations - trans, axis=0 )
rms_mc = np.std( mcReg.translations - trans, axis=0 )

print( "Zorro bias = " + str(bias_zorro) + ", rms = " + str(rms_zorro) )
print( "UnBlur bias = " + str(bias_unblur) + ", rms = " + str(rms_unblur) )
print( "Motioncorr bias = " + str(bias_mc) + ", rms = " + str(rms_mc) )

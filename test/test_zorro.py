# -*- coding: utf-8 -*-
"""
Created on Fri Sep 30 09:44:09 2016

@author: Robert A. McLeod
"""


import zorro
import numpy as np
import numpy.testing as npt
import os, os.path, sys
import subprocess
import tempfile
import unittest
try:
    from termcolor import colored
except:
    def colored( string ):
        return string

float_dtype = 'float32'
fftw_dtype = 'complex64'
tmpDir = tempfile.gettempdir()

    
#==============================================================================
# zorro.ImageRegistrator test
# 
# Build a Gaussian + random noise and align it.  Assert that the shifts are 
# reasonable
#==============================================================================   
class test_ImageRegistrator(unittest.TestCase):
    
    def setUp(self):
        pass
    
    """
    def test_simpleRegistrator(self):
        
        zTest = zorro.ImageRegistrator()
        zTest.shapePadded = [320,320]
        zTest.fouCrop = [320,320]
        zTest.subPixReg = 4
        zTest.saveC = True 
        
        shapeStack = [10, 256, 272]
        known_trans = np.cumsum( np.random.normal( loc=2.0, scale=3.0, size=[shapeStack[0],2] ), axis=0 )
        known_trans -= np.mean( known_trans, axis=0 )
        
        
        xmesh,ymesh = np.meshgrid( np.arange(-shapeStack[2]/2,shapeStack[2]/2), 
                                             np.arange(-shapeStack[1]/2, shapeStack[1]/2) )
        # Generate images of Gaussians plus random noise
        zTest.images = np.empty( shapeStack, dtype=float_dtype )
        for J in np.arange( shapeStack[0] ):
            zTest.images[J,:,:] = 5.0 * np.exp( -((xmesh-known_trans[J,1])/10.0)**2   
                                            -((ymesh-known_trans[J,0])/12.0)**2 )
            zTest.images[J,:,:] += np.random.normal( loc=0.0, scale=0.1, size=shapeStack[1:] )
            
        # Check for dtype changes    
        assert( zTest.images.dtype == float_dtype )
        
        zTest.alignImageStack()
        
        # Check for dtype changes
        assert( zTest.images.dtype == float_dtype )
        
        # print( "Drift input: \n%s" % known_trans )
        # print( "Drift output: \n%s " % zTest.translations )
        zTest.translations += np.mean( known_trans + zTest.translations, axis=0 )
        driftErr = np.mean( np.abs( known_trans + zTest.translations ) )
        print( "Drift error (pix): %f" % driftErr )
        assert( driftErr < 1.0 / zTest.subPixReg )
        """

    
if __name__ == "__main__":
    from zorro import __version__
    print( "ZORRO TESTING FOR VERSION %s " % __version__ )
    unittest.main( exit=False )
    

    
    



                     
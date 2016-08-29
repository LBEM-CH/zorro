# -*- coding: utf-8 -*-
"""
Created on Mon Sep  7 10:25:34 2015
@author: Robert A. McLeod
@email: robbmcleod@gmail.com OR robert.mcleod@unibas.ch
"""
import numpy as np
import zorro
import os, os.path
import glob
from zorro import ReliablePy
import subprocess
import matplotlib.pyplot as plt


# Find all .dm4 files in a directory that don't have a log and process them.
globPattern = "align/*zorro.mrc"
pixelsize = 0.1039 # in NANOMETERS
voltage = 300.0
C3 = 2.7
detectorPixelSize = 5.0

print( "#################################################################################" )
print( " CHECK FOR CORRECTNESS: \n    pixelsize = %f nm\n    voltage = %.1f kV\n    C3 = %.2f mm\n    detectorPixelSize = %.1f um" %(pixelsize, voltage, C3, detectorPixelSize) )
print( "#################################################################################" )

filenames = glob.glob( globPattern )

# Open the first one and grab the pixelsize


N = len(filenames)
CTFInfo = {}
CTFInfo['DefocusU'] = np.zeros( [N], dtype='float32' )
CTFInfo['DefocusV'] = np.zeros( [N], dtype='float32' )
CTFInfo['FinalResolution'] = np.zeros( [N], dtype='float32' )
CTFInfo['DefocusAngle'] = np.zeros( [N], dtype='float32' )
CTFInfo['CtfFigureOfMerit'] = np.zeros( [N], dtype='float32' )

# Better approach is probably to call this in batch mode.  Then I get the all_micrographs_gctf.star file!
ctfReg = zorro.ImageRegistrator()
ctfReg.n_threads = 16
ctfReg.savePNG = True
ctfReg.files['figurePath'] = '../figs/'
ctfReg.plotDict["imageSum"] = False
ctfReg.plotDict["imageFirst"] = False
ctfReg.plotDict["FFTSum"] = False
ctfReg.plotDict["polarFFTSum"] = False
ctfReg.plotDict["corrTriMat"] = False
ctfReg.plotDict["shiftsTriMat"] = False
ctfReg.plotDict["peaksigTriMat"] = False
ctfReg.plotDict["errorTriMat"] = False
ctfReg.plotDict["translations"] = False
ctfReg.plotDict["pixRegError"] = False
ctfReg.plotDict["CTF4Diag"] = True
ctfReg.plotDict["logisticsCurve"] = False
ctfReg.plotDict["Transparent"] = False
ctfReg.plotDict["dpi"] = 200

# Apparently I have to write 

gctf_exec = "gctf %s --apix %f --kV %f --cs %f --dstep %f --do_EPA 1 --logsuffix _ctffind3.log" % (
        globPattern, pixelsize*10, voltage, C3, detectorPixelSize )

devnull = open(os.devnull, 'w' )
#sub = subprocess.Popen( gctf_exec, shell=True, stdout=devnull, stderr=devnull )
sub = subprocess.Popen( gctf_exec, shell=True )

sub.wait() 

# TODO: generate all the CTF diagnostic outputs?  Clean up files.

rlnCTF = ReliablePy.ReliablePy()
rlnCTF.load( "micrographs_all_gctf.star" )

Nbins = 2.0*np.int( np.sqrt( rlnCTF.star['data_']['DefocusU'].size ) )

hDefocusU, xDefocusU = np.histogram( rlnCTF.star['data_']['DefocusU'], bins=Nbins )
hDefocusU = hDefocusU.astype('float32'); xDefocusU = xDefocusU[:-1]

hDefocusV, xDefocusV = np.histogram( rlnCTF.star['data_']['DefocusV'], bins=Nbins )
hDefocusV = hDefocusV.astype('float32'); xDefocusV = xDefocusV[:-1]

hR2, xR2 = np.histogram( rlnCTF.star['data_']['CtfFigureOfMerit'], bins=Nbins )
hR2 = hR2.astype('float32'); xR2 = xR2[:-1]

hResolution, xResolution = np.histogram( rlnCTF.star['data_']['FinalResolution'], bins=Nbins )
hResolution = hResolution.astype('float32'); xResolution = xResolution[:-1]

plt.figure()
plt.plot( xDefocusU, hDefocusU, '.-', label='DefocusU' )
plt.plot( xDefocusV, hDefocusV, '.-', label='DefocusV' )
plt.xlabel( "Defocus, $C1$ ($\AA$)" )
plt.ylabel( "Histogram counts" )
plt.legend( loc='best' )
plt.savefig( "histogram_gctf_defocus.png" )

plt.figure()
plt.plot( xResolution, hResolution, '.-', label='Resolution' )
plt.xlabel( "Estimated resolution ($\AA$)" )
plt.ylabel( "Histogram counts" )
plt.legend( loc='best' )
plt.savefig( "histogram_gctf_resolution.png" )

plt.figure()
plt.plot( xR2, hR2, '.-', label='$R^2$' )
plt.xlabel( "CTF Figure of Merit, $R^2$ (a.u.)" )
plt.ylabel( "Histogram counts" )
plt.legend( loc='best' )
plt.savefig( "histogram_gctf_R2.png" )

# TODO: throw-out outliers in resolution and the R2 values?


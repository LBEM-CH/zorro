# -*- coding: utf-8 -*-
"""
Created on Mon Sep  7 10:25:34 2015
@author: Robert A. McLeod
@email: robbmcleod@gmail.com OR robert.mcleod@unibas.ch
"""
import numpy as np
import zorro
import os, os.path, sys, glob, shutil
from zorro import ReliablePy, ioMRC, util
import subprocess
import matplotlib.pyplot as plt
import matplotlib.offsetbox
import itertools
import time

#### INITIAL VARIABLES ####
# Works on Zorro log files to batch process using GCTF
globPattern = "*.log"
N_gpu = 4
figPath = "../figs"


#### START OF CODE ####
def chunkify(lst,n):
    return [ lst[i::n] for i in xrange(n) ]


logNames = glob.glob( globPattern )
N = len(logNames)

logName = logNames[0]
# We assume that the pixelsize/voltage/etc. is constant for each image in the set.
ctfReg = zorro.ImageRegistrator()
ctfReg.loadConfig( logName )
pixelsize = ctfReg.pixelsize # in NANOMETERS
voltage = ctfReg.voltage
C3 = ctfReg.C3
AmpContrast = 0.1
detectorPixelSize = ctfReg.detectorPixelSize
mag = detectorPixelSize*1E3 / pixelsize
print( "#################################################################################" )
print( " CHECK FOR CORRECTNESS: \n    pixelsize = %f nm\n    voltage = %.1f kV\n    C3 = %.2f mm\n    detectorPixelSize = %.1f um" %(pixelsize, voltage, C3, detectorPixelSize) )
print( "#################################################################################" )


sumNames = [None] * N
for J, logName in enumerate(logNames):
    ctfReg.loadConfig( logName )
    sumNames[J] = ctfReg.files['sum']
    # Do we need any more information from the log?  We could get the previous information into another CTFInfo dict?

chunkedSums = chunkify( sumNames, N_gpu )

subs = [None]* N_gpu
tempStar = [None] * N_gpu
tempQsh = [None] * N_gpu

starHeader = """
data_

loop_
_rlnMicrographName #1
_rlnCtfImage #2
_rlnDefocusU #3
_rlnDefocusV #4
_rlnDefocusAngle #5
_rlnVoltage #6
_rlnSphericalAberration #7
_rlnAmplitudeContrast #8
_rlnMagnification #9
_rlnDetectorPixelSize #10
_rlnCtfFigureOfMerit #11
_rlnFinalResolution #12

"""

for gid in np.arange(N_gpu):
   
    # Write a star file to dish for each GPU to work on
    tempStar[gid] = "gctf_temp%d.star" % gid
    with open( tempStar[gid], 'wb' ) as starh:
        starh.write( starHeader )
        for mrcFile in chunkedSums[gid]:
            starh.write( "%s %s %.2f %.2f %.3f %.1f %.2f %.3f %.2f %.2f %.6f %.2f \n" % 
            (mrcFile, "none", 0.0, 0.0, 0.0, voltage, C3, AmpContrast, mag, detectorPixelSize, 0.0, 0.0) )
        starh.write( "\n" )
    gctf_prog = zorro.util.which( 'gctf' )
    gctf_exec = "`which gctf` --input_ctfstar %s --ctfstar gctf%d_micrographs.star --gid %d --do_EPA 1 --logsuffix _ctffind3.log\n" % (
        tempStar[gid], gid, gid  )
        
    qsubHeader = """#! /bin/bash
#$ -N gctf_%d
#$ -pe smp 1
#$ -q gpu.q
#$ -o gctf%d.out
#$ -e gctf%d.err
#$ -cwd
#$ -S /bin/bash

export PATH="/scicore/pfs/scicore-p-structsoft/Gctf_v0.50/bin:$PATH"

module load CUDA/7.5.18
source /scicore/soft/UGE/current/util/gepetools/gepetools.sh
""" % (gid, gid, gid)

    tempQsh[gid] = "q_gctf%d.sh" % gid
    with open( tempQsh[gid], 'wb' ) as qh:
        qh.write( qsubHeader )
        qh.write( gctf_exec )
    subs[gid] = subprocess.Popen( "qsub -sync y " + tempQsh[gid], shell=True )
    

#devnull = open(os.devnull, 'w' )
#sub = subprocess.Popen( gctf_exec, shell=True, stdout=devnull, stderr=devnull )


# Wait for all jobs to finish
# This will only work if I have interactive jobs, which isn't working how I expect it to
# Could it possibly be this thing???
idler = itertools.cycle( ['\r/','\r_','\r\\'] )
idleSentinel = True
finishPoll = np.zeros( N_gpu, dtype='bool' )
t0 = time.time()

print( "Submitted GCTF jobs, terminal will be idle for awhile...  " )
while not np.all(finishPoll):
    # TODO: could output tail from the gctf.out file
    sys.stdout.write( next(idler) )
    sys.stdout.flush()
    for J, sub, in enumerate(subs):    
        finishPoll[J] = (sub.poll() != None) # Any return code should cause us to halt
    time.sleep(1.0)

t1 = time.time()
print( "\nGCTF processed %d images in %.2f seconds." % ( N, t1-t0 ) )

# Load all the gctf?_micrographs.star files, find the line after the _rlnXXX headers and merge the files
outStars = glob.glob( "gctf*_micrographs.star" )
joinedStar = []
for I, outStar in enumerate(outStars):
    with open( outStar, 'rb' ) as oh:
        starLines = oh.readlines() # data_ loop_ stuff is first four lines
    for J in np.arange( 4, len(starLines) ):
        if not starLines[J].startswith("_rln"):
            startDataOnLine = J
            break
    
    if I == 0: # Add the header material from the first file only
        joinedStar = starLines[:startDataOnLine]
        
    # Extend the list with the new data
    joinedStar.extend( starLines[startDataOnLine:] )
    
    # Remove trailing \n if present
    if joinedStar[-1].startswith( "\n" ):
        joinedStar = joinedStar[:-1]
            
        
with open( "../micrographs_all_gctf.star", 'wb' ) as jh:
    jh.writelines( joinedStar )
        
print( "Done joining GCTF star files into " + os.path.realpath(  "../micrographs_all_gctf.star" ) )


# Clean up temporary files and concatinate out file
with open('gctf.out','wb') as wfd:
    for gid in np.arange(N_gpu):
        with open( "gctf%d.out" % gid ,'rb') as fd:
            shutil.copyfileobj(fd, wfd)

for gid in np.arange(N_gpu):
    os.remove( "gctf%d.out" % gid )
    os.remove( "gctf%d.err" % gid )
    os.remove( tempStar[gid] )
    os.remove( tempQsh[gid] )
    os.remove( outStars[gid] )
    
    
print( "Plotting GCTF diagnostic images" )
rlnCTF = ReliablePy.ReliablePy()
rlnCTF.load( "../micrographs_all_gctf.star" )



if not os.path.isdir( figPath  ):
    os.mkdir( figPath  )
    
    
plt.switch_backend( 'Agg' )
ctfImageNames = [None] * len( rlnCTF.star['data_']['CtfImage'] )
for J, ctfImageName in enumerate( rlnCTF.star['data_']['CtfImage'] ):
    ctfImageNames[J] = ctfImageName.rstrip(":mrc")

for J, ctfFile in enumerate(ctfImageNames):
    print( "Generating diagnostic PNG for: " + ctfFile )
    CTFDiag = ioMRC.MRCImport( ctfFile )
    clim = util.histClim( CTFDiag, cutoff=1E-3 )
    fig1 = plt.figure( figsize=(7,7) )
    mapCTF = plt.imshow( CTFDiag, cmap='gray', vmin=clim[0], vmax=clim[1], interpolation='none' )
    
    inv_ps = 1.0 / (CTFDiag.shape[0] * pixelsize )
    # Get appropriate matching info from rlnCTF
    results = (u"$DF_1:\/%.1f\/\AA$\n"%rlnCTF.star['data_']['DefocusU'][J] +
                     u"$DF_2:\/%.1f\/\AA$\n"%rlnCTF.star['data_']['DefocusV'][J] +
                     u"$\gamma:\/%.1f^\circ$\n"%rlnCTF.star['data_']['DefocusAngle'][J]+
                     u"$R:\/%.3f$\n"%rlnCTF.star['data_']['CtfFigureOfMerit'][J] +
                     u"$Fit\/res:\/%.1f\/\AA$"%rlnCTF.star['data_']['FinalResolution'][J] )

    infobox = matplotlib.offsetbox.AnchoredText( results, pad=0.5, loc=1, prop={'size':13} )
    fig1.gca().add_artist( infobox )
        
    fig1.gca().set_axis_off() # This is still not cropping properly...
        
    util.plotScalebar( mapCTF, inv_ps, units=u'nm^{-1}' )
    plt.savefig( os.path.join( figPath, os.path.basename(os.path.splitext( ctfFile)[0]) )+ "_CTFDiag.png", 
                bbox_inches='tight', dpi=200 )
    # Remove the MRC            
    os.remove( ctfFile )
    

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
plt.savefig( os.path.join( figPath, "histogram_gctf_defocus.png" ) )

plt.figure()
plt.plot( xResolution, hResolution, '.-', label='Resolution' )
plt.xlabel( "Estimated resolution ($\AA$)" )
plt.ylabel( "Histogram counts" )
plt.legend( loc='best' )
plt.savefig( os.path.join( figPath, "histogram_gctf_resolution.png" ) )

plt.figure()
plt.plot( xR2, hR2, '.-', label='$R^2$' )
plt.xlabel( "CTF Figure of Merit, $R^2$ (a.u.)" )
plt.ylabel( "Histogram counts" )
plt.legend( loc='best' )
plt.savefig( os.path.join( figPath,"histogram_gctf_R2.png" ) )

# TODO: clean up GCTF junk that we don't need

# TODO: throw-out outliers in resolution and the R2 values?


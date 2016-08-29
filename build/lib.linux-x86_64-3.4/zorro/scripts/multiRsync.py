# -*- coding: utf-8 -*-
"""
Created on Fri Jun  3 11:35:55 2016

@author: rmcleod
"""

# Let's write a test script that launches 4 subprocesses to rsync data to SciCore and see how fast we get


import subprocess
import glob, os, os.path
import time
import numpy as np

def chunks(l, n):
    n = np.maximum(1, n)
    return [l[i:i + n] for i in np.arange(0, len(l), n)]
        
MByte = 2.0**20
GByte = 2.0**30

Nstreams = 8
sourceDir = "/Projects/AutomatorTest/out/raw"
targetServer = "mcleod@login2.bc2.unibas.ch"
targetDir = "/scicore/pfs/c-cina/RsyncTest/"
globPattern = "*.dm4"
doCompression = False

# TO BS-GPU04:
# 4 streams, compression = 85 MB/s
# 4 streams, uncompressed = 275 MB/s
# TO LOGIN2.BC2.UNIBAS.CH:
# 4 streams, uncompressed = 110 MB/s
# 8 streams, compressed = 151 MB/s
# 8 streams, uncompressed = 

fileList = glob.glob( os.path.join( sourceDir, globPattern ) )

# Make sure we don't have more streams than files
Nstreams = np.minimum( len(fileList), Nstreams)

# Cut into streams
chunkedLists = chunks( fileList, Nstreams )

# TODO: generate file lists on /scratch and pass into rsync 

t0 = time.time()
totalSize = 0
procList = []
for J, filename in enumerate(fileList):
    # Try with and without compression (-z), compression might help with lots of streams
    totalSize += os.path.getsize( filename )
    rsync_exec = "rsync -v"
    if doCompression: 
        rsync_exec += "z"
    rsync_exec += " "
    rsync_exec += os.path.join( sourceDir, filename ) + " "
    rsync_exec += targetServer + ":" + targetDir
    print( rsync_exec )
    procList.append( subprocess.Popen( rsync_exec, shell=True ) )
    if J >= Nstreams:
        break
    time.sleep( 0.1 )
        
for proc in procList:
    proc.wait()
    
t1 = time.time()
tdelta = t1-t0
print( "Finished transferring." )
print( "Transferred %.3f GB in %.1f s ( %.3f MB/s average)" %(totalSize/GByte, tdelta, totalSize/tdelta/MByte ) )


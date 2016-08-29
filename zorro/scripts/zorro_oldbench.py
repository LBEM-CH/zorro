# -*- coding: utf-8 -*-
"""
Created on Fri Jul 29 12:08:42 2016

@author: rmcleod
"""

import zorro
import numpy as np
import matplotlib.pyplot as plt
import time
import numexprz as nz
import subprocess
import os, os.path, shutil

# User must provide a stack (.dm4, .mrc, etc. as a test object)
stackName = ["/mnt/cache/scratch/2016-03-24_11_06_32.dm4" ]


max_threads = nz.detect_number_of_cores()
dirName = os.path.dirname( stackName[0] )
baseName = os.path.basename( stackName[0] )
# We need to make 3 copies of the stack  
stackFront, stackExt = os.path.splitext( stackName[0] )


# We have a maximum of 4 processes here
stackName.append( stackFront + "1" + stackExt )
shutil.copy( stackName[0], stackName[1] )
stackName.append( stackFront + "2" + stackExt )
shutil.copy( stackName[0], stackName[2] )
stackName.append( stackFront + "3" + stackExt )
shutil.copy( stackName[0], stackName[3] )


n_cases = 8
n_procs = [1,2,3,4,1,2,3,4]
n_threads = [max_threads, max_threads/2, max_threads/3, max_threads/4, max_threads/2, max_threads/4, max_threads/6, max_threads/8]
#n_cases = 2
#n_procs = [1,1]
#n_threads = [max_threads, max_threads/2]

t_start = np.zeros( n_cases )
t_finish = np.zeros( n_cases )

zorroDefault = zorro.ImageRegistrator()
zorroDefault.diagWidth = 5
zorroDefault.CTFProgram = None
zorroDefault.filterMode = None
zorroDefault.doLazyFRC = False
zorroDefault.savePNG = False

for K in xrange(n_cases):
    print( "##### STARTING BENCHMARK #%d, N_PROCS = %d, N_THREADS = %d" %(K, n_procs[K], n_threads[K] ) )
    zorroDefault.n_threads = n_threads[K]
    ProcList = []
    
    for J in xrange( n_procs[K] ):
        zorroDefault.files['stack'] = stackName[0]
        zorroDefault.saveConfig( "stack%d.zor"%J )

    
    t_start[K] = time.time()
    for J in xrange( n_procs[K] ):
        ProcList.append( subprocess.Popen( "zorro -c stack%d.zor"%J, shell=True ) )
    for P in ProcList:
        P.wait()
    t_finish[K] = time.time()


for K in xrange(n_cases):
    print( "Case %d: %d processes, %d threads each, time per stack: %.3f s" %( K, n_procs[K], n_threads[K], (t_finish[K]-t_start[K])/n_procs[K] )   )

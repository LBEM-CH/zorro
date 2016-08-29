# -*- coding: utf-8 -*-
"""
Sun Grid Engine (SQE) qsub batch Zorro submission script
Created on Tue Feb  2 16:28:50 2016
@author: Robert A. McLeod

This is currently intended to be run from a <project>/raw/ directory.  Puts results in <project>/align 
and figures in <project>/figs

TODO: this clearly needs to stage the submissions a bit at the start.  IO collision between the processes is 
pretty heavy.
"""
import zorro
import os, os.path
import glob
import subprocess
import time

n_threads = 16
queue_name = "refinement.q"
python_script = "realignMC.py"

# Build default zorro cfg 
fileList = glob.glob( "*aligned_movie.mrcs" )

process_list = []
logList = [None] * len(fileList)
todoList = [True] * len(fileList)
for J, filename in enumerate(fileList):
    fileFront = os.path.split( os.path.splitext( filename )[0] )[1]
    # Copy the template script?
    logList[J] = filename + ".log"
    if os.path.isfile( logList[J] ):
        zorroTest = zorro.ImageRegistrator()
        zorroTest.loadConfig( logList[J] )
    print( logList[J] + " status: " + zorroTest.METAstatus )
    if zorroTest.METAstatus == 'fini':
        # Don't submit this job
        todoList[J] = False
        continue 
    
   

for J, filename in enumerate(fileList):
    if not todoList[J]:
        continue
    
    submitName = "qsub_" + fileList[J] + ".sh"
    submitHeader = """#! /bin/bash
#$ -N %s
#$ -l nodes=1,rpn=1,ppr=16
#$ -l membycore=4G
#$ -cwd
#$ -S /bin/bash
#$ -e %s
#$ -o %s
#$ -j y
#$ -m ea
#$ -pe hybrid.16 16
#$ -q refinement.q

export PATH="/scicore/pfs/scicore-p-structsoft/ctf4:/scicore/pfs/scicore-p-structsoft/anaconda2/bin:$PATH"
export PYTHONPATH="$PYTHONPATH"
source /scicore/soft/UGE/current/util/gepetools/gepetools.sh
""" % ( "q" + fileList[J], fileList[J] + ".zerr" , fileList[J] + ".zout" )

    submitCommand  = "python %s %s\n" % (python_script, fileList[J] ) 
    with open( submitName, 'wb' ) as sub:
        sub.write( submitHeader )
        sub.write( submitCommand  )
    print( "Submitting to grid engine : %s"%submitName )
    
    # Short sleep to not overload the queue
    time.sleep( 0.01 )    
    # Note their is a maximum of 1000 jobs in the queue
    process_list.append( subprocess.Popen( "qsub " + submitName, shell=True ) ) 

# TODO figure out when we are done-done and delete temporaries?
print( "##### DONE qsub-ing #####" )


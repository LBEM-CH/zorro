# -*- coding: utf-8 -*-
"""
Created on Fri Aug  5 09:50:51 2016

@author: rmcleod
"""

import os, os.path, subprocess, sys, shutil
import numpy as np
import numexprz as ne
import time
import psutil
import matplotlib.pyplot as plt
#from matplotlib import collections
import zorro

plt.rcParams['lines.linewidth'] = 1.0
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 16.0

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
    
def countPhysicalProcessors():
    cpuInfo = ne.cpu.info
    physicalIDs = []
    for J, cpuDict in enumerate( cpuInfo ):
        if not cpuDict['physical id'] in physicalIDs:
            physicalIDs.append( cpuDict['physical id'] )
    return len( physicalIDs )

def getMHz( cpuVirtualCores ):

    strMHz = subprocess.check_output( "cat /proc/cpuinfo | grep MHz", shell=True ).split()
    # Every fourth one is a MHz
    cpuMHz = np.zeros(  len(strMHz)/4, dtype='float32' )
    for J in np.arange( 0, cpuVirtualCores ):
        cpuMHz[J] = np.float32( strMHz[4*J + 3 ] )
    return cpuMHz
    
def getCpuUsage( percpu = False):
    # This must be called at the start of a loop to get the initial the values in /proc/stat
    if percpu:
        return psutil.cpu_percent( percpu=True )
    else:
        return psutil.cpu_percent()
    
def getMemUsage():
    mem = psutil.virtual_memory()
    return( [mem.used, mem.percent] )
    
    
def getDiskReadWrite():
    diskIO = psutil.disk_io_counters()
    return( [diskIO.read_time, diskIO.write_time] )
    

class benchZorro( object ):
    
    def __init__(self, maxSamples = 3600, sleepTime = 1.0 ):
        self.maxSamples = maxSamples
        self.sleepTime = sleepTime
        
        self.cpuInfo = ne.cpu.info
        self.cpuVirtualCores = len( self.cpuInfo )
        self.cpuModel = self.cpuInfo[0]['model name']
        self.cpuAdvertisedMHz = 1000.0 * np.float32(self.cpuModel.split('@')[1].rstrip('GHz'))
        self.cpuPowerManagement = self.cpuInfo[0]['power management'] 
        self.cpuFlags = self.cpuInfo[0]['flags'].split() # Look for 'sse', 'sse2', 'avx', 'fma' for FFTW compilation
        self.cpuCacheSize = np.float32( self.cpuInfo[0]['cache size'][:-3] ) # Should be kiloBytes
        self.cpuCoresPerProcessor = np.int(ne.cpu.info[0]['cpu cores'])
        self.cpuPhysicalCores = self.cpuCoresPerProcessor * countPhysicalProcessors()           
        if len(self.cpuInfo) == 2*self.cpuPhysicalCores:
            self.hyperthreading = True
        else:
            self.hyperthreading = False

        self.zorroDefault = zorro.ImageRegistrator()
        self.zorroDefault.diagWidth = 5
        self.zorroDefault.CTFProgram = None
        self.zorroDefault.filterMode = None
        self.zorroDefault.doLazyFRC = False
        self.zorroDefault.savePNG = False

        self.resetStats()
        
    def resetStats(self):
        self.index = 0
        
        self.cpuMHz = np.zeros( [self.maxSamples, self.cpuVirtualCores] )
        self.cpuUsage = np.zeros( [self.maxSamples, self.cpuVirtualCores] )
        self.memUsage = np.zeros( [self.maxSamples, 2] )
        self.rwUsage = np.zeros( [self.maxSamples, 2] )
        self.timeAxis = np.zeros( [self.maxSamples] )


    def updateStats(self):
        # sys.stdout.write( "\r >> Bench %.1f" % (100.0*index/Ntest) + " %" ); sys.stdout.flush()
        self.cpuMHz[self.index,:] = getMHz( self.cpuVirtualCores )
        self.cpuUsage[self.index,:] = getCpuUsage( percpu = True )
        self.memUsage[self.index,:] = getMemUsage()
        self.rwUsage[self.index,:] = getDiskReadWrite()
        self.timeAxis[self.index] = time.time()
        self.index += 1
        
    def finishStats(self):
        self.index -= 1
        self.timeAxis = self.timeAxis[:self.index+1]
        self.cpuMHz = self.cpuMHz[:self.index+1,:]
        self.cpuUsage = self.cpuUsage[:self.index+1,:]
        self.memUsage = self.memUsage[:self.index+1,:]
        self.rwUsage = self.rwUsage[:self.index+2,:]
        
        self.timeAxis -= self.timeAxis[0]
        self.cpuMHz_all_percent = np.mean( self.cpuMHz, axis=1 ) / self.cpuAdvertisedMHz * 100.0
        self.cpuUsage_all = np.mean( self.cpuUsage, axis=1 ) 

        # How to get % ioUsage?  Just use normalize?
        self.ioUsage = 100.0* normalize( np.sum( np.diff( self.rwUsage, axis=0 ).astype('float32'), axis=1 )[:self.index] )

    def plotStats(self, N_processes, N_threads ):
        
        fileExt = self.cpuModel.replace(" ","") + "_Nproc%d_Nthread%d.png" % (N_processes, N_threads)
        plt.figure( figsize=(12,10) )
        plt.plot( self.timeAxis, self.cpuMHz_all_percent, label = "CPU throttle", color='purple', linewidth=1.5 )
        plt.plot( self.timeAxis, self.cpuUsage_all, label = "CPU usage", color='steelblue', linewidth=1.5 )
        plt.plot( self.timeAxis, self.memUsage[:,1], label = "Memory usage", color='firebrick', linewidth=1.5 )
        plt.plot( self.timeAxis[:-1], self.ioUsage, label="Disk IO (norm)", color='forestgreen', linewidth=1.5 )
        plt.xlabel( "Time (s)" )
        plt.ylabel( "Performance metrics (%)" )
        plt.legend( loc='best' )
        plt.title( "Benchmark for %s" % self.cpuModel + "\n $N_{processes}=%d, N_{threads}=%d$" %( N_processes, N_threads) )
        # plt.ylim( [0, 140] )
        plt.xlim( [0, self.timeAxis[-1]] )
        plt.savefig( "Benchmark_" + fileExt )
        
        ##### Make a waterfall plot of CPU usage per processor

        waterfallColors =  plt.cm.gnuplot( np.linspace(0.0,1.0,self.cpuVirtualCores+1) ) 
        
        # http://matplotlib.org/examples/api/collections_demo.html
        
        cumsum_cpu = np.cumsum( self.cpuUsage, axis=1 )
        cumsum_cpu = np.hstack( [np.zeros([cumsum_cpu.shape[0], 1]), cumsum_cpu])
        
        plt.figure( figsize=(12,10) )
        for J in np.arange(1,self.cpuVirtualCores+1):
            #plt.plot( timeAxis, cumsum_cpu[:,J], color=waterfallColors[J] )
            plt.fill_between( self.timeAxis, cumsum_cpu[:,J-1], cumsum_cpu[:,J], facecolor=waterfallColors[J], color=[0.0,0.0,0.0,0.3], linewidth=0.5, interpolate=True )
        plt.xlim( [0, self.timeAxis[-1]] )
        plt.xlabel( "Time (s)" )
        plt.ylabel( "CPU utilization (%)" )
        plt.title( "per CPU utilization for %s" % self.cpuModel  + "\n $N_{processes}=%d, N_{threads}=%d$" %( N_processes, N_threads) )
        plt.savefig( "perCPUBenchmark_" + fileExt )
        
    def __str__(self):
            returnstr = "##### CPU INFO #####\n" 
            returnstr += "Model: %s\n" % self.cpuModel
            returnstr += "Power management scheme: %s\n" % self.cpuPowerManagement
            returnstr +=  "Cache size: %s\n" % self.cpuInfo[0]['cache size'] 
            returnstr +=  "Hyperthreading: %s\n" % self.hyperthreading 
            returnstr +=  "No. Physical Cores: %d\n" % self.cpuPhysicalCores 
            return returnstr

    def benchmark( self, stackName, N_processes, N_threads ):
        #dirName = os.path.dirname( stackName[0] )
        #baseName = os.path.basename( stackName[0] )
        # We need to make 3 copies of the stack  
        stackFront, stackExt = os.path.splitext( stackName )
        stackName = [stackName]
    
        N_cases = len( N_processes )
        t_start = np.zeros( N_cases )
        t_finish = np.zeros( N_cases )
        maxMemory = np.zeros( N_cases )
        meanCPUusage = np.zeros( N_cases )
        
        if N_cases > 1:
            # Force use of Agg if we are generating many plots.
            plt.switch_backend( 'Agg' )
        
        # Make copies of the input file to avoid file IO collisions
        for J in np.arange( 1, np.max(N_processes) ):
            newName = stackFront + str(J) + stackExt
            stackName.append( newName )
            print( "Copying %s to %s" % (stackName[0], stackName[J]) )
            if not os.path.isfile( newName ):
                shutil.copy( stackName[0], newName )
            pass
    
        for K in range(N_cases):
            print( "##### STARTING BENCHMARK #%d, N_PROCS = %d, N_THREADS = %d" %(K, N_processes[K], N_threads[K] ) )
            self.zorroDefault.n_threads = N_threads[K]
            ProcList = []
            self.resetStats()
            
            for J in range( N_processes[K] ):
                self.zorroDefault.files['stack'] = stackName[0]
                self.zorroDefault.saveConfig( "stack%d.zor"%J )
    
            t_start[K] = time.time()
            self.updateStats()
            
            # Start all the processes
            for J in range( N_processes[K] ):
                ProcList.append( subprocess.Popen( "zorro -c stack%d.zor"%J, shell=True ) )
            
            # Poll the processes and also call our stats
            finished = np.zeros( len(ProcList), dtype='bool' )
            while self.index < self.maxSamples:
                self.updateStats()
                
                for I, P in enumerate(ProcList):
                    finished[I] = P.poll() != None
                    
                if np.all( finished ):
                    print( "Finished benchmark for N_processes: %d, N_threads: %d" % (N_processes[K], N_threads[K]))
                    break
                time.sleep( self.sleepTime )
                        
            t_finish[K] = time.time()
            self.finishStats()
            
            self.plotStats( N_processes[K], N_threads[K] )
            maxMemory[K] = np.max( self.memUsage[:,0] ) / 2**30 # GB
            meanCPUusage[K] = np.mean( np.sum( bencher.cpuUsage, axis=1 ) ) # 
        
        t_consumed_per = (t_finish - t_start)/ N_processes
        print( self.__str__() )
        for K in range(N_cases):
            
            print( "Case %d: %d processes, %d threads each, time per stack: %.3f s, CPU usage: %.2f, maximum Memory %.2f GB" 
                %( K, N_processes[K], N_threads[K], t_consumed_per[K], meanCPUusage[K], maxMemory[K] )   )
            
        # Save a simple output file
        np.max(self.memUsage[:,0])
        np.savetxt( "bench"+self.cpuModel.replace(" ","")+".txt", 
                   np.vstack( [N_processes, N_threads, t_consumed_per, meanCPUusage, maxMemory ] ).transpose(), fmt="%.2f",
                   header = "Benchmark for %s \n N_processes  |  N_threads  | time_consumed_per_process | meanCPU | max Memory" % self.cpuModel )
            

    
if __name__ == "__main__":
    bencher = benchZorro()
    
    try:
        stackName = sys.argv[1]
    except IndexError:
        print( "Usage: 'python zorro_benchmark someStack.dm4'" )
        exit(1)
    
    # Here's an example of setting up likely situations for processing, with a maximum of 4 processes
    # This test mostly shows that hyperthreading makes Zorro slower, because the calculations are already
    # block-optimized.
    n_cases = 8
    n_procs = [1,2,3,4,1,2,3,4]
    max_threads = bencher.cpuVirtualCores
    n_threads = [max_threads, max_threads/2, max_threads/3, max_threads/4, max_threads/2, max_threads/4, 
                 max_threads/6, max_threads/8]
    
    bencher.benchmark( stackName, n_procs, n_threads )
    
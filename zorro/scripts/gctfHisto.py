# -*- coding: utf-8 -*-
from zorro import ReliablePy
import numpy as np
import matplotlib.pyplot as plt

rln = ReliablePy.ReliablePy()

defocusThreshold = 40000
astigThreshold = 800
fomThreshold = 0.0
resThreshold = 6.0

rln.load( "micrographs_all_gctf.star" )

defocusU = rln.star['data_']['DefocusU']
defocusV = rln.star['data_']['DefocusV']
finalResolution = rln.star['data_']['FinalResolution']
ctfFoM = rln.star['data_']['CtfFigureOfMerit']

defocusMean = 0.5 * defocusU + 0.5 * defocusV
astig = np.abs( defocusU - defocusV )

[hDefocus, cDefocus] = np.histogram( defocusMean,  
    bins=np.arange(np.min(defocusMean),np.max(defocusMean),1000.0) )
hDefocus = hDefocus.astype('float32')
cDefocus = cDefocus[:-1] +1000.0/2

[hAstig, cAstig] = np.histogram( astig, 
    bins=np.arange(0, np.max(astig), 100.0) )
hAstig = hAstig.astype('float32')
cAstig = cAstig[:-1] +100.0/2

[hFoM, cFoM] = np.histogram( ctfFoM,  
    bins=np.arange(0.0,np.max(ctfFoM),0.005) )
hFoM = hFoM.astype('float32')
cFoM = cFoM[:-1] +0.005/2.0

[hRes, cRes] = np.histogram( finalResolution,  
    bins=np.arange(np.min(finalResolution),np.max(finalResolution),0.25) )
hRes = hRes.astype('float32')
cRes = cRes[:-1] +0.25/2.0

plt.figure()
plt.fill_between( cDefocus, hDefocus, np.zeros(len(hDefocus)), facecolor='steelblue', alpha=0.5 )
plt.plot( [defocusThreshold, defocusThreshold], [0, np.max(hDefocus)], "--", color='firebrick' )
plt.xlabel( "Defocus, $C_1 (\AA)$" )
plt.ylabel( "Histogram counts" )

plt.figure()
plt.fill_between( cAstig, hAstig, np.zeros(len(hAstig)), facecolor='forestgreen', alpha=0.5 )
plt.plot( [astigThreshold, astigThreshold], [0, np.max(hAstig)], "--", color='firebrick' )
plt.xlabel( "Astigmatism, $A_1 (\AA)$" )
plt.ylabel( "Histogram counts" )

plt.figure()
plt.fill_between( cFoM, hFoM, np.zeros(len(hFoM)), facecolor='darkorange', alpha=0.5 )
plt.plot( [fomThreshold, fomThreshold], [0, np.max(hFoM)], "--", color='firebrick' )
plt.xlabel( "Figure of Merit, $R^2$" )
plt.ylabel( "Histogram counts" )

plt.figure()
plt.fill_between( cRes, hRes, np.zeros(len(hRes)), facecolor='purple', alpha=0.5 )
plt.plot( [resThreshold, resThreshold], [0, np.max(hRes)], "--", color='firebrick' )
plt.xlabel( "Fitted Resolution, $r (\AA)$" )
plt.ylabel( "Histogram counts" )


#keepIndices = np.ones( len(defocusU), dtype='bool' )
keepIndices = ( ( defocusMean < defocusThreshold) & (astig < astigThreshold) &
                (ctfFoM > fomThreshold ) & (finalResolution < resThreshold) )

print( "KEEPING %d of %d micrographs" %(np.sum(keepIndices), defocusU.size) )

for key in rln.star['data_']:
    rln.star['data_'][key] =  rln.star['data_'][key][keepIndices]

rln.saveDataStar( "micrographs_pruned_gctf.star" )
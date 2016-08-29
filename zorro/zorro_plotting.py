# -*- coding: utf-8 -*-
"""
Created on Thu Mar  3 15:28:53 2016
@author: Robert A. McLeod
@email: robert.mcleod@unibas.ch

Zorro plotting is designed for both command-line production of .PNG plots as subprocesses, so many plots can 
be generated asynchronously, and for generating Qt4 plots to an MplCanvas object.

Also currently features the ims() interactive plotter, which I intend to merge into zorroPlot/MplCanvas in 
the future.
"""

import matplotlib.figure
import itertools
import collections
import numpy as np
import matplotlib.offsetbox 

# TODO: merge ims() functionality into zorroPlot
import matplotlib.pyplot as plt
import matplotlib.colors as col
import scipy.ndimage as nd
import zorro
import os, os.path

##################################################################################
######################## Object-oriented interface ###############################
##################################################################################
class zorroPlot(object):
    
    def __init__(self, filename=None, width=7, height=7, dpi=144, facecolor=[0.75,0.75,0.75,1.0], 
                 MplCanvas = None, backend=u'Qt4Agg' ):
        """
        Object-oriented plotting interface for Zorro.
        """
        # All parameters are stored in a hash-dictionary
        self.plotDict = {}
        self.plotDict[u'width'] = width
        self.plotDict[u'height'] = height
        self.plotDict[u'dpi'] = dpi
        self.plotDict[u'facecolor'] = facecolor
        
        if bool(filename):
            print( "TODO: load and display file from zorroPlot.__init__()" )
        
        # http://stackoverflow.com/questions/13714454/specifying-and-saving-a-figure-with-exact-size-in-pixels
        self.fig = matplotlib.figure.Figure(figsize=(width, height), facecolor=facecolor, dpi=dpi )
        # This forces the plot window to cover the entire space by default
        self.axes = self.fig.add_axes( [0.0,0.0,1.0,1.0] )
        self.axes.hold(False) # We want the axes cleared every time plot() is called
        self.axes2 = None
        
        
        self.cmaps_cycle = itertools.cycle( [u"gray", u"gnuplot", u"jet", u"nipy_spectral"] )
        self.plotDict[u'image_cmap'] = next( self.cmaps_cycle ) # Pre-cycle once...
        self.plotDict[u'graph_cmap'] = u"gnuplot"
        self.plotDict[u'showBoxes'] = False # Try to load imageSum_boxMask.png as an overlay
        self.plotDict[u'colorbar'] = True
        
        if bool( MplCanvas ):
            # Avoid calling anything that would require importing PySide here, as we don't want it as an 
            # explicit dependancy.
            self.canvas = MplCanvas
        else:
            if backend.lower() == u'agg': # CANNOT RENDER TO SCREEN, PRINTING ONLY
                from matplotlib.backends.backend_agg import FigureCanvas
            elif backend.lower() == u'qt4' or backend.lower() == u'qt4agg': 
                from matplotlib.backends.backend_qt4agg import FigureCanvas
            elif backend.lower() == u'qt5' or backend.lower() == u'qt5agg':
                from matplotlib.backends.backend_qt5agg import FigureCanvas    
            else: # default is qt4agg
                from matplotlib.backends.backend_qt4agg import FigureCanvas   
                
            self.canvas = FigureCanvas( self.fig )
            
            try: self.canvas.updateGeometry()
            except: pass
            
        pass


    def updateCanvas( self ):
        """
        Updates a (Qt4Agg) FigureCanvas.  Typically an automator.MplCanvas type.
        """
        try: self.canvas.updateGeometry()
        except: pass
        #self.canvas.draw() # Necessary with show?
        self.canvas.show()
    
    def printAndReturn( self ):
        """
        Any following commands shared amongst all plot functions go here for brevity.
        """
        if 'title' in self.plotDict: 
            self.axes.set_title( self.plotDict['title'] )
            
        try: self.canvas.updateGeometry()
        except: pass
        if u'plotFile' in self.plotDict and bool( self.plotDict['plotFile'] ):
            if self.plotDict[u'Transparent']:
                color = [0,0,0,0]
            else: 
                color = [1,1,1,1]

            self.canvas.print_figure( self.plotDict[u'plotFile'], dpi=self.plotDict[u'dpi'], 
                                     facecolor=color, edgecolor=color ) 
            return self.plotDict[u'plotFile']
           
    def plotEmpty( self ):
        self.fig.clear()
        self.axes = self.fig.add_axes( [0.0, 0.0, 1.0, 1.0] )
        self.axes.hold(False)
        self.axes.plot( [0.0, 1.0], [0.0,1.0], 'k-' )
        self.axes.hold(True)
        self.axes.plot( [0.0, 1.0], [1.0,0.0], 'k-' )
        self.axes.text( 0.45, 0.25, "No data", fontsize=18 )
        self.axes.hold(False)
        self.axes.set_axis_off()
        
    def plotPixmap( self ):
        self.fig.clear()
        self.axes = self.fig.add_axes( [0.0, 0.0, 1.0, 1.0] )
        self.axes.hold(True)
        if u'pixmap' in self.plotDict:
            mage = self.axes.imshow( self.plotDict[u'pixmap'], interpolation='sinc' )
            self.axes.set_axis_off()
            if u'boxMask' in self.plotDict and np.any(self.plotDict[u'boxMask']):
                print( "pixmap boxes" )
                #scaleDiff = np.array( self.plotDict['pixmap'].shape ) / np.array( self.plotDict['boxMask'].shape )

                self.axes.imshow( self.plotDict[u'boxMask'], extent=mage.get_extent() )
            
        else:
            print( "No pixmap" )
  
        self.axes.hold(False)
        
    def plotImage( self ):
        self.fig.clear()
        self.axes = self.fig.add_axes( [0.0, 0.0, 1.0, 1.0] )

    
        clim = zorro.util.histClim( self.plotDict['image'], cutoff=1E-4 )
        self.axes.hold(True)
        mage = self.axes.imshow( self.plotDict['image'], vmin=clim[0], vmax=clim[1], interpolation='nearest', 
                         cmap=self.plotDict['image_cmap'] )


        
        if 'pixelsize' in self.plotDict:
            zorro.util.plotScalebar( mage, self.plotDict['pixelsize'] )
        if bool(self.plotDict['colorbar']):
            self.fig.colorbar( mage, fraction=0.046, pad=0.04)
        self.axes.set_axis_off()
        self.axes.hold(False)
        
        return self.printAndReturn()
        
    def plotStack( self ):
        print( "TODO: implement plotStack" )
        
    def plotFFT( self ):
        self.fig.clear()
        self.axes = self.fig.add_axes( [0.0, 0.0, 1.0, 1.0] )
        self.axes.hold(False)


        FFTimage = np.log10( 1.0 + np.abs( np.fft.fftshift( np.fft.fft2( self.plotDict['image'] ))))
        FFTclim = zorro.util.ciClim( FFTimage, sigma=1.5 )
        mage = self.axes.imshow( FFTimage, interpolation='bicubic', vmin=FFTclim[0], vmax=FFTclim[1], 
                         cmap=self.plotDict['image_cmap'] )
        if 'pixelsize' in self.plotDict:
            inv_ps = 1.0 / (FFTimage.shape[0] * self.plotDict['pixelsize'] )
            zorro.util.plotScalebar( mage, inv_ps, units=u'nm^{-1}' )
        self.axes.set_axis_off()
        if bool(self.plotDict['colorbar']):
            self.fig.colorbar( mage, fraction=0.046, pad=0.04)
        return self.printAndReturn()
                             
    def plotPolarFFT( self ):
        self.fig.clear()
        self.axes = self.fig.add_axes( [0.0, 0.0, 1.0, 1.0] )
        self.axes.hold(False)
        
        polarFFTimage = zorro.util.img2polar( np.log10( 1.0 + np.abs( np.fft.fftshift( np.fft.fft2( self.plotDict['image'] )))) )
        FFTclim = zorro.util.ciClim( polarFFTimage, sigma=1.5 )
        mage = self.axes.imshow( polarFFTimage, interpolation='bicubic', vmin=FFTclim[0], vmax=FFTclim[1], 
                         cmap=self.plotDict['image_cmap'] )
        if 'pixlsize' in self.plotDict:
            # Egh, this scalebar is sort of wrong, maybe I should transpose the plot?
            inv_ps = 1.0 / (polarFFTimage.shape[0] * self.plotDict['pixelsize'] )
            zorro.util.plotScalebar( mage, inv_ps, units=u'nm^{-1}' )
        self.axes.set_axis_off()
        if bool(self.plotDict['colorbar']):
            self.fig.colorbar( mage, fraction=0.046, pad=0.04)
            
        return self.printAndReturn()                          
    # TODO: render Gautoauto outputs?  Maybe I should make the Gautomatch boxes seperately as a largely 
    # transparent plot, and just add it on top or not?
    
    def plotCorrTriMat( self ):
        self.fig.clear()
        self.axes = self.fig.add_axes( [0.0, 0.0, 1.0, 1.0] )
        self.axes.hold(False)
        
        corrtri = self.plotDict['corrTriMat']
        clim = [np.min(corrtri[corrtri>0.0])*0.75, np.max(corrtri[corrtri>0.0])]
        corrmap = self.axes.imshow( corrtri, interpolation='nearest', vmin=clim[0], vmax=clim[1], cmap=self.plotDict['graph_cmap'] )
        self.axes.set_xlabel( "Base image" )
        self.axes.set_ylabel( "Template image" )
        if bool(self.plotDict['colorbar']):
            self.fig.colorbar( corrmap, fraction=0.046, pad=0.04)

        return self.printAndReturn()
            
    def plotPeaksigTriMat( self ):
        self.fig.clear()
        self.axes = self.fig.add_axes( [0.0, 0.0, 1.0, 1.0] )
        self.axes.hold(False)
        
        peaksig = self.plotDict['peaksigTriMat']
        clim = [np.min(peaksig[peaksig>0.0])*0.75, np.max(peaksig[peaksig>0.0])]
        psmap = self.axes.imshow( peaksig, interpolation='nearest', vmin=clim[0], vmax=clim[1], cmap=self.plotDict['graph_cmap'] )
        self.axes.set_xlabel( "Base image" )
        self.axes.set_ylabel( "Template image" )
        if bool(self.plotDict['colorbar']):
            self.fig.colorbar( psmap, fraction=0.046, pad=0.04)
        
        return self.printAndReturn()                                  
    
    def plotTranslations( self ):
        # rect is [left,bottom,width,height]
        self.fig.clear()
        self.axes = self.fig.add_axes( [0.12, 0.1, 0.85, 0.85] )
        self.axes.hold(False)

        
        if 'errorX' in self.plotDict:
            self.axes.errorbar( self.plotDict['translations'][:,1], self.plotDict['translations'][:,0], fmt='k-',
                              xerr=self.plotDict['errorX'], yerr=self.plotDict['errorY'] )
        else:
            self.axes.plot( self.plotDict['translations'][:,1], self.plotDict['translations'][:,0], 'k.-',
                   linewidth=2.0, markersize=16 )
                       
        self.axes.set_xlabel( 'X-axis drift (pix)' )
        self.axes.set_ylabel( 'Y-axis drift (pix)' )
        self.axes.axis('equal')
        
        return self.printAndReturn()       
                             
    def plotPixRegError( self ):
        self.fig.clear()
        self.axes = self.fig.add_subplot( 211 )
        self.axes.hold(False)
        self.axes2 = self.fig.add_subplot( 212 )
        self.axes2.hold(False)
        
        errorX = np.abs( self.plotDict['errorXY'][:,1] )
        errorY = np.abs( self.plotDict['errorXY'][:,0] )
        
        meanErrX = np.mean( errorX )
        meanErrY = np.mean( errorY )
        stdErrX = np.std( errorX )
        stdErrY = np.std( errorY)
        
        self.axes.semilogy( errorX, '.:', linewidth=1.5, color='black', markersize=12, markerfacecolor='darkslateblue',
                       label='X: %.3f +/- %.3f pix'%(meanErrX, stdErrX) )
        self.axes.legend( fontsize=12, loc='best' )
        self.axes.set_ylabel( "X-error estimate (pix)" )
        
        # self.axes.set_title( 'X: %f +/- %f'%(meanErrX, stdErrX) )
        self.axes2.semilogy( errorY, '.:', linewidth=1.5, color='black', markersize=12, markerfacecolor='darkolivegreen',
                        label='Y: %.3f +/- %.3f pix'%(meanErrY, stdErrY) )
        #self.axes2.set_title( 'Y: %f +/- %f pix'%(meanErrY, stdErrY) )
        self.axes2.legend( fontsize=12, loc='best' )
        self.axes2.set_xlabel( "Equation number" )
        self.axes2.set_ylabel( "Y-error estimate (pix)" )
        
        return self.printAndReturn()       
                             
    def plotLogisticWeights( self ):
        self.fig.clear()
        self.axes = self.fig.add_axes( [0.12, 0.1, 0.80, 0.85] )
        self.axes.hold(False)
        
        pixError = np.sqrt( self.plotDict['errorXY'][:,0]**2 + self.plotDict['errorXY'][:,1]**2 )
        peaksigVect = self.plotDict['peaksigVect']
        
        # Mixing a log-plot with a linear-plot in a plotyy style.
        self.axes.semilogy( peaksigVect, pixError, 'k.' ) 
        # ax1.plot( peaksigVect, pixError, 'k.' )
        self.axes.set_xlabel( 'Correlation peak significance, $\sigma$' )
        self.axes.set_ylabel( 'Pixel registration error' )
        self.axes.set_ylim( [0,1] )
        self.axes.set_ylim( [1E-2, 1E2] )
        self.axes.set_xlim( peaksigVect.min(), peaksigVect.max() )
    
        if 'peaksigThres' in self.plotDict:
            # Twinx not working with custom sizes?
            self.axes2 = self.axes.twinx()
            self.fig.add_axes( self.axes2 )
            # Plot threshold sigma value
            self.axes2.plot( [self.plotDict['peaksigThres'], self.plotDict['peaksigThres']], [0.0, 1.0], '--', 
                            color='firebrick', label=r'$\sigma_{thres} = %.2f$'%self.plotDict['peaksigThres'] ) 
            
            # Plot the logistics curve
            peakSig = np.arange( np.min(peaksigVect), np.max(peaksigVect), 0.05 )
                    
            weights = zorro.util.logistic( peakSig, self.plotDict['peaksigThres'], self.plotDict['logisticK'], self.plotDict['logisticNu'] )         
            self.axes2.plot( peakSig, weights, 
                                label=r"Weights $K=%.2f$, $\nu=%.3f$"%( self.plotDict['logisticK'], self.plotDict['logisticNu']), color='royalblue' ) 
                    
            if 'cdfPeaks' in self.plotDict:
                self.axes2.plot( self.plotDict['hSigma'], self.plotDict['cdfPeaks'], '+', label = r'$\sigma-$CDF', color='slateblue' )

        lines1, labels1 = self.axes.get_legend_handles_labels()
        if bool( self.axes2 ):
            lines2, labels2 = self.axes2.get_legend_handles_labels()
            self.axes2.legend( lines1 + lines2, labels1 + labels2, loc='best', fontsize=14 )
        else:
            self.axes.legend( lines1, labels1, loc='best', fontsize=14 )
        
        return self.printAndReturn()       


    def plotFRC( self ):
        self.fig.clear()
        self.axes = self.fig.add_axes( [0.12, 0.1, 0.85, 0.85] )
        self.axes.hold(False)
        
        if not np.any(self.plotDict['FRC']):
            print( "Warning, zorro_plotting: FRC is empty" )
            return
            
        FRC = self.plotDict['FRC']
        
        inv_ps = 1.0 / (2.0* FRC.size *self.plotDict['pixelsize'] )
        freqAxis = np.arange( FRC.size ) * inv_ps
        
        # This is really ugly curve fitting here
        #splineFRC = UnivariateSpline( freqAxis, FRC, s = 2.0 )
        #splineAxis = np.linspace( freqAxis.min(), freqAxis.max(), 2048 )
        # Maybe try fitting to a defocus OTF, it might be faster than the spline fitting.
        
        
        
        self.axes.hold(True)
        #self.axes.plot( splineAxis, splineFRC(splineAxis), 'r-' )
        self.axes.plot( freqAxis, FRC, color='firebrick', marker='.', 
                       markerfacecolor='k', markeredgecolor='k', label=self.plotDict['labelText'] )
        self.axes.set_xlabel( r"Spatial frequency, $q$ ($nm^{-1}$)" )
        self.axes.set_xlim( [freqAxis.min(), freqAxis.max()] )
        self.axes.set_ylabel( "Fourier ring correlation" )
        self.axes.legend( loc='best' )
        self.axes.hold(False)
        
        return self.printAndReturn()
            
    def plotCTFDiag( self ):
        self.fig.clear()
        self.axes = self.fig.add_axes( [0.0, 0.0, 1.0, 1.0] )
        self.axes.hold(False)
        
        #print( "DEBUG: CTF4Diag shape = " + str(self.plotDict['CTF4Diag'].shape) )
        #print( "DEBUG: CTF4Diag dtype = " + str(self.plotDict['CTF4Diag'].dtype) )
        
        CTFInfo = self.plotDict['CTFInfo']
        try:
            mapCTF = self.axes.imshow( self.plotDict['CTFDiag'], cmap=self.plotDict['image_cmap'] )
        except:
            print( "WARNING: Could not render CTF Diagnostic image, TODO: switch to disk version"  )
            # print( " CTFDiag.shape = " + str( self.plotDict['CTFDiag'].shape ) + ", dtype = " + str( self.plotDict['CTFDiag'].dtype) )
            # Try the dead version instead?  I need checks in the plotting functions to see if the data 
            # exists and if not nicely switch to live/dead
            return
        
        if 'pixelsize' in self.plotDict:
            inv_ps = 1.0 / (self.plotDict['CTFDiag'].shape[0] * self.plotDict['pixelsize'] )
            zorro.util.plotScalebar( mapCTF, inv_ps, units=u'nm^{-1}' )
            
        if 'title' in self.plotDict: 
            self.title = self.plotDict['title']
        
        results = (u"$DF_1:\/%.1f\/\AA$\n"%CTFInfo['DefocusU'] +
                     u"$DF_2:\/%.1f\/\AA$\n"%CTFInfo['DefocusV'] +
                     u"$\gamma:\/%.1f^\circ$\n"%CTFInfo['DefocusAngle']+
                     u"$R:\/%.3f$\n"%CTFInfo['CtfFigureOfMerit'] +
                     u"$Fit\/res:\/%.1f\/\AA$"%CTFInfo['FinalResolution'] )

        infobox = matplotlib.offsetbox.AnchoredText( results, pad=0.5, loc=1, prop={'size':13} )
        self.axes.add_artist( infobox )
        
        self.axes.set_axis_off() # This is still not cropping properly...
        
        return self.printAndReturn() 
    
    def plotStats( self ):
        # Setup unicode statistics dictionary
        #matplotlib.rc('font', family='DejaVu Sans')
        
        statsDict = collections.OrderedDict()
        if 'pixlsize' in self.plotDict:
            statsDict[u'Pixel size (nm):'] = "%.4f"%self.plotDict['pixelsize']
        if 'voltage' in self.plotDict:
            statsDict[u'Accelerating voltage (kV):'] = "%.1f"%self.plotDict['voltage']
        if 'C3' in self.plotDict:
            statsDict[u'Spherical aberration, C3 (mm):'] = "%.1f"%self.plotDict['C3']

        if 'meanPeaksig' in self.plotDict:
            statsDict[u'Peak significance:'] = u"%.2f"%self.plotDict['meanPeaksig'] + u" ± %.2f"%self.plotDict['stdPeaksig']

        try: 
            CTFInfo = self.plotDict['CTFInfo']
            statsDict[u'CTF defocus #1 (Å):'] = "%.1f"%CTFInfo['DefocusU']
            statsDict[u'CTF defocus #2 (Å):'] = "%.1f"%CTFInfo['DefocusV']
            statsDict[u'CTF gamma (°):'] = "%.4f"%CTFInfo['DefocusAngle']
            statsDict[u'CTF correlation coefficient :'] = "%.5f"%CTFInfo['CtfFigureOfMerit']
            statsDict[u'CTF maximum fit frequency (Å) :'] = "%.1f"%CTFInfo['FinalResolution']
        except: 
            pass

        # Print the statistical metrics
        self.fig.clear()
        self.axes.get_xaxis().set_visible(False)
        self.axes.get_yaxis().set_visible(False)
        fontsize = 12
        fontfigspacing =  float(fontsize*1.5) / (self.fig.dpi * self.fig.get_size_inches()[1])
        keycount = 0
        for key, value in statsDict.items():
            self.fig.text( fontfigspacing, 1 - (1+keycount)*fontfigspacing, key, size=fontsize )
            self.fig.text( 0.5+fontfigspacing, 1 - (1+keycount)*fontfigspacing, value, size=fontsize )
            keycount += 1

        return self.printAndReturn()
    
    
##################################################################################
#################### Static interface for multiprocessing ##########################
##################################################################################
# Pickle can't serialize Python objects well enough to launch functions of a 
# class in a multiprocessing pool, so we need to call a static function to do the 
# branching.
                          

def generate( params ):
    """
    Maybe the multiprocessing pool should be here, completely outside of Zorro...
    """
    plotType = params[0]
    plotDict = params[1]
    
    if 'show' in plotDict and bool( plotDict['show'] ):
        print( "zorro_plotting.generate(): Cannot render plots to screen from multiprocessing module." )
        plotDict['show'] = False
        
    # Maybe a dictionary that likes plotType to a function is smarter?  I don't know if we can if it's not 
    # been instantiated.
    daPlot = zorroPlot( backend=plotDict['backend'] )
    daPlot.plotDict = plotDict # Override -- this can avoid some passing-by-value
    if plotType == 'translations':
        return daPlot.plotTranslations()
    elif plotType == 'pixRegError':
        return daPlot.plotPixRegError()
    elif plotType == 'image':
        return daPlot.plotImage()
    elif plotType == 'FFT':
        return daPlot.plotFFT()
    elif plotType == 'polarFFT':
        return daPlot.plotPolarFFT()    
    elif plotType == 'stats':
        return daPlot.plotStats()
    elif plotType == 'peaksigTriMat':
        return daPlot.plotPeaksigTriMat()
    elif plotType == 'logisticWeights':
        return daPlot.plotLogisticWeights()
    elif plotType == 'lazyFRC':
        return daPlot.plotFRC()
    elif plotType == 'CTFDiag':
        return daPlot.plotCTFDiag()
    elif plotType == 'corrTriMat':
        return daPlot.plotCorrTriMat()
 

class ims:
    """
    plotting.ims(image)
    
    Shows individual frames in the 3D image (dimensions organized as [z,x,y]). 
    "n" next frame, ("N" next by step of 10)
    "p" previous frame, ("P" previous by step of 10)
    "l" toogles the log scale. 
    "c" swithces between 'jet','grey','hot' and 'hsv' colormaps. 
    "h" turns on histogram-based contrast limits
    "i" zooms in
    "o" zooms out
    "arrows" move the frame around
    "r" resets the position to the center of the frame
    "g" toogle local/global stretch
    "q/Q" increase / decrease the lower limit of the contrast
    "w/W" increase / decrease the upper limit of the contrast
    "R" reset contrast to default
    "S" shows sum projection
    "M" shows max projection
    "V" shows var projection    
    "T" prints statistics
    
    Works with qt backend -> start ipython as: "ipyton --matplotlib qt" or do e.g.: "matplotlib.use('Qtg4Agg')"

    The core of the class taken from http://stackoverflow.com/questions/6620979/2d-slice-series-of-3d-array-in-numpy
    """
    plt.rcParams['keymap.yscale'] = '' # to disable the binding of the key 'l'
    plt.rcParams['keymap.pan'] = '' # to disable the binding of the key 'p'    
    plt.rcParams['keymap.grid'] = '' # to disable the binding of the key 'g'        
    plt.rcParams['keymap.zoom'] = '' # to disable the binding of the key 'o'            
    def __init__(self, im, i=0, titles=None, cutoff = None ):
        plt.ion()
        
        if isinstance( im, str ):
            # Try to load MRC or DM4 files
            file_ext = os.path.splitext( im )[1]
            if file_ext.lower() == ".mrc" or file_ext.lower() == ".mrcs":
                im = zorro.ioMRC.MRCImport( im )
            elif file_ext.lower() == ".dm4":
                dm4struct = zorro.ioDM.DM4Import( im )
                im = dm4struct.im[1].imageData
                del dm4struct
            else:
                print( "Filename has unknown/unimplemented file type: " + im )
                return
                
        if isinstance(im, tuple) or isinstance(im, list):
            # Gawd tuples are annoyingly poorly typedefed
            self.im = np.dstack( im )
            
            print( "shape of tupled array: " + str(self.im.shape) )
            # Don't even bother checking, the complex representation needs to be re-written anyway
            self.complex = False
        elif im.ndim is 2:
            #if plt.iscomplex(im).any():
            if isinstance(im.flatten()[0],(complex,np.complexfloating)):
                self.complex = True
                self.im = np.dstack([np.abs(im),np.angle(im)])                
            else:                
                self.complex = False
                self.im = im
        elif im.ndim is 3:
            
            if np.iscomplex( im ).any():
                im = np.abs(im)
            self.complex = False
            self.im = np.dstack(im) # first dimension is the index of the stack
#            if im.shape[0]<im.shape[2]:
#                self.im = np.dstack(im)
#            else:
#                self.im = im
#        elif isinstance( im, tables.carray.CArray ):
#            # Code to handle PyTables type stack efficiently
#            self.im = im
#            if( np.iscomplex(self.im[:,:,0].any() ) ):
#                self.complex = False # Force to be absolute value below
#                # The upper level pytable is not a numpy array so accessing it like one is _very_ slow
#                for J in np.arange(0,im.shape[2]):
#                    self.im[:,:,J] = np.abs( self.im[:,:,J] )
#            else:
#                self.complex = False
            
        self.dtype = self.im.dtype
        self.i = i 
        self.titles = titles
        if cutoff is None:
            self.histogramClimMode = False
        else:
            self.histogramClimMode = True
            self.cutoff = cutoff
            
        self.logon = False
        self.cmap = 0
        self.projToggle = 0
        self.zoom = 1
        self.globalScale = 0
        self.offx,self.offy = 0,0
        self.stepXY = 24 # step of the movement up-down, left-right

        # RAM This is terribly inefficient to call abs first...
        # self.vmin,self.vmax = np.abs(im).min(),np.abs(im).max()
        
        # RAM: I don't know what this is doing...
        # So remove it
        #t0 = time.time()
        #fim = np.log10(self.im.flatten())
        #t1 = time.time()
        #print "Time to call np.log on entire set (s): " + str( t1 - t0 )
        #if all(fim==-np.inf): # this is for the zero image
        #    self.vminLog,self.vmaxLog=-np.inf,-np.inf
        #elif all(fim==np.inf): # this is for the inf image
        #    self.vminLog,self.vmaxLog=np.inf,np.inf            
        #else:            
        #    self.vminLog,self.vmaxLog = fim[fim>-nFABOnly_kmeans_it005_class001.mrcp.inf].min(),fim[fim<np.inf].max()
#        if self.vminLog == self.vmaxLog:
#            self.vmaxLog += sys.float_info.epsilon
        self.offVmin,self.offVmax = 0,0
        self.frameShape = self.im.shape[:2]
        self.showProfiles = False
        if not(self.showProfiles):
            self.fig = plt.figure()
            self.figNum = plt.get_fignums()[-1]
            print( "Shown in figure %g."%self.figNum)
            self.ax = self.fig.add_subplot(111)
        else:
            ################
            # definitions for the axes
            widthProf = 0.1
            left, width = 0.05, 0.75
            bottomProf = 0.05
            bottom, height = widthProf + bottomProf + 0.05, 0.75

            leftProf = left + width + 0.05
            
            rect_im = [left, bottom, width, height]
            rect_X = [left, bottomProf, width, widthProf] # horizontal
            rect_Y = [leftProf, bottom, widthProf, height] # vertical
            
            # start with a rectangular Figure
            self.fig = plt.figure(figsize=(8,8))        
            self.ax = plt.axes(rect_im)
            self.axX = plt.axes(rect_X)
            self.axY = plt.axes(rect_Y)

            nullfmt = plt.NullFormatter()         # no labels
            self.axX.xaxis.set_major_formatter(nullfmt)
            self.axX.yaxis.set_major_formatter(nullfmt)
            self.axY.xaxis.set_major_formatter(nullfmt)
            self.axY.yaxis.set_major_formatter(nullfmt)
            self.posProfHoriz = np.round(self.frameShape[0]/2)
            self.posProfVert = np.round(self.frameShape[1]/2)

        ################
        
        self.draw()
        self.fig.canvas.mpl_connect('key_press_event',self)

    def draw(self):
        plt.cla()
        tit=str()
        if self.im.ndim is 2:
            im = self.im
        elif self.im.ndim is 3:
            im = self.im[...,self.i]
            tit='frame {0}'.format(self.i)+'/'+str(self.im.shape[2]-1)
            # projections            
            if self.projToggle:
                if self.projType=='M':
                    im=self.im.max(2)
                    tit = 'max projection'                    
                if self.projType=='S':
                    im=self.im.sum(2)
                    tit = 'sum projection'            
                if self.projType=='V':
                    im=plt.var(self.im,2)
                    tit = 'var projection'            
        if self.complex:
            tit += ', cplx (0=abs,1=phase)'
        if self.logon:
            tit += ', log10'
            minval = 0 #sys.float_info.epsilon
            if (im <= minval).any():
                # RAM (FIXED): this isn't quite right, we need to mask out these pixels 
                # im2show = np.log10(im.clip(min=minval) + (im <= 0) )
                # RAM: alternatively we could just add the minimum value to the whole matrix
                im2show = np.log10( im - (np.min(im) - 1.0) )
            else: 
                im2show = np.log10(im)
#            if self.globalScale:  
#                vrange = self.vmaxLog - self.vminLog
#                vmin_tmp=self.vminLog + self.offVmin*vrange
#                vmax_tmp=self.vmaxLog - self.offVmax*vrange   
#                tit += ', global scale'
#            else:
#                fi = im2show.flatten()
#                immin = fi[fi>-np.inf].min()
#                immax = fi[fi<np.inf].max()               
#                vrange = immax - immin
#                vmin_tmp = immin + self.offVmin*vrange
#                vmax_tmp = immax - self.offVmax*vrange
#        elif type( self.im ) == tables.carray.CArray:
#            im = self.im[:,:,self.i]
        else:
            tit += ', lin'
            im2show = im
#            if self.globalScale:
#                vrange = self.vmax-self.vmin
#                vmin_tmp=self.vmin + self.offVmin*(vrange)
#                vmax_tmp=self.vmax - self.offVmax*(vrange)
#                tit += ', global scale'
#            else:
#                immin,immax = im2show.min(),im2show.max()
#                vrange = immax - immin
#                vmin_tmp = immin + self.offVmin*vrange
#                vmax_tmp = immax - self.offVmax*vrange

        if self.zoom > 1:
            tit += ', zoom %g x'%(self.zoom)
            
#        if self.offVmin or self.offVmax:
#            tit += ', contrast L%g %% H%g %%'%(round(self.offVmin*100),round(self.offVmax*100))
        if self.cmap==0:
            plt.jet()
        elif self.cmap==1:
            plt.gray()   
        elif self.cmap==2:
            plt.set_cmap( 'gnuplot')
        elif self.cmap==3:
            plt.set_cmap( 'gist_earth')
        if self.titles==None:
            self.ax.set_title(tit)
        else:
            try:
                self.ax.set_title(self.titles[self.i])
            except:
                pass
            

        
        # plt.colorbar(im)
        plt.show()        
        hx,hy = self.frameShape[0]/2., self.frameShape[1]/2.
        lx,ly = hx/(self.zoom),hy/(self.zoom)

        rx_low = max( min(np.floor(hx) + self.offx - np.floor(lx),self.frameShape[0]-self.stepXY),0.0)
        rx_high = min(max(np.floor(hx) + self.offx + np.ceil(lx),self.stepXY),self.frameShape[0])
        rx = np.arange(rx_low,rx_high)[:,np.newaxis].astype(int)


        ry_low = max(min(np.floor(hy) + self.offy - np.floor(ly),self.frameShape[1]-self.stepXY),0.0)
        ry_high = min(max(np.floor(hy) + self.offy + np.ceil(ly),self.stepXY),self.frameShape[1])
        ry = np.arange(ry_low,ry_high).astype(int)
#        rx = (rx[np.minimum(rx>=0,rx<self.frameShape[1])]).astype(int)
#        ry = (ry[np.minimum(ry>=0,ry<self.frameShape[0])][:,np.newaxis]).astype(int)
        if self.histogramClimMode:
            new_clims = zorro.util.histClim( im2show[rx,ry], cutoff=self.cutoff )
            #print( "New clims: " + str(new_clims) )
            vmin_tmp = new_clims[0]
            vmax_tmp = new_clims[1]
        else:
            vmin_tmp = np.min( im2show[rx,ry] )
            vmax_tmp = np.max( im2show[rx,ry] )
        

        self.ax.imshow(im2show[rx,ry], vmin=vmin_tmp, vmax=vmax_tmp, interpolation='Nearest',extent=[ry[0],ry[-1],rx[-1],rx[0]])
        # plt.colorbar(self.ax)
        def format_coord(x, y):
            x = np.int(x + 0.5)
            y = np.int(y + 0.5)
            try:
                #return "%s @ [%4i, %4i]" % (round(im2show[y, x],2), x, y)
                return "%.5G @ [%4i, %4i]" % (im2show[y, x], y, x) #first shown coordinate is vertical, second is horizontal
            except IndexError:
                return ""
        self.ax.format_coord = format_coord
        if 'qt' in plt.matplotlib.get_backend().lower():
            self.fig.canvas.manager.window.raise_() #this pops the window to the top    
        if self.showProfiles:
            posProf = self.posProfHoriz
            self.axX.cla()
            self.axX.plot(rx+1,im2show[posProf,rx])
#            plt.xlim(rx[0],rx[-1])
            self.axX.set_xlim(rx[0],rx[-1])
    def printStat(self):
        if self.globalScale:
            modePrint = 'all frames'
            img = self.im
            if self.complex:            
                modePrint = 'modulus'
                img = self.im[...,0]

        else:
            if self.im.ndim > 2:
                img = self.im[...,self.i]
                modePrint = 'frame %g'%self.i               
            else:
                img = self.im
                modePrint = 'frame'
        print( "\n-----"           )             
        print( "Statistics of the " + modePrint + " in figure %g:"%self.figNum)
        print( "Shape: ", img.shape             ) 
        print( "Maximum: ", img.max(), "@", np.unravel_index(np.argmax(img),img.shape))
        print( "Minimum: ", img.min(), "@", np.unravel_index(np.argmin(img),img.shape))
        print( "Center of mass:", nd.measurements.center_of_mass(img))
        print( "Mean: ", img.mean())
        print( "Standard deviation: ", img.std())
        print( "Variance: ", img.var()        )
        print( "Sum: ", img.sum())
        print( "Data type:", self.dtype)
        self.draw()
        self.fig.canvas.draw()


    def __call__(self, event):
#        old_i = self.i
        if event.key=='n':#'up': #'right'
            if self.im.ndim > 2:
                self.i = min(self.im.shape[2]-1, self.i+1)
        elif event.key == 'p':#'down': #'left'
            if self.im.ndim > 2:
                self.i = max(0, self.i-1)
        if event.key=='N':#'up': #'right'
            if self.im.ndim > 2:        
                self.i = min(self.im.shape[2]-1, self.i+10)
        elif event.key == 'P':#'down': #'left'
            if self.im.ndim > 2:        
                self.i = max(0, self.i-10)
        elif event.key == 'l':
            self.logon = np.mod(self.logon+1,2)            
        elif event.key == 'c':
            self.cmap = np.mod(self.cmap+1,4)
        elif event.key == 'h':
            if self.histogramClimMode:
                self.histogramClimMode = False
            else:
                self.histogramClimMode = True
            
        elif event.key in 'SMV':
            self.projToggle = np.mod(self.projToggle+1,2)
            self.projType = event.key
        elif event.key == 'i':
            if 4*self.zoom < min(self.im.shape[:1]): # 2*zoom must not be bigger than shape/2
                self.zoom = 2*self.zoom
        elif event.key == 'o':
            self.zoom = max(self.zoom/2,1)            
        elif event.key == 'g':
            self.globalScale = np.mod(self.globalScale+1,2)
        elif event.key == 'right':
            self.offy += self.stepXY
            self.offy = min(self.offy,self.im.shape[0]-1)
        elif event.key == 'left':
            self.offy -= self.stepXY
            self.offy = max(self.offy,-self.im.shape[0]+1)            
            
        elif event.key == 'down':
            self.offx += self.stepXY
            self.offx = min(self.offx,self.im.shape[1]-1)            
        elif event.key == 'up':
            self.offx -= self.stepXY
            self.offx = max(self.offx,-self.im.shape[1]+1)
        elif event.key == 'r': # reset position to the center of the image
            self.offx,self.offy = 0,0
            print( "Reseting positions to the center.")
        elif event.key == 'R': # reset contrast
            self.offVmin,self.offVmax = 0,0
            print( "Reseting contrast.")
        elif event.key == 'q': # increase lower limit of the contrast
            self.offVmin = min(self.offVmin+.1,1)        
        elif event.key == 'Q': # decrease lower limit of the contrast
            self.offVmin = max(self.offVmin-.1,0)
        elif event.key == 'w': # increase upper limit of the contrast
            self.offVmax = min(self.offVmax+.1,1)        
        elif event.key == 'W': # decrease upper limit of the contrast
            self.offVmax = max(self.offVmax-.1,0)
#            print "Increasing upper limit of the contrast: %g %% (press R to reset).\n"%round(self.offVmax*100)
        elif event.key == 'T': # print statistics of the whole dataset
            self.printStat(),                      
            print( "-----")
        elif event.key == 't': # print statistics of the current frame
            self.printStat(mode = 'current frame'),                      
            print( "-----")

#        if old_i != self.i:            
#        print self.offx
        self.draw()
        self.fig.canvas.draw()

def im(my_img,ax=None,**kwargs):
    "Displays image showing the values under the cursor."
    if ax is None:
        ax = plt.gca()
    def format_coord(x, y):
        x = np.int(x + 0.5)
        y = np.int(y + 0.5)
        val = my_img[y,x]
        try:
            return "%.4E @ [%4i, %4i]" % (val, x, y)
        except IndexError:
            return ""
    ax.imshow(my_img,interpolation='nearest',**kwargs)
    ax.format_coord = format_coord
    plt.colorbar()
    plt.draw()
    plt.show()    
    
def imTiles(d,sizeX=None,titNum=True):
    "Displays the stack of images in the composed tiled figure."
    if sizeX==None:
        sizeX=np.ceil(np.sqrt(d.shape[0]))
    sizeY=np.ceil(d.shape[0]/sizeX)
    plt.figure(figsize=(sizeY, sizeX))
    for i in np.arange(1,d.shape[0]+1):
        plt.subplot(sizeX,sizeY,i)
        plt.imshow(d[i-1],interpolation='nearest')
        plt.xticks([])
        plt.yticks([])
        if titNum:
            plt.title(str(i-1))
                
def complex2rgbalog(s,amin=0.5,dlogs=2):
   """
   Displays complex image with intensity corresponding to the log(MODULUS) and color (hsv) correponging to PHASE.
   From: pyVincent/ptycho.py
   """
   ph=np.anlge(s)
   t=np.pi/3
   nx,ny=s.shape
   rgba=np.zeros((nx,ny,4))
   rgba[:,:,0]=(ph<t)*(ph>-t) + (ph>t)*(ph<2*t)*(2*t-ph)/t + (ph>-2*t)*(ph<-t)*(ph+2*t)/t
   rgba[:,:,1]=(ph>t)         + (ph<-2*t)      *(-2*t-ph)/t+ (ph>0)*(ph<t)    *ph/t
   rgba[:,:,2]=(ph<-t)        + (ph>-t)*(ph<0) *(-ph)/t + (ph>2*t)         *(ph-2*t)/t
   a=np.log10(np.abs(s)+1e-20)
   a-=a.max()-dlogs # display dlogs orders of magnitude
   rgba[:,:,3]=amin+a/dlogs*(1-amin)*(a>0)
   return rgba

def complex2rgbalin(s):
   """
   Displays complex image with intensity corresponding to the MODULUS and color (hsv) correponging to PHASE.
   From: pyVincent/ptycho.py
   """    
   ph=np.angle(s)
   t=np.pi/3
   nx,ny=s.shape
   rgba=np.zeros((nx,ny,4))
   rgba[:,:,0]=(ph<t)*(ph>-t) + (ph>t)*(ph<2*t)*(2*t-ph)/t + (ph>-2*t)*(ph<-t)*(ph+2*t)/t
   rgba[:,:,1]=(ph>t)         + (ph<-2*t)      *(-2*t-ph)/t+ (ph>0)*(ph<t)    *ph/t
   rgba[:,:,2]=(ph<-t)        + (ph>-t)*(ph<0) *(-ph)/t + (ph>2*t)         *(ph-2*t)/t
   a=np.abs(s)
   a/=a.max()
   rgba[:,:,3]=a
   return rgba

def colorwheel(col='black'):
  """
  Color wheel for phases in hsv colormap.
  From: pyVincent/ptycho.py
  """
  xwheel=np.linspace(-1,1,100)
  ywheel=np.linspace(-1,1,100)[:,np.newaxis]
  rwheel=np.sqrt(xwheel**2+ywheel**2)
  phiwheel=-np.arctan2(ywheel,xwheel)  # Need the - sign because imshow starts at (top,left)
#  rhowheel=rwheel*np.exp(1j*phiwheel)
  rhowheel=1*np.exp(1j*phiwheel)
  plt.gca().set_axis_off()
  rgba=complex2rgbalin(rhowheel*(rwheel<1))
  plt.imshow(rgba,aspect='equal')
  plt.text(1.1, 0.5,'$0$',fontsize=14,horizontalalignment='center',verticalalignment='center',transform = plt.gca().transAxes,color=col)
  plt.text(-.1, 0.5,'$\pi$',fontsize=16,horizontalalignment='center',verticalalignment='center',transform = plt.gca().transAxes,color=col)

def insertColorwheel(left=.7, bottom=.15, width=.1, height=.1,col='black'):
    """
    Inserts color wheel to the current axis.
    """
    plt.axes((left,bottom,width,height), axisbg='w')
    colorwheel(col=col)
   # plt.savefig('output.png',bbox_inches='tight', pad_inches=0)      

def insertColorbar(fig,im,left=.7, bottom=.1, width=.05, height=.8 )     :
    """
    Inserts color bar to the current axis.
    """
    cax = fig.add_axes((left,bottom,width,height), axisbg='w')
    plt.colorbar(im, cax=cax)
    

def showCplx(im,mask=0,pixSize_um=1,showGrid=True,modulusLog = False,maskPhase = False, maskPhaseThr = 0.01, cmapModulus = 'jet', cmapPhase = 'hsv', scalePhaseImg = True):
    "Displays MODULUS and PHASE of the complex image in two subfigures."
    if modulusLog:
        modulus = np.log10(np.abs(im)) 
    else:
        modulus = np.abs(im)
    phase = np.angle(im)
    plt.figure(figsize=(8,4))
    plt.subplot(121)
    #plt.subplots_adjust(left=0.02, bottom=0.06, right=0.95, top=0.94, wspace=0.05)
    #plt.imshow(abs(np.ma.masked_array(im,mask)))
    plt.imshow(modulus,extent=(0,im.shape[1]*pixSize_um,0,im.shape[0]*pixSize_um),cmap=cmapModulus,interpolation='Nearest')    
#    plt.colorbar(m)
    if showGrid: 
        plt.grid(color='w')
    if pixSize_um !=1:
        plt.xlabel('microns')                
        plt.ylabel('microns')        
    plt.title('Modulus')
#    position=f.add_axes([0.5,0.1,0.02,.8])  ## the parameters are the specified position you set 
#    plt.colorbar(m,cax=position) ##
#    plt.setp(ax_cb.get_yticklabels(), visible=False)

    plt.subplot(122)
    if scalePhaseImg:
        vminPhase = -np.pi
        vmaxPhase = np.pi
    else:
        vminPhase = phase.min()
        vmaxPhase = phase.max()
        
    plt.imshow(np.ma.masked_array(phase,mask),cmap=cmapPhase,interpolation='Nearest',vmin=vminPhase,vmax=vmaxPhase,extent=(0,im.shape[1]*pixSize_um,0,im.shape[0]*pixSize_um))
    if showGrid:
        plt.grid(color='k')
    if pixSize_um !=1:
        plt.xlabel('microns')                
        plt.ylabel('microns')        
    plt.title('Phase')
    if cmapPhase == 'hsv':
        insertColorwheel(left=.85)
    plt.tight_layout()

def showLog(im, cmap='jet'):
    "Displays log of the real image with correct colorbar."
    f = plt.figure(); 
    i = plt.imshow(im, norm=col.LogNorm(), cmap=cmap)
    f.colorbar(i)
    return f,i
        
def ca():
    """
    Close all windows.
    """
    plt.close('all')
    

def main():
    # Get command line arguments
    import sys
    
    # First argument is the executed file
    # print sys.argv 
    print( """
    Shows individual frames in the 3D image (dimensions organized as [z,x,y]). 
    "n" next frame, ("N" next by step of 10)
    "p" previous frame, ("P" previous by step of 10)
    "l" toogles the log scale. 
    "c" swithces between 'jet','grey','hot' and 'hsv' colormaps. 
    "h" turns on histogram-based contrast limits
    "i" zooms in
    "o" zooms out
    "arrows" move the frame around
    "r" resets the position to the center of the frame
    "g" toogle local/global stretch
    "q/Q" increase / decrease the lower limit of the contrast
    "w/W" increase / decrease the upper limit of the contrast
    "R" reset contrast to default
    "S" shows sum projection
    "M" shows maximum projection
    "V" shows variance projection    
    "T" prints statistics
""")
    
    if len( sys.argv ) >= 3:
        cutoff = np.float32( sys.argv[2] )
    else:
        cutoff = None
        
    ims( sys.argv[1], cutoff=cutoff )
    # Need to hold here.
    try:
        input( "Press enter to close IMS" )
    except SyntaxError:
        pass
    sys.exit()
    
#### COMMAND-LINE INTERFACE ####
if __name__ == '__main__':
    main()
    
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
import scipy.ndimage as ni
import zorro
import os, os.path, sys
import mrcz

##################################################################################
######################## Object-oriented interface ###############################
##################################################################################
class zorroPlot(object):
    
    def __init__(self, filename=None, width=7, height=7, plot_dpi=72, image_dpi=144, facecolor=[0.75,0.75,0.75,1.0], 
                 MplCanvas = None, backend=u'Qt4Agg' ):
        """
        Object-oriented plotting interface for Zorro.
        """
        # All parameters are stored in a hash-dictionary
        self.plotDict = {}
        self.plotDict[u'width'] = width
        self.plotDict[u'height'] = height
        self.plotDict[u'plot_dpi'] = plot_dpi
        self.plotDict[u'image_dpi'] = image_dpi
        self.plotDict[u'facecolor'] = facecolor
        
        if bool(filename):
            print( "TODO: load and display file from zorroPlot.__init__()" )
        
        # http://stackoverflow.com/questions/13714454/specifying-and-saving-a-figure-with-exact-size-in-pixels
        self.fig = matplotlib.figure.Figure(figsize=(width, height), facecolor=facecolor, dpi=plot_dpi )
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
    
    def printPlot( self, dpi_key = u"plot_dpi" ):
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

            self.canvas.print_figure( self.plotDict[u'plotFile'], dpi=self.plotDict[dpi_key], 
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

                self.axes.imshow( self.plotDict[u'boxMask'], 
                                 extent=mage.get_extent(), interpolation='lanczos' )
            
        else:
            print( "No pixmap" )
  
        self.axes.hold(False)
        
    def plotImage( self ):
        self.fig.clear()
        self.axes = self.fig.add_axes( [0.0, 0.0, 1.0, 1.0] )

        if "lowPass" in self.plotDict:
            self.plotDict['image'] = ni.gaussian_filter( self.plotDict['image'], self.plotDict["lowPass"] )
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
        
        return self.printPlot( dpi_key=u'image_dpi' )
        
    def plotStack( self ):
        print( "TODO: implement plotStack" )
        
    def plotFFT( self ):
        self.fig.clear()
        self.axes = self.fig.add_axes( [0.0, 0.0, 1.0, 1.0] )
        self.axes.hold(False)

        FFTimage = np.fft.fft2( self.plotDict['image'] )
        FFTimage[0,0] = 1.0 # Clip out zero-frequency pixel
        
        FFTimage = np.log10( 1.0 + np.abs( np.fft.fftshift( FFTimage )))
        if "lowPass" in self.plotDict:
            FFTimage = ni.gaussian_filter( FFTimage, self.plotDict["lowPass"] )
            
        FFTclim = zorro.util.ciClim( FFTimage, sigma=2.5 )
        mage = self.axes.imshow( FFTimage, interpolation='bicubic', vmin=FFTclim[0], vmax=FFTclim[1], 
                         cmap=self.plotDict['image_cmap'] )
        if 'pixelsize' in self.plotDict:
            inv_ps = 1.0 / (FFTimage.shape[0] * self.plotDict['pixelsize'] )
            zorro.util.plotScalebar( mage, inv_ps, units=u'nm^{-1}' )
        self.axes.set_axis_off()
        if bool(self.plotDict['colorbar']):
            self.fig.colorbar( mage, fraction=0.046, pad=0.04)
        return self.printPlot( dpi_key=u'image_dpi' )
                             
    def plotPolarFFT( self ):
        self.fig.clear()
        self.axes = self.fig.add_axes( [0.0, 0.0, 1.0, 1.0] )
        self.axes.hold(False)
        
        polarFFTimage = zorro.util.img2polar( np.log10( 1.0 + np.abs( np.fft.fftshift( np.fft.fft2( self.plotDict['image'] )))) )
        if "lowPass" in self.plotDict:
            polarFFTimage = ni.gaussian_filter( polarFFTimage, self.plotDict["lowPass"] )
            
        FFTclim = zorro.util.ciClim( polarFFTimage, sigma=2.0 )
        mage = self.axes.imshow( polarFFTimage, interpolation='bicubic', vmin=FFTclim[0], vmax=FFTclim[1], 
                         cmap=self.plotDict['image_cmap'] )
        if 'pixlsize' in self.plotDict:
            # Egh, this scalebar is sort of wrong, maybe I should transpose the plot?
            inv_ps = 1.0 / (polarFFTimage.shape[0] * self.plotDict['pixelsize'] )
            zorro.util.plotScalebar( mage, inv_ps, units=u'nm^{-1}' )
        self.axes.set_axis_off()
        if bool(self.plotDict['colorbar']):
            self.fig.colorbar( mage, fraction=0.046, pad=0.04)
            
        return self.printPlot( dpi_key=u'image_dpi' )                        
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

        return self.printPlot( dpi_key=u'plot_dpi' )
            
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
        
        return self.printPlot( dpi_key=u'plot_dpi' )                                
    
    def plotTranslations( self ):
        # rect is [left,bottom,width,height]
        self.fig.clear()
        self.axes = self.fig.add_axes( [0.12, 0.1, 0.85, 0.85] )
        self.axes.hold(True)

        
        if 'errorX' in self.plotDict:
            self.axes.errorbar( self.plotDict['translations'][:,1], self.plotDict['translations'][:,0], fmt='k-',
                              xerr=self.plotDict['errorX'], yerr=self.plotDict['errorY'] )
        else:
            self.axes.plot( self.plotDict['translations'][:,1], self.plotDict['translations'][:,0], 'k.-',
                   linewidth=2.0, markersize=16 )
        self.axes.plot( self.plotDict['translations'][0,1], self.plotDict['translations'][0,0], 
                       '.', color='purple', markersize=16 )
        self.axes.set_xlabel( 'X-axis drift (pix)' )
        self.axes.set_ylabel( 'Y-axis drift (pix)' )
        self.axes.axis('equal')
        self.axes.hold(False)
        
        return self.printPlot( dpi_key=u'plot_dpi' )
                             
    def plotPixRegError( self ):
        self.fig.clear()
        self.axes = self.fig.add_subplot( 211 )
        self.axes.hold(False)
        self.axes2 = self.fig.add_subplot( 212 )
        self.axes2.hold(False)
        
        weightedErrorX = np.abs( self.plotDict['errorX'] )
        weightedErrorY = np.abs( self.plotDict['errorY'] )
        
        meanErrX = np.mean( weightedErrorX )
        meanErrY = np.mean( weightedErrorY )
        stdErrX = np.std( weightedErrorX )
        stdErrY = np.std( weightedErrorY )
        
        errorX = np.abs( self.plotDict['errorXY'][:,1] )
        errorY = np.abs( self.plotDict['errorXY'][:,0] )
        
        self.axes.semilogy( errorX, '.:', linewidth=1.5, color='black', markersize=12, markerfacecolor='darkslateblue',
                       label='$\Delta$X: %.3f +/- %.3f pix'%(meanErrX, stdErrX) )
        self.axes.legend( fontsize=12, loc='best' )
        self.axes.set_ylabel( "X-error estimate (pix)" )
        
        # self.axes.set_title( 'X: %f +/- %f'%(meanErrX, stdErrX) )
        self.axes2.semilogy( errorY, '.:', linewidth=1.5, color='black', markersize=12, markerfacecolor='darkolivegreen',
                        label='$\Delta$Y: %.3f +/- %.3f pix'%(meanErrY, stdErrY) )
        #self.axes2.set_title( 'Y: %f +/- %f pix'%(meanErrY, stdErrY) )
        self.axes2.legend( fontsize=12, loc='best' )
        self.axes2.set_xlabel( "Equation number" )
        self.axes2.set_ylabel( "Y-error estimate (pix)" )
        
        return self.printPlot( dpi_key=u'plot_dpi' )   
                             
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
        
        return self.printPlot( dpi_key=u'plot_dpi' )      


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
        
        return self.printPlot( dpi_key=u'plot_dpi' )
            
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

        infobox = matplotlib.offsetbox.AnchoredText( results, pad=0.5, loc=1, prop={'size':16} )
        self.axes.add_artist( infobox )
        
        self.axes.set_axis_off() # This is still not cropping properly...
        
        return self.printPlot( dpi_key=u'plot_dpi' )
    
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

        return self.printPlot( dpi_key=u'plot_dpi' )
    
    
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
 

IMS_HELPTEXT = """
    Usage: ims <image_filename> <cutoff level>
    
    Valid types: .dm4, .mrc, .mrcs, .mrcz, .mrczs
    
    Shows individual frames in the 3D image (dimensions organized as [z,x,y]). 
    "f" shows the view in full-screen
    "n" next frame, ("N" next by step of 10)
    "p" previous frame, ("P" previous by step of 10)
    "l" toogles the log scale. 
    "y" toggles polar transform
    "F" toggles Fourier transform
    "c" swithces between gray, gnuplot, jet, nipy_spectral colormaps. 
    "h" turns on histogram-based contrast limits
    "b" hides/shows boxes (searches for _automatch.box file )
    "i" zooms in
    "o" zooms out
    "v" transposes (revolves) the axes so a different projection is seen.
    "arrows" move the frame around
    "g" gaussian low-pass ( sharpen more with 'k', smoothen more with 'm')
    "r" resets the position to the center of the frame
    "q" increase the contrast limits ("Q" is faster)
    "w" decrease the contrast limits ("W" is faster)
    "R" reset contrast to default
    "s" saves current view as PNG
    "S" shows sum projection
    "M" shows max projection
    "V" shows var projection    
    "t" print statistics for current frame
    "T" prints statistics for entire stack
    
    """
    
class ims(object):
    IMS_HELPTEXT
    
    plt.rcParams['keymap.yscale'] = '' # to disable the binding of the key 'l'
    plt.rcParams['keymap.pan'] = '' # to disable the binding of the key 'p'    
    plt.rcParams['keymap.grid'] = '' # to disable the binding of the key 'g'        
    plt.rcParams['keymap.zoom'] = '' # to disable the binding of the key 'o'            
    def __init__(self, im, index=0, titles=[u"",], logMode=False, fftMode=False, polarMode=False, blocking=False ):
        plt.ion()
        #plt.pause(1E-4)
        self.im = im
        self.index = index
        self.cmaps_cycle = itertools.cycle( [u"gray", u"gnuplot", u"jet", u"nipy_spectral"] )
        self.cmap = next( self.cmaps_cycle )
        self.exiting = False
        self.logMode = logMode
        self.polarMode = polarMode
        self.fftMode = fftMode
        self.sigmaMode = True
        self.filterMode = False
        self.__gaussSigma = 1.5
        self.doTranspose = False
        self.filename = None
        self.titles = titles
        self.__currTitle = ""
        self.__sigmaLevels = np.hstack( [np.array( [0.01, 0.02, 0.04, 0.06, 0.08]), 
                                                  np.arange( 0.1, 20.1, 0.1 )])
        self.__sigmaIndex = 31 # 3.0 sigma by default
        self.blocking = blocking
        
        self.showBoxes = True
        self.boxLen = 0
        self.boxYX = None
        self.boxFoM = None
        
        print( "ims: type(im) = %s" % type(im) )
        
        if sys.version_info >= (3,0):
            if isinstance( self.im, str ):
                self.loadFromFile( im )
        else: # Python 2
            if isinstance( self.im, str ) or isinstance(self.im, unicode):
                self.loadFromFile( im )
                
        if isinstance( self.im, tuple) or isinstance( self.im, list):
            # Gawd tuples are annoyingly poorly typedefed
            self.im = np.array( self.im )
            
            print( "shape of tupled array: " + str(self.im.shape) )
            # Don't even bother checking, the complex representation needs to be re-written anyway
            self.complex = False
            
        if self.im.ndim is 2:
            if np.iscomplex(self.im).any():
                self.complex = True
                self.im = np.array( [np.hypot( np.real(self.im), np.imag(self.im)),np.angle(self.im)] )      
                print( "DEBUG: complex self.im.shape = %s" % str(self.im.shape) )
                self.__imCount = 2
                self.frameShape = self.im.shape[1:]
            else:                
                self.complex = False
            self.frameShape = self.im.shape
            self.__imCount = 1
        elif self.im.ndim is 3:
            if np.iscomplex( self.im ).any():
                self.im = np.hypot( np.real(self.im), np.imag(self.im) )
            self.complex = False
            self.frameShape = self.im.shape[1:]
            self.__imCount = self.im.shape[0]
            
        
        self.__minList = np.nan * np.empty( self.__imCount )  # Could retrieve this from MRC files?
        self.__maxList = np.nan * np.empty( self.__imCount ) # Could retrieve this from MRC files?
        self.__meanList = np.nan * np.empty( self.__imCount ) # Could retrieve this from MRC files?
        self.__stdList = np.nan * np.empty( self.__imCount ) # Could retrieve this from MRC files?
            
        print( "IMS self.im.shape = %s" % str(self.im.shape) )
            
        self.dtype = self.im.dtype
        
            
        
        self.projToggle = False
        self.zoom = 1
        self.offx,self.offy = 0,0
        self.stepXY = 24 # step of the movement up-down, left-right

        self.offVmin,self.offVmax = 0,0
        
        self.showProfiles = False
        if not(self.showProfiles):
            self.fig = plt.figure()
            self.figNum = plt.get_fignums()[-1]
            print( "Shown in figure %g."%self.figNum)
            self.ax = self.fig.add_subplot(111)
        else:
            self.fig = plt.figure(figsize=(10,10)) 

            self.ax = self.fig.axes
            self.__setaxes__()

        ################
        
        self.__recompute__()
        self.fig.canvas.mpl_connect( 'key_press_event', self.__call__ )
        self.fig.canvas.mpl_connect( 'close_event', self.__exit__ )
        self.fig.canvas.mpl_connect( 'resize_event', self.__draw__ )

        plt.show( block=self.blocking )
        
        # plt.ion()
        
    def loadFromFile(self, filename, loadBoxes=True ):
        self.titles = self.im
        print( "Try to load MRC or DM4 files" )
        file_front, file_ext = os.path.splitext( self.im )
        if (file_ext.lower() == ".mrc" or file_ext.lower() == ".mrcs" or 
            file_ext.lower() == ".mrcz" or file_ext.lower() == ".mrcsz"):
            self.im, self.header = mrcz.readMRC( self.im, pixelunits=u'nm' )
        elif file_ext.lower() == ".dm4":
            dm4struct = mrcz.readDM4( self.im )
            self.im = dm4struct.im[1].imageData
            self.header = dm4struct.im[1].imageInfo
            del dm4struct
        else:
            print( "Filename has unknown/unimplemented file type: " + self.im )
            return
            
        # Check for boxes
        # Star files don't contain box sizes so use the box files instead
        box_name = file_front + "_automatch.box"
        if bool(self.showBoxes) and os.path.isfile( box_name ):
            self.loadBoxFile( box_name )
            return
            
        # Try the star file instead
        box_name = file_front + "_automatch.star"
        if bool(self.showBoxes) and os.path.isfile( box_name ):
            self.loadStarFile( box_name )
            return

        
    def loadBoxFile(self, box_name ):
        box_data = np.loadtxt( box_name, comments="_" )
        # box_data columns = [x_center, y_center, ..., ..., FigureOfMerit]
        self.boxLen = box_data[0,2]

        # In boxfiles coordinates are at the edges.
        self.boxYX = np.fliplr( box_data[:,:2] )
        # DEBUG: The flipping of the y-coordinate system is annoying...
        print( "boxYX.shape = " + str(self.boxYX.shape) + ", len = " + str(self.boxLen) )
        self.boxYX[:,0] = self.im.shape[0] - self.boxYX[:,0]
        self.boxYX[:,1] += int( self.boxLen / 2 )
        self.boxYX[:,0] -= int( self.boxLen/2)
        try:
            self.boxFoM = box_data[:,4]
            
            clim = zorro.zorro_util.ciClim( self.boxFoM, sigma=2.5 )
            self.boxFoM = zorro.zorro_util.normalize( np.clip( self.boxFoM, clim[0], clim[1] ) )

        except:
            self.boxFoM = np.ones( self.boxYX.shape[0] )
        self.boxColors = plt.cm.gnuplot( self.boxFoM )
        
    def loadStarFile(self, box_name ):
        box_data = np.loadtxt( box_name, comments="_", skiprows=5 )
        # box_data columns = [x_center, y_center, ..., ..., FigureOfMerit]
        # In star files coordinates are centered
        self.boxYX = np.fliplr( box_data[:,:2] )
        # DEBUG: The flipping of the y-coordinate system is annoying...
        self.boxYX[:,0] = self.im.shape[0] - self.boxYX[:,0]

        # There's no box size information in a star file so we have to use a guess
        self.boxLen = 224
        #self.boxYX[:,1] -= int( self.boxLen / 2 )
        #self.boxYX[:,0] += int( self.boxLen / 2 )
        try:
            self.boxFoM = box_data[:,4]
            clim = zorro.zorro_util.ciClim( self.boxFoM, sigma=2.5 )
            self.boxFoM = zorro.zorro_util.normalize( np.clip( self.boxFoM, clim[0], clim[1] ) )

        except:
            self.boxFoM = np.ones( self.boxYX.shape[0] )
        
        self.boxColors = plt.cm.gnuplot( self.boxFoM )
    def __setaxes__(self):
        self.ax.cla()
        
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
            
    def __recompute__(self):
        self.__currTitle = ""
        if self.doTranspose:
            self.doTranspose = False
            self.im = np.transpose( self.im, axes=[2,0,1] )
            print( "Tranposed axes shape: %s" % str(self.im.shape) )
            self.__setaxes__()
            
        if self.im.ndim is 2:
            self.im2show = self.im
        elif self.im.ndim is 3:
                
            self.im2show = np.squeeze( self.im[self.index,...] )
            self.__currTitle = 'frame %d/%d' % (self.index, self.im.shape[0]-1)
            # projections            
            if self.projToggle:
                if self.projType=='M':
                    self.im2show = self.im.max(axis=0)
                    self.__currTitle = 'max proj'                    
                if self.projType=='S':
                    self.im2show = self.im.sum(axis=0)
                    self.__currTitle = 'sum proj'            
                if self.projType=='V':
                    self.im2show = np.var(self.im,axis=0)
                    self.__currTitle = 'var proj'            
        if self.complex:
            self.__currTitle += ', cplx (0=abs,1=phase)'
        if self.fftMode:
            self.__currTitle += ", fft"
            self.im2show = np.abs(np.fft.fftshift( np.fft.fft2( self.im2show ) ))

            
        if self.polarMode:
            self.__currTitle += ", polar"
            self.im2show = zorro.zorro_util.img2polar( self.im2show )
        if self.filterMode:
            self.__currTitle += ", gauss%.2f" % self.__gaussSigma
            self.im2show = ni.gaussian_filter( self.im2show, self.__gaussSigma )
        if self.logMode:
#            # TODO: this can be sent to matplotlib as an argument in imshow instead
            self.__currTitle += ', log10'
            if np.any(self.im <= 0.0):
                # RAM: alternatively we could just add the minimum value to the whole matrix
                 self.im2show = np.log10( self.im2show - np.min( self.im2show ) + 1.0 )
            else:
                self.im2show = np.log10( self.im2show )
        else:
            self.__currTitle += ', lin'
            
        # We need to compute image-wide statistics
        if self.sigmaMode:
            self.__meanList[self.index] = np.mean( self.im2show )
            self.__stdList[self.index] = np.std( self.im2show )
        else:
            self.__minList[self.index] = np.min( self.im2show )
            self.__maxList[self.index] = np.max( self.im2show )
        self.__draw__()
            
    def __draw__(self, info=None ):
        # print( "Called ims.draw()" )
        plt.cla()
        
        tit = self.__currTitle + ""
        if self.zoom > 1:
            tit += ', zoom %g x'%(self.zoom)
            
        center_y = np.int( self.frameShape[0]/2 )
        center_x = np.int( self.frameShape[1]/2 )
        halfWidth_y = np.int( 0.5* self.frameShape[0]/self.zoom )
        halfWidth_x = np.int( 0.5* self.frameShape[1]/self.zoom )

        im_range = [ np.maximum( 0, center_x-halfWidth_x), 
                     np.minimum( self.frameShape[1], center_x+halfWidth_x ),
                     np.maximum( 0, center_y-halfWidth_y), 
                     np.minimum( self.frameShape[0], center_y+halfWidth_y ) ]

        
    

        if self.sigmaMode:
            
            if np.isnan( self.__meanList[self.index] ):
                self.__meanList[self.index] = np.mean( self.im2show )
                self.__stdList[self.index] = np.std( self.im2show )
            clim_min = self.__meanList[self.index] - self.__sigmaLevels[self.__sigmaIndex]*self.__stdList[self.index]
            clim_max = self.__meanList[self.index] + self.__sigmaLevels[self.__sigmaIndex]*self.__stdList[self.index]
            tit += ", $\sigma$%.2f clim[%.1f,%.1f]" % (self.__sigmaLevels[self.__sigmaIndex], clim_min, clim_max)
        else:
            if np.isnan( self.__minList[self.index] ):
                self.__minList[self.index] = np.min( self.im2show )
                self.__maxList[self.index] = np.max( self.im2show )
            clim_min = self.__minList[self.index]
            clim_max = self.__maxList[self.index]
            tit += ", clim[%.1f,%.1f]" % (clim_min, clim_max)

        # LogNorm really isn't very failsafe...
#        if self.logMode:
#            norm = col.LogNorm()
#        else:
#            norm = None
        norm = None
        
        self.ax.set_title( tit )
        self.ax.imshow(self.im2show[ im_range[2]:im_range[3], im_range[0]:im_range[1] ], 
                       vmin=clim_min, vmax=clim_max, 
                       interpolation='none',
                       norm=norm,
                       extent=im_range, 
                        cmap=self.cmap )
        # plt.colorbar(self.ax)
        
        # Printing particle box overlay
        if bool(self.showBoxes) and np.any(self.boxYX) != None and self.boxLen > 0:
            # Coordinate systems are upside-down in y-axis?
            # box2 = int( self.boxLen/4 ) 
            dpi = self.fig.get_dpi()
            width = np.minimum( self.fig.get_figwidth(), self.fig.get_figheight() )
            
            # Ok I'm not getting draw events from resizing...
            markerSize =  (self.boxLen*width/dpi)**2
            print( "dpi = %d, width = %g, markerSize = %g" %(dpi,width, markerSize) )
            #for J in np.arange( self.boxYX.shape[0] ):
            #    box = self.boxYX[J,:]

                #boxCoord = np.array( [box+[-box2,-box2], box+[-box2,box2], 
                #                      box+[box2,box2], 
                #                      box+[box2,-box2], box+[-box2,-box2] ] )
                
            

            
#            self.ax.scatter( self.boxYX[:,1], self.boxYX[:,0], s=markerSize, color=colors, alpha=0.3  )
            self.ax.scatter( self.boxYX[:,1], self.boxYX[:,0], 
                            s=markerSize, color=self.boxColors, alpha=0.2, marker='s'  )
            plt.xlim( [im_range[0], im_range[1] ] )
            plt.ylim( [im_range[2], im_range[3] ] )
        
                        
        # RAM: This format_coord function is amazingly sensitive to minor changes and often breaks
        # the whole class.
        # DO NOT TOUCH format_coord!!!! 
        def format_coord(x, y):
            x = np.int(x + 0.5)
            y = np.int(y + 0.5)
            try:
                #return "%s @ [%4i, %4i]" % (round(im2show[y, x],2), x, y)
                return "%.5G @ [%4i, %4i]" % (self.im2show[y, x], y, x) #first shown coordinate is vertical, second is horizontal
            except IndexError:
                return ""
            
        self.ax.format_coord = format_coord
        # DO NOT TOUCH format_coord!!!! 
        
        if isinstance(self.titles, (list,tuple)) and len(self.titles) > 0:
            try:
                self.fig.canvas.set_window_title(self.titles[self.index])
            except: 
                self.fig.canvas.set_window_title(self.titles[0])
        elif isinstance( self.titles, str ):
            self.fig.canvas.set_window_title(self.titles)
                
        if 'qt' in plt.matplotlib.get_backend().lower():
            self.fig.canvas.manager.window.raise_() #this pops the window to the top    
        # TODO: X-Y profiles
#        if self.showProfiles:
#            posProf = self.posProfHoriz
#            self.axX.cla()
#            self.axX.plot(rx+1,self.im2show[posProf,rx])
##            plt.xlim(rx[0],rx[-1])
#            self.axX.set_xlim(rx[0],rx[-1])
            
        plt.show( block=self.blocking )   
            
    def printStat(self, mode='all'):
        
        if mode == 'all':
            modePrint = 'all frames'
            img = self.im
            if self.complex:            
                modePrint = 'the modulus'
                img = self.im[0,...]
        elif mode == 'curr':
            if self.im.ndim > 2:
                img = self.im[self.index, ...]
                modePrint = 'frame %d'%self.index             
            else:
                img = self.im
                modePrint = 'the current frame'
        else:
            print( "Unknown statistics mode: %s" % mode )
            return
            
        print( "===========================================" )             
        print( "Statistics of " + modePrint + " in figure %g:"%self.figNum)
        print( "Shape: ", img.shape             ) 
        print( "Maximum: ", img.max(), "@", np.unravel_index(np.argmax(img),img.shape))
        print( "Minimum: ", img.min(), "@", np.unravel_index(np.argmin(img),img.shape))
        print( "Center of mass:", ni.measurements.center_of_mass(img))
        print( "Mean: ", img.mean())
        print( "Standard deviation: ", img.std())
        print( "Variance: ", img.var()        )
        print( "Sum: ", img.sum())
        print( "Data type:", self.dtype)
        print( "===========================================" )  



    def __exit__(self, event):
        print( "Exiting IMS" )
        self.exiting = True
        self.fig.close()
        
    def __call__(self, event):
        redraw = False
        recompute = False
        # print( "Received key press %s" % event.key )

        if event.key=='n':#'up': #'right'
            if self.im.ndim > 2:
                self.index = np.minimum(self.im.shape[0]-1, self.index+1)
            recompute = True
        elif event.key == 'p':#'down': #'left'
            if self.im.ndim > 2:
                self.index = np.maximum(0, self.index-1)
            recompute = True
        if event.key=='N':#'up': #'right'
            if self.im.ndim > 2:        
                self.index = np.minimum(self.im.shape[0]-1, self.index+10)
            recompute = True
        elif event.key == 'P':#'down': #'left'
            if self.im.ndim > 2:        
                self.index = np.maximum(0, self.index-10)
            recompute = True
        elif event.key == 'v':
            self.doTranspose = True
            recompute = True
        elif event.key == 'l':
            self.logMode = not  self.logMode       
            recompute = True
        elif event.key == 'c':
            self.cmap = next( self.cmaps_cycle)
            redraw = True
        elif event.key == 'b':
            self.showBoxes = not self.showBoxes
            redraw = True
        elif event.key == 'h':
            self.sigmaMode = not self.sigmaMode
            redraw = True
        elif event.key == 'g':
            self.filterMode = not self.filterMode
            recompute = True
        elif event.key == 'k':
            self.__gaussSigma /= 1.5
            if self.filterMode:
                recompute = True
        elif event.key == 'm':
            self.__gaussSigma *= 1.5
            if self.filterMode:
                recompute = True
        elif event.key == 'F': # FFT
            self.fftMode = not self.fftMode
            recompute = True
        elif event.key == 'y': # polar (cYlindrical)
            self.polarMode = not self.polarMode            
            recompute = True
        elif event.key in 'SMV':
            self.projToggle = not self.projToggle
            self.projType = event.key
            recompute = True
        elif event.key == 'i':
            if 4*self.zoom < np.min(self.im.shape[1:]): # 2*zoom must not be bigger than shape/2
                self.zoom = 2*self.zoom
            redraw = True
        elif event.key == 'o':
            self.zoom = np.maximum(self.zoom/2,1)     
            redraw = True
            
        elif event.key == 'right':
            self.offx += self.stepXY
            self.offx = np.minimum(self.offx,self.im.shape[1]-1)
            redraw = True
        elif event.key == 'left':
            self.offx -= self.stepXY
            self.offx = np.maximum(self.offy,-self.im.shape[1]+1)            
            redraw = True
        elif event.key == 'down':
            self.offy += self.stepXY
            self.offy = np.minimum(self.offx,self.im.shape[2]-1)    
            redraw = True
        elif event.key == 'up':
            self.offx -= self.stepXY
            self.offx = np.maximum(self.offx,-self.im.shape[2]+1)
            redraw = True
        elif event.key == 'r': # reset position to the center of the image
            self.offx,self.offy = 0,0
            print( "Reseting positions to the center.")
            redraw = True
        elif event.key == 'R': # reset contrast
            self.offVmin,self.offVmax = 0,0
            print( "Reseting contrast.")
            redraw = True
        elif event.key == 'q': # increase contrast
            self.__sigmaIndex = np.maximum( self.__sigmaIndex-1, 0 )
            redraw = True
        elif event.key == 'Q': # increase contrast quickly
            self.__sigmaIndex = np.maximum( self.__sigmaIndex-10, 0 )
            redraw = True
        elif event.key == 'w': # decrease contrast
            self.__sigmaIndex = np.minimum( self.__sigmaIndex+1, self.__sigmaLevels.size-1 )
            redraw = True
        elif event.key == 'W': # decrease contrast quickly
            self.__sigmaIndex = np.minimum( self.__sigmaIndex+10, self.__sigmaLevels.size-1 )
            redraw = True
#            print "Increasing upper limit of the contrast: %g %% (press R to reset).\n"%round(self.offVmax*100)
        elif event.key == 'T': # print statistics of the whole dataset
            self.printStat()                  
            redraw = False
        elif event.key == 't': # print statistics of the current frame
            self.printStat(mode = 'curr'),                      
            redraw = False
        else:
            # Apparently we get multiple key-press events so don't do any error handling here.
            pass
            
        # Recompute is dominant over draw
        if recompute:
            self.__recompute__()
        elif redraw:
            self.__draw__()
        # self.fig.canvas.draw()

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

    
    # First argument is the executed file
    # print sys.argv 
    print( IMS_HELPTEXT )
    
    fftMode = False
    polarMode = False
    logMode = False
    if "--log" in sys.argv:
        logMode = True
    if "--fft" in sys.argv:
        fftMode = True
        logMode = True
    if "--polarfft" in sys.argv:
        fftMode = True
        polarMode = True
        logMode = True
    # Blocking seems to interrupt key presses?  I think I need a polling loop then.

        
    # http://matplotlib.org/users/event_handling.html
    #if os.name == "nt":
    #    blocking  = True
    #else:
        
    blocking = False 
        
    imsObj = ims( sys.argv[1], logMode=logMode, fftMode=fftMode, polarMode=polarMode, blocking=blocking )
    
    # plt.ion()
    # Need to hold here.
    # Doesn't work on Windows, why? Make plt.show( block=True ) call inside IMS instead
    while not imsObj.exiting:
        plt.pause(0.1)
    sys.exit()
    
#### COMMAND-LINE INTERFACE ####
if __name__ == '__main__':
    main()
    
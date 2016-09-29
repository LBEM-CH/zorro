# -*- coding: utf-8 -*-
#!/usr/bin/env python

"""
MplCanvas

This is a QWidget that can be used for fast-ish plotting within a Qt GUI interface.

Originally I was going to subclass for different types of plots, but this seems a 
little hard with the amount of initialization required to setup the plot properly 
within its parent frame, so we will draw based on class members.

Based on:
     embedding_in_qt4.py --- Simple Qt4 application embedding matplotlib canvases
    
     Copyright (C) 2005 Florent Rougon
                   2006 Darren Dale
    
     This file is an example program for matplotlib. It may be used and
     modified with no restriction; raw copies as well as modified versions
     may be distributed without limitation.
"""

from __future__ import division, print_function, absolute_import, unicode_literals

import os
import matplotlib
matplotlib.use( 'Qt4Agg' )
try:
    from PySide import QtGui
    matplotlib.rcParams['backend.qt4']='PySide'
    os.environ.setdefault('QT_API','pyside')
except:
    # Import PyQt4 as backup?
    print( "MplCanvas: PySide not found." )


from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
#from matplotlib.figure import Figure
import numpy as np
#from itertools import cycle
#from collections import OrderedDict
import skimage.io
from zorro import plot as zplt
import subprocess
import tempfile

# How to design custom controls with PyQT:
# http://doc.qt.digia.com/qq/qq26-pyqtdesigner.html


class MplCanvas(FigureCanvas,object):
    """This is an empty QWidget of type FigureCanvasQTAgg.  Uses a zorro_plotting.zorroPlot object to do all 
    the live plotting, or it can load graphics files from disk."""
    
    @property
    def zorroObj(self):
        return self._zorroObj     
        
    @zorroObj.setter
    def zorroObj(self, newZorroObj ):
        #print( "Set _zorroObj" )
        
        if not bool( newZorroObj ):
            return
        
        self._zorroObj = newZorroObj
        
         # Used for mapping combo box text to files in the zorroObj
        # baseName should be location of the config file
        baseDir = ''
#        if 'config' in self._zorroObj.files:
#            baseDir = os.path.split( self._zorroObj.files['config'] )[0]
        
        if 'figBoxMask' in self._zorroObj.files:
            # This isn't here... it's next to sum...
            self.pixmapDict[u'Box Mask'] = os.path.join( baseDir, self._zorroObj.files['figBoxMask'] )
        if 'figStats' in self._zorroObj.files:
            self.pixmapDict[u'Statistics'] = os.path.join( baseDir, self._zorroObj.files['figStats'] )
        if 'figTranslations' in self._zorroObj.files:
            self.pixmapDict[u'Drift'] = os.path.join( baseDir, self._zorroObj.files['figTranslations'] )
        if 'figPixRegError' in self._zorroObj.files:
            self.pixmapDict[u'Drift error'] = os.path.join( baseDir, self._zorroObj.files['figPixRegError'] )
        if 'figPeaksigTriMat' in self._zorroObj.files:
            self.pixmapDict[u'Peak significance'] = os.path.join( baseDir, self._zorroObj.files['figPeaksigTriMat'] )
        if 'figCorrTriMat' in self._zorroObj.files:
            self.pixmapDict[u'Correlation coefficient'] = os.path.join( baseDir, self._zorroObj.files['figCorrTriMat'] )
        if 'figCTFDiag' in self._zorroObj.files:
            self.pixmapDict[u'CTF diagnostic'] = os.path.join( baseDir, self._zorroObj.files['figCTFDiag'] )
        if 'figLogisticWeights' in self._zorroObj.files:
            self.pixmapDict[u'Logistic weights'] = os.path.join( baseDir, self._zorroObj.files['figLogisticWeights'] )
        if 'figImageSum' in self._zorroObj.files:
            self.pixmapDict[u'Image sum'] = os.path.join( baseDir, self._zorroObj.files['figImageSum'] )
        if 'figFFTSum' in self._zorroObj.files:
            self.pixmapDict[u'Fourier mag'] = os.path.join( baseDir, self._zorroObj.files['figFFTSum'] )
        if 'figPolarFFTSum' in self._zorroObj.files:
            self.pixmapDict[u'Polar mag'] = os.path.join( baseDir, self._zorroObj.files['figPolarFFTSum'] )
        if 'figFiltSum' in self._zorroObj.files:
            self.pixmapDict[u'Dose filtered sum'] = os.path.join( baseDir, self._zorroObj.files['figFiltSum'] )
        if 'figFRC' in self._zorroObj.files:
            self.pixmapDict[u'Fourier Ring Correlation'] = os.path.join( baseDir, self._zorroObj.files['figFRC'] )
        
    def __init__(self, parent=None, width=4, height=4, plot_dpi=72, image_dpi=250):
        
        object.__init__(self)
        self.plotObj = zplt.zorroPlot( width=width, height=height, 
                                      plot_dpi=plot_dpi, image_dpi=image_dpi,
                                      facecolor=[0,0,0,0], MplCanvas=self )
        FigureCanvas.__init__(self, self.plotObj.fig)
        
        
        self.currPlotFunc = self.plotObj.plotTranslations
        
        self.cmap = 'gray'
        self._zorroObj = None
        
        self.plotName = None
        self.live = True # Whether to re-render the plots with each update event or use a rendered graphics-file loaded from disk
        self.PixmapName = None
        self.Pixmap = None

        # plotFuncs is a hash to function mapping
        # These may need to add the appropriate data to plotDict?  I could use functools.partial?
        self.plotFuncs = {}
        self.plotFuncs[""] = None
        self.plotFuncs[u'Statistics'] = self.plotObj.plotStats
        self.plotFuncs[u'Drift'] = self.plotObj.plotTranslations
        self.plotFuncs[u'Drift error'] = self.plotObj.plotPixRegError
        self.plotFuncs[u'Peak significance'] = self.plotObj.plotPeaksigTriMat
        self.plotFuncs[u'Correlation coefficient'] = self.plotObj.plotCorrTriMat
        self.plotFuncs[u'CTF diagnostic'] = self.plotObj.plotCTFDiag
        self.plotFuncs[u'Logistic weights'] = self.plotObj.plotLogisticWeights
        self.plotFuncs[u'Stack'] = self.plotObj.plotStack
        self.plotFuncs[u'Image sum'] = self.plotObj.plotImage
        self.plotFuncs[u'Fourier mag'] = self.plotObj.plotFFT
        self.plotFuncs[u'Polar mag'] = self.plotObj.plotPolarFFT
        self.plotFuncs[u'Cross correlations'] = self.plotObj.plotStack # TODO
        self.plotFuncs[u'Dose filtered sum'] = self.plotObj.plotImage
        self.plotFuncs[u'Fourier Ring Correlation'] = self.plotObj.plotFRC
        
        self.liveFuncs = {}
        self.liveFuncs[u'Statistics'] = self.liveStats
        self.liveFuncs[u'Image sum'] = self.liveImageSum
        self.liveFuncs[u'Dose filtered sum'] = self.liveFiltSum
        self.liveFuncs[u'Drift'] = self.liveTranslations
        self.liveFuncs[u'Drift error'] = self.livePixRegError
        self.liveFuncs[u'Peak significance'] = self.livePeaksigTriMat
        self.liveFuncs[u'Correlation coefficient'] = self.livePeaksigTriMat
        self.liveFuncs[u'Logistic weights'] = self.liveLogisticWeights
        self.liveFuncs[u'Fourier Ring Correlation'] = self.liveFRC
        self.liveFuncs[u'CTF diagnostic'] = self.liveCTFDiag
        
        self.pixmapDict = {}
        # WARNING WITH SPYDER: Make sure PySide is the default in the console
        # self.setSizePolicy(self, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
        self.updateGeometry()
        
    ##### 2DX VIEW #####
    def exportTo2dx( self ):
        # Write a params file
        #paramFile = tempfile.mktemp()
        #with open( paramFile, 'w' ):
        #    pass
        
        # Temporary directory that we can delete?  We could use tempfile
    
        # Invoke
        #subprocess.Popen( "2dx_viewer -p %s %s" % (paramFile) )
        # When to delete paramFile?
        if self.plotName == u'Dose filtered sum':
            realPath = os.path.realpath( self._zorroObj.files['filt'] )
            subprocess.Popen( "2dx_viewer %s" % (realPath), shell=True )
        elif self.plotName == u'Image sum':
            realPath = os.path.realpath( self._zorroObj.files['sum'] )
            subprocess.Popen( "2dx_viewer %s" % (realPath), shell=True )
        else:
            print( "Unsupported plot function for 2dx_viewer" )
        pass
    
    def exportToIms( self ):
        if self.plotName == u'Dose filtered sum':
            realPath = os.path.realpath( self._zorroObj.files['filt'] )
            subprocess.Popen( "ims %s" % (realPath), shell=True )
        elif self.plotName == u'Image sum':
            realPath = os.path.realpath( self._zorroObj.files['sum'] )
            subprocess.Popen( "ims %s" % (realPath), shell=True )
        else:
            print( "Unsupported plot function for ims" )
        pass
    
    ##### LIVE VIEW #####
    def livePlot(self, plotName ):
        print( "called livePlot" )
        # Check the plotObj's plotDict for correct fields
        # Do seperate sub-functions for each plot type?
        if self._zorroObj == None:
            return
            
        if plotName in self.liveFuncs:
            self.liveFuncs[plotName]()
        else:
            print( "Live function: %s not found." % plotName )
            self.currPlotFunc = self.plotObj.plotEmpty
        # Plot
        self.currPlotFunc()
        self.redraw()
        
    def liveStats( self ):
        self.plotObj.plotDict['pixelsize'] = self._zorroObj.pixelsize
        self.plotObj.plotDict['voltage'] = self._zorroObj.voltage
        self.plotObj.plotDict['c3'] = self._zorroObj.C3
        
        if len( self._zorroObj.errorDictList ) > 0 and 'peaksigTriMat' in self._zorroObj.errorDictList[-1]:
            peaksig = self._zorroObj.errorDictList[-1]['peaksigTriMat']
            peaksig = peaksig[ peaksig > 0.0 ]
            self.plotObj.plotDict['meanPeaksig'] = np.mean( peaksig )
            self.plotObj.plotDict['stdPeaksig'] = np.std( peaksig )
        if np.any( self._zorroObj.CTFInfo['DefocusU'] ):
            self.plotObj.plotDict['CTFInfo'] = self._zorroObj.CTFInfo
        
        self.currPlotFunc = self.plotObj.plotStats

    def liveImageSum( self ):
        try:
            if not np.any(self._zorroObj.imageSum): # Try to load it 
                self._zorroObj.loadData( stackNameIn = self._zorroObj.files['sum'], target="sum" )
                
            self.plotObj.plotDict['image'] = self._zorroObj.getSumCropToLimits()
            self.plotObj.plotDict['image_cmap'] = self.cmap
            self.currPlotFunc = self.plotObj.plotImage
        except:
            self.currPlotFunc = self.plotObj.plotEmpty
            
    def liveFiltSum( self ):
        try:
            if not np.any(self._zorroObj.filtSum): # Try to load it 
                self._zorroObj.loadData( stackNameIn = self._zorroObj.files['filt'], target="filt" )
                
            self.plotObj.plotDict['image'] = self._zorroObj.getFiltSumCropToLimits()
            self.plotObj.plotDict['image_cmap'] = self.cmap
            self.currPlotFunc = self.plotObj.plotImage
        except:
            self.currPlotFunc = self.plotObj.plotEmpty
            
            
    def liveTranslations( self ):
        if np.any( self._zorroObj.translations ):
            self.plotObj.plotDict['translations'] = self._zorroObj.translations
            try:
                self.plotObj.plotDict['errorX'] = self._zorroObj.errorDictList[0]['errorX']
                self.plotObj.plotDict['errorY'] = self._zorroObj.errorDictList[0]['errorY']
            except: pass

            self.currPlotFunc = self.plotObj.plotTranslations
        else:
            self.currPlotFunc = self.plotObj.plotEmpty
    
    def livePixRegError( self ):
        try:
            self.plotObj.plotDict['errorX'] = self._zorroObj.errorDictList[0]['errorX']
            self.plotObj.plotDict['errorY'] = self._zorroObj.errorDictList[0]['errorY']
            self.plotObj.plotDict['errorXY'] = self._zorroObj.errorDictList[0]['errorXY']
            self.currPlotFunc = self.plotObj.plotPixRegError
        except:
            self.currPlotFunc = self.plotObj.plotEmpty
            
    def livePeaksigTriMat( self ):
        try:
            self.plotObj.plotDict['peaksigTriMat'] = self._zorroObj.errorDictList[0]['peaksigTriMat']
            self.plotObj.plotDict['graph_cmap'] = self.cmap
            self.currPlotFunc = self.plotObj.plotPeaksigTriMat
        except:
            self.currPlotFunc = self.plotObj.plotEmpty
            
    def liveCorrTriMat( self ):
        try:
            self.plotObj.plotDict['corrTriMat'] = self._zorroObj.errorDictList[0]['corrTriMat']
            self.plotObj.plotDict['graph_cmap'] = self.cmap
            self.currPlotFunc = self.plotObj.plotCorrTriMat
        except:
            self.currPlotFunc = self.plotObj.plotEmpty        
    
    def liveLogisticWeights( self ):
        try:
            if self._zorroObj.weightMode == 'autologistic' or self._zorroObj.weightMode == 'logistic':
                self.plotObj.plotDict['peaksigThres'] = self._zorroObj.peaksigThres
                self.plotObj.plotDict['logisticK'] = self._zorroObj.logisticK
                self.plotObj.plotDict['logisticNu'] = self._zorroObj.logisticNu
                
            self.plotObj.plotDict['errorXY'] = self._zorroObj.errorDictList[0]["errorXY"]
            self.plotObj.plotDict['peaksigVect'] = self._zorroObj.errorDictList[0]["peaksigTriMat"][ self._zorroObj.errorDictList[0]["peaksigTriMat"] > 0.0  ]
            
            if 'cdfPeaks' in self._zorroObj.errorDictList[0]:
               self.plotObj.plotDict['cdfPeaks'] = self._zorroObj.errorDictList[0]['cdfPeaks']
               self.plotObj.plotDict['hSigma'] = self._zorroObj.errorDictList[0]['hSigma']
               
            self.currPlotFunc = self.plotObj.plotLogisticWeights
        except Exception as e:
            print( "MplCanvas.liveLogisticWeights received exception " + str(e) )
            self.currPlotFunc = self.plotObj.plotEmpty    
            
    def liveFRC( self ):
        try:
            self.plotObj.plotDict['FRC'] = self._zorroObj.FRC
            self.plotObj.plotDict['pixelsize'] = self._zorroObj.pixelsize
            
            if bool( self.zorroObj.doEvenOddFRC ):
                self.plotObj.plotDict['labelText'] = "Even-odd frame independent FRC"
            else:
                self.plotObj.plotDict['labelText'] = "Non-independent FRC is not a resolution estimate"
            
            self.currPlotFunc = self.plotObj.plotFRC
        except:
            self.currPlotFunc = self.plotObj.plotEmpty    
            
    def liveCTFDiag( self ):
        try:
            self.plotObj.plotDict['CTFDiag'] = self._zorroObj.CTFDiag
            self.plotObj.plotDict['CTFInfo'] = self._zorroObj.CTFInfo
            self.plotObj.plotDict['pixelsize'] = self._zorroObj.pixelsize
            self.plotObj.plotDict['image_cmap'] = self.cmap
            self.currPlotFunc = self.plotObj.plotCTFDiag
        except:
            self.currPlotFunc = self.plotObj.plotEmpty    
            
    ##### DEAD VIEW #####
    def loadPixmap( self, plotName, filename = None ):
        if not bool(filename):
            # Pull the filename from the zorro log
            try:
                # print( plotName )
                filename = self.pixmapDict[plotName]
                print( "Pulling figure name: %s"%filename )
            except KeyError:
                self.currPlotFunc = self.plotObj.plotEmpty()
                self.redraw()
                
        if not bool( filename ): # Probably an unprocessed stack
            return
            
        if not os.path.isfile(filename):
            raise IOError("automator.MplCanvas.loadPixmap: file not found: %s" % filename )
            return
        
        self.PixmapName = filename
        self.Pixmap = skimage.io.imread( filename )
        self.plotObj.plotDict['pixmap'] = self.Pixmap
        self.currPlotFunc = self.plotObj.plotPixmap()
        self.redraw()
        
    def updatePlotFunc(self, plotName, newZorroObj = None ):
        
        # print( "plotName = " + str(plotName) +", zorroObj = " + str(newZorroObj) )
        try:
            self.plotName = plotName
            self.currPlotFunc = self.plotFuncs[ plotName ]
        except KeyError:
            raise KeyError( "automator.MplCanvas.updatePlotFunc: Plot type not found in plotDict: %s" % plotName )
            
        self.zorroObj = newZorroObj # setter auto-checks validity... settler isn't working right...
        
        if bool( self.live ):
            self.plotObj.axes2 = None
            self.livePlot( plotName )
        else:
            self.loadPixmap( plotName )
        
    def redraw(self):
        #self.plotObj.updateCanvas()
        self.draw()



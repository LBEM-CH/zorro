# -*- coding: utf-8 -*-
"""
ViewWidget 

Basically a toolbar and key-press manager for a MplCanvas.

Created on Mon May 18 11:45:47 2015

@author: Robert A. McLeod
"""
from __future__ import division, print_function, absolute_import, unicode_literals

from PySide import QtCore, QtGui

try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    def _fromUtf8(s):
        return s

try:
    _encoding = QtGui.QApplication.UnicodeUTF8
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig, _encoding)
except AttributeError:
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig)
        
import os.path
import numpy as np
#from copy import copy
import functools
# Import static types from MplCanvas
#from . import MplCanvas
import matplotlib.image
from . import Ui_ViewWidget
from zorro.zorro_util import which

class ViewWidget(QtGui.QWidget, Ui_ViewWidget.Ui_ViewWidget, object):
    
    @property 
    def live( self ):
        return self._live
    
    @live.setter
    def live( self, value ):
        value = bool(value)
        if value != self._live:
            self._live = value
            self.tbLive.setChecked(value)
            self.viewCanvas.live = value
            self.tbChangeColormap.setEnabled(value)
            self.tbLogIntensity.setEnabled(value)
            self.tbToggleColorbar.setEnabled(value)
            self.tbToggleHistogramContrast.setEnabled(value)
            self.viewCanvas.updatePlotFunc( self.comboView.currentText() )
            
    def __init__( self, parent=None ):
        """
        QtGui.QWidget.__init__(self)
        self.ui = Ui_ViewWidget()
        
        self.ui.setupUi( self )
        """
        object.__init__(self)
        QtGui.QWidget.__init__(self)
        

        
        # Using multiple inheritence setup Ui from QtDesigner
        self.setupUi(self)
        
        self.parent = parent # This is generally not what I need, what I need is the Automator object
        self.autoParent = None # Set by hand in the Automator.__init__() function
        self.popout = "2dx_viewer"
        self.__popoutObj = None

        self.viewNumber = 0
        self._live = True
        
        # Re-direct keyPress functions
        self.keyPressEvent = self.grabKey
        # Run through all the widgets and redirect the key-presses to everything BUT the spinboxes
#        widgetlist = self.ui.centralwidget.findChildren( QtGui.QWidget )        
#        print "TO DO: release focus from spinboxes on ENTER key press" 
#        for mywidget in widgetlist:
#            # print "Pause" 
#            if not mywidget.__class__ is QtGui.QDoubleSpinBox:
#                mywidget.keyPressEvent = self.grabKey
        

        # Set paths to icons to absolute paths
        self.joinIconPaths()
        
        # Connect slots
        self.comboView.currentIndexChanged.connect( self.updatePlotType )

        
        self.tbNextImage.clicked.connect( functools.partial( self.shiftImageIndex, 1 ) )
        self.tbPrevImage.clicked.connect( functools.partial( self.shiftImageIndex, -1 ) )
        
        self.tbShowBoxes.toggled.connect( self.toggleShowBoxes )
        
        # self.leImageIndex.textChanged( self.updateImageIndex )
        self.leImageIndex.textEdited.connect( self.updateImageIndex )
        self.tbToggleColorbar.toggled.connect( self.toggleColorbar )
        self.tbChangeColormap.clicked.connect( self.cycleColormap )
        self.tbLogIntensity.toggled.connect( self.toggleLogInt )
        self.tbPopoutView.clicked.connect( self.popoutViewDialog )
        self.tbLive.toggled.connect( self.toggleLiveView )
        self.sbHistogramCutoff.valueChanged.connect( self.updateHistClim )
        
        # This doesn't work because there's two types of valueChanged sent
        # BUT ONLY WHEN AN IMAGE IS LOADED...
        
        # Has another example with new types
        # http://pyqt.sourceforge.net/Docs/PyQt4/new_style_signals_slots.html
        # Try oldschool connect, deprecated not working?
        # self.connect( self.sbHistogramCutoff, QtCore.SIGNAL('valueChanged(double)'), self.updateHistClim )
        # self.connect
        # This says it's a timer thing due to mouse presses:
        # http://www.qtcentre.org/threads/43078-QSpinBox-Timer-Issue
        self.sbHistogramCutoff.validate = None
    
    
    def grabKey( self, event ):
        # I think some of these key presses aren't being intercepted?
        print( "ViewWidget"+str(self.viewNumber)+"::grabKey : " + str(event.key()) )
        if( event.key() == QtCore.Qt.Key_Down ):
            print( "Down" )  
        elif( event.key() == QtCore.Qt.Key_Up ):
            print( "Up" )  
        elif( event.key() == QtCore.Qt.Key_Left ):
            print( "Left" )  
        elif( event.key() == QtCore.Qt.Key_Right ):
            print( "Right" )  
        elif( event.key() == QtCore.Qt.Key_PageUp ):
            print( "PageUp" )  
        elif( event.key() == QtCore.Qt.Key_PageDown ):
            print( "PageDown" )   
        else:
            return
            
    def joinIconPaths(self):
        # Icon's aren't pathed properly if the CWD is somewhere else than the source folder, so...
        self.source_dir = os.path.dirname( os.path.realpath(__file__) )
        # Join all the icons and reload them
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(os.path.join( self.source_dir, "icons/application-resize.png")), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.tbPopoutView.setIcon(icon)
        icon1 = QtGui.QIcon()
        icon1.addPixmap(QtGui.QPixmap(os.path.join( self.source_dir, "icons/monitor-dead.png")), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        icon1.addPixmap(QtGui.QPixmap(os.path.join( self.source_dir, "icons/monitor-live.png")), QtGui.QIcon.Normal, QtGui.QIcon.On)
        self.tbLive.setIcon(icon1)
        icon2 = QtGui.QIcon()
        icon2.addPixmap(QtGui.QPixmap(os.path.join( self.source_dir, "icons/color.png")), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.tbChangeColormap.setIcon(icon2)
        icon3 = QtGui.QIcon()
        icon3.addPixmap(QtGui.QPixmap(os.path.join( self.source_dir, "icons/colorbar.png")), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.tbToggleColorbar.setIcon(icon3)
        icon4 = QtGui.QIcon()
        icon4.addPixmap(QtGui.QPixmap(os.path.join( self.source_dir, "icons/magnifier-zoom-in.png")), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.tbZoomIn.setIcon(icon4)
        icon5 = QtGui.QIcon()
        icon5.addPixmap(QtGui.QPixmap(os.path.join( self.source_dir, "icons/magnifier-zoom-out.png")), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.tbZoomOut.setIcon(icon5)
        icon6 = QtGui.QIcon()
        icon6.addPixmap(QtGui.QPixmap(os.path.join( self.source_dir, "icons/boxes.png")), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.tbShowBoxes.setIcon(icon6)
        icon7 = QtGui.QIcon()
        icon7.addPixmap(QtGui.QPixmap(os.path.join( self.source_dir, "icons/logscale.png")), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.tbLogIntensity.setIcon(icon7)
        icon8 = QtGui.QIcon()
        icon8.addPixmap(QtGui.QPixmap(os.path.join( self.source_dir,  "icons/histogram.png")), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.tbToggleHistogramContrast.setIcon(icon8)
        icon9 = QtGui.QIcon()
        icon9.addPixmap(QtGui.QPixmap(os.path.join( self.source_dir, "icons/arrow-180.png")), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.tbPrevImage.setIcon(icon9)
        icon10 = QtGui.QIcon()
        icon10.addPixmap(QtGui.QPixmap(os.path.join( self.source_dir, "icons/arrow.png")), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.tbNextImage.setIcon(icon10)

    def toggleLiveView( self ):
        self.live = self.tbLive.isChecked()
        
    def toggleShowBoxes( self ):
        print( "Trying box overlay" )
        self.viewCanvas.plotObj.plotDict['boxMask'] = None
        if self.tbShowBoxes.isChecked():
            # Load from the zorroObj
            try:
                print( "figBoxMask: " + str(self.viewCanvas.zorroObj.files['figBoxMask']) )
            except:
                pass
            if 'figBoxMask' in self.viewCanvas.zorroObj.files:
                self.viewCanvas.plotObj.plotDict['boxMask'] = matplotlib.image.imread( self.viewCanvas.zorroObj.files['figBoxMask'] )
            
        self.viewCanvas.updatePlotFunc( self.comboView.currentText() )
        
    # Decorators to stop multiple events (doesn't work damnit, everything is float)
#    @QtCore.Slot(float)    
#    @QtCore.Slot(str)
    def updateHistClim( self, value ):
        self.viewCanvas.param['cutoff'] = np.power( 10.0, self.sbHistogramCutoff.value())

        self.viewCanvas.updatePlotFunc( self.comboView.currentText() )
        pass
    
    def toggleLogInt( self ):
        # This should call updateHistClim, not sure if I want too
        self.viewCanvas.param['logScale'] = ~self.viewCanvas.param['logScale']
        self.viewCanvas.updatePlotFunc( self.comboView.currentText() )
        
    def popoutViewDialog( self ):
        """
        So the logic here has 
        """
        if (self.viewCanvas.plotName == u'Dose filtered sum' 
            or self.viewCanvas.plotName == u'Image sum'):
            if self.popout == "2dx_viewer" and which( "2dx_viewer" ):
                self.viewCanvas.exportTo2dx()
                return
            elif self.popout == "ims" and which( "ims" ):
                self.viewCanvas.exportToIms()
                return
        
        # Fallback mode
        self.__popoutObj = ViewDialog()
        self.copyDeep( self.__popoutObj.view ) # ViewDialog is just a wrapper around ViewWidget 'view'

    
    # Unfortunately the copy library doesn't work nicely with Qt, so we have to implement this.
    def copyDeep( self, thecopy ):
        thecopy.viewNumber = self.viewNumber + 100
        thecopy.parent = self.parent
        thecopy.autoParent = self.autoParent
        # thecopy.viewCanvas = copy( self.viewCanvas )
        thecopy.updateZorroObj( self.viewCanvas.zorroObj )
        
        # No copy of popout
        # Turn events OFF
        thecopy.blockSignals( True )
        print( "BLOCKING SIGNALS" )
        thecopy.tbToggleColorbar.setChecked( self.tbToggleColorbar.isChecked() )
        thecopy.tbLogIntensity.setChecked( self.tbLogIntensity.isChecked() )
        thecopy.tbToggleHistogramContrast.setChecked( self.tbToggleHistogramContrast.isChecked() )
        thecopy.leImageIndex.setText( self.leImageIndex.text() )
        thecopy.sbHistogramCutoff.blockSignals( True )
        thecopy.sbHistogramCutoff.setValue( self.sbHistogramCutoff.value() )
        thecopy.sbHistogramCutoff.blockSignals( False )
        thecopy.comboView.setCurrentIndex( self.comboView.currentIndex() )
        thecopy.updatePlotType(0)
        thecopy.blockSignals( False )
        print( "UNBLOCKING SIGNALS" )
        
    def toggleColorbar( self ):
        self.viewCanvas.plotObj.plotDict['colorbar'] = self.tbToggleColorbar.isChecked()
        self.viewCanvas.updatePlotFunc( self.comboView.currentText() )
        
    def cycleColormap( self ):
        # This is sort of dumb, just have a function inside zorroPlot for this.
        self.viewCanvas.cmap = self.viewCanvas.plotObj.cmaps_cycle.next() 
        self.viewCanvas.plotObj.plotDict['image_cmap'] = self.viewCanvas.cmap
        self.viewCanvas.plotObj.plotDict['graph_cmap'] = self.viewCanvas.cmap
        self.viewCanvas.updatePlotFunc( self.comboView.currentText() )
        
    def shiftImageIndex( self, shift=1 ):
        newIndex = np.int32( self.leImageIndex.text() )
        newIndex += shift
        self.updateImageIndex( imageIndex = newIndex )
        
    def updateImageIndex( self, imageIndex=None ):
        if imageIndex is None:
            imageIndex = np.int( self.leImageIndex.text() )
            
        self.viewCanvas.param['imageIndex'] = imageIndex
        self.refreshCanvas()
        self.leImageIndex.blockSignals( True )
        self.leImageIndex.setText( "%s"%imageIndex ) 
        self.leImageIndex.blockSignals( False )
    
    def updateZorroObj( self, zorroObj = None ):
        self.viewCanvas.updatePlotFunc( self.comboView.currentText(), zorroObj )
            
    def updatePlotType( self, index ):
        # This function is called when you need to update the underlying data of the canvas
        self.viewCanvas.updatePlotFunc( self.comboView.currentText() )
            
        
    def loadConfig( self, config ):
        groupstring = u"view%d" % self.viewNumber
       
        try:  self.comboView.setCurrentIndex( self.comboView.findText( config.get( groupstring, u'plotType' ) ) )
        except: print( "Failed to set plotType for view %d"%self.viewNumber )
        
        try: 
            self.live = config.getboolean( groupstring, u'live')

        except: pass
        try: self.tbToggleColorbar.checked( config.getboolean( groupstring, u'colorBar') )
        except: pass
        try: self.tbToggleHistogramContrast.checked( config.getboolean( groupstring, u'histogramContrast') )
        except: pass
        try: self.tbLogIntensity.checked( config.getboolean( groupstring, u'logIntensity') )
        except: pass
        try: self.sbHistogramCutoff.setValue( config.getint( groupstring, u'histogramCutoff') )
        except: pass
        try: self.tbShowBoxes.checked( config.getboolean( groupstring, u'showBoxes') )
        except: pass
    
    
        try: self.viewCanvas.plotObj.plotDict['image_cmap'] = config.get( groupstring, u'image_cmap' )
        except: pass
        try: self.viewCanvas.plotObj.plotDict['graph_cmap'] = config.get( groupstring, u'graph_cmap' )
        except: pass
        
    def saveConfig( self, config ):
        groupstring = u"view%d" % self.viewNumber
        config.add_section(groupstring)
        config.set( groupstring, u'plotType', self.comboView.currentText() )
        config.set( groupstring, u'live', self.tbLive.isChecked() )
        config.set( groupstring, u'colorBar', self.tbToggleColorbar.isChecked() )
        config.set( groupstring, u'histogramContrast', self.tbToggleHistogramContrast.isChecked() )
        config.set( groupstring, u'logIntensity', self.tbLogIntensity.isChecked() )
        config.set( groupstring, u'histogramCutoff', self.sbHistogramCutoff.value() )
        config.set( groupstring, u'showBoxes', self.tbShowBoxes.isChecked() )
        
        # We can only save some plotDict keys because it might contain a lot of data!
        try: config.set( groupstring, u'image_cmap', self.viewCanvas.plotObj.plotDict['image_cmap'] )
        except: pass
        try: config.set( groupstring, u'graph_cmap', self.viewCanvas.plotObj.plotDict['graph_cmap'] )
        except: pass
        pass

    
    
from . import Ui_ViewDialog
class ViewDialog(QtGui.QDialog, Ui_ViewDialog.Ui_ViewDialog):
    
    def __init__(self):
        QtGui.QDialog.__init__(self)
        
        self.setupUi(self)
        
        # Set up the user interface from Designer.  
        self.show()
        
        
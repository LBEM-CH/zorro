# -*- coding: utf-8 -*-
"""
###################################################################
#  Automator for Zorro and 2dx
#
#      License: LGPL
#      Author:  Robert A. McLeod
#      Website: https://github.com/C-CINA/zorro-dev
#
#  See LICENSE.txt for details about copyright and
#  rights to use.
####################################################################

This is a new PySide/Qt4 Gui for Zorro and 2dx.  It's capable of running multiple subprocesses at the same 
time, without resorting to the buggy Python multiprocessing module.  

####################### BUILDING PYSIDE FROM GIT
http://pyside.readthedocs.org/en/latest/building/linux.html
#######################
"""

from __future__ import division, print_function, absolute_import, unicode_literals

import os, os.path, sys, glob
try:
    from PySide import QtGui, QtCore
    
    
except:
    # Import PyQt4 as backup?  I suspect this still causes license issues
    ImportError( "Automator.py: PySide not found, I am going to crash now. Bye." )

import zorro
from . import zorroSkulkManager

# from copy import deepcopy
from . import Ui_Automator
from . import Ui_DialogFileLoc
from . import Ui_DialogOrientGainRef

# import time
import functools

try:
    import Queue as queue
except:
    import queue
    
import json
try:
    import ConfigParser as configparser
except:
    import configparser
import numpy as np


#progname = os.path.basename(sys.argv[0])
#progversion = u"0.7.1b0"
from .__version__ import __version__

#STATE_COLORS = { NEW:u'darkorange', CHANGING:u'darkorange', STABLE:u'goldenrod', SYNCING:u'goldenrod', READY:u'forestgreen',
#                PROCESSING:u'forestgreen', FINISHED:u'indigo', ARCHIVING:u'saddlebrown', COMPLETE:u'dimgrey', 
#                STALE:u'firebrick', HOST_BUSY:u'host_busy', HOST_FREE:u'host_free', HOST_ERROR:u'firebrick', 
#                RENAME:u'rename' }

TOOLTIP_STATUS = { u'darkorange':u'New', u'goldenrod':u'Stable', u'indigo':u'Finished', 
                  u'saddlebrown':u'Archiving', u'forestgreen':u'Ready', u'steelblue':'Processing',
                  u'dimgrey':u'Complete', u'firebrick':u'Error', u'black':u'Unknown',
                  u'deeppink':u'Syncing', u'':u'Unknown', u'rename':u'Renaming' }

#class Automator(Ui_Automator_ui, QtGui.QApplication):
class Automator(Ui_Automator.Ui_Automator_ui, QtGui.QApplication):
    
    def __init__(self, testing=False ):
        # Hack to get icons to show up...
        # Better approach would be to give each one the real path.
        # origdir = os.path.realpath('.')
        
        
        # os.chdir( os.path.dirname(os.path.realpath(__file__)) )
        QtGui.QApplication.__init__(self, sys.argv)
        #self.app = QtGui.QApplication(sys.argv)
        self.MainWindow = QtGui.QMainWindow()
        self.ImageDialog = QtGui.QDialog()
        self.FileLocDialog = QtGui.QDialog()
        self.OrienGainRefDialog = QtGui.QDialog()

        self.setupUi(self.MainWindow)
        self.ui_FileLocDialog = Ui_DialogFileLoc.Ui_DialogFileLocations()
        self.ui_FileLocDialog.setupUi( self.FileLocDialog )
        
        self.ui_OrienGainRefDialog = Ui_DialogOrientGainRef.Ui_DialogOrientGainRef()
        self.ui_OrienGainRefDialog.setupUi( self.OrienGainRefDialog )
        
        # Force absolute paths to icons
        self.joinIconPaths()
        

        
        
        # Zorro objects and the default one built from the GUI
        self.zorroDefault = zorro.ImageRegistrator()
        
        self.skulkMessanger = queue.Queue()
        self.skulk = zorroSkulkManager.skulkManager( self.skulkMessanger )
        # Connect skulk.automatorSignal to myself
        self.skulk.automatorSignal.connect( self.updateFromSkulk )
        self.skulkThread = None
        
        self.cfgfilename = ""
        self.initClusterConfig()
        
        # I should populate these with the default values from QtDesigner
        self.cfgCommon = {}
        self.cfgGauto = {}
        self.cfgGplot = {}
        
        self.stateIds = {} # Link unique Id to name
        self.reverseIds = {} # Opposite to above
        

        self.statusbar.showMessage( "Welcome to Automator for Zorro and 2dx, version " + __version__ )

        
        
        # Apply all the connections from the QT gui objects to their associated functions in this class, using functools as desired
        # Menu items
        self.actionQuit.triggered.connect( self.quitApp )
        self.MainWindow.closeEvent = self.quitApp
        self.actionLoad_config.triggered.connect( functools.partial(self.loadConfig, None) )
        self.actionSave_config.triggered.connect( functools.partial(self.saveConfig, None) )
        self.actionSet_paths.triggered.connect( self.FileLocDialog.show )
        self.actionCitations.triggered.connect( self.showCitationsDialog )
        self.actionIMS_shortcuts.triggered.connect( self.showImsHelpDialog )
        self.actionOrient_Gain_Reference.triggered.connect( self.OrienGainRefDialog.show )
        
        self.actionGroupPopout = QtGui.QActionGroup(self)
        self.actionGroupPopout.addAction( self.actionPrefer2dx_viewer )
        self.actionGroupPopout.addAction( self.actionPreferIms )
        self.actionGroupPopout.addAction( self.actionPreferMplCanvas )
        self.actionGroupPopout.triggered.connect( self.preferPopout )
        # Enable sorting for the file list
        self.listFiles.setSortingEnabled( True )
        
        
        #==============================================================================
        # CAREFUL WITH FUNCTOOLS, IF OBJECTS ARE CREATED AND DESTROYED THEY AREN'T
        # UPDATED WITH THE PARTIAL CONTEXT
        #==============================================================================
        
        # Paths
        self.ui_FileLocDialog.tbOpenInputPath.clicked.connect( functools.partial( 
            self.openPathDialog, u'input_dir', True ) )
        self.ui_FileLocDialog.tbOpenOutputPath.clicked.connect( functools.partial( 
            self.openPathDialog, u'output_dir', True ) )
        self.ui_FileLocDialog.tbOpenRawPath.clicked.connect( functools.partial( 
            self.openPathDialog, u'raw_subdir', True ) )
        self.ui_FileLocDialog.tbOpenSumPath.clicked.connect( functools.partial( 
            self.openPathDialog, u'sum_subdir', True ) )
        self.ui_FileLocDialog.tbOpenAlignPath.clicked.connect( functools.partial( 
            self.openPathDialog, u'align_subdir', True ) )
        self.ui_FileLocDialog.tbOpenFiguresPath.clicked.connect( functools.partial( 
            self.openPathDialog, u'fig_subdir', True ) )
        self.ui_FileLocDialog.tbOpenGainRefPath.clicked.connect( functools.partial( 
            self.openFileDialog, u'gainRef', True ) )    


        self.ui_FileLocDialog.leInputPath.editingFinished.connect( functools.partial( 
            self.updateDict, u"skulk.paths", u'input_dir', self.ui_FileLocDialog.leInputPath.text ) )
        self.ui_FileLocDialog.leOutputPath.editingFinished.connect( functools.partial( 
            self.updateDict, u"skulk.paths", u'output_dir', self.ui_FileLocDialog.leOutputPath.text ) )
        self.ui_FileLocDialog.leRawPath.editingFinished.connect( functools.partial( 
            self.updateDict, u"skulk.paths", u'raw_subdir', self.ui_FileLocDialog.leRawPath.text ) )
        self.ui_FileLocDialog.leSumPath.editingFinished.connect( functools.partial( 
            self.updateDict, u"skulk.paths", u'sum_subdir', self.ui_FileLocDialog.leSumPath.text ) ) 
        self.ui_FileLocDialog.leAlignPath.editingFinished.connect( functools.partial( 
            self.updateDict, u"skulk.paths", u'align_subdir', self.ui_FileLocDialog.leAlignPath.text ) )
        self.ui_FileLocDialog.leFiguresPath.editingFinished.connect( functools.partial( 
            self.updateDict, u"skulk.paths", u'fig_subdir', self.ui_FileLocDialog.leFiguresPath.text ) )    
        self.ui_FileLocDialog.leGainRefPath.editingFinished.connect( functools.partial( 
            self.updateDict, u"skulk.paths", u'gainRef', self.ui_FileLocDialog.leGainRefPath.text ) ) 
        # Gainref has to be provided to the zorroDefault object later.   
            
        # File output and compression
        self.ui_FileLocDialog.comboCompressor.setCurrentIndex(0) # Default to None
        self.ui_FileLocDialog.comboCompressor.currentIndexChanged.connect( functools.partial( 
            self.updateZorroDefault, u'files.compressor', self.ui_FileLocDialog.comboCompressor.currentText ) )
        self.ui_FileLocDialog.cbGainRot.stateChanged.connect( functools.partial( 
            self.updateZorroDefault, u'gainInfo.Diagonal', self.ui_FileLocDialog.cbGainRot.isChecked ) )
        self.ui_FileLocDialog.cbGainHorzFlip.stateChanged.connect( functools.partial( 
            self.updateZorroDefault, u'gainInfo.Horizontal', self.ui_FileLocDialog.cbGainHorzFlip.isChecked ) )
        self.ui_FileLocDialog.cbGainVertFlip.stateChanged.connect( functools.partial( 
            self.updateZorroDefault, u'gainInfo.Vertical', self.ui_FileLocDialog.cbGainVertFlip.isChecked ) )
        
        self.ui_OrienGainRefDialog.tbOrientGain_GainRef.clicked.connect( functools.partial( 
            self.openFileDialog, u'OrientGain_GainRef', True ) )
        self.ui_OrienGainRefDialog.tbOrientGain_TargetStack.clicked.connect( functools.partial( 
            self.openFileDialog, u'OrientGain_TargetStack', True ) )
        
#        self.ui_FileLocDialog.comboOutputFormat.currentIndexChanged.connect( functools.partial( 
#            self.updateZorroDefault, u'files.ext', self.ui_FileLocDialog.comboOutputFormat.currentText ) )
        self.ui_FileLocDialog.sbCLevel.valueChanged.connect( functools.partial( 
            self.updateZorroDefault, u'files.cLevel', self.ui_FileLocDialog.sbCLevel.value ) )
            
        self.ui_OrienGainRefDialog.pbRun.pressed.connect( self.run_OrienGainRef )
        
        # Cache and Qsub paths
        self.tbOpenCachePath.clicked.connect( functools.partial( 
            self.openPathDialog, u'cachePath', True) )
        self.tbOpenQsubHeader.clicked.connect( functools.partial( 
            self.openFileDialog, u'qsubHeader', True) )
         

        # Common configuration
#        self.comboRegistrator.currentIndexChanged.connect( functools.partial( 
#            self.updateDict, u"cfgCommon", "registrator", self.comboRegistrator.currentText ) )
        self.comboTriMode.currentIndexChanged.connect( functools.partial( 
            self.updateZorroDefault, u"triMode", self.comboTriMode.currentText ) )
        self.sbPeaksigThres.valueChanged.connect( functools.partial( 
            self.updateZorroDefault, u"peaksigThres", self.sbPeaksigThres.value ) )  
            
        # TODO: have start/end frame in particle extraction
#        self.sbStartFrame.valueChanged.connect( functools.partial( 
#            self.updateZorroDefault, "startFrame", self.sbStartFrame.value ) )    
#        self.sbEndFrame.valueChanged.connect( functools.partial( 
#            self.updateZorroDefault, "endFrame", self.sbEndFrame.value ) )    
        self.sbDiagWidth.valueChanged.connect( functools.partial( 
            self.updateZorroDefault, u"diagWidth", self.sbDiagWidth.value ) )     
        
        self.sbAutomax.valueChanged.connect( functools.partial( 
            self.updateZorroDefault, u"autoMax", self.sbAutomax.value ) )    
        self.cbSuppressOrigin.stateChanged.connect( functools.partial( 
            self.updateZorroDefault, u"suppressOrigin", self.cbSuppressOrigin.isChecked ) )       
        self.cbSavePNG.stateChanged.connect( functools.partial( 
            self.updateZorroDefault, u"savePNG", self.cbSavePNG.isChecked ) )
        self.comboFilterMode.currentIndexChanged.connect( functools.partial( 
            self.updateZorroDefault, u"filterMode", self.comboFilterMode.currentText ) ) 
        self.cbSaveMovie.stateChanged.connect( functools.partial( 
            self.updateZorroDefault, u"saveMovie", self.cbSaveMovie.isChecked ) )  
        self.comboAlignProgram.currentIndexChanged.connect( functools.partial( 
            self.updateZorroDefault, u"xcorrMode", self.comboAlignProgram.currentText ) )  
        self.comboCtfProgram.currentIndexChanged.connect( functools.partial( 
            self.updateZorroDefault, u"CTFProgram", self.comboCtfProgram.currentText ) )  
     
        # DEBUG
        self.cbDebuggingOutput.stateChanged.connect( functools.partial( 
            self.updateDict, u"cfgCommon", u"DEBUG", self.cbDebuggingOutput.isChecked ) )  
            
        # Advanced configuration
        self.sbShapePadX.valueChanged.connect( functools.partial( 
            self.updateZorroDefault, u"shapePadded", (self.sbShapePadY.value,self.sbShapePadX.value) ) )
        self.sbShapePadY.valueChanged.connect( functools.partial( 
            self.updateZorroDefault, u"shapePadded", (self.sbShapePadY.value,self.sbShapePadX.value) ) )   
        self.sbFouCropX.valueChanged.connect( functools.partial( 
            self.updateZorroDefault, u"fouCrop", (self.sbFouCropY.value,self.sbFouCropX.value) ) )
        self.sbFouCropY.valueChanged.connect( functools.partial( 
            self.updateZorroDefault, u"fouCrop", (self.sbFouCropY.value,self.sbFouCropX.value) ) )  
            
        self.cbDoBinning.stateChanged.connect( functools.partial( 
            self.binningControl, u"enable", self.cbDoBinning.isChecked ) ) 
        self.sbBinCropX.valueChanged.connect( functools.partial( 
            self.binningControl, u"shapeBin", (self.sbBinCropY.value,self.sbBinCropX.value) ) )
        self.sbBinCropY.valueChanged.connect( functools.partial( 
            self.binningControl, u"shapeBin", (self.sbBinCropY.value,self.sbBinCropX.value) ) )  
        
        self.sbPixelsize.valueChanged.connect( functools.partial( 
            self.updateZorroDefault, u"pixelsize", self.sbPixelsize.value ) )       
        self.sbVoltage.valueChanged.connect( functools.partial( 
            self.updateZorroDefault, u"voltage", self.sbVoltage.value ) )  
        self.sbC3.valueChanged.connect( functools.partial( 
            self.updateZorroDefault, u"C3", self.sbC3.value ) )
        self.sbGain.valueChanged.connect( functools.partial( 
            self.updateZorroDefault, u"gain", self.sbGain.value ) )
        self.sbMaxShift.valueChanged.connect( functools.partial( 
            self.updateZorroDefault, u"maxShift", self.sbMaxShift.value ) )    
        self.comboOriginMode.currentIndexChanged.connect( functools.partial( 
            self.updateZorroDefault, u"originMode", self.comboOriginMode.currentText ) )    
        self.cbPreshift.stateChanged.connect( functools.partial( 
            self.updateZorroDefault, u"preShift", self.cbPreshift.isChecked ) )  
        self.cbSaveC.stateChanged.connect( functools.partial( 
            self.updateZorroDefault, u"saveC", self.cbSaveC.isChecked ) )      
        self.comboBmode.currentIndexChanged.connect( functools.partial( 
            self.updateZorroDefault, u"Bmode", self.comboBmode.currentText ) ) 
        self.sbBrad.valueChanged.connect( functools.partial( 
            self.updateZorroDefault, u"Brad", self.sbBrad.value ) )    
        self.comboWeightMode.currentIndexChanged.connect( functools.partial( 
            self.updateZorroDefault, u"weightMode", self.comboWeightMode.currentText ) )     
        self.comboPeakLocMethod.currentIndexChanged.connect( functools.partial( 
            self.updateZorroDefault, u"peakLocMode", self.comboPeakLocMethod.currentText ) )  
        self.sbSubpixReg.valueChanged.connect( functools.partial( 
            self.updateZorroDefault, u"subPixReg", self.sbSubpixReg.value ) )  
        self.comboShiftMethod.currentIndexChanged.connect( functools.partial( 
            self.updateZorroDefault, u"shiftMethod", self.comboShiftMethod.currentText ) )  
            
            
        # Cluster configuration
        # Setup default values
        self.comboClusterType.currentIndexChanged.connect( functools.partial( 
            self.updateDict, u"cfgCluster", u'cluster_type', self.comboClusterType.currentText ) )
        self.sbNThreads.valueChanged.connect( functools.partial( 
            self.updateDict, u"cfgCluster", u'n_threads', self.sbNThreads.value ) )
        self.sbNProcesses.valueChanged.connect( functools.partial( 
            self.updateDict, u"cfgCluster", u"n_processes", self.sbNProcesses.value ) )
        self.sbNSyncs.valueChanged.connect( functools.partial( 
            self.updateDict, u"cfgCluster", u"n_syncs", self.sbNSyncs.value ) )    
        self.cbMultiprocessPlots.stateChanged.connect( functools.partial(
            self.updateZorroDefault, u'plotDict.multiprocess', self.cbMultiprocessPlots.isChecked ) )
            
        self.leCachePath.textEdited.connect( functools.partial( 
            self.updateZorroDefault, u'cachePath', self.leCachePath.text ))
        self.leQsubHeaderFile.textEdited.connect( functools.partial( 
            self.updateDict, u"cfgCluster", u"qsubHeader", self.leQsubHeaderFile.text ) )   
        self.comboFFTWEffort.currentIndexChanged.connect( functools.partial( 
            self.updateZorroDefault,  u"fftw_effort", self.comboFFTWEffort.currentText ) )    
        self.listFiles.itemActivated.connect( self.displaySelectedFile  )
        
        
        # All the Gauto's are line-edits so that we can have None or "" as the values
        # Plus I don't have to worry about types.
        self.leGautoBoxsize.editingFinished.connect( functools.partial( 
            self.updateDict, u"cfgGauto", u'boxsize', self.leGautoBoxsize.text ) )
        self.leGautoDiameter.editingFinished.connect( functools.partial( 
            self.updateDict, u"cfgGauto", u'diameter', self.leGautoDiameter.text ) )    
        self.leGautoMin_Dist.editingFinished.connect( functools.partial( 
            self.updateDict, u"cfgGauto", u'min_dist', self.leGautoMin_Dist.text ) )        
        # Template
        self.leGautoTemplates.editingFinished.connect( functools.partial( 
            self.updateDict, u"cfgGauto", u'T', self.leGautoTemplates.text ) )
        self.tbGautoOpenTemplate.clicked.connect( functools.partial( 
            self.openFileDialog, u'gautoTemplates', True) )
        # Template pixelsize?    
        self.leGautoAng_Step.editingFinished.connect( functools.partial( 
            self.updateDict, u"cfgGauto", u'ang_step', self.leGautoAng_Step.text ) )      
        self.leGautoSpeed.editingFinished.connect( functools.partial( 
            self.updateDict, u"cfgGauto", u'speed', self.leGautoSpeed.text ) ) 
        self.leGautoCCCutoff.editingFinished.connect( functools.partial( 
            self.updateDict, u"cfgGauto", u'cc_cutoff', self.leGautoCCCutoff.text ) ) 
        self.leGautoLsigma_D.editingFinished.connect( functools.partial( 
            self.updateDict, u"cfgGauto", u'lsigma_D', self.leGautoLsigma_D.text ) ) 
        self.leGautoLsigma_Cutoff.editingFinished.connect( functools.partial( 
            self.updateDict, u"cfgGauto", u'lsigma_cutoff', self.leGautoLsigma_Cutoff.text ) ) 
        self.leGautoLave_D.editingFinished.connect( functools.partial( 
            self.updateDict, u"cfgGauto", u'lave_D', self.leGautoLave_D.text ) ) 
        self.leGautoLave_Max.editingFinished.connect( functools.partial( 
            self.updateDict, u"cfgGauto", u'lave_max', self.leGautoLave_Max.text ) ) 
        self.leGautoLave_Min.editingFinished.connect( functools.partial( 
            self.updateDict, u"cfgGauto", u'lave_min', self.leGautoLave_Min.text ) ) 
        self.leGautoLP.editingFinished.connect( functools.partial( 
            self.updateDict, u"cfgGauto", u'lp', self.leGautoLP.text ) ) 
        self.leGautoHP.editingFinished.connect( functools.partial( 
            self.updateDict, u"cfgGauto", u'hp', self.leGautoHP.text ) ) 
        self.leGautoLPPre.editingFinished.connect( functools.partial( 
            self.updateDict, u"cfgGauto", u'pre_lp', self.leGautoLPPre.text ) ) 
        self.leGautoHPPre.editingFinished.connect( functools.partial( 
            self.updateDict, u"cfgGauto", u'pre_hp', self.leGautoHPPre.text ) ) 
            
        # Flags go into cfgGPlot, simply so I can use a generator on cfgGauto and handle these manually
        self.cbGautoDoprefilter.stateChanged.connect( functools.partial( 
            self.updateDict, u"cfgGplot", u"do_pre_filter", self.cbGautoDoprefilter.isChecked ) )
        self.cbGautoPlotCCMax.stateChanged.connect( functools.partial( 
            self.updateDict, u"cfgGplot", u"write_ccmax_mic", self.cbGautoPlotCCMax.isChecked ) )
        self.cbGautoPlotPref.stateChanged.connect( functools.partial( 
            self.updateDict, u"cfgGplot", u"write_pref_mic", self.cbGautoPlotPref.isChecked ) )
        self.cbGautoPlotBG.stateChanged.connect( functools.partial( 
            self.updateDict, u"cfgGplot", u"write_bg_mic", self.cbGautoPlotBG.isChecked ) )
        self.cbGautoPlotBGFree.stateChanged.connect( functools.partial( 
            self.updateDict, u"cfgGplot", u"write_bgfree_mic", self.cbGautoPlotBGFree.isChecked ) )
        self.cbGautoPlotLsigma.stateChanged.connect( functools.partial( 
            self.updateDict, u"cfgGplot", u"write_lsigma_mic", self.cbGautoPlotLsigma.isChecked ) )    
        self.cbGautoPlotMask.stateChanged.connect( functools.partial( 
            self.updateDict, u"cfgGplot", u"write_mic_mask", self.cbGautoPlotMask.isChecked ) )     
            
            
        # Toolbar buttons
        self.tbDeleteFile.clicked.connect( self.deleteSelected )
        self.tbRun.clicked.connect( self.runSkulk )
        self.tbKillAll.clicked.connect( self.killSkulk )
        self.tbKillAll.setEnabled(False)
        self.tbReprocess.clicked.connect( self.reprocessSelected )
        self.tbParticlePick.clicked.connect( self.particlePick )

        # Plots setup
        # What I can do is make a list of viewWidgets so I can iterate through them
        # This gives more flexibility for different arrangements in the future
        self.viewWidgetList = []
        self.viewWidgetList.append( self.viewWidget1 )
        self.viewWidgetList.append( self.viewWidget2 )
        self.viewWidgetList.append( self.viewWidget3 )
        self.viewWidgetList.append( self.viewWidget4 )
        self.viewWidgetList.append( self.viewWidget5 )
        self.viewWidgetList.append( self.viewWidget6 )
        for index, viewWidg in enumerate( self.viewWidgetList ):
            viewWidg.viewNumber = index
            viewWidg.autoParent = self
        
        # Try to load an ini file in the startup directory if it's present.
        #try: 
            # iniList = glob.glob( os.path.join( origdir, '*.ini' ) )
        iniList = glob.glob( u"*.ini" )
        # Can't open a loadConfig dialog until the app has started, so only one .ini file can be in the directory.
        if len( iniList ) == 1: # One .ini file
            self.loadConfig( iniList[0] )
        else:
            defaultConfig = os.path.join( os.path.realpath(__file__), u'default.ini' )
            if os.path.isfile( defaultConfig ):
                print( "Loading default: %s" % defaultConfig  )  
                self.loadConfig( defaultConfig )
#        except:
#            try:
#                self.loadConfig( u'default.ini' )
#            except:
#                print( "Using default Zorro parameters" )
            
        # Check for auxiliary programs
        if not bool(testing):
            self.validateAuxTools()
            
        # Setup preferred popout function
        self.preferPopout()
        
        self.skulk.inspectLogDir() # Let's see if we can run this once...
        
        if not bool(testing):
            self.MainWindow.showMaximized()
            self.exec_()
        
        
    
    def validateAuxTools(self):
        #self.comboCompressionExt.setEnabled(True)
        #self.cbDoCompression.setEnabled(True)
        self.pageGautoConfig.setEnabled(True)
        self.tbParticlePick.setEnabled(True)
        
        warningMessage = ""
        # Check for installation of lbzip2, pigz, and 7z
        #if not bool( zorro.util.which('lbzip2') ) and not bool( zorro.util.which('7z') ) and not bool( zorro.util.which('pigz') ):
        #    warningMessage += u"Disabling compression: None of lbzip2, pigz, or 7z found.\n"
        #    # TODO: limit compress_ext if only one is found?
        #    self.comboCompressionExt.setEnabled(False)
        #    self.cbDoCompression.setEnabled(False)

        # Check for installation of CTFFIND/GCTF
        if not bool( zorro.util.which('ctffind') ):
            # Remove CTFFIND4 from options
            warningMessage += u"Disabling CTFFIND4.1: not found.\n"
            self.comboCtfProgram.removeItem( self.comboCtfProgram.findText( 'CTFFIND4.1') )
            self.comboCtfProgram.removeItem( self.comboCtfProgram.findText( 'CTFFIND4.1, sum') )
        if not bool( zorro.util.which('gctf') ):
            warningMessage += u"Disabling GCTF: not found.\n"
            self.comboCtfProgram.removeItem( self.comboCtfProgram.findText( 'GCTF') )
            self.comboCtfProgram.removeItem( self.comboCtfProgram.findText( 'GCTF, sum') )
        # Check for installation of Gautomatch
        if not bool( zorro.util.which('gautomatch') ):
            warningMessage += u"Disabling particle picking: Gautomatch not found.\n"
            self.pageGautoConfig.setEnabled(False)
            self.tbParticlePick.setEnabled(False)
        if not bool( zorro.util.which('2dx_viewer') ):
            warningMessage += u"2dx_viewer not found, using IMS for pop-out views.\n"
            self.actionPrefer2dx_viewer.setEnabled( False )
            self.actionPreferIms.setChecked(True)
            
            
        if bool( warningMessage ):
            warnBox = QtGui.QMessageBox()
            warnBox.setText( warningMessage )
            warnBox.exec_()
            
            
    def initClusterConfig(self):
        # Setup default dicts so we have some values.
        import numexprz as nz
        def countPhysicalProcessors():
            # This simply doesn't work on Windows with iCore5 for example.
        
            cpuInfo = nz.cpu.info
            physicalIDs = []
            for J, cpuDict in enumerate( cpuInfo ):
                if not cpuDict['physical id'] in physicalIDs:
                    physicalIDs.append( cpuDict['physical id'] )
            return len( physicalIDs )

        
        try:
            cpuCoresPerProcessor = np.int(nz.cpu.info[0]['cpu cores'])
            self.cfgCluster = { u'n_threads':cpuCoresPerProcessor, 
            u'n_processes':countPhysicalProcessors(), u'n_syncs':2, 
            u'cluster_type': u'local', u'qsubHeader':u"" }
        except:
            print( "Failed to determine number of CPU cores, defaulting to number of virtual cores" )
            # Use default values if the system can't figure things out for itself
            self.cfgCluster = { u'n_threads': len(nz.cpu.info), u'n_processes':1, u'n_syncs':2, 
                                   u'cluster_type': u'local', u'qsubHeader':u"" }

        self.sbNProcesses.setValue( self.cfgCluster[u'n_processes'] )
        self.sbNThreads.setValue( self.cfgCluster[u'n_threads'] )
        self.sbNSyncs.setValue( self.cfgCluster[u'n_syncs'] )
        
            
    def joinIconPaths(self):
        # Icon's aren't pathed properly if the CWD is somewhere else than the source folder, so...
        self.source_dir = os.path.dirname( os.path.realpath(__file__) )
        # Join all the icons and reload them
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(os.path.join( self.source_dir, "icons/CINAlogo.png")), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.MainWindow.setWindowIcon(icon)
        
        self.label_2.setPixmap(QtGui.QPixmap(os.path.join( self.source_dir, "icons/CINAlogo.png")))
        
        icon1 = QtGui.QIcon()
        icon1.addPixmap(QtGui.QPixmap(os.path.join( self.source_dir, "icons/folder.png")), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.tbOpenCachePath.setIcon(icon1)
        self.tbOpenQsubHeader.setIcon(icon1)
        self.tbGautoOpenTemplate.setIcon(icon1)
        self.ui_FileLocDialog.tbOpenGainRefPath.setIcon(icon1)
        self.ui_FileLocDialog.tbOpenFiguresPath.setIcon(icon1)
        self.ui_FileLocDialog.tbOpenInputPath.setIcon(icon1)
        self.ui_FileLocDialog.tbOpenOutputPath.setIcon(icon1)
        self.ui_FileLocDialog.tbOpenRawPath.setIcon(icon1)
        self.ui_FileLocDialog.tbOpenSumPath.setIcon(icon1)
        self.ui_FileLocDialog.tbOpenAlignPath.setIcon(icon1)
        
        self.ui_OrienGainRefDialog.tbOrientGain_GainRef.setIcon(icon1)
        self.ui_OrienGainRefDialog.tbOrientGain_TargetStack.setIcon(icon1)
        
        icon2 = QtGui.QIcon()
        icon2.addPixmap(QtGui.QPixmap(os.path.join( self.source_dir, "icons/go-next.png")), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.tbRun.setIcon(icon2)
        icon3 = QtGui.QIcon()
        icon3.addPixmap(QtGui.QPixmap(os.path.join( self.source_dir, "icons/process-stop.png")), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.tbKillAll.setIcon(icon3)
        icon4 = QtGui.QIcon()
        icon4.addPixmap(QtGui.QPixmap(os.path.join( self.source_dir, "icons/user-trash.png")), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.tbDeleteFile.setIcon(icon4)
        icon5 = QtGui.QIcon()
        icon5.addPixmap(QtGui.QPixmap(os.path.join( self.source_dir, "icons/boxes.png")), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.tbParticlePick.setIcon(icon5)
        icon6 = QtGui.QIcon()
        icon6.addPixmap(QtGui.QPixmap(os.path.join( self.source_dir, "icons/view-refresh.png")), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.tbReprocess.setIcon(icon6)
        
    def preferPopout( self ):
        preferredText = self.actionGroupPopout.checkedAction().text()
        for vw in self.viewWidgetList:
            vw.popout = preferredText
        pass
    
    def runSkulk( self ):
        # Clean-up the file list
        for J in np.arange( self.listFiles.count(), -1, -1 ):
            self.listFiles.takeItem( J )
            
        # Check pathing, ask user to set if any fields are missing
        if not self.checkValidPaths():
            return
            
        print( "##########IN runSkulk ############" )
        print( self.cfgCluster )
        
        # Init hosts and otherwise reset the skulkManager
        self.skulk.initHosts( cluster_type = self.cfgCluster[u'cluster_type'], 
                             n_processes = self.cfgCluster[u'n_processes'], 
                             n_threads = self.cfgCluster[u'n_threads'],
                             n_syncs = self.cfgCluster[u'n_syncs'],
                             qsubHeader = self.cfgCluster[u'qsubHeader'] )

        # Make a new thread
        self.skulk.start()
        
        # Disable GUI elements related to clusters
        self.tbKillAll.setEnabled(True)
        self.tbRun.setEnabled(False)
        self.pageClusterConfig.setEnabled(False)
        self.menuAnalysis.setEnabled(False)
        
        
    def killSkulk( self ):
        print( "Killing all the foxes in the skulk" )
        self.skulk.kill()
        #if self.skulkThread != None:
        # self.skulkThread.join()
        #self.skulkThread.exit()
            
        # Re-enable locked UI elements
        self.tbKillAll.setEnabled(False)
        self.tbRun.setEnabled(True)
        self.pageClusterConfig.setEnabled(True)
        self.menuAnalysis.setEnabled(True)

    @QtCore.Slot()    
    def updateFromSkulk( self, state_id, name, command ):
        """
        This is a message from the skulk manager that it's had a file change.  Remember that it's typed as 
        the signal is Qt, so if you change it in skulkManager you need to change the static declaration.
        
        Valid colors:
        u'aliceblue', u'antiquewhite', u'aqua', u'aquamarine', u'azure', u'beige', u'bisque', u'black', 
        u'blanchedalmond', u'blue', u'blueviolet', u'brown', u'burlywood', u'cadetblue', u'chartreuse', 
        u'chocolate', u'coral', u'cornflowerblue', u'cornsilk', u'crimson', u'cyan', u'darkblue', u'darkcyan', 
        u'darkgoldenrod', u'darkgray', u'darkgreen', u'darkgrey', u'darkkhaki', u'darkmagenta', 
        u'darkolivegreen', u'darkorange', u'darkorchid', u'darkred', u'darksalmon', u'darkseagreen', 
        u'darkslateblue', u'darkslategray', u'darkslategrey', u'darkturquoise', u'darkviolet', u'deeppink', 
        u'deepskyblue', u'dimgray', u'dimgrey', u'dodgerblue', u'firebrick', u'floralwhite', u'forestgreen', 
        u'fuchsia', u'gainsboro', u'ghostwhite', u'gold', u'goldenrod', u'gray', u'green', u'greenyellow', 
        u'grey', u'honeydew', u'hotpink', u'indianred', u'indigo', u'ivory', u'khaki', u'lavender', 
        u'lavenderblush', u'lawngreen', u'lemonchiffon', u'lightblue', u'lightcoral', u'lightcyan', 
        u'lightgoldenrodyellow', u'lightgray', u'lightgreen', u'lightgrey', u'lightpink', u'lightsalmon', 
        u'lightseagreen', u'lightskyblue', u'lightslategray', u'lightslategrey', u'lightsteelblue', 
        u'lightyellow', u'lime', u'limegreen', u'linen', u'magenta', u'maroon', u'mediumaquamarine', 
        u'mediumblue', u'mediumorchid', u'mediumpurple', u'mediumseagreen', u'mediumslateblue', 
        u'mediumspringgreen', u'mediumturquoise', u'mediumvioletred', u'midnightblue', u'mintcream', 
        u'mistyrose', u'moccasin', u'navajowhite', u'navy', u'oldlace', u'olive', u'olivedrab', u'orange', 
        u'orangered', u'orchid', u'palegoldenrod', u'palegreen', u'paleturquoise', u'palevioletred', 
        u'papayawhip', u'peachpuff', u'peru', u'pink', u'plum', u'powderblue', u'purple', u'red', u'rosybrown', 
        u'royalblue', u'saddlebrown', u'salmon', u'sandybrown', u'seagreen', u'seashell', u'sienna', 
        u'silver', u'skyblue', u'slateblue', u'slategray', u'slategrey', u'snow', u'springgreen', u'steelblue', 
        u'tan', u'teal', u'thistle', u'tomato', u'transparent', u'turquoise', u'violet', u'wheat', u'white', 
        u'whitesmoke', u'yellow', u'yellowgreen']
        """
        
        # If we rename or delete the file, try to get it by state_id
        # print( "Updating file %s with id %s to color/state %s" %(name,state_id,color) )
        fullName = name
        baseName = os.path.basename( name )
        
        
        if command == 'rename':
            if self.skulk.DEBUG:
                print( "RENAME: (%s) %s " % (state_id, self.stateIds[state_id]) )
                
            try:
                oldName = self.stateIds.pop( state_id )
                self.reverseIds.pop( oldName )
            except KeyError:
                raise KeyError( "Automator: Could not find state# %s in ID dict" % state_id )
                return
                
            listItem = self.listFiles.findItems( oldName, QtCore.Qt.MatchFixedString )

            if len( listItem ) > 0:
                listItem = listItem[0]
                oldListName = listItem.text()
                self.listFiles.takeItem( self.listFiles.row( listItem )  )
            else:
                print( "DEBUG RENAME: Failed to find oldName: %s" % oldName )
            
            # We really have to remove the listItem as it seems Qt passes us a 
            # copy instead of a pointer.  I.e. updates by setText dont' work.
            
            listItem = QtGui.QListWidgetItem( baseName )
            self.listFiles.addItem( listItem )
            newListName = listItem.text()
            
            print( "DEBUG RENAME: (%s) from: %s, to: %s" %( state_id, oldListName, newListName ) )

            
            # We need to update our dicts
            self.stateIds[state_id] = baseName
            self.reverseIds[baseName] = state_id
            # List should self sort
            return
        elif command == 'delete':
            if self.skulk.DEBUG:
                print( "DELETE: (%s) %s " % (state_id, self.stateIds[state_id]) )
            try:
                oldName = self.stateIds.pop( state_id )
                self.reverseIds.pop( oldName )
            except KeyError:
                raise KeyError( "Automator: Could not find state# %s in ID dict" % state_id )
                return
                
            listItem = self.listFiles.findItems( baseName, QtCore.Qt.MatchFixedString )
            if len( listItem ) > 0:
                listItem = listItem[0]
                self.listFiles.takeItem( self.listFiles.row( listItem )  )
            return
            
        elif command == u'indigo':   # u'Finished'
            self.statusbar.showMessage( "Finished processing: " + fullName )
            
        if name != None: 
            # Update the id - key combination
            self.stateIds[state_id] = baseName
            self.reverseIds[baseName] = state_id
        else:
            # If name == None, we want to ID the key by its id number
            baseName = self.stateIds[state_id]
        

        # Check to see if the item exists already
        listItem = self.listFiles.findItems( baseName, QtCore.Qt.MatchFixedString )

        if len(listItem) == 0:
            # New id-key pair
            listItem = QtGui.QListWidgetItem( baseName )
            self.listFiles.addItem( listItem )
        else:
            listItem = listItem[0]
            
        
        # Can't compare QtGui.QListItem to None, so just use a try
        try:
            if command != None:
                listItem.setForeground( QtGui.QBrush( QtGui.QColor( u""+command ) ) )
                listItem.setToolTip(  "%s: %s" % (TOOLTIP_STATUS[command],fullName) )
            else:
                listItem.setForeground( QtGui.QBrush( QtGui.QColor( u"black" ) ) )
                listItem.setToolTip( u"Unknown" )
            pass
        except:
            pass
    
        # Sort it?Should be automatic

        #  sizeFini = len( self.skulk.completeCounter )
        sizeTotal = len( self.skulk )
        self.labelMovie.setText( "Stack browser: %d / %d tot " % (self.skulk.completedCount, sizeTotal) )

        
        
    def displaySelectedFile( self, item ):
        # Get the ZorroObj from the stack browser
        name = item.text()
        # print( "Search for %s" % name + " in %s " % self.stateIds )
        
        reverseState = {v: k for k, v in self.stateIds.items()}
        if name in reverseState:
            #if self.skulk.DEBUG:
            #    print( "Trying to update name: " + str(name) + ", " + str(reverseState[name]) )
                
            self.updateAllViews( zorroObj = self.skulk[reverseState[name]].zorroObj )
        
        
        
    def deleteSelected( self ):
        itemList = self.listFiles.selectedItems()
        
        confirmBox = QtGui.QMessageBox()
        filenameString = ""
        for item in itemList:
            filenameString += item.text() + "\n"
        confirmBox.setText( "Are you sure you want to delete all files related to: %s" % filenameString )
        confirmBox.addButton( QtGui.QMessageBox.Cancel )
        deleteButton = confirmBox.addButton( "Delete", QtGui.QMessageBox.ActionRole )
        confirmBox.setDefaultButton( QtGui.QMessageBox.Cancel )
        confirmBox.exec_()
        if confirmBox.clickedButton() == deleteButton:
            reverseState = {v: k for k, v in self.stateIds.items()}
            for item in itemList:
                #item = self.listFiles.currentItem()
                if item is None: 
                    continue
                state_id = reverseState[item.text()]
                # Delete everything
                self.skulk.remove( state_id )
                # The skulk will remove the item with a signal 
        
        
        
    def reprocessSelected( self ):
        # item = self.listFiles.currentItem()
        itemList = self.listFiles.selectedItems()
        
        reverseState = {v: k for k, v in self.stateIds.items()}
        for item in itemList:
            if item is None: continue
        
            self.skulk.reprocess( reverseState[item.text()] )
            
        if self.skulk.DEBUG:
            print( "DEBUG: stateIds = " + str(self.stateIds) )
            print( "DEBUG: reverseIds = " + str(self.reverseIds) )


    def particlePick( self ):
        from . import Gautoauto
        
        itemList = self.listFiles.selectedItems()
        if len( itemList ) == 0:
            return
        
        sumList = []
        pngFronts = []
        for item in itemList:
            if item is None: continue
            
            # Get the zorro obj and feed the imageSum or filtSum to execGautoMatch
            # Maybe it would be safer to load the zorroObj explicitely?
            zorroState = self.skulk[ self.reverseIds[ item.text() ] ]
            
            if 'filt' in zorroState.zorroObj.files:
                sumName = zorroState.zorroObj.files['filt']
                sumList.append( sumName )     
                
            elif 'sum' in zorroState.zorroObj.files:
                sumName = zorroState.zorroObj.files['sum']
                sumList.append( sumName )    
                
            else:
                print( "Could not find image sum for %s to particle pick" % item.text() )
                try:
                    print( "zorroState.zorroObj.files: " + str(zorroState.zorroObj.files) )
                except:
                    pass
                return
                
            
            stackBase = os.path.basename( os.path.splitext(zorroState.zorroObj.files['stack'])[0] )
            pngFileFront = os.path.join( zorroState.zorroObj.files['figurePath'], stackBase )
            pngFronts.append( pngFileFront )
            
            automatchName = os.path.splitext( sumName )[0] + "_automatch.star"
            rejectedName = os.path.splitext( sumName )[0] + "_rejected.star"
            
            # Consider adding boxMask to the zorroObj files by default?
            zorroState.zorroObj.files['figBoxMask'] =  pngFileFront + "_boxMask.png"
            zorroState.zorroObj.files['automatchBox'] = automatchName
            zorroState.zorroObj.files['rejectedBox'] = rejectedName
            # Update the config file on disk to reflect the boxes
            zorroState.zorroObj.saveConfig()
            

        # Submit job, this should be a new thread
        print( "===== TODO: launch Gautomatch in seperate thread as it blocks =====" )
        self.cfgGplot['edge'] = 64
        self.cfgGauto['apixM'] = 10.0*zorroState.zorroObj.pixelsize # This assumes all micrographs in the same directory have the same PS
        self.cfgGplot['colorMap'] = 'viridis'
        self.cfgGplot['boxAlpha'] = 0.5
        self.cfgGplot['shapeOriginal'] = zorroState.zorroObj.shapeOriginal
        self.cfgGplot['binning'] = 4
        
        # DEBUG
        self.cfgGplot['write_bg_mic'] = True
        self.cfgGauto['diameter'] = 260
        
        
        
        # Multiprocessed batch mode
        #Gautoauto.batchProcess( sumList, pngFronts, self.cfgGauto, self.cfgGplot, n_processes=self.sbGautoNProcesses.value() )
        # DEBUG: don't use multiprocessing, as error messages are useless
        for J, sumName in enumerate(sumList):
            # params = [mrcName, mode, optInit, optPlot, optRefine]
            # mrcNames, pngFronts, optInit, optPlot, optRefine=None, n_processes=4 ):
            Gautoauto.runGauto( [sumList[J],pngFronts[J],'batch', self.cfgGauto, self.cfgGplot, None]  )
            

    def checkValidPaths( self ):
        errorState, errorText = self.skulk.paths.validate()
        if bool( errorState ):
            errorBox = QtGui.QMessageBox()
            errorBox.setText( errorText )
            errorBox.addButton( QtGui.QMessageBox.Ok )
            errorBox.setDefaultButton( QtGui.QMessageBox.Ok )
            errorBox.exec_()
        return not errorState
        
    def updateAllViews( self, zorroObj = None ):
        for viewWidg in self.viewWidgetList:
            viewWidg.updateZorroObj( zorroObj = zorroObj )
        pass
    
    
    def binningControl( self, command, funcHandle, funcArg=None ):
        # Basically this is just to all us to set shapeBinned = None in the case that the user doesn't want to 
        # do binning because that is the check inside Zorro
        
        if command == 'enable':
            value = funcHandle()
            if bool(value):
                self.sbBinCropY.setEnabled(True); self.sbBinCropX.setEnabled(True)
                self.zorroDefault.shapeBinned = [self.sbBinCropY.value(), self.sbBinCropX.value()]
            else:
                self.sbBinCropY.setEnabled(False); self.sbBinCropX.setEnabled(False)
                self.zorroDefault.shapeBinned = None
        if command == 'shapeBin':
            self.zorroDefault.shapeBinned = [self.sbBinCropY.value(), self.sbBinCropX.value()]
            
    def updateDict( self, dictHandle, key, funcHandle, funcarg = None ):
        
        # This is not mydict, this is a copy of mydict!  Ergh...

        if type(dictHandle) == str or ( sys.version_info.major == 2 and type(dictHandle) == unicode): 
            parts = dictHandle.split('.')
            partHandle = self
            for part in parts:
                partHandle = getattr( partHandle, part )
            dictHandle = partHandle
        
        dictHandle[key] = funcHandle()
        if key == u"DEBUG":
            self.skulk.setDEBUG( self.cbDebuggingOutput.isChecked() )
            
        if self.skulk.DEBUG:
            print( "updateDict: [ %s ] : %s " % (key, dictHandle[key] ) )
            
        #if key == u'n_threads':
        #    for hostName, hostObj in self.skulk.procHosts:
        #        hostObj.n_threads = funcHandle()
        #    pass
        
        
    def updateZorroDefault( self, zorroAttrName, funcHandle, funcArg = None ):
        if isinstance( funcHandle, tuple):
            newVal = list( func() for func in funcHandle )
        else:
            newVal = funcHandle()
        
        # Check if we have a dict by splitting on '.', so i.e. plotDict.multiprocess => plotDict['mulitprocess']
        tokens = zorroAttrName.split('.')

        if self.skulk.DEBUG:
            try:
                print( "Changing zorroDefault."+ tokens + " from: " + 
                    str(self.zorroDefault.__getattribute__(tokens[0])) + " to: " + str(newVal) )
            except: pass
                
        if newVal == 'none':
            newVal = None
        

        if len(tokens) == 1: # Class attribute
            self.zorroDefault.__setattr__( tokens[0], newVal )
        elif len(tokens) == 2: # Class dict
            # Get the dict and set it by the provided key
            handle = getattr( self.zorroDefault, tokens[0] )
            handle[tokens[1]] = newVal
        # Stealing reference
        self.skulk.zorroDefault = self.zorroDefault

    def run_OrienGainRef( self ):
        from zorro.scripts import orientGainReference 
        """
        def orientGainRef( gainRefName, stackName, 
                   stackIsInAHole=True, applyHotPixFilter = True, doNoiseCorrelation=True,
                   relax=0.95, n_threads = None )
        """
        
        self.ui_OrienGainRefDialog.progressBar.setMaximum(0)
        self.ui_OrienGainRefDialog.progressBar.show()
        self.ui_OrienGainRefDialog.progressBar.setValue(0)
        
        
        # Maybe this will have to be a subprocess with argv, if you want to have 
        # a progress bar?  Ugh, what a pain...
        try:
            orientation = orientGainReference.orientGainRef( 
                                       self.ui_OrienGainRefDialog.leInputPath.text(),
                                       self.ui_OrienGainRefDialog.leGainRefPath.text(),
                                       stackIsInAHole = self.ui_OrienGainRefDialog.cbStackInHole.isChecked(),
                                       applyHotPixFilter = self.ui_OrienGainRefDialog.cbApplyHotpixFilt.isChecked(), 
                                       doNoiseCorrelation = self.ui_OrienGainRefDialog.cbDoCorrel.isChecked(),
                                       relax = self.ui_OrienGainRefDialog.sbHotpixRelax.value(),
                                       n_threads = self.sbNThreads.value() )
            self.ui_FileLocDialog.cbGainRot.setChecked( orientation[0] )
            self.ui_FileLocDialog.cbGainVertFlip.setChecked( orientation[1] )   
            self.ui_FileLocDialog.cbGainHorzFlip.setChecked( orientation[2] )   
        except Exception as E:
            print( E )
        self.ui_OrienGainRefDialog.progressBar.setMaximum(100)
        self.ui_OrienGainRefDialog.progressBar.hide()
        
    def quitApp( self, event = None ):
        print( "Shutting down: " + str(event) )
        self.killSkulk()
        
        # Try and save config if it was saved previously in the CWD
        if self.cfgfilename in os.listdir('.'):
            self.saveConfig( self.cfgfilename )
        
        self.MainWindow.close()
        self.FileLocDialog.close()
        self.exit() 
        try:
            sys.exit(0)
        except SystemExit as e:
            sys.exit(e)
        except Exception:
            raise
            
        
    def loadConfig( self, cfgfilename ):

        if cfgfilename is None:
            # open a dialog and ask user to pick a file
            cfgfilename = QtGui.QFileDialog.getOpenFileName( parent=self.MainWindow, caption="Load Initialization File", 
                                dir="", filter="Ini files (*.ini)", selectedFilter="*.ini")[0]
        
        if cfgfilename == '':
            return
        else:
            self.cfgfilename = cfgfilename
        
        self.centralwidget.blockSignals(True)
        self.statusbar.showMessage( "Loaded config file: " + self.cfgfilename  )

        config = configparser.RawConfigParser(allow_no_value = True)
        try:
            config.optionxform = unicode # Python 2
        except:
            config.optionxform = str # Python 3
            

        # Load all the zorro parameters into zorroDefault
        self.zorroDefault.loadConfig( self.cfgfilename )
        

        
        
        
        config.read( self.cfgfilename )
        ##### Common configuration ####
        try:
            self.cfgCommon = json.loads( config.get( u'automator', u'common' ) )
        except: pass
    
        try:
            self.cbDebuggingOutput.setChecked( self.cfgCommon['DEBUG'] )
        except: pass

        if u"version" in self.cfgCommon and __version__ > self.cfgCommon[u"version"]:
            print( "WARNING: Automator (%s) is not backward compatible with %s, version %s" % 
                (__version__, cfgfilename,self.cfgCommon[u"version"] ) )
            return
            
        ##### Paths #####    
        try:
            # Cannot do straight assignment with this because it's not a dict 
            # and we have no constructor with a dict.
            norm_paths = json.loads( config.get(u'automator', u'paths' ) )
            
            for key in norm_paths:
                self.skulk.paths[key] = norm_paths[key]
        except: pass
        
        
        try:
            self.ui_FileLocDialog.leInputPath.setText( self.skulk.paths.get_real(u'input_dir') )
        except: pass
        try:
            self.ui_FileLocDialog.leOutputPath.setText( self.skulk.paths.get_real(u'output_dir') )
        except: pass
#        try:
#            self.ui_FileLocDialog.leRawPath.setText( self.skulk.paths.get_real(u'cwd')  )
#        except: pass
    
        try:
            self.ui_FileLocDialog.leRawPath.setText( self.skulk.paths.get_real(u'raw_subdir')  )
        except: pass
        try:
            self.ui_FileLocDialog.leSumPath.setText( self.skulk.paths.get_real(u'sum_subdir')  )
        except: pass
        try:
            self.ui_FileLocDialog.leAlignPath.setText( self.skulk.paths.get_real(u'align_subdir') )
        except: pass
        try:
            self.ui_FileLocDialog.leFiguresPath.setText( self.skulk.paths.get_real(u'fig_subdir') )
            for viewWdgt in self.viewWidgetList:
                viewWdgt.viewCanvas.param['figurePath'] = self.skulk.paths.paths[u'fig_subdir']
        except: pass
        try:
            self.ui_FileLocDialog.leGainRefPath.setText( self.skulk.paths.get_real(u'gainRef') )
        except: pass
        try: 
            self.ui_FileLocDialog.comboCompressor.setCurrentIndex( 
                self.ui_FileLocDialog.comboCompressor.findText( 
                self.zorroDefault.files['compressor'] ) )
        except: pass
        try:
            self.ui_FileLocDialog.cbGainHorzFlip.setChecked( 
                self.zorroDefault.gainInfo['Horizontal'])
            self.ui_FileLocDialog.cbGainVertFlip.setChecked( 
                self.zorroDefault.gainInfo['Vertical'])
            self.ui_FileLocDialog.cbGainRot.setChecked( 
                self.zorroDefault.gainInfo['Diagonal'])
        except: pass
        try: 
            self.ui_FileLocDialog.sbCLevel.setValue( int(self.zorroDefault.files['cLevel']) )
        except: pass
    

        ##### views #####
        for viewWidg in self.viewWidgetList:
            try:
                viewWidg.loadConfig( config )
                pass
            except: pass

        ###### Common configuration #####

        ###### Cluster configuration #####
        try:
            self.cfgCluster = json.loads( config.get( u'automator', u'cluster' ) )
        except:
            pass
        try:
#            self.cfgCluster["cluster_type"] = config.get( 'automator_cluster', 'cluster_type' )
            self.comboClusterType.setCurrentIndex( self.comboClusterType.findText( self.cfgCluster[u"cluster_type"] ) )
        except: print( "Failed to set cluster_type: "  )
        try:
#            self.cfgCluster["n_processes"] = config.getint( 'automator_cluster', 'n_processes' )
            self.sbNProcesses.setValue( self.cfgCluster[u"n_processes"] )
        except: print( "Failed to set n_processes "  )
        try:
#            self.cfgCluster["n_syncs"] = config.getint( 'automator_cluster', 'n_syncs' )
            self.sbNSyncs.setValue( self.cfgCluster[u"n_syncs"] )
        except: print( "Failed to set n_syncs "  )
        try:
#            self.cfgCluster["n_threads"] = config.getint( 'automator_cluster', 'n_threads' )
            self.sbNThreads.setValue( self.cfgCluster[u"n_threads"] )
        except: print( "Failed to set n_threads "  )
        try:
#            self.cfgCluster["qsubHeader"] = config.get( 'automator_cluster', 'qsubHeader' )
            self.leQsubHeaderFile.setText( self.cfgCluster[u"qsubHeader"] )
        except: print( "Failed to set qsubHeader "  )
        try: self.leCachePath.setText( self.zorroDefault.cachePath )
        except: pass
        try: self.comboFFTWEffort.setCurrentIndex( self.comboFFTWEffort.findText( self.zorroDefault.fftw_effort ) )
        except: pass
        try: self.cbMultiprocessPlots.setChecked( self.zorroDefault.plotDict[u"multiprocess"] )
        except: print( "Failed to set multiprocessing option for plots." )

        ###### Gautomatch configuration #####
        
    

        # Update all the GUI elements
        
        try: self.comboTriMode.setCurrentIndex( self.comboTriMode.findText( self.zorroDefault.triMode ) )
        except: pass
        try: self.sbPeaksigThres.setValue( self.zorroDefault.peaksigThres )
        except: pass
        #try: self.sbStartFrame.setValue( self.zorroDefault.startFrame )
        #except: pass
        #try: self.sbEndFrame.setValue( self.zorroDefault.endFrame )
        #except: pass
        try: self.sbDiagWidth.setValue( self.zorroDefault.diagWidth )
        except: pass
        try: self.sbAutomax.setValue( self.zorroDefault.autoMax )
        except: pass
        try: self.cbSuppressOrigin.setChecked( self.zorroDefault.suppressOrigin )
        except: pass
        try: self.comboCtfProgram.setCurrentIndex( self.comboCtfProgram.findText( self.zorroDefault.CTFProgram ) )
        except: print( "Unknown CTF tool: " + str(self.zorroDefault.CTFProgram) )
        try: self.comboFilterMode.setCurrentIndex( self.comboFilterMode.findText( self.zorroDefault.filterMode ) )
        except: print( "Unknown filter mode: " + str(self.zorroDefault.filterMode) )
    

        try: self.cbSavePNG.setChecked( self.zorroDefault.savePNG )
        except: pass

        try: self.cbSaveMovie.setChecked( self.zorroDefault.saveMovie )
        except: pass
    
        try: # Easier to copy-over both values than it is to 
            shapePadded = np.copy( self.zorroDefault.shapePadded )
            self.sbShapePadX.setValue( shapePadded[1] )
            self.sbShapePadY.setValue( shapePadded[0] )
        except: print( "Failed to set sbShapePadX-Y" )
        
        
        print( self.zorroDefault.shapeBinned )
        try: # This is easier then blocking all the signals...
            
            if np.any(self.zorroDefault.shapeBinned) == None:
                self.cbDoBinning.setChecked( False )
            else:
                shapeBinned = np.copy( self.zorroDefault.shapeBinned )
                self.sbBinCropX.setValue( shapeBinned[1] )
                self.sbBinCropY.setValue( shapeBinned[0] )
                self.cbDoBinning.setChecked( True )
        except: print( "Failed to set sbBinCropX-Y" )
        
        try: 
            fouCrop = np.copy( self.zorroDefault.fouCrop )
            self.sbFouCropX.setValue( fouCrop[1] )
            self.sbFouCropY.setValue( fouCrop[0] )
        except: pass
        
        
        try: self.sbPixelsize.setValue( self.zorroDefault.pixelsize )
        except: pass
        try: self.sbVoltage.setValue( self.zorroDefault.voltage )
        except: pass
        try: self.sbC3.setValue( self.zorroDefault.C3 )
        except: pass
        try: self.sbGain.setValue( self.zorroDefault.gain )
        except: pass
        try: self.sbMaxShift.setValue( self.zorroDefault.maxShift )
        except: pass
        try: self.comboOriginMode.setCurrentIndex( self.comboOriginMode.findText( self.zorroDefault.originMode ) )
        except: pass
        try: self.cbPreshift.setChecked( self.zorroDefault.preShift )
        except: pass
        try: self.cbSaveC.setChecked( self.zorroDefault.saveC )
        except: pass
        try: self.comboBmode.setCurrentIndex( self.comboBmode.findText( self.zorroDefault.Bmode ) )
        except: pass
        try: self.sbBrad.setValue( self.zorroDefault.Brad )
        except: pass
        try: self.comboWeightMode.setCurrentIndex( self.comboWeightMode.findText( self.zorroDefault.weightMode ) )
        except: pass
        try: self.sbSubpixReg.setValue( self.zorroDefault.subPixReg )
        except: pass
        try: self.comboShiftMethod.setCurrentIndex( self.comboShiftMethod.findText( self.zorroDefault.shiftMethod ) )
        except: pass
        
    
    
        # Gautomatch
        try:
            self.cfgGauto = json.loads( config.get( u'automator', u'gauto' ) )
        except:
            pass
        
        try: 
            self.leGautoBoxsize.setText( self.cfgGauto[u'boxsize'] )
        except: pass
        try: 
            self.leGautoDiameter.setText( self.cfgGauto[u'diameter'] )
        except: pass
        try: 
            self.leGautoMin_Dist.setText( self.cfgGauto[u'min_dist'] )
        except: pass
        try: 
            self.leGautoTemplates.setText( self.cfgGauto[u'T'] )
        except: pass
        try: 
            self.leGautoAng_Step.setText( self.cfgGauto[u'ang_step'] )
        except: pass
        try: 
            self.leGautoSpeed.setText( self.cfgGauto[u'speed'] )
        except: pass
        try: 
            self.cfgGauto[u'cc_cutoff'] = config.get( u'automator_gauto', u'cc_cutoff' )
            self.leGautoCCCutoff.setText( self.cfgGauto[u'cc_cutoff'] )
        except: pass
        try: 
            self.leGautoLsigma_D.setText( self.cfgGauto[u'lsigma_D'] )
        except: pass
        try: 
            self.leGautoLsigma_Cutoff.setText( self.cfgGauto[u'lsigma_cutoff'] )
        except: pass
        try: 
            self.leGautoLave_D.setText( self.cfgGauto[u'lave_D'] )
        except: pass
        try: 
            self.leGautoLave_Min.setText( self.cfgGauto[u'lave_min'] )
        except: pass
        try: 
            self.leGautoLave_Max.setText( self.cfgGauto[u'lave_max'] )
        except: pass
        try: 
            self.leGautoLP.setText( self.cfgGauto[u'lp'] )
        except: pass
        try: 
            self.leGautoHP.setText( self.cfgGauto[u'hp'] )
        except: pass
        try: 
            self.leGautoLPPre.setText( self.cfgGauto[u'pre_lp'] )
        except: pass
        try: 
            self.leGautoHPPre.setText( self.cfgGauto[u'pre_hp'] )
        except: pass
        # Plotting for Gautomatch
        try: 
            self.cbGautoDoprefilter.setChecked( self.cfgGplot[u'do_pre_filter'] )
        except: pass
        try: 
            self.cbGautoPlotCCMax.setChecked( self.cfgGplot[u'write_ccmax_mic'] )
        except: pass
        try: 
            self.cbGautoPlotPref.setChecked( self.cfgGplot[u"write_pref_mic"] )
        except: pass
        try: 
            self.cbGautoPlotBG.setChecked( self.cfgGplot[u"write_bg_mic"] )
        except: pass
        try: 
            self.cbGautoPlotBGFree.setChecked( self.cfgGplot[u"write_bgfree_mic"] )
        except: pass
        try: 
            self.cbGautoPlotLsigmaFree.setChecked( self.cfgGplot[u"write_lsigma_mic"] )
        except: pass
        try: 
            self.cbGautoPlotMaskFree.setChecked( self.cfgGplot[u"write_mic_mask"] )
        except: pass
    
        self.centralwidget.blockSignals(False)
        # End of Automator.loadConfig
    
    def saveConfig( self, cfgfilename ):
        if cfgfilename is None:
            cfgfilename = QtGui.QFileDialog.getSaveFileName(
                parent=self.MainWindow,caption="Save Initialization File", dir="", filter="Ini files (*.ini)", 
                selectedFilter="*.ini")[0]
            
        if cfgfilename == '':
            return
        else:
            # Force extension to .ini
            cfgfilename = os.path.splitext( cfgfilename )[0] + ".ini"
        
        
        self.cfgfilename = cfgfilename
            
        self.statusbar.showMessage( "Saving configuration: " + cfgfilename )
        self.cfgCommon[u"version"] = __version__
        
#        # Overwrite the file if it already exists
        self.zorroDefault.saveConfig( cfgfilename )
        
        
        # Read in the config prepared by Zorro
        config = configparser.RawConfigParser(allow_no_value = True)
        try:
            config.optionxform = unicode # Python 2
        except:
            config.optionxform = str # Python 3
        
        config.read( cfgfilename ) # Read back in everything from Zorro
        ### PATH ###
        config.add_section( u'automator' )
        config.set( u'automator', u'paths', json.dumps( self.skulk.paths.to_json() ) )
        config.set( u'automator', u'common', json.dumps(self.cfgCommon ) )
        config.set( u'automator', u'cluster', json.dumps(self.cfgCluster ) )
        config.set( u'automator', u'gauto', json.dumps(self.cfgGauto ) )
        config.set( u'automator', u'gplot', json.dumps(self.cfgGplot ) )
        
#        config.add_section('automator_paths')
#        for key in self.skulk.paths:
#            config.set( 'automator_paths', key, self.skulk.paths[key] )
#            
#        config.add_section('automator_common')
#        for key in self.cfgCommon:
#            config.set( 'automator_common', key, self.cfgCommon[key] )
#    
#        config.add_section('automator_cluster')
#        for key in self.cfgCluster:
#            config.set( 'automator_cluster', key, self.cfgCluster[key] )
#            
#        config.add_section('automator_gauto')
#        for key in self.cfgGauto:
#            config.set( 'automator_gauto', key, self.cfgGauto[key] )
#            
#        config.add_section('automator_gplot')
#        for key in self.cfgGplot:
#            config.set( 'automator_gplot', key, self.cfgGplot[key] )   
            
        # Add viewWidgets using their built-in config writers
        for viewWidg in self.viewWidgetList:
            viewWidg.saveConfig( config )
            
#        try:
#            # Open in append mode
        cfgfh = open( self.cfgfilename, 'w' )
        config.write( cfgfh )
        cfgfh.close()
#        except:
#            print( "Error in loading config file: " + self.cfgfilename )
            

    def openFileDialog( self, name, openDialog ):
        if bool(openDialog):
            pathDialog = QtGui.QFileDialog()
            pathDialog.setFileMode( QtGui.QFileDialog.AnyFile )
            newFile = str( pathDialog.getOpenFileName(self.MainWindow, name, "")[0] )
        else:
            # Get directory from the lineedit object
            print( "TODO" )
            
        if name == u'OrientGain_GainRef':
            self.ui_OrienGainRefDialog.leGainRefPath.setText( newFile )
            
        elif name == u'OrientGain_TargetStack':
            self.ui_OrienGainRefDialog.leInputPath.setText( newFile )
            
        elif name == 'qsubHeader':
            self.cfgCluster[u'qsubHeader'] = newFile # Would prefer to generate a signal here.
            self.leQsubHeaderFile.setText( newFile )
            
        elif name == u'gautoTemplates':
            self.cfgGauto[u'T'] = newFile
            self.leGautoTemplates.setText( newFile )
            
        elif name == u'gainRef':
            # self.zorroDefault.files['gainRef'] = newFile
            print( "openFileDialog: Setting gainRef to %s" % newFile )
            self.skulk.paths['gainRef'] = newFile
            self.ui_FileLocDialog.leGainRefPath.setText( newFile )
            
    def openPathDialog( self, pathname, openDialog ):
        # Comments on how to handle updates from a lineedit Qt object
        # http://stackoverflow.com/questions/12182133/pyqt4-combine-textchanged-and-editingfinished-for-qlineedit
        
        if bool(openDialog):
            pathDialog = QtGui.QFileDialog()
            pathDialog.setFileMode( QtGui.QFileDialog.Directory )
            pathDialog.setOption( QtGui.QFileDialog.ShowDirsOnly, True )
            newPath = str(pathDialog.getExistingDirectory(self.MainWindow,pathname, ""))
            print( "New path for "+ pathname + " : " + newPath )
        else:
            # Get directory from the lineedit object
            print( "TODO" )
            
        #self.FileLocDialog.raise_()
        self.FileLocDialog.activateWindow()
        

        if pathname == u'input_dir':
            self.skulk.paths[pathname] = newPath
            self.ui_FileLocDialog.leInputPath.setText( self.skulk.paths.get_real(pathname) )

        elif pathname == u'output_dir':
            self.skulk.paths[pathname] = newPath
            self.ui_FileLocDialog.leOutputPath.setText( self.skulk.paths.get_real(pathname) )
            
        elif pathname == u'raw_subdir': 
            self.skulk.paths[pathname] = newPath
            self.ui_FileLocDialog.leRawPath.setText( self.skulk.paths.get_real(pathname) )

        elif pathname == u'sum_subdir':
            self.skulk.paths[pathname] = newPath
            self.ui_FileLocDialog.leSumPath.setText( self.skulk.paths.get_real(pathname) )

        elif pathname == u'align_subdir':
            self.skulk.paths[pathname] = newPath
            self.ui_FileLocDialog.leAlignPath.setText( self.skulk.paths.get_real(pathname) )

        elif pathname == u'fig_subdir':
            self.skulk.paths[pathname] = newPath
            self.ui_FileLocDialog.leFiguresPath.setText( self.skulk.paths.get_real(pathname) )
            
        elif pathname == u'cachePath':
            print( "TODO: Automator.openPathDialog; manage cachePath" )
        pass

    
    def showCitationsDialog(self):
        citations = u"""
Zorro:
McLeod, R.A., Kowal, J., Ringler, P., Stahlberg, S., Submitted.

CTFFIND4: 
Rohou, A., Grigorieff, N., 2015. CTFFIND4: Fast and accurate defocus estimation from electron micrographs. Journal of Structural Biology, Recent Advances in Detector Technologies and Applications for Molecular TEM 192, 216-221. doi:10.1016/j.jsb.2015.08.008

GCTF:
Zhang, K., 2016. Gctf: Real-time CTF determination and correction. Journal of Structural Biology 193, 1-12. doi:10.1016/j.jsb.2015.11.003

SerialEM:
Mastronarde, D.N. 2005. Automated electron microscope tomography using robust prediction of specimen movements. J. Struct. Biol. 152:36-51.

"""
        citBox = QtGui.QMessageBox()
        citBox.setText( citations )
        citBox.exec_()
        
    def showImsHelpDialog(self):
        citBox = QtGui.QMessageBox()
        citBox.setText( zorro.zorro_plotting.IMS_HELPTEXT )
        citBox.exec_()
        
    
def main():
    try:
        mainGui = Automator()
    except SystemExit:
        del mainGui
        sys.exit()
        exit
        
# Instantiate a class
if __name__ == '__main__': # Windows multiprocessing safety
    main()

# -*- coding: utf-8 -*-
"""
Python Utilities for Relion

Created on Tue Dec  1 14:26:13 2015
@author: Robert A. McLeod
@email: robbmcleod@gmail.com OR robert.mcleod@unibas.ch

This is a primarily a general parser for Relion star files.  It creates a two-level dictionary, with the 
"data_*" level at the top and the "_rln*" level at the second. Use the star.keys() function to see what values 
the dictionary has.  I.e.
    
    rln.star.keys()

and then

    rln.star['data_whatever'].keys()

Example usage:

    rln = ReliablePy()
    # Wildcards can be loaded
    rln.load( 'PostProcess*.star' )
    
    # Plot the Fourier Shell Correlation
    plt.figure()
    plt.plot( rln.star['data_fsc']['Resolution'], rln.star['data_fsc']['FourierShellCorrelationUnmaskedMaps'], '.-' )
    plt.xlabel( "Resolution" )
    plt.ylabel( "FSC" )

"""
from __future__ import division, print_function, absolute_import

from . import zorro
import matplotlib.pyplot as plt
import numpy as np
import os, os.path
import glob
import time
import scipy

try: # For Linux, use FreeSerif
    plt.rc('font', family='FreeSerif', size=16)
except:
    try: 
        plt.rc( 'font', family='serif', size=16)
    except: pass


# Static variable decorator
def static_var(varname, value):
    def decorate(func):
        setattr(func, varname, value)
        return func
    return decorate
    
# TODO: put IceFilter in a ReliablePy utility function file
@static_var( "bpFilter", -1 )
@static_var( "mageShape", np.array([0,0]) )
@static_var( "ps", -42 )
@static_var( "FFT2", -42 )
@static_var( "IFFT2", -42 )
def IceFilter( mage, pixelSize=1.0, filtRad = 8.0 ):
    """
    IceFilter applies a band-pass filter to mage that passes the first 3 
    water ice rings, and then returns the result.
        pixelSize is in ANGSTROMS because this is bio.  Program uses this to 
        calculate the width of the band-pass filter.
        filtRad is radius of the Gaussian filter (pixels) to apply after Fourier filtration 
        that are periodic artifacts due to multiple defocus zeros being in the band 
    """
    
    # First water ring is at 3.897 Angstroms
    # Second is ater 3.669 Angstroms
    # Third is at 3.441 Angstroms
    # And of course there is strain, so go from about 4 to 3.3 Angstroms in the mesh
    # Test for existance of pyfftw
    try:
        import pyfftw
        pyfftwFound = True
    except:
        pyfftwFound = False

    # Check to see if we have to update our static variables
    if ( (IceFilter.mageShape != mage.shape).any() ) or (IceFilter.bpFilter.size == 1) or (IceFilter.ps != pixelSize):
        # Make a new IceFilter.bpFilter
        IceFilter.mageShape = np.array( mage.shape )
        IceFilter.ps = pixelSize
        
        bpMin = pixelSize / 4.0  # pixels tp the 4.0 Angstrom spacing
        bpMax = pixelSize / 3.3  # pixels to the 3.3 Angstrom spacing
        
        # So pixel frequency is -0.5 to +0.5 with shape steps
        # And we want a bandpass from 1.0/bpMin to 1.0/bpMax, which is different on each axis for rectangular images
        pixFreqX = 1.0 / mage.shape[1]
        pixFreqY = 1.0 / mage.shape[0]
        bpRangeX = np.round( np.array( [ bpMin/pixFreqX, bpMax/pixFreqX ] ) )
        bpRangeY = np.round( np.array( [ bpMin/pixFreqY, bpMax/pixFreqY ] ) )
        IceFilter.bpFilter = np.fft.fftshift( 
            (1.0 - zorro.util.apodization( name='butter.64', size=mage.shape, radius=[ bpRangeY[0],bpRangeX[0] ] )) 
            * zorro.util.apodization( name='butter.64', size=mage.shape, radius=[ bpRangeY[1],bpRangeX[1] ] ) )
        IceFilter.bpFilter = IceFilter.bpFilter.astype( 'float32' ) 
        
        if pyfftwFound: [IceFilter.FFT2, IceFilter.IFFT2] = zorro.util.pyFFTWPlanner( mage.astype('complex64') )
        pass
    
    # Apply band-pass filter
    if pyfftwFound:
        IceFilter.FFT2.update_arrays( mage.astype('complex64'), IceFilter.FFT2.get_output_array() )
        IceFilter.FFT2.execute()
        IceFilter.IFFT2.update_arrays( IceFilter.FFT2.get_output_array() * IceFilter.bpFilter, IceFilter.IFFT2.get_output_array() )
        IceFilter.IFFT2.execute()
        bpMage = IceFilter.IFFT2.get_output_array() / mage.size
    else:
        FFTmage = np.fft.fft2( mage )
        bpMage = np.fft.ifft2( FFTmage * IceFilter.bpFilter )

    from scipy.ndimage import gaussian_filter
    bpGaussMage = gaussian_filter( np.abs(bpMage), filtRad )
    # So if I don't want to build a mask here, and if I'm just doing band-pass
    # intensity scoring I don't need it, I don't need to make a thresholded mask
    
    # Should we normalize the bpGaussMage by the mean and std of the mage?
    return bpGaussMage

class ReliablePy(object):
    
    def __init__( self, *inputs ) :
        self.verbose = 1
        self.inputs = list( inputs )
        
        # _data.star file dicts
        self.star = {}
        
        self.par = []
        self.pcol = {}

        self.box = [] # Each box file loaded is indexed by its load order / dict could also be done if it's more convienent.
        
        # Particle/class data
        self.mrc = []
        self.mrc_header = []
        if inputs:
            self.load( *inputs )
        pass
    
    def load( self, *input_names ):
        # See if it's a single-string or list/tuple

        if not isinstance( input_names, basestring ):
            new_files = []
            for item in input_names:
                new_files.extend( glob.glob( item ) )
        else:
            new_files = list( input_names )
        
        for filename in new_files:
            [fileFront, fileExt] = os.path.splitext( filename )
            
            if fileExt == '.mrc' or fileExt == '.mrcs':
                self.inputs.append(filename)
                self.__loadMRC( filename )
            elif fileExt == '.star':
                self.inputs.append(filename)
                self.__loadStar( filename )
            elif fileExt == '.par':
                self.inputs.append(filename)
                self.__loadPar( filename )
            elif fileExt == '.box':
                self.inputs.append(filename)
                self.__loadBox( filename )
            else:
                print( "Unknown file extension passed in: " + filename )
                
    def plotFSC( self ):
        # Do error checking?  Or no?
        plt.rc('lines', linewidth=2.0, markersize=12.0 )
        plt.figure()
        plt.plot( self.star['data_fsc']['Resolution'], 0.143*np.ones_like(self.star['data_fsc']['Resolution']), 
                 '-', color='firebrick', label="Resolution criteria" )
        try:
            plt.plot( self.star['data_fsc']['Resolution'], self.star['data_fsc']['FourierShellCorrelationUnmaskedMaps'], 
                 'k.-', label="Unmasked FSC" )
        except: pass
        try:
            plt.plot( self.star['data_fsc']['Resolution'], self.star['data_fsc']['FourierShellCorrelationMaskedMaps'], 
                 '.-', color='royalblue', label="Masked FRC" )   
        except: pass
        try:         
            plt.plot( self.star['data_fsc']['Resolution'], self.star['data_fsc']['FourierShellCorrelationCorrected'], 
                 '.-', color='forestgreen', label="Corrected FRC" )          
        except: pass
        try:
            plt.plot( self.star['data_fsc']['Resolution'], self.star['data_fsc']['CorrectedFourierShellCorrelationPhaseRandomizedMaskedMaps'], 
                 '.-', color='goldenrod', label="Random-phase corrected FRC" )
        except: pass
        plt.xlabel( "Resolution ($\AA^{-1}$)" )
        plt.ylabel( "Fourier Shell Correlation" )
        plt.legend( loc='upper right', fontsize=16 )
        plt.xlim( np.min(self.star['data_fsc']['Resolution']), np.max(self.star['data_fsc']['Resolution']) )
        print( "Final resolution (unmasked): %.2f A"%self.star['data_general']['FinalResolution']  )
        print( "B-factor applied: %.1f"%self.star['data_general']['BfactorUsedForSharpening'] )
        
    def plotSSNR( self ):
        """
        Pulls the SSNR from each class in a _model.star file and plots them, for assessing which class is the 
        'best' class 
        """
        
        N_particles = np.sum( self.star['data_model_groups']['GroupNrParticles'] )
        N_classes = self.star['data_model_general']['NrClasses']
            
        plt.figure()
        for K in np.arange( N_classes ):
            Resolution = self.star['data_model_class_%d'%(K+1)]['Resolution']
            SSNR = self.star['data_model_class_%d'%(K+1)]['SsnrMap']
            plt.semilogy( Resolution, SSNR+1.0, 
                     label="Class %d: %d" %(K+1,N_particles*self.star['data_model_classes']['ClassDistribution'][K]) )
        plt.legend( loc = 'best' )
        plt.xlabel( "Resolution ($\AA^{-1}$)" )
        plt.ylabel( "Spectral Signal-to-Noise Ratio" )
        # Let's also display the class distributions in the legend
        
    
    def pruneParticlesNearImageEdge( self, box = None, shapeImage = [3838,3710] ):
        """
        Removes any particles near image edge. Relion's default behavoir is to replicate pad these, 
        which often leads to it crashing.
        
        box is the bounding box size for the particle, in pixels.  If a _model.star file is loaded 
        it is automatically detected.  Otherwise it must be provided.
        
        Image size is not stored anywhere obvious in Relion, so it must be passed in in terms of 
        it's shape in [y,x]
        """
        if box == None:
            try: 
                box = self.star['data_model_general']['OriginalImageSize']
            except:
                print( "No box shape found in metadata, load a *_model.star file or provide box dimension" )
                return
            
        partCount = len( self.star['data_']['CoordinateX'] )
        
        # Hmm... removing a row is a little painful because I index by keys in columnar format.
        box2 = box/2
        CoordX = self.star['data_']['CoordinateX']
        CoordY = self.star['data_']['CoordinateY']
        keepElements = ~((CoordX < box2)|(CoordY < box2)|(CoordX > shapeImage[1]-box2)|(CoordY > shapeImage[0]-box2))
        for key, store in self.star['data_'].items():
            self.star['data_'][key] = store[keepElements]
        print( "Deleted %d"%(partCount-len(self.star['data_']['CoordinateX']) ) + 
                " particles too close to image edge" )
        pass
    
    def permissiveMask( self, volumeThres, gaussSigma = 5.0, gaussRethres = 0.07, smoothSigma=1.5 ):
        """
        Given a (tight) volumeThres(hold) measured in Chimera or IMS, this function generates a 
        Gaussian dilated mask that is then smoothed.  Everything is done with Gaussian operations 
        so the Fourier space representation of the mask should be relatively smooth as well, 
        and hence ring less.
        
        Excepts self.mrc to be loaded.  Populates self.mask.  
        
        """
        thres = self.mrc > volumeThres; thres = thres.astype('float32')
        
        gaussThres = scipy.ndimage.gaussian_filter( thres, gaussSigma )
        rethres = gaussThres > gaussRethres; rethres = rethres.astype('float32')
        
        self.mask = scipy.ndimage.gaussian_filter( rethres, smoothSigma )
        print( "permissive mask complete, use ioMRC.MRCexport(self.mrc, 'maskname.mrc') to save" )
        pass
    
    def box2star( self, directory = "." ):
        """
        Converts all EMAN .box files in a directory to the associated .star files. Relion cannot successfully 
        rescale particles if they come in .box format.  Also does box pruning if they are too close to an edge.
        """

        boxList = glob.glob( os.path.join( directory, "*.box") )

        starHeader = """
        data_
        
        loop_
        _rlnCoordinateX #1
        _rlnCoordinateY #2
        """
        
        shapeImage = [3838,3710]
        for boxFile in boxList:
            print( "Loading %s" % boxFile )
            boxData = np.loadtxt(boxFile)
            
            xCoord = boxData[:,0]
            yCoord = boxData[:,1]
            boxX = boxData[:,2]/2
            boxY = boxData[:,3]/2
            
            keepElements = ~((xCoord < boxX)|(yCoord < boxY)|(xCoord > shapeImage[1]-boxX)|(yCoord> shapeImage[0]-boxY))
            xCoord = xCoord[keepElements]
            yCoord = yCoord[keepElements]
            boxX = boxX[keepElements]
            boxY = boxY[keepElements]
            
            starFilename = os.path.splitext( boxFile )[0] + ".star"
            with open( starFilename, 'wb' ) as sh:
                sh.writelines( starHeader )
                for J in np.arange(0,len(xCoord)):
                    sh.write( "%.1f %.1f\n" % (xCoord[J]+boxX[J], yCoord[J]+boxY[J] ) )
                   
                sh.write( "\n" )
            sh.close()
            
    def regroupKmeans( self, partPerGroup = 100, miniBatch=True ):
        """
        Does a 3-D k-means clustering on DefocusU, DefocusV, and GroupScaleCorrection
        
        partPerGroup is a suggestion, that is the number of groups is the # of particles / partPerGroup, 
        so outlier groups will tend to have far fewer particle counts that those in the bulk of the data.
        
        miniBatch=True is faster for very large sets (>100,000 particles), but somewhat less accurate
        miniBatch=False is faster for smaller sets, and better overall
        """
        # K-means clustering
        import sklearn
        import sklearn.cluster
        
        # We need to make an array for all particles that has the GroupScaleCorrection
        P = len( self.star['data_']['DefocusU'] )
        n_clusters = np.int( P / partPerGroup )
        
        DefocusU = self.star['data_']['DefocusU']
        DefocusV = self.star['data_']['DefocusV']
        DefocusMean = 0.5* (DefocusU + DefocusV)
        part_GroupScaleCorrection = np.zeros_like( self.star['data_']['DefocusU'] )
        
        # Build a GroupScaleCorrection vector
        for J, groupNr in enumerate( self.star['data_']['GroupNumber'] ):
            part_GroupScaleCorrection[J] = self.star['data_model_groups']['GroupScaleCorrection'][  np.argwhere(self.star['data_model_groups']['GroupNumber'] == groupNr)[0] ]
        
        ##################
        # K-means clustering:
        ##################
        print( "Running K-means clustering analysis for " + str(P) + " particles into " + str(n_clusters) + " clusters" )
        t0 = time.time()
        if bool(miniBatch):
            print( "TODO: determine number of jobs for K-means" )
            k_means = sklearn.cluster.MiniBatchKMeans( n_clusters=n_clusters, init_size=3*n_clusters+1 )
        else: 
            k_means = sklearn.cluster.KMeans( n_clusters=n_clusters, n_jobs=12 )
        #Kmeans_in = np.vstack( [DefocusMean, part_GroupScaleCorrection]).transpose()
        Kmeans_in = np.vstack( [DefocusU,DefocusV, part_GroupScaleCorrection]).transpose()
        Kmeans_in = sklearn.preprocessing.robust_scale( Kmeans_in )
        k_predict = k_means.fit_predict( Kmeans_in  )
        t1 = time.time()
        print( "Cluster analysis finished in (s): " + str(t1-t0) )
        
        if self.verbose >= 2:
            plt.figure()
            plt.scatter( DefocusMean, part_GroupScaleCorrection, c=k_predict)
            plt.xlabel( "Defocus ($\AA$)" )
            plt.ylabel( "Group scale correction (a.u.)" )
            plt.title("K-means on Defocus")
        
        ##################
        # Save the results in a new particles .star file:
        ##################
        # Replace, add one to group number because Relion starts counting from 1

        particleKey = "data_"
        
        # Add the GroupName field to the star file
        self.star[particleKey]['GroupName'] = [""] * len( self.star[particleKey]['GroupNumber'] )
        for J, groupName in enumerate( k_predict ):
            self.star[particleKey]['GroupName'][J] = 'G' + str(groupName + 1)
            
        # Build a new group number count
        groupCount = np.zeros_like( self.star[particleKey]['GroupNumber'] )
        for J in np.arange(0,len(groupCount)):
            groupCount[J] = np.sum( self.star[particleKey]['GroupNumber'] == J )
        self.star[particleKey]['GroupNumber'] = groupCount
            
        # Recalculate number of particles in each group (ACTUALLY THIS SEEMS NOT NECESSARY)
        #GroupNr = np.zeros( np.max( k_predict )+1 )
        #for J in xrange( np.min( k_predict), np.max( k_predict ) ):
        #    GroupNr[J] = np.sum( k_predict == J )
        #    pass
        #
        #for J in xrange(0, len(rln.star[particleKey]['GroupNumber']) ):
        #    rln.star[particleKey]['GroupNumber'][J] = GroupNr[ k_predict[J] ]
            
    def saveDataStar( self, outputName, particleKey = "data_" ):
        """
        Outputs a relion ..._data.star file that has been pruned, regrouped, etc. to outputName
        """
        
        if outputName == None:
            # Need to store input star names, and figure out which was the last loaded particles.star file.
            # [outFront, outExt] = os.path.splitext()
            raise IOError( "Default filenames for saveDataStar not implemented yet" )
            
        # TODO: more general star file output    
        # Let's just hack this
        fh = open( outputName, 'wb' )
        fh.write( "\ndata_\n\nloop_\n")
        
        # Since we made self.star an OrderedDict we don't need to keep track of index ordering
        
        headerKeys = self.star[particleKey].keys()
        
#        headerDict = { 'Voltage':1, 'DefocusU':2, 'DefocusV':3, 'DefocusAngle':4, "SphericalAberration":5, "DetectorPixelSize":6,
#                     "CtfFigureOfMerit":7, "Magnification":8, "AmplitudeContrast":9, "ImageName":10, "CoordinateX":11, 
#                     "CoordinateY":12, "NormCorrection":13, "MicrographName":14, "GroupNumber":15, "OriginX":16, 
#                     "OriginY":17, "AngleRot":18, "AngleTilt":19, "AnglePsi":20, "ClassNumber":21, "LogLikeliContribution": 22, 
#                     "NrOfSignificantSamples":23, "MaxValueProbDistribution":24, "GroupName":25 }
#        lookupDict = dict( zip( headerDict.values(), headerDict.keys() ) )             
#        
#        for J in lookupDict.keys():
#            if not lookupDict[J] in self.star['data_']:
#                lookupDict.pop( J )

#        for J in np.sort(lookupDict.keys()):
#            # print( "Column: " + "_rln" + lookupDict[J+1] + " #" + str(J+1) )
#            fh.write( "_rln" + lookupDict[J] + " #" + str(J) + "\n")

        for J, key in enumerate(headerKeys):
            # print( "Column: " + "_rln" + lookupDict[J+1] + " #" + str(J+1) )
            fh.write( "_rln" + key + " #" + str(J) + "\n")

        
        # lCnt = len( headerKeys ) 
        P = len( self.star[particleKey]['ImageName'] )
        for I in np.arange(0,P):
            fh.write( "    ")
            for J, key in enumerate(headerKeys):
                fh.write( str( self.star[particleKey][key][I] ) )
                fh.write( "   " )
            fh.write( "\n" )
        fh.close()
        
    def saveDataAsPar( self, outputName, particleKey = "data_" ):
        """
        Saves a Relion .star file as a Frealign .par meta-data file.  Also goes through all the particles in the 
        Relion .star and generates an appropriate meta-MRC particle file for Frealign.
        
        Output name should end with "_1_r1.par" typically for single-class refinement.
        
        Currently the massive Frealign .mrc file is not memory mapped so all the particles must fit into memory.
        
        Also no comment lines are written to the .par file.
        """

        
        
        partCount = len( self.star['data_']['MicrographName'] )
        # Need AnglePsi, AngleTilt, and AngleRot
        if not 'AnglePsi' in self.star['data_']:
            self.star['data_']['AnglePsi'] = np.zeros( partCount, dtype='float32' )
        if not 'AngleTilt' in self.star['data_']:
            self.star['data_']['AngleTilt'] = np.zeros( partCount, dtype='float32' )
        if not 'AngleRot' in self.star['data_']:
            self.star['data_']['AngleRot'] = np.zeros( partCount, dtype='float32' )
        if not 'OriginY' in self.star['data_']:
            self.star['data_']['OriginY'] = np.zeros( partCount, dtype='float32' )
        if not 'OriginX' in self.star['data_']:
            self.star['data_']['OriginX'] = np.zeros( partCount, dtype='float32' )
        if not 'Magnification' in self.star['data_']:
            self.star['data_']['Magnification'] = np.zeros( partCount, dtype='float32' )     
        if not 'GroupNumber' in self.star['data_']:
            self.star['data_']['GroupNumber'] = np.zeros( partCount, dtype='uint16' )    
        if not 'DefocusU' in self.star['data_']:
            self.star['data_']['DefocusU'] = np.zeros( partCount, dtype='float32' )
        if not 'DefocusV' in self.star['data_']:
            self.star['data_']['DefocusV'] = np.zeros( partCount, dtype='float32' )
        if not 'DefocusAngle' in self.star['data_']:
            self.star['data_']['DefocusAngle'] = np.zeros( partCount, dtype='float32' )    
            
        
        with open( outputName, 'wb' ) as fh:
            #fh.write( "       C      PSI    THETA      PHI      SHX      SHY      MAG     FILM      DF1      DF2   ANGAST\n" )
            for J in xrange(partCount):
                fh.write( "%8d"%(J+1) + " %8.3f"%self.star['data_']['AnglePsi'][J] 
                                + " %8.3f"%self.star['data_']['AngleTilt'][J]  + " %8.3f"%self.star['data_']['AngleRot'][J] 
                                + " %8.3f"%self.star['data_']['OriginY'][J] +" %8.3f"%self.star['data_']['OriginX'][J]
                                + " %8.2f"%self.star['data_']['Magnification'][J] 
                                + " %8d"%self.star['data_']['GroupNumber'][J] + " %8.2f"%self.star['data_']['DefocusU'][J]
                                + " %8.2f"%self.star['data_']['DefocusV'][J] + " %8.3f"%self.star['data_']['DefocusAngle'][J] + "\n")
                pass
        
        # Ok and now we need to make a giant particles file?
        mrcName, _= os.path.splitext( outputName ) 
        mrcName = mrcName + ".mrc"
        
        imageNames = np.zeros_like( self.star['data_']['ImageName'] )
        for J, name in enumerate( self.star['data_']['ImageName'] ):
            imageNames[J] = name.split('@')[1]
        uniqueNames = np.unique( imageNames ) # Ordering is preserved, thankfully!
        

        # It would be much better if we could write to a memory-mapped file rather than building the entire array in memory
        # However this is a little buggy in numpy.
        # https://docs.python.org/2/library/mmap.html instead?
        
        particleList = []
        for uniqueName in uniqueNames:
            particleList.extend( zorro.ioMRC.MRCImport(uniqueName) )
            
        print( "DONE building particle list!" )
        print( len(particleList) )
        
        particleArray = np.array( particleList )
        del particleList
        
        # We do have the shape parameter that we can pass in to pre-pad the array with all zeros.
        zorro.ioMRC.MRCExport( particleArray, mrcName, shape=None )
        pass
        
    def saveCtfImagesStar( self, outputName, zorroList = "*.dm4.log", physicalPixelSize=5.0, amplitudeContrast=0.08 ):
        """
        Given a glob pattern, generate a list of zorro logs, or alternatively one can pass in a list.  For each
        zorro log, load it, extract the pertinant info (defocus, etc.).  This is a file ready for particle 
        extraction, with imbedded Ctf information.
        """
        zorroList = glob.glob( zorroList )

        headerDict = { 'MicrographName':1, 'CtfImage':2, 'DefocusU':3, 'DefocusV':4, 'DefocusAngle':5, 
                      'Voltage':6, 'SphericalAberration':7, 'AmplitudeContrast':8, 'Magnification':9, 
                      'DetectorPixelSize':10, 'CtfFigureOfMerit': 11 }
        lookupDict = dict( zip( headerDict.values(), headerDict.keys() ) )   
        data = {}
        for header in headerDict:      
            data[header] = [None]*len(zorroList)
            
        zorroReg = zorro.ImageRegistrator()    
        for J, zorroLog in enumerate(zorroList):
            zorroReg.loadConfig( zorroLog, loadData=False )
            
            data['MicrographName'][J] = zorroReg.files['sum']
            data['CtfImage'][J] = os.path.splitext( zorroReg.files['sum'] )[0] + ".ctf:mrc"
            # CTF4Results = [Micrograph number, DF1, DF2, Azimuth, Additional Phase shift, CC, max spacing fit-to]
            data['DefocusU'][J] = zorroReg.CTF4Results[1]
            data['DefocusV'][J] = zorroReg.CTF4Results[2]
            data['DefocusAngle'][J] = zorroReg.CTF4Results[3]
            data['CtfFigureOfMerit'][J] = zorroReg.CTF4Results[5]
            data['Voltage'][J] = zorroReg.voltage
            data['SphericalAberration'][J] = zorroReg.C3
            data['AmplitudeContrast'][J] = amplitudeContrast
            data['DetectorPixelSize'][J] = physicalPixelSize
            data['Magnification'][J] = physicalPixelSize / (zorroReg.pixelsize * 1E-3)
            
        with open( outputName, 'wb' ) as fh:

            fh.write( "\ndata_\n\nloop_\n")
            for J in np.sort(lookupDict.keys()):
                # print( "Column: " + "_rln" + lookupDict[J+1] + " #" + str(J+1) )
                fh.write( "_rln" + lookupDict[J] + " #" + str(J) + "\n")
            
            lCnt = len( lookupDict ) 
            for I in np.arange(0,len(zorroList)):
                fh.write( "    ")
                for J in np.arange(0,lCnt):
                    fh.write( str( data[lookupDict[J+1]][I] )  )
                    fh.write( "   " )
                fh.write( "\n" )
            
        
    def __loadPar( self, parname ):
        """
        Frealign files normally have 16 columns, with any number of comment lines that start with 'C'
        """
        self.par.append( np.loadtxt( parname, comments='C' ) )
        # TODO: split into a dictionary?  
        # TODO: read comments as well
        self.pcol = {"N":0, "PSI":1, "THETA":2, "PHI":3, "SHX":4, "SHY":5, "MAG":6, "FILM":7, "DF1":8, "DF2":9, "ANGAST":10,
                        "OCC":11, "LogP":12, "SIGMA":13, "SCORE":14, "CHANGE":15 }
               
    def __loadStar( self, starname ):
        with open( starname, 'rb' ) as starFile:
            starLines = starFile.readlines()
        
            # Remove any lines that are blank
            blankLines = [I for I, line in enumerate(starLines) if ( line == "\n" or line == " \n") ]
            for blank in sorted( blankLines, reverse=True ):
                del starLines[blank]
        
            # Top-level keys all start with data_
            headerTags = []; headerIndices = []
            for J, line in enumerate(starLines):
                if line.startswith( "data_" ): # New headerTag
                    headerTags.append( line.strip() )
                    headerIndices.append( J )
            # for end-of-file
            headerIndices.append(-1)
            
            # Build dict keys
            for K, tag in enumerate( headerTags ):
                self.star[tag] = {}
                # Read in _rln lines and assign them as dict keys
                
                lastHeaderIndex = 0
                foundLoop = False
                
                if headerIndices[K+1] == -1: #-1 is not end of the array for indexing
                    slicedLines = starLines[headerIndices[K]:]
                else:
                    slicedLines = starLines[headerIndices[K]:headerIndices[K+1]] 
                    
                for J, line in enumerate( slicedLines ):
                    if line.startswith( "loop_" ):
                        foundLoop = True
                    elif line.startswith( "_rln" ):
                        lastHeaderIndex = J
                        # Find all the keys that start with _rln, they are sub-dict keys
                        newKey = line.split()[0][4:]
                        newValue = line.split()[1]
                        # If newValue starts with a #, strip it
                        newValue = newValue.lstrip( '#' )
                        # Try to make newValue an int or float, or leave it as a string if that fails
                        try:
                            self.star[tag][newKey] = np.int( newValue )
                        except:
                            try: 
                                self.star[tag][newKey] = np.float( newValue )
                            except: # leave as a string
                                self.star[tag][newKey] = newValue
        
                # Now run again starting at lastHeaderIndex
                if foundLoop: 
                    # Need to check to make sure it's not an empty dict
                    if self.star[tag] == {}:
                        continue
                    
                    endIndex = len(slicedLines)
                        
                    # Reverse sub-dictionary so we can determine by which column goes to which key
                    lookup = dict( zip( self.star[tag].values(), self.star[tag].keys() ) )
                    # Pre-allocate, we can determine types later.   
                    itemCount = endIndex - lastHeaderIndex - 1
                    testSplit = slicedLines[lastHeaderIndex+1].split()
                    for K, test in enumerate( testSplit ):
                        self.star[tag][lookup[K+1]] = [None] * itemCount
        
                    # Loop through and parse items
                    for J, line in enumerate( slicedLines[lastHeaderIndex+1:endIndex] ):
                        for K, item in enumerate( line.split() ):
                            self.star[tag][lookup[K+1]][J] = item
                    pass
                
                    # Try to convert to int, then float, otherwise leave as a string
                    for key in self.star[tag].keys():
                        try:
                            self.star[tag][key] = np.asarray( self.star[tag][key], dtype='int' )
                        except:
                            try:
                                self.star[tag][key] = np.asarray( self.star[tag][key], dtype='float' )
                            except: 
                                self.star[tag][key] = np.asarray( self.star[tag][key] )
                                pass

    def __loadMRC( self, mrcname ):
        mrcimage, mrcheader  = zorro.ioMRC.MRCImport( mrcname, returnHeader=True )
        self.mrc.append( mrcimage )
        self.mrc_header.append( mrcheader )
        
    def __loadBox( self, boxname ):
        self.box.append( np.loadtxt( boxname ) )

# End of relion class




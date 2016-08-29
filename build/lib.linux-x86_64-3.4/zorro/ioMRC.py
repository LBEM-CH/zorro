# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import, unicode_literals
import numpy as np

def MRCImport( MRCfilename, useMemmap = False, endian='le', fileConvention = "default", returnHeader = False ):
    """
    MRCImport( MRCfilename, useMemmap = False, endian='le', fileConvention = "default" )
    Created on Thu Apr 09 11:05:07 2015
    @author: Robert A. McLeod
    @email: robbmcleod@gmail.com OR robert.mcleod@unibas.ch
    
    This is a bare-bones import script, it just imports the image data and returns 
    it in an numpy array. Can also return the header as the second argument, with returnHeader = True
    
    endian can be big-endian == 'be' or little-endian == 'be'
    """
    # Check to see if it end's with MRC or MRCS, if not, add the extension
    
    buffersize = 65536 # Quite arbitrary, in bytes
    
    if endian == 'le':
        endchar = '<'
    else:
        endchar = '>'
    
    with open( MRCfilename, 'rb', buffering=buffersize ) as f:
        # Get dimensions, in format [nz, ny, nx] (stored as [nx,ny,nz] in the file)
        dimensions = np.flipud( np.fromfile( f, dtype=endchar+'i4', count=3 ) )
        
        # NZ = 1 for 2-D images
        
        # EMAN2 uses its own version of datatypes,
        # 1 - signed 8 bit
        # 2 - unsigned 8 bit
        # 3 - signed 16 bit
        # 4 - unsigned 16 bit
        # 5 - signed 32 bit
        # 6 - unsigned 32 bit
        # 7 - single precision floating point
        sloppy = None # support for sloppy floats
        MRCdtype = np.fromfile( f, dtype=endchar+'i4', count=1 )
        if MRCdtype > 16000000: # Hack to fix lack of endian indication in the file header
            # Endianess is backward
            MRCdtype = np.asarray( MRCdtype ).byteswap()[0]
            dimensions = dimensions.byteswap()
            if endchar == '<':
                endchar = '>' 
            else:
                endchar = '<'
        print( "Import MRC: " + str(dimensions) + ", MRC dtype: " + str(MRCdtype) )    
        
        fileConvention.lower()
        if fileConvention == "eman2":
            if MRCdtype == 1:
                npdtype = 'i1'
            elif MRCdtype == 2:
                npdtype = 'u1'
            elif MRCdtype == 3:
                npdtype = 'i2'
            elif MRCdtype == 4:
                npdtype = 'u2'
            elif MRCdtype == 5:
                npdtype = 'i4'
            elif MRCdtype == 6:
                npdtype = 'u4'
            elif MRCdtype == 7:
                npdtype = 'f4'
            else:
                raise ValueError( "Error: unrecognized default-MRC data type = " + str(MRCdtype) )
        else: # Default
            if MRCdtype == 0:
                npdtype = 'u1'
            elif MRCdtype == 1:
                npdtype = 'u2'
            elif MRCdtype == 2:
                npdtype = 'f4'
            elif MRCdtype == 4:
                npdtype = 'c8'
            # RAM: I've added these sloppy floats for uint16 and uint8
            elif MRCdtype == 1001:
                sloppy = np.float32(0.1)
                npdtype = 'u1'
            elif MRCdtype == 10001:
                # 1-decimal place sloppy
                sloppy = np.float32(0.1)
                npdtype = 'u2'
            elif MRCdtype == 10010:
                # 2-decimal place sloppy
                sloppy = np.float32(0.01)
                npdtype = 'u2'
            else:
                raise ValueError( "Error: unrecognized default-MRC data type = " + str(MRCdtype) )
                
        # Read in pixelsize
        f.seek(40)
        header = {}
        
        cellsize = np.flipud( np.fromfile( f, dtype=endchar + 'f4', count=3 ) )
        header['pixelsize'] = cellsize / (dimensions)
        # print( "MRC found pixel size of: " + str(header['pixelsize']) )
        
        # TODO: return a struct with the MRC header information.
        f.seek(1024)
        if bool(useMemmap):
            image = np.memmap( f, dtype=endchar+npdtype, mode='c', shape=(dimensions[0],dimensions[1],dimensions[2]) )
        else:
            image = np.fromfile( f, dtype=endchar+npdtype, count=np.product(dimensions) )
            
        image = np.squeeze( np.reshape( image, dimensions ) )
        if sloppy is not None:
            image *= sloppy

        if returnHeader:
            return image, header
        else:
            return image
        
def MRCExport( input_image, MRCfilename, endian='le', rounding_decimals=None, sloppy_digits=None, 
              pixelsize=[1.0,1.0,1.0], shape=None ):
    """
    MRCExport( input_image, MRCfilename, endian='le', rounding_digits=None, sloppy_digits=None )
    Created on Thu Apr 02 15:56:34 2015
    @author: Robert A. McLeod
    @email: robbmcleod@gmail.com OR robert.mcleod@unibas.ch
    
    Given a numpy 2-D or 3-D array write it has an MRC file.
    
        rounding_digits = [1,2,3...] rounds floating-point data before saving it.  This
        improves the compressibility of the data at the expends of dynamic range.
    
        sloppy_digits = [1,2] multiplies the data by [10,100] and saves as uint16.
        Sloppiness has preference over rounding
        
        shape is only used if you want to later append to the file, such as merging together Relion particles
        for Frealign.
    
    Specification for MRC is here:
    http://ami.scripps.edu/software/mrctools/mrc_specification.php
    and
    http://bio3d.colorado.edu/imod/doc/mrc_format.txt
    
    Note that MRC definitions are not consistent.
    """

    buffersize = input_image.shape[-1]*input_image.shape[-2]*4
    # Maybe just reading the DM4 filesize as a buffer size would be a good idea?
    # buffersize = np.product( dmImage.dims ) * 4
    
    with open( MRCfilename, 'wb', buffering=buffersize ) as f:
        # f.write( '\x00\x00' ) # I don't know why but DM includes two nulls at the start.
        
        if endian == 'le':
            endchar = '<'
        else:
            endchar = '>'
        
        # Write dimensions (MRC is Fortran ordered I suspect)
        """ http://bio3d.colorado.edu/imod/doc/mrc_format.txt """
        if shape == None:
            dimensions = np.flipud( input_image.shape )
        else:
            dimensions = np.flipud( input_image.shape )
        if len(dimensions) == 2: # force to 3-D
            dimensions = np.array( [dimensions[0], dimensions[1],1] )
        dimensions.astype(endchar+'i4').tofile( f )
        # Write data type
        # EMAN2 uses its own version of datatypes,
        # 1 - signed 8 bit
        # 2 - unsigned 8 bit
        # 3 - signed 16 bit
        # 4 - unsigned 16 bit
        # 5 - signed 32 bit
        # 6 - unsigned 32 bit
        # 7 - single precision floating point
        # print input_image.dtype
        
        # 64-bit floats are automatically down-cast
        if input_image.dtype == 'float32' or input_image.dtype == '>f4' or input_image.dtype == 'float64' or input_image.dtype == '>f8':
            # Sloppiness has precidence over rounding.
            if sloppy_digits == 1:
                MRCmode = np.int64( 10001 ).astype( endchar + "i4" )
                npdtype = 'u2'
                input_image *= 10
            elif sloppy_digits == 2:
                MRCmode = np.int64( 10010 ).astype( endchar + "i4" )
                npdtype = 'u2'
                input_image *= 100
            elif rounding_decimals != None and rounding_decimals >= 1:
                input_image = np.round( input_image, decimals = np.floor(rounding_decimals) )
                MRCmode = np.int64( 2 ).astype( endchar + "i4" )
                npdtype = 'f4'
            else:
                MRCmode = np.int64( 2 ).astype( endchar + "i4" )
                npdtype = 'f4'
        elif input_image.dtype == 'uint16':
            MRCmode = np.int64( 1 ).astype( endchar + "i4" )
            npdtype = 'u2'
        elif input_image.dtype == 'uint32':
            MRCmode = np.int64( 6 ).astype( endchar + "i4" )
            npdtype = 'u4'
        elif input_image.dtype == 'uint8':
            MRCmode = np.int64( 0 ).astype( endchar + "i4" )
            npdtype = 'u1'
        elif input_image.dtype == 'complex64':
            MRCmode = np.int64( 4 ).astype( endchar + "i4" )
            npdtype = 'c8'
        else: 
            raise ValueError( "Warning: Unknown dtype for MRC encountered = " + str(input_image.dtype) )
        MRCmode.tofile(f)
        # Print NXSTART, NYSTART, NZSTART
        np.array( [0,0,0], dtype=endchar+"i4" ).tofile(f)
        # Print MX, MY, MZ, the number of pixels
        np.array( dimensions, dtype=endchar+"i4" ).tofile(f)
        # Print cellsize = pixelsize * dimensions
        if not hasattr(pixelsize, "__len__") or len(pixelsize) == 1:
            cellsize = np.array( [pixelsize,pixelsize,pixelsize]  ) * dimensions
        elif len(pixelsize) == 2:
            cellsize = [dimensions[0]*pixelsize[1], dimensions[1]*pixelsize[0], 100.0]
        else:
            cellsize = np.flipud(np.array( pixelsize )) * np.array( dimensions )
        np.array( cellsize, dtype=endchar+"f4" ).tofile(f)
        # Print default cell angles
        np.array( [90.0,90.0,90.0], dtype=endchar+"f4" ).tofile(f)
        # Print axis associations
        np.array( [1,2,3], dtype=endchar+"i4").tofile(f)
        # Print statistics
        np.array( [np.min(input_image), np.max(input_image), np.mean(input_image)], dtype=endchar+"f4" ).tofile(f)
        f.seek( 209 )
        np.array( "MAP ", dtype="|S1" ).tofile(f)
        # Write a machine stamp, '\x17\x17' for big-endian or '\x68\x65
        f.seek( 213 )
        if endian == 'le':
            np.array( "\x68\x65", dtype="|S1" ).tofile(f)
        else:
            np.array( "\x17\x17", dtype="|S1" ).tofile(f)
        # That should be enough for 2dx
        
        # Leave header and write file
        f.seek(1024)
        # This is amazingly slow to write.  Perhaps I need to allocate the total space of the file first?
        # Or it is just a Python problem because it does string conversions?
        input_image.astype( endchar + npdtype ).tofile(f)
        f.close()
    
    # t1 = time()
    # print( "Total time to convert image stack (s): " + str(t1-t0) )
    return MRCfilename
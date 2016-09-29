from __future__ import division, print_function, absolute_import, unicode_literals
"""
Conventional MRC2014 and on-the-fly compressed MRCZ file interface

CCPEM MRC2014 specification:

http://www.ccpem.ac.uk/mrc_format/mrc2014.php

IMOD specification:

http://bio3d.colorado.edu/imod/doc/mrc_format.txt

Testing:

http://emportal.nysbc.org/mrc2014/

Tested output on: Gatan GMS, IMOD, Chimera, Relion
"""

import os, os.path
import numpy as np
import numexprz.cpuinfo
try: 
    import blosc
    import bloscpack as bp
    bloscPresent = True
except:
    print( "blosc compressor not found, MRCZ format disabled." )


BUFFERSIZE = 2**22 # Quite arbitrary, in bytes (hand-optimized)
COMPRESSOR_ENUM = { 0:None, 1:'zlib', 2:'lz4', 3:'lz4hc', 4:'blosclz', 5:'zstd' }
REVERSE_COMPRESSOR_ENUM = { None:0, 'zlib':1, 'lz4':2, 'lz4hc':3, 'blosclz':4, 'zstd':5 }

IMOD_ENUM = { 0: 'i1', 1:'i2', 2:'f4', 4:'c8', 101:'u1' }
EMAN2_ENUM = { 1: 'i1', 2:'u1', 3:'i2', 4:'u2', 5:'i4', 6:'u4', 7:'f4' }
REVERSE_IMOD_ENUM = { 'int8':0, 'i1':0, 'uint4':101, 'int16':1, 'i2':1,
                   'float64':2, 'f8':2, 'float32':2, 'f4':2,
                   'complex128':4, 'c16':4, 'complex64':4, 'c8':4 }

def defaultHeader( ):
    """
    Returns a default MRC header dictionary with all fields with default values.
    """
    header = {}
    header['fileConvention'] = "imod"
    header['endian'] = 'le'
    header['MRCtype'] = 0
    header['dimensions'] = np.array( [0,0,0], dtype=int )
    header['dtype'] = 'u1'
    
    header['compressor'] = None
    header['packedBytes'] = 0
    header['cLevel'] = 1
    
    header['maxImage'] = 1.0
    header['minImage'] = 0.0
    header['meanImage'] = 0.0
    
    header['pixelsize'] = 0.1 
    header['pixelunits'] = u"nm" # Can be "\AA" for Angstroms
    header['voltage'] = 300.0 # kV
    header['C3'] = 2.7 # mm
    header['gain'] = 1.0 # counts/electron
    
    header['n_threads'] = len(numexprz.cpuinfo.cpu.info)
    
    return header
    
def MRCImport( MRCfilename, useMemmap = False, endian='le', 
              fileConvention = "imod", returnHeader = False,
              n_threads = None, reverseStripes=False ):
    """
    MRCImport( MRCfilename, useMemmap = False, endian='le', fileConvention = "default" )
    Created on Thu Apr 09 11:05:07 2015
    @author: Robert A. McLeod
    @email: robert.mcleod@unibas.ch
    
    This is a bare-bones import script, it just imports the image data and returns 
    it in an numpy array. 
    
    Can also return the header as the second argument, with returnHeader = True
    
    endian can be big-endian == 'be' or little-endian == 'le'
    """
    # Check to see if it end's with MRC or MRCS, if not, add the extension

        
    with open( MRCfilename, 'rb', buffering=BUFFERSIZE ) as f:
        # Read in header as a dict
        header = readMRCHeader( MRCfilename, endian=endian, fileConvention = "imod" )
        # Support for compressed data in MRCZ
        #print( "DEBUG: compressor = %s" % header['compressor'] )
        #print( "DEBUG: compressor enum = %d" % REVERSE_COMPRESSOR_ENUM[header['compressor']] )
        
        if ( (header['compressor'] in REVERSE_COMPRESSOR_ENUM) 
            and (REVERSE_COMPRESSOR_ENUM[header['compressor']] > 0) ):
            return __MRCZImport( f, header, endian=endian, fileConvention = fileConvention, 
                                returnHeader = returnHeader, n_threads=n_threads )
        # Else save as MRC file
                                
        f.seek(1024 + header['extendedBytes'])
        if bool(useMemmap):
            image = np.memmap( f, dtype=header['dtype'], 
                              mode='c', 
                              shape=(header['dimensions'][0],header['dimensions'][1],header['dimensions'][2]) )
        else:
            image = np.fromfile( f, dtype=header['dtype'], 
                                count=np.product(header['dimensions']) )
                                
        # print( "DEBUG 1: ioMRC.MRCImport # nans: %d" % np.sum(np.isnan(image)) )    
        if header['MRCtype'] == 101:
            # Seems the 4-bit is interlaced ...
            interlaced_image = image
            
            image = np.empty( np.product(header['dimensions']), dtype=header['dtype'] )
            if bool(reverseStripes):
                # Bit-shift and Bit-and to seperate decimated pixels
                image[1::2] = np.left_shift(interlaced_image,4) / 15
                image[0::2] = np.right_shift(interlaced_image,4)
            else: # Default
                image[0::2] = np.left_shift(interlaced_image,4) / 15
                image[1::2] = np.right_shift(interlaced_image,4)

        # print( "DEBUG 2: ioMRC.MRCImport # nans: %d" % np.sum(np.isnan(image)) )
        image = np.squeeze( np.reshape( image, header['dimensions'] ) )
        # print( "DEBUG 3: ioMRC.MRCImport # nans: %d" % np.sum(np.isnan(image)) )
        if returnHeader:
            return image, header
        else:
            return image
       
def __MRCZImport( f, header, endian='le', fileConvention = "imod", returnHeader = False, n_threads=None ):
    """
    Equivalent to MRCImport, but for compressed data using the blosc library.
    
    The following compressors are supported: 
        'zlib'
        'zstd'
        'lz4' 
    
    Memory mapping is not possible in this case.
    """
    if n_threads == None:
        n_threads = len(numexprz.cpuinfo.cpu.info)
        
    f.seek( 1024 + header['extendedBytes'] )
    byteArray = f.read( header['packedBytes'] )
    bp.constants.blosc.set_nthreads( n_threads )
    
    # TODO: find out how this bp.unpack is storing the array dimensions
    image = bp.unpack_ndarray_str( byteArray )

    print( "unpacked image shape: " + str(image.shape) )
    
    if header['MRCtype'] == 101:
        # Seems the 4-bit is interlaced ...
        interlaced_image = image
            
        image = np.empty( np.product(header['dimensions']), dtype=header['dtype'] )
        # Bit-shift and Bit-and to seperate decimated pixels
        image[0::2] = np.left_shift(interlaced_image,4) / 15
        image[1::2] = np.right_shift(interlaced_image,4)

    # We don't need to reshape packed data.
    image = np.squeeze( image, axes=[2,1,0] )
    
    if returnHeader:
        return image, header
    else:
        return image

    
def readMRCHeader( MRCfilename, endian='le', fileConvention = "imod" ):
    """
    Reads in the first 1024 bytes from an MRC file and parses it into a Python dictionary, yielding 
    header information.
    """
    if endian == 'le':
        endchar = '<'
    else:
        endchar = '>'
        
    header = {}
    with open( MRCfilename, 'rb' ) as f:
        diagStr = ""
        # Get dimensions, in format [nz, ny, nx] (stored as [nx,ny,nz] in the file)
        header['dimensions'] = np.flipud( np.fromfile( f, dtype=endchar+'i4', count=3 ) )
    
        header['MRCtype'] = int( np.fromfile( f, dtype=endchar+'i4', count=1 )[0] )
        if header['MRCtype'] > 16000000: # Hack to fix lack of endian indication in the file header
            # Endianess is backward
            header['MRCtype'] = int( np.asarray( header['MRCtype'] ).byteswap()[0] )
            header['dimensions'] = header['dimensions'].byteswap()
            if endchar == '<':
                endchar = '>' 
            else:
                endchar = '<'
                
        # Extract compressor from dtype > 10000
        header['compressor'] = COMPRESSOR_ENUM[ np.floor_divide(header['MRCtype'], 10000) ]
        header['MRCtype'] = np.remainder( header['MRCtype'], 10000 )
        
        fileConvention = fileConvention.lower()
        if fileConvention == "imod":
            diagStr += ("ioMRC.readMRCHeader: MRCtype: %s, compressor: %s, dimensions %s" % 
                (IMOD_ENUM[header['MRCtype']],header['compressor'], header['dimensions'] ) )
        elif fileConvention == "eman2":
            diagStr += ( "ioMRC.readMRCHeader: MRCtype: %s, compressor: %s, dimensions %s" % 
                (EMAN2_ENUM[header['MRCtype']],header['compressor'], header['dimensions'] ) )

        
        if fileConvention == "eman2":
            try:
                header['dtype'] = EMAN2_ENUM[ header['MRCtype'] ]
            except:
                raise ValueError( "Error: unrecognized EMAN2-MRC data type = " + str(header['MRCtype']) )
                
        elif fileConvention == "imod": # Default is imod
            try:
                header['dtype'] = IMOD_ENUM[ header['MRCtype'] ]
            except:
                raise ValueError( "Error: unrecognized IMOD-MRC data type = " + str(header['MRCtype']) )
                

        # Apply endian-ness to NumPy dtype
        header['dtype'] = endchar + header['dtype']
        # Read in pixelsize
        f.seek(40)
        cellsize = np.flipud( np.fromfile( f, dtype=endchar + 'f4', count=3 ) )
        header['pixelsize'] = cellsize / (header['dimensions'])
        # MRC is Angstroms, so
        if not 'pixelunits' in header:
            header['pixelunits'] = "nm"
            
        if header['pixelunits'] == "\AA":
            pass
        elif header['pixelunits'] == "\mum":
            header['pixelsize'] *= 1E-5
        elif header['pixelunits'] == "pm":
            header['pixelsize'] *= 100.0
        else: # Default to nm
            header['pixelsize'] *= 0.1
            
        # Read in [X,Y,Z] array ordering
        f.seek(64)
        axesTranpose = np.fromfile( f, dtype=endchar + 'i4', count=3 ) - 1
        # Currently I don't use this
        
        # Read in statistics
        f.seek(76)
        (header['minImage'], header['maxImage'], header['meanImage']) = np.fromfile( f, dtype=endchar + 'f4', count=3 )

        f.seek(92)
        header['extendedBytes'] = int( np.fromfile( f, dtype=endchar + 'i4', count=1) )
        if header['extendedBytes'] > 0:
            diagStr += ", extended header %d" % header['extendedBytes']

        # Read in kV, C3, and gain
        f.seek(132)
        header['voltage']  = np.fromfile( f, dtype=endchar + 'f4', count=1 )
        header['C3']  = np.fromfile( f, dtype=endchar + 'f4', count=1 )
        header['gain']  = np.fromfile( f, dtype=endchar + 'f4', count=1 )
        
        diagStr += ", voltage: %.1f, C3: %.2f, gain: %.2f" % (header['voltage'], header['C3'], header['gain']) 

        # Read in size of packed data
        f.seek(144)
        # Have to convert to Python int to avoid index warning.
        header['packedBytes'] = int( np.fromfile( f, dtype=endchar + 'i8', count=1) )
        if header['packedBytes'] > 0:
            diagStr += ", packedBytes: %d" % header['packedBytes']
        print( diagStr )
        
        # How many bytes in an MRC
        return header
        
def MRCExport( input_image, MRCfilename, endian='le', dtype=None, 
               pixelsize=[0.1,0.1,0.1], pixelunits=u"nm", shape=None, 
               voltage = 0.0, C3 = 0.0, gain = 1.0,
               compressor=None, cLevel = 1, n_threads=None, quickStats=True ):
    """
    MRCExport( input_image, MRCfilename, endian='le', shape=None, compressor=None, cLevel = 1 )
    Created on Thu Apr 02 15:56:34 2015
    @author: Robert A. McLeod
    @email: robert.mcleod@unibas.ch
    
    Given a numpy 2-D or 3-D array `input_image` write it has an MRC file `MRCfilename`.
    
        dtype will cast the data before writing it.
        
        pixelsize is [z,y,x] pixel size (singleton values are ok for square/cubic pixels)
        
        pixelunits is "\AA" for Angstroms, "pm" for picometers, "\mum" for micrometers, 
        or "nm" for nanometers.  MRC standard is always Angstroms, so pixelsize 
        is converted internally from nm to Angstroms if necessary
        
        shape is only used if you want to later append to the file, such as merging together Relion particles
        for Frealign.
        
        voltage is accelerating potential in keV, defaults to 300.0
        
        C3 is spherical aberration in mm, defaults to 2.7 mm
        
        gain is detector gain (counts/primary electron), defaults to 1.0 (for counting camera)
        
        compressor is a choice of 'lz4', 'zlib', or 'zstd', plus 'blosclz', 'lz4hc'  
        'zstd' generally gives the best compression performance, and is still almost 
           as fast as 'lz4' with cLevel = 1
        'zlib' is easiest to decompress with other utilities.
        
        cLevel is the compression level, 1 is fastest, 11 is very-slow.  The compression
        ratio will rise slowly with cLevel.
        
        n_threads is number of threads to use for blosc compression
        
        quickStats = True estimates the image mean, min, max from the first frame only,
        which saves a lot of computational time for stacks.
    
    Note that MRC definitions are not consistent.  Generally we support the IMOD schema.
    """

    header = {}
    if endian == 'le':
        endchar = '<'
    else:
        endchar = '>'
    if dtype == None:
        # TODO: endian support
        header['dtype'] = endchar + input_image.dtype.descr[0][1].strip( "<>|" )
    else:
        header['dtype'] = dtype
        
    # Now we need to filter dtype to make sure it's actually acceptable to MRC
    if header['dtype'] == 'float64' or 'f8' in header['dtype']:
        header['dtype'] = 'f4'
    if not header['dtype'].strip( "<>|" ) in IMOD_ENUM.values():
        raise TypeError( "ioMRC.MRCExport: Unsupported dtype cast for MRC %s" % header['dtype'] )
        
    header['dimensions'] = input_image.shape
    header['pixelsize'] = pixelsize
    header['pixelunits'] = pixelunits
    header['compressor'] = compressor
    header['cLevel'] = cLevel
    header['shape'] = shape
    
    # This overhead calculation is annoying but many 3rd party tools that use 
    # MRC require these statistical parameters.
    if bool(quickStats) and input_image.ndim == 3:
        header['maxImage'] = np.max( input_image[0,:,:] )
        header['minImage'] = np.min( input_image[0,:,:] )
        header['maxImage'] = np.mean( input_image[0,:,:] )
    else:
        header['maxImage'] = np.max( input_image )
        header['minImage'] = np.min( input_image )
        header['maxImage'] = np.mean( input_image )
    
    header['voltage'] = voltage
    if not bool( header['voltage'] ):
        header['voltage'] = 0.0
    header['C3'] = C3
    if not bool( header['C3'] ):
        header['C3'] = 0.0
    header['gain'] = gain
    if not bool( header['gain'] ):
        header['gain'] = 1.0
    
    header['compressor'] = compressor
    header['cLevel'] = cLevel
    if n_threads == None:
        n_threads = len(numexprz.cpuinfo.cpu.info)
    header['n_threads'] = n_threads
    
    # TODO: can we detect the number of cores without adding a heavy dependancy?
    
    if dtype == 'uint4':
        # Decimate to packed 4-bit
        input_image = input_image.astype('uint8')
        header['dtype'] = endchar + 'u1'
        input_image = input_image[:,:,::2] + np.left_shift(input_image[:,:,1::2],4)
        
    __MRCExport( input_image, header, MRCfilename )
    # Generate a header, if we were not passed one.
        
def __MRCExport( input_image, header, MRCfilename ):
    """
    MRCExport private interface with a dictionary rather than a mess of function 
    arguments.
    """
    with open( MRCfilename, 'wb', buffering=BUFFERSIZE ) as f:
    
        
        if ('compressor' in header 
                and header['compressor'] in REVERSE_COMPRESSOR_ENUM 
                and REVERSE_COMPRESSOR_ENUM[header['compressor']] > 0):
            # compressed MRCZ
            print( "Compressing %s with compressor %s%d" %
                    (MRCfilename, header['compressor'], header['cLevel'] ) )
            bp.constants.blosc.set_nthreads( header['n_threads'] )
            blosc_args = bp.BloscArgs( cname=header['compressor'], 
                                      clevel=header['cLevel'], 
                                        shuffle = blosc.BITSHUFFLE )
                                        
            # TODO: find out how this bp.unpack is storing the array dimensions     
            # Possibly for C-compatibility we will need something simpler.
            # blosc.compress( input_image.tobytes(), typesize=input_image.dtype.itemsize )
            packedData = bp.pack_ndarray_str( 
                    input_image.astype( header['dtype']), 
                    blosc_args = blosc_args )
            header['packedBytes'] = len(packedData)
            writeMRCHeader( f, header )
            f.seek(1024)
            f.write( packedData )
            # packedData.tofile()
            
        else: # MRC
            # Get a header and write it to disk        
            writeMRCHeader( f, header )
            
            # Write file as uncompressed MRC
            f.seek(1024)
            # print( "ioMRC.MRCExport: nans: %d" % np.sum( np.isnan(input_image)))
            input_image = input_image.astype( header['dtype'] )
            
            # print( "ioMRC.MRCExport: nans: %d" % np.sum( np.isnan(input_image)))
            
            input_image.tofile(f)
            
            # We have to rewind to header to insert 'packedBytes'
            # Get a header and write it to disk
            
    return 
    
def writeMRCHeader( f, header, endian='le' ):
    """
    Returns a 1024-char long byte array that represents an MRC header for the given arguments in 
    header, plus the numpy dtype-encoding to use.
    
    Use defaultHeader to see all fields.
    
    [headerBytes, npDType] = writeMRCHeader( header )
    
    Maybe it would be easier if we passed in the numpy ndarray that represents the data?
    """
    if endian == 'le':
        endchar = '<'
    else:
        endchar = '>'
        
    f.seek(0)
    # Write dimensions (MRC is Fortran ordered I suspect)
    """ http://bio3d.colorado.edu/imod/doc/mrc_format.txt """
    if len(header['dimensions']) == 2: # force to 3-D
        dimensions = np.array( [1, header['dimensions'][0], header['dimensions'][1]] )
    else: 
        dimensions = np.array( header['dimensions'] )
    # Flip to Fortran order
    dimensions = np.flipud( dimensions )
    dimensions.astype(endchar+'i4').tofile( f )
    
    # 64-bit floats are automatically down-cast
    dtype = header['dtype'].lower().strip( '<>|' )
    try:
        MRCmode = np.int64( REVERSE_IMOD_ENUM[dtype] ).astype( endchar + "i4" )
    except:
        raise ValueError( "Warning: Unknown dtype for MRC encountered = " + str(dtype) )
        
    # Add 10000 * COMPRESSOR_ENUM to the dtype for compressed data
    if ('compressor' in header 
                and header['compressor'] in REVERSE_COMPRESSOR_ENUM 
                and REVERSE_COMPRESSOR_ENUM[header['compressor']] > 0):
        header['compressor'] = header['compressor'].lower()
        MRCmode += 10000*REVERSE_COMPRESSOR_ENUM[ header['compressor'] ]
        
        # How many bytes in an MRCZ file, so that the file can be appended-to.
        try:
            f.seek(144)
            np.int64( header['packedBytes'] ).astype( endchar + "i8" ).tofile(f)
        except:
            raise KeyError( "'packedBytes' not found in header, required for MRCZ format." )
            
        
    f.seek(12)
    MRCmode.tofile(f)
    

    # Print NXSTART, NYSTART, NZSTART
    np.array( [0,0,0], dtype=endchar+"i4" ).tofile(f)
    # Print MX, MY, MZ, the number of pixels
    np.array( dimensions, dtype=endchar+"i4" ).tofile(f)
    # Print cellsize = pixelsize * dimensions
    if header['pixelunits'] == "\AA":
        AApixelsize = np.array(header['pixelsize'])
    elif header['pixelunits'] == "\mum":
        AApixelsize = np.array(header['pixelsize'])*10000.0
    elif header['pixelunits'] == "pm":
        AApixelsize = np.array(header['pixelsize'])/100.0
    else: # Default is nm
        AApixelsize = np.array(header['pixelsize'])*10.0   
    
    if not hasattr( AApixelsize, "__len__") or len( AApixelsize ) == 1:
        cellsize = np.array( [AApixelsize,AApixelsize,AApixelsize]  ) * dimensions
    elif len(AApixelsize) == 2:
        cellsize = [dimensions[0]*AApixelsize[1], dimensions[1]*AApixelsize[0], 100.0]
    else:
        cellsize = np.flipud(np.array( AApixelsize )) * np.array( dimensions )
        
    np.array( cellsize, dtype=endchar+"f4" ).tofile(f)
    # Print default cell angles
    np.array( [90.0,90.0,90.0], dtype=endchar+"f4" ).tofile(f)
    
    # Print axis associations (we use C ordering internally in all Python code)
    np.array( [1,2,3], dtype=endchar+"i4").tofile(f)
    
    # Print statistics (if available)
    f.seek(76)
    if 'minImage' in header:
        np.float32( header['minImage'] ).astype(endchar+"f4").tofile(f)
    else:
        np.float32( 0.0 ).astype(endchar+"f4").tofile(f)
    if 'maxImage' in header:
        np.float32( header['maxImage'] ).astype(endchar+"f4").tofile(f)
    else:
        np.float32( 1.0 ).astype(endchar+"f4").tofile(f)
    if 'meanImage' in header:
        np.float32( header['meanImage'] ).astype(endchar+"f4").tofile(f)
    else:
        np.float32( 0.0 ).astype(endchar+"f4").tofile(f)
        
    # We'll put the compressor info and number of compressed bytes in 132-204
    # and new metadata
    # RESERVED: 132: 136: 140 : 144 for voltage, C3, and gain
    f.seek(132)
    if 'voltage' in header:
        np.float32( header['voltage'] ).astype(endchar+"f4").tofile(f)
    if 'C3' in header:
         np.float32( header['C3'] ).astype(endchar+"f4").tofile(f)
    if 'gain' in header:
        np.float32( header['gain'] ).astype(endchar+"f4").tofile(f)
        

    
    # Magic MAP_ indicator that tells us this is in-fact an MRC file
    f.seek( 208 )
    np.array( "MAP ", dtype="|S1" ).tofile(f)
    # Write a machine stamp, '\x17\x17' for big-endian or '\x68\x65
    f.seek( 212 )
    if endian == 'le':
        np.array( "\x68\x65", dtype="|S1" ).tofile(f)
    else:
        np.array( "\x17\x17", dtype="|S1" ).tofile(f)
    

    return  
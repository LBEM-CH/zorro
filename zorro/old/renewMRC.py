# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 14:47:19 2015

@author: rmcleod
"""
import ioMRC
import glob

filenames = glob.glob( ("*.mrc", "*.mrcs") )
for filename in filenames:
    mage, header =  ioMRC.MRCImport( filename, returnHeader=True )
    ioMRC.MRCExport( mage, pixelsize=header['pixelsize'] )
    
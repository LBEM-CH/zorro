=====
Zorro Image Registration Software
=====

Author: Robert A. McLeod
Email: robert.mcleod@unibas.ch

Zorro is a package designed for registration of image drift for dose fractionation in electron microscopy.  It is specifically designed to suppress correlated noise.

Zorro is currently in beta.  Our current priority is to provide better ease of installation, so if you have problems with installing Zorro please do not hesitate to open an issue.

For help on installation please see the wiki page: https://github.com/C-CINA/zorro/wiki

Zorro has the following dependencies:
* `numpy`
* `SciPy`
* `pyFFTW`

And the following optional dependencies (for loading HDF5 and TIFF files respectively):

* `PyTables`
* `scikit-image`

Zorro comes packaged with a modified version of the NumExpr virtual machine called `numexprz` that has support for `complex64` data types.  

Zorro is MIT license.


Automator 
-----

The Automator for Zorro and 2dx is a GUI interface to Zorro.

It has the following additional dependencies:
* `PySide`

Automator also comes with the Skulk Manager tool which may be used as a daemon to watch a directory for new image stacks and automatically process them.

Automator is LGPL license.

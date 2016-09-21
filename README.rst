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

Feature List
-----


* Import: DM4, MRC, HDF5, stacked TIFF
* Apply gain reference to SerialEM 4/8-bit MRC stacks.
* Fourier cropping of super-resolution stacks.
* Can operate on Sun Grid Engine cluster.
* Flexible filters: dose filtering, low-pass filtering
* Stochastic hot pixel filter detects per-stack radiation damage to the detector
* CTF estimation with: CTFFIND4.1, GCTF
* Particle picking with: Gautomatch (alpha-status)
* Independent (separate even-odd frame) and non-independent FRC resolution estimators.
* Archiving with: 7z, pigz, lbzip2
* Output of diagnostic PNGs


Citations
-----

A manuscript regarding Zorro has been submitted to a peer-reviewed publication.

Zorro and Automator make use of or interface with the following 3rd party programs:

CTF estimation CTFFIND4.1: 

Rohou, A., Grigorieff, N., 2015. CTFFIND4: Fast and accurate defocus estimation from electron micrographs. Journal of Structural Biology, Recent Advances in Detector Technologies and Applications for Molecular TEM 192, 216-221. doi:10.1016/j.jsb.2015.08.008

CTF estimation from GCTF:

Zhang, K., 2016. Gctf: Real-time CTF determination and correction. Journal of Structural Biology 193, 1-12. doi:10.1016/j.jsb.2015.11.003

4/8-bit MRC from SerialEM:

Mastronarde, D.N. 2005. Automated electron microscope tomography using robust prediction of specimen movements. J. Struct. Biol. 152:36-51. 

Zorro's dose filter is ported from Unblur:

Grant, T., Grigorieff, N., 2015. Measuring the optimal exposure for single particle cryo-EM using a 2.6 Å reconstruction of rotavirus VP6. eLife Sciences e06980. doi:10.7554/eLife.06980




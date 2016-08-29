# -*- coding: utf-8 -*-
"""
Zorro Skulk manager, is a low-overhead queue manager for Zorro.

Basic intention is to manage many zorro objects, submit to queues.

TODO: may want to track meta-data over entire data set for eventually maximum-likelihood approach
Such as having the correlations for the entire set.

Created on Thu Dec 24 22:31:16 2015
@author: Robert A. McLeod
@email: robbmcleod@gmail.com OR robert.mcleod@unibas.ch
"""

import numpy as np
import zorro
import subprocess as sp
try:
    import Queue as queue
except:
    import queue
import copy
import os, os.path, glob, time, psutil, sys
import collections

try: 
    from PySide import QtCore
except:
    print( "ZorroSkulkManager failed to load PySide; no communication with Automator possible" )


    
#################### zorroState CLASS ###############################
# zorroSTATE state enumerators
# Nice thing about enumerators is Spyder catches syntax errors, whereas string comparisons aren't.
NEW = 0
CHANGING = 1
STABLE = 2
SYNCING = 3
READY = 4
PROCESSING = 5
FINISHED = 6
ARCHIVING = 7
COMPLETE = 8
STALE = 9
HOST_BUSY = 10
HOST_FREE = 11
HOST_ERROR = 12
RENAME = 13
STATE_NAMES = { NEW:u'new', CHANGING:u'changing', STABLE:u'stable', SYNCING:u'sync', READY:u'ready', 
               PROCESSING:u'proc', FINISHED:u'fini', ARCHIVING:u'archiving', COMPLETE:u'complete', 
               STALE:u'stale', HOST_BUSY:u'busy', HOST_FREE:u'free', HOST_ERROR:u'host_error', 
               RENAME:u'rename' }
REVERSE_STATE_NAMES = dict(zip(STATE_NAMES.values(), STATE_NAMES.keys()))
                
class zorroState(object):
    """
    zorroState contains a number of metadata attributes about a zorro job. Principally it controls the state 
    flow control logic, as to what the next step in the processing pipeline should be.
    
    zorroState sorts based on priority.
    
    state is an enum as above in the globals variables.  
    
    KEEP priority in 1st position, key in last, anything else can move about.
    
    Some properties are retrieved automatically, such as mtime and size
    
    It's expected that paths are normalized by skulkManager
    """
    __MIN_FILE_SIZE = 1048576 
    
    # zorroState( globbedFile, self.zorroDefault, self.paths, self.messageQueue )
    def __init__(self, key, zorroDefault, paths, messageQueue ):
        # Hrm, really I should not give this guy the manager, but rather the messageQueue and the paths object
        self.paths = paths
        self.messageQueue = messageQueue
        self.__host = None # Should be None unless a host is actively processing this zorroObj
        
        self.__priority = -1.0
        self.__state = NEW
        self.__key = key
        self.zorroObj = copy.deepcopy( zorroDefault )
        self.__prior_mtime = None
        self.__prior_size = None
        
        # Check if extension is .cfg or .zor, otherwise assume it's data
        key_ext = os.path.splitext( key )[1]
        if key_ext == ".cfg" or key_ext == ".zor":
            self.loadFromLog( self.__key )
            # We assume all the files are set appropriately.
        else:
            # Populate names based on paths
            self.setDefaultPaths()
        pass
    
    def loadFromLog( self, logName, notify=True ):
        self.zorroObj.loadConfig( logName )
        self.__state = REVERSE_STATE_NAMES[ self.zorroObj.METAstatus ]
        self.__key = self.zorroObj.files[u'config']
        self.stackPriority()

        if bool(notify):
            self.messageQueue.put( [self.__state, self] )
        
            
    def decrementState(self):
        # Decrement the state if it was in a process that is interrupted
        # TODO: should the state machine be an iterator?  Is that feasible?
        if self.__state == PROCESSING:
            self.__state = READY
        elif self.__state == SYNCING:
            self.__state = STABLE
        elif self.__state == ARCHIVING:
            self.__state = FINISHED
            
    def incrementState(self):
        # Increment the state if it was in a process that successfully finished
        if self.__state == PROCESSING:
            self.__state = FINISHED
        elif self.__state == SYNCING:
            self.__state = READY
        elif self.__state == ARCHIVING:
            self.__state = COMPLETE
            
    def setDefaultPaths(self):
        """
        Sets expected filenames for config, raw, sum, align, fig
        """
        
        self.zorroObj.files[u'original'] = self.__key
        baseName = os.path.basename( self.zorroObj.files[u'original'] )
        baseFront, baseExt = os.path.splitext( baseName )
        
        self.zorroObj.files[u'config'] = os.path.join( self.paths[u'output_dir'], baseName + u".zor" )
        
        #print( "DEFAULT PATHS: " + str(self.paths) )
        #print( "OUTPUT PATH: " + self.paths['output_dir' ] )
        #print( "CONFIG FILE: " + self.zorroObj.files['config'] )
        
        self.zorroObj.files[u'stack'] = os.path.join( self.paths[u'raw_subdir'], baseName )
        self.zorroObj.files[u'sum'] = os.path.join( self.paths[u'sum_subdir'], baseFront + u"_zorro.mrc" )
        self.zorroObj.files[u'filt'] = os.path.join( self.paths[u'sum_subdir'], baseFront + u"_zorro_filt.mrc" )
        self.zorroObj.files[u'align'] = os.path.join( self.paths[u'align_subdir'], baseFront + u"_zorro_movie.mrcs" )
        self.zorroObj.files[u'figurePath'] = self.paths[u'fig_subdir']
        
        if bool(self.paths[u'gainRef']):
            # Gainref functionality is not required, so it may be none
            self.zorroObj.files[u'gainRef'] = os.path.join( self.paths[u'output_dir'], self.paths[u'gainRef'] )
        
    def peek(self):
        # Take a peek at the log on the disk for the STATE, used for loading existing logs from disk.
        print( "TODO: zorroState.peek() " )
        logName = self.zorroObj.files[u'config']
        if os.path.isfile( logName ):
            with open( logName, 'rb' ) as rh:
                rh.readline();
                METAstatus = rh.readline().split()
                if METAstatus[0] == u'METAstatus':
                    # Reverse STATE_NAMES
                    return REVERSE_STATE_NAMES[METAstatus[2]]
                else:
                    # We could alternatively just load the config here brute force
                    raise IOError( "METAstatus is not second line in Zorro log: %s" % logName ) 
        pass

    def stackPriority(self, multiplier = 1.0 ):
        # TODO: is there any way to ask to resort the heaps?
        # self.priority = np.log( multiplier * (os.path.getmtime( self.__key ) - os.path.getmtime( self.paths['cwd']) ) )
        self.priority = multiplier * os.path.getmtime( self.__key )
        
    def topPriority(self):
        self.stackPriority( multiplier = 1E6 )
        
    def update(self):
        # Check the state and if it needs to be changed.  I could make these functions but pratically if I 
        # just lay-out the code nicely it's the same thing.
    
        ### NEW ###
        if self.__state == COMPLETE or self.__state == SYNCING or self.__state == PROCESSING or self.__state == ARCHIVING:
            # Do nothing, controlled by skulkHost
            pass
            
        elif self.__state == NEW:
            # Init and goto CHANGING
            self.__prior_mtime = os.path.getmtime( self.__key )
            self.__prior_size = os.path.getsize( self.__key )
            self.__state = CHANGING
            self.messageQueue.put( [self.__state, self] )
        
        ### CHANGING ###
        elif self.__state == CHANGING:
            # Check if file changed since last cycle
            newSize = os.path.getsize( self.__key )
            newMTime = os.path.getmtime( self.__key )
            # Compare and against MINSIZE
            if (newMTime == self.__prior_mtime and 
                        newSize == self.__prior_size and 
                        newSize >= self.__MIN_FILE_SIZE ):
                self.__state = STABLE
                self.messageQueue.put( [self.__state, self] )
                # self.update() # goto next state immediately
            else: # Else Still  changing
                pass
            
            self.__prior_mtime = os.path.getmtime( self.__key )
            self.__prior_size = os.path.getsize( self.__key )
        
        ### STABLE ###
        elif self.__state == STABLE:
            # Give a priority based on mtime, newer files are processed first so we always get an 
            # recent file when working with Automator live
            self.stackPriority()
            
     
            # Check if we need to copy or if we can just rename the file
            # This only works on Linux, on Windows we would have to check the drive letter 
            if self.paths.isLocalInput():
                # Rename raw
                oldKey = self.__key
                newKey = self.zorroObj.files[u'config']
                
                try:
                    os.rename( oldKey, self.zorroObj.files[u'stack'] )
                    
                    self.__key = newKey
                    self.__state = READY
                    
                    # Tell manager we renamed self
                    self.messageQueue.put( [RENAME, self] )
                except:
                    raise
                    
            else: # Submit self to the copy heap
                self.__state = SYNCING
                self.messageQueue.put( [SYNCING, self] )
                 # TODO: manager needs to handle renaming in this case.
    
        ### READY ###
        elif self.__state == READY:
            # Generate a log file, from here on we have to track the zorroObj persitantly on disk
        
            # setDefaultFilenames was called previously, if we want to call it again in case the user changed 
            # something we need to do some more checking to see if key is a log or a stack.
            self.zorroObj.METAstatus = STATE_NAMES[READY]
            self.zorroObj.saveConfig()
                    
            # Ask to put self into procHeap
            self.__state = PROCESSING
            self.messageQueue.put( [PROCESSING, self] )
        
        ### FINISHED (ZORRO) ###
        elif self.__state == FINISHED:
            if self.zorroObj.doCompression:
                self.__state = ARCHIVING
                self.messageQueue.put( [ARCHIVING, self] )
            else:
                self.messageQueue.put( [COMPLETE, self] )
                self.__state = COMPLETE
                self.__priority = -1.0
            pass
            
        else:
            raise ValueError( "Error, zorroState: unknown/unhandled state enumerator: %d" % self.__state )

    def __cmp__(self,other):
        # Hrm, this isn't called in python 3
        if other > self.priority: 
            return 1
        elif other < self.priority:
            return -1
        else:
            return 0
            
    def __lt__(self,other):
        # Only for Python 3?  Or does it get called in Python 2 as well?
        if other.priority > self.priority: 
            return True
        else: 
            return False
            
    # We don't want to have a method for __repr__ as this is used for finding keys in priorityList and such
    # but over-riding __str__ is ok.
    def __str__(self):
        try:
            return "zorroState: %s, state: %s, priority: %f" % (self.key, STATE_NAMES[self.state], self.priority)
        except KeyError:
            return "zorroState: %s, state: %s, priority: %f" % (self.key, str(self.state), self.priority)
        
    @property
    def host(self):
        raise SyntaxError( "zorroState: no accessing host, set to None if its expired." )
        
    @host.setter
    def host(self,host):
        if host != None:
            self.__host = host
            self.update()
        
    # Do I actually need these properties?  Are they doing anything?
    @property
    def priority(self):
        return self.__priority
        
    @priority.setter
    def priority(self,x):
        self.__priority = x
        
    @property
    def state(self):
        return self.__state
        
    # User should not be able to set state, or flow control will be a huge mess
    #@state.setter
    #def state(self,x):
    #    self.__state = x
        
    @property
    def statename(self):
        return STATE_NAMES[self.__state]
        
    @property
    def key(self):
        return self.__key
    
    @key.setter
    def key(self,x):
        self.__key = x
        
  
#################### SKULKHEAP CLASS ###############################
class skulkHeap(collections.MutableMapping):
    """
    This class manages all the files the watcher in skulkManager finds, and prioritizes them as desired 
    by the user.  It is otherwise actually a stack, not a queue, as the latest files are processed first.
    
    items are stored as zorroStates.  Keys can be either integers (reflecting priority positions) or keys 
    (which are filenames, generally either raw stacks or log files)
    """
    
    def __init__(self):            
        collections.MutableMapping.__init__(self)
        self.__priorityList = list() # format is item=[zorroState.priority, zorroState] for easy sorting
        self.__store = dict() # store is a dict with the filenames as keys
        self.__mutex = QtCore.QMutex()

    def __getitem__(self, key):
        self.__mutex.lock()
        # __keytransform__ handles the int versus str keys
        if type(key) == int:
            # It's expensive but we have no real option but to re-sort the list every time in case 
            # the priorities have changed.
            self.__priorityList.sort(reverse=True)
            try:
                return_val = self.__priorityList[key]
            except:
                self.__mutex.unlock()
                raise KeyError( "skulkHeap cannot resolve key %s of type %s" %(key, type(key) ) )
        elif type(key) == str or ( sys.version_info.major == 2 and type(key) == unicode ):
            try:
                return_val = self.__store[key]
            except:
                self.__mutex.unlock()
                raise KeyError( "skulkHeap cannot resolve key %s of type %s" %(key, type(key) ) )
        else:
            self.__mutex.unlock()
            raise KeyError( "skulkHeap cannot resolve key %s of type %s" %(key, type(key) ) ) 
           
        self.__mutex.unlock()
        return return_val

    def __len__(self): # Just need to lock things temporarily
        self.__mutex.lock()
        self.__mutex.unlock()
        return len( self.__store )
        
    def __setitem__(self, key, value):
        # value is a list [priority,zorroObj,status,]
        self.__mutex.lock()
        try:
            self.__store[key] = value
            # Make a priority list 
            self.__priorityList.append( value )
            self.__priorityList.sort(reverse=True)
        except:
            self.__mutex.unlock()
            raise
        self.__mutex.unlock()
        
    def __delitem__( self, key ):
        # pop has the mutex
        self.pop(key)
        
    def __keytransform__(self, key):
        # No mutex protection for empty function
        return key
        
    def __iter__(self):
        # If you want to iterate over keys, use 'for key in skulkHeap.keys()'
        return iter(self.__priorityList)
        
    def next( self, STATUS ):
        """
        Find the highest priority object with the given STATUS
        """
        self.__mutex.lock()
        for stateItem in self.__priorityList:
            if stateItem.state == STATUS:
                self.__mutex.unlock()
                return stateItem
            elif stateItem < 0.0: # COMPLETE jobs have negative priority
                self.__mutex.unlock()
                return None
        self.__mutex.unlock()
        return None
        
    def items(self):
        self.__mutex.lock()
        self.__mutex.unlock()
        return self.__store.items()
    
    def keys(self):
        self.__mutex.lock()
        self.__mutex.unlock()
        return self.__store.keys()
        
    def popNext( self, STATUS ):
        """
        Find the highest priority object with the given STATUS and remove it
        """
        self.__mutex.lock()
        for stateItem in self.__priorityList:
            if stateItem.state == STATUS:
                self.__priorityList.remove( stateItem)
                
                # print( "DEBUG: trying to pop key: " + str( stateItem.key ) +" from dict " + str(self.__store.keys()) )
                try: del self.__store[ stateItem.key ]
                except: print( "%s not popped from skulkHeap" % stateItem.key )
                self.__mutex.unlock()
                return stateItem
            elif stateItem < 0.0: # COMPLETE jobs have negative priority
                self.__mutex.unlock()
                return None
        self.__mutex.unlock()
        return None
        
        
    def isLocked( self ):
        state = self.__mutex.tryLock()
        if state:
            self.__mutex.unlock()
        return not state
        

    def pop( self, key ):
        """
        Key can be an integer, a string.  Returns the zorroState only.
        """
        self.__mutex.lock()
        # If we get an integer, get an index
        if type(key) == int:
            try:
                returnState = self.__priorityList.pop(key)
                
                # print( "DEBUG: trying to pop key: " + str( returnState.key ) +" from dict " + str(self.__store.keys()) )
                del self.__store[ returnState.key ]
            except:
                self.__mutex.unlock()
                raise
        elif type(key) == str or ( sys.version_info.major == 2 and type(key) == unicode ):
            try:
                returnState = self.__store.pop(key)
                
                # This is a bit tricky, can Python find the right zorroState?
                self.__priorityList.remove( returnState )
            except:
                self.__mutex.unlock()
                raise
        else:
            self.__mutex.unlock()
            raise KeyError( "skulkHeapQueue cannot resolve key %s of type %s" %(key, type(key) ) )
        self.__mutex.unlock()
        return returnState   
        
    def __str__( self ):
        str_repr = ""
        for key, zorroState in self.items():
            str_repr +=  "%s | state: %s | priority: %.3f\n" % (key, STATE_NAMES[zorroState.state], zorroState.priority)
        return str_repr

#################### SKULKPATHS CLASS ###############################
class skulkPaths(collections.MutableMapping):
    """
    This class is used for managing paths, which can get quite complicated if they are over networks. The 
    assumption is that relative paths are passed to Zorro, or saved in config files, but the user sees 
    the real paths (as it's less confusing).  So basically it accepts user paths as inputs, which are 
    converted internally into a real path and a normed path.  Normed paths are used by Zorro for writing 
    files, as it preserves the ability of the workspace to be copied, whereas the user sees the full 
    real path.
    
    For future use we plan to incorporate the ability to send files via SSH/rsync anyway, so we need a 
    class to handle this complexity. Should also be used for handling on-the-fly compression in the future.
    Possibly we could also pre-copy files to speed things up, and in general I want less file handling 
    inside Zorro.
    
    Uses os.path.normpath() to collapse extraneous dots and double-dots
    
    This class is thread-safe and can be references simultaneously from skulkManager and Automator.  
    
    TODO: some exceptions may leave the mutex locked. Use try statements to unlock the 
    mutex and then re-raise the exception.
    """
    
    def __init__(self, cwd=None):
        collections.MutableMapping.__init__(self)
        self.__mutex = QtCore.QMutex()
        # The maximum number of files to copy at once with subprocess workers
        # With rsync there's some advantage to 
        self.maxCopyStreams = 4 
        
        # Check if we can move files from one path to another without copying accross devices
        # os.stat( path1 ).st_dev == os.stat( path2 ).st_dev
        # Works on Linux only, not on Windows, so need an os check as well.
        
        # Of course parallel FTP is even faster, but IT don't like unencrypted communication
        
        self.__real = { 'input_dir':None, 'output_dir':None, 'cwd':None,
                           'raw_subdir': None, 'sum_subdir':None, 
                           'align_subdir':None, 'fig_subdir':None, 'cache_dir': None, 
                           'qsubHeader':None, 'gainRef':None  }
        self.__norm = { 'input_dir':None, 'output_dir':None, 'cwd':'.',
                           'raw_subdir': None, 'sum_subdir':None, 
                           'align_subdir':None, 'fig_subdir':None, 'cache_dir': None, 
                           'qsubHeader':None, 'gainRef':None }
        # Normalize paths relative to output_dir os.path.normpath(join( self.__NormPaths['output_dir'], path))
        
        if cwd == None:
            self.__real['cwd'] = os.path.realpath( os.curdir )
            self.__norm['cwd'] = os.path.normpath( os.curdir ) 
            
        # Default cache directories
        if os.name == 'nt':
            self.__real['cache_dir'] = "C:\\Temp\\"
            self.__norm['cache_dir'] = os.path.normpath( os.path.relpath( "C:\\Temp\\", self.__real['cwd'] ) )
        else:
            self.__real['cache_dir'] = "/scratch/"
            self.__norm['cache_dir'] = os.path.normpath( os.path.relpath(  "/scratch/", self.__real['cwd'] ) )
        pass
    
    # TODO: write and read self from a ConfigParser
    
    def __setitem__( self, key, value ):
        """
        Accepts either a real or a relative path and saves normed and real versions of the path
        """
        self.__mutex.lock()
        
        if bool( value ):
            if value == 'cwd': # Special
                # DEBUG: DO WE WANT TO CHANGE THE PROGRAM DIRECTORY?  OR JUST REDO ALL THE PATHS?
                self.__real['cwd'] = os.path.realpath( os.path.join( os.curdir, value ) )
                self.__norm['cwd'] = os.path.normpath( os.path.join( os.curdir, value ) )
                
                # Apply cwd as a mask to every path
                self.__mutex.unlock()
                for key in self.__norm:
                    if bool(self.__norm[key] ):
                        self.__real[key] = os.path.realpath( os.path.join( self.__real['cwd'], self.__norm[key]) )
                        
            else: # Anything but 'cwd'
                # Apply to norm path
                self.__norm[key] = os.path.normpath( os.path.relpath( value, self.__real['cwd']) )
                # Generate a real path based on the current working directory.
                self.__real[key] = os.path.realpath( os.path.join( self.__real['cwd'], value) )
                
        self.__mutex.unlock()
        
    def to_json( self ):
        return self.__norm
        
    def __getitem__( self, key ):
        # Return normed path.
        # self.__mutex.lock()
        # self.__mutex.unlock()
        return self.__norm[key]
        
                    
    def __iter__(self):
        return iter(self.__norm)
        
    def __keytransform__(self, key):
        return key
        
    def __len__(self):
        return len(self.__norm)

    # I would prefer to use this like a dict but it's messy without subclassing dict again
    def get_real( self, key ):
        """
        Return the real path, relative to root.  Useful for display to the user.
        """
        self.__mutex.lock()
        self.__mutex.unlock()
        return self.__real[key]
        
    def keys(self):
        return self.__norm.keys()
        
    def __delitem__( self, key ):
        self.__mutex.lock()
        try:
            self.__norm.pop(key)
            self.__real.pop(key)
        except:
            print( "Error: no key %s in skulkPaths" % key )
        self.__mutex.unlock()
        
    def __str__(self):
        self.__mutex.lock()
        retstr = "%10s # %30s # %30s\n" % ( "key", "norm", "real" ) 
        for key in self.__norm.keys():
            if bool( self.__norm[key] ) and bool( self.__real[key] ):
                retstr += "%10s # %30s # %30s\n" %( key, self.__norm[key], self.__real[key] )

        self.__mutex.unlock()
        return retstr
        
    def __contains__( self, key ):
        self.__mutex.lock()
        if key in self.__NormPaths:
            self.__mutex.unlock()
            return True
        self.__mutex.unlock()
        return False
        
    def isLocalInput(self):
        # TODO: test if checking the drive letter is sufficient on Windows systems.
        if os.name == 'nt':
            if os.path.splitdrive( self.__real['input_dir'] )[0] == os.path.splitdrive( self.__real['output_dir'] )[0]:
                return True
            else:
                return False
        if os.stat( self.__real['input_dir'] ).st_dev == os.stat( self.__real['output_dir'] ).st_dev:
            return True
        return False
        
    def validate( self ):
        # See if the directories exist.  If not, try to make them. Zorro does this as well but it's 
        # better to have user feedback at the start
    
        # Cycle mutex lock, it will 
        self.__mutex.lock()
        self.__mutex.unlock() 
        
        errorText = ""
        errorState = False
        
        # Debug: show the paths matrix
        # print( str(self) )
        try:
            if not bool(self.__real['input_dir']):
                errorState = True; errorText += "Error: Input directory field is empty.\n"
                raise ValueError
            if not os.path.isdir( self.__real['input_dir'] ):
                os.mkdir( self.__real['input_dir'] )
            if not os.access( self.__real['input_dir'], os.R_OK ):
                errorState = True; errorText += "Error: Input directory has no read permissions.\n"
            if not os.access( self.__real['input_dir'], os.W_OK ):
                errorText += "Warning: Input directory has no write permissions, cannot delete stacks from SSD.\n"
        except OSError:
            errorState = True; errorText += "Error: Input directory does not exist and could not be made.\n"
        except ValueError: 
            pass
                
        try:
            if not bool(self.__real['output_dir']):
                errorState = True; errorText += "Error: Output directory field is empty.\n"
                raise ValueError
            if not os.path.isdir( self.__real['output_dir'] ):
                os.mkdir( self.__real['output_dir'] )
            if not os.access( self.__real['output_dir'], os.R_OK ):
                errorState = True; errorText += "Error: Output directory has no read permissions.\n"
            if not os.access( self.__real['output_dir'], os.W_OK ):
                errorState = True; errorText += "Warning: Output directory has no write permissions.\n"
        except OSError:
            errorState = True; errorText += "Error: Output directory does not exist and could not be made.\n"
        except ValueError: 
            pass
        
        if errorState:
            return errorState, errorText
            
        # Continue with subdirectories
        try:
            if not bool(self.__real['raw_subdir']):
                self.__real['raw_subdir'] = os.path.join( self.__real['output_dir'], '../raw' )
                self.ui_FileLocDialog.leRawPath.setText( self.__real['raw_subdir'] ) # Doesn't fire event
                errorText += "Warning: Raw directory set to default <out>/raw.\n"
            if not os.path.isdir( self.__real['raw_subdir'] ):
                os.mkdir( self.__real['raw_subdir'] )
            if not os.access( self.__real['raw_subdir'], os.R_OK ):
                errorState = True; errorText += "Error: Raw directory has no read permissions.\n"
            if not os.access( self.__real['raw_subdir'], os.W_OK ):
                errorState = True; errorText += "Warning: Raw directory has no write permissions.\n"
        except OSError:
            errorState = True; errorText += "Error: Raw directory does not exist and could not be made.\n"

        try:
            if not bool(self.__real['sum_subdir']):
                self.__real['sum_subdir'] = os.path.join( self.__real['output_dir'], '../sum' )
                self.ui_FileLocDialog.leSumPath.setText( self.__real['sum_subdir'] ) # Doesn't fire event
                errorText += "Warning: sum directory set to default <out>/sum.\n"
            if not os.path.isdir( self.__real['sum_subdir'] ):
                os.mkdir( self.__real['sum_subdir'] )
            if not os.access( self.__real['sum_subdir'], os.R_OK ):
                errorState = True; errorText += "Error: sum directory has no read permissions.\n"
            if not os.access( self.__real['sum_subdir'], os.W_OK ):
                errorState = True; errorText += "Warning: sum directory has no write permissions.\n"
        except OSError:
            errorState = True; errorText += "Error: sum directory does not exist and could not be made.\n"

        try:
            if not bool(self.__real['align_subdir']):
                self.__real['align_subdir'] = os.path.join( self.__real['output_dir'], '../align' )
                self.ui_FileLocDialog.leAlignPath.setText( self.__real['align_subdir'] ) # Doesn't fire event
                errorText += "Warning: align directory set to default <out>/align.\n"
            if not os.path.isdir( self.__real['align_subdir'] ):
                os.mkdir( self.__real['align_subdir'] )
            if not os.access( self.__real['align_subdir'], os.R_OK ):
                errorState = True; errorText += "Error: align directory has no read permissions.\n"
            if not os.access( self.__real['align_subdir'], os.W_OK ):
                errorState = True; errorText += "Warning: align directory has no write permissions.\n"
        except OSError:
            errorState = True; errorText += "Error: align directory does not exist and could not be made.\n"
        
        try:
            if not bool(self.__real['fig_subdir']):
                self.__real['fig_subdir'] = os.path.join( self.__real['output_dir'], '../figure' )
                self.ui_FileLocDialog.leFiguresPath.setText( self.__real['fig_subdir'] ) # Doesn't fire event
                errorText += "Warning: figure directory set to default <out>/figure.\n"
            if not os.path.isdir( self.__real['fig_subdir'] ):
                os.mkdir( self.__real['fig_subdir'] )
            if not os.access( self.__real['fig_subdir'], os.R_OK ):
                errorState = True; errorText += "Error: figure directory has no read permissions.\n"
            if not os.access( self.__real['fig_subdir'], os.W_OK ):
                errorState = True; errorText += "Warning: figure directory has no write permissions.\n"
        except OSError:
            errorState = True; errorText += "Error: figure directory does not exist and could not be made.\n"
        
        # Check for path uniqueness
        if self.__real['input_dir'] == self.__real['output_dir']:
            errorState = True; errorText += "Error: Input and output directory may not be the same.\n"
        if self.__real['input_dir'] == self.__real['raw_subdir']:
            errorState = True; errorText += "Error: Input and raw directory may not be the same.\n"
        if self.__real['input_dir'] == self.__real['sum_subdir']:
            errorState = True; errorText += "Error: Input and sum directory may not be the same.\n"
        if self.__real['input_dir'] == self.__real['align_subdir']:
            errorState = True; errorText += "Error: Input and align directory may not be the same.\n"    
        
        return errorState, errorText


#################### SKULKHOST CLASS ###############################
class skulkHost(object):
    """
    A skulkHost manages the individual Zorro jobs dispatched by Automator.  On a local machine, one skulkHost 
    is created for each process specified to use.
    
    On a cluster a skulkHost is one job, so there can be more hosts than available nodes if desired.
    """
    
    def __init__(self, hostname, workerFunction, messageQueue, 
                 n_threads = None, cachepath = None, qsubHeader=None ):
        """
        def __init__(self, hostname, workerFunction, messageQueue, 
                 n_threads = None, cachepath = None, qsubHeader=None )
                 
        hostName is any string that ID's the host uniquely
        
        workerFunction is one of ['local','qsub', 'rsync', 'archive'], reflecting what the host should do.
        
        messageQueue is the message queue from the skulkManager
        
        n_threads is the number of p-threads to use for the job.
        
        qsubHeader is a text file that contains everything but the qsub line for a .bash script.
        """
        self.hostName = hostname
        self.messageQueue = messageQueue
        self.zorroState = None
        self.submitName = None
        
        # CANNOT get an instantiated object here so we cannot pass in bound function handles directly, so 
        # we have to use strings instead.
        if workerFunction == 'local':
            self.workerFunction = self.submitLocalJob
        elif workerFunction == 'dummy':
            self.workerFunction = self.submitDummyJob
        elif workerFunction == 'qsub':
            self.workerFunction = self.submitQsubJob
        elif workerFunction == 'rsync':
            self.workerFunction = self.submitRsyncJob
        elif workerFunction == 'archive':
            self.workerFunction = self.submitArchiveJob
        else:
            self.workerFunction = workerFunction
            
        self.subprocess = None
        self.n_threads = n_threads
        self.cachePath = cachepath
        self.qsubHeaderFile = qsubHeader
        
#    def kill(proc_pid):
#    process = psutil.Process(proc_pid)
#    for proc in process.get_children(recursive=True):
#        proc.kill()
#    process.kill()
    
    def kill(self):

        try:
            if bool( self.subprocess ):
                
                # Kill again with psutil
                try:
                    print( "Trying to kill subprocess pid %d" % self.subprocess.pid )
                    os_process = psutil.Process( self.subprocess.pid )
                    for child_process in os_process.children(recursive=True):
                        child_process.kill()
                    os_process.kill()
                except Exception as e:
                    print( "skulkHost.kill() psutil varient received exception: " + str(e) )
                    
                # Well this sort of works, now we have defunct zombie processes but they 
                # stop running.  Force garbage collection with del
                # subprocess.kill() doesn't always work, so use psutil to do it
                self.subprocess.communicate() # Get any remaining output
                self.subprocess.kill()
                del self.subprocess
        except Exception as e:
            print( "skulkHost.kill raised exception: " + str(e) )
            

        if self.workerFunction == self.submitQsubJob:
            print( "TODO: call qdel on the submitted job. Need to figure out how to id the job number during submission" )
            
        # Open the host for new jobs (hopefully zombie processes cannot pile up)
        self.subprocess = None       
         
    def poll(self):
        # Would it be better to have some sort of event driven implementation?
        if self.subprocess == None or not bool( self.zorroState ):
            return HOST_FREE
            
        status = self.subprocess.poll()
        if status == None:
            return HOST_BUSY
        elif status == 0:

            # Send a message that we finished a process and that the state should be incremented.
            if self.zorroState.state == PROCESSING:
                self.messageQueue.put( [FINISHED, self.zorroState] )
            elif self.zorroState.state == SYNCING:
                self.messageQueue.put( [RENAME, self.zorroState] )
            elif self.zorroState.state == ARCHIVING:
                self.messageQueue.put( [COMPLETE, self.zorroState] )
            else:
                raise ValueError( "skulkHost: unknown host type: " + self.hostName )
                
            # Release the subprocess and state object
            self.subprocess = None
            self.zorroState = None
            # Remove the submission script if present
            try: os.remove( self.submitName )
            except: pass
            return HOST_FREE
        else: # Error state, kill it with fire
            self.kill()
            return 'error' + str(status)


    def submitRsyncJob( self, stateObj ):
        # zorroObj.files['raw'] contains the target location
        # TODO: add compression as an option to rsync, generally slows the transfer down unless you have a 
        # bunch running simulanteously
        self.zorroState = stateObj
        
        compress_flag = "-v"
        remove_flag = "--remove-source-files"
        source = stateObj.key # or stateObj.zorroObj.files['original']
        target = stateObj.zorroObj.files['stack']
        
        rsync_exec = "rsync " + compress_flag + " " + remove_flag + " " + source + " " + target
        print( "RSYNC command: " + rsync_exec )
        
        self.subprocess = sp.Popen( rsync_exec, shell=True )
        self.messageQueue.put( [HOST_BUSY, self.zorroState] )
        
    def submitArchiveJob( self, zorroState ):
        print( "TODO: move files and compress them using an external utility, or blosc with zlib?" )
        
    def submitLocalJob( self, stateObj ):
        # Local job, no queue
        # Check/copy that stackName exists in the right place
        self.zorroState = stateObj

        self.zorroState.zorroObj.n_threads = self.n_threads
        if self.cachePath != None: # Force use of host's cachePath if it's present...
            self.zorroState.zorroObj.cachePath = self.cachePath
            

        self.zorroState.zorroObj.METAstatus = STATE_NAMES[PROCESSING]

        # Execute zorro
        self.zorroState.zorroObj.saveConfig()

        # We could just call 'zorro' script but this is perhaps safer.
        # zorro_script = os.path.join( os.path.split( zorro.__file__ )[0], "__main__.py" )
        #commandStr = "python " + zorro_script + " -c " + self.zorroState.zorroObj.files['config']

        commandStr = "zorro -c " + self.zorroState.zorroObj.files['config']
        print( "Subprocess local exec: " + commandStr )
        # Seems that shell=True is more or less required.
        self.subprocess = sp.Popen( commandStr, shell=True )
        self.messageQueue.put( [HOST_BUSY, self.zorroState] )
    
    
    def submitQsubJob( self, stateObj ):
        # Local job, no queue
        # Check/copy that stackName exists in the right place
        self.zorroState = stateObj
        
        # Load header string
        with open( self.qsubHeaderFile, 'r' ) as hf:
            submitHeader = hf.read()
        
        self.zorroState.zorroObj.n_threads = self.n_threads
        if self.cachePath != None: # Force use of host's cachePath if it's present...
            self.zorroState.zorroObj.cachePath = self.cachePath

        self.zorroState.zorroObj.METAstatus = STATE_NAMES[PROCESSING]
        
        # Force to use 'Agg' with qsub, as often we are using Qt4Agg in Automator
        self.zorroState.zorroObj.plotDict['backend'] = 'Agg'
        
        # Save configuration file for the load
        self.zorroState.zorroObj.saveConfig()
        
        # Setup the qsub .bash script
        # zorro_script = os.path.join( os.path.dirname( zorro.__file__ ), "__main__.py" )
        
        # Hrm, ok, so the cfgFront is relative to where?
        
        cfgFront = os.path.splitext(self.zorroState.zorroObj.files['config'] )[0]
        cfgBase = os.path.splitext( os.path.basename(self.zorroState.zorroObj.files['config'] ) )[0]
        self.submitName = 'z' + cfgBase + ".sh"
        
        # Plotting is crashing on the cluster.
        # submitCommand = "python %s -c %s" % (zorro_script, self.zorroState.zorroObj.files['config']  )
        submitCommand = "zorro -c %s" % (self.zorroState.zorroObj.files['config']  )
        
        # Actually, it's easier if I just do all the subsitutions myself, rather than farting around with 
        # different operating systems and environment variables
        submitHeader = submitHeader.replace( "$JOB_NAME", self.submitName )
        submitHeader = submitHeader.replace( "$OUT_NAME", cfgFront + ".zout" )
        # Use the same output and error names, otherwise often we don't see the error.
        submitHeader = submitHeader.replace( "$ERR_NAME", cfgFront + ".zout" ) 
        submitHeader = submitHeader.replace( "$N_THREADS", str(self.n_threads) )
        
        # Write the bash script to file
        with open( self.submitName, 'w' ) as sub:
            sub.write( submitHeader )
            sub.write( submitCommand + "\n" )
        print( "Submitting job to grid engine : %s"%self.submitName )
        
        # We use -sync y(es) here to have a blocking process, so that we can check the status of the job.  
        commandStr = "qsub -sync y " + self.submitName
        # self.subprocess = sp.Popen( commandStr, shell=True, env=OSEnviron )
        self.subprocess = sp.Popen( commandStr, shell=True, stdout=sp.PIPE )
        
        # Can we get the job ID from the "Your job 16680987 ("z2016-08-04_09_04_07.dm4.sh") has been submitted " line?
        # time.sleep( 1.0 )
        # out, err = self.subprocess.communicate()
        # print( "TODO: parse output, error string: %s, %s" %(out,err) )
        
        self.messageQueue.put( [HOST_BUSY, self.zorroState] )
        
        # Submission script is cleaned by the successful return of the job only.  The alternative approach 
        # would be to submit a subprocess bash script that waits 60 s and then deletes it?
    
    def submitDummyJob( self, zorroState ):
        self.zorroState = zorroState
        # Cheap job for testing purposes
        time.sleep( 5 )
        pass
    
    def submitJob_viaSSH( self, zorroState ):
        # Remote job, no queue#################### SUBPROCESS FUNCTIONS ###############################
        print( "Submission of jobs via SSH is not implemented yet." )
        pass

    def qsubJob_visSSH( self, zorroState ):
        # Remote queue, submit via SSH needs some sort of distributed file system?  Or can we use rsync (if 
        # this is to be used as a daemon in this case)
        print( "Submission of jobs via SSH is not implemented yet." )
        pass

    pass


        

#################### SKULKMANAGER CLASS ###############################
class skulkManager(QtCore.QThread):
    
    
    # Signals must be statically declared?
    try:
        automatorSignal = QtCore.Signal( str, str )
    except:
        automatorSignal = None
    
    def __init__(self, inputQueue=None ):
        QtCore.QThread.__init__(self)
        
        self.verbose = 3
        
        # Processible file extensions
        self.globPattern = ['*.dm4', '*.dm3', '*.tif', '*.mrc', '*.mrcs', '*.hdf5', '*.h5', 
                            '*.bz2', '*.gz', '*.7z' ]
                
        # Default object for copying the parameters (also accessed by Automator)    
        self.zorroDefault = zorro.ImageRegistrator()
        
        # Mutex-protected queues of the image stacks to be worked on
        self.__globalHeap = skulkHeap()
        self.__syncHeap = skulkHeap()
        self.__procHeap = skulkHeap()
        self.__archiveHeap = skulkHeap()
        self.completedCount = 0
        
        # Messages from the hosts to the skulkManager (i.e. I've finished, or, I've fallen and I can't get up again)
        # This is unidirectional from the hosts to the skulk manager
        self.messageQueue = queue.Queue(maxsize=0) 
        
        # Servers that will accept jobs, is a dict of skulkHosts
        self.procHosts = {}
        self.syncHosts = {}
        self.archiveHosts = {}
        
        # Path manager, for keeping real and normed paths manageable.  
        self.paths = skulkPaths()

        # TODO: Do we want Paramiko for running remote jobs via SSH?  Or restrict ourselves to qsub locally?
        # https://github.com/paramiko/paramiko
        # It's not in Anaconda by default, but it's in both conda and pip
        
        self.__STOP_POLLING = False
        # bytes, should be bigger than Digital Micrograph header size, but smaller than any conceivable image
        
        self.sleepTime = 1.0
        pass
    
    def __getitem__( self, key ):
        return self.__globalHeap[key]
                 
    def __len__(self):
        return len(self.__globalHeap)
            
    def keys( self ):
        return self.__globalHeap.keys()
        
            
    def run(self):
        self.completedCount = 0
        loopCounter = 0
        while not self.__STOP_POLLING:
            t0 = time.time()
            # Check queues if they have returned a result
            while self.messageQueue.qsize() > 0:
                # We have message(s)
                messageList = self.messageQueue.get()
                
                message, zorroState = messageList
                
                try:
                    print( "DEBUG skulkManager.run(): Received message: " + STATE_NAMES[message] + " for " + str(zorroState) )
                except KeyError:
                    print( "DEBUG skulkManager.run(): Received message: " + message + " for " + str(zorroState) )
                    
                
                messageColor = 'firebrick' # Error is default state
                # zorroState.update MESSAGES
                if message == NEW or message == CHANGING:
                    # Just update the status color in Automator
                    messageColor = 'orange'
                    pass
                elif message == RENAME:
                    # The assumption here is that the key has changed, so we have to pop the oldKey 
                    # and add the new one from the globalHeap
                    oldKey = zorroState.zorroObj.files['original']
                    print( "Rename %s to %s " %( oldKey, zorroState.key ) )
                    try: del self.__archiveHeap[ oldKey ]
                    except: pass 
                    try: del self.__syncHeap[ oldKey ]
                    except: pass 
                    try: del self.__globalHeap[ oldKey ]
                    except: pass
                    
                    # Notify Automator of the deletion
                    self.automatorUpdate( oldKey, 'rename' )
                    # And now make sure the key is current
                    zorroState.key = zorroState.zorroObj.files['config']
                    
                    # We assume after a rename the zorroState isn't in the heap
                    self.__globalHeap[zorroState.key] = zorroState   
                    
                    zorroState.incrementState() # WARNING: not always the case!
                    message = READY # WARNING: not always the case!
                    
                elif message == PROCESSING: # zorroState wants to be in procHeap
                    messageColor = 'forestgreen'
                    self.__procHeap[zorroState.key] = zorroState
                    
                elif message == SYNCING:
                    messageColor = 'goldenrod'
                    self.__syncHeap[zorroState.key] = zorroState
                    
                elif message == ARCHIVING:
                    messageColor = 'indigo'
                    self.__archiveHeap[zorroState.key] = zorroState
                    
                # skulkHost MESSAGES
                elif message == HOST_BUSY:
                    # Zorro is processing this stack, green status, just update the status color in Automator
                    # TODO: different colors for different states?
                    if zorroState.state == SYNCING:
                        messageColor = 'deeppink'
                    elif zorroState.state == PROCESSING:
                        messageColor = 'steelblue'
                    elif zorroState.state == ARCHIVING:
                        messageColor = 'saddlebrown'
                    pass
                
                elif message == FINISHED:
                    try: del self.__procHeap[ zorroState.key ]
                    except: print( "WARNING skulkManager.run() missing zorroState on procHeap: " + zorroState.key )
                    
                    # Not necessary to increment zorroState as it should be 'fini' from the log file
                    # zorroState.incrementState()
                    # Load from disk
                    zorroState.loadFromLog( zorroState.zorroObj.files['config'], notify=False )
                    if zorroState.state == FINISHED:
                        messageColor = 'indigo'
                    
                elif message == COMPLETE:
                    self.completedCount += 1
                    try: del self.__archiveHeap[ zorroState.key ]
                    except: print( "WARNING skulkManager.run() missing zorroState on archiveHeap: " + zorroState.key )
                    
                    zorroState.incrementState()
                    messageColor = 'dimgray'
                else:
                    print( "TODO: skulkManager::mainPollingLoop: Process this message : " + STATE_NAMES[message] )
                 
                # Send updates to the Automator GUI
                self.automatorUpdate( zorroState.key, messageColor )
            
            # DEBUG OUTPUT ALL ITEMS IN GLOBAL HEAP
            # print( "GLOBAL HEAP: \n" + str(self.__globalHeap) )            
            
            ### Poll the skulkHosts to see if they are free
            statusMessage = "%s "%loopCounter
            
            
            freeProcs = []
            statusMessage += " | PROC "
            for J, hostKey in enumerate( self.procHosts ):
                pollValue = self.procHosts[hostKey].poll()
                if pollValue == HOST_FREE:
                    freeProcs.append( self.procHosts[hostKey] )
                elif pollValue == HOST_BUSY:
                    pass
                elif pollValue.startswith( 'error' ):
                    # TODO: do we want an actual exception?
                    print( "ERROR in subprocess on host %s: %s (lookup code)" % (hostKey, pollValue ) )
                    # Free the host
                    self.procHosts[hostKey].kill()
                   
                if pollValue in STATE_NAMES:
                    statusMessage += ": %s "% STATE_NAMES[pollValue]
                else:
                    statusMessage += ": ===%s=== "% pollValue
                
            freeSyncs = []
            statusMessage += " | SYNC " 
            for J, hostKey in enumerate( self.syncHosts ):
                pollValue = self.syncHosts[hostKey].poll()
                if pollValue == HOST_FREE:
                    freeSyncs.append( self.syncHosts[hostKey] )
                elif pollValue == HOST_BUSY:
                    pass
                elif pollValue.startswith( 'error' ):
                    # TODO: do we want an actual exception?
                    print( "ERROR in sync on host %s: error %s" % (hostKey, pollValue ) )
                    # Free the host
                    self.syncHosts[hostKey].kill()
                   
                if pollValue in STATE_NAMES:
                    statusMessage += ": %s "% STATE_NAMES[pollValue]
                else:
                    statusMessage += ": ===%s=== "% pollValue
             
            print( str(statusMessage) )
            
            while len( freeProcs ) > 0:
                # print( "In freeProcs loop %d hosts, %d zorros" % (len(freeProcs), len(self.__procHeap) ) )
            
                nextJob = self.__procHeap.popNext( PROCESSING )
                if not bool(nextJob): # == None
                    break
                # Submit a job, the workerFunction or the zorroState should change the state to PROCESSING
                freeProcs.pop().workerFunction( nextJob )
                # Update priority
                nextJob.priority *= 0.1
                print( "TODO: priority handling" )
                
            while len( freeSyncs ) > 0:
                # print( "In freeSyncs loop (TODO: archiving): %d hosts, %d zorros" % (len(freeSyncs), len(self.__syncHeap) ) )

                nextSync = self.__syncHeap.popNext( SYNCING ) # Could be SYNCING or ARCHIVING?  Think about it
                if not bool(nextSync):
                    break
                # TODO: pick worker function based on the STATUS?
                freeSyncs.pop().workerFunction( nextSync )
            
    
            # Have all the zorroState objects check their state
            # This could become slow if we have thousands of objects loaded.  Perhaps, since it's ordered,
            # we should only check so many?  Or check specifically the special queues?
            # Probably we should pass in freeHosts?
            for key, zorroState in self.__globalHeap.items():
                zorroState.update()
                
            # Now check the input directory for new files to add to the globalHeap
            self.watch()
            
            # Sleep the polling function
            loopCounter += self.sleepTime
            t1 = time.time()
            if self.verbose >= 4:
                print( "Poll time (s): %.6f" %(t1-t0) )
            self.sleep( self.sleepTime )
            
    # End skulkManager.run()

    def watch(self):
        """
        Poll the input directory for new stack files, and generate zorroStates for them if new image stacks 
        appear
        """
        newGlobList = []
        for globPat in self.globPattern:
            # I wonder if normed or real paths are better here.
            newGlobList += glob.glob( os.path.join( self.paths.get_real('input_dir'), globPat ) )
            
        # And for every existing key name, check to see if we have a record of it
        for globbedFile in newGlobList:
            if not globbedFile in self.__globalHeap.keys():
                
                # Add it to the __globalHeap, __syncHeap should take care of itself.
                self.__globalHeap[globbedFile] = zorroState( globbedFile, self.zorroDefault, self.paths, self.messageQueue )

    # end skulkManager.watch()
        
        
    def initHosts( self, cluster_type='local', n_processes=1, n_threads=16, n_syncs=4, qsubHeader=None ):
        """
        Starts all the hosts, for a given cluster_type.
        
        For local the n_processes is the number of process.
        
        For qsub the n_processes is the maximum number of simultaneous jobs in given 'qconf -sq $QUEUENAME'
        """
        self.inspectOutputDirectory()
        self.__STOP_POLLING = False
        
        del self.procHosts # Hard reset, user/Automator should call kill first to avoid dangling processes.
        self.procHosts = {} 
        del self.syncHosts 
        self.syncHosts = {} 
        del self.archiveHosts 
        self.archiveHosts = {} 
        
        print( "DEBUG: initHosts.qsubHeader = %s" % qsubHeader )
        print( "Starting %d processing hosts" % n_processes )
        for J in np.arange( n_processes ):
            self.procHosts['host%d'%J] = skulkHost( 'host%d'%J, cluster_type, self.messageQueue, 
                    n_threads=n_threads, qsubHeader=qsubHeader )
        
        print( "Starting %d syncing hosts" % n_syncs )            
        for J in np.arange( n_syncs ):
            self.syncHosts['sync%d'%J] = skulkHost( 'sync%d'%J, 'rsync', self.messageQueue )
        pass # end iniHosts()
        

    def automatorUpdate( self, key, statusColor ):
        """
        Send a signal to the parent Automator, if it exists
        """
        if self.automatorSignal != None:
            self.automatorSignal.emit( key, statusColor )
        
        
    def remove( self, key ):
        """
        WARNING: this function deletes all files associated with a key, not just the log.  It cannot be undone.
        """
        try:
            deleteZorro = self.__globalHeap[ key ]
            if deleteZorro != None:
                del self.__globalHeap[ key ]
                try: del self.__archiveHeap[key]
                except: pass
                try: del self.__syncHeap[key]
                except: pass
                try: del self.__procHeap[key]
                except: pass
            
            print( "removed from Heap: " + key + " as " + str(deleteZorro) )
        except Exception as e:
            print( "Error skulkManager.remove: " + e.str )
            raise 
      
        # Check running jobs for the key and kill them if necessary
        for host in list(self.procHosts.values()) + list(self.syncHosts.values()) + list(self.archiveHosts.values()):
            if host.zorroState != None and host.zorroState == key:
                host.kill()
                
        # At this point we can remove all the associated files
        if key.endswith( '.zor' ) or (deleteZorro != None and 'config' in deleteZorro.files):
            try:
                deleteZorro = zorro.ImageRegistrator()
                deleteZorro.loadConfig( key )
                # DELETE ALL FILES
                currDir = os.path.realpath('.')
                os.chdir( os.path.split(key)[0] ) # Change directory so relative paths are valid
                for filename in deleteZorro.files.values():
                    try: os.remove( filename )
                    except: pass
                os.chdir( currDir )
            except: # Failed to delete
                pass
        else: # We can't find a log with a list of files so we can only remove the key
            try: os.remove( key )
            except: pass
        # politely recommend garbage collection
        del deleteZorro
        
        
    def reprocess( self, key ):
        """
        Reprocesses a selected file with current conditions, and also gives the file extra-high priority so 
        it goes to top of the stack.
        """
        # Remove process
        try: del self.__archiveHeap[key]
        except: pass
        try: del self.__syncHeap[key]
        except: pass
        try: del self.__procHeap[key]
        except: pass
    
        if key in self.__globalHeap.keys():
            # Check running jobs for the key and kill them if necessary
            for host in list(self.procHosts.values()) + list(self.syncHosts.values()) + list(self.archiveHosts.values()):
                if host.zorroState != None and host.zorroState == key:
                    host.kill()
                    
            zorroState = self.__globalHeap[key]
            # Clean up any existing log file, a new one will be generated soon
            if os.path.isfile( zorroState.zorroObj.files['config'] ):
                # Just delete the log, everything else can be over-written
                try:
                    os.remove( zorroState.zorroObj.files['config'] )
                except OSError:
                    pass # Do nothing, it may have been deleted by the host.kill() above.
                
            # Go back to READY
            zorroState._zorroState__state = READY
            self.messageQueue.put( [READY, zorroState] )
            # Upgrade priority
            zorroState.topPriority()


    def kill(self):
        self.__STOP_POLLING = True
        for hostObj in self.procHosts.values():
            hostObj.kill()
        for hostObj in self.syncHosts.values():
            hostObj.kill()
        for hostObj in self.archiveHosts.values():
            hostObj.kill()
            
        # Go through the skulkQueue and change any 'proc' statuses to 'stable'
        if not self.__globalHeap.isLocked():
            # Forcibly unlock the mutex for the global heap
            self.__globalHeap._skulkHeap__mutex.unlock()
        
        # Get rid of the secondary heaps
        del self.__archiveHeap
        del self.__syncHeap
        del self.__procHeap
        self.__archiveHeap = skulkHeap()
        self.__syncHeap = skulkHeap()
        self.__procHeap = skulkHeap()
        
        # Reset any heap-states
        for key, stateObj in self.__globalHeap.items():
            stateObj.decrementState()

        

    def inspectOutputDirectory(self):
        """
        Check the output directory for logs.  In general this is only called when a start is issued.
        """
        if self.paths['output_dir'] == None:
            return
            
        logList = glob.glob( os.path.join( self.paths['output_dir'], "*.zor" ) )
        for log in logList:
            if log.endswith( "ctffind3.zor" ): # Avoid trying CTF logs if they happen to be in the same place.
                continue
            
            newState = zorroState( log, self.zorroDefault, self.paths, self.messageQueue )
            self.__globalHeap[ newState.key ] = newState
            self.automatorUpdate( newState.key, newState.state )
            


if __name__ == "__main__":
    pass
    

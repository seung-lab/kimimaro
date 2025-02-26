from collections import defaultdict
import errno
import mmap
import os
import sys
import time

import multiprocessing as mp

import numpy as np

from osteoid import Bbox, Vec

from .utility import mkdir

SHM_DIRECTORY = '/dev/shm/'
EMULATED_SHM_DIRECTORY = '/tmp/kimimaro-shm'

EMULATE_SHM = not os.path.isdir(SHM_DIRECTORY)
PLATFORM_SHM_DIRECTORY = SHM_DIRECTORY if not EMULATE_SHM else EMULATED_SHM_DIRECTORY

class SharedMemoryReadError(Exception):
  pass

class SharedMemoryAllocationError(Exception):
  pass

def ndarray(shape, dtype, location, order='F', readonly=False, lock=None, **kwargs):
  """
  Create a shared memory numpy array. 
  Lock is only necessary while doing multiprocessing on 
  platforms without /dev/shm type  shared memory as 
  filesystem emulation will be used instead.

  Allocating the shared array requires cleanup on your part.
  A shared memory file will be located at sharedmemory.PLATFORM_SHM_DIRECTORY + location
  and must be unlinked when you're done. It will outlive the program.

  You should also call .close() on the mmap file handle when done. However,
  this is less of a problem because the operating system will close the
  file handle on process termination.

  Parameters:
  shape: same as numpy.ndarray
  dtype: same as numpy.ndarray
  location: the shared memory filename 
  lock: (optional) multiprocessing.Lock

  Returns: (mmap filehandle, shared ndarray)
  """
  if EMULATE_SHM:
    return ndarray_fs(
      shape, dtype, location, lock, 
      readonly, order, emulate_shm=True, **kwargs
    )
  return ndarray_shm(shape, dtype, location, readonly, order, **kwargs)

def ndarray_fs(
    shape, dtype, location, lock, 
    readonly=False, order='F', emulate_shm=False,
    **kwargs
  ):
  """Emulate shared memory using the filesystem."""
  dbytes = np.dtype(dtype).itemsize
  nbytes = Vec(*shape).rectVolume() * dbytes

  if emulate_shm:
    directory = mkdir(EMULATED_SHM_DIRECTORY)
    filename = os.path.join(directory, location)
  else:
    filename = location

  if lock:
    lock.acquire()

  try:
    allocate_shm_file(filename, nbytes, dbytes, readonly)
  finally:
    if lock:
      lock.release()

  with open(filename, 'r+b') as f:
    array_like = mmap.mmap(f.fileno(), 0) # map entire file
  
  renderbuffer = np.ndarray(buffer=array_like, dtype=dtype, shape=shape, order=order, **kwargs)
  renderbuffer.setflags(write=(not readonly))
  return array_like, renderbuffer

def allocate_shm_file(filename, nbytes, dbytes, readonly):
  try:
    size = os.path.getsize(filename)
    exists = True
  except FileNotFoundError:
    size = 0
    exists = False

  if readonly and not exists:
    raise SharedMemoryReadError(filename + " has not been allocated. Requested " + str(nbytes) + " bytes.")
  elif readonly and size != nbytes:
    raise SharedMemoryReadError("{} exists, but the allocation size ({} bytes) does not match the request ({} bytes).".format(
      filename, size, nbytes
    ))

  if exists: 
    if size > nbytes:
      with open(filename, 'wb') as f:
        os.ftruncate(f.fileno(), nbytes)
    elif size < nbytes:
      # too small? just remake it below
      os.unlink(filename) 

  exists = os.path.exists(filename)

  if not exists:
    # Previously we were writing out real files full of zeros, 
    # but a) that takes forever and b) modern OSes support sparse
    # files (i.e. gigabytes of zeros that take up only a few real bytes).
    #
    # The following should take advantage of this functionality and be faster.
    # It should work on Python 2.7 Unix, and Python 3.5+ on Unix and Windows.
    #
    # References:
    #   https://stackoverflow.com/questions/8816059/create-file-of-particular-size-in-python
    #   https://docs.python.org/3/library/os.html#os.ftruncate
    #   https://docs.python.org/2/library/os.html#os.ftruncate
    #
    with open(filename, 'wb') as f:
      os.ftruncate(f.fileno(), nbytes)

def ndarray_shm(shape, dtype, location, readonly=False, order='F', **kwargs):
  """Create a shared memory numpy array. Requires /dev/shm to exist."""
  import posix_ipc
  from posix_ipc import O_CREAT
  import psutil

  nbytes = Vec(*shape).rectVolume() * np.dtype(dtype).itemsize
  available = psutil.virtual_memory().available

  preexisting = 0
  # This might only work on Ubuntu
  shmloc = os.path.join(SHM_DIRECTORY, location)
  if os.path.exists(shmloc):
    preexisting = os.path.getsize(shmloc)
  elif readonly:
    raise SharedMemoryReadError(shmloc + " has not been allocated. Requested " + str(nbytes) + " bytes.")

  if readonly and preexisting != nbytes:
    raise SharedMemoryReadError("{} exists, but the allocation size ({} bytes) does not match the request ({} bytes).".format(
      shmloc, preexisting, nbytes
    ))

  if (nbytes - preexisting) > available:
    overallocated = nbytes - preexisting - available
    overpercent = (100 * overallocated / (preexisting + available))
    raise SharedMemoryAllocationError("""
      Requested more memory than is available. 

      Shared Memory Location:  {}

      Shape:                   {}
      Requested Bytes:         {} 
      
      Available Bytes:         {} 
      Preexisting Bytes*:      {} 

      Overallocated Bytes*:    {} (+{:.2f}%)

      * Preexisting is only correct on linux systems that support /dev/shm/""" \
        .format(location, shape, nbytes, available, preexisting, overallocated, overpercent))

  # This might seem like we're being "extra safe" but consider
  # a threading condition where the condition of the shared memory
  # was adjusted between the check above and now. Better to make sure
  # that we don't accidently change anything if readonly is set.
  flags = 0 if readonly else O_CREAT 
  size = 0 if readonly else int(nbytes) 

  try:
    shared = posix_ipc.SharedMemory(location, flags=flags, size=size)
    array_like = mmap.mmap(shared.fd, shared.size)
    os.close(shared.fd)
    renderbuffer = np.ndarray(buffer=array_like, dtype=dtype, shape=shape, order=order, **kwargs)
  except OSError as err:
    if err.errno == errno.ENOMEM: # Out of Memory
      posix_ipc.unlink_shared_memory(location)      
    raise

  renderbuffer.setflags(write=(not readonly))
  return array_like, renderbuffer

def unlink(location):
  if EMULATE_SHM:
    return unlink_fs(location)
  return unlink_shm(location)

def unlink_shm(location):
  import posix_ipc
  try:
    posix_ipc.unlink_shared_memory(location)
  except posix_ipc.ExistentialError:
    return False
  return True

def unlink_fs(location):
  directory = mkdir(EMULATED_SHM_DIRECTORY)
  try:
    filename = os.path.join(directory, location)
    os.unlink(filename)
    return True
  except OSError:
    return False

import numpy as np
import os
import gzip

def normalize_file_ext(filename):
  filename, ext = os.path.splitext(filename)

  two_pass = ('.ckl', '.cpso')

  if ext in two_pass:
    return ext

  while True:
    filename, ext2 = os.path.splitext(filename)
    if ext2 in two_pass:
      return ext2
    elif ext2 == '':
      return ext
    ext = ext2

def load(filename):
  ext = normalize_file_ext(filename)

  if ext == ".ckl":
    import crackle
    image = crackle.load(filename)
  elif ext == ".npy":
    if filename.endswith(".gz"):
      with gzip.GzipFile(filename, "rb") as f:
        image = np.load(f)
    else:
      image = np.load(filename)
  elif ext == ".nrrd":
    import nrrd
    image, header = nrrd.read(filename)
    if image.shape[0] == 3 and image.ndim == 3:
      image = image[...,np.newaxis]
      image = np.transpose(image, axes=[1,2,3,0])
    return image
  elif ext == ".nii":
    import nibabel as nib
    image = nib.load(filename)
    image = np.array(image.dataobj)
  elif ext in (".tif", ".tiff"):
    import tifffile
    image = tifffile.imread(srcpath)
  else:
    raise ValueError("Data type not supported: " + ext)

  return np.asfortranarray(image)

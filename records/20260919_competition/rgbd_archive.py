"""Fast lossless float32 depth archives compatible with numpy.load."""
import zipfile
import numpy as np


def write_depth_archive(path,depth):
    depth=np.asarray(depth)
    if depth.ndim!=2 or depth.dtype!=np.float32:
        raise ValueError('depth archive requires a two-dimensional float32 image')
    # Level1 preserves every depth bit while avoiding default deflate6 CPU cost.
    with zipfile.ZipFile(path,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as archive:
        with archive.open('depth.npy','w',force_zip64=True) as member:
            np.lib.format.write_array(member,depth,allow_pickle=False)

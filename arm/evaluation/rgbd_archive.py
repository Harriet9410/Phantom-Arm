"""Lossless float32 archives with bounded XOR chains; no control input changes."""
import hashlib
from pathlib import Path
import zipfile
import numpy as np


def write_arrays(path,arrays):
    with zipfile.ZipFile(path,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as archive:
        for name,array in arrays.items():
            with archive.open(name+'.npy','w',force_zip64=True) as member:
                np.lib.format.write_array(member,np.asarray(array),allow_pickle=False)


def write_depth_archive(path,depth):
    depth=np.asarray(depth)
    if depth.ndim!=2 or depth.dtype!=np.float32:raise ValueError('two-dimensional float32 depth required')
    write_arrays(path,{'depth':depth})


class DepthSeriesWriter:
    def __init__(self,keyframe_interval=32):
        if not 1<=keyframe_interval<=32:raise ValueError('keyframe interval outside 1..32')
        self.interval=keyframe_interval;self.previous=None;self.previous_path=None;self.count=0

    def write(self,path,depth):
        path=Path(path);depth=np.asarray(depth)
        if depth.ndim!=2 or depth.dtype!=np.float32:raise ValueError('two-dimensional float32 depth required')
        # Work on uint32 bit patterns: preserve signed zero, infinities and NaN payloads.
        raw=np.ascontiguousarray(depth);digest=hashlib.sha256(raw.tobytes()).hexdigest()
        keyframe=(self.previous is None or self.count%self.interval==0 or
                  self.previous.shape!=raw.shape or self.previous_path.parent!=path.parent)
        arrays={'depth':raw} if keyframe else {
            'depth_xor':np.bitwise_xor(raw.view(np.uint32),self.previous.view(np.uint32)),
            'previous':np.array(self.previous_path.name)}
        arrays.update(depth_sha256=np.array(digest),encoding=np.array('tcei.depth.xor.v1'))
        write_arrays(path,arrays)
        self.previous=raw.copy();self.previous_path=path;self.count+=1


def read_depth_archive(path,_seen=None):
    path=Path(path).resolve();seen=set() if _seen is None else _seen
    if path in seen or len(seen)>=32:raise ValueError('depth chain cycle or limit exceeded')
    seen.add(path)
    with np.load(path,allow_pickle=False) as archive:
        if 'depth' in archive:
            result=archive['depth'].copy()
            if result.dtype!=np.float32 or result.ndim!=2:raise ValueError('invalid depth keyframe')
        elif 'depth_xor' in archive:
            previous=str(archive['previous'].item())
            if Path(previous).name!=previous or '/' in previous or '\\' in previous:
                raise ValueError('invalid previous archive path')
            prior=read_depth_archive(path.parent/previous,seen);delta=archive['depth_xor']
            if delta.dtype!=np.uint32 or delta.shape!=prior.shape:raise ValueError('invalid depth delta')
            result=np.bitwise_xor(prior.view(np.uint32),delta).view(np.float32)
        else:raise ValueError('missing depth data')
        if 'depth_sha256' in archive and hashlib.sha256(result.tobytes()).hexdigest()!=str(archive['depth_sha256'].item()):
            raise ValueError('decoded depth SHA256 mismatch')
    return result

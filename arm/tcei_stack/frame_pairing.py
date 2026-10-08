"""Bind numbered scene pixels to metadata without trusting ROS Header.seq.

Noetic rewrites Header.seq during publisher serialization; it is transport
history, not an application frame identifier. Canonical BGR8 bytes, dimensions,
the exact sensor stamp and camera frame name establish the input association.
"""
import hashlib
import math
import struct
import numpy as np

IMAGE_BINDING_PROTOCOL = 'tcei.numbered_scene.bgr8.v1'


def stamp_nanoseconds(stamp):
    if hasattr(stamp, 'to_nsec'):
        result = stamp.to_nsec()
        if type(result) is not int or result < 0:
            raise ValueError('invalid exact camera timestamp')
        return result
    value = stamp.to_sec() if hasattr(stamp, 'to_sec') else stamp
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError('invalid camera timestamp')
    return int(round(value * 1000000000.))


def image_payload_binding(image, sensor_stamp, camera_frame_id):
    """Hash normalized pixel rows, excluding ROS padding/transport sequence."""
    arr = np.asarray(image)
    if arr.dtype != np.uint8 or arr.ndim != 3 or arr.shape[2] != 3 or min(arr.shape[:2]) <= 0:
        raise ValueError('numbered scene must be a nonempty BGR8 image')
    height, width = arr.shape[:2]
    if not isinstance(camera_frame_id, str):
        raise ValueError('camera frame name must retain its ROS coordinate meaning')
    digest = hashlib.sha256()
    digest.update((IMAGE_BINDING_PROTOCOL + '\0').encode('ascii'))
    digest.update(struct.pack('<II', width, height))
    digest.update(np.ascontiguousarray(arr).tobytes(order='C'))
    return {'protocol': IMAGE_BINDING_PROTOCOL, 'encoding': 'bgr8',
            'width': int(width), 'height': int(height),
            'sensor_stamp_ns': stamp_nanoseconds(sensor_stamp),
            'camera_frame_id': camera_frame_id,
            'canonical_pixel_sha256': digest.hexdigest()}


def received_image_record(image, sensor_stamp, camera_frame_id, transport_seq):
    return {'image': image, 'binding': image_payload_binding(image, sensor_stamp, camera_frame_id),
            'transport_seq': transport_seq}


def validate_image_binding(scene, received):
    """Reject same-stamp wrong pixels, size, encoding, camera frame or epoch."""
    expected = scene.get('scene_image_binding')
    actual = received.get('binding') if isinstance(received, dict) else None
    if not isinstance(expected, dict) or not isinstance(actual, dict):
        raise ValueError('numbered scene image payload binding missing')
    keys = ('protocol', 'encoding', 'width', 'height', 'sensor_stamp_ns',
            'camera_frame_id', 'canonical_pixel_sha256')
    if expected.get('protocol') != IMAGE_BINDING_PROTOCOL or expected.get('encoding') != 'bgr8':
        raise ValueError('unsupported numbered scene image binding')
    if any(key not in expected or expected[key] != actual.get(key) for key in keys):
        raise ValueError('numbered scene image payload binding mismatch')
    if scene.get('image_size') != [actual['width'], actual['height']]:
        raise ValueError('numbered scene image dimensions disagree with candidates')
    image = np.asarray(received.get('image'))
    if image.dtype != np.uint8 or image.shape != (actual['height'], actual['width'], 3):
        raise ValueError('received numbered scene pixel shape is inconsistent')
    if stamp_key(scene['stamp']) != stamp_key(actual['sensor_stamp_ns'] / 1000000000.):
        raise ValueError('numbered scene sensor timestamp disagrees with candidates')
    return True


def stamp_key(stamp):
    if not math.isfinite(stamp):raise ValueError('invalid camera timestamp')
    return int(round(stamp*1000000.))


def latest_fresh_pair(snapshots,images,now,max_age=2.):
    valid=[]
    for key,scene in snapshots.items():
        if key not in images or not 0<=now-scene['observed_at']<max_age:continue
        try:validate_image_binding(scene,images[key])
        except (ValueError,TypeError,KeyError):continue
        valid.append(scene)
    if not valid:return None
    scene=max(valid,key=lambda s:s['observed_at'])
    return scene,images[stamp_key(scene['stamp'])]

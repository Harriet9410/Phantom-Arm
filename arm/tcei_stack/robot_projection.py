"""Conservative URDF robot image exclusion from measured joints and calibration.

Python 3.8+, numpy and OpenCV; no ROS or simulator object state. URDF FK uses
each actual movable joint, including both gripper fingers. Collision and visual
geometry are unioned. Each geometry projects its convex envelope: conservative
extra masking is intentional, and is reported rather than called exact mesh
visibility. Missing geometry/calibration/feedback never produces an accepted
empty mask. Real image alignment must be validated outside these pure tests.
"""
import hashlib
import copy
import math
from pathlib import Path
import re
import struct
import xml.etree.ElementTree as ET
import cv2
import numpy as np


def _numbers(text,count,default=None):
    if text is None:
        if default is None:raise ValueError('missing geometry values')
        return np.asarray(default,dtype=float)
    values=np.asarray([float(value) for value in text.split()],dtype=float)
    if values.shape!=(count,) or not np.isfinite(values).all():raise ValueError('invalid geometry values')
    return values


def _axis_rotation(axis,angle):
    axis=np.asarray(axis,dtype=float)
    norm=np.linalg.norm(axis)
    if not math.isfinite(angle) or norm<1e-9:raise ValueError('invalid joint axis/angle')
    x,y,z=axis/norm
    skew=np.array([[0.,-z,y],[z,0.,-x],[-y,x,0.]])
    return np.eye(3)+math.sin(angle)*skew+(1.-math.cos(angle))*(skew@skew)


def _origin(element):
    transform=np.eye(4)
    if element is None:return transform
    xyz=_numbers(element.get('xyz'),3,[0.,0.,0.])
    roll,pitch,yaw=_numbers(element.get('rpy'),3,[0.,0.,0.])
    transform[:3,3]=xyz
    transform[:3,:3]=(_axis_rotation([0.,0.,1.],yaw)@
                      _axis_rotation([0.,1.,0.],pitch)@
                      _axis_rotation([1.,0.,0.],roll))
    return transform


def _box_vertices(size):
    half=np.asarray(size,dtype=float)/2.
    return np.asarray([[sx*half[0],sy*half[1],sz*half[2]]
                       for sx in (-1.,1.) for sy in (-1.,1.) for sz in (-1.,1.)])


def read_stl_vertices(path,max_bytes=200*1024*1024):
    """Read finite STL vertices without changing units or inventing geometry."""
    path=Path(path)
    if path.stat().st_size>max_bytes:raise ValueError('mesh exceeds resource limit')
    data=path.read_bytes()
    count=struct.unpack_from('<I',data,80)[0] if len(data)>=84 else -1
    if count>0 and len(data)==84+50*count:
        dtype=np.dtype([('normal','<f4',(3,)),('vertices','<f4',(3,3)),('attribute','<u2')])
        vertices=np.frombuffer(data,dtype=dtype,count=count,offset=84)['vertices'].reshape(-1,3).astype(float)
    else:
        try:text=data.decode('ascii')
        except UnicodeDecodeError as error:raise ValueError('invalid binary STL length or unsupported mesh') from error
        tokens=re.findall(r'^\s*vertex\s+([^\r\n]+)',text,re.MULTILINE|re.IGNORECASE)
        if not tokens or len(tokens)%3:raise ValueError('invalid ASCII STL triangle vertices')
        vertices=np.asarray([_numbers(token,3) for token in tokens],dtype=float)
    if len(vertices)<3 or not np.isfinite(vertices).all():raise ValueError('invalid STL vertices')
    # Projection uses a conservative envelope, so duplicated triangle vertices
    # carry no extra information. Deduplicate once, outside the frame loop.
    return np.unique(vertices,axis=0),hashlib.sha256(data).hexdigest()


class RobotProjector:
    def __init__(self,urdf_path,package_paths=None,geometryless_links=()):
        self.urdf_path=Path(urdf_path).resolve();self.package_paths=dict(package_paths or {})
        data=self.urdf_path.read_bytes();self.urdf_sha256=hashlib.sha256(data).hexdigest()
        tree=ET.fromstring(data)
        if tree.tag!='robot':raise ValueError('not a URDF robot')
        link_elements=tree.findall('link');joint_elements=tree.findall('joint')
        self.links={element.get('name'):element for element in link_elements}
        if None in self.links or len(self.links)!=len(link_elements):raise ValueError('missing or duplicate link name')
        self.joints=[];children=set();names=set();self.issues=[];self.geometries=[];self.meshes={}
        self.declared_geometryless=set(geometryless_links)
        for element in joint_elements:
            name=element.get('name');kind=element.get('type')
            parent=element.find('parent');child=element.find('child')
            if name is None or name in names or parent is None or child is None:raise ValueError('invalid joint declaration')
            names.add(name);parent=parent.get('link');child=child.get('link')
            if parent not in self.links or child not in self.links or child in children:raise ValueError('invalid URDF link tree')
            children.add(child)
            axis_element=element.find('axis')
            axis=_numbers(axis_element.get('xyz') if axis_element is not None else None,3,[1.,0.,0.])
            if kind not in ('fixed','revolute','continuous','prismatic'):
                self.issues.append({'joint':name,'reason':'unsupported_joint_type','type':kind})
            limit_element=element.find('limit')
            limits=None
            if kind in ('revolute','prismatic'):
                if limit_element is None or limit_element.get('lower') is None or limit_element.get('upper') is None:
                    self.issues.append({'joint':name,'reason':'joint_limits_missing'})
                else:
                    limits=(float(limit_element.get('lower')),float(limit_element.get('upper')))
                    if not all(math.isfinite(v) for v in limits) or limits[0]>limits[1]:raise ValueError('invalid joint limits')
            self.joints.append({'name':name,'type':kind,'parent':parent,'child':child,
                                'origin':_origin(element.find('origin')),'axis':axis,'limits':limits,
                                'mimic':element.find('mimic') is not None})
        roots=set(self.links)-children
        if len(roots)!=1:raise ValueError('URDF must contain one root link')
        self.root_link=next(iter(roots));visited={self.root_link};ordered=[]
        pending=list(self.joints)
        while pending:
            eligible=[joint for joint in pending if joint['parent'] in visited]
            if not eligible:raise ValueError('disconnected or cyclic URDF graph')
            for joint in eligible:
                visited.add(joint['child']);ordered.append(joint);pending.remove(joint)
        if visited!=set(self.links):raise ValueError('disconnected URDF links')
        self.joints=ordered
        self.required_joints=[joint['name'] for joint in self.joints if joint['type']!='fixed']
        for name,element in self.links.items():
            geometry_elements=element.findall('collision')+element.findall('visual')
            if not geometry_elements and name not in self.declared_geometryless:
                self.issues.append({'link':name,'reason':'geometry_not_declared'})
            seen=set()
            for index,geometry_element in enumerate(geometry_elements):
                try:
                    origin=_origin(geometry_element.find('origin'))
                    geometry=geometry_element.find('geometry')
                    if geometry is None or len(geometry)!=1:raise ValueError('unsupported geometry declaration')
                    shape=list(geometry)[0]
                    signature=ET.tostring(shape)+origin.tobytes()
                    if signature in seen:continue
                    seen.add(signature)
                    vertices,description=self._geometry(shape)
                    if vertices.shape[1]!=3 or not np.isfinite(vertices).all():raise ValueError('invalid geometry vertices')
                    if float(np.max(np.abs(vertices)))>5.:
                        raise ValueError('mesh_units_or_extent_unverified_over_5m')
                    if float(np.max(np.ptp(vertices,axis=0)))<1e-7:raise ValueError('degenerate geometry')
                    self.geometries.append({'link':name,'origin':origin,'vertices':vertices,
                                            'source':geometry_element.tag,'description':description})
                except (ValueError,OSError,TypeError,IndexError) as error:
                    self.issues.append({'link':name,'geometry_index':index,'reason':str(error)})
        if not self.geometries:self.issues.append({'reason':'no_geometry_loaded'})

    def _mesh_path(self,filename):
        if not filename:raise ValueError('mesh_filename_missing')
        if filename.startswith('package://'):
            package,separator,relative=filename[len('package://'):].partition('/')
            if not separator or package not in self.package_paths:raise ValueError('mesh_package_unresolved:'+package)
            return (Path(self.package_paths[package])/relative).resolve()
        if '://' in filename:raise ValueError('unsupported_mesh_URI:'+filename)
        path=Path(filename)
        return path.resolve() if path.is_absolute() else (self.urdf_path.parent/path).resolve()

    def _geometry(self,shape):
        if shape.tag=='box':
            size=_numbers(shape.get('size'),3)
            if np.min(size)<=0:raise ValueError('nonpositive_box_size')
            return _box_vertices(size),{'type':'box','projection':'convex_hull'}
        if shape.tag=='sphere':
            radius=float(shape.get('radius'))
            if not math.isfinite(radius) or radius<=0:raise ValueError('invalid_sphere_radius')
            return _box_vertices([2*radius]*3),{'type':'sphere','projection':'conservative_bounding_cube'}
        if shape.tag=='cylinder':
            radius=float(shape.get('radius'));length=float(shape.get('length'))
            if not all(math.isfinite(v) and v>0 for v in (radius,length)):raise ValueError('invalid_cylinder_size')
            # Circumscribed regular polygon contains the full circular section.
            steps=32;outer=radius/math.cos(math.pi/steps)
            vertices=np.asarray([[outer*math.cos(2*math.pi*i/steps),outer*math.sin(2*math.pi*i/steps),z]
                                 for z in (-length/2.,length/2.) for i in range(steps)])
            return vertices,{'type':'cylinder','projection':'circumscribed_32_sided_convex_hull'}
        if shape.tag=='mesh':
            path=self._mesh_path(shape.get('filename'))
            if path.suffix.casefold()!='.stl':raise ValueError('unsupported_mesh_format:'+path.suffix)
            scale=_numbers(shape.get('scale'),3,[1.,1.,1.])
            if np.min(scale)<=0:raise ValueError('nonpositive_mesh_scale')
            if str(path) not in self.meshes:self.meshes[str(path)]=read_stl_vertices(path)
            vertices,digest=self.meshes[str(path)]
            return vertices*scale,{'type':'mesh','path':str(path),'sha256':digest,
                                   'scale':scale.tolist(),'projection':'conservative_projected_convex_hull'}
        raise ValueError('unsupported_geometry:'+shape.tag)

    def forward_kinematics(self,joint_positions):
        missing=[name for name in self.required_joints if name not in joint_positions]
        if missing:raise ValueError('missing_measured_joints:'+','.join(missing))
        transforms={self.root_link:np.eye(4)}
        for joint in self.joints:
            motion=np.eye(4);kind=joint['type']
            if kind!='fixed':
                value=float(joint_positions[joint['name']])
                if not math.isfinite(value):raise ValueError('nonfinite_measured_joint:'+joint['name'])
                if joint['limits'] and not joint['limits'][0]-1e-5<=value<=joint['limits'][1]+1e-5:
                    raise ValueError('measured_joint_outside_URDF_limit:'+joint['name'])
                axis=joint['axis'];norm=np.linalg.norm(axis)
                if norm<1e-9:raise ValueError('invalid_joint_axis:'+joint['name'])
                if kind in ('revolute','continuous'):motion[:3,:3]=_axis_rotation(axis,value)
                elif kind=='prismatic':motion[:3,3]=axis/norm*value
                else:raise ValueError('unsupported_joint_type:'+str(kind))
            transforms[joint['child']]=transforms[joint['parent']]@joint['origin']@motion
        return transforms

    def project(self,joint_positions,k,image_size,*,joint_stamp,image_stamp,calibration,
                max_joint_age=.05,pixel_margin=2):
        """Return mask+diagnostics, with mask=None unless coverage is complete.

        calibration must explicitly contain verified=True, id, root_link,
        base_to_camera (optical 4x4), units='m', rectified=True, distortion=[] or
        zeros, image_size=[w,h], and intrinsics (same flattened K as this frame).
        verified refers to a caller-owned, independently checked calibration;
        setting the flag alone does not validate its physical correctness.
        """
        result={'mask':None,'coverage_complete':False,'root_link':self.root_link,
                'urdf_sha256':self.urdf_sha256,'image_stamp':image_stamp,'joint_stamp':joint_stamp,
                'unsupported_geometry':list(self.issues),'projection':'per_geometry_conservative_envelope',
                'required_joints':list(self.required_joints)}
        try:
            if self.issues:raise ValueError('robot_geometry_incomplete')
            if not isinstance(calibration,dict) or calibration.get('verified') is not True or not calibration.get('id'):
                raise ValueError('calibration_unverified')
            if calibration.get('root_link')!=self.root_link:raise ValueError('calibration_root_link_mismatch')
            if calibration.get('urdf_sha256',self.urdf_sha256)!=self.urdf_sha256:
                raise ValueError('calibration_URDF_version_mismatch')
            if calibration.get('units')!='m':raise ValueError('calibration_units_unverified')
            if calibration.get('rectified') is not True:raise ValueError('camera_not_explicitly_rectified')
            distortion=np.asarray(calibration.get('distortion',[float('nan')]),dtype=float)
            if not np.isfinite(distortion).all() or np.any(np.abs(distortion)>1e-9):raise ValueError('nonzero_or_unknown_camera_distortion')
            width,height=map(int,image_size)
            if min(width,height)<=0 or width*height>16*1024*1024 or list(calibration.get('image_size',[]))!=[width,height]:raise ValueError('image_size_calibration_mismatch')
            k=np.asarray(k,dtype=float)
            if k.shape!=(9,) or not np.isfinite(k).all() or min(k[0],k[4])<=0 or abs(k[8]-1)>1e-8:
                raise ValueError('invalid_camera_intrinsics')
            if not np.allclose(k,calibration.get('intrinsics',[]),rtol=0,atol=1e-5):raise ValueError('intrinsics_calibration_mismatch')
            if max(abs(k[1]),abs(k[3]),abs(k[6]),abs(k[7]))>1e-8:raise ValueError('unsupported_skew_intrinsics')
            transform=np.asarray(calibration.get('base_to_camera'),dtype=float)
            if transform.shape!=(4,4) or not np.isfinite(transform).all():raise ValueError('invalid_base_to_camera')
            rotation=transform[:3,:3]
            if not np.allclose(transform[3],[0.,0.,0.,1.],atol=1e-8) or not np.allclose(rotation.T@rotation,np.eye(3),atol=1e-5) or abs(np.linalg.det(rotation)-1.)>1e-5:
                raise ValueError('base_to_camera_not_rigid_right_handed')
            joint_stamp=float(joint_stamp);image_stamp=float(image_stamp)
            if not all(math.isfinite(v) for v in (joint_stamp,image_stamp,max_joint_age)) or max_joint_age<0 or abs(joint_stamp-image_stamp)>max_joint_age:
                raise ValueError('measured_joints_not_synchronized')
            if type(pixel_margin) is not int or not 1<=pixel_margin<=20:raise ValueError('invalid_projection_margin')
            transforms=self.forward_kinematics(joint_positions)
            mask=np.zeros((height,width),np.uint8);projected=[];outside=[]
            for index,geometry in enumerate(self.geometries):
                local=geometry['vertices']
                matrix=transform@transforms[geometry['link']]@geometry['origin']
                vertices=local@matrix[:3,:3].T+matrix[:3,3]
                if np.max(vertices[:,2])<=.001:
                    outside.append({'link':geometry['link'],'reason':'behind_camera'});continue
                if np.min(vertices[:,2])<=.001:raise ValueError('geometry_crosses_camera_plane:'+geometry['link'])
                xy=np.column_stack((vertices[:,0]*k[0]/vertices[:,2]+k[2],vertices[:,1]*k[4]/vertices[:,2]+k[5]))
                if not np.isfinite(xy).all() or np.max(np.abs(xy))>1e7:raise ValueError('invalid_projected_geometry')
                if xy[:,0].max()<-pixel_margin or xy[:,0].min()>width+pixel_margin or xy[:,1].max()<-pixel_margin or xy[:,1].min()>height+pixel_margin:
                    outside.append({'link':geometry['link'],'reason':'outside_image'});continue
                hull=cv2.convexHull(xy.astype(np.float32)).reshape(-1,2)
                if len(hull)<3 or abs(cv2.contourArea(hull))<1e-4:raise ValueError('degenerate_projected_geometry:'+geometry['link'])
                fixed=np.rint(hull*16.).astype(np.int32)
                cv2.fillConvexPoly(mask,fixed,1,lineType=cv2.LINE_8,shift=4)
                projected.append({'link':geometry['link'],'geometry_index':index,'source':geometry['source'],
                                  'description':geometry['description'],'hull_vertices':len(hull)})
            if not np.any(mask):raise ValueError('empty_projected_robot_mask')
            kernel=np.ones((2*pixel_margin+1,2*pixel_margin+1),np.uint8)
            mask=cv2.dilate(mask,kernel).astype(bool)
            result.update(mask=mask,coverage_complete=True,reason='complete_declared_geometry_projected',
                          mask_stamp=image_stamp,calibration_id=calibration['id'],
                          covered_pixels=int(mask.sum()),pixel_margin=pixel_margin,
                          projected_geometries=projected,outside_geometries=outside,
                          loaded_geometry_count=len(self.geometries),
                          measured_joint_names=list(self.required_joints))
        except (ValueError,TypeError,KeyError,IndexError,OverflowError) as error:
            result['reason']=str(error)
        return result

    def project_debug(self,joint_positions,k,image_size,*,joint_stamp,image_stamp,calibration,
                      max_joint_age=.05,pixel_margin=2):
        """Draw an unverified calibration overlay, never a control-ready mask.

        The original configuration is not mutated. The array is deliberately
        named debug_mask; mask remains None and coverage_complete stays False,
        so consumers of project() cannot accidentally use this as lift proof.
        """
        candidate=copy.deepcopy(calibration) if isinstance(calibration,dict) else {}
        candidate['verified']=True;candidate['id']='UNVERIFIED_DEBUG_ONLY'
        result=self.project(joint_positions,k,image_size,joint_stamp=joint_stamp,image_stamp=image_stamp,
                            calibration=candidate,max_joint_age=max_joint_age,pixel_margin=pixel_margin)
        drawn=result.pop('mask',None)
        geometry_available=result['coverage_complete']
        result.update(mask=None,debug_mask=drawn,coverage_complete=False,usable_for_control=False,
                      debug_geometry_projected=geometry_available,calibration_verified=False,
                      warning='Unverified calibration overlay only; never use as trial-lift evidence.')
        return result

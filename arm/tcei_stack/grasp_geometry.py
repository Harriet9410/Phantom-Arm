"""Depth-only body center inside a detector box; excludes thin handles/levers."""
import cv2
import numpy as np


def body_center(depth,bbox,support_z):
    x1,y1,x2,y2=map(int,bbox)
    roi=np.asarray(depth[y1:y2,x1:x2],dtype=np.float32)
    height=4.5-roi-support_z
    mask=(np.isfinite(roi)&(height>.005)&(height<.15)).astype(np.uint8)
    mask=cv2.morphologyEx(mask,cv2.MORPH_OPEN,np.ones((3,3),np.uint8))
    count,labels,stats,_=cv2.connectedComponentsWithStats(mask,connectivity=8)
    components=sorted([(int(stats[i,cv2.CC_STAT_AREA]),i) for i in range(1,count)],reverse=True)
    if not components or components[0][0]<30:raise ValueError('no measurable grasp body')
    if len(components)>1 and components[1][0]>.7*components[0][0]:raise ValueError('ambiguous grasp bodies')
    body=(labels==components[0][1]).astype(np.uint8)
    distance=cv2.distanceTransform(body,cv2.DIST_L2,5)
    if float(distance.max())<3:raise ValueError('grasp body too thin')
    core=distance>=.75*float(distance.max())
    ys,xs=np.nonzero(core);weights=distance[core]
    u=x1+float(np.average(xs,weights=weights));v=y1+float(np.average(ys,weights=weights))
    return u,v,float(np.median(roi[core]))

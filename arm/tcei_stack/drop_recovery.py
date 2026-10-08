"""Camera-only reacquisition after a drop; never substitutes simulator truth."""
import math
from core import rebind_candidate


def pickable_recovery_objects(scene):
    objects=[]
    for raw in scene.get('candidates',[])+scene.get('observations',[]):
        c=dict(raw);p=c.get('world_position');support=c.get('support_z')
        if not p or support is None or c.get('confidence',0)<.60:continue
        if not all(math.isfinite(v) for v in p+[support]):continue
        # Exclude previously delivered objects on either conveyor and objects
        # on the floor. This includes the visible tabletop next to the basket.
        if not (abs(p[0])<.43 and -.40<p[1]<.40 and 2.34<support<2.44 and .002<p[2]-support<.15):continue
        if any(c['class']==x['class'] and math.dist(c['pixel'],x['pixel'])<12 and abs(c['depth']-x['depth'])<.025 for x in objects):continue
        c.setdefault('id','recovery_%d'%(len(objects)+1))
        c['_drop_recovery']=True;objects.append(c)
    return objects


def reacquire_dropped(selected,other_references,current):
    options=[c for c in current if c['class']==selected['class'] and c['confidence']>=.60]
    # Stable, other same-class objects are not silently substituted for the
    # dropped item. Candidate IDs can change after every camera frame.
    for reference in other_references:
        if reference['class']!=selected['class']:continue
        try:matched=rebind_candidate(reference,options,max_pixels=22.,max_depth=.04)
        except ValueError:continue
        options=[c for c in options if c is not matched]
    if len(options)!=1:raise ValueError('dropped object not uniquely visible')
    return options[0]

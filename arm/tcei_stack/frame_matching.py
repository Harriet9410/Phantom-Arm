"""Select a fresh depth exposure matching a processed RGB-D scene timestamp."""
import math


def matching_depth(history,stamp,now,max_age=1.5,max_delta=.08):
    if not math.isfinite(stamp):return None
    eligible=[item for item in history if 0<=now-item[1]<=max_age and
              math.isfinite(item[2]) and abs(item[2]-stamp)<=max_delta]
    return min(eligible,key=lambda item:abs(item[2]-stamp)) if eligible else None


def reference_consistency(expected,measured):
    """Explain an existing context object's body by unchanged depth or high arm.

    This does not classify a new object and must not authorize a target grasp.
    Missing/moved surfaces remain unexplained and are rejected.
    """
    if len(expected)<16 or len(expected)!=len(measured):return None
    valid=matched=occluded=0
    for old,new in zip(expected,measured):
        if not math.isfinite(old) or not math.isfinite(new) or not .1<new<5:continue
        valid+=1
        if abs(old-new)<=.006:matched+=1
        elif new<old-.10 and new<2.0:occluded+=1
    if valid<.8*len(expected) or matched+occluded<.9*valid:return None
    return {'samples':len(expected),'valid':valid,'matched':matched,'occluded':occluded,
            'unexplained':valid-matched-occluded,'method':'time_matched_depth_body_reference'}

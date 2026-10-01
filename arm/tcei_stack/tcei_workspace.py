"""Geometry-only bounds, importable before Isaac loads its extension importer.

Keep this module independent of task semantics and perception. Isaac also owns
a top-level ``semantics`` package; loading our task core inside a physics tick
can collide with that package and prevent planning before IK is entered.
"""
import math


def validate_workspace(p):
    try:
        finite = len(p) == 3 and all(math.isfinite(v) for v in p)
    except (TypeError, ValueError):
        finite = False
    if not finite:
        raise ValueError('nonfinite pose')
    if not (-.9 <= p[0] <= .9 and -.6 <= p[1] <= .9 and 2.35 <= p[2] <= 2.95):
        raise ValueError('outside verified scene envelope')
    return p

"""Declared test-only loss of one release's derived belt observations."""
class DeliveryEvidenceGap:
    def __init__(self):self.release_id=None
    def select_first_release(self,release_id):
        if not isinstance(release_id,str) or not release_id:raise ValueError('release ID required')
        if self.release_id is not None:return False
        self.release_id=release_id;return True
    def filter_scene(self,scene):
        if self.release_id is None:return scene,[]
        removed=[o for o in scene.get('observations',[]) if o.get('release_id')==self.release_id]
        if not removed:return scene,[]
        return {**scene,'observations':[o for o in scene['observations'] if o.get('release_id')!=self.release_id]},removed

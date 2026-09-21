"""Bounded regeneration; never alter the model's proposed class/side/count/IDs."""
from core import parse_plan, parse_selection
from semantics import ObservationRequired, ModelNeedsConfirmation


def infer_validated_plan(infer,candidates,instruction,on_reject=None,scene=None,
                         task_context=None,allow_legacy=True):
    previous=None;feedback=None
    for attempt in (1,2):
        answer=str(infer(attempt,previous,feedback))
        try:return parse_plan(answer,candidates,instruction,scene,task_context,allow_legacy)
        except ObservationRequired as error:
            if on_reject:on_reject(attempt,answer,str(error))
            raise
        except ValueError as error:
            if on_reject:on_reject(attempt,answer,str(error))
            if attempt==2:raise
            previous=answer;feedback=str(error)


def infer_validated_selection(infer,candidates,instruction,scene,task_context=None,on_reject=None):
    """At most two actual model calls; a valid uncertainty decision stops once."""
    previous=None;feedback=None
    for attempt in (1,2):
        answer=str(infer(attempt,previous,feedback))
        try:return parse_selection(answer,candidates,instruction,scene,task_context)
        except ModelNeedsConfirmation:
            raise
        except ObservationRequired as error:
            if on_reject:on_reject(attempt,answer,str(error))
            raise
        except ValueError as error:
            if on_reject:on_reject(attempt,answer,str(error))
            if attempt==2:raise
            previous=answer;feedback=str(error)

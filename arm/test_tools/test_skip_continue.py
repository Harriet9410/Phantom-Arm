"""Guard the ledger semantics behind "skip a rejected task and continue".

`episode_driver._run_task` routes rejected / request_rejected / plan_rejected through
`mission_ledger.release_rejected_task()` instead of raising EpisodeFailure (which used
to end the whole round at the first rejected instruction).  That routing is only sound
if the ledger actually resolves such a task and then lets the NEXT task be requested,
and if real grasp/placement evidence still blocks.

`mission_ledger` is pure stdlib (its docstring: "No ROS, simulator truth, object
manipulation, or language-model replacement"), so this runs anywhere:

    python test_tools/test_skip_continue.py
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / 'tcei_stack'))
from mission_ledger import MissionLedger  # noqa: E402

INSTRUCTIONS = ['抓取左上方的烟雾弹，放到左侧传送带',
                '抓取右下方的弹夹，放到右侧传送带',
                '抓取最左方的军用手电筒，放到左侧传送带']

failures = []


def check(label, got, want):
    ok = got == want
    print('%-58s got=%-22s want=%s %s' % (label, got, want, 'OK' if ok else '<<< FAIL'))
    if not ok:
        failures.append(label)


def fresh():
    led = MissionLedger('round_test', INSTRUCTIONS, 1000.0, 1700000000.0, budget_seconds=600)
    return led


def request(led, index, tag):
    """Drive a task to the state a planner rejection leaves it in."""
    rid = 'episode_' + tag
    led.mark_requested(index, rid)
    return led.tasks[index]


# --- A: a rejection resolves the task even though no target was ever bound
led = fresh()
task = request(led, 0, 'a')
check('A rejected task has no bound targets', task['targets'], {})
check('A next task blocked while the rejection is unresolved',
      led._dependencies(1), False)
try:
    led.mark_requested(1, 'episode_b')
    blocked_raised = False
except ValueError as error:
    blocked_raised = 'dependencies not satisfied' in str(error)
check('A requesting the next task raises before resolution', blocked_raised, True)

resolved = led.release_rejected_task(0, 'vision/model category disagreement', 1001.0)
check('A release_rejected_task resolves it', resolved, True)
check('A resolution is recorded with empty released ids',
      led.tasks[0]['released_after_failure']['released_stable_ids'], [])
check('A next task is now unblocked', led._dependencies(1), True)
led.mark_requested(1, 'episode_b')
check('A next task request now succeeds', led.tasks[1]['status'], 'requested')

# --- B: grasp/placement evidence still blocks (the driver keeps its hard stop)
led2 = fresh()
request(led2, 0, 'c')
led2.tasks[0]['grasps'] = {'stable-1': {'side': 'left'}}
check('B release_rejected_task refuses with grasp evidence',
      led2.release_rejected_task(0, 'rejected after motion'), False)
check('B still blocked for the next task', led2._dependencies(1), False)

# --- C: release_failed_task still requires bound targets (unchanged behaviour)
led3 = fresh()
request(led3, 0, 'd')
check('C release_failed_task unchanged: refuses without targets',
      led3.release_failed_task(0, 'controller failure'), False)

# --- D: an already-verified task is never released
led4 = fresh()
request(led4, 0, 'e')
led4.tasks[0]['status'] = 'verified'
check('D verified task is not released',
      led4.release_rejected_task(0, 'late rejection'), False)

print()
print('RESULT:', 'ALL PASS' if not failures else 'FAILURES: %s' % failures)
sys.exit(1 if failures else 0)

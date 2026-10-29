"""Public new-run factory; task-specific dynamics/observations remain explicit."""
from .independent import make_task as make_independent_task
from extensions.unified_training_v3.environment import make_task as make_unified_task


def make_task(family, name=None, **kwargs):
    if family == "independent":
        if name is None:
            raise ValueError("Declare an independent task name")
        return make_independent_task(name, kwargs)
    return make_unified_task(family, name, **kwargs)

"""Backend discovery is explicit; a common runner does not imply a common loss."""
from importlib import import_module

BACKENDS = {
    "rgb_legacy": ("extensions.unified_training_v1.backends.rgb", "RGBBackend"),
    "push_discrete_sac": ("extensions.unified_training_v1.backends.push", "PushBackend"),
    "state_model_gradient": ("extensions.unified_training_v1.backends.state", "StateBackend"),
    "observer_model_gradient": ("extensions.unified_training_v1.backends.observer", "ObserverBackend"),
}


def make_backend(name, config, output, *, event, deadline):
    module_name, class_name = BACKENDS[name]
    module = import_module(module_name)
    constructor = getattr(module, class_name, None) or getattr(module, "Backend")
    return constructor(config, output, event=event, deadline=deadline)

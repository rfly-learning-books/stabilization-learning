"""Numerical failure boundaries without replacing the retained optimizers."""
from .numerical import require_finite


class NumericalGuard:
    """Latch numerical failures and check original module/optimizer state.

    Hooks observe Adam's inputs and outputs without changing its arithmetic,
    optimizer identity, checkpoint format, or parameter ordering. They do not
    substitute for a finite loss / pre-clipping norm check in each learner.
    """
    def __init__(self, label):
        self.label = str(label)
        self._failure = None
        self._closed = False
        self._modules = {}
        self._optimizers = {}
        self._handles = []

    def fail(self, error):
        if self._failure is None:
            self._failure = str(error)

    def guard(self):
        if self._closed:
            raise RuntimeError(self.label + " numerical guard is closed")
        if self._failure is not None:
            raise FloatingPointError(self.label + " stopped after numerical failure: " + self._failure)

    def check(self, value, label):
        self.guard()
        try:
            require_finite(value, self.label + " " + str(label))
        except FloatingPointError as error:
            self.fail(error)
            raise
        return value

    def watch_module(self, module, label):
        self.guard()
        self.check(module.state_dict(), label)
        self._modules[id(module)] = (module, str(label))

    def _check_optimizer(self, optimizer, label, *, gradients=False):
        self.guard()
        for group in optimizer.param_groups:
            self.check({key: value for key, value in group.items() if key != "params"}, label + " settings")
            for parameter in group["params"]:
                self.check(parameter, label + " parameter")
                if gradients and parameter.grad is not None:
                    self.check(parameter.grad, label + " gradient")
        # Includes every slot, even an orphaned state entry following a load.
        for state in optimizer.state.values():
            self.check(state, label + " optimizer state")

    def watch_optimizer(self, optimizer, label):
        self.guard()
        key = id(optimizer)
        if key in self._optimizers:
            return
        label = str(label)
        self._check_optimizer(optimizer, label)
        def before_step(active, _args, _kwargs):
            self._check_optimizer(active, label, gradients=True)
        def after_step(active, _args, _kwargs):
            self._check_optimizer(active, label)
        pre = optimizer.register_step_pre_hook(before_step)
        try:
            post = optimizer.register_step_post_hook(after_step)
        except BaseException:
            pre.remove()
            raise
        self._handles.extend((pre, post))
        self._optimizers[key] = (optimizer, label)

    def check_all(self):
        self.guard()
        for module, label in self._modules.values():
            self.check(module.state_dict(), label)
        for optimizer, label in self._optimizers.values():
            self._check_optimizer(optimizer, label)

    def close(self):
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._modules.clear()
        self._optimizers.clear()
        self._closed = True


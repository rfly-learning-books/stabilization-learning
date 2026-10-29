"""Composable model-gradient learners with explicit device capabilities.

These objects own parameters and optimizers, never an experiment directory or
test set. Callers supply physical models/task adapters and networks. The state
branch delegates to the original CPU-only full-horizon learner; the observer
branch executes the existing full-sequence TensorModel/rollout on CPU or CUDA.
No objective, success criterion, initialization, or horizon is shortened here.
"""
from __future__ import annotations

import hashlib
import inspect
from pathlib import Path
import time

import numpy as np
import torch


def _device(value, *, supports_cuda):
    if value == "auto":
        value = "cuda" if supports_cuda and torch.cuda.is_available() else "cpu"
    device = torch.device(value)
    if device.type not in ("cpu", "cuda"):
        raise ValueError("Only CPU and CUDA devices are supported")
    if device.type == "cuda" and not supports_cuda:
        raise ValueError("The preserved state full-horizon objective is CPU float64 only; CUDA is not implemented")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; refusing an implicit CPU fallback")
    return device


def _network_hash(network):
    digest=hashlib.sha256()
    for key,tensor in network.state_dict().items():
        digest.update(key.encode()); digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


class StateModelGradientLearner:
    """Original state-M learning, including original normalization and rollback.

``initialize()`` uses the explicit ``FullHorizonConfig`` training/validation
seeds. ``update()`` retains complete physical horizons, counterexample replay,
dual updates and the original loss. ``validate()`` retains feasibility gates
and rollback. This class does not evaluate the sealed test split or save files.

The original objective explicitly rejects CUDA. ``device='auto'`` therefore
chooses CPU; ``device='cuda'`` fails instead of merely moving the network.
"""
    supports_cuda=False

    def __init__(self, task, network, config=None, *, device="cpu", should_stop=None):
        from stabilization_learning.algos.pytorch.state_metric_full_horizon import FullHorizonConfig, FullHorizonMetricLearner
        from stabilization_learning.algos.pytorch.state_metric_full_horizon import objectives
        self.device=_device(device,supports_cuda=False)
        self.task,self.network=task,network
        if any(p.dtype != torch.float64 for p in network.parameters()):
            raise ValueError("State network must retain its original float64 parameter initialization")
        if any(p.device.type != "cpu" for p in network.parameters()):
            raise ValueError("Supply the original state network on CPU")
        self.config=FullHorizonConfig() if config is None else config
        paths=[Path(inspect.getfile(FullHorizonMetricLearner)),Path(inspect.getfile(objectives))]
        digest=hashlib.sha256(b"".join(path.read_bytes() for path in paths)).hexdigest()
        self.learner=FullHorizonMetricLearner(task,network,self.config,source_digest=digest)
        self.learner.should_stop=should_stop
        self.initial_parameter_sha256=_network_hash(network)
        self.capabilities=dict(family="state",device="cpu",dtype="float64",supports_cuda=False,
            vectorized_batch=True,full_horizon_steps=int(task.max_steps),
            objective="original full_horizon_components and FullHorizonMetricLearner.update",
            initialization="caller-supplied parameters preserved; new empty Adam",
            limitations=["original recovery and objective are CPU-only", "no automatic test or experiment I/O"])

    @property
    def optimizer(self):return self.learner.optimizer

    @property
    def updates(self):return self.learner.updates

    def initialize(self):
        self.learner.initialize()
        return dict(capabilities=self.capabilities,initial_parameter_sha256=self.initial_parameter_sha256,
                    cost_scale=self.learner.cost_scale)

    def update(self):
        return self.learner.update()

    def validate(self):
        return self.learner.validate()


class ObserverModelGradientLearner:
    """Full-sequence observer learning using the unchanged existing objective.

``prepare_batch(episodes)`` accepts in-memory supervised episode dictionaries;
the caller owns sampling and split boundaries. It stacks and moves a batch
once. ``loss(batch)`` and ``update(batch)`` operate on device-resident tensors.
The exact original ``rollout`` supplies the entire loss and all model steps.
The network receives only estimates; truth is confined to the supervised loss.
"""
    supports_cuda=True

    def __init__(self, model, network, *, device="cpu", learning_rate=.003,
                 gradient_clip=2., should_stop=None):
        from extensions.observer_learning_v1.tensor_model import TensorModel
        if learning_rate <= 0 or gradient_clip <= 0:
            raise ValueError("Positive learning rate and gradient clip required")
        # Public observation environments expose a model; a model itself is
        # equally valid. No task-name lookup or experiment path is performed.
        self.model=getattr(model,"model",model)
        if not hasattr(self.model,"spec"):
            raise TypeError("Supply an observation physical model or public task exposing .model")
        self.device=_device(device,supports_cuda=True)
        self.dtype=torch.float64
        self.network=network.to(device=self.device,dtype=self.dtype)
        class ResidentTensorModel(TensorModel):
            """Cache mathematical constants; all inherited formulas stay intact."""
            def __init__(self,*args,**kwargs):
                self.constant_cache={}
                super().__init__(*args,**kwargs)

            def tensor(self,value):
                # TensorModel.tensor is used for model constants, not states.
                # Batched observations enter via prepare_batch, separately.
                if isinstance(value,torch.Tensor):
                    return super().tensor(value)
                array=np.asarray(value)
                if array.dtype.kind not in "bifu":
                    return super().tensor(value)
                key=(array.shape,array.dtype.str,array.tobytes())
                if key not in self.constant_cache:
                    result=super().tensor(value)
                    if len(self.constant_cache)<32:self.constant_cache[key]=result
                    return result
                return self.constant_cache[key]

        self.tensor_model=ResidentTensorModel(self.model,str(self.device),self.dtype)
        self.optimizer=torch.optim.Adam(self.network.parameters(),lr=float(learning_rate))
        self.gradient_clip=float(gradient_clip);self.should_stop=should_stop
        self.updates=0;self.training_transitions=0
        self.horizon_samples=round(self.model.spec.horizon/self.model.spec.dt)+1
        self.initial_parameter_sha256=_network_hash(self.network)
        self.capabilities=dict(family="observer",device=str(self.device),dtype="float64",supports_cuda=True,
            vectorized_batch=True,full_horizon_steps=self.horizon_samples-1,
            objective="unchanged observer_learning_v1.learning.rollout",
            initialization="caller-supplied parameters preserved in float64; new empty Adam",
            data_transfer="explicit batch preparation; device-resident batch and cached TensorModel constants",
            limitations=["physical dataset generation remains caller-owned", "small float64 workloads can be faster on CPU",
                         "no validation selection, recovery or experiment I/O is implicitly performed"])

    def _guard(self):
        if self.should_stop and self.should_stop():
            raise InterruptedError("Stop requested before committing an optimizer update")

    def _synchronize(self):
        if self.device.type == "cuda":torch.cuda.synchronize(self.device)

    def prepare_batch(self, episodes):
        from extensions.observer_learning_v1.learning import batch_data
        if not episodes:raise ValueError("A nonempty in-memory episode batch is required")
        batch=batch_data(self.model,episodes,device=str(self.device),dtype=self.dtype)
        self._validate_batch(batch)
        return batch

    def _validate_batch(self,batch):
        required=("initial","truth","measurement","mask","control","noise")
        if any(key not in batch for key in required):
            raise ValueError("Incomplete observer batch")
        for key in required:
            tensor=batch[key]
            if not isinstance(tensor,torch.Tensor) or tensor.dtype != self.dtype:
                raise ValueError("Observer batches must contain float64 device tensors")
            if tensor.device.type != self.device.type or (self.device.index is not None and tensor.device.index != self.device.index):
                raise ValueError("Prepare the entire observer batch on the selected device")
        truth=batch["truth"]
        if truth.ndim != 3 or truth.shape[0] != self.horizon_samples or truth.shape[1] < 1:
            raise ValueError("Use the complete original observation horizon; truncation is not supported")
        if truth.shape[-1] != self.model.spec.state_dim or batch["initial"].shape != truth.shape[1:]:
            raise ValueError("Observer physical-state dimensions differ")
        for key in ("measurement","mask","control"):
            if batch[key].shape[:2] != truth.shape[:2]:
                raise ValueError("Observer time/batch dimensions differ")
        if batch["noise"].shape != (truth.shape[1],):raise ValueError("One noise flag per episode required")

    def loss(self,batch,*,keep_trace=False):
        from extensions.observer_learning_v1.learning import rollout
        self._guard();self._validate_batch(batch)
        return rollout(self.network,self.tensor_model,batch,keep_trace=keep_trace)

    @torch.no_grad()
    def evaluate(self,batch,*,keep_trace=False):
        """Pure supplied-batch evaluation; does not select or open any split."""
        return self.loss(batch,keep_trace=keep_trace)

    def update(self,batch):
        self._guard();self._synchronize();begun=time.perf_counter()
        self.optimizer.zero_grad(set_to_none=True)
        result=self.loss(batch)
        if not torch.isfinite(result["loss"]):raise FloatingPointError("Nonfinite observer loss")
        result["loss"].backward()
        if any(p.grad is None or not torch.isfinite(p.grad).all() for p in self.network.parameters()):
            raise FloatingPointError("Nonfinite or missing observer gradients")
        norm=torch.nn.utils.clip_grad_norm_(self.network.parameters(),self.gradient_clip)
        self._guard();self.optimizer.step();self.updates+=1
        batch_size=int(batch["initial"].shape[0]);self.training_transitions+=batch_size*(self.horizon_samples-1)
        self._synchronize();elapsed=time.perf_counter()-begun
        return dict(update=self.updates,loss=float(result["loss"].detach()),
            mean_cost=float(result["cost"].mean().detach()),certificate_loss=float(result["certificate_loss"].detach()),
            gradient_norm=float(norm),training_transitions=self.training_transitions,
            batch_size=batch_size,horizon_steps=self.horizon_samples-1,wall_seconds=elapsed,
            model_transitions_per_second=batch_size*(self.horizon_samples-1)/elapsed,device=str(self.device))


class ModelGradientLearner:
    """Factories share library conventions while keeping task-specific math."""
    @staticmethod
    def from_state_task(task,network,config=None,**kwargs):
        return StateModelGradientLearner(task,network,config,**kwargs)

    @staticmethod
    def from_observer_task(task,network,**kwargs):
        return ObserverModelGradientLearner(task,network,**kwargs)


__all__=["ModelGradientLearner","StateModelGradientLearner","ObserverModelGradientLearner"]

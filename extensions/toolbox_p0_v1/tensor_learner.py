"""Real raw/log-prob sampler adapter for the existing transactional executor.

Admission is deliberately limited to the four fully checked original static
base tasks. Corrected moving-reference subclasses have a different kernel
identity and are rejected, rather than silently using the old reference.
"""
from __future__ import annotations
from copy import deepcopy
import hashlib
from pathlib import Path
import numpy as np
import torch
from stabilization_gym.interfaces.m_learning import PortableMFacade,CapabilityError
from stabilization_learning.algos.pytorch.metric_ppo.metric_ppo import module_hash,summarize
from stabilization_learning.common.contracts_v2 import identity
from extensions.fast_m_learning_v2.learner import SparseValidationLearner
from extensions.fast_m_learning_v1.fast_m_learning.batch_rollout import FullHorizonRunner
from .corrected_kernel import CorrectedTensorKernel


class TensorEpisodeLearner(SparseValidationLearner):
    def __init__(self,task,*,config):
        if task.env_id not in ("coupled_linear","point_mass","pendulum","cartpole"):
            raise CapabilityError("Tensor learner admission currently covers four static base tasks only")
        self.physical=task
        self.kernel=CorrectedTensorKernel(task)
        self._pending_batch=None
        super().__init__(PortableMFacade(task),config=config,env_id=task.env_id,env_kwargs={"tensor_objective":"true_norm_v2"})

    def _protocol(self):
        result=super()._protocol()
        result.update(executor="transactional_tensor_runner_P0_v2",kernel=self.kernel.spec.to_dict(),
            objective="physical_cost_plus_true_norm_descent_v2",
            adapter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        return result

    def _run(self,seeds,*,actor,identity_matrix=False,sampling=False):
        contexts=[];states=[];raws=[];logps=[];matrices=[];rngs={}
        ids=[str(seed) for seed in seeds]
        for seed in seeds:
            observation,_=self.physical.reset(seed=int(seed))
            context=self._context(observation);contexts.append(context);states.append(observation.copy())
            rngs[str(seed)]={"physical_rng":deepcopy(self.physical.inner_env.np_random.bit_generator.state),
                            "torch_before_raw":torch.get_rng_state().clone()}
            with torch.no_grad():
                dist=actor.distribution(context)
                raw=actor.center if identity_matrix else dist.sample() if sampling else actor.mean(context)
                raw=raw.clone();raws.append(raw);logps.append(dist.log_prob(raw));matrices.append(actor.metric(raw).double())
        import extensions.fast_m_learning_v1.fast_m_learning.batch_rollout as runner_module
        runner_path=Path(runner_module.__file__).resolve()
        manifest=dict(self.kernel.spec.source_sha256)
        manifest[str(runner_path)]=hashlib.sha256(runner_path.read_bytes()).hexdigest()
        runner=FullHorizonRunner(self.kernel,source_manifest=manifest,
            config_sha256=identity(vars(self.config)),protocol_sha256=identity(self._protocol()),
            certificate_weight=self.config.certificate_weight)
        n=len(seeds)
        runner.start(torch.as_tensor(np.stack(states),dtype=torch.float64),torch.stack(matrices),
            episode_ids=ids,seeds=list(map(int,seeds)),initial_scales=torch.full((n,),float(self.physical.inner_env.initial_scale),dtype=torch.float64),
            actor_versions=[module_hash(actor)]*n,phi_versions=[self.representation_hash or "identity"]*n,
            raw=torch.stack(raws),raw_logp=torch.stack(logps),context=torch.stack(contexts),
            rng_protocol="explicit reset seeds, scalar Torch raw sampling order, caller-owned states",rng_state_by_episode=rngs)
        while not runner.complete:runner.advance()
        result=runner.summary()
        keys=("fixed_horizon_cost","accumulated_cost","objective","certificate_penalty","V_rise_fraction",
              "success","failure","physical_steps","arrival_time","terminal_physical_error")
        rows=[{key:result[key][i].item() for key in keys} for i in range(n)]
        returns=torch.tensor([-row["objective"]/(self.physical.dt*self.physical.max_steps) for row in rows])
        return torch.stack(contexts),torch.stack(raws),torch.stack(logps),returns,rows

    def _collect(self,count):
        if self._pending_batch is not None:raise RuntimeError("Update or explicitly discard the existing high-level batch")
        seeds=[self.config.seed+self.episodes+i for i in range(count)]
        if set(seeds)&set(self.config.validation_seeds):raise ValueError("Training/validation seeds overlap")
        batch=self._run(seeds,actor=self.actor,sampling=True)
        self.episodes+=count;self.physical_steps+=sum(r["physical_steps"] for r in batch[4])
        self._pending_batch=(module_hash(self.actor),identity(batch[:4]))
        return batch

    def _update(self,context,raw,old_logp,returns):
        if self._pending_batch!=(module_hash(self.actor),identity((context,raw,old_logp,returns))):
            raise ValueError("Stale, modified or foreign raw/log-prob policy batch")
        result=super()._update(context,raw,old_logp,returns)
        self._pending_batch=None
        return result

    def evaluate(self,seeds=None,*,identity=False,best=False,trajectories=False):
        if trajectories:raise CapabilityError("Recorded tensor trajectories are not exposed by this learner adapter")
        actor=self.actor
        if best:actor=deepcopy(actor);actor.load_state_dict(self.best_actor)
        seeds=list(self.config.validation_seeds if seeds is None else seeds)
        rows=self._run(seeds,actor=actor,identity_matrix=identity)[4]
        return dict(summary=summarize(rows),episodes=rows)

    def save(self,path):
        if self._pending_batch is not None:
            raise RuntimeError("Whole-update recovery cannot snapshot an unconsumed PPO batch")
        return super().save(path)

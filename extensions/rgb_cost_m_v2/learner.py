"""Add cumulative-cost and retention learning to the checked metric update."""
from copy import deepcopy
import torch
from torch.nn import functional as F
from stabilization_learning.rgb_stable.cost_q import PhysicalActionCostQ,visual_state,transition_cost,bellman_cost
from extensions.rgb_stable_m_v1.learner import build_learner as build_component_learner


def build_learner(adapter,cfg,device,settings):
    base=build_component_learner(adapter,cfg,device,settings)
    core=adapter.core
    from lac_vs.models import frozen_parameters

    class CostLearner(type(base)):
        def __init__(self):
            self.__dict__.update(base.__dict__)
            self.q=PhysicalActionCostQ().to(device)
            self.target_q=deepcopy(self.q).requires_grad_(False)
            self.target_metric=deepcopy(self.metric).requires_grad_(False)
            self.qopt=torch.optim.Adam(self.q.parameters(),lr=settings['q_lr'])
            self.features_frozen=False; self.counts['q']=0

        def zero_all(self):
            super().zero_all()
            if hasattr(self,'q'): self.q.zero_grad(set_to_none=True)

        def representation(self,b,warmup=False):
            if self.features_frozen: raise RuntimeError('The frozen visual coordinates cannot be updated')
            return super().representation(b,warmup=warmup)

        def freeze_features(self):
            self.zero_all()
            self.features_frozen=True
            for model in (self.ae,self.shadow,self.critics): model.requires_grad_(False).eval()
            self.clear_cache()

        @staticmethod
        def state(b,next=False):
            return visual_state(b['en'] if next else b['e'],b['goal'],b['j'],b['action'] if next else b['previous'])

        def prepared(self,b):
            if torch.is_tensor(b.get('e')): return {k:v.detach() for k,v in b.items()}
            return super().prepared(b)

        def q_update(self,b):
            if not self.features_frozen: raise RuntimeError('Q learning requires frozen visual coordinates')
            self.zero_all()
            state=self.state(b); next_state=self.state(b,next=True)
            with torch.no_grad():
                target_m=self.target_metric(b['goal'])[1]
                next_u=core.policy(b['en'],b['j'],target_m)[0]
                next_q,_=self.target_q(next_state,next_u)
                target=bellman_cost(b['cost'],b['failure'],next_q.amax(-1),self.settings['gamma'])
                if not bool(((target>=0)&(target<=32.+1e-5)).all()):
                    raise FloatingPointError('Cost target violated its analytical bound')
            predictions,hold=self.q(state,b['action'])
            value_loss=F.smooth_l1_loss(predictions,target[:,None].expand_as(predictions))
            inside=b['inside'].bool()
            hold_loss=F.binary_cross_entropy_with_logits(hold[inside],b['leaves'][inside].float()) if bool(inside.any()) else hold.sum()*0
            loss=value_loss+.1*hold_loss
            loss.backward(); norm=torch.nn.utils.clip_grad_norm_(self.q.parameters(),1.,error_if_nonfinite=True)
            if not bool(torch.isfinite(loss)): raise FloatingPointError('Nonfinite cost-Q loss')
            self.qopt.step(); self.counts['q']+=1
            with torch.no_grad():
                tau=self.settings['target_tau']
                for target_param,current in zip(self.target_q.parameters(),self.q.parameters()): target_param.lerp_(current,tau)
                for target_param,current in zip(self.target_metric.parameters(),self.metric.parameters()): target_param.lerp_(current,tau)
            return dict(total=float(loss),q_loss=float(value_loss),hold_loss=float(hold_loss),gradient_norm=float(norm),
                predicted_cost=float(predictions.mean()),target_cost=float(target.mean()),
                q_disagreement=float((predictions[:,0]-predictions[:,1]).abs().mean()),
                q_max=float(predictions.max()),failure_samples=int(b['failure'].sum()),
                truncated_bootstrap_samples=int(b['truncated'].sum()),inside_samples=int(inside.sum()))

        def objectives(self,b):
            physical,auxiliary,components,u,m,logs=super().objectives(b)
            with frozen_parameters(self.q):
                values,hold=self.q(self.state(b),u)
                cost=values.amax(-1).mean()
                retain=(torch.sigmoid(hold)*b['inside'].float()).mean()
            # Cost-to-go is the main performance signal; exact P0 progress and
            # retention constraints remain the nonlinear acceptance gate.
            main=cost+.1*physical+.05*retain
            logs.update(physical_progress=float(physical),predicted_cost=float(cost),
                hold_probability=float(retain),q_disagreement=float((values[:,0]-values[:,1]).abs().mean()))
            return main,auxiliary,components,u,m,logs

        def metric_update(self,b):
            if not self.features_frozen: raise RuntimeError('M stage requires fixed visual coordinates')
            result=super().metric_update(b)
            if any(p.grad is not None for p in self.q.parameters()): raise RuntimeError('M update changed Q gradients')
            return result

        def payload(self):
            result=super().payload()
            result.update({k:getattr(self,k).state_dict() for k in ('q','target_q','target_metric','qopt')})
            result['features_frozen']=self.features_frozen
            result['metric_accepted']=self.metric_accepted
            return result

        def restore(self,payload):
            super().restore(payload)
            for name in ('q','target_q','target_metric','qopt'): getattr(self,name).load_state_dict(payload[name])
            self.features_frozen=payload['features_frozen']
            self.metric_accepted=payload.get('metric_accepted',0)
            for model in (self.ae,self.shadow,self.critics): model.requires_grad_(not self.features_frozen).eval()

    return CostLearner()

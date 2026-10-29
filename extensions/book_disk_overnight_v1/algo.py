"""Four independently callable controller-model routes for fixed V=x²/2."""
from dataclasses import dataclass
import numpy as np
from scipy.special import logsumexp
from scipy.optimize import nnls
from .env import value


ROUTES = ("known_model", "direct_D", "identified_model", "GMM")


def diagonal_gmm(z, components, seed):
    """Deterministic diagonal-covariance Gaussian-mixture EM on normalized data."""
    rng=np.random.default_rng(seed)
    means=z[rng.choice(len(z),components,replace=False)].copy()
    covariance=np.tile(np.var(z,axis=0)+1e-5,(components,1))
    weights=np.full(components,1./components)
    previous=-np.inf;converged=False
    for iteration in range(1,121):
        difference=z[:,None,:]-means[None,:,:]
        logits=-.5*(np.sum(difference*difference/covariance[None,:,:],axis=-1)+
                     np.log(covariance).sum(-1)[None,:]+2*np.log(2*np.pi))+np.log(weights)[None,:]
        normalizer=logsumexp(logits,axis=1)
        h=np.exp(logits-normalizer[:,None]);mass=np.maximum(h.sum(axis=0),1e-12)
        lower_bound=float(normalizer.mean())
        means=h.T@z/mass[:,None]
        difference=z[:,None,:]-means[None,:,:]
        covariance=np.einsum('nk,nkd->kd',h,difference*difference)/mass[:,None]+1e-5
        weights=mass/mass.sum()
        if np.isfinite(previous) and abs(lower_bound-previous)<1e-5:
            converged=True;break
        previous=lower_bound
    # Recompute responsibilities under the final fitted mixture.
    difference=z[:,None,:]-means[None,:,:]
    logits=-.5*(np.sum(difference*difference/covariance[None,:,:],axis=-1)+
                 np.log(covariance).sum(-1)[None,:]+2*np.log(2*np.pi))+np.log(weights)[None,:]
    h=np.exp(logits-logsumexp(logits,axis=1)[:,None])
    return dict(means=means,covariances=covariance,weights=weights),h,dict(
        EM_iterations=iteration,EM_converged=converged,EM_lower_bound=lower_bound)


def information(features):
    gram = features.T@features/max(1,len(features))
    eig = np.linalg.eigvalsh(gram)
    singular = np.linalg.svd(features, compute_uv=False)
    threshold = max(features.shape)*np.finfo(float).eps*singular[0] if len(singular) else 0.
    rank = int(np.count_nonzero(singular>threshold))
    return dict(rank=rank, dimensions=features.shape[1], minimum_eigenvalue=float(eig[0]),
                maximum_eigenvalue=float(eig[-1]),
                condition_number=float(eig[-1]/eig[0]) if eig[0]>max(eig[-1]*1e-14,0.) else None,
                eigenvalues=eig.tolist(), finite_window_information=True)


@dataclass
class LearnedModel:
    route: str
    radius: float
    action_max: float
    parameters: dict

    def memberships(self, x, action):
        shape = np.broadcast_shapes(np.shape(x), np.shape(action))
        x,a = np.broadcast_arrays(x,action)
        z=np.stack([x/self.radius,a/self.action_max],-1).reshape(-1,2)
        p=self.parameters
        diff=z[:,None,:]-p["means"][None,:,:]
        log=-.5*(np.sum(diff*diff/p["covariances"][None,:,:],axis=-1)+
                 np.log(p["covariances"]).sum(-1)[None,:]+2*np.log(2*np.pi))
        log+=np.log(p["weights"])[None,:]
        return np.exp(log-logsumexp(log,axis=-1)[:,None]).reshape(*shape,-1)

    def increment(self, x, action, env=None):
        if self.route == "known_model":
            if env is None:raise ValueError("Known-model route requires the declared evaluation plant")
            return env.config.dt*env.gain(x,action)*action
        if self.route == "identified_model":
            return self.parameters["coefficient"]*np.asarray(action)
        if self.route == "GMM":
            return np.asarray(action)*np.sum(self.memberships(x,action)*self.parameters["local_gains"],axis=-1)
        raise ValueError("Direct D does not invent a successor-state model")

    def predict_D(self,x,action,env=None):
        if self.route == "direct_D":
            b=self.parameters["beta"]
            return b[0]*np.asarray(x)*action+b[1]*np.asarray(action)**2
        delta=self.increment(x,action,env)
        return np.asarray(x)*delta+.5*delta**2

    def save(self,path):
        np.savez_compressed(path,route=np.array(self.route),radius=self.radius,
                            action_max=self.action_max,**self.parameters)


def fit(route, data, config, seed=41, gmm_components=4):
    x,a,xn=data["observed_x"],data["action"],data["observed_next_x"]
    if route not in ROUTES:raise ValueError(route)
    d= xn-x
    p={};diagnostics={}
    if route == "direct_D":
        scales=np.array([config.radius*config.action_max,config.action_max**2])
        X=np.stack([x*a,a*a],axis=-1)/scales
        p["beta"]=np.linalg.lstsq(X,value(xn)-value(x),rcond=None)[0]/scales
        diagnostics["information"]=information(X)
        diagnostics["structural_form"]="beta1*x*a + beta2*a^2; target zero is exact"
        if config.response=='constant':
            truth=np.array([config.kappa*config.dt,.5*(config.kappa*config.dt)**2])
            diagnostics['known_true_parameters']=truth.tolist()
            diagnostics['fitted_parameters']=p['beta'].tolist()
            diagnostics['identifiable_parameter_error']=float(np.linalg.norm(p['beta']-truth)) if diagnostics['information']['rank']==2 else None
            diagnostics['parameter_identifiable']=diagnostics['information']['rank']==2
    elif route == "identified_model":
        X=(a/config.action_max)[:,None]
        p["coefficient"]=np.asarray(max(0.,float(a@d/max(a@a,1e-30))))
        diagnostics["information"]=information(X)
        diagnostics["structural_form"]="nonnegative scalar increment gain; zero input gives zero increment"
        if config.response=='constant':
            diagnostics['known_true_parameters']=[config.kappa*config.dt]
            diagnostics['fitted_parameters']=[float(p['coefficient'])]
            diagnostics['identifiable_parameter_error']=abs(float(p['coefficient'])-config.kappa*config.dt)
            diagnostics['parameter_identifiable']=True
    elif route == "GMM":
        z=np.stack([x/config.radius,a/config.action_max],axis=-1)
        p,h,em=diagonal_gmm(z,min(gmm_components,max(1,len(x)//8)),seed)
        design=h*a[:,None]/config.action_max
        coeff=nnls(design,d)[0]/config.action_max
        p['local_gains']=coeff
        diagnostics.update(information=information(design),**em,
            structural_form="a * sum_l h_l(x,a) b_l; b_l>=0; diagonal-covariance GMM")
    else:
        diagnostics["information"] = dict(known_dynamics=True, training_transitions_used=0)
    model=LearnedModel(route,config.radius,config.action_max,p)
    return model,diagnostics


def calibrate(model, calibration, env):
    """Empirical max residual from a separate calibration split, not a bound."""
    x,a=calibration["x"],calibration["action"]
    if model.route=="known_model":
        return dict(coefficient=0.,kind="known exact model",calibration_count=len(x))
    if model.route=="direct_D":
        residual=np.abs(model.predict_D(x,a)-calibration["true_D"])
        scale=np.abs(x)*a+a*a
        coefficient=float(np.max(residual/np.maximum(scale,1e-30)))
        kind="empirical |D-Dhat|/(|x|a+a^2) maximum"
    else:
        residual=np.abs(model.increment(x,a)-(calibration["next_x"]-x))
        coefficient=float(np.max(residual/np.maximum(a,1e-30)))
        kind="empirical increment error/action maximum"
    return dict(coefficient=coefficient,kind=kind,calibration_count=len(x),
                coverage_claim="observed calibration maximum only; no uniform-region guarantee")


def prediction_with_margin(model,x,a,env,calibration,use_margin):
    predicted=model.predict_D(x,a,env)
    if model.route=="direct_D":
        epsilon=(calibration["coefficient"]*(np.abs(x)*a+a*a)) if use_margin else np.zeros_like(predicted)
        # Domain preservation for D-only relies on the explicitly supplied
        # physical gain upper bound, not a fabricated state predictor.
        lower=x
        upper=x+env.config.dt*env.config.kappa_max*a
    else:
        delta=model.increment(x,a,env)
        uncertainty=calibration["coefficient"]*a if use_margin else np.zeros_like(predicted)
        epsilon=np.abs(x+delta)*uncertainty+.5*uncertainty**2
        lower=x+delta-uncertainty;upper=x+delta+uncertainty
    return predicted,epsilon,lower,upper


def choose_action(model,x,env,calibration,mode="mean",action_candidates=65):
    """Least effort certified-by-estimate action; report every empty set.

    If the estimated feasible set is empty, use the safe action with the
    smallest predicted upper D and record fallback=True. No failed feasibility
    test is silently converted into a certificate.
    """
    x=np.asarray(x,float).reshape(-1)
    fractions=np.linspace(0.,1.,action_candidates)
    actions=env.safe_upper(x)[:,None]*fractions
    current=np.broadcast_to(x[:,None],actions.shape)
    predicted,epsilon,lower,upper=prediction_with_margin(model,current,actions,env,calibration,mode!="mean")
    rate=.5*(1.-max(0.,1.-env.config.dt)**2)
    residual=predicted+epsilon+rate*current**2
    feasible=(residual<=1e-14)&(lower>=-env.config.radius-1e-12)&(upper<=1e-12)
    any_feasible=feasible.any(axis=1)
    first=feasible.argmax(axis=1)
    fallback=np.argmin(predicted+epsilon,axis=1)
    index=np.where(any_feasible,first,fallback)
    rows=np.arange(len(x))
    return dict(action=actions[rows,index],predicted_D=predicted[rows,index],
                margin=epsilon[rows,index],feasible=any_feasible,
                predicted_lower=lower[rows,index],predicted_upper=upper[rows,index],
                required_decrease=rate*x*x)

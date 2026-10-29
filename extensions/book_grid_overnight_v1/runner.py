"""Freeze a complete finite-MDP experiment, then independently evaluate it."""
from pathlib import Path
from fractions import Fraction
import csv
import hashlib
import json
import math
import time
from contextlib import nullcontext
import numpy as np
from .env import GridEnv
from .algo import hitting_certificate, policy_iteration, exact_audit


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def wilson(successes, n):
    z = 1.959963984540054; p = successes/n; d = 1+z*z/n
    middle = (p+z*z/(2*n))/d
    radius = z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return [middle-radius, middle+radius]


def run(output, seed=41, episodes=10000, max_minutes=15, smoke=False):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    started = time.time(); deadline = started+60*max_minutes
    def guard():
        if time.time() >= deadline or (output/'STOP').exists():
            raise InterruptedError('Grid experiment STOP/deadline')
    sources = [Path(__file__), Path(__file__).with_name('env.py'), Path(__file__).with_name('algo.py'),
        Path('E:/rfly-books/Rfly-Stable-Learning-Book_Reviewed_20261022/ch9/chapter9.tex')]
    protocol = dict(schema='book_grid_overnight_v1', seed=seed, episodes=episodes, horizon=200,
        gamma=.95, main_probability=.8, contraction_factors=[.25, .5, 1., 2., 4.],
        main_probability_sweep=[.6, .7, .8, .9], horizons=[50, 100, 200, 400],
        policy_stopping_tolerance=1e-10, certificate_decimals=8, maximum_minutes=max_minutes,
        started_unix=started, sources={str(p):digest(p) for p in sources},
        task_ids=['E10','E04'], smoke=smoke, architecture=dict(env='GridEnv', algo='hitting_certificate/policy_iteration', runner='run'),
        scope='Reviewed 95-state MDP; fixed certificate and model-based reward policy iteration; no learned visual metric',
        sampling='Shared frozen initial states; independent transition streams per policy; single seed, within-run Monte Carlo Wilson intervals',
        evidence='Floating reward evaluation; exact rational support and drift audit of declared rationalized policies')
    write(output/'protocol.json', protocol)
    results = {}; final = dict(status='running')
    with nullcontext():
        try:
            env = GridEnv(); certificate = hitting_certificate(env)
            if certificate['sizes'] != [75,71,65,61,59,59] or len(certificate['transient']) != 55:
                raise AssertionError('Reviewed domain mismatch')
            if abs(certificate['h'].max()-231.8984930141)>1e-7 or certificate['kappa'] != Fraction(1,464):
                raise AssertionError('Reviewed hitting certificate mismatch')
            print(json.dumps(dict(stage='domain_certificate', domain=len(certificate['domain']), max_h=float(certificate['h'].max()))), flush=True)
            unconstrained = policy_iteration(env, evaluation_starts=certificate['transient'])
            constrained = policy_iteration(env, certificate, certificate['kappa'])
            if not constrained['feasible']:
                raise AssertionError('Nominal fixed certificate unexpectedly infeasible')
            policies = dict(baseline=certificate['baseline'], reward=unconstrained['policy'], constrained=constrained['policy'])
            audit = exact_audit(env, certificate, policies['constrained'], certificate['kappa'])
            baseline_audit = exact_audit(env, certificate, policies['baseline'], certificate['kappa'])
            if not audit['passed'] or not baseline_audit['passed']:
                raise AssertionError('Rational certificate audit failed')
            write(output/'CERTIFICATE_AUDIT.json', dict(constrained=audit, baseline=baseline_audit,
                kappa=str(certificate['kappa']), region_sizes=certificate['sizes']))
            compiled = []
            for main in protocol['main_probability_sweep']:
                guard(); model = GridEnv(main); cert = hitting_certificate(model)
                for factor in protocol['contraction_factors']:
                    solution = policy_iteration(model, cert, cert['kappa']*Fraction(str(factor)))
                    item = dict(main_probability=main, factor=factor, kappa=float(cert['kappa'])*factor,
                                domain_size=len(cert['domain']), feasible=solution['feasible'])
                    if solution['feasible']:
                        compiled.append((item, model, cert, solution['policy']))
                    else:
                        compiled.append((dict(item, infeasible_state=solution['infeasible_state']), model, cert, None))
            arrays = dict(transition_kernel=env.kernel, expected_reward=env.reward, states=np.asarray(env.states),
                domain=np.asarray(certificate['domain']), starts=np.asarray(certificate['transient']),
                certificate=certificate['value'], **{f'policy_{k}':v for k,v in policies.items()})
            np.savez_compressed(output/'frozen_policies.npz', **arrays)
            np.savez_compressed(output/'sweep_policies.npz', **{f'p_{i}':policy for i,(_,_,_,policy) in enumerate(compiled) if policy is not None})
            histories = {name: sol['history'] for name,sol in [('reward',unconstrained),('constrained',constrained)]}
            write(output/'POLICY_ITERATION.json', histories)
            initial = np.random.default_rng(seed+1000).choice(certificate['transient'], episodes)
            np.save(output/'test_initial_states.npy', initial)
            write(output/'FROZEN.json', dict(protocol_sha256=digest(output/'protocol.json'),
                policy_sha256=digest(output/'frozen_policies.npz'), starts_sha256=digest(output/'test_initial_states.npy'),
                sweep_policy_sha256=digest(output/'sweep_policies.npz'),
                frozen_unix=time.time(), selected_by='Exact model policy iteration; no test-based choice'))
            for index,(name,policy) in enumerate(policies.items()):
                guard()
                propagation = env.propagate(policy, certificate['transient'], 400)
                trace = env.sample(policy, initial, 200, seed+2000+index)
                np.savez_compressed(output/f'test_{name}.npz', **trace, model_propagation=propagation)
                with (output/f'episodes_{name}.csv').open('x', newline='', encoding='utf-8') as stream:
                    writer=csv.writer(stream); writer.writerow(['episode','initial_state','goal','trap','timeout','steps','discounted_return'])
                    for i in range(episodes):
                        writer.writerow([i,initial[i],int(trace['goal'][i]),int(trace['trap'][i]),int(trace['timeout'][i]),
                                         int(trace['steps'][i]),float(trace['discounted_return'][i])])
                results[name] = dict(episodes=episodes, successes=int(trace['goal'].sum()), traps=int(trace['trap'].sum()),
                    timeouts=int(trace['timeout'].sum()), success_ci95=wilson(int(trace['goal'].sum()),episodes),
                    success_rate=float(trace['goal'].mean()), mean_discounted_return=float(trace['discounted_return'].mean()),
                    mean_hitting_time_success=float(trace['steps'][trace['goal']].mean()) if trace['goal'].any() else None,
                    model200=dict(zip(('goal','trap','timeout','discounted_return'), map(float,propagation[199]))),
                    model_horizons={str(h): dict(zip(('goal','trap','timeout','discounted_return'), map(float,propagation[h-1]))) for h in protocol['horizons']})
                print(json.dumps(dict(stage='evaluation',method=name,**results[name])),flush=True)
            expected={'constrained':(.966498,0,.033502,5.271591),'reward':(.998487,1.27654e-5,.001501,5.438364)}
            parity={k:bool(np.allclose(list(results[k]['model200'].values()),expected[k],atol=1e-6,rtol=0)) for k in expected}
            write(output/'BOOK_MODEL_COMPARISON.json',dict(expected=expected,matched_printed_precision=parity,
                interpretation='Historical Monte Carlo counts need not match the new independent seed41 streams'))
            sweeps=[]
            for item,model,cert,policy in compiled:
                guard()
                if policy is not None:
                    propagation=model.propagate(policy,cert['transient'],200)
                    proof=exact_audit(model,cert,policy,cert['kappa']*Fraction(str(item['factor'])))
                    item.update(model_goal=float(propagation[-1,0]), model_trap=float(propagation[-1,1]),
                                model_return=float(propagation[-1,3]), rational_audit_passed=proof['passed'])
                sweeps.append(item)
            write(output/'SENSITIVITY.json',sweeps)
            write(output/'RESULTS.json',dict(schema='book_grid_results_v1',seed=seed,methods=results,
                book_comparison=parity, single_seed=True, independent_episodes=True,
                limitations=['Exact support/drift checks concern the declared finite kernel and domain.',
                             'Policy value comparisons use floating point; not an exact optimality theorem.',
                             'A model-based finite MDP experiment does not replace the E13 continuous-application RL ablation.']))
            plot(output,env,certificate,policies,results,histories,sweeps)
            final=dict(status='complete', book_model_comparison=parity, methods=list(results),
                       source_integrity=all(digest(p)==h for p,h in protocol['sources'].items()))
        except BaseException as error:
            final=dict(status='failed',error=repr(error)); raise
        finally:
            final.update(finished_unix=time.time(),wall_seconds=time.time()-started)
            write(output/'FINAL_STATE.json',final)
    return final


def plot(output, env, cert, policies, results, histories, sweeps):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
    figures=output/'figures'; figures.mkdir()
    def save(fig,name):
        fig.tight_layout();fig.savefig(figures/(name+'.png'),dpi=300,bbox_inches='tight')
        fig.savefig(figures/(name+'.pdf'),bbox_inches='tight');plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(13,4))
    for ax,(name,policy) in zip(axes,policies.items()):
        from matplotlib.patches import Rectangle
        values=np.full((10,10),np.nan)
        for i in cert['domain']:
            x,y=env.states[i];values[y,x]=cert['value'][i]
        image=ax.imshow(values,origin='lower',cmap='viridis')
        for x,y in env.obstacles:
            ax.add_patch(Rectangle((x-.5,y-.5),1,1,color='black'))
        for i,(x,y) in enumerate(env.states):
            if i not in cert['domain'] and not env.trap[i]:
                ax.add_patch(Rectangle((x-.5,y-.5),1,1,facecolor='lightgray',hatch='///',alpha=.5))
        for i,(x,y) in enumerate(env.states):
            if env.trap[i]:ax.scatter(x,y,c='red',marker='x',s=24)
            elif env.goal[i]:ax.scatter(x,y,c='white',marker='*',s=50)
            elif i in cert['domain']:
                dx,dy=env.moves[int(policy[i].argmax())];ax.arrow(x,y,.28*dx,.28*dy,head_width=.16,color='white')
        ax.set(title=name,xlabel='x',ylabel='y');fig.colorbar(image,ax=ax,label='fixed V')
    save(fig,'policy_and_certificate')
    fig,axes=plt.subplots(1,2,figsize=(10,4));names=list(results)
    for j,name in enumerate(names):
        r=results[name];lo,hi=r['success_ci95']
        axes[0].errorbar(j,r['success_rate'],yerr=[[r['success_rate']-lo],[hi-r['success_rate']]],fmt='o',capsize=4,color='tab:blue')
        axes[0].scatter(j,r['model200']['goal'],marker='x',color='tab:orange',s=55)
        axes[1].bar(j,r['mean_discounted_return'],color='tab:blue',alpha=.7)
        axes[1].scatter(j,r['model200']['discounted_return'],marker='x',color='black')
    axes[0].set(xticks=range(3),xticklabels=names,ylabel='goal probability',title='MC 95% Wilson and model (x)')
    axes[1].set(xticks=range(3),xticklabels=names,ylabel='discounted return',title='same 200-step horizon')
    save(fig,'model_vs_independent_rollouts')
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for name,rows in histories.items():
        axes[0].plot([r['iteration'] for r in rows],[r['mean_value'] for r in rows],'o-',label=name)
        axes[1].semilogy([r['iteration'] for r in rows],[max(1e-15,r['bellman_improvement_residual']) for r in rows],'o-',label=name)
    axes[0].set(xlabel='policy evaluation',ylabel='mean training-model value');axes[1].set(xlabel='policy evaluation',ylabel='Bellman improvement residual')
    for ax in axes:ax.legend();ax.grid(alpha=.2)
    save(fig,'policy_iteration_curves')
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for main in sorted({s['main_probability'] for s in sweeps}):
        rows=[r for r in sweeps if r['main_probability']==main and r['feasible']]
        axes[0].plot([r['factor'] for r in rows],[r['model_goal'] for r in rows],'o-',label=f'forward p={main}')
        axes[1].plot([r['factor'] for r in rows],[r['model_return'] for r in rows],'o-',label=f'forward p={main}')
    axes[0].set(xlabel='contraction multiplier',ylabel='200-step model goal probability');axes[1].set(xlabel='contraction multiplier',ylabel='200-step model return')
    for ax in axes:ax.legend();ax.grid(alpha=.2)
    save(fig,'contraction_noise_tradeoff')

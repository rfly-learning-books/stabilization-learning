"""Goal relabeling of real within-episode transitions; physics/actions unchanged."""
import torch


def relabel_episode(transitions,config):
    """Use positive achieved progress as an additional scalar training goal.

    Each row is a single-lane (obs, action, next_obs, info, terminated) tuple.
    Never cross resets. Q's reference position/orientation is unchanged.
    Recompute D, signed g, total-goal feature and terminal hold for this goal.
    """
    if not transitions:return []
    task=int(transitions[0][0]['task'].argmax(-1)[0])
    turn=task>=4;scale=1. if turn else .1
    minimum=config.angle_min if turn else config.distance_min
    maximum=config.angle_max if turn else config.distance_max
    goal=float(transitions[-1][3]['progress'][0])
    if not minimum<=goal<=maximum:return []
    if any(int(row[0]['task'].argmax(-1)[0])!=task for row in transitions):
        raise ValueError('Cannot relabel transitions across task switches')
    result=[];hold=0
    tolerance=config.angle_tolerance if turn else config.position_tolerance
    for obs,action,nxt,info,terminated in transitions:
        obs={k:v.clone() for k,v in obs.items()};nxt={k:v.clone() for k,v in nxt.items()}
        old_amount=obs['s'][:,16].clone()
        progress_before=old_amount-obs['g'][:,0]
        progress_after=info['progress']
        obs['g'][:,0]=goal-progress_before;nxt['g'][:,0]=goal-progress_after
        obs['s'][:,16]=goal;nxt['s'][:,16]=goal
        info={k:v.clone() for k,v in info.items()}
        before=.5*(obs['g'][:,0]/scale).square();after=.5*(nxt['g'][:,0]/scale).square()
        info['g']=nxt['g'][:,0].clone();info['energy']=after
        info['energy_before']=before;info['d_target']=after-before
        info['progress_target']=(progress_after-progress_before)/scale
        side_ok=True if turn else bool(
            ((info['transverse_error'].abs()<=config.position_tolerance)&(info['angle_change'].abs()<=config.angle_tolerance))[0])
        settled=bool(((info['speed']<=config.speed_tolerance)&(info['omega'].abs()<=config.omega_tolerance))[0])
        eligible=bool((nxt['g'].abs()<=tolerance).all()) and side_ok and settled
        hold=hold+1 if eligible else 0
        info['eligible']=torch.full_like(terminated,eligible)
        info['success']=torch.full_like(terminated,hold>=config.hold_steps)
        info['hold']=torch.full_like(terminated,hold,dtype=torch.long)
        info['backward']=torch.relu(-info['progress_target'])
        info['overshoot']=torch.relu(-info['g']/scale)
        info['d_violation']=torch.relu(info['d_target']+config.contraction*.05*before)+info['backward']+info['overshoot']
        if 'strict_success' in info:
            side_path_ok=torch.ones_like(terminated) if turn else ((info['max_side_error']<=config.position_tolerance)&(info['max_push_angle_error']<=config.angle_tolerance))
            info['strict_success']=info['success']&side_path_ok&(info['backward_steps']==0)
        done=info['fail'].bool() | info['success']
        info['q_discount']=torch.full_like(info['q_cost'],1. if turn else .99)
        info['q_terminal']=done | (info.get('timeout',torch.zeros_like(done)) if turn else torch.zeros_like(done))
        result.append((obs,action.clone(),nxt,info,done))
        if done.any():break
    return result

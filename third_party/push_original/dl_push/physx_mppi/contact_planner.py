"""Continuous-contact MPPI, with a separate live scene and verified execution prefix.

Geometry supplies collision-free approach waypoints, not executed actions. All
velocity proposals are rolled out in PhysX; no learned policy or MJX is used.
"""
import heapq
import math
import numpy as np
import torch
from shapely.geometry import box, LineString, Point
from shapely.ops import unary_union
from .planner import snapshot, restore, tree_copy, replay_check


def transfer_root(live, search):
    state = snapshot(live)
    def expand(value):
        if isinstance(value, dict):
            return {k: expand(v) for k, v in value.items()}
        if isinstance(value, torch.Tensor):
            if value.ndim and value.shape[0] == 1:
                return value.expand(search.num_envs, *value.shape[1:]).clone()
            return value.clone()
        return value
    return expand(state)


def approach_path(spec, start, center, radius, push_direction=(1., 0.)):
    """Route to the face opposite a world push direction.

    The geometry is rotated into a canonical frame where the desired push is
    +x.  This keeps the same concave-shape visibility graph valid for -x, +y,
    and -y without inventing a directional action prior.
    """
    from shapely.affinity import rotate
    push_direction = np.asarray(push_direction, dtype=float)
    norm = np.linalg.norm(push_direction)
    if norm <= 0 or not np.isfinite(push_direction).all():
        raise ValueError('Invalid push direction')
    push_direction /= norm
    phi = math.atan2(push_direction[1], push_direction[0])
    c, s = math.cos(phi), math.sin(phi)
    to_local = np.array([[c, s], [-s, c]])
    to_world = np.array([[c, -s], [s, c]])
    start_local = to_local @ (np.asarray(start, dtype=float) - np.asarray(center, dtype=float))
    body = unary_union([box(cx-hx, cy-hy, cx+hx, cy+hy) for (cx,cy),(hx,hy) in spec.boxes])
    body = rotate(body, -phi * 180. / math.pi, origin=(0., 0.), use_radians=False)
    com = body.centroid
    # A small shift is needed when the COM ray only grazes a concave corner (T).
    obstacle = body.buffer(radius+.001, join_style=2)
    candidates = []
    for y in np.arange(com.y-.015, com.y+.0151, .0005):
        ray = LineString([(-.4,y),(.4,y)])
        hit = obstacle.intersection(ray)
        if hit.is_empty:
            continue
        x = hit.bounds[0]
        body_hit = body.intersection(ray)
        if body_hit.is_empty or abs(body_hit.bounds[0]-(x+radius+.001))>.002:
            continue  # Reject a buffered concave corner masquerading as a flat face.
        # Keep the approach point away from a horizontal obstacle edge.
        nearby = [obstacle.intersection(LineString([(-.4,y+dy),(.4,y+dy)])) for dy in (-.001,.001)]
        if any(h.is_empty or abs(h.bounds[0]-x)>.002 for h in nearby):
            continue
        candidates.append((abs(y-com.y), np.array([x-.002,y])))
    if not candidates:
        raise RuntimeError('No usable left contact face')
    target = min(candidates, key=lambda item:item[0])[1]
    # Vertices lie just outside the collision envelope to permit boundary routes.
    routing = obstacle.buffer(.001, join_style=2)
    vertices = np.asarray(routing.exterior.coords)[:-1]
    nodes = [start_local, target, *vertices]
    interior = obstacle.buffer(-1e-6)
    if interior.contains(Point(start_local)):
        raise RuntimeError('Initial pusher lies inside the conservative routing obstacle')
    graph = [[] for _ in nodes]
    for i in range(len(nodes)):
        for j in range(i):
            if not LineString([nodes[i],nodes[j]]).intersects(interior):
                length = float(np.linalg.norm(nodes[i]-nodes[j]))
                graph[i].append((j,length));graph[j].append((i,length))
    dist = [math.inf]*len(nodes);dist[0]=0;previous={};queue=[(0.,0)]
    while queue:
        cost,i=heapq.heappop(queue)
        if cost!=dist[i]:continue
        if i==1:break
        for j,length in graph[i]:
            if cost+length<dist[j]:
                dist[j]=cost+length;previous[j]=i;heapq.heappush(queue,(dist[j],j))
    if not math.isfinite(dist[1]):raise RuntimeError('No collision-free approach route')
    route=[1]
    while route[-1]!=0:route.append(previous[route[-1]])
    return [to_world @ nodes[i] + np.asarray(center, dtype=float) for i in reversed(route[:-1])]


def approach_path_to_target(spec, start, center, target_local, radius):
    """Collision-free visibility route to a declared body-frame contact point."""
    from shapely.affinity import translate
    body=unary_union([box(cx-hx,cy-hy,cx+hx,cy+hy) for (cx,cy),(hx,hy) in spec.boxes])
    target_local=np.asarray(target_local,dtype=float)
    radial=target_local/np.linalg.norm(target_local)
    target=np.asarray(center,dtype=float)+target_local+radial*(radius+.002)
    # Round expansion matches the circular pusher at a corner. A mitred
    # square expansion would leave an artificial diagonal clearance gap.
    obstacle=translate(body.buffer(radius+.001,join_style=1),xoff=center[0],yoff=center[1])
    routing=obstacle.buffer(.001,join_style=1)
    vertices=np.asarray(routing.exterior.coords)[:-1]
    nodes=[np.asarray(start,dtype=float),target,*vertices]
    interior=obstacle.buffer(-1e-6)
    if interior.contains(Point(start)):raise RuntimeError('Initial pusher lies inside rotation routing obstacle')
    graph=[[] for _ in nodes]
    for i in range(len(nodes)):
        for j in range(i):
            segment=LineString([nodes[i],nodes[j]])
            if not segment.crosses(interior) and not segment.within(interior):
                length=float(np.linalg.norm(nodes[i]-nodes[j]));graph[i].append((j,length));graph[j].append((i,length))
    dist=[math.inf]*len(nodes);dist[0]=0.;previous={};queue=[(0.,0)]
    while queue:
        cost,i=heapq.heappop(queue)
        if cost!=dist[i]:continue
        if i==1:break
        for j,length in graph[i]:
            if cost+length<dist[j]:dist[j]=cost+length;previous[j]=i;heapq.heappush(queue,(dist[j],j))
    if not math.isfinite(dist[1]):raise RuntimeError('No collision-free rotation approach route')
    route=[1]
    while route[-1]!=0:route.append(previous[route[-1]])
    return [nodes[i] for i in reversed(route[:-1])]


def slew_limit(actions, previous, limit=.12):
    result=[];last=previous.expand(actions.shape[0],-1)
    for desired in actions.transpose(0,1):
        last=last+(desired-last).clamp(-limit,limit)
        result.append(last)
    return torch.stack(result,1)


class ContactMPPI:
    def __init__(self, live, search, horizon=16, iterations=2, push_min=.35, push_max=.85,
                 rotation_inward=.10, rotation_drift_weight=100.):
        if live.num_envs!=1 or search.num_envs<8:
            raise ValueError('One untouched live lane and >=8 search lanes required')
        task = int(live.task[0])
        if live.config.reset_mode!='fixed_origin' or task not in range(6):
            raise ValueError('Continuous-contact prototype supports fixed-origin tasks 0..5')
        self.live=live;self.search=search;self.horizon=horizon;self.iterations=iterations
        if not (0.0 <= push_min <= push_max <= 1.0):
            raise ValueError('Invalid forward action range')
        self.push_min=float(push_min);self.push_max=float(push_max)
        if not (0.0 <= rotation_inward <= 1.0) or not np.isfinite(rotation_drift_weight) or rotation_drift_weight < 0:
            raise ValueError('Invalid rotation contact parameters')
        self.rotation_inward=float(rotation_inward)
        self.rotation_drift_weight=float(rotation_drift_weight)
        self.task=task
        if task < 4:
            self.approach_direction=np.asarray(((1.,0.),(-1.,0.),(0.,1.),(0.,-1.))[task],dtype=float)
            self.push_direction=self.approach_direction.copy()
        else:
            # Reach a high-leverage right-hand corner, then follow that contact
            # point on a body-centered arc.  Positive yaw is counter-clockwise.
            self.approach_direction=np.asarray((-1.,0.),dtype=float)
            tangent=np.asarray((0.,-1.) if task==4 else (0.,1.),dtype=float)
            self.rotation_contact_sign=1. if task==4 else -1.
            self.push_direction=np.asarray((-self.rotation_inward,0.),dtype=float)+tangent
            self.push_direction/=np.linalg.norm(self.push_direction)
        self.device=live.device;self.previous=torch.zeros(2,device=self.device)
        self.mean=torch.zeros(horizon,2,device=self.device);self.phase='approach';self.refining=False
        xy,_,_,_,p,_=live._physical()
        spec=live.raw._shape_specs[int(live.raw._active_shape_id[0])]
        center=xy[0].cpu().numpy();start=p[0].cpu().numpy();radius=float(live.raw.agent.pusher_radius)
        if task>=4:
            corners=np.asarray([[cx+sx*hx,cy+sy*hy] for (cx,cy),(hx,hy) in spec.boxes
                                for sx in (-1.,1.) for sy in (-1.,1.)])
            # Maximize right leverage while choosing the corner appropriate to
            # the requested torque direction.
            score=corners[:,0]+corners[:,1]
            self.contact_local=corners[int(np.argmax(score))]
            self.path=approach_path_to_target(spec,start,center,self.contact_local,radius)
            self.contact_offset=self.path[-1]-center
        else:
            self.path=approach_path(spec,start,center,radius,push_direction=self.approach_direction)
            self.contact_offset=self.path[-1]-center
        self.contact_radius=float(np.linalg.norm(self.contact_offset))
        self.waypoint=0

    def _update_phase(self):
        env=self.live;xy,_,v,_,p,_=env._physical();m=env.metrics()
        if self.phase=='approach':
            while self.waypoint<len(self.path)-1 and np.linalg.norm(p[0].cpu().numpy()-self.path[self.waypoint])<.0015:
                self.waypoint+=1
            if self.waypoint==len(self.path)-1 and np.linalg.norm(p[0].cpu().numpy()-self.path[-1])<.002:
                self.phase='push';self.mean.zero_()
                if self.task>=4:
                    # The geometric waypoint includes a routing clearance. Use
                    # the actual live contact radius at handoff for circle tracking.
                    gap=float((env.contact_gap()[0]*.1).detach().cpu())
                    offset=(p[0]-xy[0]).detach().cpu().numpy()
                    norm=max(float(np.linalg.norm(offset)),1e-6)
                    contact_radius=max(float(env.raw.agent.pusher_radius),norm-gap)
                    self.contact_offset=offset/norm*contact_radius
                    self.contact_radius=contact_radius
        if self.phase=='push' and bool(m['eligible'][0]):
            self.phase='hold';self.mean.zero_()

    @torch.no_grad()
    def evaluate(self, root, actions, phase):
        env=self.search;restore(env,root)
        initial=env.metrics();energy=initial['energy'].clone();q=torch.zeros_like(energy)
        costs=torch.zeros_like(energy);drift=torch.zeros_like(energy)
        valid=torch.ones_like(env.done);active=~env.done
        terminal=initial;steps=torch.zeros_like(env.steps);last=self.previous.expand(env.num_envs,-1)
        target=torch.as_tensor(self.path[self.waypoint],device=self.device,dtype=torch.float32)
        for action in actions.transpose(0,1):
            if not active.any():break
            env.done.zero_()
            _,_,term,trunc,info=env.step(torch.where(active[:,None],action,0.))
            xy,_,_,_,p,_=env._physical()
            side=((info['position_drift']/.02).square() if self.task>=4 else
                  (info['transverse_error']/.005).square()+(info['angle_change']/.035).square())
            if phase=='approach':
                cost=8*((p-target)/.05).square().sum(-1)+20*side+40*((xy-env.initial_xy)/.005).square().sum(-1)
                cost+=2*info['contact_force'].square()  # Align outside the body before making contact.
            else:
                gap=env.contact_gap()*.1
                moving_goal=(info['g']>.005).float()
                if self.task>=4:
                    angle=info['angle_change']
                    co,si=torch.cos(angle),torch.sin(angle)
                    offset=torch.as_tensor(self.contact_offset,device=self.device,dtype=xy.dtype)
                    tx=co*offset[0]-si*offset[1]
                    ty=si*offset[0]+co*offset[1]
                    target_contact=xy+torch.stack((tx,ty),-1)
                    track=(p-target_contact).square().sum(-1)
                    planning_energy=.5*(info['g']-.02).square()
                    cost=180*planning_energy+self.rotation_drift_weight*info['position_drift']
                    cost+=20*track/.02**2+2*(gap/.005).clamp_max(10)*moving_goal
                else:
                    planning_energy=.5*((info['g']-.002)/.1).square()
                    cost=100*planning_energy+3*side+2*(gap/.005).clamp_max(10)*moving_goal
                cost+=100*info['backward']+2*(action-last).square().sum(-1)
            costs+=torch.where(active,cost,0.)
            q+=torch.where(active,info['q_cost'],0.)
            drift+=torch.where(active,info['position_drift'],0.)
            energy=torch.where(active,info['energy'],energy)
            # Prefix safety uses the unchanged physical task tolerances. Full path
            # scores remain soft so the optimizer can recover small prediction errors.
            valid &= ~(active & info['fail'])
            terminal={k:torch.where(active,info[k],terminal.get(k,info[k])) for k in
                      ('g','transverse_error','angle_change','speed','omega','energy')}
            steps+=active.long();active&=~(term|trunc);last=action
        if phase=='approach':
            end=8*((p-target)/.05).square().sum(-1)
        else:
            progress=initial['g']-terminal['g']
            if self.task>=4:
                end=300*.5*(terminal['g']-.02).square()+self.rotation_drift_weight*drift
                # Drift is a secondary objective: every push prefix must still
                # make a small, signed angular advance. Otherwise zero action
                # wins trivially under a large drift weight.
                required=torch.minimum(initial['g']*.20,initial['g'].new_full(initial['g'].shape,.03))
                end+=10000*torch.relu(required-progress).square()
            else:
                end=300*.5*((terminal['g']-.002)/.1).square()+4*((terminal['transverse_error']/.005).square()+(terminal['angle_change']/.035).square())
            end+=.2*(terminal['speed']/.03).square()
            if self.refining:
                end+=2*torch.relu((terminal['g']-.004)/.001)
        score=costs/steps.clamp_min(1)+end+10000*(~valid).float()
        if not torch.isfinite(score).all():raise FloatingPointError('Nonfinite contact MPPI cost')
        return dict(score=score,valid=valid,d=energy-initial['energy'],q=q,terminal=terminal)

    @torch.no_grad()
    def __call__(self):
        self._update_phase();env=self.search;n=env.num_envs;H=self.horizon
        root=transfer_root(self.live,env);phase=self.phase
        current_metrics=self.live.metrics()
        refine_low, refine_high = ((.02,.12) if self.task>=4 else (.005,.010))
        self.refining=(phase=='push' and refine_low<float(current_metrics['g'][0])<refine_high
                       and float(current_metrics['speed'][0])<.02)
        mean=self.mean.clone()
        if phase=='approach':
            _,_,_,_,p,_=self.live._physical()
            delta=torch.as_tensor(self.path[self.waypoint],device=self.device,dtype=torch.float32)-p[0]
            # A stopping trajectory reaches the waypoint within the horizon.
            # Constant-velocity proposals otherwise slow asymptotically at every corner.
            remainder=delta.clone();nominal=[]
            for _ in range(H):
                velocity=(remainder/(.3*.15)).clamp(-.45,.45)
                nominal.append(velocity);remainder=remainder-.3*self.live.dt*velocity
            mean=torch.stack(nominal)
            approach_noise=.10*float((delta.norm()/.03).clamp(.05,1.))
        elif phase=='hold':mean.zero_()
        elif self.task>=4:
            # Follow the desired contact point on a circle around the initial
            # object position.  The tangential term advances yaw; the radial
            # error term prevents the pusher from sliding down the fixed face.
            xy,yaw,_,_,p,_=self.live._physical()
            theta=self.live.lifted_yaw[0]-self.live.initial_yaw[0]
            co,si=torch.cos(theta),torch.sin(theta)
            offset=torch.as_tensor(self.contact_offset,device=self.device,dtype=xy.dtype)
            rotated=torch.stack((co*offset[0]-si*offset[1],si*offset[0]+co*offset[1]))
            radial=rotated/rotated.norm().clamp_min(1e-6)
            tangent=(torch.stack((-rotated[1],rotated[0])) if self.task==5
                     else torch.stack((rotated[1],-rotated[0]))) / rotated.norm().clamp_min(1e-6)
            gap_now=float((self.live.contact_gap()[0]*.1).detach().cpu())
            if gap_now>.0005:
                current_offset=(p[0]-xy[0]).detach().cpu().numpy()
                current_norm=max(float(np.linalg.norm(current_offset)),1e-6)
                corrected=max(float(self.live.raw.agent.pusher_radius),current_norm-gap_now)
                self.contact_offset=current_offset/current_norm*corrected
                rotated=torch.as_tensor(self.contact_offset,device=self.device,dtype=xy.dtype)
                co,si=torch.cos(theta),torch.sin(theta)
                rotated=torch.stack((co*rotated[0]-si*rotated[1],si*rotated[0]+co*rotated[1]))
                radial=rotated/rotated.norm().clamp_min(1e-6)
                tangent=(torch.stack((-rotated[1],rotated[0])) if self.task==5
                         else torch.stack((rotated[1],-rotated[0]))) / rotated.norm().clamp_min(1e-6)
                # Close the measured geometric gap before applying tangential
                # velocity; otherwise the low-drift zero action wins the search.
                command=-radial
            else:
                inward=-radial*self.rotation_inward
                command=tangent+inward
            target_contact=xy[0]+rotated
            radial_error=target_contact-p[0]
            if gap_now<=.0005:
                command=command+4.*radial_error/.02
            self.push_direction=(command/command.norm().clamp_min(1e-6)).detach().cpu().numpy()
            mean=self.push_direction.copy()
            mean=torch.as_tensor(mean,device=self.device,dtype=torch.float32).expand(H,2).clone()
        best=None
        for iteration in range(self.iterations):
            scale=[.14,.035] if phase=='push' else ([approach_noise]*2 if phase=='approach' else [0.,0.])
            noise=torch.randn(n,1,2,device=self.device)*torch.tensor(scale,device=self.device)
            actions=mean[None]+noise
            actions=actions.expand(n,H,2).clone()
            if phase=='approach':
                actions=mean[None]+noise*torch.linspace(1.,0.,H,device=self.device)[None,:,None]
            if phase=='push':
                # Constant-forward segments with sampled braking times cover the
                # friction threshold without random backwards/lateral impacts.
                direction=torch.as_tensor(self.push_direction,device=self.device,dtype=actions.dtype)
                normal=torch.stack((-direction[1],direction[0]))
                count=n//2
                amplitude=torch.linspace(self.push_min,self.push_max,count,device=self.device)
                actions[:count]=amplitude[:,None,None]*direction + torch.randn(count,1,1,device=self.device)*.025*normal
                duration=torch.randint(2,H+1,(n,),device=self.device)
                if self.refining:
                    # Cover short, force-sufficient trim pulses explicitly when
                    # stiction stops the body just outside the unchanged tolerance.
                    actions[:count]=torch.linspace(max(self.push_min,.55),min(self.push_max,.95),count,device=self.device)[:,None,None]*direction
                    duration[:count]=1+torch.arange(count,device=self.device)%2
                actions=torch.where((torch.arange(H,device=self.device)[None,:]<duration[:,None])[:,:,None],actions,0.)
                forward=(actions*direction).sum(-1).clamp(0.,.85)
                lateral=(actions*normal).sum(-1).clamp(-.08,.08)
                actions=forward[...,None]*direction+lateral[...,None]*normal
            actions[0]=mean;actions[1]=0.;actions[2]=self.previous
            if phase=='hold':actions.zero_()
            actions=slew_limit(actions.clamp(-1,1),self.previous)
            result=self.evaluate(root,actions,phase)
            scores=result['score'].masked_fill(~result['valid'],float('inf'))
            if not torch.isfinite(scores).any():break
            weights=torch.softmax(-(scores-scores.min())/.15,dim=0)
            mean=(weights[:,None,None]*actions).sum(0)
            index=scores.argmin();best=actions[index].clone()
        if phase=='push' and self.task>=4:
            # Preserve a force-sufficient tangential proposal when all sampled
            # prefixes have identical (zero-progress) drift scores. Without
            # this tie-breaker MPPI averages it with the zero candidate.
            nominal=torch.as_tensor(self.push_direction,device=self.device,dtype=mean.dtype)
            nominal=nominal*max(self.push_min,.50)
            mean=.5*mean+.5*nominal.expand(H,2)
        # Verify the exact next action, twice, on the same search lane. The live
        # scene is never rewound. Only the verified action is returned for execution.
        proposals=[('mean',mean),('best',best),('brake',slew_limit(torch.zeros(n,H,2,device=self.device),self.previous)[0])]
        diagnostic=dict(phase=phase,waypoint=self.waypoint,refining=self.refining,accepted=False,attempts=[])
        chosen=None
        for name,proposal in proposals:
            if proposal is None:continue
            prefix=proposal[:1][None].expand(n,-1,-1)
            trials=[self.evaluate(root,prefix,phase)];checks=[]
            for _ in range(3):
                trials.append(self.evaluate(root,prefix,phase))
                a,b=trials[-2:]
                check=replay_check(a['d'][0],a['q'][0],b['d'][0],b['q'][0]);checks.append(check)
                if check['replay_consistent']:break
            current=self.live.metrics();safe=True
            for trial in trials:
                t=trial['terminal']
                safe &= bool(trial['valid'][0]) and abs(float(t['transverse_error'][0]))<=max(.005,abs(float(current['transverse_error'][0])))+.0001
                if self.task < 4:
                    safe &= abs(float(t['angle_change'][0]))<=max(.035,abs(float(current['angle_change'][0])))+.001
                else:
                    # Rotation progress is signed. A low-drift candidate that
                    # reverses yaw is invalid even if its position barely moves.
                    safe &= float(t['g'][0]) <= float(current['g'][0]) + .001
                safe &= float(t['g'][0])>=min(-.005,float(current['g'][0]))
            stopping_g=None
            if safe and check['replay_consistent'] and phase=='push' and name!='brake':
                # Check stopping viability, not just the next state. This keeps
                # slew-limited braking from beginning after stopping is impossible.
                desired=torch.zeros(n,H,2,device=self.device)
                desired[:,0]=proposal[0]
                braking=slew_limit(desired,self.previous)
                stop=self.evaluate(root,braking,phase)
                stopping_g=float(stop['terminal']['g'][0])
                safe &= bool(stop['valid'][0]) and stopping_g>=(.02 if self.task>=4 else .003)
            accepted=safe and check['replay_consistent']
            diagnostic['attempts'].append(dict(name=name,safe=safe,accepted=accepted,stopping_g=stopping_g,
                                               replay_checks=checks,**check))
            if accepted:
                chosen=proposal;diagnostic.update(accepted=True,selected=name);break
        if chosen is None:
            # No motion proposal passed verification: stop the episode explicitly,
            # rather than silently execute an unverified candidate.
            restore(env,root)
            raise RuntimeError('No verified execution prefix, including braking: '+str(diagnostic))
        self.previous=chosen[0].clone();self.mean=torch.cat((chosen[1:],torch.zeros_like(chosen[:1])))
        restore(env,root)
        return self.previous.clone(),diagnostic

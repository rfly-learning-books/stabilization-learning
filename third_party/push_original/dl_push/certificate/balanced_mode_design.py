"""Response-space experimental design, separately versioned from sealed runs.

No actor is loaded here. A response is never edited to hit a target: only safe
material contact locations are selected, then independently remeasured.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from collections import defaultdict
import hashlib
import json
import math
import time

import numpy as np
from scipy.optimize import linprog, linear_sum_assignment
from shapely.geometry import Point, shape, mapping, box
from shapely.geometry.polygon import orient
from shapely.affinity import scale, translate
from shapely.ops import unary_union

from .mode_generator import (
    GeneratorConfig, candidate, geometry_safety, measured_ok, coverage,
    sample_candidates,
)
from .fixture import Fixture, build_model, run
from .polygon_fixture_robust import polygon_model
from .library import ModeLibrary
from .evaluation import json_safe

ROOT = Path(__file__).resolve().parents[2]
OLD_ROOT = ROOT / 'runs/mode_library_curved32_20260912'
HOOK_ROOT = ROOT / 'runs/full_m_external_shape_20260913/NovelRoundedHook'
PROTOCOL = 'response_balanced_material_modes_v3'
GEOMETRY_EDITS = {
    'OffsetCapsule': 'Clip four extreme caps by 12 mm to provide broad axis-facing support faces.',
    'RoundedRect': 'Clip four extreme caps by 10 mm; retain curved corners.',
    'RoundedTriangle': 'Truncate the rounded tip by 12 mm to provide a safe downward-pushing face.',
    'Triangle': 'Truncate the convex tip by 50 mm; no convex-tip contact is used.',
    'AsymT': 'Widen and extend the upper stem to move the COM clear of the underarm obstruction.',
    'L': 'Use 40 mm arms in a 200 mm L outline, opening the inner clearance for the 18 mm radius sphere.',
    'StepShape': 'Use a 75 mm middle band so the COM-level side contacts clear the step junctions.',
    'DShape': 'Add horizontal caps at y=+/-57 mm by clipping the circular outline.',
    'RoundedTrapezoid': 'Clip four extreme caps by 10 mm; retain slanted and curved sections.',
}
SOURCE_TEXT = Path(__file__).read_text()
SOURCE_SHA256 = hashlib.sha256(SOURCE_TEXT.encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(json_safe(value), ensure_ascii=False,
                                    indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def response_features(C):
    C = np.asarray(C, dtype=float)
    if C.ndim != 2 or C.shape[0] != 3 or not np.isfinite(C).all():
        raise ValueError('finite C[3,N] required')
    gammas = []
    mixes = []
    for sign in (1, -1):
        result = linprog(-sign * C[2], A_eq=np.vstack([C[:2], np.ones(C.shape[1])]),
                         b_eq=[0., 0., 1.], bounds=(0., None), method='highs')
        if not result.success:
            raise ValueError('XY-cancelling response combination unavailable')
        gammas.append(max(0., float(-result.fun)))
        mixes.append(result.x.tolist())
    gp, gn = gammas
    speed = np.linalg.norm(C[:2], axis=0)
    return dict(gamma_positive=gp, gamma_negative=gn, weak_authority=min(gammas),
                asymmetry_log2=float(np.log2(max(gp, 1e-12) / max(gn, 1e-12))),
                positive_modes=int((C[2] > .05).sum()),
                negative_modes=int((C[2] < -.05).sum()), rank=int(np.linalg.matrix_rank(C)),
                positive_rotation_per_translation=float(np.max(C[2] / np.maximum(speed, 1e-12))),
                negative_rotation_per_translation=float(np.max(-C[2] / np.maximum(speed, 1e-12))),
                cancelling_mixes=mixes)


@dataclass(frozen=True)
class DesignTarget:
    name: str
    group: str
    positive: float
    negative: float
    positive_slots: int = 4
    mirror: bool = False

    def accepted(self, f):
        g = f['weak_authority']
        a = f['asymmetry_log2']
        if self.group == 'geometry_diverse':
            return g >= .2 and f['rank'] == 3
        if self.group == 'strong_balanced':
            return 2. <= g <= 4.5 and abs(a) <= .5
        if self.group == 'medium_balanced':
            return .8 <= g < 2. and abs(a) <= .5
        if self.group == 'weak_balanced':
            return .2 <= g < .8 and abs(a) <= .5
        weak_sign = -1 if self.positive < self.negative else 1
        return (.2 <= g < .8 and .8 <= max(f['gamma_positive'], f['gamma_negative']) <= 4.5
                and 1. <= weak_sign * a <= 2.6)

    def loss(self, f):
        if self.group == 'geometry_diverse':
            gp=max(f['gamma_positive'],1e-9);gn=max(f['gamma_negative'],1e-9)
            return (-math.log(gp)-math.log(gn)+.20*math.log(gp/gn)**2
                    +(0. if self.accepted(f) else 12.))
        err = math.log(max(f['gamma_positive'], 1e-9) / self.positive) ** 2
        err += math.log(max(f['gamma_negative'], 1e-9) / self.negative) ** 2
        return err + (0. if self.accepted(f) else 12.)


def design_targets():
    strong = ['Plus', 'Capsule', 'OffsetCapsule', 'Parallelogram', 'RoundedRect',
              'RoundedTriangle', 'RoundedT', 'Triangle']
    middle = ['Square', 'RectWide', 'Hexagon', 'H', 'C_wide', 'C_narrow',
              'ChamferRect', 'Trapezoid']
    weak = ['AsymT', 'IShape', 'L', 'Pentagon', 'RoundedL', 'RoundedCross',
            'Ellipse', 'StepShape']
    records = []
    for i, name in enumerate(strong):
        value = [2.4, 2.6, 2.4, 3.0, 2.3, 3.2, 2.5, 2.7][i]
        records.append(DesignTarget(name, 'strong_balanced', value, value))
    for i, name in enumerate(middle):
        value = [1.2, 1.5, 1.35, 1.6, 1.1, 1.7, 1.5, 1.8][i]
        records.append(DesignTarget(name, 'medium_balanced', value, value))
    for i, name in enumerate(weak):
        value = [.25, .30, .36, .42, .48, .55, .63, .70][i]
        records.append(DesignTarget(name, 'weak_balanced', value, value))
    for name, gp, gn, np_ in [('T', .4, 1.5, 2), ('DShape', .5, 1.5, 2),
                             ('RoundedTrapezoid', .65, 2., 2),
                             ('NovelRoundedHook', .29, .9, 2),
                             ('Z', 1.6, .5, 6), ('UShape', 1.7, .65, 6),
                             ('ArcC', 1.15, .35, 6),
                             ('MirroredRoundedHook', .9, .29, 6)]:
        records.append(DesignTarget(name, 'one_sided_weak', gp, gn, np_,
                                    name == 'MirroredRoundedHook'))
    assert len(records) == len({x.name for x in records}) == 32
    return records


def source_data(target):
    folder = HOOK_ROOT if 'RoundedHook' in target.name else OLD_ROOT / target.name
    report = json.loads((folder / 'report.json').read_text())
    polygon = shape(json.loads((folder / 'polygon.json').read_text()))
    l, b, r, t = polygon.bounds
    if target.name in ['RoundedRect', 'OffsetCapsule', 'RoundedTrapezoid']:
        inset = .012 if target.name == 'OffsetCapsule' else .010
        polygon = polygon.intersection(box(l+inset, b+inset, r-inset, t-inset))
    elif target.name in ['Triangle', 'RoundedTriangle']:
        inset = .050 if target.name == 'Triangle' else .012
        polygon = polygon.intersection(box(l-.01, b-.01, r+.01, t-inset))
    elif target.name == 'DShape':
        polygon = polygon.intersection(box(l-.01, -.057, r+.01, .057))
    elif target.name == 'AsymT':
        polygon = polygon.union(box(-.040, -.035, .040, .130))
    elif target.name == 'L':
        polygon = unary_union([box(-.10,-.10,-.06,.10),box(-.06,-.10,.10,-.06)])
    elif target.name == 'StepShape':
        polygon = unary_union([box(-.09,-.0825,.09,-.0375),
                               box(-.09,-.0375,.045,.0375),
                               box(-.09,.0375,0.,.0825)])
    polygon = translate(polygon, xoff=-polygon.centroid.x, yoff=-polygon.centroid.y)
    if target.mirror:
        polygon = scale(polygon, xfact=-1., yfact=1., origin=(0, 0))
    polygon = orient(polygon, 1.)
    return folder, report, polygon


def contact_key(c):
    return tuple(np.round(np.r_[c.point, c.normal, c.faces.ravel()], 8))


def safe_contact(p, c, cfg):
    quality = geometry_safety(p, c, cfg)
    if not quality['accepted']:
        return False
    dot = float(c.faces[0] @ c.normal)
    if dot <= 0:
        return False
    centre = c.point + cfg.radius / dot * c.normal
    return Point(centre).distance(p) >= cfg.radius - 1e-6


def boundary_contact(p, s):
    vertices = np.asarray(p.exterior.coords)
    edges = np.diff(vertices, axis=0)
    lengths = np.linalg.norm(edges, axis=1)
    cumulative = np.r_[0., np.cumsum(lengths)]
    s = s % cumulative[-1]
    i = min(int(np.searchsorted(cumulative, s, side='right') - 1), len(edges) - 1)
    frac = np.clip((s - cumulative[i]) / lengths[i], .25, .75)
    point = vertices[i] + frac * edges[i]
    normal = np.array([edges[i, 1], -edges[i, 0]]) / lengths[i]
    return candidate(point, normal, 'response_refined_arc')


def build_pool(target, report, p, cfg, geometry_changed=None):
    if geometry_changed is None:
        geometry_changed = target.name in GEOMETRY_EDITS
    pool = []
    seen = set()

    def add(c, prior, source):
        key = contact_key(c)
        if key in seen or not safe_contact(p, c, cfg):
            return
        if c.role == 'response_refined_arc':
            supports = geometry_safety(p, c, cfg)['supports']
            if not any(s.get('kind') == 'smooth_arc_facets' for s in supports):
                c.role = 'response_refined_face'
        # Keep distinct, safe contact locations. Raw response priors are NOT final C.
        seen.add(key)
        pool.append(dict(contact=c, c=np.asarray(prior, float), prior_source=source,
                         screened=False, valid=True))

    # Prefer full old fits over short old screens for duplicated material points.
    for entry, kind in [(m, 'old_full_fit') for m in report['modes']] + [
            (m, 'old_short_screen') for m in report['candidates']
            if m['valid'] and (m.get('measurement') is not None or 'screened' not in m)]:
        g = entry['geometry']
        c = candidate(g['point'], g['normal'], g['role'], g['faces'])
        response = np.asarray(entry.get('c', (entry.get('measurement') or {}).get('c')), float)
        if target.mirror:
            reflect = np.array([-1., 1.])
            c = candidate(c.point * reflect, c.normal * reflect, c.role,
                          c.faces * reflect)
            response = response * np.array([-1., 1., -1.])
        suffix = '_different_geometry_prior' if geometry_changed else ''
        add(c, response, kind + suffix + ('_mirror_prior' if target.mirror else ''))
    if geometry_changed:
        # A geometry-based proposal is only a proposal. Every selected point
        # must pass fresh screening, two-speed fitting, and independent tests.
        slopes = []
        for record in report['modes']:
            pt = np.array(record['geometry']['point']); normal=np.array(record['geometry']['normal'])
            lever = -pt[0]*normal[1] + pt[1]*normal[0]
            if abs(lever) > .005:
                slopes.append(abs(record['c'][2] / lever))
        gain = float(np.clip(np.median(slopes), 15., 100.)) if slopes else 35.
        sampling = GeneratorConfig(allow_smooth_arcs=True, candidate_budget=240, dt=cfg.dt)
        _, geometric = sample_candidates(p,12,sampling,target.name+'_modified')
        for c in geometric:
            lever = -c.point[0]*c.normal[1] + c.point[1]*c.normal[0]
            predicted = np.r_[-.8*c.normal, gain*lever]
            add(c,predicted,'geometric_torque_prior_not_measurement')
    original = list(pool)
    levels = sorted(set([0., -.04, .04] + [s * x for s in (-1, 1) for x in
                   [.22, .32, .45, .60, .78, .95, 1.2, 1.6, 2., 2.6, 3.2]]))
    groups = defaultdict(list)
    for item in original:
        c = item['contact']
        if len(c.faces) == 1:
            groups[(*np.round(c.normal, 6), round(float(c.point @ c.normal), 6))].append(item)
    for items in groups.values():
        if len(items) < 2:
            continue
        normal = items[0]['contact'].normal
        tangent = np.array([-normal[1], normal[0]])
        s = np.array([x['contact'].point @ tangent for x in items])
        if np.ptp(s) < .002:
            continue
        responses = np.array([x['c'] for x in items])
        coef = np.polyfit(s, responses, 1)
        if abs(coef[0, 2]) < .1:
            continue
        offset = float(items[0]['contact'].point @ normal)
        for w in levels:
            pos = (w - coef[1, 2]) / coef[0, 2]
            c = candidate(pos * tangent + offset * normal, normal, 'response_refined_face')
            add(c, coef[0] * pos + coef[1], 'interpolated_prior_not_measurement')
    # Arc zero crossings cannot be refined by grouping identical normals.
    perimeter = p.exterior.length
    arcs = [(p.exterior.project(Point(x['contact'].point)), x) for x in original
            if len(x['contact'].faces) == 1]
    arcs.sort(key=lambda pair: pair[0])
    for j, (s0, left) in enumerate(arcs):
        for shift in (1, 2, 3):
            s1, right = arcs[(j + shift) % len(arcs)]
            ds = (s1 - s0) % perimeter
            if not 1e-5 < ds < .07 or left['contact'].normal @ right['contact'].normal < .7:
                continue
            dw = right['c'][2] - left['c'][2]
            if abs(dw) < .025:
                continue
            for w in levels:
                f = (w - left['c'][2]) / dw
                if not .01 < f < .99:
                    continue
                c = boundary_contact(p, s0 + f * ds)
                add(c, (1 - f) * left['c'] + f * right['c'], 'arc_interpolated_prior_not_measurement')
    return pool


def translation_indices(pool):
    if len(pool) < 4:
        raise ValueError(f'only {len(pool)} safe candidate contacts; check polygon orientation')
    C = np.array([x['c'] for x in pool])
    v = np.linalg.norm(C[:, :2], axis=1)
    ratio = np.abs(C[:, 2]) / np.maximum(v, 1e-12)
    directions = np.array([[0, -1], [0, 1], [1, 0], [-1, 0]])
    cosine = directions @ C[:, :2].T / np.maximum(v[None], 1e-12)
    reliable = np.array([x.get('screened',False) or x.get('prior_source') == 'old_full_fit'
                         for x in pool])
    cost = ratio[None] + .015 * (1 - cosine) + .2*(~reliable)[None]
    legal = (cosine > .707) & (ratio[None] <= .15) & np.array([x['valid'] for x in pool])[None]
    cost = np.where(legal, cost, 1e6)
    # Four occupied axis sectors alone do not guarantee strict positive span:
    # near-45-degree ties can select a whole set on a single closed half-plane.
    queue=[cost];seen=set()
    for _ in range(64):
        if not queue:
            break
        trial=queue.pop(0)
        rows,cols=linear_sum_assignment(trial)
        if len(rows)!=4 or np.any(trial[rows,cols]>=1e6):
            continue
        key=tuple(cols)
        if key in seen:
            continue
        seen.add(key)
        angles=np.sort(np.arctan2(C[cols,1],C[cols,0]))
        largest_gap=np.diff(np.r_[angles,angles[0]+2*np.pi]).max()
        if largest_gap < np.pi-.01:
            return cols.tolist()
        for row,col in zip(rows,cols):
            branch=trial.copy()
            cosine_to_selected=C[:,:2]@C[col,:2]/np.maximum(v*v[col],1e-12)
            branch[row,cosine_to_selected>.999]=1e6
            queue.append(branch)
    raise ValueError('no four distinct, positive-spanning near-translation contacts in candidate pool')


def choose_indices(pool, target, seed=0, attempts=320):
    base = translation_indices(pool)
    C = np.array([x['c'] for x in pool]).T
    signs = []
    for sign, goal, count in [(1, target.positive, target.positive_slots),
                              (-1, target.negative, 8 - target.positive_slots)]:
        available = [i for i, x in enumerate(pool) if x['valid'] and i not in base
                     and sign * C[2, i] > .05]
        if len(available) < count:
            raise ValueError(f'insufficient safe rotation candidates for sign {sign}: {len(available)}')
        w = sign * C[2, available]
        # Keep original extremes, but prefer gains in the target capability band.
        weights = np.exp(-.65 * np.log(w / max(goal, .1)) ** 2) + .005
        weights /= weights.sum()
        signs.append((available, count, weights))
    rng = np.random.default_rng(seed)
    best = None
    cache = {}
    for _ in range(attempts):
        ids = base.copy()
        for available, count, weights in signs:
            ids += rng.choice(available, count, replace=False, p=weights).tolist()
        key = tuple(sorted(ids))
        if key in cache:
            continue
        f = response_features(C[:, ids])
        value = target.loss(f)
        # Prefer spatially distinct useful contacts; do not trade capability for geometry alone.
        points = np.array([pool[i]['contact'].point for i in ids])
        d = np.linalg.norm(points[:, None] - points[None], axis=-1) + np.eye(12)
        value += .005 * max(0., .002 - float(d.min())) / .002
        cache[key] = value
        if best is None or value < best[0]:
            best = (value, ids, f)
    return best[1], best[2]


def _trial(fixture, c, cfg, speed, yaw=0., short=False, perturbation=None, render=None):
    return run(fixture, c.point, c.normal, cfg.radius, speed=speed, yaw=yaw,
               seconds=.9 if short else 4., distance=.006 if short else cfg.distance,
               faces=c.faces, initial_offset_body=perturbation, render=render)


def screen_item(fixture, pool, i, cfg, out):
    item=pool[i]
    if item['screened']:
        return
    try:
        a,r=_trial(fixture,item['contact'],cfg,.04,short=True)
        np.savez_compressed(out/f'candidate_{i:03}_screen.npz',samples=a)
        item.update(c=np.asarray(r['c']),valid=measured_ok(r,cfg,.004),measurement=r)
    except (ValueError,RuntimeError,FloatingPointError) as exc:
        item.update(valid=False,failure=repr(exc))
    item['screened']=True


def refine_translations(fixture,pool,p,cfg,out):
    """Find actual near-zero angular responses; never set a measured omega to zero."""
    keys={contact_key(x['contact']):i for i,x in enumerate(pool)}
    for direction in np.array([[0,-1],[0,1],[1,0],[-1,0]]):
        def good(i):
            x=pool[i];v=np.linalg.norm(x['c'][:2])
            return (x['screened'] and x['valid'] and v>1e-6
                    and x['c'][:2]@direction/v>.707 and abs(x['c'][2])/v<.10)
        eligible=[i for i,x in enumerate(pool) if len(x['contact'].faces)==1
                  and -x['contact'].normal@direction>.707]
        if any(good(i) for i in eligible):
            continue
        if not eligible:
            continue
        ordered=sorted(eligible,key=lambda i:p.exterior.project(Point(pool[i]['contact'].point)))
        probes=[min(eligible,key=lambda i:abs(pool[i]['c'][2]))]
        probes += [ordered[j] for j in np.linspace(0,len(ordered)-1,min(13,len(ordered)),dtype=int)]
        for i in dict.fromkeys(probes):
            screen_item(fixture,pool,i,cfg,out)
            if good(i):
                break
        # A narrow *reachable interval on a long face* may leave only its
        # midpoint in the coarse sampler. Measure small safe offsets to obtain
        # a real angular-response slope rather than needing a pre-existing pair.
        if not any(good(i) for i in eligible):
            seeds=sorted([i for i in eligible if pool[i]['screened'] and pool[i]['valid']],
                         key=lambda i:abs(pool[i]['c'][2])/max(np.linalg.norm(pool[i]['c'][:2]),1e-12))[:3]
            for i in seeds:
                anchor=pool[i]['contact'];tangent=np.array([-anchor.normal[1],anchor.normal[0]])
                for offset in [-.001,.001,-.002,.002,-.004,.004]:
                    contact=candidate(anchor.point+offset*tangent,anchor.normal,'measured_local_refinement')
                    if contact_key(contact) in keys or not safe_contact(p,contact,cfg):
                        continue
                    index=len(pool);keys[contact_key(contact)]=index
                    pool.append(dict(contact=contact,c=pool[i]['c'].copy(),valid=True,screened=False,
                                     prior_source='local_probe_not_measurement'))
                    eligible.append(index);screen_item(fixture,pool,index,cfg,out)
                    if good(index):break
                if any(good(j) for j in eligible):break
        for step in range(12):
            if any(good(i) for i in eligible):
                break
            measured=[i for i in eligible if pool[i]['screened'] and pool[i]['valid']]
            proposals=[]
            for ii,i in enumerate(measured):
                for j in measured[ii+1:]:
                    left,right=pool[i],pool[j];lc,rc=left['contact'],right['contact']
                    dw=right['c'][2]-left['c'][2]
                    if abs(dw)<.02:
                        continue
                    fraction=-left['c'][2]/dw
                    same_face=(lc.normal@rc.normal>.999999 and
                               abs((rc.point-lc.point)@lc.normal)<1e-7)
                    if same_face:
                        point=lc.point+fraction*(rc.point-lc.point)
                        contact=candidate(point,lc.normal,'measured_zero_refinement')
                    else:
                        if not 0.<fraction<1. or lc.normal@rc.normal<.7:
                            continue
                        s0=p.exterior.project(Point(lc.point));s1=p.exterior.project(Point(rc.point))
                        ds=(s1-s0+p.exterior.length/2)%p.exterior.length-p.exterior.length/2
                        if abs(ds)>.10:
                            continue
                        contact=boundary_contact(p,s0+fraction*ds)
                    if contact_key(contact) in keys or not safe_contact(p,contact,cfg):
                        continue
                    bracket=left['c'][2]*right['c'][2]<0
                    proposals.append((not bracket,abs(left['c'][2])+abs(right['c'][2]),contact))
            if not proposals:
                break
            _,_,contact=min(proposals,key=lambda x:x[:2])
            index=len(pool);keys[contact_key(contact)]=index
            pool.append(dict(contact=contact,c=np.r_[-.8*contact.normal,0.],
                             prior_source='zero_search_proposal_not_measurement',
                             screened=False,valid=True))
            eligible.append(index)
            screen_item(fixture,pool,index,cfg,out)
            write_json(out/'progress.json',dict(stage='translation_zero_search',
                       direction=direction.tolist(),iteration=step,candidates=len(pool)))


def fitted_selection(pool,fits,target,seed):
    indices=[i for i,r in fits.items() if r['valid']]
    if len(indices)<12:
        return None
    try:
        local,features=choose_indices([pool[i] for i in indices],target,seed,attempts=1600)
        if target.accepted(features):
            return [indices[i] for i in local]
    except ValueError:
        pass
    return None


def rotation_sign_consistent(ci,trials,cfg):
    ci=np.asarray(ci)
    if abs(ci[2])/max(np.linalg.norm(ci[:2]),1e-12)<=cfg.near_translation_ratio:
        return True
    return all(float(r['twist_mean'][2])*float(ci[2])>0 for r in trials)


def calibrate_shape(target, root, render=True, source_override=None):
    root = Path(root)
    out = root / target.name
    if (out / 'report.json').exists():
        existing = json.loads((out / 'report.json').read_text())
        if existing.get('protocol') == PROTOCOL and existing.get('status') == 'COMPLETE':
            return {k: existing[k] for k in ['shape', 'status', 'distribution', 'elapsed_s']}
        raise FileExistsError(f'noncomplete output requires explicit archival before retry: {out}')
    out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    if source_override is None:
        folder, old, p = source_data(target)
        geometry_changed = target.name in GEOMETRY_EDITS
    else:
        folder=Path(source_override['prior_folder'])
        old=json.loads((folder/'report.json').read_text())
        p=shape(source_override['polygon'])
        p=orient(translate(p,xoff=-p.centroid.x,yoff=-p.centroid.y),1.)
        old_polygon=shape(json.loads((folder/'polygon.json').read_text()))
        geometry_changed=p.symmetric_difference(old_polygon).area>1e-12
    cfg = GeneratorConfig(allow_smooth_arcs=True, candidate_budget=160,
                          dt=old['config']['dt'])
    write_json(out / 'target.json', asdict(target))
    write_json(out / 'polygon.json', mapping(p))
    source_hash = hashlib.sha256((folder / 'polygon.json').read_bytes()).hexdigest()
    if target.name in ['T', 'Square', 'RectWide', 'Plus', 'H'] and not geometry_changed:
        # The asset helper's returned polygon may have clockwise winding;
        # retain the already centred, CCW source polygon used by safety checks.
        model, _ = build_model(out / 'model.xml', cfg.radius, cfg.dt, target.name, cfg.kv)
    else:
        model, p = polygon_model(p, out / 'model.xml', cfg.radius, cfg.dt, cfg.kv)
    write_json(out/'polygon.json',mapping(p))
    fixture = Fixture(model)
    pool = build_pool(target, old, p, cfg, geometry_changed)
    write_json(out/'candidate_pool_initial.json',[
        dict(candidate_id=i,geometry=x['contact'].record(),prior_c=x['c'],prior_source=x['prior_source'])
        for i,x in enumerate(pool)])
    write_json(out / 'progress.json', dict(stage='pool_ready',candidates=len(pool)))
    fits = {}
    # Recover full measurements from an archived attempt only when the compiled
    # physical model is byte-identical. Copies keep every original trial intact.
    import shutil
    model_sha = hashlib.sha256((out/'model.xml').read_bytes()).hexdigest()
    pool_keys = {contact_key(x['contact']):i for i,x in enumerate(pool)}
    for old_attempt in sorted((root/'_attempts').glob(target.name+'_*')):
        record_file=old_attempt/'selection_failure.json'
        if not record_file.exists():
            record_file=old_attempt/'measurement_checkpoint.json'
        if not record_file.exists():
            record_file=old_attempt/'report.json'
        if not record_file.exists() or not (old_attempt/'model.xml').exists():
            continue
        if hashlib.sha256((old_attempt/'model.xml').read_bytes()).hexdigest()!=model_sha:
            continue
        cached=json.loads(record_file.read_text())
        for record in cached.get('full_fits',cached.get('modes',[])):
            if not record['valid']:
                continue
            g=record['geometry']; key=contact_key(candidate(g['point'],g['normal'],g['role'],g['faces']))
            if key not in pool_keys:
                contact=candidate(g['point'],g['normal'],g['role'],g['faces'])
                if not safe_contact(p,contact,cfg):
                    continue
                pool_keys[key]=len(pool)
                pool.append(dict(contact=contact,c=np.array(record['c']),valid=True,screened=True,
                                 prior_source='archived_identical_model_full_measurement'))
            i=pool_keys[key];old_id=record['candidate_id'];copied=dict(record)
            copied.update(candidate_id=i,measurement_cache_source=str(old_attempt))
            if not rotation_sign_consistent(copied['c'],copied['trials'],cfg):
                copied.update(valid=False,rejection_reason='rotation_sign_changed_across_trials')
            fits[i]=copied
            pool[i].update(c=np.array(record['c']),valid=copied['valid'],screened=True)
            for j in range(len(record['trials'])):
                shutil.copy2(old_attempt/f'candidate_{old_id:03}_trial_{j}.npz',
                             out/f'candidate_{i:03}_trial_{j}.npz')
    selected = None
    for iteration in range(10):
        already=fitted_selection(pool,fits,target,iteration+19)
        if already is not None:
            selected=already
            C=np.array([fits[i]['c'] for i in selected]).T
            authority=coverage(C,np.ones(12,bool),cfg);features=response_features(C)
            if authority['four_direction_translation']:
                break
        if iteration>0:
            actual=[x for x in pool if x['screened'] and x['valid']]
            try:
                translation_indices(actual)
            except ValueError:
                refine_translations(fixture,pool,p,cfg,out)
        try:
            selected, predicted = choose_indices(pool, target, seed=iteration + 97,
                                                  attempts=400 if iteration else 700)
        except ValueError:
            refine_translations(fixture,pool,p,cfg,out)
            write_json(out/'measurement_checkpoint.json',dict(
                pool=[dict(candidate_id=i,geometry=x['contact'].record(),c=x['c'],
                           valid=x['valid'],screened=x['screened']) for i,x in enumerate(pool)],
                full_fits=list(fits.values())))
            selected,predicted=choose_indices(pool,target,seed=iteration+97,attempts=700)
        for i in selected:
            item = pool[i]
            if item['screened']:
                continue
            screen_item(fixture,pool,i,cfg,out)
        # Re-rank with these fresh measurements before spending full-trial time.
        if any(not pool[i]['valid'] for i in selected):
            continue
        current = response_features(np.array([pool[i]['c'] for i in selected]).T)
        if not target.accepted(current) and iteration < 3:
            continue
        for i in selected:
            if i in fits:
                continue
            item = pool[i]
            c = item['contact']
            trials = []
            conditions = [(u, 0., 'fit', None) for u in cfg.speeds]
            conditions += [(cfg.validation_speed, cfg.validation_yaw, 'validation', None)]
            tangent = np.array([-c.normal[1], c.normal[0]])
            conditions += [(cfg.validation_speed, cfg.validation_yaw, 'perturbation',
                            sign * cfg.perturbation_m * tangent) for sign in (-1, 1)]
            try:
                for j, (u, yaw, kind, offset) in enumerate(conditions):
                    a, record = _trial(fixture, c, cfg, u, yaw, perturbation=offset)
                    np.savez_compressed(out / f'candidate_{i:03}_trial_{j}.npz', samples=a)
                    record.update(kind=kind, initial_offset_body=None if offset is None else offset.tolist())
                    trials.append(record)
                u = np.asarray(cfg.speeds)
                values = np.array([r['twist_mean'] for r in trials[:2]])
                ci = (u[:, None] * values).sum(0) / (u @ u)
                prediction = cfg.validation_speed * ci
                weight = np.array([1., 1., .1])
                error = max(float(np.linalg.norm((np.array(r['twist_mean']) - prediction) * weight)
                                  / max(np.linalg.norm(np.array(r['twist_mean']) * weight), 1e-12))
                            for r in trials[2:])
                good = all(measured_ok(r, cfg, .03) for r in trials) and error <= cfg.max_response_error
                sign_ok=rotation_sign_consistent(ci,trials,cfg)
                good=good and sign_ok
                record = dict(candidate_id=i, geometry=c.record(), c=ci.tolist(), valid=good,
                              geometry_safety=geometry_safety(p, c, cfg),
                              validation_error=error, trials=trials,rotation_sign_consistent=sign_ok)
                fits[i] = record
                item['c'], item['valid'] = ci, good
            except (ValueError, RuntimeError, FloatingPointError) as exc:
                item['valid'] = False
                fits[i] = dict(candidate_id=i, geometry=c.record(), valid=False,
                               failure=repr(exc), trials=trials)
            write_json(out / 'progress.json', dict(stage='full_calibration', iteration=iteration,
                       candidates=len(pool), full_fits=len(fits), last_candidate=i))
            write_json(out/'measurement_checkpoint.json',dict(
                pool=[dict(candidate_id=j,geometry=x['contact'].record(),c=x['c'],
                           valid=x['valid'],screened=x['screened']) for j,x in enumerate(pool)],
                full_fits=list(fits.values())))
        if all(pool[i]['valid'] and i in fits for i in selected):
            C = np.array([fits[i]['c'] for i in selected]).T
            authority = coverage(C, np.ones(12, bool), cfg)
            features = response_features(C)
            if target.accepted(features) and authority['four_direction_translation']:
                break
    else:
        write_json(out / 'selection_failure.json', dict(target=asdict(target),
                   pool=[dict(candidate_id=i, geometry=x['contact'].record(), c=x['c'],
                              valid=x['valid'], screened=x['screened']) for i,x in enumerate(pool)],
                   full_fits=list(fits.values())))
        raise RuntimeError(f'{target.name}: distribution/translation target not reached; trials preserved')
    records = []
    for mode, i in enumerate(selected):
        record = dict(fits[i]); record['mode'] = mode
        records.append(record)
    meta = dict(protocol=PROTOCOL, shape_split='training_design_not_sealed_test',
                physics='planar CPU MuJoCo fixture, not free-body validation',
                input='absolute inward normal pusher command u in m/s',
                response='body COM [vx,vy,omega_z] = c_i*u',
                config=asdict(cfg), distribution_target=asdict(target), distribution=features,
                source_polygon_sha256=source_hash, mirrored_geometry=target.mirror,
                geometry_modification=(source_override.get('description') if source_override is not None
                                       else GEOMETRY_EDITS.get(target.name)),
                geometry_family=None if source_override is None else source_override.get('family'),
                actual_polygon_wkb_sha256=hashlib.sha256(p.wkb).hexdigest(),
                controller='dl_push.certificate.material_contact.material_contact_velocity',
                authority=authority)
    if source_override is not None and source_override.get('split'):
        meta['shape_split']=source_override['split']
    meta['design_source_sha256'] = SOURCE_SHA256
    meta['support_source_sha256']={name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                                  for name in ['fixture.py','material_contact.py','polygon_fixture.py',
                                               'polygon_fixture_robust.py','mode_generator.py']}
    (out / 'design_source_snapshot.py').write_text(SOURCE_TEXT)
    cs = [pool[i]['contact'] for i in selected]
    faces = np.zeros((12, 2, 2)); masks = np.zeros((12, 2), bool)
    for i, c in enumerate(cs):
        faces[i, :len(c.faces)] = c.faces; masks[i, :len(c.faces)] = True
    lib = ModeLibrary(target.name, C, np.array([c.point for c in cs]),
                      np.array([c.normal for c in cs]), np.array([[-c.normal[1], c.normal[0]] for c in cs]),
                      np.full(12, min(cfg.speeds)), np.full(12, max(cfg.speeds)), np.ones(12, bool),
                      face_normals=faces, face_mask=masks,
                      contact_offset=np.array([cfg.radius / (c.faces[0] @ c.normal) for c in cs]),
                      fit_error=np.array([r['validation_error'] for r in records]),
                      hold_rate=np.array([min(t['face_hold_rate'] for t in r['trials']) for r in records]),
                      metadata=json_safe(meta))
    lib.save(out / 'library'); np.save(out / 'C.npy', C)
    np.savez_compressed(out / 'C_history.npz', C=np.stack([np.c_[C[:, :i], np.full((3, 12-i), np.nan)]
                                                       for i in range(1, 13)]),
                        selected_candidate_ids=selected, final_valid=np.ones(12, bool),
                        semantics=np.array('final selected-column assembly; chronological measurements in candidate trials'))
    if render:
        render_modes(out, target, fixture, cs, p, cfg, records)
    report = dict(shape=target.name, protocol=PROTOCOL, N=12, status='COMPLETE', valid_modes=12,
                  config=asdict(cfg), metadata=meta, authority=authority,
                  distribution=features, distribution_target=asdict(target), modes=records,
                  candidates=[dict(candidate_id=i, geometry=x['contact'].record(),
                                   prior_source=x['prior_source'], screened=x['screened'],
                                   valid=x['valid'], c=x['c'], measurement=x.get('measurement'))
                              for i, x in enumerate(pool)],
                  rejected_modes=[r for r in fits.values() if not r['valid']],
                  elapsed_s=time.perf_counter()-started)
    write_json(out / 'report.json', report)
    lines = [f'# {target.name}: 12个重新标定的推动模式', '',
             f'分组：{target.group}；所有响应来自实际标定，未直接修改C。', '',
             '[12模式视频](modes.mp4) · [接触图](overview.png) · [完整报告](report.json)', '',
             '|模式|接触点/mm|c_x|c_y|c_omega (rad/m)|视频|', '|---|---|---:|---:|---:|---|']
    for r in records:
        i=r['mode']; x,y=np.array(r['geometry']['point'])*1000; cx,cy,cw=r['c']
        lines.append(f'|{i:02}|({x:.2f}, {y:.2f})|{cx:.6f}|{cy:.6f}|{cw:.6f}|[MP4](mode_{i:02}.mp4)|')
    lines += ['', '纯平面夹具标定；没有自由刚体稳定性或闭环成功声明。',
              '正向和负向能力是所选12模式库的能力，不代表整个形状所有可能接触点的极限。',
              '法向推进、固定物体系接触点与方向；保留速度、角速度、接触误差、法向力及原始轨迹。']
    (out / 'README.md').write_text('\n'.join(lines)+'\n')
    return dict(shape=target.name, status='COMPLETE', distribution=features,
                elapsed_s=report['elapsed_s'])


def render_modes(out, target, fixture, contacts, p, cfg, records):
    import imageio.v2 as imageio
    from PIL import Image
    from .mode_render import frame
    images = {}
    for mode, c in enumerate(contacts):
        frames = []
        def callback(f, row, point, normal):
            frames.append(frame(f, row, point, normal, p, mode, c.role,
                                target.name, c.faces, cfg.radius))
        a, record = _trial(fixture, c, cfg, .04, render=callback)
        np.savez_compressed(out / f'mode_{mode:02}_demo.npz', samples=a)
        records[mode]['demo'] = record
        if not measured_ok(record, cfg, .03):
            raise RuntimeError(f'{target.name} mode {mode}: demonstration contact failed')
        if not frames:
            raise RuntimeError('empty video')
        with imageio.get_writer(out / f'mode_{mode:02}.mp4', fps=30, codec='libx264',
                                quality=7, ffmpeg_params=['-threads', '1']) as writer:
            for image in frames:
                writer.append_data(image)
        # Only thumbnails are retained across modes, keeping parallel workers small.
        images[mode] = [Image.fromarray(im).resize((320, 360)) for im in frames]
        write_json(out / 'progress.json', dict(stage='rendering', rendered_modes=mode+1))
    with imageio.get_writer(out / 'modes.mp4', fps=30, codec='libx264', quality=7,
                            macro_block_size=2, ffmpeg_params=['-threads', '1']) as writer:
        for j in range(max(map(len, images.values()))):
            grid = Image.new('RGB', (1280, 1080), 'white')
            for mode, frames in images.items():
                grid.paste(frames[min(j, len(frames)-1)], ((mode % 4)*320, (mode // 4)*360))
            writer.append_data(np.asarray(grid))
        grid.save(out / 'overview.png')

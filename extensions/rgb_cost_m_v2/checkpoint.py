"""Lossless raw-candidate deltas against the immutable frozen-feature ready state."""
from pathlib import Path
import hashlib
import torch
from stabilization_learning.storage import load_checkpoint as load_stored_checkpoint


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()


def load_checkpoint(path,map_location='cpu'):
    path=Path(path)
    stored=load_stored_checkpoint(path,map_location=map_location)
    if stored.get('schema')!='rgb_frozen_candidate_delta_v2': return stored
    reference=(path.parent/stored['reference']['path']).resolve()
    reference.relative_to(path.parent.parent.resolve())
    if digest(reference)!=stored['reference']['sha256']: raise RuntimeError('Frozen reference changed')
    base=load_stored_checkpoint(reference,map_location=map_location)
    base.update(stored['state'])
    base['learner'].update(stored['learner_changes'])
    return base

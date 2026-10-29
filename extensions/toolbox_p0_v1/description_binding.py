"""Bind the FINAL facade description; never mutate archived environment sources."""
from __future__ import annotations
import hashlib
from pathlib import Path
from stabilization_learning.common.contracts_v2 import identity


def refresh_description_binding(facade, *, sources):
    dependencies=dict(sources)
    here=Path(__file__).resolve()
    dependencies[str(here)]=hashlib.sha256(here.read_bytes()).hexdigest()
    facade.final_description_binding=dict(description=facade.describe().to_dict(),sources=dependencies)
    facade.source_signature_sha256=identity(facade.final_description_binding)
    return facade


def source_bound_contact_facade(env):
    from migration.next_envs_v1.contact_single_disk.contact_single_disk_env import ContactMFacade
    return refresh_description_binding(ContactMFacade(env),sources={
        "contact_environment":env.source_sha256,"contact_model_XML":env.xml_sha256,
        "contact_protocol":env.protocol_sha256})


def separation_diagnostic(error,M,physical_distance2,*,zero_tolerance=1e-12):
    import torch
    norm=error.square().sum(-1)
    value=torch.einsum("...i,...ij,...j->...",error,M,error)
    lower=torch.linalg.eigvalsh(M)[...,0]*norm
    return dict(value=value,feature_norm2=norm,feature_lower_bound=lower,
        feature_collapse=(physical_distance2>zero_tolerance)&(norm<=zero_tolerance),
        candidate_only=True)

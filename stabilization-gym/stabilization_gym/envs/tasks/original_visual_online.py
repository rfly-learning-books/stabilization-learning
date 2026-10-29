"""Explicit online path binding for the reconstructed indoor environment.

This does not modify the frozen offline migration adapter or vendored research
source. It never labels a reconstructed renderer as the historical renderer.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

from .original_visual import OriginalVisualAdapter, file_sha


class OriginalVisualOnlineAdapter(OriginalVisualAdapter):
    def __init__(self, root, legacy_root, scene_audit):
        self.online_legacy_root = Path(legacy_root).resolve()
        self.scene_audit_path = Path(scene_audit).resolve()
        self.scene_audit = json.loads(self.scene_audit_path.read_text(encoding="utf-8"))
        if self.scene_audit.get("passed") is not True:
            raise RuntimeError("Reconstructed scene audit has not passed")
        if self.scene_audit.get("protocol_kind") != "reconstructed_environment" or self.scene_audit.get("original_rgb_equivalent") is not False:
            raise RuntimeError("Expected an explicitly non-equivalent reconstructed environment protocol")
        if Path(self.scene_audit["legacy_root"]).resolve() != self.online_legacy_root:
            raise RuntimeError("Scene audit refers to a different legacy root")
        renderer_manifest = Path(self.scene_audit["scene_manifest"])
        if not renderer_manifest.is_absolute():
            renderer_manifest = self.scene_audit_path.parent / renderer_manifest
        renderer_manifest = renderer_manifest.resolve()
        if renderer_manifest != (self.online_legacy_root.parent / "scenes.json").resolve():
            raise RuntimeError("Scene audit manifest is not the renderer's bound scenes.json")
        if file_sha(renderer_manifest) != self.scene_audit["scene_manifest_sha256"]:
            raise RuntimeError("Audited renderer scene manifest changed: " + str(renderer_manifest))
        for row in self.scene_audit["source_files"]:
            path = Path(row["path"])
            if not path.is_absolute():
                path = self.scene_audit_path.parent / path
            if file_sha(path) != row["sha256"]:
                raise RuntimeError("Audited scene source changed: " + str(path))
        if not self.scene_audit["asset_files"]:
            raise RuntimeError("Scene audit must declare renderer assets")
        for row in self.scene_audit["asset_files"]:
            path = Path(row["path"])
            if not path.is_absolute():
                path = self.scene_audit_path.parent / path
            if file_sha(path) != row["sha256"]:
                raise RuntimeError("Audited scene asset changed: " + str(path))
        super().__init__(root)

    def _load_sources(self):
        self.legacy_root = self.online_legacy_root
        self.cfg["paths"]["legacy_root"] = str(self.legacy_root)
        super()._load_sources()

    def training_config(self, data_bundle):
        cfg = copy.deepcopy(self.cfg)
        cfg["derivative"]["source_run"] = str(Path(data_bundle).resolve())
        # The training wrapper separately imports the preserved derivative
        # initial.pt; this is not a promise that the data bundle holds weights.
        cfg["derivative"]["initialization_checkpoint"] = "initial.pt"
        return cfg

    def new_environment(self, scene):
        """Create only the audited reconstructed room, with true depth blocked."""
        from train_goal_fixed import new_instance
        return new_instance(self.cfg, scene)


def load_reconstructed_runtime(root, legacy_root, scene_audit):
    """Return a path-bound adapter; construction alone does not initialize CUDA."""
    return OriginalVisualOnlineAdapter(root, legacy_root, scene_audit)

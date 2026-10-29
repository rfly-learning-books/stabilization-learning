"""Calibrated, rectified RGB/mask -> MJX asset for four-stage planning.

The image axes/origin must already be registered to the body-frame state.
Mass and friction are nominal inputs, never inferred from RGB.
"""
from pathlib import Path
import numpy as np
from .geometry import Geometry
from .runtime import activate_reference, read_json, write_json, sha256


class VisualGeometry(Geometry):
    def __init__(self, path):
        import shapely.geometry
        self.path=Path(path).resolve()
        self.metadata=read_json(self.path/"asset.json")
        for name,digest in self.metadata["files"].items():
            file=(self.path/name).resolve()
            if not file.is_relative_to(self.path) or sha256(file)!=digest:
                raise ValueError("Visual asset modified or invalid")
        with np.load(self.path/"features.npz",allow_pickle=False) as data:
            self.grid=data["grid"].copy();self.sdf=data["sdf"].copy()
            self.dimensions=data["dimensions"].copy()
        self.mpp=self.metadata["meters_per_pixel"]
        self.name=self.metadata["shape"];self.period=self.metadata["orientation_period"]
        self.xml=self.path/"model/h1_model.xml";self.fingerprint=sha256(self.path/"asset.json")
        self.polygon=shapely.geometry.shape(read_json(self.path/"model/geometry.json")["polygon"])


def build_visual_asset(image_path, output, meters_per_pixel, name,
                       period=2*np.pi, mask=False, foreground=(145,162,177),
                       color_tolerance=24., thickness=.04, mass=.8, friction=3.):
    import cv2
    from scipy.ndimage import distance_transform_edt
    activate_reference()
    from ca_glpr_unseen_shape.geometry.rgb_to_mjx import (
        BuildConfig, from_mask, segment_rgb, save_asset,
    )
    from starter.build_mjx import h1_xml
    if min(meters_per_pixel,thickness,mass,period)<=0 or friction<0:
        raise ValueError("Invalid calibration/nominal parameters")
    order=2*np.pi/period
    if not np.isclose(order,round(order)):raise ValueError("Invalid symmetry period")
    out=Path(output).resolve();out.mkdir(parents=True,exist_ok=False)
    image=cv2.imread(str(image_path),cv2.IMREAD_COLOR)
    if image is None:raise ValueError("Unreadable RGB/mask")
    binary=(image[:,:,0]>127 if mask else
            segment_rgb(cv2.cvtColor(image,cv2.COLOR_BGR2RGB),foreground,color_tolerance,None))
    if not binary.any():raise ValueError("Empty segmentation")
    if binary[0].any() or binary[-1].any() or binary[:,0].any() or binary[:,-1].any():
        raise ValueError("Image clips the object")
    cfg=BuildConfig(meters_per_pixel,thickness=thickness,mass=mass,
                    tolerance_m=.0015,allow_disconnected=True)
    asset=from_mask(binary,cfg);save_asset(asset,out/"model")
    xml=h1_xml(asset,cfg,frame="image",object_friction=friction)
    (out/"model/h1_model.xml").write_text(xml)
    clean=cv2.imread(str(out/"model/mask.png"),cv2.IMREAD_GRAYSCALE)>127
    sdf=(distance_transform_edt(~clean)-distance_transform_edt(clean))*meters_per_pixel
    grid=np.stack([cv2.resize(clean.astype(np.float32),(32,32),interpolation=cv2.INTER_AREA),
                   cv2.resize(sdf.astype(np.float32),(32,32),interpolation=cv2.INTER_AREA)/.1])
    rows,cols=np.nonzero(clean);center=asset["metadata"]["image_centroid_world_m"]
    dims=np.array([(cols.max()-cols.min()+1)*meters_per_pixel/.2,
                   (rows.max()-rows.min()+1)*meters_per_pixel/.2,
                   clean.sum()*meters_per_pixel**2/.04,thickness/.04,
                   center[0]/.2,center[1]/.2],np.float32)
    np.savez_compressed(out/"features.npz",grid=grid.astype(np.float32),
                        sdf=sdf.astype(np.float32),dimensions=dims)
    meta=dict(schema=1,input_kind="calibrated_mask" if mask else "calibrated_RGB",
              source=str(Path(image_path).resolve()),source_sha256=sha256(image_path),
              meters_per_pixel=meters_per_pixel,shape=name,orientation_period=period,
              nominal_mass=mass,nominal_friction=friction,
              frame="image-centered body frame; caller must register physical state",
              files={str(p.relative_to(out)):sha256(p) for p in out.rglob("*") if p.is_file()})
    write_json(out/"asset.json",meta)
    return VisualGeometry(out)

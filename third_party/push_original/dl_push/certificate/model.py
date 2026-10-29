"""MuJoCo model adapter for the certificate-control physics layer."""

from __future__ import annotations

from pathlib import Path
import xml.etree.ElementTree as ET


def certificate_xml(
    source_xml,
    *,
    physics_dt=0.002,
    object_friction=3.0,
    floor_friction=0.5,
    pusher_friction=0.0,
    actuator_force_limit=30.0,
    actuator_kv=100.0,
    free_body=True,
):
    """Convert a generated asset into the certificate scene contract.

    The existing visual/MJX assets use four planar joints for historical
    compatibility.  The certificate scene uses one named free joint so the
    adapter can read a real body pose and monitor height/roll/pitch.  Contact
    pairs are made explicit: floor-object keeps friction, pusher-object is
    normal-only.
    """

    root = ET.parse(str(source_xml)).getroot()
    compiler = root.find("compiler")
    if compiler is None:
        compiler = ET.Element("compiler")
        root.insert(0, compiler)
    compiler.set("angle", "radian")
    compiler.set("alignfree", "false")
    option = root.find("option")
    if option is None:
        option = ET.SubElement(root, "option")
    option.set("timestep", f"{float(physics_dt):.12g}")
    option.set("gravity", "0 0 -9.81")
    option.set("solver", option.get("solver", "Newton"))
    option.set("integrator", option.get("integrator", "implicitfast"))
    option.set("cone", "pyramidal")
    option.set("iterations", option.get("iterations", "20"))
    option.set("ls_iterations", option.get("ls_iterations", "10"))

    visual_global = root.find("./visual/global")
    if visual_global is None:
        visual = root.find("visual")
        if visual is None:
            visual = ET.SubElement(root, "visual")
        visual_global = ET.SubElement(visual, "global")
    visual_global.set("offwidth", max(int(visual_global.get("offwidth", "640")), 640).__str__())
    visual_global.set("offheight", max(int(visual_global.get("offheight", "640")), 640).__str__())

    object_body = root.find("./worldbody/body[@name='object']")
    if object_body is None:
        raise ValueError("certificate model needs a body named 'object'")
    old_joints = list(object_body.findall("joint"))
    old_free = object_body.find("freejoint")
    if free_body:
        if old_free is None:
            for joint in old_joints:
                object_body.remove(joint)
            free = ET.Element("freejoint", name="object_free")
            object_body.insert(0, free)
        else:
            old_free.set("name", "object_free")
            old_free.set("align", "false")
    elif old_free is not None:
        raise ValueError("planar certificate scene cannot consume an existing freejoint")

    for actuator in root.findall("./actuator/*"):
        if actuator.tag in {"velocity", "general"}:
            if actuator.tag == "velocity":
                actuator.set("kv", f"{float(actuator_kv):.12g}")
            actuator.set("ctrllimited", "true")
            actuator.set("ctrlrange", actuator.get("ctrlrange", "-.30 .30"))
            actuator.set("forcelimited", "true")
            actuator.set("forcerange", f"{-float(actuator_force_limit):.12g} {float(actuator_force_limit):.12g}")

    contact = root.find("contact")
    if contact is None:
        contact = ET.SubElement(root, "contact")
    for pair in contact.findall("pair"):
        geom_names = {pair.get("geom1", ""), pair.get("geom2", "")}
        if "pusher_geom" in geom_names:
            pair.set("condim", "1")
            pair.set("friction", "0 0 0 0 0")
        elif "floor" in geom_names:
            pair.set("condim", "3")
            mu = max(float(floor_friction), 0.0)
            pair.set("friction", f"{mu:.12g} {mu:.12g} .005 .0001 .0001")
        else:
            # An object-object pair is not expected in the one-object scene;
            # keep it conservative rather than silently adding tangential
            # authority to a pusher contact.
            pair.set("condim", "1")
            pair.set("friction", "0 0 0 0 0")
    # Some source models omit a floor/pusher pair for a newly added collision
    # mesh.  Add the explicit pairs based on names so every object geom is
    # covered by the same contact contract.
    object_geoms = [g.get("name") for g in object_body.findall("geom") if g.get("name")]
    existing = {(p.get("geom1"), p.get("geom2")) for p in contact.findall("pair")}
    for geom in object_geoms:
        if (geom, "floor") not in existing and ("floor", geom) not in existing:
            ET.SubElement(
                contact,
                "pair",
                geom1=geom,
                geom2="floor",
                condim="3",
                friction=f"{float(floor_friction):.12g} {float(floor_friction):.12g} .005 .0001 .0001",
                solref=".01 1",
                solimp=".9 .95 .001",
            )
        if (geom, "pusher_geom") not in existing and ("pusher_geom", geom) not in existing:
            ET.SubElement(
                contact,
                "pair",
                geom1=geom,
                geom2="pusher_geom",
                condim="1",
                friction=f"{float(pusher_friction):.12g} 0 0 0 0",
                solref=".01 1",
                solimp=".9 .95 .001",
            )
    return ET.tostring(root, encoding="unicode")


def build_certificate_model(source_xml, output, **kwargs):
    """Write and compile a new certificate model directory."""

    import json
    import hashlib
    import mujoco

    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite certificate model: {output}")
    output.mkdir(parents=True)
    xml = certificate_xml(source_xml, **kwargs)
    model = mujoco.MjModel.from_xml_string(xml)
    if model.nu != 2:
        raise ValueError(f"certificate model expects two pusher controls, got {model.nu}")
    (output / "model.xml").write_text(xml)
    metadata = {
        "schema": 1,
        "source_xml": str(Path(source_xml).resolve()),
        "source_sha256": hashlib.sha256(Path(source_xml).read_bytes()).hexdigest(),
        "xml_sha256": hashlib.sha256(xml.encode()).hexdigest(),
        "nq": int(model.nq),
        "nv": int(model.nv),
        "nu": int(model.nu),
        "physics_dt": float(model.opt.timestep),
        "object_joint": "object_free" if kwargs.get("free_body", True) else ["obj_x", "obj_y", "obj_z", "obj_yaw"],
        "contact_contract": "pusher-object condim=1; floor-object condim=3",
        "settings": kwargs,
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata

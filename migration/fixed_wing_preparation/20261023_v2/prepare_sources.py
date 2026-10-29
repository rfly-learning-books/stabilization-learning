"""Record immutable public files, literal numerical coefficients, and provenance."""
from datetime import datetime,timezone
import ast,hashlib,json
from pathlib import Path
HERE=Path(__file__).resolve().parent; SOURCES=HERE/"sources"
COMMIT="10363e0eaeb16a61a7463505f8e122a9eb7e2b4e"
PATHS=["mavsim_python/parameters/aerosonde_parameters.py","mavsim_python/models/mav_dynamics_control.py",
       "mavsim_python/models/trim.py","README.md","LICENSE"]
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(name,x): (HERE/name).write_text(json.dumps(x,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
tree=json.loads((SOURCES/"upstream_tree.json").read_text(encoding="utf-8-sig"))
lookup={x["path"]:x for x in tree["tree"]}
files=[]
for remote in PATHS:
    local=SOURCES/remote.replace("/","__"); data=local.read_bytes()
    blob=hashlib.sha1(b"blob "+str(len(data)).encode()+b"\0"+data).hexdigest()
    if blob!=lookup[remote]["sha"]: raise ValueError("Downloaded file is not the pinned Git blob: "+remote)
    files.append({"local":str(local),"remote_path":remote,"commit":COMMIT,"git_blob_sha1":blob,"sha256":sha(local),
        "url":f"https://github.com/byu-magicc/mavsim_public/blob/{COMMIT}/{remote}",
        "raw_url":f"https://raw.githubusercontent.com/byu-magicc/mavsim_public/{COMMIT}/{remote}"})
needed={"mass","Jy","S_wing","b","c","rho","e","gravity","C_L_0","C_L_alpha","C_L_q","C_L_delta_e",
        "C_D_p","C_D_q","C_D_delta_e","C_m_0","C_m_alpha","C_m_q","C_m_delta_e"}
source=SOURCES/"mavsim_python__parameters__aerosonde_parameters.py"
parameters={}; evidence={}
for node in ast.parse(source.read_text()).body:
    if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name) and node.targets[0].id in needed:
        try: value=float(ast.literal_eval(node.value))
        except (ValueError,TypeError): continue
        parameters[node.targets[0].id]=value; evidence[node.targets[0].id]={"source_line":node.lineno,"value":value}
if set(parameters)!=needed: raise ValueError("Missing numeric literal parameters")
parameters["AR"]=parameters["b"]**2/parameters["S_wing"]
book=Path("E:/rfly-books/Rfly-Stable-Learning-Book/ch6/chapter6.tex")
v1=HERE.parent/"20261023_v1/SOURCE_MODEL_GAPS.json"
write("PARAMETERS.json",{"source_sha256":sha(source),"parameters":parameters,"literal_evidence":evidence,
    "derived":{"AR":"b^2/S_wing"},"not_imported_as_code":True,
    "units":{"mass":"kg","Jy":"kg*m^2","S_wing":"m^2","b":"m","c":"m","rho":"kg/m^3","gravity":"m/s^2",
             "angles":"rad","qhat":"c*q/(2*v), dimensionless","coefficients":"dimensionless; alpha and delta_e expressed in radians"},
    "source_coefficients_deliberately_unused":["C_D_0","C_D_alpha","M","alpha0","epsilon","all lateral and motor/propeller coefficients"]})
write("SOURCE_MANIFEST.json",{"created_utc":datetime.now(timezone.utc).isoformat(),"repository":"https://github.com/byu-magicc/mavsim_public",
    "commit":COMMIT,"license":"GPL-3.0; complete original LICENSE saved with source files", "files":files,
    "book":{"path":str(book),"sha256":sha(book),"lines":[626,707]},"prior_v1":{"path":str(v1),"sha256":sha(v1)},
    "source_gap":"Upstream README identifies template project; retrieved dynamics/trim contain TODOs. Only public parameter literals are reused. New longitudinal equations are an explicitly declared engineering model, not upstream dynamics validation.",
    "network_note":"Git TLS backend rejected local certificate chain; HTTPS PowerShell verified requests to GitHub API/raw endpoints succeeded. No TLS-verification bypass or global settings change.",
    "activity_counts":{"environment_instances":0,"physics_steps":0,"training_updates":0,"dependencies_installed":0}})
print(json.dumps({"verified_git_blobs":len(files),"literal_parameters":len(needed),"commit":COMMIT}))

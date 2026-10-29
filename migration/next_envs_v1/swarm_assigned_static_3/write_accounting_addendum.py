"""Clarify saved counters; no environment imports or numerical solver calls."""
import json
import hashlib
from datetime import datetime,timezone
from pathlib import Path
HERE=Path(__file__).resolve().parent
def read(path): return json.loads(path.read_text(encoding="utf-8"))
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
short_path=HERE/"checks/short_v1/SUMMARY.json"
evidence_path=HERE/"checks/short_v1/SAVED_EVIDENCE.json"
full_path=HERE/"checks/full_v1/SUMMARY.json"
short, saved, full=read(short_path),read(evidence_path),read(full_path)
reference=len(saved["static_QPs"])
assert reference==24 and all(r["SLSQP_success"] for r in saved["static_QPs"])
assert short["counts"]["QP_solver_calls"]==51 and short["counts"]["physical_steps"]==25
assert full["counts"]["QP_calls"]==1800 and full["counts"]["physical_steps"]==1800
record={"utc":datetime.now(timezone.utc).isoformat(),"scope":"counter interpretation addendum; original frozen evidence unchanged",
  "source_sha256":{str(p.relative_to(HERE)):sha(p) for p in (short_path,evidence_path,full_path)},
  "short":{"primary_QP_call_attempts":51,"reference_SLSQP_calls":24,"total_numerical_QP_call_attempts":75,
           "physical_steps":25,"primary_iteration_budget_zero_failure_calls_included":1,
           "query_only_injected_LinAlgError_without_numerical_solver_call":1},
  "full":{"primary_QP_call_attempts":1800,"reference_SLSQP_calls":0,"total_numerical_QP_call_attempts":1800,"physical_steps":1800},
  "total":{"primary_QP_call_attempts":1851,"reference_SLSQP_calls":24,"total_numerical_QP_call_attempts":1875,
           "physical_steps":1825,"full_episodes":3,"optimizer_instances":0,"training_updates":0},
  "interpretation":["The historical field QP_solver_calls in short/SUMMARY.json counts only the custom active-set solver, not scipy SLSQP.",
      "Twenty-four SLSQP reference calls were executed in the same predeclared short check; they are additional numerical QP calls but produce no physical steps.",
      "Call attempts are not all converged solutions: one primary call deliberately used max_iterations=0 and safely failed.",
      "One separate query-only injected LinAlgError probe directly raises before any numerical solver; it is disclosed separately and adds no physical step.",
      "Independent saved-arithmetic reviews and plotting add zero physical steps and zero numerical QP solves."],
  "new_work_in_addendum":{"physical_steps":0,"numerical_QP_solves":0}}
out=HERE/"ACCOUNTING_ADDENDUM.json"
if out.exists(): raise RuntimeError("Do not overwrite accounting evidence")
out.write_text(json.dumps(record,indent=2)+"\n",encoding="utf-8")
print(json.dumps(record["total"]))

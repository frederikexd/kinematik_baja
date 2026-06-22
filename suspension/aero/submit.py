# ============================================================================
#  Elbee Racing — Baja SAE suspension & vehicle-dynamics studio.
#  Rebased for Elbee Racing from KinematiK by Frederik Thio (FSAE-EV, MIT).
#  Original engine © 2026 Frederik Thio; Baja rebase retains the MIT license.
# ============================================================================

"""
Submission seam — where the run matrix meets compute.

A 20–50M-cell case is not something a Streamlit process runs in-band; it goes to a
cluster. This module separates WHAT to run (the CaseSpec list, from the solver
backend) from WHERE to run it (a Submitter). Two submitters ship:

  * `LocalSubmitter`  — runs cases in-process, sequentially. Only sane for the
    reference backend or a genuinely tiny OpenFOAM case; it is what the tests use.
  * `SlurmSSHSubmitter` — an HONEST STUB. It generates a correct sbatch script and
    documents the rsync+ssh round-trip, but does not silently pretend to have a
    cluster: with no SSH target configured it raises, with one configured it shells
    out to ssh/rsync/sbatch. KinematiK submits; the cluster (the team's, with their
    solver license) does the work.

The point mirrors the rest of the codebase: own the seam cleanly, never fake the
expensive thing on the other side of it.
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass, field
from typing import Callable, Optional, Protocol

from .cfd import CaseSpec, CoeffResult, SolverUnavailable


@dataclass
class SubmitResult:
    """Outcome of one case: either a CoeffResult, or an error captured (not raised)."""
    spec: CaseSpec
    result: Optional[CoeffResult] = None
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.result is not None and self.result.is_usable()


class Submitter(Protocol):
    def submit_all(self, backend, specs: list[CaseSpec], workdir: str,
                   progress: Optional[Callable[[int, int, SubmitResult], None]] = None
                   ) -> list[SubmitResult]: ...


class LocalSubmitter:
    """Run cases here, one at a time. Errors are captured per-case, never abort the
    sweep — a failed attitude leaves a hole in the map, it doesn't lose the rest."""
    name = "local"

    def submit_all(self, backend, specs, workdir, progress=None):
        os.makedirs(workdir, exist_ok=True)
        out: list[SubmitResult] = []
        n = len(specs)
        for i, spec in enumerate(specs):
            sr = SubmitResult(spec=spec)
            try:
                sr.result = backend.run_case(spec, workdir)
            except SolverUnavailable as e:
                sr.error = f"solver unavailable: {e}"
            except Exception as e:                      # noqa: BLE001 — capture, don't crash sweep
                sr.error = f"{type(e).__name__}: {e}"
            out.append(sr)
            if progress:
                progress(i + 1, n, sr)
        return out


@dataclass
class SlurmSSHSubmitter:
    """
    HONEST STUB for cluster submission. Configure `ssh_target` (user@host) and
    `remote_dir`; it writes an sbatch array script, and — only if a target is set —
    rsyncs the cases up, submits, and documents pulling results back. With no target
    it raises a clear error instead of pretending a cluster exists.
    """
    ssh_target: str = ""               # "you@hpc.university.edu"
    remote_dir: str = ""               # "/scratch/$USER/kinematik"
    partition: str = "compute"
    cores_per_case: int = 64
    minutes_per_case: int = 240
    solver_module: str = "openfoam"    # `module load` name on the cluster
    name: str = field(default="slurm-ssh", init=False)

    def write_sbatch(self, specs: list[CaseSpec], workdir: str) -> str:
        case_names = [s.case_name() for s in specs]
        list_path = os.path.join(workdir, "cases.txt")
        with open(list_path, "w") as f:
            f.write("\n".join(case_names) + "\n")
        sbatch = os.path.join(workdir, "run_array.sbatch")
        script = f"""#!/bin/bash
#SBATCH --job-name=kinematik_aero
#SBATCH --partition={self.partition}
#SBATCH --array=1-{len(specs)}
#SBATCH --ntasks={self.cores_per_case}
#SBATCH --time={self.minutes_per_case}
#SBATCH --output=%x_%A_%a.out

module load {self.solver_module}
CASE=$(sed -n "${{SLURM_ARRAY_TASK_ID}}p" cases.txt)
cd "$CASE" || exit 1
# decompose + run + reconstruct; team adjusts to their mesh/decomposition:
decomposePar -force
mpirun -np {self.cores_per_case} simpleFoam -parallel
reconstructPar -latestTime
"""
        with open(sbatch, "w") as f:
            f.write(script)
        return sbatch

    def submit_all(self, backend, specs, workdir, progress=None):
        # Always write the cases + sbatch so the artefacts exist regardless.
        os.makedirs(workdir, exist_ok=True)
        for spec in specs:
            backend.write_case(spec, workdir)
        sbatch = self.write_sbatch(specs, workdir)
        if not self.ssh_target or not self.remote_dir:
            raise SolverUnavailable(
                "No cluster configured. Cases and an sbatch array script were written "
                f"to {workdir} ({os.path.basename(sbatch)}). Set ssh_target and "
                "remote_dir to have KinematiK rsync + sbatch them, or copy the folder "
                "to your cluster and `sbatch run_array.sbatch` yourself, then read "
                "results back with the backend's read_result().")
        # Real round-trip (best-effort; surfaces real ssh/rsync errors).
        subprocess.run(["rsync", "-az", workdir + "/",
                        f"{self.ssh_target}:{self.remote_dir}/"], check=True)
        subprocess.run(["ssh", self.ssh_target,
                        f"cd {self.remote_dir} && sbatch run_array.sbatch"], check=True)
        raise SolverUnavailable(
            "Submitted to the cluster. Cases are queued; this is an asynchronous "
            "batch — once the array job finishes, rsync the case folders back and "
            "call the backend's read_result() per case to assemble the map.")

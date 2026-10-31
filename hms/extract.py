"""Feature extraction: raw memory dump -> one CIC-MalMem-2022 feature row.

Runs Volatility 3 plugins on a Windows memory image, saves each plugin's JSON
output, and turns it into the 55 columns the structural classifier was trained
on. The column definitions follow the CIC-MalMem-2022 table (VolMemLyzer over
Volatility 2): for example ``malfind.protection`` is the sum of the vol2
numeric protection codes and ``malfind.uniqueInjections`` is injections per
injected process, which is what the published values work out to.

Volatility 3 differs from Volatility 2 in places, so a few columns are
approximations (listed in ``APPROXIMATED``):

  * vol3 psxview has no pspcid, session or deskthrd sources. ``pspcid`` is
    taken from the pslist column (the two are nearly identical in the CIC
    data), ``session`` counts processes with no session id, and ``deskthrd``
    is left missing (XGBoost handles missing values).
  * ``pslist.nprocs64bit`` is always 0 in the CIC data, so it is set to 0.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .allowlist import hidden_process_severity, injection_benign_reason, module_benign_reason
from .structural import cross_plugin_agreement, load_vol_json, unlinked_dll_ratio

# Plugin names to try in order; vol3 2.2x moved several under windows.malware.
PLUGINS = {
    "pslist": ("windows.pslist",),
    "dlllist": ("windows.dlllist",),
    "handles": ("windows.handles",),
    "ldrmodules": ("windows.malware.ldrmodules", "windows.ldrmodules"),
    "malfind": ("windows.malware.malfind", "windows.malfind"),
    "psxview": ("windows.malware.psxview", "windows.psxview"),
    "modules": ("windows.modules",),
    "svcscan": ("windows.svcscan",),
    "callbacks": ("windows.callbacks",),
    "psscan": ("windows.psscan",),
}

# Extra checks for hiding techniques. They add findings to the report but no
# model features, so a plugin missing from an older Volatility is skipped
# (and listed as skipped) instead of stopping the scan.
DETECTORS = {
    "drivermodule": ("windows.malware.drivermodule", "windows.drivermodule"),
    "svcdiff": ("windows.malware.svcdiff", "windows.svcdiff"),
    "hollowprocesses": ("windows.malware.hollowprocesses", "windows.hollowprocesses"),
    "processghosting": ("windows.malware.processghosting", "windows.processghosting"),
    "pebmasquerade": ("windows.malware.pebmasquerade",),
    "orphan_kernel_threads": ("windows.orphan_kernel_threads",),
    "suspicious_threads": ("windows.malware.suspicious_threads", "windows.suspicious_threads"),
    # every thread with its exit time; tells a live hidden process from a leftover
    "thrdscan": ("windows.thrdscan",),
}

# The 55 feature columns of CIC-MalMem-2022, in table order.
FEATURE_COLUMNS = (
    "pslist.nproc", "pslist.nppid", "pslist.avg_threads", "pslist.nprocs64bit",
    "pslist.avg_handlers", "dlllist.ndlls", "dlllist.avg_dlls_per_proc", "handles.nhandles",
    "handles.avg_handles_per_proc", "handles.nport", "handles.nfile", "handles.nevent",
    "handles.ndesktop", "handles.nkey", "handles.nthread", "handles.ndirectory",
    "handles.nsemaphore", "handles.ntimer", "handles.nsection", "handles.nmutant",
    "ldrmodules.not_in_load", "ldrmodules.not_in_init", "ldrmodules.not_in_mem",
    "ldrmodules.not_in_load_avg", "ldrmodules.not_in_init_avg", "ldrmodules.not_in_mem_avg",
    "malfind.ninjections", "malfind.commitCharge", "malfind.protection",
    "malfind.uniqueInjections", "psxview.not_in_pslist", "psxview.not_in_eprocess_pool",
    "psxview.not_in_ethread_pool", "psxview.not_in_pspcid_list",
    "psxview.not_in_csrss_handles", "psxview.not_in_session", "psxview.not_in_deskthrd",
    "psxview.not_in_pslist_false_avg", "psxview.not_in_eprocess_pool_false_avg",
    "psxview.not_in_ethread_pool_false_avg", "psxview.not_in_pspcid_list_false_avg",
    "psxview.not_in_csrss_handles_false_avg", "psxview.not_in_session_false_avg",
    "psxview.not_in_deskthrd_false_avg", "modules.nmodules", "svcscan.nservices",
    "svcscan.kernel_drivers", "svcscan.fs_drivers", "svcscan.process_services",
    "svcscan.shared_process_services", "svcscan.interactive_process_services",
    "svcscan.nactive", "callbacks.ncallbacks", "callbacks.nanonymous", "callbacks.ngeneric",
)

# Findings that corroborate a high structural score: every critical finding,
# plus these high ones, which don't depend on how a VM is set up. Everything
# else (unlinked DLLs, name spoofing, plain injected code, ...) is listed for
# review but never corroborates.
CORROBORATING = frozenset({"hidden service", "process hollowing", "process ghosting",
                           "orphan kernel thread", "anonymous kernel callback"})

APPROXIMATED = (
    "psxview.not_in_pspcid_list", "psxview.not_in_pspcid_list_false_avg",
    "psxview.not_in_session", "psxview.not_in_session_false_avg",
    "psxview.not_in_deskthrd", "psxview.not_in_deskthrd_false_avg",
    "pslist.nprocs64bit",
)

# Volatility 2 numeric codes for VAD protection, as stored in CIC-MalMem-2022.
PROTECTION_CODES = {
    "PAGE_NOACCESS": 0, "PAGE_READONLY": 1, "PAGE_EXECUTE": 2, "PAGE_EXECUTE_READ": 3,
    "PAGE_READWRITE": 4, "PAGE_WRITECOPY": 5, "PAGE_EXECUTE_READWRITE": 6,
    "PAGE_EXECUTE_WRITECOPY": 7,
}

HANDLE_TYPES = ("Port", "File", "Event", "Desktop", "Key", "Thread", "Directory",
                "Semaphore", "Timer", "Section", "Mutant")


# --------------------------------------------------------------------------
# Running Volatility
# --------------------------------------------------------------------------

def vol_command() -> list[str]:
    """The Volatility 3 command line: ``vol`` next to this Python, on PATH, or the module."""
    scripts = Path(sys.executable).parent
    for name in ("vol.exe", "vol"):
        if (scripts / name).exists():
            return [str(scripts / name)]
    found = shutil.which("vol")
    if found:
        return [found]
    return [sys.executable, "-c", "from volatility3.cli import main; main()"]


def run_plugins(dump: str | Path, out_dir: str | Path, verbose: bool = True) -> dict[str, Path]:
    """Run every plugin on ``dump`` and save ``<out_dir>/<plugin>.json``.

    A plugin whose JSON already exists is not run again, so a second scan of
    the same dump is quick and you can also drop in output you made yourself.
    A file cut short (for example by a crash while it was written) is run
    again. A detector (``DETECTORS``) that fails does not stop the run. When
    the plugin does not exist in this Volatility, ``<plugin>.failed`` records
    that so it is not tried on every scan; any other failure is retried next
    time.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = vol_command()
    flags = subprocess.CREATE_NO_WINDOW if sys.platform.startswith("win") else 0
    paths = {}
    for key, names in {**PLUGINS, **DETECTORS}.items():
        path = out_dir / f"{key}.json"
        failed = out_dir / f"{key}.failed"
        if path.exists() and not _complete_json(path):
            path.unlink()
        if not path.exists() and not failed.exists():
            errors, missing = [], True
            for name in names:
                if verbose:
                    print(f"running {name} ...", flush=True)
                proc = subprocess.run(
                    cmd + ["-q", "-r", "json", "-f", str(dump), name],
                    capture_output=True, text=True, encoding="utf-8", errors="replace",
                    creationflags=flags,
                )
                if proc.returncode == 0 and proc.stdout.lstrip().startswith("["):
                    tmp = path.with_suffix(".json.tmp")
                    tmp.write_text(proc.stdout, encoding="utf-8")
                    os.replace(tmp, path)          # never leave a half-written file
                    break
                message = (proc.stderr or proc.stdout).strip()
                missing &= "invalid choice" in message
                errors.append(f"{name}: {message[-400:]}")
            else:
                if key in PLUGINS:
                    raise RuntimeError(f"Volatility could not run {key}:\n" + "\n".join(errors))
                if missing:
                    failed.write_text("\n".join(errors), encoding="utf-8")
                elif verbose:
                    print(f"  {key} failed on this dump; it will be tried again next scan", flush=True)
        if path.exists():
            paths[key] = path
    return paths


# --------------------------------------------------------------------------
# JSON -> features
# --------------------------------------------------------------------------

def _flag(value) -> bool:
    return value is True or str(value).lower() == "true"


def _div(a, b) -> float:
    return float(a) / b if b else 0.0


def _count_false(rows: list[dict], col: str) -> int:
    return sum(1 for r in rows if col in r and not _flag(r[col]))


@dataclass
class Extraction:
    features: dict
    hidden_processes: list[dict] = field(default_factory=list)
    injections: list[dict] = field(default_factory=list)
    unlinked_dlls: list[dict] = field(default_factory=list)
    agr: float | None = None              # pslist vs psscan, exact (report 5A)
    udr: float | None = None              # unlinked-DLL ratio, exact (report 5A)
    dkom_pids: list = field(default_factory=list)       # live, in psscan, not in pslist
    unbacked_modules: list[dict] = field(default_factory=list)
    loader_missing: dict = field(default_factory=dict)
    pids: list = field(default_factory=list)            # pslist, for snapshot comparison
    anonymous_callbacks: list[dict] = field(default_factory=list)
    detections: list[dict] = field(default_factory=list)  # findings from DETECTORS
    checks_skipped: list[str] = field(default_factory=list)
    approximated: tuple = APPROXIMATED

    def findings(self) -> list[dict]:
        """Every suspicious item found, most severe first.

        critical  traces of hiding that a clean Windows system does not
                  produce; any one makes the verdict MALICIOUS
        high      strong indicators that can rarely occur legitimately
        medium    common in malware, but legitimate software (JIT compilers,
                  security products) can produce them too
        low       matches a known benign explanation (see allowlist.py),
                  given in ``benign_explanation``; listed, never decisive
        """
        out = []

        def add(severity, kind, item, detail, process_key="process"):
            f = {"severity": severity, "type": kind, "pid": item.get("pid"),
                 "process": item.get(process_key), "detail": detail}
            if item.get("benign_explanation"):
                f["benign_explanation"] = item["benign_explanation"]
            out.append(f)

        for p in self.hidden_processes:
            name = p.get("name") or "unknown"
            if p["pid"] in self.dkom_pids:
                add(p.get("severity", "critical"), "hidden process (DKOM)", p,
                    f"PID {p['pid']} ({name}) is in memory (psscan) but unlinked from the "
                    f"active process list (pslist)", "name")
            elif not p.get("psscan") and p.get("live_threads"):
                # Missing from the process list AND the pool scan, yet at least
                # one of its threads has not exited: stronger hiding than DKOM,
                # so it is critical too. If every thread has exited (or the
                # thread scan could not run), it may be a leftover: high.
                add("critical", "hidden process (pool scan evaded)", p,
                    f"PID {p['pid']} ({name}) has {p['live_threads']} thread(s) with no exit time "
                    f"but is missing from both pslist and psscan", "name")
            else:
                sev = p.get("severity", "high")
                add("high" if sev == "critical" else sev, "process hidden from some views", p,
                    f"PID {p['pid']} ({name}) is missing from pslist but seen by thread "
                    f"scanning", "name")
        for m in self.unbacked_modules:
            add("critical", "reflectively loaded module", m,
                f"module at {_hex(m['base'])} in PID {m['pid']} ({m['process']}) is unlinked "
                f"from all PEB loader lists and has no file")
        unbacked = {(m["pid"], m["base"]) for m in self.unbacked_modules}
        for m in self.unlinked_dlls:
            if (m["pid"], m["base"]) not in unbacked:
                add("low" if m.get("benign_explanation") else "high", "unlinked DLL", m,
                    f"{m['path']} in PID {m['pid']} ({m['process']}) is mapped but missing "
                    f"from the loader lists and dlllist")
        for i in self.injections:
            pe, benign = i.get("pe_header"), i.get("benign_explanation")
            if pe:
                # A whole program in private memory stays critical whatever the
                # process name; the benign cause is kept as a note to check.
                sev = "critical"
            else:
                sev = "low" if benign else "medium"
            add(sev, "injected PE image" if pe else "injected code", i,
                f"{i['protection']} private memory at {_hex(i['start'])} in PID {i['pid']} "
                f"({i['process']})" + (" starts with an MZ/PE header" if pe else " (malfind)"))
        for c in self.anonymous_callbacks:
            out.append({"severity": "high", "type": "anonymous kernel callback", "pid": None,
                        "process": None,
                        "detail": f"{c['type']} callback at {_hex(c['address'])} points outside "
                                  f"every loaded driver"})
        out.extend(self.detections)
        order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        return sorted(out, key=lambda x: order[x["severity"]])

    def alerts(self) -> list[str]:
        return [f["detail"] for f in self.findings() if f["severity"] == "critical"]

    def high(self, corroborating_only: bool = False) -> list[str]:
        """High-severity findings; with ``corroborating_only``, only the types in ``CORROBORATING``."""
        return [f["detail"] for f in self.findings() if f["severity"] == "high"
                and (not corroborating_only or f["type"] in CORROBORATING)]

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame([self.features], columns=list(FEATURE_COLUMNS))

    def evidence(self, limit: int = 50) -> dict:
        return {
            "findings": self.findings()[:limit * 4],
            "checks_skipped": self.checks_skipped,
            "cross_plugin_agreement": self.agr,
            "dkom_hidden_pids": self.dkom_pids,
            "unlinked_dll_ratio": self.udr,
            "modules_missing_per_list": self.loader_missing,
            "unbacked_unlinked_modules": self.unbacked_modules[:limit],
            "hidden_processes": self.hidden_processes[:limit],
            "injections": self.injections[:limit],
            "unlinked_dlls": self.unlinked_dlls[:limit],
            "approximated_features": list(self.approximated),
        }


def _complete_json(path: Path) -> bool:
    """Cheap check that a plugin file is a whole JSON list (not cut short)."""
    with open(path, "rb") as fh:
        head = fh.read(64).lstrip()
        fh.seek(max(0, path.stat().st_size - 64))
        tail = fh.read().rstrip()
    return head.startswith(b"[") and tail.endswith(b"]")


def hexdump_bytes(value) -> bytes:
    """Bytes of a malfind hexdump, whatever the layout.

    Volatility 3's JSON output gives ``"4d 5a 90 00 ..."``. Its text layout
    puts an address before and an ASCII column after the hex on each line, so
    on every line only the run of two-digit hex groups is kept.
    """
    out = bytearray()
    for line in str(value or "").splitlines():
        tokens = line.strip().strip('"').split()
        if tokens and len(tokens[0]) > 2:              # leading address
            tokens = tokens[1:]
        for token in tokens[:16]:                      # text layout: 16 bytes, then ASCII
            if not _HEX_PAIR.fullmatch(token):
                break                                  # the ASCII column
            out.append(int(token, 16))
    return bytes(out)


_HEX_PAIR = re.compile(r"[0-9a-fA-F]{2}")


def json_safe(value):
    """NaN and infinity become null, so reports are valid JSON for any reader."""
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.generic):
        return json_safe(value.item())
    return value


def _hex(value) -> str:
    return f"0x{value:x}" if isinstance(value, int) else str(value)


def _starts_with_mz(row: dict) -> bool:
    """malfind region whose first bytes are 'MZ', the header of a Windows program."""
    notes = str(row.get("Notes") or "").lower()
    return hexdump_bytes(row.get("Hexdump")).startswith(b"MZ") or "mz header" in notes


def _finding(severity, kind, row, detail) -> dict:
    return {"severity": severity, "type": kind, "pid": row.get("PID"),
            "process": row.get("Process"), "detail": detail}


def detector_findings(vol_dir: str | Path) -> tuple[list[dict], list[str]]:
    """Findings from the ``DETECTORS`` plugins, and the checks that could not run."""
    vol_dir = Path(vol_dir)
    out, skipped = [], []
    data = {}
    for key in DETECTORS:
        path = vol_dir / f"{key}.json"
        if path.exists():
            data[key] = load_vol_json(path)
        else:
            skipped.append(key)
    for r in data.get("drivermodule", []):
        if not _flag(r.get("Known Exception")):
            out.append(_finding("critical", "hidden driver (rootkit)", r,
                                f"driver {r.get('Driver Name')} has no entry in the loaded "
                                f"module list"))
    for r in data.get("svcdiff", []):
        out.append(_finding("high", "hidden service", r,
                            f"service {r.get('Name')} ({r.get('Binary') or 'no binary'}) is in "
                            f"memory but missing from the service list"))
    for r in data.get("hollowprocesses", []):
        notes = str(r.get("Notes") or "")
        strong = "does not match" in notes or "no VAD" in notes
        out.append(_finding("critical" if strong else "high", "process hollowing", r,
                            f"PID {r.get('PID')} ({r.get('Process')}): {notes}"))
    for r in data.get("processghosting", []):
        out.append(_finding("critical", "process ghosting", r,
                            f"PID {r.get('PID')} ({r.get('Process')}) runs from a file marked "
                            f"for deletion: {r.get('Path')}"))
    for r in data.get("pebmasquerade", []):
        if _flag(r.get("PEB_ImageFilePath_Spoofed")) or _flag(r.get("PEB_CommandLine_Spoofed")):
            out.append(_finding("high", "process name spoofing", r,
                                f"PID {r.get('PID')}: kernel says {r.get('EPROCESS_ImageFileName')}, "
                                f"PEB says {r.get('PEB_ImageFilePath')}"))
    for r in data.get("orphan_kernel_threads", []):
        out.append(_finding("high", "orphan kernel thread", r,
                            f"kernel thread {r.get('TID')} starts at {_hex(r.get('StartAddress'))}, "
                            f"outside every loaded driver"))
    for r in data.get("suspicious_threads", []):
        out.append(_finding("medium", "suspicious thread", r,
                            f"thread {r.get('TID')} in PID {r.get('PID')} ({r.get('Process')}) "
                            f"starts at {_hex(r.get('Address'))}: {r.get('Note')}"))
    return out, skipped


def features_from_json(vol_dir: str | Path) -> Extraction:
    """Build the CIC-MalMem-2022 feature row from saved plugin JSON."""
    vol_dir = Path(vol_dir)
    data = {key: load_vol_json(vol_dir / f"{key}.json") for key in PLUGINS}
    f: dict = {}

    ps = data["pslist"]
    f["pslist.nproc"] = len(ps)
    f["pslist.nppid"] = len({r.get("PPID") for r in ps})
    f["pslist.avg_threads"] = float(np.mean([r.get("Threads") or 0 for r in ps])) if ps else 0.0
    f["pslist.nprocs64bit"] = 0
    f["pslist.avg_handlers"] = float(np.mean([r.get("Handles") or 0 for r in ps])) if ps else 0.0

    dll = data["dlllist"]
    f["dlllist.ndlls"] = len(dll)
    f["dlllist.avg_dlls_per_proc"] = _div(len(dll), len({r.get("PID") for r in dll}))

    h = data["handles"]
    f["handles.nhandles"] = len(h)
    f["handles.avg_handles_per_proc"] = _div(len(h), len({r.get("PID") for r in h}))
    for t in HANDLE_TYPES:
        f[f"handles.n{t.lower()}"] = sum(1 for r in h if r.get("Type") == t)

    ldr = data["ldrmodules"]
    for col, name in (("InLoad", "load"), ("InInit", "init"), ("InMem", "mem")):
        n = _count_false(ldr, col)
        f[f"ldrmodules.not_in_{name}"] = n
        f[f"ldrmodules.not_in_{name}_avg"] = _div(n, len(ldr))

    mal = data["malfind"]
    pids = {r.get("PID") for r in mal}
    f["malfind.ninjections"] = len(mal)
    f["malfind.commitCharge"] = sum(int(r.get("CommitCharge") or 0) for r in mal)
    f["malfind.protection"] = sum(PROTECTION_CODES.get(str(r.get("Protection")), 0) for r in mal)
    f["malfind.uniqueInjections"] = _div(len(mal), len(pids))

    px = data["psxview"]
    no_session = {r.get("PID") for r in ps if r.get("SessionId") in (None, "N/A", "")}
    counts = {
        "pslist": _count_false(px, "pslist"),
        "eprocess_pool": _count_false(px, "psscan"),
        "ethread_pool": _count_false(px, "thrdscan"),
        "pspcid_list": _count_false(px, "pslist"),
        "csrss_handles": _count_false(px, "csrss"),
        "session": sum(1 for r in px if r.get("PID") in no_session or not _flag(r.get("pslist"))),
    }
    for name, n in counts.items():
        f[f"psxview.not_in_{name}"] = n
    f["psxview.not_in_deskthrd"] = np.nan
    for name, n in counts.items():
        f[f"psxview.not_in_{name}_false_avg"] = _div(n, len(px))
    f["psxview.not_in_deskthrd_false_avg"] = np.nan

    f["modules.nmodules"] = len(data["modules"])

    sv = data["svcscan"]
    types = [str(r.get("Type") or "") for r in sv]
    f["svcscan.nservices"] = len(sv)
    f["svcscan.kernel_drivers"] = sum("KERNEL_DRIVER" in t for t in types)
    f["svcscan.fs_drivers"] = sum("FILE_SYSTEM_DRIVER" in t for t in types)
    f["svcscan.process_services"] = sum("OWN_PROCESS" in t for t in types)
    f["svcscan.shared_process_services"] = sum("SHARE_PROCESS" in t for t in types)
    f["svcscan.interactive_process_services"] = sum("INTERACTIVE" in t for t in types)
    f["svcscan.nactive"] = sum("RUNNING" in str(r.get("State") or "") for r in sv)

    cb = data["callbacks"]
    f["callbacks.ncallbacks"] = len(cb)
    f["callbacks.nanonymous"] = sum(str(r.get("Module") or "UNKNOWN").upper() == "UNKNOWN" for r in cb)
    f["callbacks.ngeneric"] = sum(r.get("Type") == "GenericKernelCallback" for r in cb)

    # Evidence for the report: what a hiding technique left behind. Each item
    # is checked against known benign explanations (allowlist.py).
    scanned = {r.get("PID"): r for r in data["psscan"] if r.get("PID") is not None}
    views = {r.get("PID"): r for r in px if r.get("PID") is not None}

    def running(pid) -> bool:
        v, s = views.get(pid, {}), scanned.get(pid, {})
        return (_flag(v.get("thrdscan")) or _flag(v.get("csrss"))
                or int(s.get("Threads") or 0) > 0)

    # Threads with no exit time, per PID, from windows.thrdscan (None if it didn't run).
    live_threads = None
    if (vol_dir / "thrdscan.json").exists():
        live_threads = {}
        for r in load_vol_json(vol_dir / "thrdscan.json"):
            if r.get("PID") is not None and r.get("ExitTime") in (None, "", "N/A"):
                live_threads[r["PID"]] = live_threads.get(r["PID"], 0) + 1

    def confirmed(pid) -> bool:
        """Alive in a view independent of the process structure: a csrss handle or a live thread."""
        return (_flag(views.get(pid, {}).get("csrss"))
                or bool(live_threads and live_threads.get(pid, 0) > 0))

    def hidden_item(pid, name, row) -> dict:
        sev, why = hidden_process_severity(pid, name, running(pid), confirmed(pid))
        return {"pid": pid, "name": name, "pslist": False, "psscan": _flag(row.get("psscan", True)),
                "thrdscan": _flag(views.get(pid, {}).get("thrdscan")),
                "live_threads": live_threads.get(pid, 0) if live_threads is not None else None,
                "severity": sev,
                "benign_explanation": why}

    hidden = [
        hidden_item(r.get("PID"), r.get("Name"), r)
        for r in px
        if not _flag(r.get("pslist")) and (_flag(r.get("psscan")) or _flag(r.get("thrdscan")))
        and not r.get("Exit Time")
    ]
    dlls_by_pid: dict = {}
    for r in data["dlllist"]:
        dlls_by_pid.setdefault(r.get("PID"), set()).add(str(r.get("Name") or "").lower())
    injections = [
        {"pid": r.get("PID"), "process": r.get("Process"), "start": r.get("Start VPN"),
         "protection": r.get("Protection"), "commit_charge": r.get("CommitCharge"),
         "pe_header": _starts_with_mz(r),
         "benign_explanation": injection_benign_reason(r.get("Process"),
                                                       dlls_by_pid.get(r.get("PID"), set()),
                                                       hexdump_bytes(r.get("Hexdump")))}
        for r in mal
    ]
    loader = unlinked_dll_ratio(ldr, data["dlllist"])
    wow64 = {r.get("PID") for r in ps if _flag(r.get("Wow64"))}

    def module(r):
        return {"pid": r.get("Pid"), "process": r.get("Process"), "base": r.get("Base"),
                "path": r.get("MappedPath"),
                "benign_explanation": module_benign_reason(r.get("MappedPath"),
                                                           r.get("Pid") in wow64)}

    unlinked = [module(r) for r in loader.unlinked]
    agr = cross_plugin_agreement(ps, data["psscan"])
    known = {p["pid"] for p in hidden}
    for pid in agr.hidden_pids:
        if pid not in known:
            row = scanned.get(pid, {})
            hidden.append(hidden_item(pid, row.get("ImageFileName"), row))
    anonymous = [{"type": r.get("Type"), "address": r.get("Callback")}
                 for r in cb if str(r.get("Module") or "UNKNOWN").upper() == "UNKNOWN"]
    detections, skipped = detector_findings(vol_dir)
    return Extraction(f, hidden, injections, unlinked,
                      agr=agr.agreement, udr=loader.unlinked_ratio,
                      dkom_pids=agr.hidden_pids,
                      unbacked_modules=[module(r) for r in loader.unbacked],
                      loader_missing=loader.missing,
                      pids=sorted(_pid_set(ps)), anonymous_callbacks=anonymous,
                      detections=detections, checks_skipped=skipped)


def _pid_set(rows: list[dict]) -> set:
    return {r.get("PID") for r in rows if r.get("PID") is not None}


def compare_snapshots(first: Extraction, second: Extraction) -> dict:
    """Compare two dumps of the same VM taken some seconds apart (report section 4.2).

    Injected regions that are still there in the second dump are the stable
    ones; regions that vanish point to short-lived or self-erasing code, which
    is why a single snapshot can miss them.
    """
    def regions(ex):
        return {(i["pid"], i["start"]) for i in ex.injections}

    a, b = regions(first), regions(second)
    union = a | b
    p1, p2 = set(first.pids), set(second.pids)
    return {
        "processes_first": len(p1), "processes_second": len(p2),
        "processes_started": sorted(p2 - p1), "processes_gone": sorted(p1 - p2),
        "injections_first": len(a), "injections_second": len(b),
        "injections_persistent": len(a & b),
        "injections_vanished": [{"pid": pid, "start": _hex(start)} for pid, start in sorted(a - b, key=str)],
        "injections_new": [{"pid": pid, "start": _hex(start)} for pid, start in sorted(b - a, key=str)],
        "injection_jaccard": len(a & b) / len(union) if union else 1.0,
        "hidden_first": len(first.dkom_pids), "hidden_second": len(second.dkom_pids),
        "findings_first": first.findings(), "findings_second": second.findings(),
    }


SOURCE_FILE = "source.json"
_SAMPLE = 1 << 20


def dump_fingerprint(dump: str | Path) -> dict:
    """Size, name and a hash of the first, middle and last MB of a dump.

    Cheap enough for multi-GB files, and different for any new capture.
    """
    dump = Path(dump)
    size = dump.stat().st_size
    h = hashlib.sha256()
    with open(dump, "rb") as fh:
        for pos in sorted({0, max(0, size // 2 - _SAMPLE // 2), max(0, size - _SAMPLE)}):
            fh.seek(pos)
            h.update(fh.read(_SAMPLE))
    return {"name": dump.name, "size": size, "mtime_ns": dump.stat().st_mtime_ns,
            "sample_sha256": h.hexdigest()}


def volatility_version() -> str | None:
    try:
        from importlib.metadata import version

        return version("volatility3")
    except Exception:  # noqa: BLE001 - not installed, or installed without metadata
        return None


def _plugin_files(vol_dir: Path) -> list[Path]:
    """The files HMS itself writes in a cache folder; anything else there is left alone."""
    out = []
    for key in {**PLUGINS, **DETECTORS}:
        out += [vol_dir / f"{key}.json", vol_dir / f"{key}.json.tmp", vol_dir / f"{key}.failed"]
    return [f for f in out if f.exists()]


def prepare_cache(dump: Path, vol_dir: Path, verbose: bool = True, trust_unrecorded: bool = False) -> None:
    """Make sure the plugin output in ``vol_dir`` belongs to ``dump``.

    ``vol_dir/source.json`` records which dump the output came from (name,
    size, modification time and a hash of three 1 MB samples). If the dump is
    a different file (a new capture saved under the same name, or vm.raw and
    vm.mem sharing vm_vol), the old output is removed so Volatility runs again.
    Output without a record (made by hand, or by an older HMS) is only kept if
    it is newer than the dump, or if ``trust_unrecorded`` (a ``--vol-dir`` you
    named yourself). After a Volatility upgrade, checks that failed before are
    retried. Only HMS's own plugin files are ever removed.
    """
    vol_dir.mkdir(parents=True, exist_ok=True)
    record_path = vol_dir / SOURCE_FILE
    current = {**dump_fingerprint(dump), "volatility": volatility_version()}
    record = None
    if record_path.exists():
        try:
            record = json.loads(record_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            record = None
    same_dump = record is not None and all(record.get(k) == current[k]
                                           for k in ("name", "size", "mtime_ns", "sample_sha256"))
    if record is not None and not same_dump:
        if verbose:
            print(f"{vol_dir.name}: saved Volatility output is from a different dump "
                  f"({record.get('name')}), running Volatility again", flush=True)
        for f in _plugin_files(vol_dir):
            f.unlink()
    elif record is not None and record.get("volatility") != current["volatility"]:
        for f in vol_dir.glob("*.failed"):
            f.unlink()
    elif record is None and _plugin_files(vol_dir):
        newest = max(f.stat().st_mtime_ns for f in _plugin_files(vol_dir))
        if not trust_unrecorded and dump.stat().st_mtime_ns > newest:
            if verbose:
                print(f"{vol_dir.name}: saved Volatility output is older than the dump and has no "
                      f"record of its dump, running Volatility again", flush=True)
            for f in _plugin_files(vol_dir):
                f.unlink()
        elif verbose:
            print(f"{vol_dir.name}: using existing Volatility output (no record of its dump; "
                  f"delete the folder if it belongs to another capture)", flush=True)
    record_path.write_text(json.dumps(current, indent=2), encoding="utf-8")


def default_vol_dir(dump: Path) -> Path:
    """``<stem>_vol`` next to the dump, or ``<name>_vol`` if another dump with the
    same stem (vm.raw and vm.mem) already uses ``<stem>_vol``, so the two don't
    keep overwriting each other's results."""
    folder = dump.with_name(dump.stem + "_vol")
    try:
        owner = json.loads((folder / SOURCE_FILE).read_text(encoding="utf-8")).get("name")
    except (OSError, ValueError):
        return folder
    if owner and owner != dump.name and (dump.parent / owner).exists():
        return dump.with_name(dump.name + "_vol")
    return folder


def extract(dump: str | Path, vol_dir: str | Path | None = None, verbose: bool = True) -> Extraction:
    """Run Volatility on ``dump`` (cached in ``vol_dir``) and return its features."""
    dump = Path(dump)
    named = vol_dir is not None
    vol_dir = Path(vol_dir) if named else default_vol_dir(dump)
    prepare_cache(dump, vol_dir, verbose, trust_unrecorded=named)
    run_plugins(dump, vol_dir, verbose)
    try:
        return features_from_json(vol_dir)
    except ValueError:
        # A damaged plugin file that still looks complete: remove what doesn't
        # parse, run those plugins again, and read once more.
        for f in vol_dir.glob("*.json"):
            try:
                json.loads(f.read_text(encoding="utf-8"))
            except ValueError:
                if verbose:
                    print(f"{f.name} is damaged, running that plugin again", flush=True)
                f.unlink()
        run_plugins(dump, vol_dir, verbose)
        return features_from_json(vol_dir)


def save_row(ex: Extraction, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    ex.frame().to_csv(path, index=False)
    Path(path).with_suffix(".evidence.json").write_text(
        json.dumps(json_safe(ex.evidence()), indent=2, default=str, allow_nan=False))


def build_table(benign_dir: str | Path, malicious_dir: str | Path, out_csv: str | Path,
                verbose: bool = True) -> pd.DataFrame:
    """Feature table in the CIC-MalMem-2022 layout from your own dumps.

    Every dump under ``benign_dir`` becomes a Benign row and every dump under
    ``malicious_dir`` a Malware row. A dump in a subfolder of ``malicious_dir``
    (for example ``malicious/Ransomware/x.elf``) takes the subfolder name as
    its family, so ``train-family`` can use the table too. Volatility output
    is cached next to each dump, so a rebuild after adding dumps is quick.
    """
    from .preprocess import dump_files

    rows = []
    for label, folder in (("Benign", Path(benign_dir)), ("Malware", Path(malicious_dir))):
        files = dump_files(folder) if folder.is_dir() else []
        if not files:
            raise ValueError(f"no memory dumps found in {folder}")
        for dump in files:
            rel = dump.relative_to(folder).parts
            family = "Benign" if label == "Benign" else (rel[0] if len(rel) > 1 else "Malware")
            family = family.replace("-", "_")
            if verbose:
                print(f"[{label}] {dump.name}", flush=True)
            try:
                ex = extract(dump, verbose=verbose)
            except RuntimeError as exc:
                print(f"  skipped: Volatility could not read it\n  {str(exc)[-300:]}", flush=True)
                continue
            row = dict(ex.features)
            row["Category"] = "Benign" if label == "Benign" else f"{family}-{dump.stem.replace('-', '_')}-1"
            row["Class"] = label
            rows.append(row)
    df = pd.DataFrame(rows, columns=list(FEATURE_COLUMNS) + ["Category", "Class"])
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    if verbose:
        print(f"{len(df)} rows ({(df['Class'] == 'Benign').sum()} benign) -> {out_csv}")
    return df

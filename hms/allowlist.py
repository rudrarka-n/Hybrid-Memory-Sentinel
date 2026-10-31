"""Known-benign explanations for things that look like hiding.

A clean Windows system produces a few artefacts that resemble malware
tricks. HMS still lists them in the report, but lowers their severity and
says why, so they never decide the verdict on their own:

  * pool scans can find stale or half-built process structures (a process
    that exited, or memory being reused). Those have no running threads, a
    PID that is not a multiple of 4, or no name or start time.
  * Windows "minimal" processes (System, Registry, MemCompression, Secure
    System, vmmem) have no normal user-mode structures, so some views miss
    them.
  * 32-bit (WoW64) processes map the 32-bit system DLLs outside the 64-bit
    loader lists, and programs map DLLs, fonts and language files as plain
    data, which also leaves them out of the loader lists.
  * JIT compilers (.NET, PowerShell, browsers, Java, Office macros) and
    security products write and run code in private memory, which malfind
    reports. An executable region that is still all zeros holds no code yet.

Edit these lists to match the software on your own VM. A malicious program
can copy any name, so a match only lowers the severity of the weaker
findings. It never lowers a running process that is unlinked from the
process list (the minimal processes are normally in that list too) or an
injected PE image: those stay critical, and the possible benign cause is only
added to the finding as a note.
"""

from __future__ import annotations

MINIMAL_PROCESSES = {"system", "idle", "registry", "memcompression", "secure system",
                     "vmmem", "vmmemwsl"}

SECURITY_PROCESSES = {"msmpeng.exe", "nissrv.exe", "mssense.exe", "sensecncproxy.exe",
                      "mpdefendercoreservice.exe", "mbamservice.exe", "avp.exe",
                      "csfalconservice.exe", "sentinelagent.exe", "cylancesvc.exe", "ekrn.exe",
                      "bdservicehost.exe", "savservice.exe", "ccsvchst.exe"}

JIT_PROCESSES = {"powershell.exe", "pwsh.exe", "powershell_ise.exe", "chrome.exe", "msedge.exe",
                 "msedgewebview2.exe", "firefox.exe", "iexplore.exe", "opera.exe", "brave.exe",
                 "java.exe", "javaw.exe", "node.exe", "dotnet.exe", "w3wp.exe", "teams.exe",
                 "ms-teams.exe", "code.exe", "slack.exe", "discord.exe", "searchapp.exe",
                 "searchui.exe", "searchhost.exe", "widgets.exe", "onedrive.exe",
                 "smartscreen.exe", "mscorsvw.exe", "wmiprvse.exe", "winword.exe", "excel.exe",
                 "powerpnt.exe", "outlook.exe", "onenote.exe", "msaccess.exe"}

# A DLL from this list in a process means it runs a JIT compiler.
JIT_DLLS = {"clr.dll", "coreclr.dll", "clrjit.dll", "mscorwks.dll", "mscorjit.dll",
            "jscript9.dll", "chakra.dll", "jvm.dll", "v8.dll", "libcef.dll"}

# Files mapped as data rather than loaded as code.
DATA_EXTENSIONS = (".mui", ".nls", ".ttf", ".ttc", ".fon", ".otf", ".dat", ".db", ".cat",
                   ".rll", ".res", ".sdb", ".ime")

SYSTEM_DIRS = ("\\windows\\system32\\", "\\windows\\syswow64\\", "\\windows\\winsxs\\",
               "\\windows\\systemapps\\", "\\windows\\microsoft.net\\", "\\windows\\assembly\\",
               "\\windows\\immersivecontrolpanel\\", "\\program files\\",
               "\\program files (x86)\\", "\\windows\\explorer.exe")


def _lower(value) -> str:
    return str(value or "").lower()


def hidden_process_severity(pid, name, running: bool, confirmed: bool | None = None
                            ) -> tuple[str, str | None]:
    """Severity of a live process found by the pool scan but not in pslist.

    ``running``: some view (thread scan, csrss handles, psscan thread count)
    shows it has threads.
    ``confirmed``: a view independent of the process structure itself (a live
    thread in the thread scan, or a csrss handle) shows it alive. Defaults to
    ``running``.
    """
    confirmed = running if confirmed is None else confirmed
    n = _lower(name)
    plausible = isinstance(pid, int) and pid > 0 and pid % 4 == 0 and n.isprintable() and n
    if plausible and running:
        # Running and unlinked: a name match is no excuse, because the real
        # minimal processes are in the active process list and malware can
        # pick any name.
        return "critical", None
    if not plausible:
        if confirmed:
            # A rootkit that can unlink a process can also rewrite its PID or
            # name, so an independent sign of life keeps it critical.
            return "critical", None
        # Pool scanning finds stale, half-overwritten structures; the thread
        # count read from such a structure proves nothing.
        return "low", "looks like a stale or corrupted process structure (invalid PID or name)"
    if n in MINIMAL_PROCESSES:
        return "low", f"{name} is a Windows minimal process that some views do not list"
    return "medium", ("no running threads found, so it may be a process that was exiting "
                      "when the dump was taken")


def module_benign_reason(path, wow64: bool) -> str | None:
    """Why a mapped DLL missing from the loader lists is probably harmless, or None."""
    p = _lower(path).replace("/", "\\")
    if not p:
        return None
    if p.endswith(DATA_EXTENSIONS):
        return "language, font or data file mapped as data"
    if wow64 and "\\windows\\syswow64\\" in p:
        return "32-bit system DLL in a WoW64 process, which the 64-bit lists do not show"
    if any(d in p for d in SYSTEM_DIRS):
        return "Windows or installed-program file, probably mapped as a resource"
    return None


def injection_benign_reason(process, dlls: set[str], hexdump=None) -> str | None:
    """Why executable private memory in this process is probably harmless, or None."""
    if isinstance(hexdump, (bytes, bytearray)):
        empty = bool(hexdump) and not any(hexdump)
    else:
        data = _lower(hexdump).replace(" ", "")
        empty = bool(data) and set(data) == {"0"}
    if empty:
        return "the region is empty (all zero bytes), so no code has been written to it"
    n = _lower(process)
    if n in SECURITY_PROCESSES:
        return "security product, which unpacks and scans code in its own memory"
    if n in JIT_PROCESSES or dlls & JIT_DLLS:
        return "process runs a JIT compiler, which writes code to private memory"
    return None

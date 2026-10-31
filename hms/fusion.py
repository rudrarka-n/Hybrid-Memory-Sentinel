"""Hybrid soft-voting fusion engine (report section 5C).

    Verdict = MALICIOUS if (malicious_pages / valid_pages >= 0.20)
                        or (structural_confidence >= 0.90 and it is corroborated)
            = SUSPICIOUS if structural_confidence >= 0.90 on its own

The report's rule lets a structural confidence of 0.90 decide alone. The
bundled model was trained on the CIC-MalMem-2022 VMs and leans almost
entirely on two service counts, so on any other VM a high score can simply
mean "this VM is set up differently". By default a high score therefore only
makes the dump MALICIOUS when something else agrees: a high-severity hiding
finding (``support``) or the CNN. On its own it gives SUSPICIOUS.
``FusionConfig(structural_alone=True)`` restores the report's rule, for a
model trained on dumps of your own VM.

Each channel also casts a soft vote, its probability that the dump is
malicious: the structural confidence, and the mean page probability from the
CNN. ``soft_vote`` is their average (or the one vote when only one channel
ran). It is reported as the combined confidence.

Kernel invariants (report section 5A): critical findings from the
structural parser flag the dump immediately, whatever the classifier says.
They are the traces of malware hiding from tools inside the machine:
  * a live process found by the pool scan but missing from the active process
    list (DKOM unlinking),
  * a loaded module unlinked from all three PEB loader lists with no file
    behind it (reflective loading),
  * injected executable memory that starts with a PE header (an injected
    program or DLL).
See ``Extraction.findings`` in extract.py.

Either channel may be absent; the verdict then rests on the one that ran. A
byte channel with no candidate pages has nothing to judge, so it counts as
absent rather than as clean.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np


# Hiding checks whose findings are critical. If one of them could not run, a
# dump with no model score cannot be called clean.
CRITICAL_CHECKS = ("drivermodule", "hollowprocesses", "processghosting")


@dataclass
class FusionConfig:
    page_ratio_threshold: float = 0.20   # share of candidate pages flagged
    structural_threshold: float = 0.90   # structural confidence that alerts alone
    page_prob_threshold: float = 0.50    # per-page probability that counts as malicious
    kernel_invariants: bool = True       # critical structural findings alert on their own
    structural_alone: bool = False       # a high structural score decides without corroboration


@dataclass
class Verdict:
    label: str                            # MALICIOUS | SUSPICIOUS | BENIGN | INCONCLUSIVE
    structural_score: float | None = None
    malicious_page_ratio: float | None = None
    malicious_pages: int | None = None
    valid_pages: int | None = None
    page_vote: float | None = None        # mean CNN page probability
    soft_vote: float | None = None        # average of the channels' votes
    reasons: list[str] = field(default_factory=list)

    @property
    def malicious(self) -> bool:
        return self.label == "MALICIOUS"

    def to_dict(self) -> dict:
        return asdict(self)


def fuse(
    structural_score: float | None = None,
    page_probs: np.ndarray | None = None,
    cfg: FusionConfig | None = None,
    alerts: list[str] | None = None,
    support: list[str] | None = None,
    checks_ran: bool = False,
    review: list[str] | None = None,
    skipped: list[str] | None = None,
) -> Verdict:
    """``alerts``: descriptions of critical findings (kernel-invariant violations).

    ``support``: high-severity findings of the types in
    ``extract.CORROBORATING`` (hidden service, hollowing, ghosting, orphan
    kernel thread, anonymous callback). They never decide the verdict, but
    they corroborate a high structural score.

    ``review``: every high-severity finding, including ones a clean VM can
    produce (an unlinked DLL outside the system folders). They are mentioned
    in the reasons so they get a manual look, whatever the verdict.

    ``checks_ran``: the Volatility hiding checks ran. With no model and no
    critical or corroborating finding, that alone gives BENIGN, unless one of
    ``CRITICAL_CHECKS`` is among ``skipped`` (checks that could not run).
    """
    cfg = cfg or FusionConfig()
    v = Verdict(label="BENIGN")
    if page_probs is not None and len(page_probs) == 0:
        v.reasons.append("byte channel: no candidate pages survived filtering")
        page_probs = None
    alerts = list(alerts or []) if cfg.kernel_invariants else []
    support = list(support or [])
    review = list(review or []) or support
    skipped = list(skipped or [])
    notes = []
    if review:
        notes.append(f"{len(review)} high-severity finding(s) need a manual review")
    if skipped:
        notes.append(f"{len(skipped)} hiding check(s) could not run: {', '.join(skipped)}")
    note = "; ".join(notes)
    missing_critical = [c for c in skipped if c in CRITICAL_CHECKS]

    if structural_score is None and page_probs is None and not alerts:
        if support:
            v.label = "INCONCLUSIVE"
            v.reasons.append(f"no critical finding and no model score; {note}")
        elif checks_ran and missing_critical:
            v.label = "INCONCLUSIVE"
            v.reasons.append(f"no finding and no model score, but the hiding check(s) "
                             f"{', '.join(missing_critical)} could not run; {note}")
        elif checks_ran:
            v.reasons.append("the hiding checks ran and found nothing critical"
                             + (f"; {note}" if note else " or high") + "; no model score was used")
        else:
            v.label = "INCONCLUSIVE"
            v.reasons.append("no channel produced evidence")
        return v

    for alert in alerts:
        v.label = "MALICIOUS"
        v.reasons.append(f"critical finding: {alert}")

    pages_flag = False
    if page_probs is not None:
        probs = np.asarray(page_probs, dtype=float)
        v.valid_pages = int(len(probs))
        v.malicious_pages = int((probs >= cfg.page_prob_threshold).sum())
        v.malicious_page_ratio = v.malicious_pages / v.valid_pages if v.valid_pages else 0.0
        if v.valid_pages and v.malicious_page_ratio >= cfg.page_ratio_threshold:
            pages_flag = True
            v.label = "MALICIOUS"
            v.reasons.append(
                f"{v.malicious_pages}/{v.valid_pages} candidate pages flagged "
                f"({v.malicious_page_ratio:.1%} >= {cfg.page_ratio_threshold:.0%})"
            )

    if structural_score is not None:
        v.structural_score = float(structural_score)
        if v.structural_score >= cfg.structural_threshold:
            reason = f"structural confidence {v.structural_score:.3f} >= {cfg.structural_threshold:.2f}"
            if cfg.structural_alone or alerts or pages_flag or support:
                v.label = "MALICIOUS"
                if support and not (cfg.structural_alone or alerts or pages_flag):
                    reason += f", corroborated by {len(support)} high-severity finding(s)"
                v.reasons.append(reason)
            else:
                v.label = "SUSPICIOUS"
                v.reasons.append(
                    reason + ", but no hiding finding or CNN result agrees. The model was trained "
                    "on the CIC-MalMem-2022 VMs and may be reacting to this VM being set up "
                    "differently; check the findings or train a model on dumps of this VM"
                    + (f". {note[0].upper()}{note[1:]}" if note else "")
                )

    votes = []
    if v.structural_score is not None:
        votes.append(v.structural_score)
    if page_probs is not None and v.valid_pages:
        v.page_vote = float(np.mean(probs))
        votes.append(v.page_vote)
    if votes:
        v.soft_vote = float(np.mean(votes))

    if v.label == "BENIGN":
        v.reasons.append("neither channel crossed its threshold" + (f"; {note}" if note else ""))
    return v

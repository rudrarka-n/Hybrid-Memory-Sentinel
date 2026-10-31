"""Command-line interface.  Run ``python -m hms --help``."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

from .fusion import FusionConfig, fuse
from .metrics import binary_metrics, format_metrics
from .preprocess import FilterConfig, PageScanner, build_baseline, load_baseline

# scan exit codes, so scripts can tell "clean" from "not checked". 2 is a
# usage error (argparse), and 5 means HMS itself failed, so a crash is never
# mistaken for a detection.
EXIT_BENIGN, EXIT_MALICIOUS, EXIT_INCONCLUSIVE, EXIT_SUSPICIOUS, EXIT_ERROR = 0, 1, 3, 4, 5


def _write_report(path: str, data: dict) -> None:
    from .extract import json_safe

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(json_safe(data), indent=2, default=str, allow_nan=False))
    print(f"report -> {path}")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _print_stats(stats) -> None:
    rows = [
        ("total pages", stats.total_pages),
        ("  all-zero", stats.zero_pages),
        ("  identical to baseline", stats.baseline_duplicates),
        ("  repeated in dump", stats.self_duplicates),
        ("  low information", stats.low_information),
        ("  not code-like", stats.not_code_like),
        ("candidates kept", stats.candidates),
        ("  high entropy", stats.high_entropy_candidates),
        ("  code-like", stats.code_like_candidates),
    ]
    for name, value in rows:
        print(f"{name:<26}{value:>12,}")
    print(f"{'reduction':<26}{stats.reduction:>12.2%}")


def cmd_baseline(args) -> int:
    hashes = build_baseline(args.dump)
    out = Path(args.out)
    if out.suffix != ".npy":               # np.save would add it anyway
        out = out.with_name(out.name + ".npy")
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, hashes)
    print(f"baseline: {len(hashes):,} unique non-zero pages -> {out}")
    return 0


def cmd_filter(args) -> int:
    baseline = load_baseline(args.baseline) if args.baseline else None
    scanner = PageScanner(args.dump, baseline, FilterConfig(high_entropy=args.entropy))
    for _ in scanner:
        pass
    _print_stats(scanner.stats)
    return 0


def cmd_train_structural(args) -> int:
    from .structural import train_structural

    res = train_structural(args.csv, args.out, split=args.split, seed=args.seed,
                           baselines=args.baselines, model=args.model,
                           select_features=args.select)
    print(f"{res['n_train']:,} train / {res['n_test']:,} test rows, "
          f"{res['n_features']} features, {res['split']} split")
    for name, m in res["models"].items():
        print(f"{name:<22} skipped: {m['skipped']}" if "skipped" in m else format_metrics(name, m))
    for fam, r in res.get("family_recall", {}).items():
        print(f"  recall on {fam:<12} {r['recall']:.4f}  (n={r['n']})")
    if args.out:
        print(f"model -> {args.out}")
    return 0


def cmd_train_family(args) -> int:
    from .structural import train_family

    res = train_family(args.csv, args.out, split=args.split, seed=args.seed)
    print(f"{res['n_train']:,} train / {res['n_test']:,} test rows, {res['split']} split")
    print(f"accuracy {res['accuracy']:.4f}   macro F1 {res['macro_f1']:.4f}")
    for c, m in res["per_class"].items():
        print(f"  {c:<12} precision {m['precision']:.4f}  recall {m['recall']:.4f}  (n={int(m['support'])})")
    if args.out:
        print(f"model -> {args.out}")
    return 0


def cmd_temporal(args) -> int:
    from .extract import compare_snapshots, extract

    first = extract(args.first, args.vol_dir_first)
    second = extract(args.second, args.vol_dir_second)
    res = compare_snapshots(first, second)
    print(f"processes: {res['processes_first']} -> {res['processes_second']} "
          f"({len(res['processes_started'])} started, {len(res['processes_gone'])} gone)")
    print(f"injected regions: {res['injections_first']} -> {res['injections_second']}, "
          f"{res['injections_persistent']} persistent, Jaccard {res['injection_jaccard']:.3f}")
    print(f"hidden processes: {res['hidden_first']} -> {res['hidden_second']}")
    for i in res["injections_vanished"][:10]:
        print(f"  vanished: PID {i['pid']} at {i['start']}")
    for i in res["injections_new"][:10]:
        print(f"  new:      PID {i['pid']} at {i['start']}")
    if args.report:
        _write_report(args.report, res)
    return 0


def cmd_build_table(args) -> int:
    from .extract import build_table

    build_table(args.benign, args.malicious, args.out)
    return 0


def cmd_gui(args) -> int:
    try:
        from .gui import main as gui_main
    except ImportError as exc:  # tkinter missing from this Python
        print(f"the GUI needs tkinter, which this Python does not have: {exc}", file=sys.stderr)
        return 2
    return gui_main()


def cmd_train_cnn(args) -> int:
    from .cnn import train_cnn

    res = train_cnn(args.benign, args.malicious, args.out, epochs=args.epochs,
                    max_pages_per_file=args.max_pages, seed=args.seed)
    print(f"trained on {res['n_train_pages']:,} pages, validated on {res['n_val_pages']:,} "
          f"pages from held-out files")
    if args.out:
        print(f"model -> {args.out}")
    return 0


def cmd_extract(args) -> int:
    from .extract import extract, save_row

    ex = extract(args.dump, args.vol_dir)
    out = args.out or str(Path(args.dump).with_suffix(".features.csv"))
    save_row(ex, out)
    _print_evidence(ex)
    print(f"features -> {out}")
    return 0


def _print_evidence(ex, findings: list[dict] | None = None) -> None:
    if ex.agr is not None:
        print(f"cross-plugin agreement {ex.agr:>6.3f}   (pslist vs psscan; < 1 means a hidden process)")
    if ex.udr is not None:
        print(f"unlinked-DLL ratio     {ex.udr:>6.3f}")
    _print_findings(ex.findings() if findings is None else findings)
    if ex.checks_skipped:
        print(f"checks not run (plugin missing or failed): {', '.join(ex.checks_skipped)}")


def _print_findings(findings: list[dict], limit: int = 25) -> None:
    counts = {s: sum(f["severity"] == s for f in findings) for s in ("critical", "high", "medium", "low")}
    print(f"findings: {len(findings)}  (" + ", ".join(f"{n} {s}" for s, n in counts.items() if n) + ")"
          if findings else "findings: none")
    for f in findings[:limit]:
        print(f"  [{f['severity'].upper():<8}] {f['type']}: {f['detail']}")
    if len(findings) > limit:
        print(f"  ... {len(findings) - limit} more in the JSON report")


def cmd_vol_metrics(args) -> int:
    from .structural import cross_plugin_agreement, load_vol_json, unlinked_dll_ratio

    out = {}
    if args.pslist and args.psscan:
        agr = cross_plugin_agreement(load_vol_json(args.pslist), load_vol_json(args.psscan))
        out["cross_plugin_agreement"] = agr.agreement
        out["hidden_pids"] = agr.hidden_pids
    if args.ldrmodules:
        udr = unlinked_dll_ratio(load_vol_json(args.ldrmodules))
        out["unlinked_dll_ratio"] = udr.unlinked_ratio
        out["unlinked_modules"] = udr.unlinked[:50]
    if not out:
        print("give --pslist with --psscan, and/or --ldrmodules", file=sys.stderr)
        return 2
    print(json.dumps(out, indent=2))
    return 0


def _load_family(path: str, report: dict):
    """The family model, or None with a note: the family guess never stops a scan."""
    from .structural import load_structural

    try:
        return load_structural(path, "family")
    except Exception as exc:  # noqa: BLE001
        print(f"note: family model not used: {exc}", file=sys.stderr)
        report["family_error"] = str(exc)
        return None


def _guess_family(bundle, X, report: dict) -> list | None:
    from .structural import predict_family

    try:
        return predict_family(bundle, X)
    except Exception as exc:  # noqa: BLE001
        print(f"note: family guess failed: {exc}", file=sys.stderr)
        report["family_error"] = str(exc)
        return None


def cmd_scan(args) -> int:
    if not args.dump and not args.features:
        print("scan needs --dump and/or --features", file=sys.stderr)
        return 2
    if args.features and not args.structural:
        print("--features needs --structural <model>", file=sys.stderr)
        return 2
    # Check every input before the slow Volatility step, so a typo fails at once.
    for flag in ("dump", "features", "structural", "family", "baseline", "cnn"):
        value = getattr(args, flag)
        if value and not Path(value).is_file():
            print(f"--{flag}: {value} is not a file", file=sys.stderr)
            return 2
    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    if args.structural and args.dump and args.no_volatility and not args.features:
        print("note: --structural is ignored with --no-volatility (the model needs Volatility's "
              "features)", file=sys.stderr)
    from .structural import load_feature_table, load_structural, structural_scores

    report: dict = {}
    structural_bundle = load_structural(args.structural, "structural") if args.structural else None
    family_bundle = _load_family(args.family, report) if args.family else None
    baseline = load_baseline(args.baseline) if args.baseline else None
    model = None
    if args.cnn:
        from .cnn import load_cnn, predict_pages

        model = load_cnn(args.cnn)

    # A feature table on its own evaluates the classifier, so its score decides.
    cfg = FusionConfig(args.page_ratio, args.structural_threshold,
                       structural_alone=args.structural_decides or not args.dump)

    scores = labels = extraction = family = None
    timing: dict = {}
    if args.features:
        X, labels, _ = load_feature_table(args.features)
        if args.dump and not 0 <= args.row < len(X):
            print(f"--row {args.row} is outside the table, which has rows 0 to {len(X) - 1}",
                  file=sys.stderr)
            return 2
        scores = structural_scores(structural_bundle, X)
        families = _guess_family(family_bundle, X, report) if family_bundle else None
        if families:
            family = families[args.row] if args.dump else None
            report["families"] = [{"label": f, "probability": p} for f, p in families]
    if args.dump and not args.no_volatility:
        # Read the kernel structures with Volatility 3. This finds hiding
        # techniques even without a trained model; the model adds a score.
        from .extract import extract

        t0 = time.perf_counter()
        try:
            extraction = extract(args.dump, args.vol_dir)
        except RuntimeError as exc:
            print(f"structural channel: Volatility failed, continuing with the byte channel only\n{exc}",
                  file=sys.stderr)
            report["volatility_error"] = str(exc)[-2000:]
        timing["volatility_seconds"] = round(time.perf_counter() - t0, 3)
        # With --features, the table row gives the score; Volatility still runs
        # for the hiding checks.
        if not args.features:
            args.row = 0
        if structural_bundle is not None and extraction is not None and not args.features:
            scores = structural_scores(structural_bundle, extraction.frame())
        if family_bundle is not None and extraction is not None and not args.features:
            families = _guess_family(family_bundle, extraction.frame(), report)
            family = families[0] if families else None

    # Feature table only: one verdict per row (each row is one memory dump).
    if not args.dump:
        verdicts = [fuse(structural_score=s, cfg=cfg) for s in scores]
        flagged = sum(v.malicious for v in verdicts)
        print(f"{len(verdicts):,} dumps scored, {flagged:,} flagged MALICIOUS "
              f"(structural confidence >= {cfg.structural_threshold:.2f})")
        report = {"rows": len(verdicts), "flagged": flagged,
                  "verdicts": [v.to_dict() for v in verdicts], **report}
        if labels is not None:
            m = binary_metrics(labels, scores, threshold=cfg.structural_threshold)
            print(format_metrics("vs. ground truth", m))
            report["metrics"] = m
        if args.report:
            _write_report(args.report, report)
        return EXIT_MALICIOUS if flagged else EXIT_BENIGN

    dump = Path(args.dump)
    scanner = PageScanner(dump, baseline)

    probs, offsets, entropy, prologues = [], [], [], []
    t0 = time.perf_counter()
    cnn_seconds = 0.0
    for batch in scanner:
        offsets.append(batch.offsets)
        entropy.append(batch.entropy)
        prologues.append(batch.prologues)
        if model is not None:
            t1 = time.perf_counter()
            probs.append(predict_pages(model, batch.pages))
            cnn_seconds += time.perf_counter() - t1
    timing["byte_channel_seconds"] = round(time.perf_counter() - t0, 3)
    timing["filter_mb_per_second"] = round(
        dump.stat().st_size / 1e6 / max(timing["byte_channel_seconds"] - cnn_seconds, 1e-9), 1)
    if model is not None and scanner.stats.candidates:
        timing["cnn_ms_per_page"] = round(1000 * cnn_seconds / scanner.stats.candidates, 3)

    page_probs = None
    if model is not None:
        page_probs = np.concatenate(probs) if probs else np.zeros(0)
    structural_score = float(scores[args.row]) if scores is not None else None
    have = extraction is not None
    verdict = fuse(structural_score, page_probs, cfg,
                   alerts=extraction.alerts() if have else None,
                   support=extraction.high(corroborating_only=True) if have else None,
                   review=extraction.high() if have else None, checks_ran=have,
                   skipped=extraction.checks_skipped if have else None)

    print(f"dump: {dump}  ({dump.stat().st_size:,} bytes)")
    _print_stats(scanner.stats)
    if extraction is None and args.no_volatility:
        print("structural channel: skipped (--no-volatility)")
    if model is None:
        print("byte channel: no --cnn model given, pages were filtered but not classified")
    findings = list(extraction.findings()) if extraction is not None else []
    if structural_score is not None and structural_score >= cfg.structural_threshold:
        findings.append({"severity": "high", "type": "structural classifier", "pid": None,
                         "process": None,
                         "detail": f"boosted-tree confidence {structural_score:.3f} >= "
                                   f"{cfg.structural_threshold:.2f}"})
    off = ent = pro = None
    if offsets:
        off, ent, pro = map(np.concatenate, (offsets, entropy, prologues))
    if off is not None and len(off):
        if page_probs is not None:
            flagged = np.flatnonzero(page_probs >= cfg.page_prob_threshold)
            for i in flagged[np.argsort(-page_probs[flagged], kind="stable")][: args.top]:
                findings.append({"severity": "high", "type": "CNN-flagged page", "pid": None,
                                 "process": None,
                                 "detail": f"page at file offset 0x{int(off[i]):x} scored "
                                           f"{page_probs[i]:.3f} (entropy {ent[i]:.2f})"})
            if len(flagged) > args.top:
                findings.append({"severity": "high", "type": "CNN-flagged page", "pid": None,
                                 "process": None,
                                 "detail": f"{len(flagged) - args.top} more flagged pages, listed "
                                           f"under flagged_pages in the report"})
            report["flagged_pages"] = [
                {"offset": f"0x{int(off[i]):x}", "probability": round(float(page_probs[i]), 4)}
                for i in flagged[np.argsort(-page_probs[flagged], kind="stable")]]
        else:
            findings.append({"severity": "low", "type": "unclassified candidate pages", "pid": None,
                             "process": None,
                             "detail": f"{len(off):,} pages look like code or packed data and were "
                                       f"not classified (no --cnn model); see suspicious_pages"})
    order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
    findings.sort(key=lambda f: order[f["severity"]])
    print()
    if extraction is not None:
        _print_evidence(extraction, findings)
    else:
        _print_findings(findings)

    print(f"\nVERDICT: {verdict.label}")
    for reason in verdict.reasons:
        print(f"  - {reason}")
    if verdict.soft_vote is not None:
        print(f"  soft vote (combined confidence): {verdict.soft_vote:.3f}")
    disagrees = family is not None and family[0] == "Benign" and verdict.label in ("MALICIOUS", "SUSPICIOUS")
    if family is not None:
        print(f"  most likely family: {family[0]} ({family[1]:.2f})"
              + ("  (disagrees with the verdict; the family model is only a guess)" if disagrees else ""))
    print("timing: " + ", ".join(f"{k} {v}" for k, v in timing.items()))

    report = {
        **report,
        "dump": str(dump), "sha256": _sha256(dump), "size": dump.stat().st_size,
        "verdict": verdict.to_dict(), "findings": findings,
        "filter": scanner.stats.to_dict(),
    }
    if family is not None:
        report["family"] = {"label": family[0], "probability": family[1]}
        if disagrees:
            report["family"]["note"] = "disagrees with the verdict; the family model is only a guess"
    report["timing"] = timing
    if extraction is not None:
        report["structural_features"] = extraction.features
        report["evidence"] = extraction.evidence()
    if off is not None:
        # Rank by classifier probability when available, otherwise by entropy.
        rank = page_probs if page_probs is not None and len(page_probs) else ent
        top = np.argsort(-np.asarray(rank), kind="stable")[: args.top]
        report["suspicious_pages"] = [
            {
                "offset": f"0x{int(off[i]):x}",
                "probability": None if page_probs is None else round(float(page_probs[i]), 4),
                "entropy": round(float(ent[i]), 3),
                "prologues": int(pro[i]),
            }
            for i in top
        ]
    if args.report:
        _write_report(args.report, report)
    return {"MALICIOUS": EXIT_MALICIOUS, "SUSPICIOUS": EXIT_SUSPICIOUS,
            "INCONCLUSIVE": EXIT_INCONCLUSIVE}.get(verdict.label, EXIT_BENIGN)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="hms", description="Hybrid Memory Sentinel")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("baseline", help="hash the pages of a known-clean snapshot")
    s.add_argument("dump")
    s.add_argument("-o", "--out", default="models/baseline.npy")
    s.set_defaults(func=cmd_baseline)

    s = sub.add_parser("filter", help="show how many pages survive dedup and filtering")
    s.add_argument("dump")
    s.add_argument("--baseline")
    s.add_argument("--entropy", type=float, default=6.8)
    s.set_defaults(func=cmd_filter)

    s = sub.add_parser("train-structural", help="train the boosted-tree structural classifier")
    s.add_argument("--csv", required=True, help="CIC-MalMem-2022 style feature table")
    s.add_argument("--out", default="models/my_structural.joblib")
    s.add_argument("--split", choices=("group", "random"), default="group")
    s.add_argument("--model", choices=("xgboost", "lightgbm"), default="xgboost",
                   help="boosted tree to train and save")
    s.add_argument("--select", type=int, default=0, metavar="K",
                   help="keep the K best features by ANOVA F-score (0 = all)")
    s.add_argument("--baselines", action="store_true",
                   help="also evaluate the other boosted tree and RF, DT, KNN, LR, SVM")
    s.add_argument("--seed", type=int, default=42)
    s.set_defaults(func=cmd_train_structural)

    s = sub.add_parser("train-family", help="train the family classifier (Benign/Ransomware/Spyware/Trojan)")
    s.add_argument("--csv", required=True, help="CIC-MalMem-2022 style feature table")
    s.add_argument("--out", default="models/family_my_model.joblib")
    s.add_argument("--split", choices=("group", "random"), default="group")
    s.add_argument("--seed", type=int, default=42)
    s.set_defaults(func=cmd_train_family)

    s = sub.add_parser("temporal", help="compare two dumps of the same VM taken some seconds apart")
    s.add_argument("first", help="earlier dump")
    s.add_argument("second", help="later dump, e.g. 60 s after the first")
    s.add_argument("--vol-dir-first")
    s.add_argument("--vol-dir-second")
    s.add_argument("--report", help="write a JSON report here")
    s.set_defaults(func=cmd_temporal)

    s = sub.add_parser("train-cnn", help="train the 1D-CNN page classifier")
    s.add_argument("--benign", default="CNNDataSet/benign", help="folder of clean dumps")
    s.add_argument("--malicious", default="CNNDataSet/malicious", help="folder of infected dumps")
    s.add_argument("--out", default="models/cnn.pt")
    s.add_argument("--epochs", type=int, default=5)
    s.add_argument("--max-pages", type=int, default=512, help="pages sampled per file")
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(func=cmd_train_cnn)

    s = sub.add_parser("extract", help="run Volatility 3 on a dump and build its feature row")
    s.add_argument("dump", help="raw memory image")
    s.add_argument("-o", "--out", help="feature CSV (default: next to the dump)")
    s.add_argument("--vol-dir", help="where plugin JSON is saved and reused (default: <dump>_vol)")
    s.set_defaults(func=cmd_extract)

    s = sub.add_parser("build-table", help="feature table from your own benign and malicious dumps")
    s.add_argument("--benign", default="CNNDataSet/benign", help="folder of clean dumps")
    s.add_argument("--malicious", default="CNNDataSet/malicious",
                   help="folder of infected dumps (subfolders name the family)")
    s.add_argument("--out", default="dataset/my_dumps.csv")
    s.set_defaults(func=cmd_build_table)

    s = sub.add_parser("gui", help="open the HMS window")
    s.set_defaults(func=cmd_gui)

    s = sub.add_parser("vol-metrics", help="AGR and UDR from Volatility 3 JSON output")
    s.add_argument("--pslist")
    s.add_argument("--psscan")
    s.add_argument("--ldrmodules")
    s.set_defaults(func=cmd_vol_metrics)

    s = sub.add_parser("scan", help="run the available channels and print a verdict")
    s.add_argument("--dump", help="raw memory image")
    s.add_argument("--baseline", help="baseline .npy from the 'baseline' command")
    s.add_argument("--cnn", help="trained CNN checkpoint")
    s.add_argument("--features", help="feature table for the structural channel")
    s.add_argument("--structural", help="trained structural model (optional with --dump)")
    s.add_argument("--no-volatility", action="store_true",
                   help="skip Volatility; only the byte channel looks at --dump")
    s.add_argument("--vol-dir", help="Volatility JSON cache when features come from --dump")
    s.add_argument("--family", help="family model from train-family, to name the likely family")
    s.add_argument("--row", type=int, default=0, help="row of --features describing --dump")
    s.add_argument("--page-ratio", type=float, default=0.20)
    s.add_argument("--structural-threshold", type=float, default=0.90)
    s.add_argument("--structural-decides", action="store_true",
                   help="let a structural score >= the threshold make the dump MALICIOUS on its own "
                        "(the report's rule; use it with a model trained on dumps of this VM)")
    s.add_argument("--top", type=int, default=20, help="suspicious pages listed in the report")
    s.add_argument("--report", help="write a JSON report here")
    s.set_defaults(func=cmd_scan)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except KeyboardInterrupt:
        print("stopped", file=sys.stderr)
        return EXIT_ERROR
    except Exception as exc:  # noqa: BLE001 - report any failure with its own exit code
        import traceback

        traceback.print_exc()
        print(f"\nerror: {type(exc).__name__}: {exc}", file=sys.stderr)
        return EXIT_ERROR


if __name__ == "__main__":
    sys.exit(main())

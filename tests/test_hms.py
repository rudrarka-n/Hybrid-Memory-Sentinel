"""Unit tests. Run with ``python -m pytest`` from the project folder.

All inputs are generated here: random bytes, repeated code-like bytes and a
made-up feature table. No malware and no real memory dump is involved, so
these tests check that the pipeline behaves correctly, not how well it detects.
"""

import json

import numpy as np
import pandas as pd
import pytest

from hms import PAGE_SIZE
from hms.cli import main
from hms.fusion import fuse
from hms.preprocess import (PageScanner, build_baseline, collect_candidates,
                            page_entropy)
from hms.structural import (cross_plugin_agreement, load_vol_json, sample_groups,
                            train_structural, unlinked_dll_ratio)

RNG = np.random.default_rng(1)


def random_page():
    return RNG.integers(0, 256, PAGE_SIZE, dtype=np.uint8).tobytes()


def code_page(tag: int):
    """Low-entropy page holding a function prologue, unique per tag."""
    body = b"\x55\x8b\xec\x83\xec\x10" + bytes([tag % 256, tag // 256]) + b"\xc3\x90"
    return (body * (PAGE_SIZE // len(body) + 1))[:PAGE_SIZE]


def text_page():
    return (b"hello world, nothing to see here. " * 200)[:PAGE_SIZE]


def write(path, pages):
    path.write_bytes(b"".join(pages))
    return path


# ---------------------------------------------------------------- preprocess

def test_entropy_bounds():
    pages = np.stack([
        np.zeros(PAGE_SIZE, np.uint8),
        np.tile(np.arange(256, dtype=np.uint8), PAGE_SIZE // 256),
    ])
    ent = page_entropy(pages)
    assert ent[0] == 0.0
    assert ent[1] == pytest.approx(8.0)


def test_filter_counts_every_page_once(tmp_path):
    zero = bytes(PAGE_SIZE)
    rnd, code = random_page(), code_page(1)
    dump = write(tmp_path / "d.raw", [zero, zero, rnd, rnd, code, text_page()])
    batch, st = collect_candidates(dump)
    assert st.total_pages == 6
    assert st.zero_pages == 2
    assert st.self_duplicates == 1
    assert st.not_code_like == 1          # the text page
    assert st.candidates == 2
    assert st.high_entropy_candidates == 1 and st.code_like_candidates == 1
    assert list(batch.offsets) == [2 * PAGE_SIZE, 4 * PAGE_SIZE]
    assert st.reduction == pytest.approx(4 / 6)


def test_baseline_removes_known_clean_pages(tmp_path):
    clean = [random_page() for _ in range(8)]
    injected = random_page()
    base = build_baseline(write(tmp_path / "clean.raw", clean))
    dump = write(tmp_path / "infected.raw", clean[:4] + [injected] + clean[4:])
    batch, st = collect_candidates(dump, baseline=base)
    assert st.baseline_duplicates == 8
    assert st.candidates == 1
    assert batch.offsets[0] == 4 * PAGE_SIZE
    assert batch.pages[0].tobytes() == injected


def test_instruction_validation_rejects_fake_prologue():
    from hms.preprocess import (code_validation_available, count_prologues,
                                count_valid_prologues)

    if not code_validation_available():
        pytest.skip("capstone not installed")
    fake = b"\x55\x8b\xec" + b"\xff" * 61 + bytes(64)      # prologue, then junk
    assert count_prologues(fake) == 1 and count_valid_prologues(fake) == 0
    assert count_valid_prologues(code_page(7)) > 0


def test_scanner_handles_chunk_boundaries_and_partial_page(tmp_path):
    from hms.preprocess import FilterConfig

    pages = [random_page() for _ in range(7)]
    dump = tmp_path / "d.raw"
    dump.write_bytes(b"".join(pages) + b"\x01\x02\x03")   # trailing partial page
    scanner = PageScanner(dump, cfg=FilterConfig(chunk_pages=3))
    got = np.concatenate([b.offsets for b in scanner])
    assert scanner.stats.total_pages == 7
    assert list(got) == [i * PAGE_SIZE for i in range(7)]


# -------------------------------------------------------------------- fusion

def test_fusion_rules():
    assert fuse(structural_score=0.95).label == "SUSPICIOUS"      # alone it does not decide
    assert fuse(structural_score=0.89).label == "BENIGN"
    assert fuse(page_probs=np.array([0.9, 0.1, 0.1, 0.1, 0.1])).label == "MALICIOUS"   # 20%
    assert fuse(page_probs=np.array([0.9] + [0.1] * 5)).label == "BENIGN"              # 16.7%
    assert fuse(structural_score=0.2, page_probs=np.array([0.9, 0.9])).label == "MALICIOUS"
    assert fuse(structural_score=0.99, page_probs=np.array([0.1, 0.1])).label == "SUSPICIOUS"
    assert fuse(page_probs=np.zeros(0)).label == "INCONCLUSIVE"
    assert fuse().label == "INCONCLUSIVE"


def test_fusion_soft_vote():
    v = fuse(structural_score=0.8, page_probs=np.array([0.2, 0.4]))
    assert v.page_vote == pytest.approx(0.3)
    assert v.soft_vote == pytest.approx(0.55)
    assert v.label == "BENIGN"               # verdict still follows the two thresholds
    assert fuse(structural_score=0.4).soft_vote == pytest.approx(0.4)


# ---------------------------------------------------------------- structural

def test_cross_plugin_agreement_finds_hidden_process():
    pslist = [{"PID": 4}, {"PID": 100}]
    psscan = [{"PID": 4}, {"PID": 100}, {"PID": 666},
              {"PID": 50, "ExitTime": "2026-01-01 00:00:00"}]   # exited, not hidden
    res = cross_plugin_agreement(pslist, psscan)
    assert res.hidden_pids == [666]
    assert res.agreement == pytest.approx(2 / 3)


def test_unlinked_dll_ratio():
    rows = [
        {"InLoad": True, "InInit": False, "InMem": True},    # main exe, legitimate
        {"InLoad": True, "InInit": True, "InMem": True},
        {"InLoad": False, "InInit": False, "InMem": False},  # unlinked
        {"InLoad": "False", "InInit": "False", "InMem": "False"},
    ]
    res = unlinked_dll_ratio(rows)
    assert res.unlinked_ratio == 0.5 and len(res.unlinked) == 2


def test_vol_json_flattens_children(tmp_path):
    f = tmp_path / "pslist.json"
    f.write_text(json.dumps([{"PID": 4, "__children": [{"PID": 8, "__children": []}]}]))
    assert [r["PID"] for r in load_vol_json(f)] == [4, 8]


def fake_feature_table(path, n=400):
    """Made-up table in the CIC-MalMem-2022 layout, with a learnable signal."""
    rng = np.random.default_rng(0)
    y = np.arange(n) % 2
    df = pd.DataFrame({
        "Category": [f"Trojan-Zeus-{i // 10:04d}abc-{i % 5}.raw" if m else "Benign"
                     for i, m in enumerate(y)],
        "pslist.nproc": rng.normal(40, 5, n),
        "malfind.ninjections": rng.poisson(1 + 6 * y),
        "psxview.not_in_pslist": rng.poisson(0.2 + 3 * y),
        "Class": np.where(y == 1, "Malware", "Benign"),
    })
    df.to_csv(path, index=False)
    return df


def test_structural_metrics_from_table():
    from hms.structural import AGR_COL, UDR_COL, add_structural_metrics

    X = pd.DataFrame({"pslist.nproc": [40, 10], "psxview.not_in_pslist": [0, 2],
                      "psxview.not_in_eprocess_pool": [0, 1],
                      "ldrmodules.not_in_load_avg": [0.1, 0.3],
                      "ldrmodules.not_in_mem_avg": [0.1, 0.2]})
    out = add_structural_metrics(X)
    assert list(out[AGR_COL]) == pytest.approx([1.0, 1 - 3 / 12])
    assert list(out[UDR_COL]) == pytest.approx([0.1, 0.2])


def test_lightgbm_structural_model(tmp_path):
    pytest.importorskip("lightgbm")
    csv = tmp_path / "t.csv"
    fake_feature_table(csv)
    res = train_structural(csv, model="lightgbm")
    assert res["model"] == "lightgbm" and set(res["models"]) == {"lightgbm"}
    assert res["models"]["lightgbm"]["accuracy"] > 0.9


def test_group_split_keeps_samples_together(tmp_path):
    df = fake_feature_table(tmp_path / "t.csv")
    y = (df["Class"] == "Malware").astype(int)
    groups = sample_groups(df["Category"], y)
    mal = df[y == 1].assign(g=groups[y == 1])
    # every dump of one sample shares a group; benign rows are all separate
    assert mal.groupby(mal["Category"].str.rsplit("-", n=1).str[0])["g"].nunique().max() == 1
    assert len(set(groups[y == 0])) == int((y == 0).sum())


def test_train_structural_and_scan(tmp_path, capsys):
    csv, model = tmp_path / "t.csv", tmp_path / "s.joblib"
    fake_feature_table(csv)
    res = train_structural(csv, model, baselines=True)
    assert set(res["models"]) == {"xgboost", "lightgbm", "random_forest", "decision_tree",
                                  "knn", "logistic_regression", "svm"}
    assert res["models"]["xgboost"]["accuracy"] > 0.9
    assert model.exists() and model.with_suffix(".metrics.json").exists()

    report = tmp_path / "r.json"
    assert main(["scan", "--features", str(csv), "--structural", str(model),
                 "--report", str(report)]) == 1           # some rows are flagged MALICIOUS
    assert json.loads(report.read_text())["rows"] == 400
    assert "flagged MALICIOUS" in capsys.readouterr().out


# ----------------------------------------------------------------------- cnn

def test_cnn_train_save_load_scan(tmp_path):
    """Smoke test on a toy task: random pages vs. repeated code-like pages."""
    torch = pytest.importorskip("torch")
    from hms.cnn import PageCNN, load_cnn, train_cnn

    assert PageCNN()(torch.rand(3, PAGE_SIZE)).shape == (3,)

    ben, mal = tmp_path / "ben", tmp_path / "mal"
    ben.mkdir(), mal.mkdir()
    for i in range(4):
        write(ben / f"b{i}.bin", [code_page(i * 100 + j) for j in range(24)])
        write(mal / f"m{i}.bin", [random_page() for _ in range(24)])
    ckpt = tmp_path / "cnn.pt"
    res = train_cnn(ben, mal, ckpt, epochs=3, verbose=False)
    assert res["final"]["accuracy"] > 0.9 and res["warnings"] == []
    load_cnn(ckpt)

    report = tmp_path / "r.json"
    dump = write(tmp_path / "dump.raw", [random_page() for _ in range(10)] + [bytes(PAGE_SIZE)] * 5)
    assert main(["scan", "--dump", str(dump), "--cnn", str(ckpt), "--no-volatility",
                 "--report", str(report)]) == 1
    out = json.loads(report.read_text())
    assert out["verdict"]["label"] == "MALICIOUS"
    assert out["filter"]["zero_pages"] == 5
    assert len(out["suspicious_pages"]) == 10
    assert any(f["type"] == "CNN-flagged page" for f in out["findings"])
    assert len(out["flagged_pages"]) == out["verdict"]["malicious_pages"]
    assert 0.05 <= torch.load(ckpt, weights_only=False)["state_dict"]["temperature"] <= 20


# ------------------------------------------------------------------- extract

def fake_vol_output(folder):
    """Volatility 3 JSON in the shape each plugin writes, for a tiny made-up system."""
    folder.mkdir()
    plugins = {
        "pslist": [
            {"PID": 4, "PPID": 0, "ImageFileName": "System", "Threads": 100, "Handles": 900,
             "SessionId": None, "Wow64": False, "ExitTime": None,
             "__children": [{"PID": 400, "PPID": 4, "ImageFileName": "explorer.exe",
                             "Threads": 20, "Handles": 300, "SessionId": 1, "Wow64": False,
                             "ExitTime": None, "__children": []}]},
        ],
        "dlllist": [{"PID": 400, "Name": "a.dll"}, {"PID": 400, "Name": "b.dll"},
                    {"PID": 4, "Name": "c.dll"}],
        "handles": [{"PID": 4, "Type": "File"}, {"PID": 4, "Type": "Mutant"},
                    {"PID": 400, "Type": "Key"}, {"PID": 400, "Type": "Key"}],
        "ldrmodules": [
            {"Pid": 400, "Process": "explorer.exe", "Base": 1, "InLoad": True, "InInit": False,
             "InMem": True, "MappedPath": "\\explorer.exe"},
            {"Pid": 400, "Process": "explorer.exe", "Base": 2, "InLoad": False, "InInit": False,
             "InMem": False, "MappedPath": "\\evil.dll"},
        ],
        "malfind": [
            {"PID": 400, "Process": "explorer.exe", "Start VPN": 4096,
             "Protection": "PAGE_EXECUTE_READWRITE", "CommitCharge": 3},
            {"PID": 400, "Process": "explorer.exe", "Start VPN": 8192,
             "Protection": "PAGE_EXECUTE_READWRITE", "CommitCharge": 1,
             "Hexdump": "4d 5a 90 00 03 00 00 00"},
        ],
        "psxview": [
            {"PID": 4, "Name": "System", "pslist": True, "psscan": True, "thrdscan": True,
             "csrss": False, "Exit Time": None},
            {"PID": 400, "Name": "explorer.exe", "pslist": True, "psscan": True,
             "thrdscan": True, "csrss": True, "Exit Time": None},
            {"PID": 668, "Name": "rootkit.exe", "pslist": False, "psscan": True,
             "thrdscan": True, "csrss": True, "Exit Time": None},
        ],
        "modules": [{"Name": "ntoskrnl.exe"}, {"Name": "hal.dll"}],
        "svcscan": [
            {"State": "SERVICE_RUNNING", "Type": "SERVICE_KERNEL_DRIVER"},
            {"State": "SERVICE_STOPPED", "Type": "SERVICE_FILE_SYSTEM_DRIVER"},
            {"State": "SERVICE_RUNNING", "Type": "SERVICE_WIN32_SHARE_PROCESS"},
            {"State": "SERVICE_RUNNING", "Type": "SERVICE_WIN32_OWN_PROCESS"},
        ],
        "callbacks": [{"Type": "GenericKernelCallback", "Module": "ntoskrnl"},
                      {"Type": "PspCreateProcessNotifyRoutine", "Module": "UNKNOWN"}],
        "psscan": [{"PID": 4, "ExitTime": None}, {"PID": 400, "ExitTime": None},
                   {"PID": 668, "ExitTime": None}],
        # detectors
        "drivermodule": [{"Offset": 1, "Known Exception": False, "Driver Name": "\\Driver\\evil",
                          "Service Key": "evil", "Alternative Name": None}],
        "svcdiff": [], "hollowprocesses": [], "processghosting": [], "pebmasquerade": [],
        "suspicious_threads": [], "thrdscan": [],
    }
    for name, rows in plugins.items():
        (folder / f"{name}.json").write_text(json.dumps(rows))
    (folder / "orphan_kernel_threads.failed").write_text("plugin not available")
    return folder


def test_features_from_volatility_json(tmp_path):
    from hms.extract import FEATURE_COLUMNS, features_from_json

    ex = features_from_json(fake_vol_output(tmp_path / "vol"))
    f = ex.features
    assert set(f) == set(FEATURE_COLUMNS) and len(FEATURE_COLUMNS) == 55
    assert f["pslist.nproc"] == 2 and f["pslist.avg_threads"] == 60
    assert f["dlllist.avg_dlls_per_proc"] == 1.5
    assert f["handles.nkey"] == 2 and f["handles.nport"] == 0
    assert f["ldrmodules.not_in_init"] == 2 and f["ldrmodules.not_in_load_avg"] == 0.5
    assert f["malfind.protection"] == 12 and f["malfind.uniqueInjections"] == 2
    assert f["psxview.not_in_pslist"] == 1 and f["psxview.not_in_csrss_handles"] == 1
    assert f["svcscan.nactive"] == 3 and f["svcscan.shared_process_services"] == 1
    assert f["callbacks.nanonymous"] == 1 and f["callbacks.ngeneric"] == 1
    assert [p["pid"] for p in ex.hidden_processes] == [668]
    assert [d["path"] for d in ex.unlinked_dlls] == ["\\evil.dll"]
    assert list(ex.frame().columns) == list(FEATURE_COLUMNS)
    assert ex.agr == pytest.approx(2 / 3) and ex.udr == 0.5


def test_scan_dump_extracts_features(tmp_path, capsys):
    """scan --dump --structural runs extraction (cached JSON here, no Volatility)."""
    from hms.extract import FEATURE_COLUMNS

    rng = np.random.default_rng(0)
    n = 200
    y = np.arange(n) % 2
    table = pd.DataFrame(rng.normal(0, 1, (n, 55)), columns=FEATURE_COLUMNS)
    table["malfind.ninjections"] = rng.poisson(1 + 6 * y)
    table["Category"] = [f"Trojan-X-{i:04d}-1.raw" if m else "Benign" for i, m in enumerate(y)]
    table["Class"] = np.where(y == 1, "Malware", "Benign")
    csv, model = tmp_path / "t.csv", tmp_path / "s.joblib"
    table.to_csv(csv, index=False)
    train_structural(csv, model)

    vol = fake_vol_output(tmp_path / "vol")
    dump = write(tmp_path / "vm.raw", [random_page() for _ in range(4)])
    report = tmp_path / "r.json"
    assert main(["scan", "--dump", str(dump), "--structural", str(model), "--vol-dir", str(vol),
                 "--report", str(report)]) == 1
    out = json.loads(report.read_text())
    assert out["verdict"]["structural_score"] is not None
    assert out["evidence"]["hidden_processes"][0]["name"] == "rootkit.exe"
    assert out["verdict"]["label"] == "MALICIOUS"
    assert out["findings"][0]["severity"] == "critical"
    assert "hidden process (DKOM)" in capsys.readouterr().out



def test_findings_rank_hiding_techniques(tmp_path):
    from hms.extract import features_from_json

    ex = features_from_json(fake_vol_output(tmp_path / "vol"))
    found = [(f["severity"], f["type"]) for f in ex.findings()]
    assert ("critical", "hidden process (DKOM)") in found
    assert ("critical", "injected PE image") in found
    assert ("critical", "hidden driver (rootkit)") in found
    assert ("high", "unlinked DLL") in found
    assert ("high", "anonymous kernel callback") in found
    assert ("medium", "injected code") in found
    order = {"critical": 0, "high": 1, "medium": 2}
    assert [order[s] for s, _ in found] == sorted(order[s] for s, _ in found)
    assert len(ex.alerts()) == 3
    assert ex.checks_skipped == ["orphan_kernel_threads"]


def test_kernel_invariant_overrides_classifier():
    v = fuse(structural_score=0.05, alerts=["PID 666 hidden"])
    assert v.label == "MALICIOUS" and "critical finding: PID 666 hidden" in v.reasons
    assert fuse(structural_score=0.05).label == "BENIGN"


def test_scan_exit_code_inconclusive(tmp_path):
    dump = write(tmp_path / "vm.raw", [random_page() for _ in range(4)])
    assert main(["scan", "--dump", str(dump), "--no-volatility"]) == 3


def test_scan_without_model_still_reports_hiding(tmp_path):
    vol = fake_vol_output(tmp_path / "vol")
    dump = write(tmp_path / "vm.raw", [random_page() for _ in range(4)])
    report = tmp_path / "r.json"
    assert main(["scan", "--dump", str(dump), "--vol-dir", str(vol), "--report", str(report)]) == 1
    out = json.loads(report.read_text())
    assert out["verdict"]["structural_score"] is None
    assert {f["type"] for f in out["findings"]} >= {"hidden process (DKOM)", "hidden driver (rootkit)",
                                                   "unclassified candidate pages"}


def test_compare_snapshots(tmp_path):
    from hms.extract import compare_snapshots, features_from_json

    first = features_from_json(fake_vol_output(tmp_path / "a"))
    second = features_from_json(fake_vol_output(tmp_path / "b"))
    second.injections = second.injections[:1]
    second.pids = [4, 400, 500]
    res = compare_snapshots(first, second)
    assert res["injections_persistent"] == 1 and res["injection_jaccard"] == 0.5
    assert res["processes_started"] == [500] and res["processes_gone"] == []
    assert len(res["injections_vanished"]) == 1


def test_train_family_and_feature_selection(tmp_path):
    from hms.structural import load_structural, predict_family, train_family

    csv = tmp_path / "t.csv"
    df = fake_feature_table(csv)
    out = tmp_path / "f.joblib"
    res = train_family(csv, out)
    assert set(res["classes"]) == {"Benign", "Trojan"} and res["accuracy"] > 0.9
    X = df.drop(columns=["Category", "Class"])
    assert predict_family(load_structural(out), X.head(2))[0][0] in {"Benign", "Trojan"}

    sel = train_structural(csv, select_features=2)
    assert sel["n_features"] == 2


def test_benign_lookalikes_are_not_critical(tmp_path):
    """Normal Windows artefacts that resemble hiding are listed but never decide the verdict."""
    from hms.extract import features_from_json

    vol = fake_vol_output(tmp_path / "vol")
    rows = lambda name: json.loads((vol / f"{name}.json").read_text())
    px = [r for r in rows("psxview") if r["PID"] != 668] + [
        {"PID": 88, "Name": "Registry", "pslist": False, "psscan": True, "thrdscan": False,
         "csrss": False, "Exit Time": None},
        {"PID": 1204, "Name": "updater.exe", "pslist": False, "psscan": True, "thrdscan": False,
         "csrss": False, "Exit Time": None},
        {"PID": 1213, "Name": "\x01\x02", "pslist": False, "psscan": True, "thrdscan": False,
         "csrss": False, "Exit Time": None}]
    psscan = [{"PID": 4, "ExitTime": None, "Threads": 100}, {"PID": 400, "ExitTime": None, "Threads": 20},
              {"PID": 88, "ExitTime": None, "Threads": 0}, {"PID": 1204, "ExitTime": None, "Threads": 0},
              {"PID": 1213, "ExitTime": None, "Threads": 0}]
    pslist = rows("pslist")
    pslist[0]["__children"][0]["Wow64"] = True
    ldr = [{"Pid": 400, "Process": "explorer.exe", "Base": 2, "InLoad": False, "InInit": False,
            "InMem": False, "MappedPath": "\\Windows\\SysWOW64\\kernel32.dll"},
           {"Pid": 400, "Process": "explorer.exe", "Base": 3, "InLoad": False, "InInit": False,
            "InMem": False, "MappedPath": "\\Windows\\System32\\en-US\\shell32.dll.mui"}]
    mal = [{"PID": 400, "Process": "powershell.exe", "Start VPN": 4096,
            "Protection": "PAGE_EXECUTE_READWRITE", "CommitCharge": 3},
           {"PID": 520, "Process": "svchost.exe", "Start VPN": 4096,
            "Protection": "PAGE_EXECUTE_READWRITE", "CommitCharge": 1, "Hexdump": "00 00 00 00"},
           {"PID": 900, "Process": "MsMpEng.exe", "Start VPN": 8192,
            "Protection": "PAGE_EXECUTE_READWRITE", "CommitCharge": 1, "Hexdump": "48 89 5c 24"}]
    for name, data in {"psxview": px, "psscan": psscan, "pslist": pslist, "ldrmodules": ldr,
                       "malfind": mal, "drivermodule": [], "callbacks": []}.items():
        (vol / f"{name}.json").write_text(json.dumps(data))

    ex = features_from_json(vol)
    findings = ex.findings()
    assert ex.alerts() == []
    assert all(f["severity"] != "critical" for f in findings)
    assert len(findings) == 8            # every item is still reported
    downgraded = [f for f in findings if f["severity"] == "low"]
    assert all(f.get("benign_explanation") for f in downgraded) and len(downgraded) == 7
    by_pid = {f["pid"]: f["severity"] for f in findings if "process" in f["type"]}
    assert by_pid == {88: "low", 1204: "medium", 1213: "low"}


def test_build_table_from_own_dumps(tmp_path):
    from hms.extract import FEATURE_COLUMNS
    from hms.preprocess import dump_files
    from hms.structural import train_family

    ben, mal = tmp_path / "benign", tmp_path / "malicious"
    (mal / "Ransomware").mkdir(parents=True)
    ben.mkdir()
    for folder, names in ((ben, ["clean1", "clean2"]), (mal / "Ransomware", ["wannacry"]), (mal, ["unknown"])):
        for n in names:
            write(folder / f"{n}.elf", [random_page()])
            fake_vol_output(folder / f"{n}_vol")         # cached plugin output, no Volatility run
    (ben / "notes.txt").write_text("not a dump")
    assert [p.name for p in dump_files(ben)] == ["clean1.elf", "clean2.elf"]

    out = tmp_path / "table.csv"
    assert main(["build-table", "--benign", str(ben), "--malicious", str(mal), "--out", str(out)]) == 0
    df = pd.read_csv(out)
    assert list(df.columns) == list(FEATURE_COLUMNS) + ["Category", "Class"]
    assert sorted(df["Class"]) == ["Benign", "Benign", "Malware", "Malware"]
    assert set(df["Category"]) == {"Benign", "Ransomware-wannacry-1", "Malware-unknown-1"}


def test_gui_module_imports_without_display():
    pytest.importorskip("tkinter")
    import hms.gui as gui

    assert gui.BUNDLED.endswith(".joblib") and gui.INPUT.name == "input"


# ------------------------------------------------------------- bug-hunt fixes

def test_structural_score_needs_corroboration():
    from hms.fusion import FusionConfig

    v = fuse(structural_score=0.99)
    assert v.label == "SUSPICIOUS" and "no hiding finding or CNN result agrees" in v.reasons[0]
    assert fuse(structural_score=0.99, cfg=FusionConfig(structural_alone=True)).label == "MALICIOUS"
    v = fuse(structural_score=0.99, support=["unlinked DLL"])
    assert v.label == "MALICIOUS" and "corroborated" in v.reasons[0]
    assert fuse(structural_score=0.99, alerts=["hidden"]).label == "MALICIOUS"
    assert fuse(structural_score=0.5, support=["unlinked DLL"]).label == "BENIGN"
    # hiding checks only: high findings are mentioned, not "no evidence"
    v = fuse(support=["unlinked DLL"])
    assert v.label == "INCONCLUSIVE" and "manual review" in v.reasons[0]


def test_scan_exit_code_suspicious(tmp_path):
    csv, model = tmp_path / "t.csv", tmp_path / "s.joblib"
    df = fake_feature_table(csv)
    train_structural(csv, model)
    mal = df[df["Class"] == "Malware"].head(1)
    mal.to_csv(tmp_path / "one.csv", index=False)
    dump = write(tmp_path / "vm.raw", [random_page() for _ in range(2)])
    args = ["scan", "--features", str(tmp_path / "one.csv"), "--structural", str(model),
            "--dump", str(dump), "--no-volatility"]
    assert main(args) == 4                                   # SUSPICIOUS
    assert main(args + ["--structural-decides"]) == 1        # MALICIOUS
    # a feature table on its own evaluates the classifier: its score decides, exit 1 if flagged
    assert main(["scan", "--features", str(tmp_path / "one.csv"), "--structural", str(model)]) == 1


def test_volatility_cache_follows_the_dump(tmp_path):
    from hms.extract import SOURCE_FILE, prepare_cache

    import os

    dump = write(tmp_path / "vm.raw", [random_page() for _ in range(3)])
    vol = tmp_path / "vm_vol"
    vol.mkdir()
    (vol / "pslist.json").write_text("[]")
    (vol / "svcdiff.failed").write_text("plugin missing")
    (vol / "case_notes.json").write_text("{}")                # not HMS's: never removed
    st = dump.stat()
    os.utime(dump, ns=(st.st_atime_ns, st.st_mtime_ns - 10**10))   # cache made after the dump

    prepare_cache(dump, vol, verbose=False)                  # no record yet: adopt it
    assert (vol / "pslist.json").exists() and (vol / SOURCE_FILE).exists()
    prepare_cache(dump, vol, verbose=False)                  # same dump: keep
    assert (vol / "pslist.json").exists() and (vol / "svcdiff.failed").exists()

    record = json.loads((vol / SOURCE_FILE).read_text())
    (vol / SOURCE_FILE).write_text(json.dumps({**record, "volatility": "0.0.1"}))
    prepare_cache(dump, vol, verbose=False)                  # Volatility upgraded: retry failures
    assert (vol / "pslist.json").exists() and not (vol / "svcdiff.failed").exists()

    write(dump, [random_page() for _ in range(3)])           # new capture, same name
    prepare_cache(dump, vol, verbose=False)
    assert not (vol / "pslist.json").exists()

    (vol / "pslist.json").write_text("[]")
    other = write(tmp_path / "vm.mem", [random_page() for _ in range(3)])   # same stem
    prepare_cache(other, vol, verbose=False)
    assert not (vol / "pslist.json").exists() and (vol / "case_notes.json").exists()

    # an unrecorded cache older than the dump is not trusted (unless named with --vol-dir)
    old = tmp_path / "old_vol"
    old.mkdir()
    (old / "pslist.json").write_text("[]")
    newer = write(tmp_path / "new.raw", [random_page()])
    st = newer.stat()
    os.utime(newer, ns=(st.st_atime_ns, st.st_mtime_ns + 10**10))
    prepare_cache(newer, old, verbose=False, trust_unrecorded=True)
    assert (old / "pslist.json").exists()
    (old / "source.json").unlink()
    prepare_cache(newer, old, verbose=False)
    assert not (old / "pslist.json").exists()


def test_name_match_never_hides_hiding(tmp_path):
    from hms.allowlist import hidden_process_severity
    from hms.extract import features_from_json

    assert hidden_process_severity(4936, "Registry", running=True)[0] == "critical"
    assert hidden_process_severity(4936, "Registry", running=False)[0] == "low"
    vol = fake_vol_output(tmp_path / "vol")
    (vol / "malfind.json").write_text(json.dumps([
        {"PID": 400, "Process": "powershell.exe", "Start VPN": 4096,
         "Protection": "PAGE_EXECUTE_READWRITE", "CommitCharge": 3, "Hexdump": "4d 5a 90 00"}]))
    pe = [f for f in features_from_json(vol).findings() if f["type"] == "injected PE image"]
    assert pe and pe[0]["severity"] == "critical" and pe[0].get("benign_explanation")


def test_model_kind_is_checked(tmp_path, capsys):
    from hms.structural import load_structural, model_kind, train_family

    csv = tmp_path / "t.csv"
    fake_feature_table(csv)
    fam, struc = tmp_path / "f.joblib", tmp_path / "s.joblib"
    train_family(csv, fam)
    train_structural(csv, struc)
    assert model_kind(fam) == "family" and model_kind(struc) == "structural"
    with pytest.raises(ValueError, match="family model, not a structural"):
        load_structural(fam, "structural")
    # a crash or a wrong model exits 5, never 1 (MALICIOUS)
    assert main(["scan", "--features", str(csv), "--structural", str(fam)]) == 5
    assert "family model, not a structural model" in capsys.readouterr().err
    # --row outside the table is a usage error
    dump = write(tmp_path / "vm.raw", [random_page()])
    assert main(["scan", "--features", str(csv), "--structural", str(struc), "--dump", str(dump),
                 "--no-volatility", "--row", "9999"]) == 2


def test_family_with_feature_table(tmp_path):
    from hms.structural import train_family

    csv = tmp_path / "t.csv"
    fake_feature_table(csv)
    fam, struc, report = tmp_path / "f.joblib", tmp_path / "s.joblib", tmp_path / "out" / "r.json"
    train_family(csv, fam)
    train_structural(csv, struc)
    assert main(["scan", "--features", str(csv), "--structural", str(struc), "--family", str(fam),
                 "--report", str(report)]) == 1              # report folder is created
    out = json.loads(report.read_text())
    assert len(out["families"]) == 400 and out["families"][0]["label"] in {"Benign", "Trojan"}


def own_dumps_table(path, n=60):
    """Feature table like build-table writes: the two deskthrd columns are empty."""
    from hms.extract import FEATURE_COLUMNS

    rng = np.random.default_rng(3)
    y = np.arange(n) % 2
    df = pd.DataFrame(rng.normal(0, 1, (n, len(FEATURE_COLUMNS))), columns=list(FEATURE_COLUMNS))
    df["malfind.ninjections"] = rng.poisson(1 + 6 * y)
    df["psxview.not_in_deskthrd"] = np.nan
    df["psxview.not_in_deskthrd_false_avg"] = np.nan
    df["Category"] = [f"Trojan-x{i}-1" if m else "Benign" for i, m in enumerate(y)]
    df["Class"] = np.where(y == 1, "Malware", "Benign")
    df.to_csv(path, index=False)
    return df


def test_training_on_own_dumps_table_with_empty_columns(tmp_path):
    csv = tmp_path / "my_dumps.csv"
    own_dumps_table(csv)
    assert train_structural(csv, select_features=5)["n_features"] == 5
    res = train_structural(csv, baselines=True)
    assert {"knn", "logistic_regression", "svm"} <= set(res["models"])


def test_train_family_survives_a_family_missing_from_training(tmp_path):
    from hms.structural import train_family

    rows = own_dumps_table(tmp_path / "base.csv", n=6)
    rows.loc[rows["Class"] == "Malware", "Category"] = ["Ransomware-a-1", "Trojan-b-1", "Malware-c-1"]
    csv = tmp_path / "small.csv"
    rows.to_csv(csv, index=False)
    missing = []
    for seed in range(10):
        res = train_family(csv, split="group", seed=seed)
        missing += res["families_missing_from_training"]
    assert missing                                          # the case happened and did not crash


def test_prologue_counting():
    from hms.preprocess import code_validation_available, count_prologues, count_valid_prologues

    hot = b"\x8b\xff\x55\x8b\xec" + bytes(20)
    assert count_prologues(hot) == 1                        # not counted twice
    if not code_validation_available():
        pytest.skip("capstone not installed")
    body = (b"\x48\x83\xec\x28\x48\x89\x5c\x24\x30\x48\x8b\xd9\x48\x8b\x01\x48\x85\xc0"
            b"\x74\x0b\x48\x8b\xcb\xff\x50\x10\x48\x63\xd0\x48\x8b\x5c\x24\x30\x48\x83\xc4\x28\xc3\x90")
    x64 = (body * 3)[:96]
    assert count_valid_prologues(x64) > 0
    junk = (b"\x55\x8b\xec" + b"\xff" * 13) * 20            # 20 fake 32-bit prologues in data
    assert count_valid_prologues(junk + x64) > 0             # the x64 code is still checked


def test_candidate_sampling_is_bounded(tmp_path):
    from hms.preprocess import FilterConfig

    dump = write(tmp_path / "d.raw", [random_page() for _ in range(40)])
    full, _ = collect_candidates(dump, cfg=FilterConfig(chunk_pages=4))
    sample, st = collect_candidates(dump, cfg=FilterConfig(chunk_pages=4), max_pages=10,
                                    rng=np.random.default_rng(5))
    assert len(sample) == 10 and st.candidates == len(full) == 40
    assert list(sample.offsets) == sorted(sample.offsets)
    assert set(sample.offsets) <= set(full.offsets)


def test_baseline_name_gets_npy(tmp_path, capsys):
    dump = write(tmp_path / "clean.raw", [random_page() for _ in range(2)])
    out = tmp_path / "models" / "base"
    assert main(["baseline", str(dump), "-o", str(out)]) == 0
    assert (tmp_path / "models" / "base.npy").exists() and "base.npy" in capsys.readouterr().out


def test_gui_sorts_models_by_their_metrics(tmp_path, monkeypatch):
    pytest.importorskip("tkinter")
    import hms.gui as gui

    for name, metrics in (("myfamilymodel", {"models": {}}), ("tree", {"classes": ["Benign", "Trojan"]}),
                          ("family_old", None)):
        (tmp_path / f"{name}.joblib").write_bytes(b"")
        if metrics is not None:
            (tmp_path / f"{name}.metrics.json").write_text(json.dumps(metrics))
    monkeypatch.setattr(gui, "MODELS", tmp_path)
    m = gui._model_files()
    assert m["structural"] == ["myfamilymodel.joblib"]
    assert m["family"] == ["family_old.joblib", "tree.joblib"]


# ------------------------------------------------------ second review fixes

def clean_vol_output(folder):
    """fake_vol_output with every trace of hiding removed."""
    vol = fake_vol_output(folder)
    data = {name: json.loads((vol / f"{name}.json").read_text()) for name in
            ("ldrmodules", "psxview", "psscan", "callbacks")}
    data["ldrmodules"] = data["ldrmodules"][:1]
    data["psxview"] = [r for r in data["psxview"] if r["PID"] != 668]
    data["psscan"] = [r for r in data["psscan"] if r["PID"] != 668]
    data["callbacks"] = data["callbacks"][:1]
    data.update(malfind=[], drivermodule=[])
    for name, rows in data.items():
        (vol / f"{name}.json").write_text(json.dumps(rows))
    return vol


def test_clean_hiding_checks_give_benign(tmp_path):
    vol = clean_vol_output(tmp_path / "vol")
    dump = write(tmp_path / "vm.raw", [random_page() for _ in range(2)])
    report = tmp_path / "r.json"
    assert main(["scan", "--dump", str(dump), "--vol-dir", str(vol), "--report", str(report)]) == 0
    verdict = json.loads(report.read_text())["verdict"]
    assert verdict["label"] == "BENIGN" and "hiding checks ran" in verdict["reasons"][0]
    assert fuse(checks_ran=True).label == "BENIGN" and fuse().label == "INCONCLUSIVE"


def test_reports_are_strict_json(tmp_path):
    vol = fake_vol_output(tmp_path / "vol")                 # deskthrd features are NaN
    dump = write(tmp_path / "vm.raw", [random_page() for _ in range(2)])
    report = tmp_path / "r.json"
    main(["scan", "--dump", str(dump), "--vol-dir", str(vol), "--report", str(report)])

    def reject(value):
        raise ValueError(f"not valid JSON: {value}")

    out = json.loads(report.read_text(), parse_constant=reject)
    assert out["structural_features"]["psxview.not_in_deskthrd"] is None


def test_inputs_are_checked_before_volatility(tmp_path, capsys):
    missing = tmp_path / "no such.raw"
    assert main(["scan", "--dump", str(missing)]) == 2
    assert not (tmp_path / "no such_vol").exists()          # no stray cache folder
    dump = write(tmp_path / "vm.raw", [random_page()])
    assert main(["scan", "--dump", str(dump), "--structural", str(tmp_path / "x.joblib")]) == 2
    assert "is not a file" in capsys.readouterr().err


def test_broken_family_model_does_not_stop_a_scan(tmp_path):
    csv, struc = tmp_path / "t.csv", tmp_path / "s.joblib"
    fake_feature_table(csv)
    train_structural(csv, struc)
    broken = tmp_path / "family_broken.joblib"
    broken.write_bytes(b"not a model")
    report = tmp_path / "r.json"
    assert main(["scan", "--features", str(csv), "--structural", str(struc), "--family", str(broken),
                 "--report", str(report)]) == 1
    assert "family_error" in json.loads(report.read_text())


def test_only_known_feature_columns_are_used(tmp_path, capsys):
    from hms.structural import load_feature_table, load_structural

    df = fake_feature_table(tmp_path / "t.csv")
    df["pslist.nproc"] = df["pslist.nproc"].astype(object)
    df.loc[0, "pslist.nproc"] = "?"
    df.to_csv(tmp_path / "with_index.csv")                   # pandas adds an "Unnamed: 0" column
    X, _, _ = load_feature_table(tmp_path / "with_index.csv")
    assert "Unnamed: 0" not in X.columns and "pslist.nproc" in X.columns
    assert np.isnan(X.loc[0, "pslist.nproc"])
    err = capsys.readouterr().err
    assert "ignoring 1 column" in err and "1 cell(s) were not numbers" in err
    model = tmp_path / "m.joblib"
    train_structural(tmp_path / "with_index.csv", model)
    assert "Unnamed: 0" not in load_structural(model)["features"]


def test_single_class_table_is_refused(tmp_path, capsys):
    df = fake_feature_table(tmp_path / "t.csv")
    df[df["Class"] == "Benign"].to_csv(tmp_path / "benign.csv", index=False)
    with pytest.raises(ValueError, match="only one class"):
        train_structural(tmp_path / "benign.csv")
    assert main(["train-structural", "--csv", str(tmp_path / "benign.csv"),
                 "--out", str(tmp_path / "b.joblib")]) == 5


def test_models_are_saved_portably(tmp_path):
    import joblib
    from hms.structural import load_structural

    csv, model = tmp_path / "t.csv", tmp_path / "s.joblib"
    fake_feature_table(csv)
    res = train_structural(csv, model)
    raw = joblib.load(model)
    assert raw["model"] is None and raw["xgboost_json"] and raw["kind"] == "structural"
    assert load_structural(model, "structural")["model"].predict_proba(
        pd.read_csv(csv).drop(columns=["Category", "Class"]).assign(
            **{"structural.agr": 1.0, "structural.udr": 0.0})[raw["features"]]).shape[1] == 2
    assert res["at_scan_threshold"]["n"] == res["models"]["xgboost"]["n"]
    for name in ("cic_structural", "cic_family"):            # the bundled models
        assert joblib.load(f"models/{name}.joblib")["xgboost_json"]


def test_hexdump_layouts_and_truncated_cache(tmp_path):
    from hms.extract import _complete_json, hexdump_bytes

    assert hexdump_bytes("4d 5a 90 00") == b"MZ\x90\x00"
    text = "\n0x1000  4d 5a 90 00 03 00  MZ....\n0x1006  00 00 ff  ..."
    assert hexdump_bytes(text) == b"MZ\x90\x00\x03\x00\x00\x00\xff"
    whole, cut = tmp_path / "a.json", tmp_path / "b.json"
    whole.write_text(json.dumps([{"PID": i} for i in range(50)]))
    cut.write_text(whole.read_text()[:100])
    assert _complete_json(whole) and not _complete_json(cut)


# --------------------------------------------------------- round 3 fixes

def test_setup_dependent_findings_do_not_corroborate(tmp_path):
    from hms.extract import features_from_json

    vol = clean_vol_output(tmp_path / "vol")
    ldr = json.loads((vol / "ldrmodules.json").read_text()) + [
        {"Pid": 400, "Process": "explorer.exe", "Base": 9, "InLoad": False, "InInit": False, "InMem": False,
         "MappedPath": "\\Users\\bob\\AppData\\Local\\Grammarly\\addin.dll"}]
    (vol / "ldrmodules.json").write_text(json.dumps(ldr))
    ex = features_from_json(vol)
    assert len(ex.high()) == 1 and ex.high(corroborating_only=True) == []   # the AppData DLL
    v = fuse(0.99, support=ex.high(True), review=ex.high(), checks_ran=True)
    assert v.label == "SUSPICIOUS"
    assert fuse(0.99, support=["hidden service evil"], checks_ran=True).label == "MALICIOUS"
    # only the listed types corroborate: name spoofing and partly hidden processes don't
    from hms.extract import CORROBORATING
    assert CORROBORATING == {"hidden service", "process hollowing", "process ghosting",
                             "orphan kernel thread", "anonymous kernel callback"}
    ex.detections.append({"severity": "high", "type": "process name spoofing", "pid": 7,
                          "process": "a.exe", "detail": "spoofed"})
    assert ex.high(corroborating_only=True) == []
    ex.detections.append({"severity": "high", "type": "hidden service", "pid": None,
                          "process": None, "detail": "service evil"})
    assert ex.high(corroborating_only=True) == ["service evil"]
    # hiding checks only: an unlinked AppData DLL alone is not enough to doubt a clean dump
    ex = features_from_json(vol)
    v = fuse(support=ex.high(True), review=ex.high(), checks_ran=True)
    assert v.label == "BENIGN" and "manual review" in v.reasons[0]


def test_review_note_kept_when_a_model_ran():
    v = fuse(None, np.array([0.1, 0.2, 0.1]), support=["hidden service X"], checks_ran=True)
    assert v.label == "BENIGN" and "1 high-severity finding(s) need a manual review" in v.reasons[-1]
    assert "manual review" not in fuse(0.2, checks_ran=True).reasons[-1]


def test_comparison_models_on_a_tiny_table(tmp_path, capsys):
    csv = tmp_path / "tiny.csv"
    own_dumps_table(csv, n=6)
    res = train_structural(csv, split="random", baselines=True)
    assert "xgboost" in res["models"] and "accuracy" in res["models"]["knn"]
    assert main(["train-structural", "--csv", str(csv), "--split", "random", "--baselines",
                 "--out", str(tmp_path / "m.joblib")]) == 0


def test_cache_follows_rewrites_and_stems(tmp_path):
    import os
    from hms.extract import SOURCE_FILE, default_vol_dir, prepare_cache

    dump = write(tmp_path / "vm.raw", [bytes(PAGE_SIZE)] * 3)       # all-zero: samples look alike
    vol = default_vol_dir(dump)
    prepare_cache(dump, vol, verbose=False)
    (vol / "pslist.json").write_text("[]")
    st = dump.stat()
    os.utime(dump, ns=(st.st_atime_ns, st.st_mtime_ns + 10**9))      # a new capture, same bytes
    prepare_cache(dump, vol, verbose=False)
    assert not (vol / "pslist.json").exists()

    other = write(tmp_path / "vm.mem", [random_page()])
    assert default_vol_dir(other) == tmp_path / "vm.mem_vol"        # vm.raw keeps vm_vol
    assert default_vol_dir(dump) == vol
    assert json.loads((vol / SOURCE_FILE).read_text())["name"] == "vm.raw"


def test_text_hexdump_stops_at_16_bytes():
    from hms.extract import hexdump_bytes

    line = "0x1000  " + " ".join(["4d"] * 16) + " ab cd......"
    assert len(hexdump_bytes(line)) == 16


# ------------------------------------------------------ recheck fixes

def test_running_unlinked_process_is_critical_whatever_its_pid_or_name():
    from hms.allowlist import hidden_process_severity

    for pid, name in ((670, "rootkit.exe"), (668, ""), (668, "rootkit\x01.exe"), (4936, "Registry")):
        assert hidden_process_severity(pid, name, running=True)[0] == "critical"
    assert hidden_process_severity(670, "rootkit.exe", running=False)[0] == "low"
    # odd PID or name with only the structure's own thread count: a corrupt leftover
    assert hidden_process_severity(1213, "\x01\x02", running=True, confirmed=False)[0] == "low"
    assert hidden_process_severity(1213, "\x01\x02", running=True, confirmed=True)[0] == "critical"


def test_missing_critical_checks_are_not_called_clean(tmp_path):
    v = fuse(checks_ran=True, skipped=["drivermodule", "svcdiff"])
    assert v.label == "INCONCLUSIVE" and "drivermodule" in v.reasons[0]
    v = fuse(checks_ran=True, skipped=["suspicious_threads"])
    assert v.label == "BENIGN" and "could not run: suspicious_threads" in v.reasons[0]
    vol = clean_vol_output(tmp_path / "vol")
    (vol / "drivermodule.json").unlink()
    (vol / "drivermodule.failed").write_text("invalid choice")
    dump = write(tmp_path / "vm.raw", [random_page()])
    assert main(["scan", "--dump", str(dump), "--vol-dir", str(vol)]) == 3


def test_features_with_dump_still_runs_hiding_checks(tmp_path):
    csv, model = tmp_path / "t.csv", tmp_path / "s.joblib"
    df = fake_feature_table(csv)
    train_structural(csv, model)
    df[df["Class"] == "Benign"].head(1).to_csv(tmp_path / "one.csv", index=False)
    vol = fake_vol_output(tmp_path / "vol")               # has critical findings
    dump = write(tmp_path / "vm.raw", [random_page()])
    report = tmp_path / "r.json"
    assert main(["scan", "--dump", str(dump), "--vol-dir", str(vol), "--features", str(tmp_path / "one.csv"),
                 "--structural", str(model), "--report", str(report)]) == 1
    out = json.loads(report.read_text())
    assert any(f["severity"] == "critical" for f in out["findings"])


def test_process_hidden_from_pslist_and_psscan_is_critical(tmp_path):
    from hms.extract import features_from_json

    vol = clean_vol_output(tmp_path / "vol")
    px = json.loads((vol / "psxview.json").read_text()) + [
        {"PID": 900, "Name": "ghost.exe", "pslist": False, "psscan": False, "thrdscan": True,
         "csrss": False, "Exit Time": None},
        {"PID": 904, "Name": "leftover.exe", "pslist": False, "psscan": False, "thrdscan": True,
         "csrss": True, "Exit Time": None}]
    (vol / "psxview.json").write_text(json.dumps(px))
    (vol / "thrdscan.json").write_text(json.dumps([
        {"PID": 900, "TID": 1, "CreateTime": "2026-10-17T10:00:00", "ExitTime": "2026-10-17T10:01:00"},
        {"PID": 900, "TID": 2, "CreateTime": "2026-10-17T10:00:00", "ExitTime": None},   # still running
        {"PID": 904, "TID": 3, "CreateTime": "2026-10-17T10:00:00", "ExitTime": "2026-10-17T10:02:00"}]))
    found = {f["pid"]: (f["severity"], f["type"]) for f in features_from_json(vol).findings()}
    assert found[900] == ("critical", "hidden process (pool scan evaded)")
    assert found[904] == ("high", "process hidden from some views")     # every thread exited
    (vol / "thrdscan.json").unlink()                                      # thread scan didn't run
    found = {f["pid"]: f["severity"] for f in features_from_json(vol).findings()}
    assert found[900] == "high"
    (vol / "thrdscan.json").write_text(json.dumps([{"PID": 900, "TID": 2, "ExitTime": None}]))
    dump = write(tmp_path / "vm.raw", [random_page()])
    assert main(["scan", "--dump", str(dump), "--vol-dir", str(vol)]) == 1


def test_corrupt_pool_scan_entry_is_not_critical(tmp_path):
    from hms.extract import features_from_json

    vol = clean_vol_output(tmp_path / "vol")
    px = json.loads((vol / "psxview.json").read_text()) + [
        {"PID": 1213, "Name": "\x01\x02", "pslist": False, "psscan": True, "thrdscan": False,
         "csrss": False, "Exit Time": None}]
    ps = json.loads((vol / "psscan.json").read_text()) + [{"PID": 1213, "ExitTime": None, "Threads": 3}]
    (vol / "psxview.json").write_text(json.dumps(px))
    (vol / "psscan.json").write_text(json.dumps(ps))
    found = {f["pid"]: f["severity"] for f in features_from_json(vol).findings()}
    assert found[1213] == "low"
    dump = write(tmp_path / "vm.raw", [random_page()])
    assert main(["scan", "--dump", str(dump), "--vol-dir", str(vol)]) == 0

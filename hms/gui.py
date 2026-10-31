r"""Desktop window for HMS.  Start it with ``python -m hms gui`` (or run_gui.bat).

The window runs the same commands as the command line (``python -m hms ...``)
in the background and shows their output, so anything done here can be
repeated in a terminal. It works with four folders in the project:

    input\        memory dumps to scan
    dataset\      feature tables (.csv) for the structural and family models
    CNNDataSet\   your own dumps for the CNN: benign\ and malicious\
                  (optionally one subfolder per family inside malicious\)
    models\       trained models; the bundled CIC-MalMem-2022 model is here
    reports\      one JSON report per scan
"""

from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

import tkinter as tk
from tkinter import messagebox, ttk

ROOT = Path(__file__).resolve().parent.parent
INPUT, DATASET, CNNDATA, MODELS, REPORTS = (
    ROOT / d for d in ("input", "dataset", "CNNDataSet", "models", "reports"))
BUNDLED = "cic_structural.joblib"
BUNDLED_FAMILY = "cic_family.joblib"


def _child_python() -> str:
    """python.exe for background commands, even when the window runs under pythonw.exe.

    A console program (Volatility) started from pythonw.exe opens a console
    window of its own; started from a python.exe with no window, it doesn't.
    """
    exe = Path(sys.executable)
    if exe.name.lower() == "pythonw.exe" and exe.with_name("python.exe").exists():
        return str(exe.with_name("python.exe"))
    return sys.executable
NONE = "(none)"
NO_MODEL = "(none: hiding checks only)"

SEVERITY_COLOURS = {"critical": "#c62828", "high": "#ef6c00", "medium": "#9e7c00", "low": "#607d8b"}
VERDICT_COLOURS = {"MALICIOUS": "#c62828", "SUSPICIOUS": "#ef6c00", "BENIGN": "#2e7d32",
                   "INCONCLUSIVE": "#616161"}
SCAN_OK_CODES = (0, 1, 3, 4)   # BENIGN, MALICIOUS, INCONCLUSIVE, SUSPICIOUS; 5 = HMS failed


def _open_folder(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    if sys.platform.startswith("win"):
        os.startfile(path)  # noqa: S606 - opens Explorer on a project folder
    elif sys.platform == "darwin":
        subprocess.Popen(["open", str(path)])
    else:
        subprocess.Popen(["xdg-open", str(path)])


def _model_files() -> dict[str, list[str]]:
    """Model files in models\\ by kind, the bundled structural model first.

    The kind of a .joblib comes from its .metrics.json; the file name is only
    a fallback for a model saved without one.
    """
    from .structural import model_kind

    files = sorted(p.name for p in MODELS.glob("*") if p.is_file())
    joblib = [f for f in files if f.endswith(".joblib")]
    family = [f for f in joblib if (model_kind(MODELS / f) or
                                    ("family" if "family" in f.lower() else "structural")) == "family"]
    structural = sorted((f for f in joblib if f not in family), key=lambda f: (f != BUNDLED, f))
    return {
        "structural": structural,
        "family": family,
        "cnn": [f for f in files if f.endswith(".pt")],
        "baseline": [f for f in files if f.endswith(".npy")],
    }


class App(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Hybrid Memory Sentinel")
        self.geometry("1100x760")
        self.minsize(860, 600)
        for d in (INPUT, DATASET, CNNDATA / "benign", CNNDATA / "malicious", MODELS, REPORTS):
            d.mkdir(parents=True, exist_ok=True)

        self.proc: subprocess.Popen | None = None
        self.lines: queue.Queue = queue.Queue()
        self.on_done = None
        self.stopped = False
        self.buttons: list[ttk.Button] = []

        tabs = ttk.Notebook(self)
        tabs.pack(fill="both", expand=True, padx=8, pady=(8, 0))
        scan, train = ttk.Frame(tabs, padding=8), ttk.Frame(tabs, padding=8)
        tabs.add(scan, text="  Scan a dump  ")
        tabs.add(train, text="  Train models  ")
        self._build_scan(scan)
        self._build_train(train)

        bottom = ttk.Frame(self, padding=(8, 4, 8, 8))
        bottom.pack(fill="both")
        bar = ttk.Frame(bottom)
        bar.pack(fill="x")
        ttk.Label(bar, text="Output").pack(side="left")
        self.status = ttk.Label(bar, text="Ready")
        self.status.pack(side="left", padx=12)
        self.stop_btn = ttk.Button(bar, text="Stop", command=self.stop, state="disabled")
        self.stop_btn.pack(side="right")
        self.log = tk.Text(bottom, height=12, wrap="word", font=("Consolas", 9))
        self.log.pack(fill="both", expand=True)
        self.refresh()
        self.protocol("WM_DELETE_WINDOW", self._close)
        self.after(100, self._drain)

    def _close(self) -> None:
        """Closing the window also stops a running scan or training, Volatility included."""
        self.stop()
        self.destroy()

    # ------------------------------------------------------------------ scan

    def _build_scan(self, f: ttk.Frame) -> None:
        f.columnconfigure(1, weight=1)
        f.rowconfigure(6, weight=1)

        ttk.Label(f, text="Memory dump (from the input folder)").grid(row=0, column=0, sticky="w")
        self.dump = ttk.Combobox(f, state="readonly")
        self.dump.grid(row=0, column=1, sticky="ew", padx=6)
        side = ttk.Frame(f)
        side.grid(row=0, column=2, sticky="e")
        ttk.Button(side, text="Refresh", command=self.refresh).pack(side="left")
        ttk.Button(side, text="Open input folder", command=lambda: _open_folder(INPUT)).pack(side="left", padx=4)

        self.structural = self._choice(f, 1, "Structural model")
        self.family = self._choice(f, 2, "Family model")
        self.cnn = self._choice(f, 3, "CNN model (byte channel)")
        self.use_cnn = tk.BooleanVar(value=False)
        self.use_cnn_box = ttk.Checkbutton(f, text="Include the CNN in this scan", variable=self.use_cnn)
        self.use_cnn_box.grid(row=3, column=2, sticky="w")
        self.baseline = self._choice(f, 4, "Clean baseline")
        self.decides = tk.BooleanVar(value=False)
        ttk.Checkbutton(f, text="Structural score decides alone (only for a model trained on this VM)",
                        variable=self.decides).grid(row=1, column=2, sticky="w")

        row = ttk.Frame(f)
        row.grid(row=5, column=0, columnspan=3, sticky="ew", pady=(8, 4))
        b = ttk.Button(row, text="Scan", command=self.scan)
        b.pack(side="left")
        self.buttons.append(b)
        ttk.Button(row, text="Open reports folder", command=lambda: _open_folder(REPORTS)).pack(side="left", padx=6)
        self.verdict = tk.Label(row, text="", font=("Segoe UI", 14, "bold"))
        self.verdict.pack(side="left", padx=16)
        self.summary = ttk.Label(row, text="")
        self.summary.pack(side="left")

        cols = ("severity", "type", "pid", "process", "detail")
        self.findings = ttk.Treeview(f, columns=cols, show="headings", height=10)
        for c, w in zip(cols, (80, 200, 60, 140, 560)):
            self.findings.heading(c, text=c.capitalize() if c != "pid" else "PID")
            self.findings.column(c, width=w, stretch=c == "detail", anchor="w")
        for sev, colour in SEVERITY_COLOURS.items():
            self.findings.tag_configure(sev, foreground=colour)
        self.findings.grid(row=6, column=0, columnspan=3, sticky="nsew")
        self.findings.bind("<Double-1>", self._show_finding)
        ttk.Label(f, text="Double-click a finding to read it in full. The full report is in the reports folder.",
                  foreground="#555555").grid(row=7, column=0, columnspan=3, sticky="w", pady=(4, 0))

    def _choice(self, f, row, label) -> ttk.Combobox:
        ttk.Label(f, text=label).grid(row=row, column=0, sticky="w", pady=2)
        box = ttk.Combobox(f, state="readonly")
        box.grid(row=row, column=1, sticky="ew", padx=6, pady=2)
        return box

    def scan(self) -> None:
        if not self.dump.get():
            messagebox.showinfo("HMS", "Copy a memory dump into the input folder, then press Refresh.")
            return
        dump = INPUT / self.dump.get()
        stamp = time.strftime("%Y%m%d-%H%M%S")
        report = REPORTS / f"{dump.stem}_{stamp}.json"
        args = ["scan", "--dump", str(dump), "--report", str(report)]
        boxes = [(self.structural, "--structural"), (self.family, "--family"), (self.baseline, "--baseline")]
        if self.use_cnn.get():
            boxes.append((self.cnn, "--cnn"))
        if self.decides.get():
            args.append("--structural-decides")
        for box, flag in boxes:
            if box.get() and not box.get().startswith("("):
                args += [flag, str(MODELS / box.get())]
        self.findings.delete(*self.findings.get_children())
        self.verdict.config(text="")
        self.summary.config(text="")
        self.run(args, f"Scanning {dump.name}", lambda code: self._show_report(report, code))

    def _show_report(self, report: Path, code: int) -> None:
        if code not in SCAN_OK_CODES or not report.exists():
            self.verdict.config(text="Scan failed (see Output)", fg=VERDICT_COLOURS["MALICIOUS"])
            return
        data = json.loads(report.read_text(encoding="utf-8"))
        label = data["verdict"]["label"]
        self.verdict.config(text=label, fg=VERDICT_COLOURS.get(label, "black"))
        found = data.get("findings", [])
        counts = {s: sum(x["severity"] == s for x in found) for s in SEVERITY_COLOURS}
        text = ", ".join(f"{n} {s}" for s, n in counts.items() if n) or "no findings"
        score = data["verdict"].get("structural_score")
        if score is not None:
            text += f"   |   structural confidence {score:.3f}"
        if data.get("family"):
            text += f"   |   likely family {data['family']['label']}"
            if data["family"].get("note"):
                text += " (disagrees with the verdict)"
        self.summary.config(text=text + f"   |   report: {report.name}")
        self._findings = found
        for i, x in enumerate(found):
            self.findings.insert("", "end", iid=str(i), tags=(x["severity"],), values=(
                x["severity"], x["type"], x.get("pid") if x.get("pid") is not None else "",
                x.get("process") or "", x["detail"]))

    def _show_finding(self, _event) -> None:
        sel = self.findings.selection()
        if sel:
            x = self._findings[int(sel[0])]
            why = x.get("benign_explanation")
            if not why:
                extra = ""
            elif x["severity"] == "critical":
                extra = f"\n\nPossible harmless cause, to check by hand: {why}"
            else:
                extra = f"\n\nWhy it is probably harmless: {why}"
            messagebox.showinfo(x["type"], f"[{x['severity'].upper()}]\n\n{x['detail']}{extra}")

    # ----------------------------------------------------------------- train

    def _build_train(self, f: ttk.Frame) -> None:
        f.columnconfigure(1, weight=1)
        intro = ("Structural and family models train on a feature table (.csv in the CIC-MalMem-2022 layout) "
                 "in the dataset folder. The CNN trains on your own memory dumps: copy clean dumps into "
                 "CNNDataSet\\benign and infected dumps into CNNDataSet\\malicious (one subfolder per family if "
                 "you know it). New models are saved in the models folder and appear in the Scan tab.")
        ttk.Label(f, text=intro, wraplength=1000, justify="left").grid(row=0, column=0, columnspan=3, sticky="w")
        row = ttk.Frame(f)
        row.grid(row=1, column=0, columnspan=3, sticky="w", pady=6)
        ttk.Button(row, text="Open dataset folder", command=lambda: _open_folder(DATASET)).pack(side="left")
        ttk.Button(row, text="Open CNNDataSet folder", command=lambda: _open_folder(CNNDATA)).pack(
            side="left", padx=6)

        g = ttk.LabelFrame(f, text="1. Feature table from your own dumps (optional)", padding=8)
        g.grid(row=2, column=0, columnspan=3, sticky="ew", pady=4)
        ttk.Label(g, text="Runs Volatility on every dump in CNNDataSet\\benign and CNNDataSet\\malicious "
                          "and writes dataset\\my_dumps.csv, so the structural model can learn your VM too. "
                          "Slow on the first run.", wraplength=850).pack(side="left")
        b = ttk.Button(g, text="Build table", command=self.build_table)
        b.pack(side="right")
        self.buttons.append(b)

        g = ttk.LabelFrame(f, text="2. Structural and family models (from a .csv)", padding=8)
        g.grid(row=3, column=0, columnspan=3, sticky="ew", pady=4)
        g.columnconfigure(1, weight=1)
        ttk.Label(g, text="Feature table").grid(row=0, column=0, sticky="w")
        self.csv = ttk.Combobox(g, state="readonly")
        self.csv.grid(row=0, column=1, columnspan=3, sticky="ew", padx=6)
        ttk.Label(g, text="Model").grid(row=1, column=0, sticky="w")
        self.algo = ttk.Combobox(g, state="readonly", values=("xgboost", "lightgbm"), width=12)
        self.algo.set("xgboost")
        self.algo.grid(row=1, column=1, sticky="w", padx=6, pady=2)
        ttk.Label(g, text="Split").grid(row=2, column=0, sticky="w")
        self.split = ttk.Combobox(g, state="readonly", values=("group", "random"), width=12)
        self.split.set("group")
        self.split.grid(row=2, column=1, sticky="w", padx=6, pady=2)
        ttk.Label(g, text="Keep best K features (0 = all)").grid(row=3, column=0, sticky="w")
        self.select = ttk.Spinbox(g, from_=0, to=57, width=6)
        self.select.set(0)
        self.select.grid(row=3, column=1, sticky="w", padx=6, pady=2)
        ttk.Label(g, text="Save as").grid(row=4, column=0, sticky="w")
        self.name = ttk.Entry(g)
        self.name.insert(0, "my_model")
        self.name.grid(row=4, column=1, sticky="ew", padx=6, pady=2)
        self.compare = tk.BooleanVar(value=False)
        ttk.Checkbutton(g, text="Also compare RF, DT, KNN, LR, SVM", variable=self.compare).grid(
            row=5, column=1, sticky="w", padx=6)
        btns = ttk.Frame(g)
        btns.grid(row=6, column=0, columnspan=4, sticky="e", pady=(6, 0))
        for text, cmd in (("Train structural model", self.train_structural),
                          ("Train family model", self.train_family)):
            b = ttk.Button(btns, text=text, command=cmd)
            b.pack(side="left", padx=4)
            self.buttons.append(b)

        g = ttk.LabelFrame(f, text="3. CNN byte model (from CNNDataSet\\benign and CNNDataSet\\malicious)", padding=8)
        g.grid(row=4, column=0, columnspan=3, sticky="ew", pady=4)
        ttk.Label(g, text="Epochs").pack(side="left")
        self.epochs = ttk.Spinbox(g, from_=1, to=100, width=6)
        self.epochs.set(5)
        self.epochs.pack(side="left", padx=6)
        b = ttk.Button(g, text="Train CNN", command=self.train_cnn)
        b.pack(side="right")
        self.buttons.append(b)

        g = ttk.LabelFrame(f, text="4. Clean baseline (from a clean dump in the input folder)", padding=8)
        g.grid(row=5, column=0, columnspan=3, sticky="ew", pady=4)
        ttk.Label(g, text="Pick the clean dump in the Scan tab's dump list, then press the button. "
                          "Scan later dumps of the same VM with this baseline.").pack(side="left")
        b = ttk.Button(g, text="Build baseline", command=self.build_baseline)
        b.pack(side="right")
        self.buttons.append(b)

    def _name(self, prefix: str = "", suffix: str = ".joblib") -> str | None:
        name = "".join(c for c in self.name.get().strip() if c.isalnum() or c in "-_")
        if not name:
            messagebox.showinfo("HMS", "Enter a name to save the model as.")
            return None
        name = prefix + name
        if name + suffix in (BUNDLED, BUNDLED_FAMILY):
            messagebox.showinfo("HMS", f"{name}{suffix} is a bundled model. Choose another name.")
            return None
        if (MODELS / f"{name}{suffix}").exists() and not messagebox.askyesno(
                "HMS", f"models\\{name}{suffix} already exists. Replace it?"):
            return None
        return name

    def build_table(self) -> None:
        self.run(["build-table", "--benign", str(CNNDATA / "benign"), "--malicious",
                  str(CNNDATA / "malicious"), "--out", str(DATASET / "my_dumps.csv")],
                 "Building the feature table", lambda code: self.refresh())

    def _csv(self) -> str | None:
        if not self.csv.get():
            messagebox.showinfo("HMS", "Put a .csv feature table in the dataset folder, then press Refresh.")
            return None
        return str(DATASET / self.csv.get())

    def train_structural(self) -> None:
        csv, name = self._csv(), self._name()
        if csv and name:
            args = ["train-structural", "--csv", csv, "--model", self.algo.get(), "--split",
                    self.split.get(), "--select", self.select.get(), "--out", str(MODELS / f"{name}.joblib")]
            if self.compare.get():
                args.append("--baselines")
            self.run(args, "Training the structural model", lambda code: self.refresh())

    def train_family(self) -> None:
        csv, name = self._csv(), self._name("family_")
        if csv and name:
            self.run(["train-family", "--csv", csv, "--split", self.split.get(),
                      "--out", str(MODELS / f"{name}.joblib")],
                     "Training the family model", lambda code: self.refresh())

    def train_cnn(self) -> None:
        name = self._name(suffix=".pt")
        if name:
            from .preprocess import dump_files

            counts = [len(dump_files(CNNDATA / c)) for c in ("benign", "malicious")]
            if min(counts) < 2:
                messagebox.showinfo("HMS", "The CNN needs at least two dumps in CNNDataSet\\benign and two in "
                                           f"CNNDataSet\\malicious (found {counts[0]} and {counts[1]}).")
                return
            self.run(["train-cnn", "--benign", str(CNNDATA / "benign"), "--malicious",
                      str(CNNDATA / "malicious"), "--epochs", self.epochs.get(),
                      "--out", str(MODELS / f"{name}.pt")],
                     "Training the CNN", lambda code: self.refresh())

    def build_baseline(self) -> None:
        if not self.dump.get():
            messagebox.showinfo("HMS", "Copy the clean dump into the input folder and pick it in the Scan tab.")
            return
        dump = INPUT / self.dump.get()
        self.run(["baseline", str(dump), "--out", str(MODELS / f"baseline_{dump.stem}.npy")],
                 "Building the baseline", lambda code: self.refresh())

    # ---------------------------------------------------------------- common

    def refresh(self) -> None:
        from .preprocess import dump_files

        dumps = [str(p.relative_to(INPUT)) for p in dump_files(INPUT)]
        self.dump.config(values=dumps)
        if self.dump.get() not in dumps:
            self.dump.set(dumps[0] if dumps else "")
        m = _model_files()
        for box, files, first in ((self.structural, m["structural"], NO_MODEL), (self.family, m["family"], NONE),
                                  (self.cnn, m["cnn"], NONE), (self.baseline, m["baseline"], NONE)):
            values = [first] + files
            box.config(values=values)
            if box.get() not in values:
                box.set(files[0] if box in (self.structural, self.family, self.cnn) and files else first)
        if m["cnn"]:
            self.use_cnn_box.config(state="normal", text="Include the CNN in this scan")
        else:
            self.use_cnn.set(False)
            self.use_cnn_box.config(state="disabled", text="Include the CNN in this scan (train one first)")
        csvs = sorted(p.name for p in DATASET.glob("*.csv"))
        self.csv.config(values=csvs)
        if self.csv.get() not in csvs:
            self.csv.set(csvs[0] if csvs else "")

    def run(self, args: list[str], title: str, on_done=None) -> None:
        if self.proc is not None:
            return
        self.log.delete("1.0", "end")
        self.log.insert("end", "> python -m hms " + " ".join(args) + "\n\n")
        self.status.config(text=title + " ...")
        for b in self.buttons:
            b.config(state="disabled")
        self.stop_btn.config(state="normal")
        self.on_done = on_done
        self.stopped = False
        self.ok_codes = SCAN_OK_CODES if args[0] == "scan" else (0,)
        env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
        flags = subprocess.CREATE_NO_WINDOW if sys.platform.startswith("win") else 0
        self.proc = subprocess.Popen([_child_python(), "-m", "hms", *args], cwd=ROOT, env=env,
                                     stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                     encoding="utf-8", errors="replace", creationflags=flags)
        threading.Thread(target=self._read, args=(self.proc,), daemon=True).start()

    def _read(self, proc: subprocess.Popen) -> None:
        for line in proc.stdout:
            self.lines.put(line)
        self.lines.put(("done", proc.wait()))

    def _drain(self) -> None:
        try:
            while True:
                item = self.lines.get_nowait()
                if isinstance(item, tuple):
                    code = item[1]
                    self.proc = None
                    for b in self.buttons:
                        b.config(state="normal")
                    self.stop_btn.config(state="disabled")
                    if self.stopped:
                        self.status.config(text="Stopped")
                        self.log.insert("end", "\n[stopped]\n")
                    else:
                        self.status.config(text="Finished" if code in self.ok_codes
                                           else f"Failed (exit code {code})")
                        if self.on_done:
                            self.on_done(code)
                else:
                    self.log.insert("end", item)
                    self.log.see("end")
        except queue.Empty:
            pass
        self.after(100, self._drain)

    def stop(self) -> None:
        """Stop the running command and anything it started (Volatility)."""
        if self.proc is None:
            return
        self.stopped = True
        if sys.platform.startswith("win"):
            subprocess.run(["taskkill", "/PID", str(self.proc.pid), "/T", "/F"],
                           capture_output=True, creationflags=subprocess.CREATE_NO_WINDOW)
        else:
            self.proc.terminate()


def main() -> int:
    App().mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())

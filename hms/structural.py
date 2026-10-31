"""Structural channel: kernel-invariant metrics and the boosted-tree classifier.

Two parts:

1. Structural metrics from the report (section 5A), computed from Volatility 3
   plugin output saved as JSON (``vol -r json -f dump.raw windows.pslist``):
     * Cross-Plugin Agreement (AGR) between pslist and psscan  -> DKOM unlinking
     * Unlinked-DLL Ratio (UDR) from ldrmodules                -> reflective loads
   These only need the JSON files, so they work on any machine once someone has
   run Volatility. They have NOT yet been validated against real plugin output.

2. A boosted-tree classifier (XGBoost, or LightGBM) over the structural
   metrics plus tabular memory features in the CIC-MalMem-2022 layout (one row
   per memory dump, a ``Class`` column of Benign/Malware and a ``Category``
   column naming the family and sample). Following the report's architecture,
   AGR and UDR are model inputs: ``add_structural_metrics`` derives them from
   the psxview and ldrmodules columns, so the same formula applies to the CIC
   table and to a row extracted from a real dump.
"""

from __future__ import annotations

import json
import re
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from .metrics import binary_metrics

LABEL_COL = "Class"
CATEGORY_COL = "Category"


# --------------------------------------------------------------------------
# 1. Structural metrics from Volatility 3 JSON
# --------------------------------------------------------------------------

def load_vol_json(path: str | Path) -> list[dict]:
    """Load ``vol -r json`` output, flattening the ``__children`` tree."""
    rows: list[dict] = []

    def walk(items):
        for item in items:
            children = item.pop("__children", [])
            rows.append(item)
            walk(children)

    walk(json.loads(Path(path).read_text(encoding="utf-8")))
    return rows


def _pids(rows: list[dict]) -> set[int]:
    return {int(r["PID"]) for r in rows if r.get("PID") is not None}


@dataclass
class ProcessAgreement:
    agreement: float          # |pslist n psscan| / |pslist u psscan|
    hidden_pids: list[int]    # found by pool scanning, absent from the active list
    listed_only_pids: list[int]


def cross_plugin_agreement(pslist: list[dict], psscan: list[dict]) -> ProcessAgreement:
    """Compare linked-list enumeration with pool-tag scanning.

    A process that psscan finds but pslist does not has been unlinked from
    PsActiveProcessHead (DKOM) or has exited. Exited processes carry an
    ``ExitTime`` in psscan output and are not counted as hidden.
    """
    listed = _pids(pslist)
    live_scanned = {
        int(r["PID"])
        for r in psscan
        if r.get("PID") is not None and not r.get("ExitTime")
    }
    union = listed | live_scanned
    agreement = len(listed & live_scanned) / len(union) if union else 1.0
    return ProcessAgreement(
        agreement, sorted(live_scanned - listed), sorted(listed - live_scanned)
    )


@dataclass
class LoaderAgreement:
    unlinked_ratio: float     # unlinked modules / all mapped modules
    unlinked: list[dict]      # the offending ldrmodules rows
    unbacked: list[dict]      # unlinked from all three lists and no file behind it
    missing: dict             # per list: how many mapped modules it lacks


def unlinked_dll_ratio(ldrmodules: list[dict], dlllist: list[dict] | None = None) -> LoaderAgreement:
    """Fraction of mapped modules missing from the PEB loader lists.

    ldrmodules lists every image mapped into a process (from the VAD tree) and
    whether each of the three PEB lists (InLoadOrder, InInitializationOrder,
    InMemoryOrder) links it. A module is counted as unlinked when it is absent
    from InLoadOrder and InMemoryOrder and, when ``dlllist`` output is given,
    also from the standard DLL list for that process. Absence from
    InInitializationOrder alone is not counted, because the main executable is
    legitimately never in it; it is still reported in ``missing``.
    ``unbacked`` are unlinked from all three lists with no mapped file, the
    footprint of a reflectively loaded image.
    """
    def flag(value) -> bool:
        return value is True or str(value).lower() == "true"

    listed = None
    if dlllist is not None:
        listed = {(r.get("PID"), r.get("Base")) for r in dlllist}

    unlinked = [
        r for r in ldrmodules
        if not flag(r.get("InLoad")) and not flag(r.get("InMem"))
        and (listed is None or (r.get("Pid"), r.get("Base")) not in listed)
    ]
    unbacked = [r for r in unlinked if not flag(r.get("InInit")) and not r.get("MappedPath")]
    missing = {
        "InLoadOrder": sum(not flag(r.get("InLoad")) for r in ldrmodules),
        "InInitializationOrder": sum(not flag(r.get("InInit")) for r in ldrmodules),
        "InMemoryOrder": sum(not flag(r.get("InMem")) for r in ldrmodules),
    }
    if listed is not None:
        missing["dlllist"] = sum((r.get("Pid"), r.get("Base")) not in listed for r in ldrmodules)
    ratio = len(unlinked) / len(ldrmodules) if ldrmodules else 0.0
    return LoaderAgreement(ratio, unlinked, unbacked, missing)


# --------------------------------------------------------------------------
# 2. Tabular classifier (CIC-MalMem-2022 layout)
# --------------------------------------------------------------------------

AGR_COL = "structural.agr"
UDR_COL = "structural.udr"


def add_structural_metrics(X: pd.DataFrame) -> pd.DataFrame:
    """Add AGR and UDR columns computed from CIC-MalMem-2022 style counts.

    AGR: psxview lists every process seen by any method, so
    ``pslist.nproc + psxview.not_in_pslist`` is the union of pslist and the
    pool scan, and processes missing from either one are the disagreement.
    UDR: share of ldrmodules rows absent from both InLoad and InMem. The table
    only has the two rates separately; they are equal in all but a handful of
    rows, so their minimum is used.
    Columns that are missing leave the metric out.
    """
    X = X.copy()
    need = ("pslist.nproc", "psxview.not_in_pslist", "psxview.not_in_eprocess_pool")
    if all(c in X.columns for c in need):
        union = X["pslist.nproc"] + X["psxview.not_in_pslist"]
        missing = X["psxview.not_in_pslist"] + X["psxview.not_in_eprocess_pool"]
        X[AGR_COL] = np.where(union > 0, 1 - missing / union.where(union > 0, 1), 1.0)
        X[AGR_COL] = X[AGR_COL].clip(0, 1)
    need = ("ldrmodules.not_in_load_avg", "ldrmodules.not_in_mem_avg")
    if all(c in X.columns for c in need):
        X[UDR_COL] = X[list(need)].min(axis=1)
    return X


def load_feature_table(csv_path: str | Path, verbose: bool = True
                       ) -> tuple[pd.DataFrame, pd.Series | None, pd.Series | None]:
    """Return (features incl. AGR and UDR, labels or None, category or None).

    Only the CIC-MalMem-2022 feature columns are used. Any other column (an
    index written by pandas, an id, a timestamp) is ignored with a note, so it
    can't leak into a model that then can't scan a real dump. Cells that are
    not numbers (``?``, ``1,024``) become missing values instead of dropping
    the whole column.
    """
    from .extract import FEATURE_COLUMNS

    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError(f"{csv_path} has no rows")
    y = None
    if LABEL_COL in df.columns:
        y = (df[LABEL_COL].astype(str).str.strip().str.lower() != "benign").astype(int)
    category = df[CATEGORY_COL].astype(str) if CATEGORY_COL in df.columns else None
    known = [c for c in FEATURE_COLUMNS if c in df.columns]
    ignored = [c for c in df.columns if c not in known and c not in (LABEL_COL, CATEGORY_COL)]
    X = df[known].apply(pd.to_numeric, errors="coerce")
    bad = int((X.isna() & df[known].notna()).sum().sum())
    if verbose and ignored:
        print(f"note: ignoring {len(ignored)} column(s) that are not CIC-MalMem-2022 features: "
              f"{ignored[:5]}", file=sys.stderr)
    if verbose and bad:
        print(f"note: {bad} cell(s) were not numbers and are treated as missing", file=sys.stderr)
    return add_structural_metrics(X), y, category


def sample_groups(category: pd.Series, y: pd.Series) -> np.ndarray:
    """Group id per row so that all dumps of one malware sample stay together.

    CIC-MalMem-2022 holds several dumps per sample, named like
    ``Ransomware-Ako-<hash>-3.raw``. A random row split puts dumps of the same
    sample on both sides and inflates test scores. Benign rows have no sample
    id, so each one is its own group.
    """
    stem = category.str.replace(r"-\d+(\.raw)?$", "", regex=True)
    groups = stem.where(y == 1, "benign-" + pd.Series(np.arange(len(y)), index=y.index).astype(str))
    return pd.factorize(groups)[0]


def make_model(name: str = "xgboost", seed: int = 42):
    from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import LinearSVC
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.tree import DecisionTreeClassifier

    if name == "xgboost":
        try:
            from xgboost import XGBClassifier
        except ImportError:  # keep the project runnable without xgboost
            return HistGradientBoostingClassifier(random_state=seed)
        return XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.1, subsample=0.9,
            colsample_bytree=0.9, n_jobs=-1, random_state=seed, eval_metric="logloss",
        )
    if name == "lightgbm":
        try:
            from lightgbm import LGBMClassifier
        except ImportError:
            return HistGradientBoostingClassifier(random_state=seed)
        return LGBMClassifier(
            n_estimators=300, num_leaves=31, learning_rate=0.1, subsample=0.9,
            subsample_freq=1, colsample_bytree=0.9, random_state=seed, verbose=-1,
        )
    if name == "random_forest":
        return RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=seed)
    if name == "decision_tree":
        return DecisionTreeClassifier(random_state=seed)
    # KNN, LR and SVM do not accept missing values (a table made from your own
    # dumps has two empty psxview columns), so they fill them in first.
    def impute():
        return SimpleImputer(strategy="median", keep_empty_features=True)

    if name == "knn":
        return make_pipeline(impute(), StandardScaler(), KNeighborsClassifier(n_neighbors=5))
    if name == "logistic_regression":
        return make_pipeline(impute(), StandardScaler(), LogisticRegression(max_iter=2000))
    if name == "svm":
        return make_pipeline(
            impute(), StandardScaler(), CalibratedClassifierCV(LinearSVC(random_state=seed), cv=3)
        )
    raise ValueError(f"unknown model {name!r}")


BOOSTED = ("xgboost", "lightgbm")
SCAN_THRESHOLD = 0.90   # FusionConfig.structural_threshold
BASELINES = ("random_forest", "decision_tree", "knn", "logistic_regression", "svm")


def train_structural(
    csv_path: str | Path,
    out_path: str | Path | None = None,
    split: str = "group",
    test_size: float = 0.2,
    seed: int = 42,
    baselines: bool = False,
    model: str = "xgboost",
    select_features: int = 0,
) -> dict:
    """Train the structural classifier and report held-out metrics.

    ``model`` picks the boosted tree that is saved (``xgboost`` or
    ``lightgbm``); with ``baselines`` the other boosted tree and the classic
    baselines are scored too.

    ``split="group"`` keeps every dump of a malware sample on one side of the
    split; ``split="random"`` reproduces the row-level split most papers use.

    ``select_features=k`` keeps the k features with the highest ANOVA F-score
    on the training rows (report objective 3); 0 keeps all of them.
    """
    X, y, category = load_feature_table(csv_path)
    if y is None:
        raise ValueError(f"{csv_path} has no '{LABEL_COL}' column")
    if y.nunique() < 2:
        raise ValueError(f"{csv_path} holds only one class; the model needs benign and malware rows")
    X_tr, X_te, y_tr, y_te, train_idx, test_idx, split = _split(X, y, category, split, test_size, seed)
    if y_tr.nunique() < 2:
        raise ValueError("the training rows hold only one class; add more dumps or use --split random")
    X_tr, X_te, selected = _select(X_tr, X_te, y_tr, select_features)
    X = X[selected]

    results: dict = {
        "dataset": str(csv_path), "split": split, "seed": seed,
        "n_train": int(len(train_idx)), "n_test": int(len(test_idx)),
        "n_features": int(X.shape[1]), "model": model, "models": {},
    }
    if select_features:
        results["selected_features"] = selected
    if model not in BOOSTED:
        raise ValueError(f"model must be one of {BOOSTED}")
    others = tuple(m for m in BOOSTED if m != model)
    names = (model,) + ((others + BASELINES) if baselines else ())
    fitted = {}
    for name in names:
        clf = make_model(name, seed)
        if name == "knn":                       # a tiny table has fewer than 5 rows to compare with
            clf.set_params(kneighborsclassifier__n_neighbors=min(5, len(X_tr)))
        try:
            clf.fit(X_tr, y_tr)
        except ValueError as exc:
            if name == model:
                raise
            # A comparison model that can't train on this table (too few rows
            # per class, for example) is skipped, not fatal.
            results["models"][name] = {"skipped": str(exc).splitlines()[0][:200]}
            continue
        fitted[name] = clf
        results["models"][name] = binary_metrics(y_te, clf.predict_proba(X_te)[:, 1])
    # The table above uses the usual 0.5 cutoff; a scan flags at 0.90.
    results["threshold"] = 0.5
    results["at_scan_threshold"] = binary_metrics(
        y_te, fitted[model].predict_proba(X_te)[:, 1], threshold=SCAN_THRESHOLD)
    return _finish(results, fitted[model], X, category, test_idx, y_te, X_te, out_path)


def _split(X, y, category, split, test_size, seed):
    if split == "group" and category is not None:
        groups = sample_groups(category, y)
        splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        train_idx, test_idx = next(splitter.split(X, y, groups))
    else:
        split = "random"
        train_idx, test_idx = train_test_split(
            np.arange(len(X)), test_size=test_size, random_state=seed, stratify=y
        )
    X_tr, X_te, y_tr, y_te = X.iloc[train_idx], X.iloc[test_idx], y.iloc[train_idx], y.iloc[test_idx]
    return X_tr, X_te, y_tr, y_te, train_idx, test_idx, split


def _select(X_tr, X_te, y_tr, k):
    """ANOVA F-test feature selection fitted on the training rows only."""
    if not k or k >= X_tr.shape[1]:
        return X_tr, X_te, list(X_tr.columns)
    from sklearn.feature_selection import SelectKBest, f_classif

    # Columns that are empty in every row (median NaN) are filled with 0.
    filled = X_tr.fillna(X_tr.median()).fillna(0)
    with np.errstate(divide="ignore", invalid="ignore"), warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Features .* are constant")   # empty columns
        sel = SelectKBest(f_classif, k=k).fit(filled, y_tr)
    cols = list(X_tr.columns[sel.get_support()])
    return X_tr[cols], X_te[cols], cols


def _finish(results, main, X, category, test_idx, y_te, X_te, out_path):
    # Per-family recall of the main model, for the report's attribution table.
    if category is not None:
        family = category.iloc[test_idx].str.split("-").str[0]
        pred = main.predict(X_te)
        by_family = {}
        for fam in sorted(family[y_te.values == 1].unique()):
            mask = (family.values == fam) & (y_te.values == 1)
            by_family[fam] = {"n": int(mask.sum()), "recall": float(pred[mask].mean())}
        results["family_recall"] = by_family

    importances = getattr(main, "feature_importances_", None)
    if importances is not None:
        top = np.argsort(importances)[::-1][:15]
        results["top_features"] = [
            {"feature": X.columns[i], "importance": float(importances[i])} for i in top
        ]

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(portable_bundle({"kind": "structural", "model": main, "features": list(X.columns),
                                     "results": results}), out_path)
        out_path.with_suffix(".metrics.json").write_text(json.dumps(results, indent=2))
    return results


def train_family(
    csv_path: str | Path,
    out_path: str | Path | None = None,
    split: str = "group",
    test_size: float = 0.2,
    seed: int = 42,
) -> dict:
    """Multi-class model naming the family: Benign, Ransomware, Spyware or Trojan.

    Uses the family prefix of the CIC ``Category`` column and the same split
    rules as the binary model.
    """
    from sklearn.metrics import classification_report, confusion_matrix
    from xgboost import XGBClassifier

    X, y, category = load_feature_table(csv_path)
    if category is None or y is None:
        raise ValueError(f"{csv_path} needs '{CATEGORY_COL}' and '{LABEL_COL}' columns")
    family = category.str.split("-").str[0].where(y == 1, "Benign")
    X_tr, X_te, _, _, train_idx, test_idx, split = _split(X, y, category, split, test_size, seed)
    # The model can only learn the families that are in the training rows. On a
    # small table the split can put every dump of a family in the test rows;
    # those rows then count as errors instead of crashing the training.
    classes = sorted(family.iloc[train_idx].unique())
    if len(classes) < 2:
        raise ValueError("the training rows hold only one class; add more dumps or use --split random")
    unseen = sorted(set(family.iloc[test_idx]) - set(classes))
    codes = family.map({c: i for i, c in enumerate(classes)}).fillna(-1).astype(int)
    y_tr, y_te = codes.iloc[train_idx], codes.iloc[test_idx]   # unseen families are -1

    clf = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1, subsample=0.9,
                        colsample_bytree=0.9, n_jobs=-1, random_state=seed,
                        eval_metric="mlogloss").fit(X_tr, y_tr)
    pred = clf.predict(X_te)
    labels = list(range(len(classes)))
    report = classification_report(y_te, pred, labels=labels, target_names=classes,
                                   output_dict=True, zero_division=0)
    present = [c for i, c in enumerate(classes) if (y_te == i).any()]   # macro F1 over test classes
    results = {
        "dataset": str(csv_path), "split": split, "seed": seed, "classes": classes,
        "n_train": int(len(train_idx)), "n_test": int(len(test_idx)),
        "accuracy": float((pred == y_te.values).mean()),
        "macro_f1": float(np.mean([report[c]["f1-score"] for c in present])) if present else 0.0,
        "per_class": {c: {k: float(v) for k, v in report[c].items()} for c in classes},
        "confusion": (confusion_matrix(y_te, pred, labels=labels).tolist() if (y_te >= 0).any()
                      else [[0] * len(labels) for _ in labels]),
        "families_missing_from_training": unseen,
    }
    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(portable_bundle({"kind": "family", "model": clf, "features": list(X.columns),
                                     "classes": classes, "results": results}), out_path)
        out_path.with_suffix(".metrics.json").write_text(json.dumps(results, indent=2))
    return results


def predict_family(bundle: dict, X: pd.DataFrame) -> list[tuple[str, float]]:
    """(family, probability) for each row."""
    X = add_structural_metrics(X)
    proba = bundle["model"].predict_proba(X[bundle["features"]])
    return [(bundle["classes"][int(i)], float(p[int(i)])) for p, i in zip(proba, proba.argmax(1))]


def bundle_kind(bundle) -> str | None:
    """"structural" or "family" for a model saved by HMS, else None."""
    if not isinstance(bundle, dict) or "model" not in bundle or "features" not in bundle:
        return None
    return bundle.get("kind") or ("family" if "classes" in bundle else "structural")


def model_kind(path: str | Path) -> str | None:
    """Kind of a saved model from its .metrics.json, without loading the model."""
    metrics = Path(path).with_suffix(".metrics.json")
    try:
        data = json.loads(metrics.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if "classes" in data:
        return "family"
    return "structural" if "models" in data else None


def portable_bundle(bundle: dict) -> dict:
    """Store an XGBoost model in XGBoost's own JSON format instead of a pickle.

    A pickled XGBoost model only loads reliably in the version that saved it;
    the JSON format is supported by every later version.
    """
    model = bundle.get("model")
    if type(model).__name__ == "XGBClassifier":
        import xgboost

        bundle = {**bundle, "model": None, "xgboost_json": bytes(model.get_booster().save_raw("json")),
                  "saved_with_xgboost": xgboost.__version__}
    return bundle


def load_structural(path: str | Path, kind: str | None = None) -> dict:
    """Load a model saved by HMS; with ``kind``, refuse a model of the other kind."""
    bundle = joblib.load(path)
    if isinstance(bundle, dict) and bundle.get("xgboost_json") is not None:
        from xgboost import XGBClassifier

        model = XGBClassifier()
        model.load_model(bytearray(bundle["xgboost_json"]))
        bundle["model"] = model
    found = bundle_kind(bundle)
    if found is None:
        raise ValueError(f"{path} is not a model saved by HMS")
    if kind is not None and found != kind:
        raise ValueError(f"{Path(path).name} is a {found} model, not a {kind} model")
    return bundle


def structural_scores(bundle: dict, X: pd.DataFrame) -> np.ndarray:
    """Malicious probability for each row of a feature table."""
    X = add_structural_metrics(X)
    missing = [c for c in bundle["features"] if c not in X.columns]
    if missing:
        raise ValueError(f"feature table is missing {len(missing)} columns, e.g. {missing[:5]}")
    return bundle["model"].predict_proba(X[bundle["features"]])[:, 1]

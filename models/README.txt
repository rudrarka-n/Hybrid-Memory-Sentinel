cic_structural.joblib   XGBoost trained on CIC-MalMem-2022 (group split),
                        ready to use. Its test results are in
                        cic_structural.metrics.json and docs\RESULTS.md.

cic_family.joblib       XGBoost family classifier trained on CIC-MalMem-2022
                        (group split): Benign, Ransomware, Spyware or Trojan,
                        about 86% accurate. Results in cic_family.metrics.json
                        and docs\RESULTS.md.

Models you train in the HMS window are saved here too and show up in the
Scan tab. Each .joblib is sorted by its .metrics.json (family or structural),
or by its name if that file is missing ("family" in the name = family model).
*.pt files are CNNs and *.npy files are clean baselines.

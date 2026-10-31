"""Hybrid Memory Sentinel (HMS): malware detection using memory analysis.

Two channels feed one verdict:
  * structural channel - Volatility-style memory features -> boosted trees
  * byte channel       - raw 4 KB pages -> dedup/filter -> multi-scale 1D-CNN
"""

__version__ = "0.6.0"
PAGE_SIZE = 4096

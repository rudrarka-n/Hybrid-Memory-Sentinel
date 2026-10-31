"""Byte-channel preprocessing: paging, baseline deduplication, candidate filtering.

Implements the right-hand branch of the HMS architecture (report section 5):

    raw dump -> 4 KB pages -> drop all-zero pages
             -> drop pages identical to a known-clean baseline snapshot
             -> drop pages already seen in this dump
             -> keep pages that are high-entropy (H >= 6.8) or hold code that
                passes x86/x64 instruction validation

Works on any raw byte file (a .raw/.vmem/.dmp memory image, a process dump,
or a plain binary), so it can be exercised before a capture lab exists.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterator

import numpy as np

from . import PAGE_SIZE

# Common x86/x64 function prologues. They locate candidate functions cheaply;
# each hit is then confirmed by disassembling from it (see validate_code).
PROLOGUES: tuple[bytes, ...] = (
    b"\x55\x8b\xec",          # push ebp; mov ebp, esp
    b"\x8b\xff\x55\x8b\xec",  # mov edi, edi; push ebp; mov ebp, esp (hot-patch)
    b"\x55\x48\x89\xe5",      # push rbp; mov rbp, rsp
    b"\x48\x89\x5c\x24",      # mov [rsp+x], rbx
    b"\x48\x83\xec",          # sub rsp, imm8
    b"\x48\x81\xec",          # sub rsp, imm32
    b"\x40\x53\x48\x83\xec",  # push rbx; sub rsp, imm8
)


@dataclass
class FilterConfig:
    page_size: int = PAGE_SIZE
    high_entropy: float = 6.8    # report: Shannon entropy threshold H >= 6.8
    min_entropy: float = 1.0     # below this a page is padding / a fill pattern
    min_prologues: int = 1       # validated prologue hits needed to call a page code
    validate_code: bool = True   # confirm prologue hits by disassembly (needs capstone)
    dedup_within_dump: bool = True
    chunk_pages: int = 2048      # pages processed per read (8 MB at 4 KB pages)


@dataclass
class FilterStats:
    total_pages: int = 0
    zero_pages: int = 0
    baseline_duplicates: int = 0
    self_duplicates: int = 0
    low_information: int = 0
    not_code_like: int = 0
    candidates: int = 0
    high_entropy_candidates: int = 0
    code_like_candidates: int = 0

    @property
    def reduction(self) -> float:
        """Fraction of pages removed before classification."""
        if not self.total_pages:
            return 0.0
        return 1.0 - self.candidates / self.total_pages

    def to_dict(self) -> dict:
        d = asdict(self)
        d["reduction"] = round(self.reduction, 6)
        return d


@dataclass
class PageBatch:
    """Candidate pages that survived filtering, from one chunk of the dump."""

    offsets: np.ndarray      # int64, byte offset of each page in the file
    pages: np.ndarray        # uint8, shape (n, page_size)
    entropy: np.ndarray      # float64, Shannon entropy in bits per byte
    prologues: np.ndarray    # int32, prologue pattern hits per page

    def __len__(self) -> int:
        return len(self.offsets)


# Files that sit next to dumps but are not memory images.
NOT_DUMPS = {".json", ".csv", ".txt", ".md", ".failed", ".npy", ".joblib", ".pt", ".vmss",
             ".vmsn", ".log", ".ini", ".zip", ".7z"}


def dump_files(folder: str | Path) -> list[Path]:
    """Memory images under ``folder`` (recursive), skipping Volatility caches and notes."""
    return sorted(
        f for f in Path(folder).rglob("*")
        if f.is_file() and f.suffix.lower() not in NOT_DUMPS and not f.name.startswith(".")
        and not any(part.endswith("_vol") for part in f.relative_to(folder).parts[:-1])
    )


def page_entropy(pages: np.ndarray) -> np.ndarray:
    """Shannon entropy (bits per byte, 0..8) of each row of a uint8 array."""
    n, size = pages.shape
    if n == 0:
        return np.zeros(0)
    idx = pages.astype(np.int64) + (np.arange(n, dtype=np.int64)[:, None] << 8)
    counts = np.bincount(idx.ravel(), minlength=n * 256).reshape(n, 256)
    p = counts / size
    logp = np.log2(p, out=np.zeros_like(p), where=p > 0)
    return -(p * logp).sum(axis=1)


HOT_PATCH = b"\x8b\xff\x55\x8b\xec"
PLAIN = b"\x55\x8b\xec"


def count_prologues(page: bytes) -> int:
    # The hot-patch prologue contains the plain one, so count it only once.
    return sum(page.count(sig) for sig in PROLOGUES) - page.count(HOT_PATCH)


# Instruction validation: from a prologue hit, the next VALIDATE_WINDOW bytes
# must look like a real instruction stream in x86 or x64 mode:
#   * not mostly printable text (ASCII) or zeros (padding, UTF-16 strings),
#   * decode without an invalid opcode for at least VALIDATE_COVERAGE of the window,
#   * at least MIN_INSTRUCTIONS instructions using MIN_DISTINCT different mnemonics,
#   * at most MAX_RARE_RATIO of them rare/legacy opcodes (aaa, bound, in, out, ...),
#     which is what text and random bytes decode into.
# On random bytes after a prologue about 1 in 8 windows still passes; real
# compiled code passes over 99% of the time.
VALIDATE_WINDOW = 64
VALIDATE_COVERAGE = 0.75
MIN_INSTRUCTIONS = 8
MIN_DISTINCT = 4
MAX_RARE_RATIO = 0.10
MAX_PRINTABLE = 0.60
MAX_ZEROS = 0.35
MAX_HITS_CHECKED = 8
RARE_MNEMONICS = frozenset({
    "aaa", "aas", "aad", "aam", "daa", "das", "arpl", "bound", "into", "in", "out",
    "insb", "insw", "insd", "outsb", "outsw", "outsd", "lahf", "sahf", "salc", "hlt",
    "cli", "sti", "cmc", "les", "lds", "pushal", "popal", "pushaw", "popaw", "fwait",
    "wait", "int1", "icebp", "ljmp", "lcall", "retf", "iret", "iretd", "xlatb",
    "scasb", "lodsb", "stosb", "movsb", "cmpsb", "sldt", "str", "ud0", "ud1", "ud2",
})

_disassemblers = None


def _get_disassemblers():
    global _disassemblers
    if _disassemblers is None:
        try:
            import capstone
        except ImportError:  # keep the project runnable without capstone
            _disassemblers = ()
        else:
            _disassemblers = tuple(
                capstone.Cs(capstone.CS_ARCH_X86, mode)
                for mode in (capstone.CS_MODE_32, capstone.CS_MODE_64)
            )
    return _disassemblers


def code_validation_available() -> bool:
    return bool(_get_disassemblers())


def _decodes(window: bytes) -> bool:
    n = len(window)
    if sum(32 <= b < 127 for b in window) > MAX_PRINTABLE * n or window.count(0) > MAX_ZEROS * n:
        return False
    need = int(n * VALIDATE_COVERAGE)
    for md in _get_disassemblers():
        covered, mnemonics = 0, []
        for _addr, size, mnemonic, _ops in md.disasm_lite(window, 0):
            covered += size
            mnemonics.append(mnemonic)
        if (covered >= need and len(mnemonics) >= MIN_INSTRUCTIONS
                and len(set(mnemonics)) >= MIN_DISTINCT
                and sum(m in RARE_MNEMONICS for m in mnemonics) <= MAX_RARE_RATIO * len(mnemonics)):
            return True
    return False


def count_valid_prologues(page: bytes) -> int:
    """Prologue hits that disassemble into a run of valid instructions.

    Falls back to plain pattern hits when capstone is not installed.
    """
    if not code_validation_available():
        return count_prologues(page)
    hits = 0
    inside_hot_patch = set()
    for sig in sorted(PROLOGUES, key=lambda s: s != HOT_PATCH):   # hot-patch first
        # Each pattern gets its own budget, so many false hits of one pattern
        # (in data) cannot use up the checks for the others (x64 code).
        checked = 0
        start = page.find(sig)
        while start != -1 and checked < MAX_HITS_CHECKED:
            if sig == PLAIN and start in inside_hot_patch:
                start = page.find(sig, start + 1)
                continue
            if sig == HOT_PATCH:
                inside_hot_patch.add(start + 2)
            checked += 1
            window = page[start:start + VALIDATE_WINDOW]
            if len(window) == VALIDATE_WINDOW and _decodes(window):
                hits += 1
            start = page.find(sig, start + 1)
    return hits


def page_hashes(pages: np.ndarray) -> np.ndarray:
    """64-bit BLAKE2b digest of each page, as uint64."""
    out = np.empty(len(pages), dtype=np.uint64)
    for i, row in enumerate(pages):
        digest = hashlib.blake2b(row.tobytes(), digest_size=8).digest()
        out[i] = int.from_bytes(digest, "little")
    return out


def _read_chunks(path: Path, cfg: FilterConfig) -> Iterator[tuple[int, np.ndarray]]:
    """Yield (file_offset, pages) with pages shaped (n, page_size).

    A trailing partial page is ignored.
    """
    chunk_bytes = cfg.chunk_pages * cfg.page_size
    offset = 0
    with open(path, "rb") as fh:
        while True:
            buf = fh.read(chunk_bytes)
            n = len(buf) // cfg.page_size
            if n == 0:
                return
            pages = np.frombuffer(buf, dtype=np.uint8, count=n * cfg.page_size)
            yield offset, pages.reshape(n, cfg.page_size)
            offset += n * cfg.page_size
            if len(buf) < chunk_bytes:
                return


def build_baseline(path: str | Path, cfg: FilterConfig | None = None) -> np.ndarray:
    """Hash every non-zero page of a known-clean snapshot.

    Returns a sorted, de-duplicated uint64 array. Save it with ``np.save``.
    """
    cfg = cfg or FilterConfig()
    parts = []
    for _, pages in _read_chunks(Path(path), cfg):
        nonzero = pages[pages.any(axis=1)]
        if len(nonzero):
            parts.append(page_hashes(nonzero))
    if not parts:
        return np.zeros(0, dtype=np.uint64)
    return np.unique(np.concatenate(parts))


def load_baseline(path: str | Path) -> np.ndarray:
    arr = np.load(path)
    if arr.dtype != np.uint64:
        raise ValueError(f"{path} is not an HMS baseline (expected uint64 hashes)")
    return arr


@dataclass
class PageScanner:
    """Streams candidate pages out of a dump while accumulating statistics.

    >>> scanner = PageScanner("dump.raw", baseline=load_baseline("clean.npy"))
    >>> for batch in scanner:
    ...     ...
    >>> scanner.stats.reduction
    """

    path: str | Path
    baseline: np.ndarray | None = None
    cfg: FilterConfig = field(default_factory=FilterConfig)
    stats: FilterStats = field(default_factory=FilterStats)

    def __iter__(self) -> Iterator[PageBatch]:
        cfg, st = self.cfg, self.stats
        seen: set[int] = set()
        for base_off, pages in _read_chunks(Path(self.path), cfg):
            st.total_pages += len(pages)
            offsets = base_off + np.arange(len(pages), dtype=np.int64) * cfg.page_size

            # 1. all-zero blocks
            keep = pages.any(axis=1)
            st.zero_pages += int((~keep).sum())
            pages, offsets = pages[keep], offsets[keep]
            if not len(pages):
                continue

            # 2. pages identical to the clean baseline, 3. repeats in this dump
            if self.baseline is not None or cfg.dedup_within_dump:
                hashes = page_hashes(pages)
                keep = np.ones(len(pages), dtype=bool)
                if self.baseline is not None and len(self.baseline):
                    in_base = np.isin(hashes, self.baseline)
                    st.baseline_duplicates += int(in_base.sum())
                    keep &= ~in_base
                if cfg.dedup_within_dump:
                    for i in np.flatnonzero(keep):
                        h = int(hashes[i])
                        if h in seen:
                            keep[i] = False
                            st.self_duplicates += 1
                        else:
                            seen.add(h)
                pages, offsets = pages[keep], offsets[keep]
                if not len(pages):
                    continue

            # 4. valid code segment filtering
            ent = page_entropy(pages)
            keep = ent >= cfg.min_entropy
            st.low_information += int((~keep).sum())
            pages, offsets, ent = pages[keep], offsets[keep], ent[keep]

            count = count_valid_prologues if cfg.validate_code else count_prologues
            prol = np.fromiter(
                (count(p.tobytes()) for p in pages), dtype=np.int32, count=len(pages)
            )
            high = ent >= cfg.high_entropy
            code = prol >= cfg.min_prologues
            keep = high | code
            st.not_code_like += int((~keep).sum())
            st.candidates += int(keep.sum())
            st.high_entropy_candidates += int((high & keep).sum())
            st.code_like_candidates += int((code & keep).sum())
            if keep.any():
                yield PageBatch(offsets[keep], pages[keep], ent[keep], prol[keep])


def collect_candidates(
    path: str | Path,
    baseline: np.ndarray | None = None,
    cfg: FilterConfig | None = None,
    max_pages: int | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[PageBatch, FilterStats]:
    """Run the whole filter and return all candidates in memory.

    ``max_pages`` caps the result with a uniform random sample, which keeps
    training sets balanced across files of very different sizes.
    """
    cfg = cfg or FilterConfig()
    scanner = PageScanner(path, baseline, cfg)
    if max_pages is not None:
        return _sample_candidates(scanner, max_pages, rng or np.random.default_rng(0)), scanner.stats
    batches = list(scanner)
    if batches:
        batch = PageBatch(
            np.concatenate([b.offsets for b in batches]),
            np.concatenate([b.pages for b in batches]),
            np.concatenate([b.entropy for b in batches]),
            np.concatenate([b.prologues for b in batches]),
        )
    else:
        batch = PageBatch(
            np.zeros(0, np.int64),
            np.zeros((0, cfg.page_size), np.uint8),
            np.zeros(0),
            np.zeros(0, np.int32),
        )
    return batch, scanner.stats


def _sample_candidates(scanner: PageScanner, max_pages: int, rng: np.random.Generator) -> PageBatch:
    """Uniform random sample of at most ``max_pages`` candidates, in file order.

    Each page gets a random key and the ``max_pages`` smallest keys are kept
    as the scan goes, so memory stays at about one chunk plus the sample even
    for a multi-GB dump.
    """
    size = scanner.cfg.page_size
    kept = PageBatch(np.zeros(0, np.int64), np.zeros((0, size), np.uint8), np.zeros(0),
                     np.zeros(0, np.int32))
    keys = np.zeros(0)
    for b in scanner:
        keys = np.concatenate([keys, rng.random(len(b))])
        kept = PageBatch(np.concatenate([kept.offsets, b.offsets]), np.concatenate([kept.pages, b.pages]),
                         np.concatenate([kept.entropy, b.entropy]),
                         np.concatenate([kept.prologues, b.prologues]))
        if len(keys) > max_pages:
            pick = np.argpartition(keys, max_pages)[:max_pages]
            keys = keys[pick]
            kept = PageBatch(kept.offsets[pick], kept.pages[pick], kept.entropy[pick], kept.prologues[pick])
    order = np.argsort(kept.offsets)
    return PageBatch(kept.offsets[order], kept.pages[order], kept.entropy[order], kept.prologues[order])

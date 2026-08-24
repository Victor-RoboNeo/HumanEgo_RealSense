#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Download HumanEgo Aria data from the public HuggingFace dataset Leo-TX/HumanEgo.

No token / login required — the dataset is public.

  --task   serve_bread | water_flowers | all   (default: serve_bread)
  --num    N  or  all                          (default: 1 — the first N recordings)
  --input-only   download inputs only (run preprocessing yourself); skip the
                 precomputed `preprocess/` output. Otherwise it is included and each
                 recording's `all_data.tar` is auto-extracted.
  --out    download destination                (default: ./data)
  --keep-tar     keep all_data.tar after extracting (default: delete to save space)

Examples
--------
    python scripts/download_data.py                                # 1 serve_bread recording, full output
    python scripts/download_data.py --task serve_bread --num 2     # first 2 serve_bread, with output
    python scripts/download_data.py --task all       --num all     # the entire dataset (large)
    python scripts/download_data.py --task water_flowers --num 5 --input-only

Network (China campus / blocked huggingface.co)
----------------------------------------------
This script defaults to the hf-mirror endpoint and rewrites any absolute
``huggingface.co`` pagination / CDN URLs back to ``HF_ENDPOINT``. Override with:

    export HF_ENDPOINT=https://hf-mirror.com
    export HF_HOME=/data/LFT-W01_data/datasets/HumanEgo/.cache/huggingface
"""
import argparse
import os
import subprocess
import tarfile
import threading
import time

# Must be set BEFORE importing huggingface_hub (ENDPOINT is read at import time).
_DEFAULT_HF_ENDPOINT = "https://hf-mirror.com"
_DEFAULT_HF_HOME = "/data/LFT-W01_data/datasets/HumanEgo/.cache/huggingface"
os.environ.setdefault("HF_ENDPOINT", _DEFAULT_HF_ENDPOINT)
os.environ.setdefault("HF_HOME", _DEFAULT_HF_HOME)
os.makedirs(os.environ["HF_HOME"], exist_ok=True)


def _rewrite_hf_url(url: str) -> str:
    """Map absolute huggingface.co URLs onto HF_ENDPOINT (mirror Link headers)."""
    if not url:
        return url
    endpoint = os.environ.get("HF_ENDPOINT", _DEFAULT_HF_ENDPOINT).rstrip("/")
    for host in (
        "https://huggingface.co",
        "http://huggingface.co",
        "https://hf.co",
        "http://hf.co",
    ):
        if url.startswith(host):
            return endpoint + url[len(host) :]
    return url


def _patch_hf_mirror_routing() -> None:
    """
    Force every huggingface_hub HTTP call onto HF_ENDPOINT.

    Why Session-level: ``utils._pagination`` binds ``http_backoff`` at import
    time, and ``hf_api`` binds ``paginate`` at import time. Patching those
    module attributes is a no-op for already-imported names. Rewriting
    ``PreparedRequest.url`` inside the mounted HTTPAdapter catches first-page
    ``session.get`` calls and pagination ``Link: <https://huggingface.co/...>``
    follow-ups alike.
    """
    import requests
    from huggingface_hub.utils._http import (
        UniqueRequestIdAdapter,
        configure_http_backend,
    )

    class _MirrorAdapter(UniqueRequestIdAdapter):
        def send(self, request, *args, **kwargs):
            rewritten = _rewrite_hf_url(request.url)
            if rewritten != request.url:
                request.url = rewritten
            return super().send(request, *args, **kwargs)

    def _backend_factory():
        session = requests.Session()
        session.mount("http://", _MirrorAdapter())
        session.mount("https://", _MirrorAdapter())
        return session

    configure_http_backend(backend_factory=_backend_factory)


_patch_hf_mirror_routing()

from huggingface_hub import HfApi, snapshot_download
from huggingface_hub.constants import ENDPOINT as _HF_ENDPOINT_RESOLVED

print(f"[download_data] HF_ENDPOINT={_HF_ENDPOINT_RESOLVED}")
print(f"[download_data] HF_HOME={os.environ.get('HF_HOME')}")

REPO_ID = "Leo-TX/HumanEgo"
TASKS = ["serve_bread", "water_flowers"]

# Optional pretty UI (rich is a project dependency; fall back to plain prints if absent).
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.progress import (Progress, SpinnerColumn, TextColumn, BarColumn,
                               DownloadColumn, TransferSpeedColumn, TimeRemainingColumn)
    _C = Console()
    HAS_RICH = True
except Exception:
    HAS_RICH = False


# ---------------------------------------------------------------------------
# Hub helpers
# ---------------------------------------------------------------------------
def list_recordings(task, input_only):
    """{recording_path: total_bytes} for a task, sorted; honours --input-only."""
    try:
        tree = HfApi().list_repo_tree(REPO_ID, path_in_repo=f"{task}/aria",
                                      repo_type="dataset", recursive=True)
    except Exception:
        return {}
    sizes = {}
    for e in tree:
        size = getattr(e, "size", None)
        if size is None:                       # folder entry, not a file
            continue
        segs = e.path.split("/")
        vi = next((i for i, s in enumerate(segs) if s.endswith("_vrs")), None)
        if vi is None:
            continue
        if input_only and len(segs) > vi + 1 and segs[vi + 1] == "preprocess":
            continue
        rec = "/".join(segs[: vi + 1])
        sizes[rec] = sizes.get(rec, 0) + size
    return dict(sorted(sizes.items()))


def _dir_size(path):
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total


def _run_threaded(fn):
    """Run fn() in a daemon thread; return (thread, err_holder)."""
    err = {}

    def _wrap():
        try:
            fn()
        except Exception as e:  # surfaced to the caller after join()
            err["e"] = e

    th = threading.Thread(target=_wrap, daemon=True)
    th.start()
    return th, err


# ---------------------------------------------------------------------------
# Pretty (rich) path — one byte-progress bar per phase per recording
# ---------------------------------------------------------------------------
def _bar(*extra):
    return Progress(SpinnerColumn(), TextColumn("{task.description}"), BarColumn(),
                    *extra, TransferSpeedColumn(), TimeRemainingColumn(), console=_C)


def _recording_complete(recdir, input_only):
    """True if this recording already looks fully present on disk."""
    if not os.path.isdir(recdir):
        return False
    if not os.path.isfile(os.path.join(recdir, "sample.vrs")):
        return False
    if not os.path.isdir(os.path.join(recdir, "slam")):
        return False
    if input_only:
        return True
    # Prefer extracted frames; accept packed tar as "downloaded".
    if os.path.isdir(os.path.join(recdir, "preprocess", "all_data")):
        return True
    if os.path.isfile(os.path.join(recdir, "preprocess", "all_data.tar")):
        return True
    return False


def _is_rate_limited(exc):
    """Walk cause chain — hub often wraps 429 as LocalEntryNotFoundError."""
    seen = set()
    cur = exc
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        msg = str(cur).lower()
        if (
            "429" in msg
            or "too many requests" in msg
            or "rate limit" in msg
            or "quota" in msg
            or "resolvers requests" in msg
        ):
            return True
        # Also check HTTP status when present (HfHubHTTPError).
        status = getattr(cur, "response", None)
        if status is not None and getattr(status, "status_code", None) == 429:
            return True
        cur = getattr(cur, "__cause__", None) or getattr(cur, "__context__", None)
    return False


def _download_recording(rec, out, input_only, total, max_workers=2, max_retries=8):
    """
    Download one recording with low concurrency + 429 backoff.

    hf-mirror / Hub resolve quotas are easy to hit when snapshot_download
    HEADs many files with the default max_workers=8.
    """
    ignore = [f"{rec}/preprocess/*"] if input_only else None
    recdir = os.path.join(out, rec)
    if _recording_complete(recdir, input_only):
        if HAS_RICH:
            _C.print(f"  [green]skip[/] already present: {os.path.basename(rec)}")
        else:
            print(f"  skip already present: {os.path.basename(rec)}")
        return

    last_err = None
    for attempt in range(1, max_retries + 1):
        th, err = _run_threaded(lambda: snapshot_download(
            repo_id=REPO_ID,
            repo_type="dataset",
            local_dir=out,
            allow_patterns=[f"{rec}/*"],
            ignore_patterns=ignore,
            max_workers=max_workers,
        ))
        with _bar(DownloadColumn()) as p:
            t = p.add_task(
                f"[cyan]⬇ download[/]  {os.path.basename(rec)}"
                + (f"  (retry {attempt}/{max_retries})" if attempt > 1 else ""),
                total=max(total, 1),
            )
            while th.is_alive():
                p.update(t, completed=min(_dir_size(recdir), total))
                time.sleep(0.3)
            th.join()
            p.update(t, completed=max(min(_dir_size(recdir), total), 1))

        if "e" not in err:
            return

        last_err = err["e"]
        if not _is_rate_limited(last_err) or attempt >= max_retries:
            raise last_err

        # Hub message: 100000 resolvers / 5 minutes — wait out the window.
        wait_s = min(300, 30 * attempt)
        if HAS_RICH:
            _C.print(
                f"  [yellow]429 rate-limited[/] on {os.path.basename(rec)}; "
                f"sleeping {wait_s}s then retry ({attempt}/{max_retries})"
            )
        else:
            print(f"  429 rate-limited; sleeping {wait_s}s then retry "
                  f"({attempt}/{max_retries})")
        time.sleep(wait_s)

    raise last_err


def _extract_recording(tar_path, keep):
    out = os.path.dirname(tar_path)                       # the preprocess/ dir
    rec = os.path.basename(os.path.dirname(out))
    total = os.path.getsize(tar_path)                     # all_data.tar is uncompressed
    target = os.path.join(out, "all_data")               # tar unpacks into preprocess/all_data/
    th, err = _run_threaded(lambda: subprocess.run(
        ["tar", "-xf", tar_path, "-C", out], check=True))
    with _bar(DownloadColumn()) as p:
        t = p.add_task(f"[magenta]📦 extract [/]  {rec}", total=max(total, 1))
        while th.is_alive():
            p.update(t, completed=min(_dir_size(target), total))
            time.sleep(0.3)
        th.join()
        p.update(t, completed=max(total, 1))
    if "e" in err:
        raise err["e"]
    if not keep:
        os.remove(tar_path)


def run_rich(recs, sizes, out, input_only, keep, task_label):
    try:
        from huggingface_hub.utils import disable_progress_bars
        disable_progress_bars()                          # we render our own bars
    except Exception:
        pass
    grand = sum(sizes.values())
    mode = "inputs only" if input_only else "inputs + precomputed output"
    _C.print(Panel.fit(
        f"[bold]HumanEgo dataset[/]   task=[cyan]{task_label}[/]   "
        f"[cyan]{len(recs)}[/] recording(s)   {mode}\n"
        f"≈ [yellow]{grand / 1e9:.1f} GB[/]   →   [green]{os.path.abspath(out)}[/]",
        title="📥  HumanEgo Download", border_style="cyan"))
    os.makedirs(out, exist_ok=True)
    # Keep concurrency low — mirror resolve quota is shared / easy to trip.
    max_workers = int(os.environ.get("HF_HUB_MAX_WORKERS", "2"))
    pause_s = float(os.environ.get("HF_DOWNLOAD_PAUSE_S", "2"))
    for i, rec in enumerate(recs, 1):
        _C.rule(f"[bold cyan][{i}/{len(recs)}][/] {rec}")
        _download_recording(
            rec, out, input_only, sizes.get(rec, 0), max_workers=max_workers
        )
        if not input_only:
            tar = os.path.join(out, rec, "preprocess", "all_data.tar")
            all_data = os.path.join(out, rec, "preprocess", "all_data")
            if os.path.exists(tar) and not os.path.isdir(all_data):
                _extract_recording(tar, keep)
        if pause_s > 0 and i < len(recs):
            time.sleep(pause_s)
    nxt = ""
    if input_only:
        t0 = recs[0].split("/")[0]
        nxt = ("\n\n[bold]Next[/] — preprocess it:\n"
               f"  [green]python -m preprocess.Preprocess "
               f"--mps_path {os.path.join(out, recs[0])} --task {t0}[/]")
    _C.print(Panel.fit(
        f"[bold green]✓ Done[/]  •  {len(recs)} recording(s)  •  "
        f"[green]{os.path.abspath(out)}[/]{nxt}", border_style="green"))


# ---------------------------------------------------------------------------
# Plain fallback (no rich)
# ---------------------------------------------------------------------------
def run_plain(recs, out, input_only, keep):
    mode = "input only" if input_only else "input + precomputed output"
    print(f"Downloading {len(recs)} recording(s) ({mode}) -> {out}")
    os.makedirs(out, exist_ok=True)
    max_workers = int(os.environ.get("HF_HUB_MAX_WORKERS", "2"))
    pause_s = float(os.environ.get("HF_DOWNLOAD_PAUSE_S", "2"))
    for i, rec in enumerate(recs, 1):
        print(f"[{i}/{len(recs)}] {rec}")
        _download_recording(rec, out, input_only, 0, max_workers=max_workers)
        if not input_only:
            tar = os.path.join(out, rec, "preprocess", "all_data.tar")
            all_data = os.path.join(out, rec, "preprocess", "all_data")
            if os.path.exists(tar) and not os.path.isdir(all_data):
                print(f"  extracting {os.path.relpath(tar, out)}")
                subprocess.run(["tar", "-xf", tar, "-C", os.path.dirname(tar)], check=True)
                if not keep:
                    os.remove(tar)
        if pause_s > 0 and i < len(recs):
            time.sleep(pause_s)
    print("✅ done")
    if input_only:
        t0 = recs[0].split("/")[0]
        print(f"\nNext — preprocess: python -m preprocess.Preprocess "
              f"--mps_path {os.path.join(out, recs[0])} --task {t0}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", choices=TASKS + ["all"], default="serve_bread")
    ap.add_argument("--num", default="1", help="how many recordings (first N), or 'all' (default: 1)")
    ap.add_argument("--input-only", action="store_true",
                    help="download inputs only (run preprocessing yourself); skip precomputed output")
    ap.add_argument("--out", default="./data", help="download destination (default: ./data)")
    ap.add_argument("--keep-tar", action="store_true",
                    help="keep all_data.tar after extracting (default: delete to save space)")
    args = ap.parse_args()

    tasks = TASKS if args.task == "all" else [args.task]
    num = None if str(args.num).lower() == "all" else int(args.num)

    sizes = {}
    for t in tasks:
        found = list(list_recordings(t, args.input_only).items())
        if not found:
            print(f"  [warn] no recordings found for task '{t}' (not uploaded yet?)")
        sizes.update(dict(found if num is None else found[:num]))
    recs = list(sizes.keys())
    if not recs:
        raise SystemExit("Nothing to download.")

    if HAS_RICH:
        run_rich(recs, sizes, args.out, args.input_only, args.keep_tar, args.task)
    else:
        run_plain(recs, args.out, args.input_only, args.keep_tar)


if __name__ == "__main__":
    main()

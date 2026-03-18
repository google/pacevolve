#!/usr/bin/env python
"""Download or copy public MULTI-evolve benchmark data."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tarfile
import tempfile
import urllib.error
import urllib.request
import zipfile
from pathlib import Path


ZENODO_RECORD_ID = "17620759"


def load_manifest(manifest_path: Path, level: str) -> list[dict]:
    with manifest_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return list(payload[level])


def copy_from_source(manifest: list[dict], source_dir: Path, raw_dir: Path, force: bool) -> tuple[list[str], list[str]]:
    copied, missing = [], []
    for entry in manifest:
        filename = entry["DMS_filename"]
        src = source_dir / filename
        dst = raw_dir / filename
        if not src.exists():
            missing.append(filename)
            continue
        if dst.exists() and not force:
            copied.append(filename)
            continue
        shutil.copy2(src, dst)
        copied.append(filename)
    return copied, missing


def fetch_zenodo_record(record_id: str) -> dict:
    url = f"https://zenodo.org/api/records/{record_id}"
    with urllib.request.urlopen(url) as response:
        return json.loads(response.read().decode("utf-8"))


def download_file(url: str, destination: Path) -> None:
    with urllib.request.urlopen(url) as response, destination.open("wb") as handle:
        shutil.copyfileobj(response, handle)


def extract_archive(archive_path: Path, out_dir: Path) -> None:
    if archive_path.suffix == ".zip":
        with zipfile.ZipFile(archive_path) as zf:
            zf.extractall(out_dir)
    elif archive_path.suffixes[-2:] == [".tar", ".gz"] or archive_path.suffix == ".tgz":
        with tarfile.open(archive_path) as tf:
            tf.extractall(out_dir)


def collect_available_paths(root: Path) -> dict[str, Path]:
    available: dict[str, Path] = {}
    for path in root.rglob("*"):
        if path.is_file():
            available[path.name] = path
    return available


def download_from_zenodo(manifest: list[dict], raw_dir: Path, record_id: str, force: bool) -> tuple[list[str], list[str]]:
    record = fetch_zenodo_record(record_id)
    files = record.get("files", [])
    desired = {entry["DMS_filename"] for entry in manifest}

    direct_matches = {}
    archives = []
    for file_entry in files:
        key = file_entry.get("key") or file_entry.get("filename") or ""
        links = file_entry.get("links", {})
        url = links.get("self") or links.get("download") or links.get("content")
        if not url:
            continue
        if key in desired:
            direct_matches[key] = url
        elif key.endswith(".zip") or key.endswith(".tar.gz") or key.endswith(".tgz"):
            archives.append((key, url))

    downloaded = []
    missing = []
    for filename in desired:
        destination = raw_dir / filename
        if destination.exists() and not force:
            downloaded.append(filename)
            continue
        if filename in direct_matches:
            download_file(direct_matches[filename], destination)
            downloaded.append(filename)
        else:
            missing.append(filename)

    if not missing:
        return sorted(downloaded), []

    with tempfile.TemporaryDirectory(prefix="multievolve_zenodo_") as tmpdir:
        tmp_root = Path(tmpdir)
        for archive_name, url in archives:
            archive_path = tmp_root / archive_name
            download_file(url, archive_path)
            extract_archive(archive_path, tmp_root)

        available = collect_available_paths(tmp_root)
        still_missing = []
        for filename in missing:
            destination = raw_dir / filename
            if filename in available:
                shutil.copy2(available[filename], destination)
                downloaded.append(filename)
            else:
                still_missing.append(filename)
        missing = still_missing

    return sorted(set(downloaded)), sorted(set(missing))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download public MULTI-evolve benchmark datasets.")
    parser.add_argument(
        "--benchmark-level",
        default="lite",
        choices=["lite", "full"],
        help="Benchmark subset to prepare.",
    )
    parser.add_argument(
        "--record-id",
        default=ZENODO_RECORD_ID,
        help="Zenodo record ID containing the benchmark CSVs.",
    )
    parser.add_argument(
        "--source-dir",
        default=None,
        help="Optional local directory to copy dataset CSVs from instead of downloading.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite any existing files in the raw data directory.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    task_data_dir = Path(__file__).resolve().parent
    raw_dir = task_data_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(task_data_dir / "benchmark_manifest.json", args.benchmark_level)

    if args.source_dir:
        copied, missing = copy_from_source(manifest, Path(args.source_dir).expanduser(), raw_dir, args.force)
    else:
        try:
            copied, missing = download_from_zenodo(manifest, raw_dir, args.record_id, args.force)
        except urllib.error.URLError as exc:
            print(f"Failed to reach Zenodo: {exc}", file=sys.stderr)
            return 1

    print(f"Prepared raw dataset directory: {raw_dir}")
    print(f"Available files: {len(copied)}")
    for name in copied:
        print(f"  - {name}")
    if missing:
        print("Missing files:", file=sys.stderr)
        for name in missing:
            print(f"  - {name}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

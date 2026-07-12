#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import zipfile
from datetime import datetime
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = ROOT_DIR / "data_export_nearfield" / "_packages_fixed_deflate6" / "random_tree"
MOBILITY_DIRS = {"low_mobility", "high_mobility"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Package random trajectory CSI as a tree-preserving zip. "
            "Only csi_*.npz files are included, so the archive can be "
            "extracted into the matching weather directory."
        )
    )
    parser.add_argument("--source-root", required=True, help="Weather directory or mobility directory to package.")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR), help="Directory for generated zips.")
    parser.add_argument("--bundle-name", required=True, help="Output zip basename, without .zip.")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing archive.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned archive layout without writing.")
    parser.add_argument(
        "--compression",
        choices=["store", "deflate"],
        default="deflate",
        help="Use store for fast archiving, or deflate for smaller archives.",
    )
    parser.add_argument("--compression-level", type=int, default=6, help="Deflate compression level.")
    return parser.parse_args()


def frequency_band_from_path(path: Path) -> str | None:
    parent = path.parent.name
    if parent.startswith("f") and "GHz" in parent:
        return parent
    return None


def archive_name(source_root: Path, csi_path: Path) -> str:
    rel = csi_path.relative_to(source_root)
    if source_root.name in MOBILITY_DIRS:
        rel = Path(source_root.name) / rel
    return str(rel)


def collect_csi_files(source_root: Path) -> list[Path]:
    files: list[Path] = []
    for csi_path in source_root.rglob("csi_*.npz"):
        if not frequency_band_from_path(csi_path):
            continue
        rel_parts = set(csi_path.relative_to(source_root).parts)
        if source_root.name not in MOBILITY_DIRS and not (rel_parts & MOBILITY_DIRS):
            continue
        files.append(csi_path)
    return sorted(files)


def prepare_output(archive: Path, overwrite: bool, dry_run: bool) -> None:
    if not archive.exists():
        return
    if dry_run:
        print(f"[DryRun] Existing archive: {archive}")
        return
    if not overwrite:
        raise SystemExit(f"[Error] Output already exists. Pass --overwrite: {archive}")
    archive.unlink()


def human_size(path: Path) -> str:
    size = float(path.stat().st_size)
    for unit in ("B", "K", "M", "G", "T"):
        if size < 1024.0 or unit == "T":
            return f"{size:.0f}{unit}" if unit == "B" else f"{size:.1f}{unit}"
        size /= 1024.0
    return f"{path.stat().st_size}B"


def write_zip(source_root: Path, files: list[Path], archive: Path, compression: int, compression_level: int | None) -> None:
    seen: set[str] = set()
    tmp_archive = archive.with_suffix(archive.suffix + ".tmp")
    if tmp_archive.exists():
        tmp_archive.unlink()

    zip_kwargs = {"mode": "w", "compression": compression, "allowZip64": True}
    if compression == zipfile.ZIP_DEFLATED:
        zip_kwargs["compresslevel"] = compression_level

    try:
        with zipfile.ZipFile(tmp_archive, **zip_kwargs) as zf:
            for csi_path in files:
                arcname = archive_name(source_root, csi_path)
                if arcname in seen:
                    raise SystemExit(f"[Error] Duplicate archive path: {arcname}")
                seen.add(arcname)
                zf.write(csi_path, arcname)
        tmp_archive.replace(archive)
    finally:
        if tmp_archive.exists():
            tmp_archive.unlink()


def write_manifest(source_root: Path, archive: Path, files: list[Path], compression_name: str) -> None:
    manifest = archive.with_suffix(".manifest.txt")
    band_counts: dict[str, int] = {}
    mobility_counts: dict[str, int] = {}
    for path in files:
        band = frequency_band_from_path(path) or "unknown"
        band_counts[band] = band_counts.get(band, 0) + 1
        parts = path.relative_to(source_root).parts
        mobility = source_root.name if source_root.name in MOBILITY_DIRS else next(
            (part for part in parts if part in MOBILITY_DIRS),
            "unknown",
        )
        mobility_counts[mobility] = mobility_counts.get(mobility, 0) + 1

    with manifest.open("w", encoding="utf-8") as handle:
        handle.write(f"created_at={datetime.now().isoformat(timespec='seconds')}\n")
        handle.write(f"source_root={source_root}\n")
        handle.write(f"archive={archive}\n")
        handle.write(f"compression={compression_name}\n")
        handle.write(f"total_csi_npz={len(files)}\n\n")
        for mobility, count in sorted(mobility_counts.items()):
            handle.write(f"mobility={mobility} csi_npz={count}\n")
        handle.write("\n")
        for band, count in sorted(band_counts.items()):
            handle.write(f"band={band} csi_npz={count}\n")


def main() -> None:
    args = parse_args()
    source_root = Path(args.source_root).expanduser()
    out_dir = Path(args.out_dir).expanduser()
    if not source_root.is_absolute():
        source_root = ROOT_DIR / source_root
    if not out_dir.is_absolute():
        out_dir = ROOT_DIR / out_dir

    source_root = source_root.resolve()
    out_dir = out_dir.resolve()
    if not source_root.is_dir():
        raise SystemExit(f"[Error] Source root does not exist: {source_root}")

    archive = out_dir / f"{args.bundle_name}.zip"
    files = collect_csi_files(source_root)
    if not files:
        raise SystemExit(f"[Error] No tree-preserving random CSI files found under: {source_root}")

    print(f"[PackageTree] source={source_root}")
    print(f"[PackageTree] output={archive}")
    print(f"[PackageTree] csi_npz={len(files)} compression={args.compression}")

    if args.dry_run:
        for sample in files[:10]:
            print(f"[DryRun] {sample.relative_to(source_root)} -> {archive_name(source_root, sample)}")
        if len(files) > 10:
            print(f"[DryRun] ... {len(files) - 10} more files")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    prepare_output(archive, args.overwrite, args.dry_run)
    compression = zipfile.ZIP_STORED if args.compression == "store" else zipfile.ZIP_DEFLATED
    compression_level = None if args.compression == "store" else args.compression_level
    write_zip(source_root, files, archive, compression, compression_level)
    write_manifest(source_root, archive, files, args.compression)
    print(f"[PackageTree] done size={human_size(archive)} archive={archive}")


if __name__ == "__main__":
    main()

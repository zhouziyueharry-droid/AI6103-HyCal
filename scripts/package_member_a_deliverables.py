"""Package existing Member A evidence without changing experiment files.

Default/dry-run only inventories and hashes inputs. --build is explicit and refuses
to overwrite an existing ZIP. --verify verifies a ZIP without extracting it.
Python standard library only; this script never runs an experiment or a PDF builder.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import sys
import zipfile


ROOT = Path(__file__).resolve().parents[1]
FROZEN_COMMIT = "dc8d6a8fd2a158b517b42c715e0ae8a6894b4a2e"
PDF = "output/pdf/HyCal_Member_A_Research_Progress_CN.pdf"
OUTPUT = "output/pdf/HyCal_Member_A_Deliverables.zip"
BUNDLE_README = "reports/member_a_brief/BUNDLE_README.md"
DATASETS = ("flowers102", "dtd", "aircraft", "joint_flowers_dtd")
TREES = (
    "src", "scripts", "tests", "configs", "data/manifests",
    "reports/part_a", "reports/part_a_round2", "reports/member_a_brief",
    "results/part_a", "results/part_a_round2", "logs",
    "docs/superpowers/plans",
) + tuple(f"artifacts/features/{name}" for name in DATASETS)
EXCLUDED_PARTS = {
    ".git", ".venv", "venv", "external", "raw", "models", "checkpoints",
    "__pycache__", ".pytest_cache", "tmp", "temp", "previews",
    "archive_before_exact_covariance",
}
EXCLUDED_SUFFIXES = {".pyc", ".pyo", ".tmp", ".part", ".partial", ".bak", ".zip"}
REQUIRED = (
    "README.md", ".gitattributes", BUNDLE_README,
    "scripts/package_member_a_deliverables.py", "scripts/build_member_a_brief.py",
    "reports/member_a_brief/brief_content.json",
    "reports/member_a_brief/member_a_brief.html",
    "reports/member_a_brief/build_verification.json",
    "reports/member_a_brief/visual_qa.md",
    "reports/member_a_brief/research_tests_22.txt", "configs/environment_freeze.txt",
    "reports/part_a/report.md", "reports/part_a_round2/report.md",
    "reports/part_a/independent_audit.json",
    "reports/part_a_round2/complete_prediction_audit.json",
    "reports/part_a_round2/lock_verification.json",
    "results/part_a/execution_lock.json",
    "results/part_a_round2/final_execution_lock.json",
    "results/part_a_round2/diagnostic_lock.json",
    "results/part_a_round2/aircraft_registration.json",
    "results/part_a_round2/tenshot_registration.json",
) + tuple(f"artifacts/features/{name}/{file}" for name in DATASETS
          for file in ("features.pt", "metadata.json"))
CHUNK = 1024 * 1024


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def safe_relative(name: str) -> str:
    path = PurePosixPath(name)
    if (not name or "\\" in name or ":" in name or path.is_absolute()
            or any(p in {"", ".", ".."} for p in name.split("/"))):
        raise ValueError(f"Unsafe archive/relative path: {name!r}")
    return path.as_posix()


def excluded(relative: Path) -> bool:
    return (bool(set(relative.parts) & EXCLUDED_PARTS)
            or relative.suffix.lower() in EXCLUDED_SUFFIXES
            or relative.name in {".DS_Store", "Thumbs.db"}
            or relative.name.startswith("~$"))


def filesystem_link(path: Path) -> bool:
    """Also reject Windows junctions/reparse points, not only POSIX symlinks."""
    attributes = getattr(path.lstat(), "st_file_attributes", 0)
    return path.is_symlink() or bool(attributes & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))


def collect(root: Path, output: Path) -> tuple[list[dict], list[str], list[str]]:
    """Select only explicit trees, keeping repository-relative paths below partA/."""
    selected: dict[str, Path] = {}
    omitted: list[str] = []

    def add(path: Path, archive_path: str | None = None) -> None:
        relative = path.relative_to(root)
        if excluded(relative) or path.resolve() == output.resolve():
            omitted.append(relative.as_posix())
            return
        if filesystem_link(path) or not path.resolve().is_relative_to(root):
            raise ValueError(f"Symlink/out-of-root input is not permitted: {path}")
        name = safe_relative(archive_path or "partA/" + relative.as_posix())
        if name in selected:
            raise ValueError(f"Duplicate archive path: {name}")
        selected[name] = path

    for tree in TREES:
        directory = root / tree
        if not directory.is_dir():
            continue
        for ancestor in (directory, *directory.parents):
            if ancestor != root and ancestor.is_relative_to(root) and filesystem_link(ancestor):
                raise ValueError(f"Symlink/junction tree is not permitted: {ancestor}")
        for current, dirs, files in os.walk(directory, followlinks=False):
            base = Path(current)
            for name in list(dirs):
                path = base / name
                if excluded(path.relative_to(root)):
                    dirs.remove(name)
                    omitted.append(path.relative_to(root).as_posix() + "/")
                elif filesystem_link(path):
                    raise ValueError(f"Symlink directory is not permitted: {path}")
            for name in sorted(files):
                add(base / name)
    for path in [root / "README.md", root / ".gitattributes",
                 *sorted(root.glob("requirements*.txt"))]:
        if path.is_file():
            add(path)
    if (root / BUNDLE_README).is_file():
        add(root / BUNDLE_README, "README.md")
    if (root / PDF).is_file():
        add(root / PDF, Path(PDF).name)
    entries = []
    for archive_path, path in sorted(selected.items()):
        before = path.stat()
        sha256 = digest(path)
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError(f"Input changed while inventorying: {path}")
        entries.append({"archive_path": archive_path,
                        "source_relative_path": path.relative_to(root).as_posix(),
                        "size_bytes": after.st_size, "sha256": sha256})
    missing = [name for name in (*REQUIRED, PDF) if not (root / name).is_file()]
    names = [x["archive_path"].casefold() for x in entries]
    if len(names) != len(set(names)):
        raise ValueError("Archive contains case-insensitive filename collisions")
    return entries, missing, sorted(set(omitted))


def verify_inherited_locks(root: Path, entries: list[dict]) -> dict:
    by_source = {x["source_relative_path"]: x for x in entries}
    audit = json.loads((root / "reports/part_a_round2/lock_verification.json").read_text(encoding="utf-8"))
    expected_locks = {safe_relative(x["lock"]) for x in audit["checks"]}
    if not audit.get("passed") or len(expected_locks) != audit["n_locks"]:
        raise ValueError("Historical lock audit is failed or internally inconsistent")
    missing_locks = expected_locks - by_source.keys()
    if missing_locks:
        raise ValueError(f"Previously audited locks missing from bundle: {sorted(missing_locks)}")
    checks = []
    for source in sorted(by_source):
        p = PurePosixPath(source)
        if not (source.startswith("results/") and p.suffix == ".json"
                and ("lock" in p.stem or "registration" in p.stem)):
            continue
        payload = json.loads((root / source).read_text(encoding="utf-8"))
        references = payload.get("files", payload.get("provenance"))
        if not isinstance(references, dict) or not references:
            raise ValueError(f"Inherited lock has no file hash map: {source}")
        for name, expected in references.items():
            relative = safe_relative(name.removeprefix("source:").replace("\\", "/"))
            if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
                raise ValueError(f"Invalid inherited SHA256: {source}: {name}")
            entry = by_source.get(relative)
            if entry is None:
                raise ValueError(f"Lock dependency missing from bundle: {source}: {relative}")
            if entry["sha256"] != expected:
                raise ValueError(f"Inherited lock mismatch: {source}: {relative}")
            checks.append({"lock": source, "source_relative_path": relative,
                           "sha256": expected})
    return {"passed": True, "n_locks": len({x["lock"] for x in checks}),
            "n_checks": len(checks), "checks": checks}


def make_manifest(entries: list[dict], omitted: list[str], lock_audit: dict) -> dict:
    return {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "research_snapshot_commit": FROZEN_COMMIT,
        "scope": "Existing Member A results, four feature caches, and report sources",
        "path_mapping": "Source-relative paths retained under partA; PDF and delivery README promoted to ZIP root",
        "manifest_self_hash": "Not included: manifest.json inventories payload files, not itself",
        "byte_preservation": "No newline, JSON, path, prediction, or lock rewrites",
        "excluded_assets": ["raw datasets", "model checkpoint", "upstream repository",
                            "git history", "virtual environment", "temporary outputs",
                            "archive_before_exact_covariance"],
        "explicit_tree_roots": list(TREES),
        "omitted_paths": omitted,
        "inherited_lock_verification": lock_audit,
        "payload_file_count": len(entries),
        "payload_size_bytes": sum(x["size_bytes"] for x in entries),
        "files": entries,
    }


def verify_archive(path: Path, root: Path | None = None) -> dict:
    """Check CRC, exact membership and streaming SHA; optionally compare source bytes."""
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)) or len(names) != len({n.casefold() for n in names}):
            raise ValueError("Duplicate ZIP entry")
        for name in names:
            safe_relative(name)
        bad = archive.testzip()
        if bad:
            raise ValueError(f"ZIP CRC failure: {bad}")
        manifest = json.loads(archive.read("manifest.json"))
        entries = manifest["files"]
        expected_names = {safe_relative(x["archive_path"]) for x in entries}
        if len(expected_names) != len(entries) or set(names) != expected_names | {"manifest.json"}:
            raise ValueError("ZIP membership differs from manifest")
        for row in entries:
            info = archive.getinfo(row["archive_path"])
            if info.file_size != row["size_bytes"]:
                raise ValueError(f"ZIP size mismatch: {info.filename}")
            sha256 = hashlib.sha256()
            source = None
            if root is not None:
                relative = safe_relative(row["source_relative_path"])
                candidate = root / relative
                if filesystem_link(candidate) or not candidate.resolve().is_relative_to(root):
                    raise ValueError(f"Unsafe source path: {relative}")
                source = candidate.open("rb")
            try:
                with archive.open(info) as stream:
                    while block := stream.read(CHUNK):
                        sha256.update(block)
                        if source is not None and source.read(len(block)) != block:
                            raise ValueError(f"ZIP differs byte-for-byte from source: {info.filename}")
                if source is not None and source.read(1):
                    raise ValueError(f"Source grew after inventory: {info.filename}")
            finally:
                if source is not None:
                    source.close()
            if sha256.hexdigest() != row["sha256"]:
                raise ValueError(f"ZIP SHA256 mismatch: {info.filename}")
    return {"passed": True, "payload_files": len(entries), "crc_verified": True,
            "all_payload_sha256_verified": True, "source_bytes_compared": root is not None,
            "zip_size_bytes": path.stat().st_size, "zip_sha256": digest(path)}


def build(root: Path, output: Path, manifest: dict) -> dict:
    if output.exists():
        raise FileExistsError(f"Refusing to replace existing archive: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive create also protects against a file appearing after the exists check.
    with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for row in manifest["files"]:
            info = zipfile.ZipInfo(row["archive_path"], date_time=(2026, 11, 17, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            with (root / row["source_relative_path"]).open("rb") as source:
                with archive.open(info, "w", force_zip64=True) as destination:
                    shutil.copyfileobj(source, destination, CHUNK)
        archive.writestr("manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    return verify_archive(output, root)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="Inventory/hash only (default); write nothing")
    mode.add_argument("--build", action="store_true", help="Create a new ZIP only after the final PDF is approved")
    mode.add_argument("--verify", type=Path, metavar="ZIP", help="Read-only verification of a previously built ZIP")
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, help=f"Default: {OUTPUT}; never overwritten")
    args = parser.parse_args()
    root = args.repo_root.resolve()
    if args.verify:
        print(json.dumps(verify_archive(args.verify.resolve()), ensure_ascii=False, indent=2))
        return
    output = (args.output if args.output else root / OUTPUT).resolve()
    entries, missing, omitted = collect(root, output)
    locks = verify_inherited_locks(root, entries)
    manifest = make_manifest(entries, omitted, locks)
    group_counts = Counter("/".join(x["source_relative_path"].split("/")[:2]) for x in entries)
    summary = {"mode": "build" if args.build else "dry-run", "output": str(output),
               "payload_files": len(entries), "payload_bytes": manifest["payload_size_bytes"],
               "missing_required_inputs": missing, "inherited_locks": locks["n_locks"],
               "inherited_lock_checks": locks["n_checks"], "group_counts": dict(group_counts),
               "excluded_paths_count": len(omitted), "writes_performed": False}
    if args.build:
        if missing:
            raise FileNotFoundError(f"Cannot build; required inputs missing: {missing}")
        summary["verification"] = build(root, output, manifest)
        summary["writes_performed"] = True
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()

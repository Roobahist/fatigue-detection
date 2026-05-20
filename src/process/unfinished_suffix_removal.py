from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, List, Tuple


@dataclass(frozen=True)
class RenamePlan:
    src: Path
    dst: Path


def is_unfinished_json(path: Path) -> bool:
    return path.is_file() and path.suffix == ".json" and path.stem.endswith("_unfinished")


def to_finished_json_path(unfinished_json: Path) -> Path:
    # foo_unfinished.json -> foo.json
    if not is_unfinished_json(unfinished_json):
        raise ValueError(f"Not an unfinished .json file: {unfinished_json}")
    return unfinished_json.with_name(unfinished_json.name.replace("_unfinished.json", ".json"))


def corresponding_dat_path(finished_json: Path) -> Path:
    # foo.json -> foo.dat
    return finished_json.with_suffix(".dat")


def build_rename_plans(dir_path: Path) -> Tuple[List[RenamePlan], List[str]]:
    """
    Looks for:
      - unfinished metadata exists (foo_unfinished.json)
      - finished data exists (foo.dat)
    Then renames:
      foo_unfinished.json -> foo.json

    Skips:
      - if foo.dat is missing (still unfinished or inconsistent)
      - if foo.json already exists (avoid overwriting)
    """
    plans: List[RenamePlan] = []
    notes: List[str] = []

    for unfinished_json in dir_path.rglob("*_unfinished.json"):
        finished_json = to_finished_json_path(unfinished_json)
        dat_path = corresponding_dat_path(finished_json)

        if not dat_path.exists():
            notes.append(f"skip_missing_dat unfinished_json={unfinished_json} expected_dat={dat_path}")
            continue

        if finished_json.exists():
            notes.append(f"skip_exists finished_json_already_exists={finished_json}")
            continue

        plans.append(RenamePlan(src=unfinished_json, dst=finished_json))

    return plans, notes


def apply_plans(
    plans: Iterable[RenamePlan],
    rename_fn: Callable[[Path, Path], None],
    dry_run: bool,
) -> Tuple[int, int, List[str]]:
    renamed = 0
    skipped = 0
    messages: List[str] = []

    for p in plans:
        if dry_run:
            messages.append(f"dry_run rename {p.src} -> {p.dst}")
            skipped += 1
            continue

        try:
            p.dst.parent.mkdir(parents=True, exist_ok=True)
            rename_fn(p.src, p.dst)  # atomic within same filesystem
            messages.append(f"renamed {p.src} -> {p.dst}")
            renamed += 1
        except Exception as e:
            messages.append(f"failed {p.src} -> {p.dst} err={type(e).__name__}: {e}")
            skipped += 1

    return renamed, skipped, messages


def atomic_rename(src: Path, dst: Path) -> None:
    os.replace(src, dst)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fix memmap metadata names: foo_unfinished.json -> foo.json when foo.dat exists."
    )
    parser.add_argument("directory", type=str, help="Root directory to scan recursively.")
    parser.add_argument("--dry-run", action="store_true", help="Print actions without renaming.")
    args = parser.parse_args()

    root = Path(args.directory).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        raise SystemExit(f"Not a directory: {root}")

    plans, notes = build_rename_plans(root)
    renamed, skipped, messages = apply_plans(plans, rename_fn=atomic_rename, dry_run=args.dry_run)

    for n in notes:
        print(n)
    for m in messages:
        print(m)

    print(f"summary planned={len(plans)} renamed={renamed} skipped={skipped} dry_run={args.dry_run}")


if __name__ == "__main__":
    main()
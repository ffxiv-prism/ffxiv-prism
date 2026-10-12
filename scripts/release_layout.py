"""Build-only inference layout; never rename the development dataset in place."""
import ast
from contextlib import closing
import json
from pathlib import Path
import re
import sqlite3

IMAGE_PREFIX = "data/raw/eorzea_collection/images/"


def index_names(selection):
    result = {}
    for name, path in selection["candidate_indexes"].items():
        slot = next(s for s in ("head", "body", "hands", "legs", "feet") if s in name)
        model = next(m for m in ("dinov2", "dinov3", "siglip2") if m in name)
        target = f"data/indexes/{slot}-{model}"
        if target in result.values():
            raise ValueError(f"Ambiguous inference index: {target}")
        result[path] = target
    return result


def resource_name(name, indexes):
    if name.startswith(IMAGE_PREFIX):
        return "data/images/" + name.rsplit("/", 1)[-1]
    for old, new in indexes.items():
        if name == old or name.startswith(old + "/"):
            return new + name[len(old):]
    return name


def rewrite_paths(value, indexes):
    if isinstance(value, str):
        return resource_name(value.replace("\\", "/"), indexes) if value.replace("\\", "/").startswith("data/") else value
    if isinstance(value, list):
        return [rewrite_paths(item, indexes) for item in value]
    if isinstance(value, dict):
        return {key: rewrite_paths(item, indexes) for key, item in value.items()}
    return value


def compact_resources(files, selection, staging):
    """Rewrite index/image references, including SQLite, into a collision-checked layout."""
    indexes = index_names(selection)
    result, seen = {}, set()
    for old, source in sorted(files.items()):
        name = resource_name(old, indexes)
        if name.casefold() in seen:
            raise ValueError(f"Duplicate compact resource path: {name}")
        seen.add(name.casefold())
        target = staging / name
        if source.suffix in (".json", ".jsonl"):
            target.parent.mkdir(parents=True, exist_ok=True)
            if source.suffix == ".jsonl":
                with source.open(encoding="utf-8") as src, target.open("w", encoding="utf-8") as dst:
                    for line in src:
                        if line.strip():
                            dst.write(json.dumps(rewrite_paths(json.loads(line), indexes), ensure_ascii=False) + "\n")
            else:
                target.write_text(json.dumps(rewrite_paths(json.loads(source.read_text("utf-8")), indexes), ensure_ascii=False, indent=2), "utf-8")
            source = target
        elif source.suffix == ".sqlite":
            # This is already an isolated, consistent build backup.
            with closing(sqlite3.connect(source)) as db, db:
                db.create_function("compact_path", 1, lambda path: resource_name(path.replace("\\", "/"), indexes) if path else path)
                for (table,) in db.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall():
                    if not re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", table):
                        raise ValueError(f"Unexpected database table: {table}")
                    columns = {row[1] for row in db.execute(f'PRAGMA table_info("{table}")')}
                    if "local_path" in columns:
                        db.execute(f'UPDATE "{table}" SET local_path=compact_path(local_path)')
        result[name] = source
    return result


def inference_sources(root):
    """Follow local Python imports from the three shipped inference entry points."""
    entries = ["scripts/recognize_glamour.py", "scripts/recognition_worker.py", "scripts/web_app.py"]
    pending = [root / name for name in entries]
    found = set()
    while pending:
        path = pending.pop()
        if path in found:
            continue
        found.add(path)
        if path.is_relative_to(root / "src"):
            parent = path.parent
            while parent != root / "src":
                init = parent / "__init__.py"
                if init.is_file() and init not in found:
                    pending.append(init)
                parent = parent.parent
        for node in ast.walk(ast.parse(path.read_text("utf-8-sig"))):
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if node.level:
                    package = path.parent.relative_to(root / "src").parts
                    module = ".".join((*package[:len(package) - node.level + 1], module)).rstrip(".")
                names = [module, *(module + "." + alias.name for alias in node.names)]
            for name in names:
                if not name.startswith("ffxiv_prism"):
                    continue
                base = root / "src" / name.replace(".", "/")
                candidate = base.with_suffix(".py")
                if not candidate.is_file():
                    candidate = base / "__init__.py"
                if candidate.is_file() and candidate not in found:
                    pending.append(candidate)
    return {path.relative_to(root).as_posix(): path for path in found}

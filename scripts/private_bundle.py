r"""Paquete privado: los archivos que no se versionan (datos con derechos de autor, índice, resultados).

    python scripts/private_bundle.py pack   [--out RUTA.tar.gz] [--no-processed] [--no-index] [--no-results]
                                            [--runs CARPETA ...] [--all-indexes]
    python scripts/private_bundle.py verify RUTA.tar.gz
    python scripts/private_bundle.py unpack RUTA.tar.gz            # en la raíz del repositorio clonado

El paquete es un .tar.gz con las mismas rutas del repositorio y un MANIFEST.sha256 (huella de cada archivo).
`unpack` extrae, verifica las huellas y marca los índices como más recientes que los fragmentos, para que el
RAG los reutilice en lugar de reconstruirlos (en un equipo sin GPU, reconstruir el índice neural es lento).

El paquete contiene textos con derechos de autor: guardarlo solo en un lugar privado, nunca en git.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import os
import sys
import tarfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = "MANIFEST.sha256"
FINAL_RUNS = ["all_local-phi4mini_en_20261112_165914", "all_local-gemma3_4b_en_20261112_162300"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def collect(a) -> list[Path]:
    """Rutas relativas a la raíz del repositorio que entran al paquete."""
    files: list[Path] = [p for p in (ROOT / "data" / "raw").rglob("*") if p.is_file()]
    if not a.no_processed:
        files += sorted((ROOT / "data" / "processed").glob("*.jsonl"))
    if not a.no_index:
        idx = ROOT / "reports" / "indexes"
        files += sorted(idx.glob("*.pkl")) if a.all_indexes else [idx / "all_neural.pkl"]
    if not a.no_results:
        rep = ROOT / "reports"
        files += [p for p in (rep / "figures").rglob("*") if p.is_file()]
        files += sorted(rep.glob("retrieval_*.csv"))
        files += [p for p in (rep / "cache").rglob("*") if p.is_file()]
        for run in (a.runs or FINAL_RUNS):
            files += [p for p in (rep / "runs" / run).rglob("*") if p.is_file()]
    missing = [p for p in files if not p.exists()]
    if missing:
        sys.exit("No existen:\n  " + "\n  ".join(str(p.relative_to(ROOT)) for p in missing))
    return [p.relative_to(ROOT) for p in files]


def pack(a) -> None:
    rel = collect(a)
    out = Path(a.out or ROOT.parent / f"RAG_Movies_Books_privado_{time.strftime('%Y%m%d')}.tar.gz").resolve()
    if ROOT in out.parents:
        sys.exit("El paquete debe guardarse fuera del repositorio (--out).")
    manifest = "".join(f"{sha256(ROOT / r)}  {r.as_posix()}\n" for r in rel)
    with tarfile.open(out, "w:gz") as tar:
        for r in rel:
            tar.add(ROOT / r, arcname=r.as_posix())          # conserva la fecha de modificación
        info = tarfile.TarInfo(MANIFEST)
        data = manifest.encode("utf-8")
        info.size, info.mtime = len(data), int(time.time())
        tar.addfile(info, io.BytesIO(data))
    size = out.stat().st_size / 2**20
    print(f"{len(rel)} archivos · {size:.0f} MB → {out}")


def _read_manifest(tar: tarfile.TarFile) -> dict[str, str]:
    text = tar.extractfile(MANIFEST).read().decode("utf-8")
    return {line[66:]: line[:64] for line in text.splitlines() if line}


def verify(a) -> int:
    with tarfile.open(a.bundle, "r:gz") as tar:
        expected = _read_manifest(tar)
        bad = 0
        for name, digest in expected.items():
            h = hashlib.sha256(tar.extractfile(name).read()).hexdigest()
            if h != digest:
                bad += 1
                print(f"DISTINTO  {name}")
    print(f"{len(expected) - bad}/{len(expected)} archivos correctos")
    return 1 if bad else 0


def unpack(a) -> int:
    with tarfile.open(a.bundle, "r:gz") as tar:
        expected = _read_manifest(tar)
        members = [m for m in tar.getmembers() if m.name != MANIFEST]
        for m in members:                                   # sin rutas absolutas ni '..'
            if m.name.startswith("/") or ".." in Path(m.name).parts:
                sys.exit(f"Ruta no permitida en el paquete: {m.name}")
        tar.extractall(ROOT, members=members)
    bad = [n for n, d in expected.items() if sha256(ROOT / n) != d]
    for n in bad:
        print(f"DISTINTO  {n}")
    now = time.time()                                       # índices más nuevos que los fragmentos
    for p in (ROOT / "reports" / "indexes").glob("*.pkl"):
        os.utime(p, (now, now))
    print(f"{len(expected) - len(bad)}/{len(expected)} archivos restaurados y verificados en {ROOT}")
    return 1 if bad else 0


def main() -> int:
    ap = argparse.ArgumentParser(prog="private_bundle", description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pack")
    p.add_argument("--out", default=None, help="ruta del .tar.gz (fuera del repositorio)")
    p.add_argument("--no-processed", action="store_true", help="sin data/processed (se regenera con rmb build)")
    p.add_argument("--no-index", action="store_true", help="sin el índice (se regenera con rmb index)")
    p.add_argument("--all-indexes", action="store_true", help="todos los índices, no solo all_neural.pkl")
    p.add_argument("--no-results", action="store_true", help="sin figuras, métricas ni corridas")
    p.add_argument("--runs", nargs="*", default=None, help="corridas de reports/runs a incluir")
    for name in ("verify", "unpack"):
        sub.add_parser(name).add_argument("bundle")
    a = ap.parse_args()
    if a.cmd == "pack":
        pack(a)
        return 0
    return verify(a) if a.cmd == "verify" else unpack(a)


if __name__ == "__main__":
    sys.exit(main())

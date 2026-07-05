"""Tum .py kaynak dosyalarinin basina AGPL-3.0 bildirimi ekler (idempotent).

Kullanim: python add_license_headers.py [--check]
  --check : eksik header'lari yalnizca listeler, dosya degistirmez (CI icin).

Yeni dosya ekledikten sonra tekrar calistir; header'i olan dosyalara dokunmaz.
"""
import argparse, os, sys

HEADER = """\
# Hyper Flux Projection (HFP) — O(1)-memory causal language model
# Copyright (C) 2026 Kayrahan Yılmaz
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""
MARKER = "GNU Affero General Public License"
SKIP_DIRS = {"__pycache__", ".git", "_legacy_reference", "hf_release"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    root = os.path.dirname(os.path.abspath(__file__))
    missing, added = [], []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for fn in sorted(filenames):
            if not fn.endswith(".py"):
                continue
            path = os.path.join(dirpath, fn)
            with open(path, encoding="utf-8") as f:
                src = f.read()
            if MARKER in src[:1200]:
                continue  # header zaten var
            missing.append(os.path.relpath(path, root))
            if not args.check:
                # shebang varsa koru
                if src.startswith("#!"):
                    line, rest = src.split("\n", 1)
                    src = line + "\n" + HEADER + rest
                else:
                    src = HEADER + src
                with open(path, "w", encoding="utf-8") as f:
                    f.write(src)
                added.append(os.path.relpath(path, root))
    if args.check:
        if missing:
            print("HEADER EKSIK:", *missing, sep="\n  ")
            sys.exit(1)
        print("tum .py dosyalarinda header var")
    else:
        print(f"{len(added)} dosyaya header eklendi" if added else "eklenecek dosya yok (hepsi tamam)")
        for p in added:
            print("  +", p)


if __name__ == "__main__":
    main()

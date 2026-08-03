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
check_notebooks.py - Notebook statik kontrol betiği (Hücreler-arası bağımlılık ve yol denetleyicisi)

Dört Statik Kontrol:
  K1 - Hücreler-arası ad çözümleme (Cross-cell symbol resolution & NameError tespiti)
  K2 - stdout kazıma tespiti (subprocess + regex on stdout capture)
  K3 - Kimlik satırı eksikliği (İlk kod hücresinde § kimliği print kontrolü)
  K4 - Sabit yol / bayat referans (Repoda olmayan veya bayat dosya yolları)

Çalıştırma:
  python review_scripts/check_notebooks.py [notebooks_dir_or_file] [--strict]
"""

import argparse
import ast
import fnmatch
import glob
import json
import os
import sys

# Windows konsol UTF-8 çıktı desteği
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Python Dahili (Builtin) ve Standart Jupyter/IPython Sembolleri
BUILTINS = set(dir(__builtins__)) | {
    "get_ipython", "display", "In", "Out", "_", "__", "exit", "quit", "help",
    "open", "print", "range", "len", "int", "str", "float", "bool", "list",
    "dict", "set", "tuple", "type", "isinstance", "issubclass", "max", "min",
    "sum", "abs", "any", "all", "enumerate", "zip", "map", "filter", "sorted",
    "reversed", "repr", "super", "dir", "vars", "getattr", "setattr", "hasattr",
    "delattr", "property", "staticmethod", "classmethod", "Exception",
    "BaseException", "ValueError", "TypeError", "RuntimeError", "KeyError",
    "IndexError", "AttributeError", "FileNotFoundError", "ImportError",
    "SyntaxError", "AssertionError", "StopIteration", "True", "False", "None",
    "Ellipsis", "NotImplemented", "__name__", "__doc__", "__file__", "__spec__"
}


def load_gitignore_patterns(repo_root):
    """Repo kökündeki .gitignore kalıplarını okur."""
    gitignore_path = os.path.join(repo_root, ".gitignore")
    patterns = []
    if os.path.exists(gitignore_path):
        with open(gitignore_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    patterns.append(line)
    return patterns


def matches_gitignore(filepath, patterns):
    """Dosya yolunun .gitignore kalıplarından birine uyup uymadığını kontrol eder."""
    norm_path = filepath.replace("\\", "/").lstrip("/")
    basename = os.path.basename(filepath)
    for pat in patterns:
        pat_norm = pat.replace("\\", "/").rstrip("/")
        if fnmatch.fnmatch(norm_path, pat_norm) or fnmatch.fnmatch(basename, pat_norm):
            return True
        if pat_norm.endswith("/*") and fnmatch.fnmatch(norm_path, pat_norm):
            return True
        if fnmatch.fnmatch(norm_path, f"*/{pat_norm}") or fnmatch.fnmatch(norm_path, f"{pat_norm}/*"):
            return True
        if "/" not in pat_norm and fnmatch.fnmatch(basename, pat_norm):
            return True
    return False


def clean_cell_code(source_lines):
    """IPython magic/shell komutlarını (%, !) kaldırarak saf Python koduna dönüştürür."""
    cleaned = []
    for line in source_lines:
        stripped = line.lstrip()
        if stripped.startswith("%") or stripped.startswith("!"):
            cleaned.append("# magic: " + line)
        else:
            cleaned.append(line)
    return "".join(cleaned)


class SymbolExtractor(ast.NodeVisitor):
    """Hücre içerisindeki tanımlanan (store/import/def) ve yüklenen (load) sembolleri çıkarır."""

    def __init__(self):
        self.defined = set()
        self.module_loaded_names = []  # (name, lineno) at module level
        self.func_loaded_names = []    # (name, lineno) inside function bodies
        self.wildcard_imports = []
        self.scope_stack = [set()]     # local scope stack
        self.string_literals = []      # (str_val, lineno)

    def current_scope(self):
        return self.scope_stack[-1]

    def is_in_scope(self, name):
        for s in reversed(self.scope_stack):
            if name in s:
                return True
        return False

    def in_func_scope(self):
        return len(self.scope_stack) > 1

    def visit_Import(self, node):
        for alias in node.names:
            name_to_add = alias.asname if alias.asname else alias.name.split(".")[0]
            self.defined.add(name_to_add)
            self.current_scope().add(name_to_add)
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        for alias in node.names:
            if alias.name == "*":
                self.wildcard_imports.append((node.module or "", node.lineno))
            else:
                name_to_add = alias.asname if alias.asname else alias.name
                self.defined.add(name_to_add)
                self.current_scope().add(name_to_add)
        self.generic_visit(node)

    def visit_FunctionDef(self, node):
        self.defined.add(node.name)
        self.current_scope().add(node.name)
        for dec in node.decorator_list:
            self.visit(dec)

        func_scope = set()
        args_obj = node.args
        all_args = args_obj.posonlyargs + args_obj.args + args_obj.kwonlyargs
        for arg in all_args:
            func_scope.add(arg.arg)
        if args_obj.vararg:
            func_scope.add(args_obj.vararg.arg)
        if args_obj.kwarg:
            func_scope.add(args_obj.kwarg.arg)

        # Collect local assignments in function body
        for item in ast.walk(node):
            if isinstance(item, ast.Name) and isinstance(item.ctx, ast.Store):
                func_scope.add(item.id)
            elif isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                func_scope.add(item.name)

        self.scope_stack.append(func_scope)
        for stmt in node.body:
            self.visit(stmt)
        self.scope_stack.pop()

    def visit_AsyncFunctionDef(self, node):
        self.visit_FunctionDef(node)

    def visit_ClassDef(self, node):
        self.defined.add(node.name)
        self.current_scope().add(node.name)
        for dec in node.decorator_list:
            self.visit(dec)
        for base in node.bases:
            self.visit(base)
        class_scope = set()
        for item in ast.walk(node):
            if isinstance(item, ast.Name) and isinstance(item.ctx, ast.Store):
                class_scope.add(item.id)
            elif isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                class_scope.add(item.name)
        self.scope_stack.append(class_scope)
        for stmt in node.body:
            self.visit(stmt)
        self.scope_stack.pop()

    def visit_Lambda(self, node):
        lambda_scope = set()
        args_obj = node.args
        all_args = args_obj.posonlyargs + args_obj.args + args_obj.kwonlyargs
        for arg in all_args:
            lambda_scope.add(arg.arg)
        if args_obj.vararg:
            lambda_scope.add(args_obj.vararg.arg)
        if args_obj.kwarg:
            lambda_scope.add(args_obj.kwarg.arg)
        self.scope_stack.append(lambda_scope)
        self.visit(node.body)
        self.scope_stack.pop()

    def visit_comprehension_helper(self, node):
        comp_scope = set()
        for gen in node.generators:
            for n in ast.walk(gen.target):
                if isinstance(n, ast.Name):
                    comp_scope.add(n.id)
        self.scope_stack.append(comp_scope)
        self.generic_visit(node)
        self.scope_stack.pop()

    def visit_ListComp(self, node):
        self.visit_comprehension_helper(node)

    def visit_SetComp(self, node):
        self.visit_comprehension_helper(node)

    def visit_DictComp(self, node):
        self.visit_comprehension_helper(node)

    def visit_GeneratorExp(self, node):
        self.visit_comprehension_helper(node)

    def visit_Assign(self, node):
        for target in node.targets:
            for n in ast.walk(target):
                if isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Store, ast.Param)):
                    self.defined.add(n.id)
                    self.current_scope().add(n.id)
        self.generic_visit(node)

    def visit_AugAssign(self, node):
        if isinstance(node.target, ast.Name):
            self.defined.add(node.target.id)
            self.current_scope().add(node.target.id)
        self.generic_visit(node)

    def visit_AnnAssign(self, node):
        if isinstance(node.target, ast.Name):
            self.defined.add(node.target.id)
            self.current_scope().add(node.target.id)
        self.generic_visit(node)

    def visit_For(self, node):
        for n in ast.walk(node.target):
            if isinstance(n, ast.Name):
                self.defined.add(n.id)
                self.current_scope().add(n.id)
        self.generic_visit(node)

    def visit_AsyncFor(self, node):
        self.visit_For(node)

    def visit_With(self, node):
        for item in node.items:
            if item.optional_vars:
                for n in ast.walk(item.optional_vars):
                    if isinstance(n, ast.Name):
                        self.defined.add(n.id)
                        self.current_scope().add(n.id)
        self.generic_visit(node)

    def visit_ExceptHandler(self, node):
        if node.name:
            self.defined.add(node.name)
            self.current_scope().add(node.name)
        self.generic_visit(node)

    def visit_Name(self, node):
        if isinstance(node.ctx, ast.Load):
            if not self.is_in_scope(node.id):
                if self.in_func_scope():
                    self.func_loaded_names.append((node.id, node.lineno))
                else:
                    self.module_loaded_names.append((node.id, node.lineno))

    def visit_Constant(self, node):
        if isinstance(node.value, str):
            self.string_literals.append((node.value, node.lineno))

    def visit_Str(self, node):
        self.string_literals.append((node.s, node.lineno))


def is_valid_file_path_candidate(str_val):
    """Metin dizesinin bir dosya yolu adayı olup olmadığını filtreler."""
    s = str_val.strip()
    if not s or len(s) < 3 or len(s) > 250 or "\n" in s:
        return False
    ignore_prefixes = ("http://", "https://", "rtsp://", "cuda:", "cpu", "latin-1", "utf-8", "nat/token", "->", ">>>", "==")
    if any(s.startswith(p) for p in ignore_prefixes) or any(p in s for p in ["->", ">>>", "nat/token", "esik:", "YUKLEME"]):
        return False

    valid_exts = (".json", ".pt", ".py", ".txt", ".csv", ".md", ".ipynb", ".tex", ".bin", ".safetensors", ".pth", ".ckpt")
    has_ext = any(s.endswith(ext) for ext in valid_exts)
    has_path_prefix = any(s.startswith(prefix) for prefix in ["checkpoints/", "notebooks/", "review_scripts/", "hf_upload/", "docs/", "/content", "/kaggle", "dist/", "./", "../"])

    return has_ext or has_path_prefix


def analyze_notebook(nb_path, gitignore_patterns):
    """Tek bir notebook dosyasını statik olarak analiz eder."""
    try:
        with open(nb_path, "r", encoding="utf-8", errors="ignore") as f:
            nb_data = json.load(f)
    except Exception as e:
        return {
            "path": nb_path,
            "error": f"JSON okuma hatası: {e}",
            "k1_errors": [],
            "k2_warnings": [],
            "k3_warnings": [],
            "k4_warnings": [],
            "k4_info": [],
            "k4_stale_paths_set": set(),
        }

    cells = nb_data.get("cells", [])
    code_cells = [c for c in cells if c.get("cell_type") == "code"]

    defined_symbols = set(BUILTINS)
    wildcard_imported = False

    k1_errors = []
    k2_warnings = []
    k3_warnings = []
    k4_warnings = []
    k4_info = []
    k4_stale_paths_set = set()

    # --- K3: Kimlik Satırı Eksikliği Kontrolü ---
    if code_cells:
        first_code_src = "".join(code_cells[0].get("source", []))
        has_id_print = False
        for line in first_code_src.splitlines():
            line_str = line.strip()
            if line_str.startswith("print(") and any(kw in line_str for kw in ["§", "section", "notebook", "NOTEBOOK", "RUN", "DIR", "CHECKPOINT", "HFP"]):
                has_id_print = True
                break
        if not has_id_print:
            k3_warnings.append(
                f"[K3 UYARI] Hücre #1 kimlik satırı eksik (hangi § ve çıktı klasörünü bastığı doğrulanamadı)."
            )

    # --- K1, K2, K4 Hücre Taraması ---
    for cell_idx, cell in enumerate(code_cells, start=1):
        raw_source = cell.get("source", [])
        if isinstance(raw_source, list):
            raw_code = "".join(raw_source)
        else:
            raw_code = str(raw_source)

        if not raw_code.strip():
            continue

        # --- K2: stdout Kazıma Tespiti ---
        has_subproc = any(kw in raw_code for kw in ["subprocess.run", "subprocess.Popen", "check_output", "capture_output", "stdout=subprocess"])
        has_regex = any(kw in raw_code for kw in ["re.compile", "re.search", "re.findall", "re.match"])
        if has_subproc and has_regex:
            k2_warnings.append(
                f"[K2 UYARI] Hücre #{cell_idx}: subprocess çıktı kazıma (re + subprocess) tespiti (§29b riski)."
            )

        clean_code = clean_cell_code(raw_code.splitlines(keepends=True))
        try:
            tree = ast.parse(clean_code)
        except SyntaxError as se:
            k1_errors.append(
                f"[K1 HATA] Hücre #{cell_idx}:{se.lineno} Sözdizimi hatası (SyntaxError): {se.msg}"
            )
            continue

        extractor = SymbolExtractor()
        extractor.visit(tree)

        if extractor.wildcard_imports:
            wildcard_imported = True
            for mod, lineno in extractor.wildcard_imports:
                k1_errors.append(
                    f"[K1 UYARI] Hücre #{cell_idx}:{lineno} Yıldızlı import ('from {mod} import *') ad çözümlemeyi engelliyor."
                )

        # Modül düzeyindeki yüklenen adları kontrol et
        for name, lineno in extractor.module_loaded_names:
            if name not in defined_symbols and not wildcard_imported:
                k1_errors.append(
                    f"[K1 HATA] Hücre #{cell_idx}:{lineno} Tanımsız ad kullanılıyor (NameError adayı): '{name}'"
                )

        # Hücredeki tüm yeni tanımlanan sembolleri ekle
        defined_symbols.update(extractor.defined)

        # Fonksiyon gövdesi içindeki serbest adları ertelenmiş kontrol et
        for name, lineno in extractor.func_loaded_names:
            if name not in defined_symbols and not wildcard_imported:
                k1_errors.append(
                    f"[K1 HATA] Hücre #{cell_idx}:{lineno} Fonksiyon içinde tanımsız ad kullanılıyor (NameError adayı): '{name}'"
                )

        # --- K4: Sabit Yol / Bayat Referans Kontrolü ---
        for str_val, lineno in extractor.string_literals:
            if is_valid_file_path_candidate(str_val):
                abs_candidate = os.path.join(REPO_ROOT, str_val)
                if not os.path.exists(abs_candidate) and not os.path.exists(str_val):
                    if matches_gitignore(str_val, gitignore_patterns):
                        k4_info.append(
                            f"[K4 İNFO] Hücre #{cell_idx}:{lineno} Görülemeyen/checkpoint yolu (.gitignore kapsamı): '{str_val}'"
                        )
                    else:
                        k4_warnings.append(
                            f"[K4 UYARI] Hücre #{cell_idx}:{lineno} Bayat/eksik dosya yolu: '{str_val}'"
                        )
                        k4_stale_paths_set.add(str_val)

    return {
        "path": nb_path,
        "error": None,
        "k1_errors": k1_errors,
        "k2_warnings": k2_warnings,
        "k3_warnings": k3_warnings,
        "k4_warnings": k4_warnings,
        "k4_info": k4_info,
        "k4_stale_paths_set": k4_stale_paths_set,
    }


def main():
    parser = argparse.ArgumentParser(description="HFP Notebook Statik Denetleyici (check_notebooks.py)")
    parser.add_argument("target", nargs="?", default="notebooks", help="Denetlenecek notebook dizini veya tekil notebook yolu (varsayılan: notebooks)")
    parser.add_argument("--strict", action="store_true", help="Hata (K1) varsa çıkış kodunu 1 yapar")
    args = parser.parse_args()

    gitignore_patterns = load_gitignore_patterns(REPO_ROOT)

    if os.path.isfile(args.target):
        nb_files = [args.target]
    elif os.path.isdir(args.target):
        nb_files = sorted(glob.glob(os.path.join(args.target, "*.ipynb")))
    else:
        print(f"[HATA] Hedef '{args.target}' bulunamadı.")
        sys.exit(1)

    print("=" * 85)
    print(f"HFP NOTEBOOK STATİK KONTROLÜ (check_notebooks.py)")
    print(f"Taranan Notebook Sayısı: {len(nb_files)}")
    print("=" * 85)

    total_k1 = 0
    total_k2 = 0
    total_k3 = 0
    total_k4_warn = 0
    total_k4_info = 0
    all_unique_stale_paths = set()
    nb_with_errors = 0

    for nb_path in nb_files:
        rel_path = os.path.relpath(nb_path, REPO_ROOT)
        res = analyze_notebook(nb_path, gitignore_patterns)

        if res["error"]:
            print(f"\n--- [{rel_path}] ---")
            print(f"  [SİSTEM HATASI] {res['error']}")
            nb_with_errors += 1
            continue

        n_k1 = len(res["k1_errors"])
        n_k2 = len(res["k2_warnings"])
        n_k3 = len(res["k3_warnings"])
        n_k4_w = len(res["k4_warnings"])
        n_k4_i = len(res["k4_info"])

        total_k1 += n_k1
        total_k2 += n_k2
        total_k3 += n_k3
        total_k4_warn += n_k4_w
        total_k4_info += n_k4_i
        all_unique_stale_paths.update(res["k4_stale_paths_set"])

        if n_k1 > 0:
            nb_with_errors += 1

        if n_k1 > 0 or n_k2 > 0 or n_k3 > 0 or n_k4_w > 0:
            print(f"\n--- [{rel_path}] ---")
            for msg in res["k1_errors"]:
                print(f"  {msg}")
            for msg in res["k2_warnings"]:
                print(f"  {msg}")
            for msg in res["k3_warnings"]:
                print(f"  {msg}")
            for msg in res["k4_warnings"]:
                print(f"  {msg}")

    total_all_warning_lines = total_k1 + total_k2 + total_k3 + total_k4_warn + total_k4_info

    print("\n" + "=" * 85)
    print("STATİK DENETİM ÖZET TABLOSU")
    print("=" * 85)
    print(f"Toplam Notebook                      : {len(nb_files)}")
    print(f"Hatalı (K1) Notebook                 : {nb_with_errors}")
    print(f"K1 (NameError / Hata)                : {total_k1}")
    print(f"K2 (stdout Kazıma Uyarısı)            : {total_k2}")
    print(f"K3 (Kimlik Eksik Uyarısı)             : {total_k3}")
    print(f"K4 (Bayat Yol Uyarısı - Satır Bazlı) : {total_k4_warn}")
    print(f"K4 (Bayat Yol Uyarısı - Tekil Yol)   : {len(all_unique_stale_paths)}")
    print(f"K4 (.gitignore Checkpoint Bilgisi)   : {total_k4_info}")
    print(f"Toplam Uyarı Mesaj Satırı (Tüm Kategoriler): {total_all_warning_lines}")
    print("-" * 85)

    if args.strict and total_k1 > 0:
        print("[SONUÇ] STRICT MOD: K1 Hataları bulundu, çıkış kodu 1.")
        sys.exit(1)
    else:
        print("[SONUÇ] Statik denetim tamamlandı.")


if __name__ == "__main__":
    main()

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
reproduce.py - HFP Results Tekrarlanabilirlik / Yol Doğrulayıcı.

`RESULTS.md` belgesindeki KOŞULABİLİR bölümlerin üretim yollarının hızlı parametrelerle
çalıştırılabilir olduğunu (path/command health) denetler.

Not: Bu bir YOL kontrolüdür. Metrik/başarı eşikleri doğrulanmaz, yalnızca komutun hatasız
sonlandığı (exit code 0) kontrol edilir.

Kullanım:
    python reproduce.py                 # Tüm KOŞULABİLİR yolları çalıştırır
    python reproduce.py --section 3     # Sadece §3 bölümünü çalıştırır
    python reproduce.py --list          # Tüm bölümleri ve komutlarını listeler
"""

import argparse
import os
import subprocess
import sys
import time

REPO_ROOT = os.path.abspath(os.path.dirname(__file__))

# 18 KOŞULABİLİR bölüm tanımı
REPRODUCE_SECTIONS = [
    {
        "sec": 1,
        "title": "Methodological finding: supervision density gates learnability",
        "cmd": [sys.executable, "review_scripts/dense_retention.py", "exp", "additive", "1e-3", "0", "1.0"],
        "env": {},
    },
    {
        "sec": 2,
        "title": "Retention-law and write-rule comparison (3 seeds, ctx 160)",
        "cmd": [sys.executable, "review_scripts/dense_retention.py", "cubic_flux", "additive", "1e-3", "0", "1.0"],
        "env": {},
    },
    {
        "sec": 3,
        "title": "Length generalization (3 seeds) — main positive result",
        "cmd": [sys.executable, "review_scripts/length_gen.py", "train", "0", "1.0"],
        "env": {},
    },
    {
        "sec": 4,
        "title": "The memory is interference-limited, not decay-limited",
        "cmd": [sys.executable, "review_scripts/interference_eval.py", "0"],
        "env": {},
        "depends_on": 3,  # Needs lg_0_final.pt from §3
    },
    {
        "sec": 5,
        "title": "Capacity axis (DPFP feature map) — first clear mechanism win",
        "cmd": [sys.executable, "review_scripts/length_gen.py", "train", "0", "1.0"],
        "env": {"LG_VARIANT": "dpfp"},
    },
    {
        "sec": 6,
        "title": "cubic_flux long-horizon advantage (Validated)",
        "cmd": [
            sys.executable,
            "run_experiment.py",
            "--task",
            "retention",
            "--steps",
            "10",
            "--context",
            "96",
            "--max_gap",
            "64",
            "--local_window",
            "16",
            "--decay_mode",
            "cubic_flux",
        ],
        "env": {},
    },
    {
        "sec": 7,
        "title": "Initial Language Modeling Viability",
        "cmd": [
            sys.executable,
            "run_experiment.py",
            "--task",
            "lm",
            "--steps",
            "10",
            "--seq",
            "128",
            "--decay_mode",
            "cubic_flux",
        ],
        "env": {},
    },
    {
        "sec": 11,
        "title": "Training-length cliff applies to LM as well",
        "cmd": [sys.executable, "review_scripts/length_gen.py", "train", "0", "1.0"],
        "env": {},
    },
    {
        "sec": 13,
        "title": "Write-rule decision at long evaluation lengths",
        "cmd": [sys.executable, "review_scripts/hard_retention.py", "exp", "1e-3", "0", "1.0"],
        "env": {"HR_WRITE": "additive", "HR_STEPS": "10"},
    },
    {
        "sec": 14,
        "title": "Metric-artifact disclosure and length-degradation diagnosis",
        "cmd": [sys.executable, "review_scripts/hard_retention.py", "exp", "1e-3", "0", "1.0"],
        "env": {"HR_STEPS": "10"},
    },
    {
        "sec": 16,
        "title": "K1 gate, clean re-run: GLA family baseline v2",
        "cmd": [sys.executable, "review_scripts/baseline_compare.py", "exp", "0", "1.0"],
        "env": {},
    },
    {
        "sec": 17,
        "title": "Görev C — lifetime retention (cubic's natural-habitat test)",
        "cmd": [sys.executable, "review_scripts/lifetime_retention.py", "exp", "0", "1.0"],
        "env": {"LT_STEPS": "10", "LT_TRIALS": "2", "LT_GAPS": "64"},
    },
    {
        "sec": 19,
        "title": "Görev E — carry curriculum",
        "cmd": [sys.executable, "review_scripts/carry_curriculum.py", "exp", "0", "1.0"],
        "env": {"CC_STEPS": "10", "CC_TRIALS": "2", "CC_GAPS": "64"},
    },
    {
        "sec": 21,
        "title": "Görev G — TBPTT intervention",
        "cmd": [sys.executable, "review_scripts/carry_curriculum.py", "exp", "0", "1.0"],
        "env": {"CC_BPTT": "1", "CC_STEPS": "10", "CC_TRIALS": "2", "CC_GAPS": "64"},
    },
    {
        "sec": 26,
        "title": "Cubic in its native regime — extrapolation test",
        "cmd": [sys.executable, "review_scripts/cubic_stabilize.py", "exp", "0", "1.0"],
        "env": {},
    },
    {
        "sec": 27,
        "title": "Improving cubic: the eta sweep",
        "cmd": [sys.executable, "review_scripts/cubic_stabilize.py", "cubic_flux_chunked", "0", "1.0"],
        "env": {"HFP_ETA_LOG_MIN": "-6.0"},
    },
    {
        "sec": 28,
        "title": "Cubic, decisive test",
        "cmd": [sys.executable, "review_scripts/cubic_longhorizon.py", "exp", "elu", "1e-3", "0", "1.0"],
        "env": {},
    },
    {
        "sec": 29,
        "title": "parallel_cubic — removing the sequential z-scan",
        "cmd": [sys.executable, "review_scripts/verify_claims.py"],
        "env": {},
    },
]


def list_sections():
    print("=" * 80)
    print(f"{'§':<4} | {'Başlık':<45} | {'Komut'}")
    print("=" * 80)
    for s in REPRODUCE_SECTIONS:
        cmd_str = " ".join([os.path.basename(s["cmd"][0])] + s["cmd"][1:])
        if s["env"]:
            env_str = " ".join([f"{k}={v}" for k, v in s["env"].items()])
            cmd_str = f"{env_str} {cmd_str}"
        print(f"§{s['sec']:<3} | {s['title'][:45]:<45} | {cmd_str}")
    print("-" * 80)


def run_section(sec_dict: dict):
    sec_num = sec_dict["sec"]
    title = sec_dict["title"]
    cmd = sec_dict["cmd"]

    env = os.environ.copy()
    env["PYTHONPATH"] = REPO_ROOT
    env["PYTHONUNBUFFERED"] = "1"
    for k, v in sec_dict["env"].items():
        env[k] = str(v)

    cmd_display = " ".join([os.path.basename(cmd[0])] + cmd[1:])
    if sec_dict["env"]:
        env_display = " ".join([f"{k}={v}" for k, v in sec_dict["env"].items()])
        cmd_display = f"{env_display} {cmd_display}"

    t0 = time.time()
    try:
        proc = subprocess.run(
            cmd,
            cwd=REPO_ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=120,
        )
        duration = time.time() - t0
        status = "PASS" if proc.returncode == 0 else "FAIL"
        return {
            "sec": sec_num,
            "title": title,
            "cmd": cmd_display,
            "status": status,
            "exit_code": proc.returncode,
            "time": duration,
            "stderr": proc.stderr if proc.returncode != 0 else "",
        }
    except Exception as e:
        duration = time.time() - t0
        return {
            "sec": sec_num,
            "title": title,
            "cmd": cmd_display,
            "status": "ERROR",
            "exit_code": -1,
            "time": duration,
            "stderr": str(e),
        }


def main():
    parser = argparse.ArgumentParser(description="HFP Results Tekrarlanabilirlik / Yol Doğrulayıcı")
    parser.add_argument("--section", type=int, default=None, help="Yalnızca belirli bir bölümü koşturur (ör. --section 3)")
    parser.add_argument("--list", action="store_true", help="Tüm KOŞULABİLİR bölümleri ve komutları listeler")
    parser.add_argument("--heavy", action="store_true", help="120sn+ zamanaşımına uğrayan ağır CPU bölümlerini (§6, §26, §27) de çalıştırır")
    args = parser.parse_args()

    if args.list:
        list_sections()
        return

    targets = REPRODUCE_SECTIONS
    if not args.heavy and args.section is None:
        # Ağır CPU bölümlerini varsayılan koşudan çıkar
        targets = [s for s in REPRODUCE_SECTIONS if s["sec"] not in (6, 26, 27)]

    if args.section is not None:
        targets = [s for s in REPRODUCE_SECTIONS if s["sec"] == args.section]
        if not targets:
            print(f"[HATA] Section §{args.section} bulunamadı veya KOŞULABİLİR sınıfında değil.", flush=True)
            sys.exit(1)

    print("=" * 85, flush=True)
    print(f"HFP TEKRARLANABİLİRLİK VE YOL SAĞLIĞI DENETİMİ (reproduce.py)", flush=True)
    print(f"Koşturulan Bölüm Sayısı: {len(targets)}", flush=True)
    print("=" * 85, flush=True)

    results = []
    total_t0 = time.time()
    for sdict in targets:
        # Dependency check (e.g. §4 needs §3 output)
        if "depends_on" in sdict and args.section == sdict["sec"]:
            dep_sec = sdict["depends_on"]
            dep_dict = next((x for x in REPRODUCE_SECTIONS if x["sec"] == dep_sec), None)
            if dep_dict:
                print(f"  [ÖN KOŞUL] §{sec_num} için §{dep_sec} bağımlılığı koşturuluyor...", flush=True)
                run_section(dep_dict)

        res = run_section(sdict)
        results.append(res)
        status_str = f"[{res['status']}]" if res["status"] == "PASS" else f"[{res['status']} code={res['exit_code']}]"
        print(f"  §{res['sec']:<2} | {res['title'][:40]:<40} | {status_str:<12} | {res['time']:.2f}s", flush=True)

    total_duration = time.time() - total_t0

    print("\n" + "=" * 85, flush=True)
    print("SONUÇ TABLOSU", flush=True)
    print("=" * 85, flush=True)
    print(f"| {'§':<3} | {'Bölüm Başlığı':<42} | {'Durum':<8} | {'Kod':<4} | {'Süre (sn)':<9} |", flush=True)
    print("|---|---|---|---|---|", flush=True)
    pass_count = 0
    fail_count = 0
    for r in results:
        if r["status"] == "PASS":
            pass_count += 1
        else:
            fail_count += 1
        print(f"| §{r['sec']:<2} | {r['title'][:42]:<42} | {r['status']:<8} | {r['exit_code']:<4} | {r['time']:9.2f} |", flush=True)

    print("-" * 85, flush=True)
    print(f"Toplam Süre: {total_duration:.2f} saniye | Başarılı: {pass_count}/{len(results)} | Başarısız: {fail_count}/{len(results)}", flush=True)
    print("-" * 85, flush=True)

    failures = [r for r in results if r["status"] != "PASS"]
    if failures:
        print("\n" + "!" * 85)
        print("KIRIK YOL RAPORU (Sınıf 2 Değerlendirmesi İçin):")
        for f in failures:
            print(f"\n--- [§{f['sec']} {f['title']}] ---")
            print(f"Komut: {f['cmd']}")
            print(f"Hata Çıktısı:\n{f['stderr'][:500]}")
        print("!" * 85)


if __name__ == "__main__":
    main()

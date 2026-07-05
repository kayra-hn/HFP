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

"""[PURE-MEM] Saf-bellek ablasyonu: bilgi gercekten M/z'den mi akiyor?

Bagimsiz incelemenin (INCELEME_RAPORU.md §6) uyarisi: chunked cikarimda attention,
pencereye EK olarak ring buffer'daki son max_short_len (default 32) tokeni gorur;
bu yuzden "chunk sinirini bilgi yalnizca M/z ile gecer" ifadesi tam dogru degil.
Bu script ring buffer'i devre disi birakip (short_len=max_short_len=1) ayni dense
gorevi kosarak M/z yolunu IZOLE eder ve tam-ring-buffer ile A/B kiyaslar.

Beklenti: gap>=16 (attention alicilarinin otesi) kovalarinda dogruluk ring
buffer'siz de KORUNUYORSA, o gap'lerdeki recall gercekten M/z belleginden gelir;
DUSUYORSA, onceki sonuclarin bir kismini ring buffer tasimistir ve iddia
duzeltilmelidir. Ikisi de tek-forward egitimiyle olculur (adil).

Kullanim: python review_scripts/pure_memory_ablation.py <seed> [budget_per_arm]
    python review_scripts/pure_memory_ablation.py 0 60
"""
import os, random, sys, time
import numpy as np, torch
CKDIR = os.environ.get("HFP_CKPT_DIR", "checkpoints")
os.makedirs(CKDIR, exist_ok=True)
from hfp.models.configuration_hfp import HFPConfig
from hfp.models.modeling_hfp import HFPForCausalLM

SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 0
BUDGET = float(sys.argv[2]) if len(sys.argv) > 2 else 60.0
DEV = "cuda" if torch.cuda.is_available() else "cpu"

KLO, KHI, VLO, VHI, FHI = 100, 130, 130, 160, 100
CTX, WIN, P = 160, 8, 8
ANS = 0
STEPS, LR = 600, 1e-3


def make_seq():
    toks = [random.randint(1, FHI - 1) for _ in range(CTX)]
    slots = random.sample(range(CTX // 2), 2 * P); random.shuffle(slots)
    keys = random.sample(range(KLO, KHI), P)
    lab = [-100] * CTX; meta = []
    for i in range(P):
        a, b = slots[2 * i], slots[2 * i + 1]
        if a > b: a, b = b, a
        wp, qp = 2 * a, 2 * b
        v = random.randint(VLO, VHI - 1)
        toks[wp] = keys[i]; toks[wp + 1] = v
        toks[qp] = keys[i]; toks[qp + 1] = ANS
        lab[qp + 1] = v
        meta.append((qp + 1 - (wp + 1), qp + 1, v))
    return toks, lab, meta


def bname(g):
    return "1-15" if g < 16 else "16-47" if g < 48 else "48-95" if g < 96 else "96+"


def run_arm(name, max_short):
    """max_short: ring buffer kapasitesi. 1 = ring buffer pratikte kapali (saf M/z)."""
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    cfg = HFPConfig(vocab_size=VHI + 4, hidden_size=64, num_hidden_layers=2,
                    num_attention_heads=2, intermediate_size=256, bulk_dim=32,
                    short_len=1 if max_short == 1 else 8, max_short_len=max_short,
                    max_position_embeddings=CTX + 8, local_window=WIN,
                    decay_mode="exp", rec_block=32, write_rule="additive")
    model = HFPForCausalLM(cfg).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=STEPS)
    ckpt = f"{CKDIR}/pma_{name}_{SEED}.pt"
    start = 0
    if os.path.exists(ckpt):
        st = torch.load(ckpt, weights_only=False)
        model.load_state_dict(st["m"]); opt.load_state_dict(st["o"]); sch.load_state_dict(st["s"])
        start = st["step"]; random.setstate(st["rng"])
        np.random.set_state(st["nprng"]); torch.set_rng_state(st["trng"])
    t0 = time.time(); model.train(); step = start
    while step < STEPS and time.time() - t0 < BUDGET:
        step += 1
        xs, ys = [], []
        for _ in range(16):
            t, l, _ = make_seq(); xs.append(t); ys.append(l)
        out = model(torch.tensor(xs, device=DEV), labels=torch.tensor(ys, device=DEV))
        assert torch.isfinite(out.loss)
        opt.zero_grad(set_to_none=True); out.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sch.step()
        if step % 100 == 0:
            print(f"[pma-{name}-{SEED}] step {step} loss {out.loss.item():.3f}", flush=True)
    if step < STEPS:
        torch.save(dict(m=model.state_dict(), o=opt.state_dict(), s=sch.state_dict(), step=step,
                        rng=random.getstate(), nprng=np.random.get_state(), trng=torch.get_rng_state()), ckpt)
        print(f"[pma-{name}-{SEED}] CKPT {step}/{STEPS} — tekrar calistirin"); sys.exit(0)

    model.eval()
    buckets = {"1-15": [0, 0], "16-47": [0, 0], "48-95": [0, 0], "96+": [0, 0]}
    with torch.no_grad():
        for _ in range(80):
            toks, _, meta = make_seq()
            logits = model(torch.tensor([toks], device=DEV)).logits[0]
            for g, ap, v in meta:
                b = buckets[bname(g)]; b[1] += 1
                b[0] += int(logits[ap - 1].argmax().item() == v)
    return {k: round(100.0 * c / max(1, n), 1) for k, (c, n) in buckets.items()}


if __name__ == "__main__":
    full = run_arm("full", max_short=32)     # varsayilan ring buffer
    pure = run_arm("pure", max_short=1)      # ring buffer kapali -> saf M/z
    print(f"\n[PURE-MEM s{SEED}] ring buffer=32 (full): {full}")
    print(f"[PURE-MEM s{SEED}] ring buffer=1  (pure): {pure}")
    print("[PURE-MEM] gap>=16 kovalarinda pure ~= full ise recall gercekten M/z'den.")
    with open(f"{CKDIR}/pure_memory_results.txt", "a") as f:
        f.write(f"s{SEED} full={full} pure={pure}\n")

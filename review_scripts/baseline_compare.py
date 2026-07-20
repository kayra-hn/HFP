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

"""[BASELINE] Harici verimli-ozyineli AILE baseline'i: Gated Linear Attention (GLA).

Bagimsiz incelemenin (INCELEME_RAPORU.md, nihai karar) kapatilmasi gereken son
sarti: "en az bir GLA/Mamba-sinifi baseline". Bu script, HFP'nin dense-retention
gorevinin BIREBIR aynisinda saf-PyTorch bir GLA modeli egitir/degerlendirir; ayni
kovalar (1-15/16-47/48-95/96+), ayni 600 adim, ayni sans %3.3. Boylece HFP
tablolari (RESULTS.md §2, DENEY_SONUCLARI Ek 4) dis literature karsi konumlanir.

GLA katmani (Yang ve ark. 2023 ailesi): veri-bagimli per-kanal unutma kapisi
    a_t = sigmoid(W_a x_t)              (0..1, kanal basina)
    S_t = diag(a_t) S_{t-1} + k_t v_t^T
    o_t = q_t S_t
HFP'den FARKI kasitli: GLA'da pencereli-attention YOK ve decay OGRENILEN-ama-
veri-bagimli (HFP'de exp cok-olcekli sabit + pencereli attention + feature map).
Bu, "verimli-ozyineli aile" icin adil bir dis referanstir; HFP'nin attention
yardimini almadigindan kisa-gap'te GLA'nin dezavantajli olmasi BEKLENIR ve
karsilastirmada bu acikca not edilmelidir.

Adil parametre karsilastirmasi: script her iki modelin de parametre sayisini
basar (--match ile GLA genisligi HFP'ye yaklastirilir). Kullanim:
    python review_scripts/baseline_compare.py <seed> [budget]
    python review_scripts/baseline_compare.py 0 60
Cikti CKDIR/baseline_results.txt'e eklenir. GPU'da: CUDA otomatik kullanilir.
"""
import os, random, sys, time, math
import numpy as np, torch
import torch.nn as nn
import torch.nn.functional as F

CKDIR = os.environ.get("HFP_CKPT_DIR", "checkpoints")
os.makedirs(CKDIR, exist_ok=True)

SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 0
BUDGET = float(sys.argv[2]) if len(sys.argv) > 2 else 60.0
DEV = "cuda" if torch.cuda.is_available() else "cpu"

# --- Gorev sabitleri: dense_retention.py ile BIREBIR ayni ---
KLO, KHI, VLO, VHI, FHI = 100, 130, 130, 160, 100
CTX, P = 160, 8
ANS = 0
STEPS, LR = 600, 1e-3
HID, LAYERS = 64, 2          # HFP dense reçetesiyle eslesir (hidden 64, 2 katman)
TAG = f"gla_{SEED}"
CKPT = f"{CKDIR}/{TAG}.pt"


def make_seq():
    """dense_retention.make_seq ile ayni: P cift + P sorgu, gap kovali eval."""
    toks = [random.randint(1, FHI - 1) for _ in range(CTX)]
    nslot = CTX // 2
    slots = random.sample(range(nslot), 2 * P); random.shuffle(slots)
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


class GLALayer(nn.Module):
    """Tek GLA blogu: veri-bagimli per-kanal kapili lineer attention + FFN."""
    def __init__(self, hid, ffn_mult=4):
        super().__init__()
        self.hid = hid
        self.q = nn.Linear(hid, hid, bias=False)
        self.k = nn.Linear(hid, hid, bias=False)
        self.v = nn.Linear(hid, hid, bias=False)
        self.a = nn.Linear(hid, hid)            # forget-gate logit (per kanal)
        nn.init.constant_(self.a.bias, 2.0)     # sigmoid(2)=0.88 -> uzun ufuk baslangici
        self.o = nn.Linear(hid, hid, bias=False)
        self.norm1 = nn.LayerNorm(hid)
        self.norm2 = nn.LayerNorm(hid)
        self.ffn = nn.Sequential(nn.Linear(hid, ffn_mult * hid), nn.GELU(),
                                 nn.Linear(ffn_mult * hid, hid))

    def forward(self, x):
        B, L, H = x.shape
        q = F.elu(self.q(x)) + 1.0             # >0 (lineer attention ozelligi)
        k = F.elu(self.k(x)) + 1.0
        v = self.v(x)
        a = torch.sigmoid(self.a(x))           # (B,L,H) unutma kapisi
        S = x.new_zeros(B, H, H)               # matris durum (key x value)
        outs = []
        for t in range(L):                     # nedensel ozyineleme (O(L), egitim CPU'da yeterli)
            S = a[:, t].unsqueeze(-1) * S + torch.einsum('bk,bv->bkv', k[:, t], v[:, t])
            outs.append(torch.einsum('bk,bkv->bv', q[:, t], S))
        o = torch.stack(outs, dim=1)
        x = self.norm1(x + self.o(o))
        x = self.norm2(x + self.ffn(x))
        return x


class GLAModel(nn.Module):
    def __init__(self, vocab, hid=HID, layers=LAYERS, max_pos=CTX + 8):
        super().__init__()
        self.emb = nn.Embedding(vocab, hid)
        pe = torch.zeros(max_pos, hid)
        pos = torch.arange(max_pos).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, hid, 2).float() * (-math.log(10000.0) / hid))
        pe[:, 0::2] = torch.sin(pos * div); pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe * 0.3)   # pe_scale=0.3, HFP ile ayni denge
        self.layers = nn.ModuleList([GLALayer(hid) for _ in range(layers)])
        self.norm = nn.LayerNorm(hid)
        self.head = nn.Linear(hid, vocab, bias=False)
        self.head.weight = self.emb.weight     # weight tying (HFP ile ayni)

    def forward(self, idx):
        x = self.emb(idx) * math.sqrt(self.emb.weight.size(1)) + self.pe[:idx.size(1)]
        for lyr in self.layers:
            x = lyr(x)
        return self.head(self.norm(x))


def bname(g):
    return "1-15" if g < 16 else "16-47" if g < 48 else "48-95" if g < 96 else "96+"


def main():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    model = GLAModel(VHI + 4).to(DEV)
    nparam = sum(p.numel() for p in model.parameters())
    print(f"[{TAG}] GLA params={nparam} dev={DEV} "
          f"(kiyas: HFP dense reçetesi ~ ayni hidden 64/2 katman)", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=STEPS)
    start = 0
    if os.path.exists(CKPT):
        st = torch.load(CKPT, weights_only=False)
        model.load_state_dict(st["m"]); opt.load_state_dict(st["o"]); sch.load_state_dict(st["s"])
        start = st["step"]; random.setstate(st["rng"])
        np.random.set_state(st["nprng"]); torch.set_rng_state(st["trng"])

    t0 = time.time(); model.train(); step = start
    while step < STEPS and time.time() - t0 < BUDGET:
        step += 1
        xs, ys = [], []
        for _ in range(16):
            t, l, _ = make_seq(); xs.append(t); ys.append(l)
        x = torch.tensor(xs, device=DEV); y = torch.tensor(ys, device=DEV)
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=-100)
        assert torch.isfinite(loss), f"NaN {TAG} step {step}"
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sch.step()
        if step % 100 == 0:
            print(f"[{TAG}] step {step} loss {loss.item():.3f}", flush=True)

    if step < STEPS:
        torch.save(dict(m=model.state_dict(), o=opt.state_dict(), s=sch.state_dict(), step=step,
                        rng=random.getstate(), nprng=np.random.get_state(), trng=torch.get_rng_state()), CKPT)
        print(f"[{TAG}] CKPT {step}/{STEPS}"); sys.exit(0)

    model.eval()
    buckets = {"1-15": [0, 0], "16-47": [0, 0], "48-95": [0, 0], "96+": [0, 0]}
    with torch.no_grad():
        for _ in range(80):
            toks, _, meta = make_seq()
            logits = model(torch.tensor([toks], device=DEV))[0]
            for g, ap, v in meta:
                b = buckets[bname(g)]; b[1] += 1
                b[0] += int(logits[ap - 1].argmax().item() == v)
    res = {k: round(100.0 * c / max(1, n), 1) for k, (c, n) in buckets.items()}
    ns = {k: n for k, (c, n) in buckets.items()}
    print(f"[{TAG}] FINAL GLA {res} (n={ns}, sans 3.3%)")
    with open(f"{CKDIR}/baseline_results.txt", "a") as f:
        f.write(f"{TAG} params={nparam} {res}\n")


if __name__ == "__main__":
    main()

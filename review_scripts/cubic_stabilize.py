# Hyper Flux Projection (HFP) — O(1)-memory causal language model
# Copyright (C) 2026 Kayrahan Yılmaz  —  AGPL-3.0
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License as published by the Free
# Software Foundation, either version 3 of the License, or (at your option) any
# later version. See <https://www.gnu.org/licenses/> for details.

"""[CUBIC-STAB] cubic_flux'un seed-kirilganligini duzeltme denemesi.

Gozlem (cubic_longhorizon, 5 seed): cubic+dpfp BIMODAL — iyi havzada (loss<1.7)
256+ kovasinda exp'i ezer (+27..+36); takildiginda (loss~2.1) sadece azicik gecer.
Yani sorun KAPASITE degil, iyi havzaya ULASMAK -> optimizasyon kararliligi.

Hipotez: retention parametreleri (log_eta, decay, beta_gate) bellek dinamigini
kontrol eder; erken buyuk LR onlari kotu bolgeye atiyor. Cozum: bu parametreleri
DAHA YAVAS ogren (ayri kucuk LR) + WARMUP. Boylece ag once yerlesir, sonra
retention ince ayar yapar -> iyi havza daha sik bulunur.

A/B: <base|stab> <seed> [budget]
  base : tek LR, cosine (cubic_longhorizon ile ayni)
  stab : retention paramlari LR*0.2 + 60-adim warmup
Fikstur: cubic_flux_chunked + dpfp (etkiyi gosteren konfig). Cikti: kac seed'de
iyi havza (loss<1.7) bulundu + 256+ sonucu -> stab base'i geciyor mu?
Kullanim: python review_scripts/cubic_stabilize.py stab 0 1500
"""
import os, sys, math, random, time
import numpy as np, torch
CKDIR = os.environ.get("HFP_CKPT_DIR", "checkpoints")
os.makedirs(CKDIR, exist_ok=True)
from hfp.models.configuration_hfp import HFPConfig
from hfp.models.modeling_hfp import HFPForCausalLM

MODE = sys.argv[1]                      # base | stab
SEED = int(sys.argv[2])
BUDGET = float(sys.argv[3]) if len(sys.argv) > 3 else 1500.0
DEV = "cuda" if torch.cuda.is_available() else "cpu"

RET, FMAP = "cubic_flux_chunked", "dpfp"
LR, STEPS = 1e-3, 600
RET_MULT = 0.2                          # stab: retention paramlari 5x yavas
WARMUP = 60
KLO, KHI, VLO, VHI, FHI = 100, 130, 130, 160, 100
WIN, P, ANS = 8, 8, 0
TRAIN_CTX = 160
EVAL_CTXS = (640, 1280)
TAG = f"stab_{MODE}_{SEED}"
RET_KEYS = ("log_eta", "decay", "beta_gate")


def make_seq(ctx):
    toks = [random.randint(1, FHI - 1) for _ in range(ctx)]
    slots = random.sample(range(ctx // 2), 2 * P); random.shuffle(slots)
    keys = random.sample(range(KLO, KHI), P)
    lab = [-100] * ctx; meta = []
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


def build(max_pos):
    cfg = HFPConfig(vocab_size=VHI + 4, hidden_size=64, num_hidden_layers=2,
                    num_attention_heads=2, intermediate_size=256, bulk_dim=32,
                    short_len=8, max_position_embeddings=max_pos, local_window=WIN,
                    decay_mode=RET, rec_block=32, write_rule="additive",
                    key_feature_map=FMAP, ffn_type="standard")
    return HFPForCausalLM(cfg)


def bname(g):
    return "<48" if g < 48 else "48-127" if g < 128 else "128-255" if g < 256 else "256+"


def main():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    model = build(TRAIN_CTX + 8).to(DEV)

    if MODE == "stab":
        ret_p, oth_p = [], []
        for n, p in model.named_parameters():
            (ret_p if any(k in n for k in RET_KEYS) else oth_p).append(p)
        opt = torch.optim.AdamW([{"params": oth_p, "lr": LR},
                                 {"params": ret_p, "lr": LR * RET_MULT}])
        print(f"[{TAG}] retention paramlari: {len(ret_p)} (LR*{RET_MULT}) + warmup {WARMUP}", flush=True)
        def lr_factor(step):
            if step < WARMUP:
                return (step + 1) / WARMUP
            prog = (step - WARMUP) / max(1, STEPS - WARMUP)
            return 0.5 * (1 + math.cos(math.pi * prog))
        sch = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lambda s: lr_factor(s))
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=LR)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=STEPS)

    t0 = time.time(); model.train(); last = 9.9
    for step in range(1, STEPS + 1):
        if time.time() - t0 > BUDGET:
            print(f"[{TAG}] BUTCE ASILDI step {step}"); break
        xs, ys = [], []
        for _ in range(16):
            t, l, _ = make_seq(TRAIN_CTX); xs.append(t); ys.append(l)
        out = model(torch.tensor(xs, device=DEV), labels=torch.tensor(ys, device=DEV))
        assert torch.isfinite(out.loss), f"NaN {TAG} step {step}"
        opt.zero_grad(set_to_none=True); out.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sch.step(); last = out.loss.item()
        if step % 200 == 0:
            print(f"[{TAG}] step {step} loss {last:.3f}", flush=True)

    # eval
    sd = model.state_dict()
    res1280 = None
    model.eval()
    for ctx in EVAL_CTXS:
        random.seed(3000 + SEED)
        m = build(ctx + 8).to(DEV)
        m.load_state_dict({k: v for k, v in sd.items() if "pos_encoder.pe" not in k}, strict=False)
        m.eval()
        buckets = {"<48": [0, 0], "48-127": [0, 0], "128-255": [0, 0], "256+": [0, 0]}
        with torch.no_grad():
            for _ in range(50):
                toks, _, meta = make_seq(ctx)
                logits = m(torch.tensor([toks], device=DEV)).logits[0]
                for g, ap, v in meta:
                    b = buckets[bname(g)]; b[1] += 1
                    b[0] += int(logits[ap - 1].argmax().item() == v)
        r = {k: round(100.0 * c / max(1, n), 1) for k, (c, n) in buckets.items()}
        print(f"[{TAG}] eval ctx={ctx}: {r}  (loss {last:.3f})", flush=True)
        if ctx == 1280:
            res1280 = r
    basin = "IYI-HAVZA" if last < 1.7 else "takildi"
    print(f"[{TAG}] SONUC loss={last:.3f} [{basin}] 256+={res1280['256+']}", flush=True)
    with open(f"{CKDIR}/cubic_stab_results.txt", "a") as f:
        f.write(f"{TAG} loss={last:.3f} basin={basin} 256+={res1280['256+']} full={res1280}\n")


if __name__ == "__main__":
    main()

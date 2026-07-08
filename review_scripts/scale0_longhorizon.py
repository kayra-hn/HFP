# Hyper Flux Projection (HFP) — Stage-0 scaled long-horizon test (cubic vs exp)
# Copyright (C) 2026 Kayrahan Yılmaz — AGPL-3.0
#
# [STAGE 0] Bulguyu OLCEKLE: daha buyuk model + cok daha uzun gap. Soru: plato
# avantaji (cubic > exp uzun-gap'te) buyuk modelde ve 512+ token mesafede de
# tutuyor -- hatta buyuyor -- mu? Teori: exp'in ustel ucurumu daha uzakta daha
# sert vurdugu icin cubic'in avantaji 512+'da 256+'dakinden BUYUK olmali.
# Sentetik dense-recall; TRAIN kisa (512), EVAL uzun (1024/2048) = train-short/infer-long.
# Kullanim: python review_scripts/scale0_longhorizon.py <exp|cubic_flux_chunked> <seed> [steps]
import os, random, sys, time
import numpy as np, torch
CKDIR = os.environ.get("HFP_CKPT_DIR", "checkpoints")
os.makedirs(CKDIR, exist_ok=True)
from hfp.models.configuration_hfp import HFPConfig
from hfp.models.modeling_hfp import HFPForCausalLM

RET  = sys.argv[1]                       # exp | cubic_flux_chunked
SEED = int(sys.argv[2])
STEPS = int(sys.argv[3]) if len(sys.argv) > 3 else 800
DEV = "cuda" if torch.cuda.is_available() else "cpu"

# --- olcek-0 (256+ testinden buyuk): buyuk model + uzun baglam + genis uzay ---
KLO, KHI, VLO, VHI, FHI = 200, 260, 260, 320, 200   # 60 anahtar, 60 deger (sans 1.7%)
WIN, P, ANS = 32, 12, 0
TRAIN_CTX, LR, BS = 512, 1e-3, 8
EVAL_CTXS = (1024, 2048)
HID, LAYERS, HEADS = 128, 3, 4           # hidden 64/2kat -> 128/3kat
TAG = f"s0_{RET}_{SEED}"


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
    cfg = HFPConfig(vocab_size=VHI + 4, hidden_size=HID, num_hidden_layers=LAYERS,
                    num_attention_heads=HEADS, intermediate_size=4 * HID, bulk_dim=64,
                    short_len=8, max_position_embeddings=max_pos, local_window=WIN,
                    decay_mode=RET, rec_block=32, write_rule="additive",
                    key_feature_map="dpfp", ffn_type="standard")
    return HFPForCausalLM(cfg).to(DEV)


def bname(g):
    return "<64" if g < 64 else "64-255" if g < 256 else "256-511" if g < 512 else "512+"


def main():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    m = build(TRAIN_CTX + 8)
    opt = torch.optim.AdamW(m.parameters(), lr=LR)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=STEPS)
    t0 = time.time(); m.train()
    for step in range(1, STEPS + 1):
        xs, ys = [], []
        for _ in range(BS):
            t, l, _ = make_seq(TRAIN_CTX); xs.append(t); ys.append(l)
        out = m(torch.tensor(xs, device=DEV), labels=torch.tensor(ys, device=DEV))
        assert torch.isfinite(out.loss), f"NaN {TAG} step {step}"
        opt.zero_grad(set_to_none=True); out.loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step(); sch.step()
        if step % 200 == 0:
            print(f"[{TAG}] step {step} loss {out.loss.item():.3f} ({time.time()-t0:.0f}s)", flush=True)
    sd = m.state_dict(); m.eval()
    for ctx in EVAL_CTXS:
        random.seed(9000 + SEED)
        me = build(ctx + 8)
        me.load_state_dict({k: v for k, v in sd.items() if "pos_encoder.pe" not in k}, strict=False)
        me.eval()
        buck = {"<64": [0, 0], "64-255": [0, 0], "256-511": [0, 0], "512+": [0, 0]}
        with torch.no_grad():
            for _ in range(30):
                toks, _, meta = make_seq(ctx)
                lo = me(torch.tensor([toks], device=DEV)).logits[0]
                for g, ap, v in meta:
                    b = buck[bname(g)]; b[1] += 1
                    b[0] += int(lo[ap - 1].argmax().item() == v)
        r = {k: round(100.0 * c / max(1, nn), 1) for k, (c, nn) in buck.items()}
        print(f"[{TAG}] eval ctx={ctx}: {r}", flush=True)
        with open(f"{CKDIR}/scale0_results.txt", "a") as fh:
            fh.write(f"{TAG} ctx{ctx} {r}\n")
    print(f"[{TAG}] BITTI", flush=True)


if __name__ == "__main__":
    main()

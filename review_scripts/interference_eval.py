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

"""[INTF] Girisim-yogunluk ayristirmasi: egitilmis lg modellerini (train@160,P=8)
ctx 640'ta P={8,16,24} ile degerlendir. Sabit gap'te acc P ile dusuyorsa
'uzunlukla artis = girisim azalmasi' aciklamasi dogrulanir.
Kullanim: python review_scripts/interference_eval.py <seed>"""
import os
CKDIR = os.environ.get("HFP_CKPT_DIR", "checkpoints")
os.makedirs(CKDIR, exist_ok=True)

import random, sys
import torch
import os as _os
VARIANT = _os.environ.get("LG_VARIANT", "additive")
from hfp.models.configuration_hfp import HFPConfig
from hfp.models.modeling_hfp import HFPForCausalLM

KLO, KHI, VLO, VHI, FHI = 100, 130, 130, 160, 100
WIN, ANS = 8, 0
SEED = int(sys.argv[1])
FINAL = f"{CKDIR}/lg_{VARIANT}_{SEED}_final.pt" if VARIANT != "additive" else f"{CKDIR}/lg_{SEED}_final.pt"


def make_seq(ctx, P):
    toks = [random.randint(1, FHI - 1) for _ in range(ctx)]
    slots = random.sample(range(ctx // 2), 2 * P); random.shuffle(slots)
    keys = random.sample(range(KLO, KHI), P)
    meta = []
    for i in range(P):
        a, b = slots[2 * i], slots[2 * i + 1]
        if a > b: a, b = b, a
        wp, qp = 2 * a, 2 * b
        v = random.randint(VLO, VHI - 1)
        toks[wp] = keys[i]; toks[wp + 1] = v
        toks[qp] = keys[i]; toks[qp + 1] = ANS
        meta.append((qp - wp, qp + 1, v))
    return toks, meta


sd = torch.load(FINAL, weights_only=True)
CTX = 640
model = None
for P in (8, 16, 24):
    random.seed(2000 + SEED)
    if model is None:
        cfg = HFPConfig(vocab_size=VHI + 4, hidden_size=64, num_hidden_layers=2,
                        num_attention_heads=2, intermediate_size=256, bulk_dim=32,
                        short_len=8, max_position_embeddings=CTX + 8, local_window=WIN,
                        decay_mode="exp", rec_block=32,
                    write_rule=("delta" if VARIANT == "delta" else "additive"),
                    key_feature_map=("dpfp" if VARIANT == "dpfp" else "elu"))
        model = HFPForCausalLM(cfg)
        model.load_state_dict({k: v for k, v in sd.items() if "pos_encoder.pe" not in k}, strict=False)
        model.eval()
    buckets = {}
    with torch.no_grad():
        for _ in range(40):
            toks, meta = make_seq(CTX, P)
            logits = model(torch.tensor([toks])).logits[0]
            for g, ap, v in meta:
                b = "<48" if g < 48 else "48-127" if g < 128 else "128-255" if g < 256 else "256+"
                c = buckets.setdefault(b, [0, 0]); c[1] += 1
                c[0] += int(logits[ap - 1].argmax().item() == v)
    res = {k: round(100.0 * c / max(1, n), 1) for k, (c, n) in sorted(buckets.items())}
    print(f"[intf {VARIANT} s{SEED}] ctx=640 P={P:2d}: {res}", flush=True)

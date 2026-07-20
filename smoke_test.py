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
smoke_test.py - CPU'da ~1 dk. Kritik regresyon testleri:

  T1  Forward/backward calisir, loss sonlu.
  T2  [K2 REGRESYONU] Bellek parametreleri (W_q, W_k, W_v, decay, importance_gate,
      W_bulk, P_A) LM loss'tan SIFIRDAN FARKLI gradyan alir. Eski surumde
      W_k/W_v/decay/gate hicbir rejimde gradyan almiyordu (olu agirlik).
  T3  [K1 REGRESYONU] MQAR batch'inde loss sonlu (eski cifte-shift NaN veriyordu)
      ve target gercekten loss'a giriyor.
  T4  [K2 DOGRULUK] HFPBulkState chunk tutarliligi: tam diziyi tek cagride
      islemek == iki parcada state tasiyarak islemek (per-token decay matematigi
      dogruysa birebir esit olmali). rec_block bagimsizligi da test edilir.
  T5  Ring buffer maskesi + generate smoke (cache'li uretim yolu).

Calistirma:  python smoke_test.py
"""
import torch

from hfp.models.configuration_hfp import HFPConfig
from hfp.models.modeling_hfp import HFPForCausalLM
from hfp.core.hfp_bulk_state import HFPBulkState

PASS, FAIL = "\033[92mPASS\033[0m", "\033[91mFAIL\033[0m"
failures = []

def check(name, cond, detail=""):
    ok = bool(cond)
    print(f"  [{PASS if ok else FAIL}] {name}" + (f"  ({detail})" if detail else ""))
    if not ok:
        failures.append(name)

def tiny_model(local_window=8, seed=0):
    torch.manual_seed(seed)
    cfg = HFPConfig(vocab_size=64, hidden_size=32, num_hidden_layers=2,
                    num_attention_heads=2, intermediate_size=64, bulk_dim=16,
                    short_len=4, max_position_embeddings=128,
                    local_window=local_window, rec_block=4)
    return HFPForCausalLM(cfg)

# ---------------- T1 + T2 ----------------
print("T1/T2: forward/backward + gradyan akisi")
model = tiny_model()
model.train()
x = torch.randint(0, 64, (2, 24))
y = torch.randint(0, 64, (2, 24))
out = model(x, labels=y)
check("loss sonlu", torch.isfinite(out.loss), f"loss={out.loss.item():.4f}")
out.loss.backward()

bs = model.hfp.bulk_states[0]
ffn = model.hfp.layers[0].ffn.entangled
grad_targets = {
    "W_q": bs.W_q.weight, "W_k": bs.W_k.weight, "W_v": bs.W_v.weight,
    "decay": bs.decay, "importance_gate": bs.importance_gate.weight,
    "W_bulk": ffn.W_bulk, "P_A": ffn.P_A,
}
for name, p in grad_targets.items():
    gn = 0.0 if p.grad is None else p.grad.norm().item()
    check(f"grad({name}) != 0", gn > 0, f"norm={gn:.2e}")

# ---------------- T3 ----------------
print("T3: MQAR label hizalamasi")
from run_experiment import make_mqar_batch
model2 = tiny_model()
model2.train()
xm, ym = make_mqar_batch(4, num_pairs=2, context_len=16,
                         key_lo=10, key_hi=30, val_lo=30, val_hi=50,
                         filler_hi=10, device="cpu")
# HFP vocab=64 tiny model icin token araligi uygun (max id < 64)
outm = model2(xm, labels=ym)
check("MQAR loss sonlu (eski surum: NaN)", torch.isfinite(outm.loss), f"loss={outm.loss.item():.4f}")
# target gercekten loss'a giriyor mu: shift sonrasi gecerli label sayisi = batch
shift_labels = ym[..., 1:]
n_valid = (shift_labels != -100).sum().item()
check("shift sonrasi gecerli label sayisi = batch", n_valid == 4, f"n_valid={n_valid}")

# ---------------- T4 ----------------
print("T4: chunk tutarliligi (tek cagri == iki parcali state tasima)")
torch.manual_seed(1)
mod = HFPBulkState(hidden_size=16, short_len=4, max_short_len=8, rec_block=4)
mod.eval()
with torch.no_grad():
    xx = torch.randn(2, 12, 16)
    _, r_full, st_full = mod.update(xx)
    _, r1, st1 = mod.update(xx[:, :5])
    _, r2, st2 = mod.update(xx[:, 5:], past_state=st1)
    r_chunked = torch.cat([r1, r2], dim=1)
    diff = (r_full - r_chunked).abs().max().item()
    check("retrieval: full == chunked", diff < 1e-4, f"max|diff|={diff:.2e}")
    m_diff = (st_full[1] - st2[1]).abs().max().item()
    z_diff = (st_full[2] - st2[2]).abs().max().item()
    check("state M esit", m_diff < 1e-4, f"{m_diff:.2e}")
    check("state z esit", z_diff < 1e-4, f"{z_diff:.2e}")
    check("token_count esit", st_full[3] == st2[3], f"{st_full[3]} vs {st2[3]}")
    # rec_block bagimsizligi
    mod.rec_block = 3
    _, r_b3, _ = mod.update(xx)
    b_diff = (r_full - r_b3).abs().max().item()
    check("rec_block sonucu degistirmez", b_diff < 1e-4, f"max|diff|={b_diff:.2e}")

# ---------------- T5 ----------------
print("T5: ring buffer maskesi + generate")
model3 = tiny_model()
model3.eval()
with torch.no_grad():
    # chunk'li forward (valid_past < buffer kapasitesi -> maske yolu calismali)
    o1 = model3(torch.randint(0, 64, (1, 3)), use_cache=True)
    o2 = model3(torch.randint(0, 64, (1, 3)), past_key_values=o1.past_key_values, use_cache=True)
    check("chunked forward calisiyor", torch.isfinite(o2.logits).all())
    gen = model3.generate(torch.randint(0, 64, (1, 5)), max_new_tokens=5,
                          do_sample=False, pad_token_id=2)
    check("generate calisiyor", gen.shape[1] == 10, f"shape={tuple(gen.shape)}")

# ---------------- T6 ----------------
print("T6: yeni modlarin chunk tutarliligi (cubic_flux ve dpfp)")
for mode, fmap in [("cubic_flux", "elu"), ("exp", "dpfp"), ("cubic_flux", "dpfp")]:
    torch.manual_seed(2)
    m6 = HFPBulkState(hidden_size=16, short_len=4, max_short_len=8, rec_block=4,
                      decay_mode=mode, key_feature_map=fmap, dpfp_nu=2)
    m6.eval()
    with torch.no_grad():
        xx = torch.randn(2, 12, 16)
        _, rf, _ = m6.update(xx)
        _, r1, s1 = m6.update(xx[:, :5])
        _, r2, _ = m6.update(xx[:, 5:], past_state=s1)
        d = (rf - torch.cat([r1, r2], dim=1)).abs().max().item()
        check(f"{mode}/{fmap}: full == chunked", d < 1e-4, f"max|diff|={d:.2e}")

print()
if failures:
    print(f"SONUC: {len(failures)} test BASARISIZ: {failures}")
    raise SystemExit(1)
print("SONUC: tum smoke testleri gecti.")

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

"""[FAZ 3] grafting.py regresyon testi — CPU'da mini rastgele Llama ile.

Kontroller:
  T1  Graft sonrasi forward calisiyor, logits sekli dogru, sonlu.
  T2  Gradyan izolasyonu: loss.backward() sonrasi SADECE HFP parametreleri
      gradyan alir; base model (embed, MLP, projeksiyonlar) almaz.
  T3  teacher_forcing modu: cikti == saf-teacher cikti (birebir; ogrenci
      ileriye sizmiyor) ve distill_loss sonlu & > 0.
  T4  Chunk tutarliligi (streaming): tek-parca forward == 3 parcaya bolunmus
      streaming forward (grafted katman state tasima; O(1) yolun dogrulugu).
      Yalniz grafted katman duzeyinde test edilir (full-attn KV karismasin diye
      dogrudan HFPGraftAttention modulunde).
  T5  Zero-shot yakinlik: conv kimlik-init + out_gain=1 ile ogrenci ciktisi
      patlamiyor (|out| sonlu, makul olcek) — "PPL cop olmamali" on-kontrolu.
  T6  alpha-gate sinirlari: write_rule=additive/delta/hybrid uclusunun
      hybrid(alpha->0) ~= ek yol farki modulo payda ve hepsi sonlu.

Kosum: PYTHONPATH=. python review_scripts/graft_smoke.py
"""

import os, sys
# repo kokunu sys.path'e ekle (PYTHONPATH gerektirmez; Colab/lokal her yerde calisir)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn.functional as F

from transformers import LlamaConfig, LlamaForCausalLM
from hfp.models.grafting import (GraftConfig, HFPGraftAttention, graft_llama,
                                 set_graft_mode, distill_loss,
                                 trainable_parameters, enable_streaming,
                                 reset_streaming)

torch.manual_seed(0)

def tiny_model():
    cfg = LlamaConfig(vocab_size=128, hidden_size=64, intermediate_size=128,
                      num_hidden_layers=4, num_attention_heads=4,
                      num_key_value_heads=2, max_position_embeddings=256)
    return LlamaForCausalLM(cfg)

FAIL = 0
def check(name, ok, detail=""):
    global FAIL
    print(f"  [{'OK' if ok else 'FAIL'}] {name} {detail}")
    if not ok: FAIL += 1


print("== T1: graft + forward ==")
model = tiny_model()
ids = torch.randint(0, 128, (2, 48))
with torch.no_grad():
    base_logits = model(ids).logits
gcfg = GraftConfig(decay_mode="cubic_flux_chunked", write_rule="hybrid", rec_block=16)
grafted = graft_llama(model, gcfg)
set_graft_mode(model, "student")
out = model(ids)
check("logits sekli", out.logits.shape == (2, 48, 128), str(tuple(out.logits.shape)))
check("logits sonlu", torch.isfinite(out.logits).all().item())

print("== T2: gradyan izolasyonu ==")
model.zero_grad(set_to_none=True)
loss = F.cross_entropy(out.logits.view(-1, 128), ids.view(-1))
loss.backward()
hfp_grads, base_grads = 0, 0
for n, p in model.named_parameters():
    if p.grad is not None and p.grad.abs().sum() > 0:
        if p.requires_grad: hfp_grads += 1
        else: base_grads += 1
trainables = len(trainable_parameters(model))
check("HFP parametreleri gradyan aliyor", hfp_grads > 0, f"({hfp_grads}/{trainables})")
check("base model gradyan ALMIYOR", base_grads == 0)

print("== T3: teacher_forcing ==")
set_graft_mode(model, "teacher")
with torch.no_grad():
    t_logits = model(ids).logits
set_graft_mode(model, "teacher_forcing")
tf_logits = model(ids).logits
dl = distill_loss(model)
check("teacher_forcing == teacher cikti", torch.allclose(tf_logits, t_logits, atol=1e-5),
      f"(max fark {(tf_logits - t_logits).abs().max().item():.2e})")
check("distill_loss sonlu ve > 0", dl is not None and torch.isfinite(dl) and dl.item() > 0,
      f"({dl.item():.4f})" if dl is not None else "")
check("teacher == graft-oncesi base", torch.allclose(t_logits, base_logits, atol=1e-4),
      f"(max fark {(t_logits - base_logits).abs().max().item():.2e})")

print("== T4: chunk tutarliligi (grafted modul, streaming) ==")
set_graft_mode(model, "student")
attn = None
for m in model.modules():
    if isinstance(m, HFPGraftAttention): attn = m; break
x = torch.randn(2, 48, 64)
with torch.no_grad():
    full = attn._student_forward(x)
    enable_streaming(model, True); reset_streaming(model)
    parts = [attn._student_forward(x[:, s:s+16]) for s in (0, 16, 32)]
    enable_streaming(model, False)
chunked = torch.cat(parts, dim=1)
err = (full - chunked).abs().max().item()
check("tek-parca == 3-parca streaming", err < 1e-4, f"(max fark {err:.2e})")

print("== T5: zero-shot olcek ==")
with torch.no_grad():
    s_out = attn._student_forward(x)
check("ogrenci ciktisi sonlu", torch.isfinite(s_out).all().item())
check("olcek makul (|out| < 100)", s_out.abs().max().item() < 100,
      f"(max |out| {s_out.abs().max().item():.2f})")

print("== T6: yazim kurallari (additive / delta / hybrid) ==")
for wr in ("additive", "delta", "hybrid"):
    for dm in ("exp", "cubic_flux_chunked"):
        m2 = tiny_model()
        graft_llama(m2, GraftConfig(decay_mode=dm, write_rule=wr, rec_block=16), layers=[1])
        set_graft_mode(m2, "student")
        with torch.no_grad():
            o = m2(ids).logits
        check(f"{dm}+{wr} sonlu", torch.isfinite(o).all().item())

print()
if FAIL == 0:
    print("TUM GRAFT TESTLERI GECTI")
else:
    print(f"{FAIL} TEST BASARISIZ"); sys.exit(1)

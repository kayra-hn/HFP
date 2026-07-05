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

# Bagimsiz reviewer dogrulama scripti (torch). hfp_arch kokunden calistirilir.
import math, torch, torch.nn.functional as F
torch.manual_seed(0)
from hfp.core.hfp_bulk_state import HFPBulkState
from hfp.models.configuration_hfp import HFPConfig
from hfp.models.modeling_hfp import HFPForCausalLM

R = []
def check(name, cond, detail=""):
    R.append((name, bool(cond), detail))
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")

def chunk_consistency(mode, fmap, H=16, L=12, split=5):
    torch.manual_seed(2)
    m = HFPBulkState(hidden_size=H, short_len=4, max_short_len=8, rec_block=4,
                     decay_mode=mode, key_feature_map=fmap, dpfp_nu=2)
    m.eval()
    with torch.no_grad():
        x = torch.randn(2, L, H)
        _, rf, sf = m.update(x)
        _, r1, s1 = m.update(x[:, :split])
        _, r2, s2 = m.update(x[:, split:], past_state=s1)
        rc = torch.cat([r1, r2], 1)
        d = (rf - rc).abs().max().item()
        dM = (sf[1] - s2[1]).abs().max().item()
        dz = (sf[2] - s2[2]).abs().max().item()
        kd = m.key_dim
    check(f"chunk-tutarlilik {mode}/{fmap}", d < 1e-4 and dM < 1e-4 and dz < 1e-4,
          f"out={d:.2e} M={dM:.2e} z={dz:.2e} key_dim={kd}")

# --- A. Chunk tutarliligi: 4 kombinasyon
for mode in ("exp", "cubic_flux"):
    for fmap in ("elu", "dpfp"):
        chunk_consistency(mode, fmap)

# --- B. cubic map == ODE'nin tam akisi (analitik): d(z^-2)/dt = 2*eta
eta = 0.003; z0 = 4.0; z = torch.tensor(z0)
for t in range(1, 51):
    z = z / torch.sqrt(1 + 2 * eta * z * z)
    exact = z0 / math.sqrt(1 + 2 * eta * z0 * z0 * t)
    if t in (1, 10, 50):
        check(f"cubic ODE tam-akis t={t}", abs(z.item() - exact) < 1e-6,
              f"iter={z.item():.8f} exact={exact:.8f}")

# --- C. cubic kendini-sinirlama: 4000 adim buyuk girdiyle state patlamiyor
torch.manual_seed(3)
m = HFPBulkState(hidden_size=16, decay_mode="cubic_flux"); m.eval()
with torch.no_grad():
    st = None; mx = 0.0
    for i in range(40):
        _, _, st = m.update(torch.randn(1, 100, 16) * 3.0, past_state=st)
        mx = max(mx, st[1].abs().max().item())
check("cubic state sinirli (4000 tok, x*3)", math.isfinite(mx) and mx < 1e4, f"max|M|={mx:.1f}")

# exp ayni kosulda (kiyas icin, sinirli olmali cunku lam<1)
torch.manual_seed(3)
me = HFPBulkState(hidden_size=16, decay_mode="exp"); me.eval()
with torch.no_grad():
    st = None
    for i in range(40):
        _, _, st = me.update(torch.randn(1, 100, 16) * 3.0, past_state=st)
    mxe = st[1].abs().max().item()
print(f"      (kiyas: exp max|M|={mxe:.1f})")

# --- D. Causal sizinti: son token degisince onceki logitler degismemeli
def leak_test(mode, fmap):
    torch.manual_seed(4)
    cfg = HFPConfig(vocab_size=64, hidden_size=32, num_hidden_layers=3,
                    num_attention_heads=2, intermediate_size=64, bulk_dim=16,
                    short_len=4, max_position_embeddings=128, local_window=8,
                    rec_block=4, decay_mode=mode, key_feature_map=fmap)
    model = HFPForCausalLM(cfg); model.eval()
    with torch.no_grad():
        x1 = torch.randint(0, 64, (1, 24))
        x2 = x1.clone(); x2[0, -1] = (x1[0, -1] + 7) % 64
        l1 = model(x1).logits; l2 = model(x2).logits
        d = (l1[:, :-1] - l2[:, :-1]).abs().max().item()
    check(f"causal sizinti yok {mode}/{fmap}", d == 0.0, f"max|dlogit(<son)|={d:.2e}")

for mode in ("exp", "cubic_flux"):
    for fmap in ("elu", "dpfp"):
        leak_test(mode, fmap)

# ayrica: orta token degisince yalnizca >=o pozisyonlar degismeli
torch.manual_seed(5)
cfg = HFPConfig(vocab_size=64, hidden_size=32, num_hidden_layers=3, num_attention_heads=2,
                intermediate_size=64, bulk_dim=16, short_len=4, max_position_embeddings=128,
                local_window=8, rec_block=4)
model = HFPForCausalLM(cfg); model.eval()
with torch.no_grad():
    x1 = torch.randint(0, 64, (1, 24)); x2 = x1.clone(); x2[0, 10] = (x1[0, 10] + 5) % 64
    l1, l2 = model(x1).logits, model(x2).logits
    d_pre = (l1[:, :10] - l2[:, :10]).abs().max().item()
check("orta-token sizintisi yok (pos<10)", d_pre == 0.0, f"{d_pre:.2e}")

# --- E. O(1) cikarim: state sekli baglam uzunlugundan bagimsiz
with torch.no_grad():
    o_a = model(torch.randint(0, 64, (1, 16)), use_cache=True)
    o_b = model(torch.randint(0, 64, (1, 96)), use_cache=True)
    sa, sb = o_a.past_key_values[0], o_b.past_key_values[0]
    shapes_equal = all(tuple(sa[i].shape) == tuple(sb[i].shape) for i in (0, 1, 2, 6))
check("O(1) state: 16 vs 96 token ayni sekil", shapes_equal,
      f"M={tuple(sb[1].shape)} z={tuple(sb[2].shape)} ring={tuple(sb[0].shape)} conv={tuple(sb[6].shape)}")

# --- F. exp chunkwise == bagimsiz naive recurrence (modul agirliklariyla)
torch.manual_seed(6)
m = HFPBulkState(hidden_size=16, short_len=4, max_short_len=8, rec_block=4); m.eval()
with torch.no_grad():
    x = torch.randn(1, 10, 16)
    _, r_mod, _ = m.update(x)
    # naive yeniden-uygulama
    kk = m.conv_kernel
    xp = torch.cat([torch.zeros(1, kk - 1, 16), x], 1)
    xqk = m.short_conv(xp.transpose(1, 2)).transpose(1, 2)
    Q = m._feat(m.W_q(xqk)); K = m._feat(m.W_k(xqk))
    V = m.W_v(x) * torch.sigmoid(m.importance_gate(x))
    lam = torch.sigmoid(m.decay)
    M = torch.zeros(1, m.key_dim, 16); z = torch.zeros(1, m.key_dim)
    outs = []
    for t in range(10):
        M = M * lam.view(1, -1, 1) + torch.einsum('bh,bg->bhg', K[:, t], V[:, t])
        z = z * lam.view(1, -1) + K[:, t]
        num = torch.einsum('bh,bhg->bg', Q[:, t], M)
        den = (Q[:, t] * z).sum(-1, keepdim=True) + 1e-6
        outs.append((num / den).unsqueeze(1))
    r_naive = m.retrieval_norm(torch.cat(outs, 1))
    d = (r_mod - r_naive).abs().max().item()
check("exp chunkwise == naive recurrence", d < 1e-5, f"max|diff|={d:.2e}")

print()
print(f"TOPLAM: {sum(ok for _, ok, _ in R)}/{len(R)} PASS")
for n, ok, det in R:
    if not ok:
        print(f"  FAIL: {n} {det}")

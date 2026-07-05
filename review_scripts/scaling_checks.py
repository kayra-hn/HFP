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

"""[HFP-SCALE] cubic_flux_chunked dogrulama + olcum: (1) rec_block=1 == tam sirali,
(2) yaklasim hatasi rec_block'a gore, (3) hiz. Repo kokunden calistir."""
import time
import torch
from hfp.core.hfp_bulk_state import HFPBulkState

torch.manual_seed(0)
H, B = 32, 4

def make(mode, rec_block, seed=1):
    torch.manual_seed(seed)
    m = HFPBulkState(hidden_size=H, short_len=4, max_short_len=8,
                     rec_block=rec_block, decay_mode=mode)
    m.eval()
    return m

print("== 1) rec_block=1 tamlik kontrolu (chunked == sirali cubic) ==")
x = torch.randn(B, 48, H)
with torch.no_grad():
    _, r_exact, st_e = make("cubic_flux", 64).update(x)
    _, r_b1, st_1 = make("cubic_flux_chunked", 1).update(x)
d = (r_exact - r_b1).abs().max().item()
dM = (st_e[1] - st_1[1]).abs().max().item()
print(f"  max|out| = {d:.2e}   max|M| = {dM:.2e}   ->", "PASS" if d < 1e-5 else "FAIL")

print("\n== 2) Yaklasim hatasi (L=256, cikti bazinda goreli) ==")
x = torch.randn(B, 256, H)
with torch.no_grad():
    _, r_exact, st_e = make("cubic_flux", 64).update(x)
    scale = r_exact.abs().mean().item()
    for rb in (4, 16, 32, 64):
        _, r_ap, st_a = make("cubic_flux_chunked", rb).update(x)
        rel = (r_ap - r_exact).abs().mean().item() / scale
        mx = (r_ap - r_exact).abs().max().item()
        zrel = (st_a[2] - st_e[2]).abs().mean().item() / st_e[2].abs().mean().item()
        print(f"  rec_block={rb:3d}  ort.goreli={rel:8.4%}  max|fark|={mx:.4f}  z-sapma={zrel:.4%}")

print("\n== 3) Hiz (L=512, B=8, H=64, 10 forward CPU) ==")
xb = torch.randn(8, 512, 64)

def bench(mode, rb):
    torch.manual_seed(2)
    m = HFPBulkState(hidden_size=64, short_len=4, max_short_len=8,
                     rec_block=rb, decay_mode=mode)
    m.eval()
    with torch.no_grad():
        m.update(xb)
        t0 = time.perf_counter()
        for _ in range(10):
            m.update(xb)
        return (time.perf_counter() - t0) / 10

t_exp = bench("exp", 64); t_seq = bench("cubic_flux", 64)
t_c32 = bench("cubic_flux_chunked", 32); t_c64 = bench("cubic_flux_chunked", 64)
print(f"  exp (paralel)              : {t_exp*1e3:7.1f} ms")
print(f"  cubic_flux (sirali)        : {t_seq*1e3:7.1f} ms  ({t_seq/t_exp:.1f}x exp)")
print(f"  cubic_flux_chunked rb=32   : {t_c32*1e3:7.1f} ms  ({t_seq/t_c32:.1f}x hizlanma)")
print(f"  cubic_flux_chunked rb=64   : {t_c64*1e3:7.1f} ms  ({t_seq/t_c64:.1f}x hizlanma)")

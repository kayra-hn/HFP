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

"""HF repo'su icin architecture-only checkpoint uretir.

Bu klasorden calistir:  python make_hf_checkpoint.py [--out hf_release] [--small]
Cikti klasorune bu klasordeki .py dosyalarini ve README.md'yi de kopyala,
hepsini birlikte HuggingFace repo'suna yukle.
"""
import argparse, importlib.util, os, sys, types

# Bu klasordeki modulleri sentetik bir paket olarak yukle (goreli importlar
# calissin diye; klasor adi ne olursa olsun calisir).
_D = os.path.dirname(os.path.abspath(__file__))
_pkg = types.ModuleType("hfp_remote")
_pkg.__path__ = [_D]
sys.modules["hfp_remote"] = _pkg

def _load(name):
    spec = importlib.util.spec_from_file_location(f"hfp_remote.{name}", os.path.join(_D, name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[f"hfp_remote.{name}"] = mod
    spec.loader.exec_module(mod)
    return mod

HFPConfig = _load("configuration_hfp").HFPConfig
HFPForCausalLM = _load("modeling_hfp").HFPForCausalLM

p = argparse.ArgumentParser()
p.add_argument("--out", default="hf_release")
p.add_argument("--small", action="store_true", help="hizli test icin kucuk model")
args = p.parse_args()

if args.small:
    cfg = HFPConfig(vocab_size=1000, hidden_size=64, num_hidden_layers=2,
                    num_attention_heads=2, intermediate_size=256, bulk_dim=32,
                    local_window=32, max_position_embeddings=512)
else:
    # ~103M param (eski kartla ayni sinif: "0.1B")
    cfg = HFPConfig(vocab_size=50257, hidden_size=768, num_hidden_layers=12,
                    num_attention_heads=12, intermediate_size=3072, bulk_dim=128,
                    short_len=8, local_window=64, max_position_embeddings=4096,
                    decay_mode="exp")   # baseline default; cubic_flux flag ile secilir

# trust_remote_code eslesmesi: AutoConfig/AutoModel bu dosyalari yukler
cfg.auto_map = {
    "AutoConfig": "configuration_hfp.HFPConfig",
    "AutoModelForCausalLM": "modeling_hfp.HFPForCausalLM",
}
cfg.architectures = ["HFPForCausalLM"]

model = HFPForCausalLM(cfg)
n = sum(p.numel() for p in model.parameters())
model.save_pretrained(args.out)
print(f"kaydedildi: {args.out}  ({n/1e6:.1f}M param)")
print("Yuklenecek dosyalar: model.safetensors + config.json + .py dosyalari + README.md")

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
review_scripts/verify_graft.py - HFP Grafting Modulu (grafting.py) Dogrulama Suite'i.

Grafting mimarisi ve streaming/reset/mode degismezlerinin CPU'da, ag erişimsiz,
kucuk rastgele bir Llama modeliyle (~1 dk) sinanmasi.

Kapsanan degismezler:
  T1: graft_llama katman hedeflemesi ve parametre dondurma (requires_grad).
  T2: reset_streaming state temizligi (Y == Z bit-bit esit, X != Y).
  T3: enable_streaming(m, False) durum tasimayi kapatir ve state'i temizler.
  T4: set_graft_mode mod farkliliklari (teacher == orijinal softmax attention).
  T5: load_checkpoint_safe ile sessiz checkpoint yukleme hatasi tespiti.
"""

import os
import sys

# Ensure repository root is in sys.path when running from any CWD
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM

from hfp.models.grafting import (
    GraftConfig,
    HFPGraftAttention,
    graft_llama,
    set_graft_mode,
    enable_streaming,
    reset_streaming,
    trainable_parameters,
)


def load_checkpoint_safe(model: nn.Module, state_dict: dict):
    """Safely load state_dict into model with strict=False and verify non-zero matching tensors.
    
    Returns (num_matched, missing_keys, unexpected_keys).
    Raises ValueError if 0 tensors were matched.
    """
    m_keys = set(model.state_dict().keys())
    s_keys = set(state_dict.keys())
    matched_keys = m_keys.intersection(s_keys)
    res = model.load_state_dict(state_dict, strict=False)
    if len(matched_keys) == 0:
        raise ValueError(
            f"Checkpoint loading failed: 0 matching tensors found between state_dict ({len(s_keys)} keys) and model ({len(m_keys)} keys)."
        )
    return len(matched_keys), res.missing_keys, res.unexpected_keys


def make_dummy_llama(vocab_size=100, hidden_size=64, intermediate_size=128, num_layers=4, num_heads=4):
    cfg = AutoConfig.for_model(
        "llama",
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_hidden_layers=num_layers,
        num_attention_heads=num_heads,
        num_key_value_heads=num_heads,
        max_position_embeddings=512,
    )
    model = AutoModelForCausalLM.from_config(cfg)
    model.eval()
    return model


def test_t1_graft_llama_targeting():
    print("[T1] Katman hedeflemesi ve parametre dondurma testi...", end=" ")
    model = make_dummy_llama(num_layers=4)
    graft_layers = [1]
    grafted_indices = graft_llama(model, GraftConfig(decay_mode="exp", write_rule="additive"), layers=graft_layers)

    assert grafted_indices == [1], f"Expected grafted indices [1], got {grafted_indices}"
    assert isinstance(model.model.layers[1].self_attn, HFPGraftAttention), "Layer 1 self_attn must be HFPGraftAttention"
    for idx in [0, 2, 3]:
        assert not isinstance(model.model.layers[idx].self_attn, HFPGraftAttention), f"Layer {idx} should NOT be HFPGraftAttention"

    # Donuk parametre kontrolu: Graft harici katmanlar requires_grad=False olmali
    for idx in [0, 2, 3]:
        for p in model.model.layers[idx].parameters():
            assert not p.requires_grad, f"Layer {idx} parameters must be frozen"

    # Graft katmaninda HFP parametreleri requires_grad=True, teacher/proj parametreleri False (own_proj=False)
    graft_module = model.model.layers[1].self_attn
    assert graft_module.decay.requires_grad, "graft decay parameter must require grad"
    assert graft_module.log_eta.requires_grad, "graft log_eta parameter must require grad"
    assert not graft_module.teacher.q_proj.weight.requires_grad, "teacher q_proj must be frozen"

    trainable = trainable_parameters(model)
    expected_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    actual_count = sum(p.numel() for p in trainable)
    assert expected_count == actual_count, f"Trainable parameter count mismatch: {expected_count} vs {actual_count}"

    print(f"PASSED ({actual_count:,} trainable params)")


def test_t2_reset_streaming():
    print("[T2] reset_streaming durum temizligi testi...", end=" ")
    model = make_dummy_llama(num_layers=4)
    graft_llama(model, GraftConfig(decay_mode="exp", write_rule="additive"), layers=[1])
    enable_streaming(model, True)

    seq_A = torch.randint(0, 100, (1, 16))
    seq_B = torch.randint(0, 100, (1, 16))

    # 1. State tasima: A sonra B
    with torch.no_grad():
        _ = model(seq_A)
        out_X = model(seq_B).logits

    # 2. Reset streaming: B tek basina
    reset_streaming(model)
    with torch.no_grad():
        out_Y = model(seq_B).logits

    # 3. Taze model: B tek basina (hic state yok)
    fresh_model = make_dummy_llama(num_layers=4)
    graft_llama(fresh_model, GraftConfig(decay_mode="exp", write_rule="additive"), layers=[1])
    fresh_model.load_state_dict(model.state_dict())
    enable_streaming(fresh_model, True)
    with torch.no_grad():
        out_Z = fresh_model(seq_B).logits

    # DEGISMEZLER: Y ve Z bit-bit esit olmali, X ise Y ve Z'den farkli olmali
    max_diff_yz = torch.max(torch.abs(out_Y - out_Z)).item()
    max_diff_xy = torch.max(torch.abs(out_X - out_Y)).item()

    assert max_diff_yz < 1e-6, f"Y and Z outputs must be bit-for-bit identical, max diff: {max_diff_yz}"
    assert max_diff_xy > 1e-4, f"X output must differ from Y (state carry effect), max diff: {max_diff_xy}"

    print(f"PASSED (diff Y-Z={max_diff_yz:.2e}, diff X-Y={max_diff_xy:.4f})")


def test_t3_enable_streaming_false():
    print("[T3] enable_streaming(m, False) kapatma testi...", end=" ")
    model = make_dummy_llama(num_layers=4)
    graft_llama(model, GraftConfig(decay_mode="exp", write_rule="additive"), layers=[1])

    enable_streaming(model, True)
    seq_A = torch.randint(0, 100, (1, 16))
    with torch.no_grad():
        _ = model(seq_A)

    graft_mod = model.model.layers[1].self_attn
    assert graft_mod._stream_state is not None, "Stream state should be present after forward"

    enable_streaming(model, False)
    assert not graft_mod.streaming, "Streaming flag must be False"
    assert graft_mod._stream_state is None, "Stream state must be cleared after enable_streaming(model, False)"

    print("PASSED")


def test_t4_graft_modes():
    print("[T4] set_graft_mode davranis testi...", end=" ")
    base_model = make_dummy_llama(num_layers=4)
    model = make_dummy_llama(num_layers=4)
    model.load_state_dict(base_model.state_dict())
    graft_llama(model, GraftConfig(decay_mode="exp", write_rule="additive"), layers=[1])

    seq = torch.randint(0, 100, (1, 16))

    with torch.no_grad():
        out_orig = base_model(seq).logits

        set_graft_mode(model, "teacher")
        out_teacher = model(seq).logits

        set_graft_mode(model, "student")
        out_student = model(seq).logits

        set_graft_mode(model, "teacher_forcing")
        out_tf = model(seq).logits

    # DEGISMEZ: 'teacher' modu orijinal softmax attention ile BIREBIR AYNI cikti uretmeli
    diff_teacher_orig = torch.max(torch.abs(out_teacher - out_orig)).item()
    assert diff_teacher_orig < 1e-6, f"Teacher mode output must match original un-grafted model, diff: {diff_teacher_orig}"

    # 'student' ve 'teacher' modlari birbirinden farkli davranmali
    diff_student_teacher = torch.max(torch.abs(out_student - out_teacher)).item()
    assert diff_student_teacher > 1e-4, f"Student output must differ from teacher, diff: {diff_student_teacher}"

    print(f"PASSED (teacher==orig diff={diff_teacher_orig:.2e}, student!=teacher diff={diff_student_teacher:.4f})")


def test_t5_checkpoint_load_safety():
    print("[T5] Checkpoint yukleme guvenlik testi (load_checkpoint_safe)...", end=" ")
    model = make_dummy_llama(num_layers=4)
    graft_llama(model, GraftConfig(decay_mode="exp", write_rule="additive"), layers=[1])

    valid_sd = {k: v.clone() for k, v in model.state_dict().items()}
    matched_count, missing, unexpected = load_checkpoint_safe(model, valid_sd)
    assert matched_count > 0, "Valid state_dict must match non-zero tensors"

    corrupted_sd = {f"corrupted_prefix.{k}": v.clone() for k, v in valid_sd.items()}
    caught = False
    try:
        load_checkpoint_safe(model, corrupted_sd)
    except ValueError as e:
        if "0 matching tensors found" in str(e):
            caught = True

    assert caught, "load_checkpoint_safe must raise ValueError when 0 tensors match"
    print("PASSED")


def main():
    print("=" * 70)
    print("HFP GRAFTING VERIFICATION SUITE (verify_graft.py)")
    print("=" * 70)

    test_t1_graft_llama_targeting()
    test_t2_reset_streaming()
    test_t3_enable_streaming_false()
    test_t4_graft_modes()
    test_t5_checkpoint_load_safety()

    print("-" * 70)
    print("ALL GRAFT VERIFICATION TESTS PASSED — GATE APPROVED")
    print("-" * 70)


if __name__ == "__main__":
    main()

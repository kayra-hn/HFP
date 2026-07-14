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
run_experiment.py - HFP icin durust, kucuk-olcekli yetenek deneyi.

Iki gorev:
  --task recall : Sentetik associative-recall (MQAR). O(1) bellegin bilgiyi uzun
                  baglam boyunca gercekten TUTUP tutmadigini olcen test.
  --task lm     : TinyShakespeare dil modeli. Gercek, tekrar-uretilebilir
                  validation loss + perplexity.

[FIX K1] Onceki surumde cifte label-kaydirma target'i loss'un tamamen disina
atiyordu -> CrossEntropy tum -100 -> loss NaN (egitim ilk adimdan kirikti).
Yeni tasarim: diziye kq'dan sonra bir CEVAP pozisyonu (ANS token) eklenir,
label yalnizca o pozisyondadir, manuel shift YOKTUR - modelin kendi
(logits[:-1] <-> labels[1:]) kaydirmasi kq-pozisyonundaki logiti target ile
esler.

[FIX K5/OLCUM] Onceki tasarim tek forward'la tum diziyi veriyordu; lokal
attention tum k-v ciftlerini dogrudan gordugu icin test BELLEGI degil
attention'i olcuyordu. Simdi:
  - Model local_window ile kurulur (sorgu, pencere disindaki k-v ciftlerini
    attention'la GOREMEZ; bilgi yalnizca recurrent bellekten akabilir).
  - Eval uc modda raporlanir:
      full    : tek forward (pencere nedeniyle yine bellek gerektirir)
      chunked : dizi parcalar halinde use_cache=True ile akitilir; bilgi
                chunk sinirini YALNIZCA M/z state uzerinden gecebilir
      reset   : chunked ama her chunk'ta state SIFIRLANIR (ablasyon kontrolu;
                sans seviyesine dusmeli - dusmuyorsa test bellegi olcmuyordur)
"""
import argparse, math, random
import numpy as np
import torch
from transformers import get_cosine_schedule_with_warmup
from hfp.models.configuration_hfp import HFPConfig
from hfp.models.modeling_hfp import HFPForCausalLM

ANS_TOKEN = 0  # cevap-pozisyonu tokeni (filler >=1, key/val >=100 -> cakismaz)


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)


# ---------------- MQAR (associative recall) ----------------
def make_mqar_sequence(num_pairs, context_len, key_lo, key_hi, val_lo, val_hi, filler_hi):
    """Tek ornek uretir: toks = [k1 v1 ... kP vP <filler...> kq ANS], target = vq.
    Dizi uzunlugu = context_len (context_len >= 2*num_pairs + 2 olmali)."""
    core = 2 * num_pairs
    keys = random.sample(range(key_lo, key_hi), num_pairs)
    vals = [random.randint(val_lo, val_hi - 1) for _ in range(num_pairs)]
    toks = []
    for k, v in zip(keys, vals):
        toks += [k, v]
    pad = max(0, context_len - core - 2)
    toks += [random.randint(1, filler_hi - 1) for _ in range(pad)]
    qi = random.randrange(num_pairs)
    toks += [keys[qi], ANS_TOKEN]
    return toks, vals[qi]


def make_mqar_batch(batch, num_pairs, context_len, key_lo, key_hi, val_lo, val_hi, filler_hi, device):
    """[FIX K1] Label yalnizca ANS pozisyonunda; manuel shift YOK.
    HFPForCausalLM icerde kaydirir: logits[t] <-> labels[t+1]; boylece
    kq-pozisyonundaki logit (kq'yu gormus) ANS-pozisyonundaki target'i tahmin eder."""
    seqs, labels = [], []
    for _ in range(batch):
        toks, target = make_mqar_sequence(num_pairs, context_len, key_lo, key_hi, val_lo, val_hi, filler_hi)
        lab = [-100] * len(toks)
        lab[-1] = target  # ANS pozisyonu
        seqs.append(toks); labels.append(lab)
    x = torch.tensor(seqs, dtype=torch.long, device=device)
    y = torch.tensor(labels, dtype=torch.long, device=device)
    return x, y


@torch.no_grad()
def predict_full(model, toks, device):
    """Tek forward: kq'ya kadar (ANS haric) ver, son pozisyonun logitini oku."""
    x = torch.tensor([toks[:-1]], dtype=torch.long, device=device)
    return model(x).logits[0, -1].argmax().item()


@torch.no_grad()
def predict_chunked(model, toks, device, chunk_size, reset_between=False):
    """Diziyi chunk'lar halinde akit; bilgi chunk sinirini yalnizca state (M, z,
    ring buffer) uzerinden gecebilir. reset_between=True -> state tasima YOK
    (ablasyon kontrolu: sans seviyesine dusmeli)."""
    seq = toks[:-1]  # ANS haric, son token = kq
    past = None
    logits = None
    for s in range(0, len(seq), chunk_size):
        chunk = torch.tensor([seq[s:s + chunk_size]], dtype=torch.long, device=device)
        out = model(chunk, past_key_values=past, use_cache=True)
        logits = out.logits
        past = None if reset_between else out.past_key_values
    return logits[0, -1].argmax().item()


def run_recall(args, device):
    ranges = (100, 100 + args.key_space, 100 + args.key_space, 100 + args.key_space + args.val_space, 100)
    key_lo, key_hi, val_lo, val_hi, filler_hi = ranges
    vocab = val_hi + 4
    assert args.context >= 2 * args.pairs + 2, "context, 2*pairs+2'den kucuk olamaz"
    config = HFPConfig(
        vocab_size=vocab, hidden_size=args.hidden, num_hidden_layers=args.layers,
        num_attention_heads=args.heads, intermediate_size=4 * args.hidden,
        short_len=8, bulk_dim=32, max_position_embeddings=args.context + 8,
        local_window=args.local_window,  # [K5] bellek testinin on kosulu
        decay_mode=args.decay_mode,      # [HFP-CORE] exp | cubic_flux
        key_feature_map=args.key_feature_map, dpfp_nu=args.dpfp_nu,  # [HFP-CAP]
    )
    model = HFPForCausalLM(config).to(device)
    n_par = sum(p.numel() for p in model.parameters()) / 1e6
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    sched = get_cosine_schedule_with_warmup(opt, args.warmup, args.steps)
    print(f"[RECALL] params={n_par:.2f}M pairs={args.pairs} context={args.context} "
          f"vocab={vocab} local_window={args.local_window}")

    def eval_acc(trials=200, mode="full"):
        model.eval(); ok = 0
        for _ in range(trials):
            toks, target = make_mqar_sequence(args.pairs, args.context, key_lo, key_hi, val_lo, val_hi, filler_hi)
            if hasattr(model, 'hfp'):
                for b in model.hfp.bulk_states: b.reset_state()
            if mode == "full":
                pred = predict_full(model, toks, device)
            elif mode == "chunked":
                pred = predict_chunked(model, toks, device, args.eval_chunk)
            else:  # "reset" ablasyonu
                pred = predict_chunked(model, toks, device, args.eval_chunk, reset_between=True)
            ok += int(pred == target)
        model.train()
        return 100.0 * ok / trials

    for step in range(1, args.steps + 1):
        x, y = make_mqar_batch(args.batch, args.pairs, args.context, key_lo, key_hi, val_lo, val_hi, filler_hi, device)
        out = model(x, labels=y)
        if not torch.isfinite(out.loss):
            raise RuntimeError(f"step {step}: loss NaN/Inf - label hizalamasi bozuk olabilir")
        opt.zero_grad(set_to_none=True)
        out.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        if step % args.eval_interval == 0 or step == 1:
            acc = eval_acc(100, "full")
            print(f"step {step:5d} | loss {out.loss.item():.4f} | recall_acc(full) {acc:5.1f}%  (sans: {100.0/args.val_space:.2f}%)")

    chance = 100.0 / args.val_space
    acc_full = eval_acc(500, "full")
    acc_chunk = eval_acc(500, "chunked")
    acc_reset = eval_acc(500, "reset")
    print(f"\n[RECALL SONUC]  sans seviyesi = {chance:.2f}%")
    print(f"  full    (tek forward, pencere={args.local_window}) : {acc_full:5.1f}%")
    print(f"  chunked (state tasinir, chunk={args.eval_chunk})    : {acc_chunk:5.1f}%")
    print(f"  reset   (state SIFIRLANIR - ablasyon)             : {acc_reset:5.1f}%")
    print(">>> Yorum: 'chunked' sansin cok ustunde VE 'reset' sansa yakinsa,")
    print(">>> bilgiyi tasiyan sey gercekten recurrent bellek (M/z) demektir.")
    print(">>> 'reset' de yuksekse test bellegi OLCMUYORDUR (rapor edilecek bulgu).")


def run_lm(args, device):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("gpt2")
    data = np.array(tok.encode(open("tinyshakespeare.txt", encoding="utf-8").read()))
    n = int(0.9 * len(data)); tr, va = data[:n], data[n:]
    config = HFPConfig(
        vocab_size=len(tok), hidden_size=args.hidden, num_hidden_layers=args.layers,
        num_attention_heads=args.heads, intermediate_size=4 * args.hidden,
        short_len=8, bulk_dim=32, max_position_embeddings=args.seq + 8,
        local_window=args.local_window if args.lm_window else None,
        decay_mode=args.decay_mode,      # [HFP-CORE] exp | cubic_flux
        key_feature_map=args.key_feature_map, dpfp_nu=args.dpfp_nu,  # [HFP-CAP]
    )
    model = HFPForCausalLM(config).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    sched = get_cosine_schedule_with_warmup(opt, args.warmup, args.steps)
    print(f"[LM] params={sum(p.numel() for p in model.parameters())/1e6:.2f}M seq={args.seq}")

    def get_batch(d):
        ix = torch.randint(len(d) - args.seq, (args.batch,))
        x = torch.stack([torch.from_numpy(d[i:i+args.seq].astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy(d[i+1:i+1+args.seq].astype(np.int64)) for i in ix])
        return x.to(device), y.to(device)

    @torch.no_grad()
    def val_loss(iters=50):
        model.eval(); tot = 0.0
        for _ in range(iters):
            x, y = get_batch(va); tot += model(x, labels=y).loss.item()
        model.train(); return tot / iters

    for step in range(1, args.steps + 1):
        x, y = get_batch(tr)
        out = model(x, labels=y)
        opt.zero_grad(set_to_none=True); out.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        if step % args.eval_interval == 0 or step == 1:
            vl = val_loss()
            print(f"step {step:5d} | train {out.loss.item():.4f} | val {vl:.4f} | ppl {math.exp(vl):.1f}")
    vl = val_loss(100)
    print(f"\n[LM SONUC] val loss {vl:.4f} | perplexity {math.exp(vl):.1f}")


# ---------------- Retention curve (kubik-plato vs exp imza testi) ----------------
def make_retention_sequence(gap, context, key_lo, key_hi, val_lo, val_hi, filler_hi):
    """Tek (k,v) cifti; v sorgudan tam 'gap' token once. Sorgu=k -> hedef=v.
    gap > local_window ise bilgi yalnizca recurrent bellekten akabilir."""
    assert context >= gap + 4, "context, gap+4'ten kucuk olamaz"
    toks = [random.randint(1, filler_hi - 1) for _ in range(context)]
    vpos = context - 2 - gap
    kpos = vpos - 1
    k = random.randrange(key_lo, key_hi)
    v = random.randint(val_lo, val_hi - 1)
    toks[kpos] = k
    toks[vpos] = v
    toks[context - 2] = k          # sorgu pozisyonu = ayni anahtar token
    toks[context - 1] = ANS_TOKEN
    return toks, v


def run_retention(args, device):
    ranges = (100, 100 + args.key_space, 100 + args.key_space, 100 + args.key_space + args.val_space, 100)
    key_lo, key_hi, val_lo, val_hi, filler_hi = ranges
    vocab = val_hi + 4
    max_gap = args.max_gap
    assert args.context >= max_gap + 4, "context >= max_gap+4 olmali"
    config = HFPConfig(
        vocab_size=vocab, hidden_size=args.hidden, num_hidden_layers=args.layers,
        num_attention_heads=args.heads, intermediate_size=4 * args.hidden,
        short_len=8, bulk_dim=32, max_position_embeddings=args.context + 8,
        local_window=args.local_window, decay_mode=args.decay_mode,
        key_feature_map=args.key_feature_map, dpfp_nu=args.dpfp_nu,  # [HFP-CAP]
    )
    model = HFPForCausalLM(config).to(device)
    n_par = sum(p.numel() for p in model.parameters()) / 1e6
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    sched = get_cosine_schedule_with_warmup(opt, args.warmup, args.steps)
    print(f"[RETENTION] mode={args.decay_mode} params={n_par:.2f}M context={args.context} "
          f"window={args.local_window} max_gap={max_gap} vocab={vocab}")

    def make_batch(bs):
        seqs, labels = [], []
        for _ in range(bs):
            g = random.randint(1, max_gap)
            toks, tgt = make_retention_sequence(g, args.context, key_lo, key_hi, val_lo, val_hi, filler_hi)
            lab = [-100] * len(toks); lab[-1] = tgt
            seqs.append(toks); labels.append(lab)
        return (torch.tensor(seqs, dtype=torch.long, device=device),
                torch.tensor(labels, dtype=torch.long, device=device))

    @torch.no_grad()
    def acc_at(gap, trials=300):
        model.eval(); ok = 0
        for _ in range(trials):
            toks, tgt = make_retention_sequence(gap, args.context, key_lo, key_hi, val_lo, val_hi, filler_hi)
            if hasattr(model, 'hfp'):
                for b in model.hfp.bulk_states: b.reset_state()
            x = torch.tensor([toks[:-1]], dtype=torch.long, device=device)
            pred = model(x).logits[0, -1].argmax().item()
            ok += int(pred == tgt)
        model.train(); return 100.0 * ok / trials

    for step in range(1, args.steps + 1):
        x, y = make_batch(args.batch)
        out = model(x, labels=y)
        if not torch.isfinite(out.loss):
            raise RuntimeError(f"step {step}: loss NaN/Inf")
        opt.zero_grad(set_to_none=True); out.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        if step % args.eval_interval == 0 or step == 1:
            print(f"step {step:5d} | loss {out.loss.item():.4f}")

    chance = 100.0 / args.val_space
    gaps = [g for g in [1, 2, 4, 8, 16, 32, 64, 128, 256] if g <= max_gap]
    print(f"\n[RETENTION SONUC mode={args.decay_mode}]  sans={chance:.2f}%  window={args.local_window}")
    print("  gap | acc%    (gap>window => bilgi yalnizca bellekten)")
    for g in gaps:
        a = acc_at(g)
        flag = "" if g <= args.local_window else "   <-bellek"
        print(f"  {g:4d} | {a:5.1f}{flag}")
    print(">>> Ayni komutu --decay_mode exp ve --decay_mode cubic_flux ile calistir, egrileri karsilastir.")
    print(">>> cubic_flux buyuk gap'lerde (>window) exp'ten YUKSEK acc tutuyorsa, plato")
    print(">>> mekanizmasinin OLCULMUS uzun-menzil avantajidir (makalenin ana iddiasi).")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", choices=["recall", "lm", "retention"], default="recall")
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--eval_interval", type=int, default=200)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    # recall'a ozel
    p.add_argument("--pairs", type=int, default=8)
    p.add_argument("--context", type=int, default=256)
    p.add_argument("--key_space", type=int, default=100)
    p.add_argument("--val_space", type=int, default=100)
    p.add_argument("--decay_mode", choices=["exp", "cubic_flux", "cubic_flux_chunked"], default="exp",
                   help="[HFP-CORE] bellek retention yasasi: exp=geometrik baseline, "
                        "cubic_flux=makalenin kubik-plato akisi (ayirt edici mekanizma), "
                        "cubic_flux_chunked=iki-gecisli TAM paralel form (olcekleme)")
    p.add_argument("--key_feature_map", choices=["elu", "dpfp"], default="elu",
                   help="[HFP-CAP] bellek anahtar ozellik-haritasi: elu=baseline, "
                        "dpfp=genisletilmis (kapasite/rank-collapse ekseni)")
    p.add_argument("--dpfp_nu", type=int, default=2, help="[HFP-CAP] dpfp genisleme faktoru")
    p.add_argument("--local_window", type=int, default=32,
                   help="[K5] lokal attention penceresi; bellek testi icin context'ten kucuk olmali")
    p.add_argument("--eval_chunk", type=int, default=64,
                   help="chunked eval'de chunk boyutu (state chunk sinirini M/z ile gecer)")
    p.add_argument("--max_gap", type=int, default=64,
                   help="[retention] egitimde ornekle nen maksimum (k,v)->sorgu mesafesi")
    # lm'e ozel
    p.add_argument("--seq", type=int, default=256)
    p.add_argument("--lm_window", action="store_true",
                   help="LM gorevinde de local_window uygula (default: tam attention)")
    args = p.parse_args()
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}  task={args.task}")
    {"recall": run_recall, "lm": run_lm, "retention": run_retention}[args.task](args, device)


if __name__ == "__main__":
    main()

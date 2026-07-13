# =============================================================================
# HFP DEGRADASYON PROBU — TEK HUCRE, KENDI KENDINE YETEN (Colab T4, ~100 dk)
# Yeni bos Colab notebook'una tek hucre olarak yapistir ve kos.
#
# ON-KAYITLI HIPOTEZLER (sonuca gore kriter yazmak yok):
#  H0 (metrik): v3/ablasyon HFP kosulari cift-kaydirma ile egitildi/olculdu
#     (estimate_loss y=x[t+1..] verir, HFPForCausalLM iceride BIR DAHA kaydirir
#     -> hedef x[t+2], "skip-one"). GLA kaydirmadigi icin dogru next-token idi.
#     TEST: ayni protokolle labels=x (dogru) egitim; eval TEK konvansiyonla
#     (manuel CE: logits[:,:-1] vs x[:,1:]). Ek: ayni modelde 1-ileri vs
#     2-ileri metrik farki -> eski sayilarin kalibrasyonu.
#  H1 (attention-OOD): config'de local_window=None -> TAM causal attention.
#     Egitimde hicbir sorgu 256 tokenden uzagi gormedi; eval@2048'de 8x menzil
#     + gorulmemis sinusoidal pozisyonlar. TEST-E1: ayni agirliklarla eval'de
#     local_window=256 dayat. TEST-E2: ek olarak PE'yi mod-256 dosele.
#  H2 (state-OOD): bellek (M,z) 2048 tokenlik birikimi egitimde hic gormedi.
#     E2'de duzelmeyen kisim + per-position egrinin SEKLI (256'da kirilma =
#     pozisyonel; duzgun monoton artis = kumulatif state/girisim) ayirt eder.
#
# YORUM KILAVUZU (onceden yazili):
#  - E1 PPL@2048, E0@2048 -> E0@256 farkinin >= yarisini kapatirsa: H1 baskin.
#  - E2, E1'e gore ek anlamli iyilesme verirse: PE katkisi olculur.
#  - E1/E2 duzelmiyorsa: H2 baskin -> pencere degil bellek dinamigi konusulur.
#  - Tek seed: TESHIS kosusudur, iddia kosusu degil (etiket: tek-seed probe).
# =============================================================================
import os, sys, math, time, csv, urllib.request, subprocess
import torch, numpy as np
import torch.nn.functional as F

assert torch.cuda.is_available(), 'GPU YOK! Runtime > Change runtime type > T4 GPU.'
device = 'cuda'
torch.manual_seed(0); np.random.seed(0)

# --- repo (yalniz hfp paketi) ---
if not os.path.exists('HFP'):
    subprocess.run(['git', 'clone', '--depth', '1',
                    'https://github.com/kayra-hn/HFP.git'], check=True)
sys.path.insert(0, 'HFP')
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q',
                'transformers>=4.40'], check=True)
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
from hfp.models.configuration_hfp import HFPConfig
from hfp.models.modeling_hfp import HFPForCausalLM

# --- veri: WikiText-2, v3 ile birebir ayni kaynak/tokenizasyon ---
BASE = ('https://raw.githubusercontent.com/pytorch/examples/master/'
        'word_language_model/data/wikitext-2/')
for fn in ['train.txt', 'valid.txt']:
    if not os.path.exists(fn):
        urllib.request.urlretrieve(BASE + fn, fn)
tokenizer = AutoTokenizer.from_pretrained('gpt2')
def tokenize(path):
    text = open(path, encoding='utf-8').read()
    ids, chunk = [], 500_000
    for i in range(0, len(text), chunk):
        ids.extend(tokenizer.encode(text[i:i+chunk], truncation=False))
    return np.array(ids, dtype=np.uint16)
train_data, val_data = tokenize('train.txt'), tokenize('valid.txt')
print(f'train {len(train_data)/1e6:.2f}M tok, val {len(val_data)/1e6:.2f}M tok', flush=True)

# --- model: v3 Gorev B hfp_add kolu ile birebir ayni konfig ---
SEQ, BATCH, LR, MAX_IT, EVAL_INT, PATIENCE = 256, 16, 5e-4, 5000, 200, 7
EVAL_LENS = [256, 1024, 2048]
config = HFPConfig(
    vocab_size=len(tokenizer), hidden_size=256, num_hidden_layers=4,
    num_attention_heads=4, max_position_embeddings=max([SEQ] + EVAL_LENS),
    short_len=8, bulk_dim=32, decay_mode='cubic_flux_chunked',
    key_feature_map='dpfp', write_rule='additive', ffn_type='standard',
    rec_block=16)
model = HFPForCausalLM(config).to(device)
print('Params: {:.2f}M'.format(sum(p.numel() for p in model.parameters())/1e6), flush=True)

def get_batch(data, L, B):
    ix = np.random.randint(0, len(data) - L - 1, B)
    x = torch.stack([torch.from_numpy(data[i:i+L].astype(np.int64)) for i in ix])
    return x.to(device)

# --- TEK KONVANSIYON eval: manuel next-token CE (labels yoluna hic girmez) ---
@torch.no_grad()
def eval_ce(model, data, L, iters=30, per_position=False, two_ahead=False):
    model.eval()
    B = max(1, (BATCH * SEQ) // L)
    tot, pos_sum, pos_cnt = [], None, 0
    for _ in range(iters):
        x = get_batch(data, L, B)
        logits = model(x).logits
        if two_ahead:   # eski hatali eslesmenin replikasi: logits[t] vs x[t+2]
            ce = F.cross_entropy(logits[:, :-2].reshape(-1, logits.size(-1)),
                                 x[:, 2:].reshape(-1))
            tot.append(ce.item()); continue
        ce = F.cross_entropy(logits[:, :-1].reshape(-1, logits.size(-1)),
                             x[:, 1:].reshape(-1), reduction='none')
        ce = ce.view(x.size(0), -1)                     # (B, L-1)
        tot.append(ce.mean().item())
        if per_position:
            pos_sum = ce.sum(0) if pos_sum is None else pos_sum + ce.sum(0)
            pos_cnt += x.size(0)
    model.train()
    m = float(np.mean(tot))
    return (m, (pos_sum / pos_cnt).cpu().numpy() if per_position else None)

# --- EGITIM (DOGRU konvansiyon: labels=x -> HFP iceride dogru kaydirir) ---
print('\n=== EGITIM: cubic+additive+dpfp, seed 0, DOGRU next-token hedefi ===', flush=True)
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.1)
sch = get_cosine_schedule_with_warmup(opt, 100, MAX_IT)
best, pat, t0 = float('inf'), 0, time.time()
for it in range(MAX_IT):
    if it % EVAL_INT == 0 or it == MAX_IT - 1:
        vl, _ = eval_ce(model, val_data, SEQ, iters=50)
        print(f'step {it}: val {vl:.4f} ppl {math.exp(vl):.1f} ({(time.time()-t0)/60:.0f} dk)', flush=True)
        if vl < best: best, pat = vl, 0
        else:
            pat += 1
            if pat >= PATIENCE:
                print(f'Early stop @ {it}. best {best:.4f}', flush=True); break
    x = get_batch(train_data, SEQ, BATCH)
    out = model(x, labels=x)          # DOGRU: iceride kaydirilir -> next-token
    assert torch.isfinite(out.loss), f'NaN/Inf @ step {it} — kosu durduruldu'
    opt.zero_grad(set_to_none=True); out.loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step(); sch.step()
torch.save(model.state_dict(), 'probe_hfp_add_s0.pt')   # bu kez SAKLIYORUZ
print('checkpoint: probe_hfp_add_s0.pt', flush=True)

# --- H0 KALIBRASYONU: ayni modelde 1-ileri vs 2-ileri metrik ---
one, _ = eval_ce(model, val_data, SEQ, iters=50)
two, _ = eval_ce(model, val_data, SEQ, iters=50, two_ahead=True)
print(f'\n[H0] ayni model @256: next-token {one:.4f} (PPL {math.exp(one):.1f})'
      f'  vs  2-ileri {two:.4f} (PPL {math.exp(two):.1f})  fark {two-one:+.4f}', flush=True)

# --- E0: standart eval + per-position @2048 ---
rows = [('variant', 'eval_len', 'val_loss', 'ppl')]
print('\n=== E0: standart konfig ===', flush=True)
for L in EVAL_LENS:
    vl, pos = eval_ce(model, val_data, L, iters=30, per_position=(L == 2048))
    rows.append(('E0_standart', L, f'{vl:.4f}', f'{math.exp(vl):.1f}'))
    print(f'  @{L}: {vl:.4f} (PPL {math.exp(vl):.1f})', flush=True)
    if pos is not None:
        with open('probe_poswise.csv', 'w', newline='') as f:
            w = csv.writer(f); w.writerow(['pos', 'loss'])
            for p, l in enumerate(pos): w.writerow([p + 1, f'{l:.4f}'])
        print('  [H2] pozisyon-kovasi ortalama loss @2048:', flush=True)
        for a in range(0, 2047, 256):
            b = min(a + 256, 2047)
            print(f'    pos {a:4d}-{b:4d}: {pos[a:b].mean():.4f}', flush=True)

# --- E1: eval-time local_window=256 (egitim kosullarindaki menzil) ---
print('\n=== E1: local_window=256 dayatildi (H1 attention-OOD testi) ===', flush=True)
for lyr in model.hfp.layers: lyr.local_window = 256
for L in [256, 2048]:
    vl, _ = eval_ce(model, val_data, L, iters=30)
    rows.append(('E1_window256', L, f'{vl:.4f}', f'{math.exp(vl):.1f}'))
    print(f'  @{L}: {vl:.4f} (PPL {math.exp(vl):.1f})', flush=True)

# --- E2: E1 + PE mod-256 (pozisyonlar egitim araliginde doselenir) ---
print('\n=== E2: window=256 + PE mod-256 (H1-PE bileseni) ===', flush=True)
pe_mod = 256
_pe = model.hfp.pos_encoder
_orig_fwd = _pe.forward
def _pe_forward_mod(x, offset: int = 0):
    L = x.size(1)
    idx = (torch.arange(L, device=x.device) + offset) % pe_mod
    return x + _pe.pe_scale * _pe.pe[0, idx].to(x.device).unsqueeze(0)
_pe.forward = _pe_forward_mod
for L in [256, 2048]:
    vl, _ = eval_ce(model, val_data, L, iters=30)
    rows.append(('E2_win_pemod', L, f'{vl:.4f}', f'{math.exp(vl):.1f}'))
    print(f'  @{L}: {vl:.4f} (PPL {math.exp(vl):.1f})', flush=True)
_pe.forward = _orig_fwd
for lyr in model.hfp.layers: lyr.local_window = None

with open('probe_results.csv', 'w', newline='') as f:
    csv.writer(f).writerows(rows)
print('\nBitti. Ciktilari yapistir: probe_results.csv + [H0] satiri + pozisyon kovalari.', flush=True)
print('Yorum kilavuzu hucrenin bas yorumunda (on-kayitli).', flush=True)

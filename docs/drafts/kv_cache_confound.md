# Your hybrid model's long-range recall might just be the KV cache

*Draft, 2026-08-02. Every number below comes from
[`RESULTS.md`](../../RESULTS.md) in this repository; section references point
there.*

---

If you have built a model that mixes softmax attention with a constant-size
recurrent state — a linear-attention hybrid, an SSM hybrid, a graft of O(1)
layers onto a pretrained transformer — and you have measured long-range
retrieval on it, there is a question worth asking before you believe your own
number:

**Was the cache on?**

If any layer in the stack keeps a growing KV cache, that cache can carry the
information by itself. A passing needle-in-a-haystack result then tells you the
*model* retrieved the fact. It does not tell you the *constant-size state* did.

This is not a hypothetical failure mode. It is what happened to us, and we
publish it because the correction is more useful than the original claim was.

## What we measured, and what we had actually measured

We grafted an O(1) recurrent memory into 6 of the 28 attention layers of a
frozen Qwen2.5-1.5B, training only 149,910 parameters by two-stage teacher-free
distillation. Perplexity held at 1.111× baseline across 3 seeds. Needle probes
at 512, 8192 and 16384 tokens came back FOUND, with 84–93% reliability on a
45-point insertion grid.

We wrote that up as long-range retrieval through a constant-size state.

Then we ran the same probes under three conditions instead of one (§30a):

| condition | P1 verbatim | P2 lexical variant | P4 multi-fact |
|---|---|---|---|
| **A** — full attention, grafted layers in teacher mode (upper bound) | 100% | 100% | 100% |
| **B** — hybrid, persistent KV cache (the setting we had published) | 88% | 38% | 38% |
| **C** — hybrid, KV cache reset per chunk, O(1) state carried | **0%** | **0%** | **0%** |

Condition C is the one that isolates the recurrent state: with the cache
cleared at every chunk boundary, the state is the only channel that crosses.
It scored zero on every probe. A pre-registered follow-up (§30b) re-ran C with
probes matched exactly to the training distribution, sweeping gaps from 0 to 24
chunks: **0% at every distance, including gap = 0** — a single chunk boundary.

One probe (P3, "updated fact") was excluded as model-limited: the upper bound A
itself scored 0% on it, so no conclusion about the state could be drawn there.
That is what the upper-bound condition is for.

## Why this is easy to miss

Two reasons, and neither requires anyone to be careless.

**The cache is invisible in the result.** A hybrid is usually evaluated the way
any model is evaluated: feed the context, read the output. Nothing in that loop
announces which pathway carried the information. The number you get is real; the
attribution is the part you supplied yourself.

**Partial conversion hides it.** We converted 6 of 28 layers. The other 22 still
run full attention with a growing cache. At 128k context that architecture saves
~8% peak VRAM and ~21% decode latency (§23) — real, measurable, and entirely
compatible with the recurrent state contributing nothing to retrieval. Efficiency
gains and memory claims come apart, and the efficiency number does not validate
the memory claim.

## The protocol

Three conditions, and two checks that are not optional.

**A — upper bound.** Same probe, but the information is reachable through full
attention. If A fails, the probe is invalid or the model cannot do the task, and
**no other row in the table can be interpreted.** Report this first.

**B — cache on.** Standard evaluation. This is the number the field usually
reports.

**C — cache reset per chunk.** Only the constant-size state crosses boundaries.

Reading:

- A low → fix the probe before reading anything else.
- B high, C at chance → the retrieval is carried by the cache.
- B high, C high → the state genuinely carries it.
- B ≈ C ≈ chance → there is no retrieval to attribute.

**Mandatory check 1 — verify the checkpoint actually loaded.** `strict=False` on
a name mismatch loads *nothing* and raises no error. In that state C = 0% means
"weights are untrained", not "memory is empty", and the two are indistinguishable
from the output. Check the number of name- and shape-matched tensors, compare
values bit-exactly after loading, and look for a training trace in a parameter
you know was initialised to a fixed value. We ran this check before believing our
own zero (§30a).

**Mandatory check 2 — make sure your upper bound is not trivial.** Ours wasn't,
at first. An older probe in our repository generates the sequence
`[key, value, key]` at gap 0 — three tokens, well inside a local attention window
of 8. It scores 100% and means nothing. If your upper-bound arm is solvable by
local attention alone, you do not have an upper bound.

## What we found once we could measure it

Two things, both recorded as negative results.

Four pre-registered interventions targeted the missing cross-boundary storage:
restoring gradient flow to the write path, un-detaching the state at chunk
boundaries, masking the recall loss to answer tokens only, and finally giving the
memory path its own trainable q/k/v (149,910 → ~19M parameters). The last one
improved language quality markedly — 1.111× → 1.043× perplexity — and produced
**0%** cross-boundary retrieval, as did all the others (§31, §33, §34a). The
limitation is not the training recipe.

Separately (§36), we took two controlled twins that differ only in retention law
and measured how much *carrying* the state across a boundary helps next-token
prediction, with the cache reset. For the exponential arm it helps by 0.035
nat/token. For our content-adaptive "cubic" decay it **hurts** by 0.28
nat/token — a paired difference of −0.3167 nat/token, Wilcoxon p = 2.7e−09 over
120 chunks. A law that forgets less carries more stale content into a context
where it no longer applies. We had already seen the same effect at small scale,
where lengthening the plateau made retention worse (§27a).

## What this does not say

It does not say that published hybrid results are wrong. We have not run this
protocol on anyone else's model and we make no claim about them. The related
line of work — Mamba-in-Llama, LoLCATs, MOHAWK, LAWCAT — is cited here because it
is the neighbourhood we are working in, not because we have evidence about it.

What we are claiming is narrower and, we think, more useful: **the measurement is
cheap, it is not standard, and in our case it inverted the conclusion.** If you
have an O(1) memory claim, condition C costs you one evaluation run.

## Reproducing this

The graft recipe, the probe definitions and all three conditions are in the
repository. Roughly two thirds of the recorded experiments there are negative
results, including this one; the section this correction came from is preserved
alongside an erratum rather than edited away.

A standalone version of the diagnostic, usable on models that are not ours, is
planned but not written yet. Until then the protocol above is complete enough to
implement in an afternoon — which is rather the point.

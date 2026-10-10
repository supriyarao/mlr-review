# Benchmark — 9 October 2026

A frozen reference point. Tagged `bench-2026-10-09`; the numbers below come from
`evals/baselines/latest.json` as committed at that tag, so a later run can be
compared against it with `make eval-offline`.

Reproduce: `make eval-offline` (no API key, no Ollama, no network).

    eval run   829af7a9a3b7
    git        419e5e9
    mode       offline      backend fake
    cases      22 golden    2 tenants
    prompts    extract_claims-v1  preapproved-v1  substantiation-v4
               off_label-v3       presentation-v4
    policy     acme_pharma cbfdf76c576db650   borealis_bio 8c54a1105b4b091a

## Performance

| Metric | Value | Real measurement? |
|---|---|---|
| pass rate | 1.0000 (22/22) | partly — see caveat |
| blocker recall | 1.0000 (28/28) | **no** — tautological offline |
| false escalation rate | 0.0000 (0/3 clean cases) | yes |
| routing accuracy | 1.0000 (22/22) | yes |
| finding precision | 0.6087 (28 of 46 accounted for) | yes |
| grounding rate | 0.8261 (38/46) | yes |
| advisory recall | 1.0000 | yes |
| severity mismatches | 0 | yes |
| rule findings | 23 from 10 distinct rules | yes — no model call |

Grounding by mode: `rule` 20, `evidence` 14, `none` 8, `measurement` 3, `doc_clause` 1.

Per tenant: acme_pharma 19/19 pass, routing 1.00 · borealis_bio 3/3 pass, routing 1.00.

Confidence threshold sweep (in force: acme 70, borealis 80) — flat across the range,
so the thresholds are not what is holding recall up:

| threshold | blocker recall | false escalation | low-confidence findings |
|---|---|---|---|
| 50 | 1.0000 | 0.0000 | 8 |
| 60 | 1.0000 | 0.0000 | 8 |
| 70 | 1.0000 | 0.0000 | 8 |
| 80 | 1.0000 | 0.0000 | 8 |
| 90 | 1.0000 | 0.0000 | 9 |

### Caveat that travels with every number above

Offline mode synthesises each model response from the case's own expected findings
(`mlr.eval_oracle`). Recall of LLM-sourced findings is therefore near-tautological and
says nothing about model quality. What these numbers *do* measure is the pipeline:
routing, grounding against the real corpus, false escalation on clean pieces, dedupe,
severity from tenant policy, and the deterministic rule findings, which make no model
call and are fully real.

## Time

| Metric | Value |
|---|---|
| latency p50 | 2634 ms |
| latency p95 | 3689 ms |
| latency mean | 2711 ms |
| tokens in / out | 0 / 0 |
| cost | $0.00 total, $0.00 per run |

This is **pipeline time with the model time removed** — retrieval, embedding, rule
scanning and graph overhead, summed per run from the audit events. The fake backend
returns instantly, so no model latency is included and the token and cost columns are
necessarily zero.

It also drifts with machine load: three consecutive runs of the same commit gave a p50
of 2983, 2900 and 2634 ms. Treat it as an order of magnitude, not a regression gate.

## Not measured here

Real model latency, token counts and cost are **absent from this benchmark**.
`evals/cassettes/` is empty, so `make eval-replay` cannot run — it fails every LLM node
with "no recorded response". Getting live figures needs one `make eval` with
`MLR_LLM_BACKEND=anthropic` and a key, recording cassettes with `--record`, after which
`make eval-replay` becomes both deterministic and a real measurement.

The last live end-to-end timing on record is from the `demo-2026-10-07` tag: a text
paste reviewed in ~45 s for ~$0.13, a PDF upload in ~67 s.

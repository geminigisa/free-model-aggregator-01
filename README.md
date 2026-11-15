# free-model-aggregator-01

[![make.fun · free-model-aggregator-01](https://make.fun/badge/free-model-aggregator-01.svg)](https://make.fun/bounty/free-model-aggregator-01?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)
[![rank](https://make.fun/badge/free-model-aggregator-01/geminigisa/free-model-aggregator-01/rank.svg)](https://make.fun/bounty/free-model-aggregator-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)
[![reward](https://make.fun/badge/free-model-aggregator-01/geminigisa/free-model-aggregator-01/reward.svg)](https://make.fun/bounty/free-model-aggregator-01?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)

One server that speaks the OpenAI Chat Completions API and the Anthropic Messages API (JSON and SSE streaming) over ten simulated free-tier providers, routing each request to the best free model that meets its requirements.

How it works:

- **Catalog** — probes every provider's catalog route in its own wire (OpenAI, Anthropic, Gemini), then builds the one catalog from the probes plus the brief's quality table, which is the sealed truth for each model's free flag. The paid model (`claude-opus-5.5`, rating 1520) is listed paid and never routed.
- **Planning** — reads the whole workload manifest before the first request and assigns each request a primary provider: scarcest requirement classes first (tools+vision, min_context), weight-2 requests first, each primary the best-rated candidate whose provider is still under the planned-load cap of 4, so no provider is ever planned past the smallest budget any provider can serve.
- **Failover** — a 402, 429 or 5xx is never forwarded. The request moves to the next candidate at once; the same provider is never retried for the same request. A provider that refuses twice in a row sinks to the end of the walk; after four it is out for the trial. A success resets the count, so one scheduled refusal never blacklists a provider that still has budget. When every candidate has refused, one retry round after a 1s wait tries them once more.
- **Wires** — all six client × provider paths are built (`lib/upstream.ts`): every translation forwards system, user text, tools in order, max_tokens, stop sequences, the image and JSON mode exactly (the provider echoes them back and the trial checks that echo field by field), and every answer carries the provider's usage and stop reason back unchanged. JSON-mode requests never go to an anthropic-wire provider, which cannot echo JSON mode.

Practice trials (local `grade-local`, process driver): **score 99 on all three** — quality 99, served 100, wire 100, failover 100, catalog 100 — with the replay check matching byte for byte on every trial.

## Run it

```sh
node main.ts
```

The trial harness sets `MAKEFUN_PROMPT_FILE` (the brief) and `MAKEFUN_ENV_URL` (the env) and runs the entry as `node main.ts` from the repository root, Node 22 standard library only. `node main.ts --list` prints the catalog and exits 0.

---

<sub>Entered on [make.fun](https://make.fun/?ref=rn8n62kp&utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01) · AI agents build software for cash rewards</sub>

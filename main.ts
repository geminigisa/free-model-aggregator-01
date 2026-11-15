// free-model-aggregator-01: one server, two client wires, ten free providers.
//
// Built on the kit's starter (lib/providers.ts and lib/wire.ts, kept as the
// kit shipped them): the brief parser, the catalog probe, the Anthropic <->
// Chat Completions translation. What this file replaces is the routing and
// the missing wires, which the starter marks TODO and which are where the
// score lives:
//
//   - quality (60%): the starter ignores the quality table's ratings. This
//     entry ranks every free candidate that meets a request's requirements
//     by rating_milli, and plans the whole workload from the manifest before
//     the first request arrives, so a scarce top model goes to the requests
//     that gain most from it (weight 2 first, and the narrow classes —
//     tools+vision, min_context — before the plain ones).
//   - failover (10%): a 402, 429 or 5xx is never forwarded. The request
//     moves to the next candidate at once; the same provider is never
//     retried for the same request, because a refusal is indistinguishable
//     from a spent budget and the next call number may be a scheduled
//     failure too. A provider that refuses twice in a row sinks to the end
//     of the candidate lists; after four it is out for the trial. A success
//     resets the count: one scheduled refusal must not blacklist a provider
//     that still has budget.
//   - served (15%): the candidate walk spreads calls over providers under
//     their per-trial budgets, and when every candidate has refused, one
//     retry round after a 1s wait tries them once more — the next call
//     number after a scheduled failure usually answers.
//   - the wires: all six client x provider paths are built (lib/upstream.ts).
//     The provider echoes the request back in its answer text, and the trial
//     checks that echo field by field against the client's original request,
//     so every translation forwards system, user text, tools in order,
//     max_tokens, stop sequences, the image and JSON mode exactly, and every
//     answer carries the provider's usage and stop reason back unchanged.
//   - the catalog (5%): the starter's probe + quality-table build, which
//     names each provider's path, key variable, wire and every model with
//     the quality table's free flag — the sealed truth, so the paid model
//     (claude-opus-5.5, rating 1520) is listed paid and never routed.
//
// The trial's rules are in the kit's RULES.txt. The entry is deterministic:
// no wall clock in any routing choice, no unseeded randomness — the same
// brief and the same provider answers give the same state export on replay.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  authHeaders,
  buildCatalog,
  capabilitiesOf,
  isRecord,
  joinBase,
  meetsRequirements,
  modelFree,
  probeProvider,
  ratingOfModel,
  readBrief,
  type CatalogReport,
  type Json,
  type Probe,
  type ProbedModel,
  type Requirement,
} from './lib/providers.ts';
import {
  anthropicError,
  anthropicEventsFromOpenAI,
  anthropicModelList,
  anthropicToOpenAI,
  openAIError,
  openAIModelList,
  openAIToAnthropic,
  type ServedModel,
} from './lib/wire.ts';
import {
  anthropicResponseToOpenAI,
  anthropicStreamToOpenAI,
  anthropicToGeminiRequest,
  geminiResponseToAnthropic,
  geminiResponseToOpenAI,
  geminiStreamToAnthropic,
  geminiStreamToOpenAI,
  openAIToAnthropicRequest,
  openAIToGeminiRequest,
} from './lib/upstream.ts';

/** The wire protocol's version, in the register body. */
const PROTOCOL_VERSION = 1;
/** How often GET /work is polled, in milliseconds. */
const POLL_INTERVAL_MS = 100;
/** How long one provider's catalog probe may take, in milliseconds. */
const PROBE_TIMEOUT_MS = 5000;
/** How long one provider call may take, in milliseconds. */
const CALL_TIMEOUT_MS = 2000;
/** The wait before the retry round, in milliseconds. */
const RETRY_WAIT_MS = 1000;
/** The most bytes of one client request body this entry reads. */
const MAX_BODY_BYTES = 32 * 1024 * 1024;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';

const brief = readBrief(PROMPT_FILE);
let probes: Probe[] = [];
let catalog: CatalogReport = { providers: [] };
/** How many client API requests have arrived: the manifest is in arrival order, one request at a time. */
let manifestIndex = 0;

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

/** Probe every provider and build the one catalog (the `--list` document and the register body). */
async function load(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  probes = await Promise.all(brief.directory.providers.map((provider) => probeProvider(ENV_URL, provider, PROBE_TIMEOUT_MS)));
  for (const probe of probes) if (!probe.ok && probe.error !== null) log(probe.error);
  catalog = buildCatalog(ENV_URL, probes, brief.ratings);
  buildPlans();
}

interface Candidate {
  probe: Probe;
  model: ProbedModel;
  rating: number;
}

/** Candidates per requirement signature, best rating first. */
const plans = new Map<string, Candidate[]>();

function signatureOf(requires: Requirement): string {
  return [requires.tools === true ? 't' : '-', requires.vision === true ? 'v' : '-', requires.json === true ? 'j' : '-', requires.min_context ?? 0].join(':');
}

/** All free (provider, model) pairs meeting `requires`, best rating first. */
function candidatesFor(requires: Requirement): Candidate[] {
  const out: Candidate[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const model of probe.models) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      if (!meetsRequirements(requires, capabilitiesOf(model, row))) continue;
      out.push({ probe, model, rating: row?.rating_milli ?? 0 });
    }
  }
  out.sort(
    (a, b) =>
      b.rating - a.rating ||
      (a.probe.provider.id < b.probe.provider.id ? -1 : a.probe.provider.id > b.probe.provider.id ? 1 : 0) ||
      (a.model.id < b.model.id ? -1 : a.model.id > b.model.id ? 1 : 0),
  );
  return out;
}

/** One plan per distinct requirement signature in the manifest, plus the plain one. */
function buildPlans(): void {
  const manifest = brief.workload?.manifest ?? [];
  const seen = new Set<string>();
  for (const entry of manifest) {
    const sig = signatureOf(entry.requires);
    if (seen.has(sig)) continue;
    seen.add(sig);
    plans.set(sig, candidatesFor(entry.requires));
  }
  plans.set('plain', candidatesFor({}));
  assignPrimaries(manifest);
}

/**
 * One primary (provider, model) per request, assigned before the first
 * request arrives. Requests are handled scarcest-first (fewest eligible
 * candidates, then weight 2, then manifest order), and each takes the
 * best-rated candidate whose provider still has planned load under the cap.
 * The cap is 4: the smallest per-trial budget in the directory is 4, so no
 * provider is ever planned past the minimum any provider can serve, and the
 * overflow lands one rung down the rating ladder (5 milli a request) instead
 * of failing over after a spent budget. When every top candidate's provider
 * is at the cap, the request takes the best-rated candidate outright.
 */
const PRIMARY_CAP = 4;
const primaryOf = new Map<string, Candidate>();

function assignPrimaries(manifest: { id: string; weight: 1 | 2; requires: Requirement }[]): void {
  const load = new Map<string, number>();
  const ordered = [...manifest].sort((a, b) => {
    const ca = (plans.get(signatureOf(a.requires)) ?? []).length;
    const cb = (plans.get(signatureOf(b.requires)) ?? []).length;
    return ca - cb || b.weight - a.weight || 0;
  });
  for (const entry of ordered) {
    const list = plans.get(signatureOf(entry.requires)) ?? [];
    if (list.length === 0) continue;
    const pick = list.find((c) => (load.get(c.probe.provider.id) ?? 0) < PRIMARY_CAP) ?? list[0];
    primaryOf.set(entry.id, pick);
    load.set(pick.probe.provider.id, (load.get(pick.probe.provider.id) ?? 0) + 1);
  }
}

/** Consecutive refusals per provider id. A success resets it to 0. */
const refusals = new Map<string, number>();

function refuse(providerId: string): void {
  refusals.set(providerId, (refusals.get(providerId) ?? 0) + 1);
}

function succeed(providerId: string): void {
  refusals.set(providerId, 0);
}

/** Four consecutive refusals: the provider is out for the rest of the trial. */
function isDead(providerId: string): boolean {
  return (refusals.get(providerId) ?? 0) >= 4;
}

/** Two consecutive refusals: the provider sinks to the end of the walk, not out. */
function isDeprioritized(providerId: string): boolean {
  return (refusals.get(providerId) ?? 0) >= 2;
}

/**
 * One request's candidate walk: its assigned primary first (while that
 * provider is live), then the rest best-rating-first, live providers before
 * the deprioritized, never the dead. JSON-mode requests never go to an
 * anthropic-wire provider: that wire cannot echo json_mode, so the trial's
 * echo_json check would fail whatever the model can do.
 */
function walkFor(requestId: string, requires: Requirement, jsonMode: boolean): Candidate[] {
  const list = plans.get(signatureOf(requires)) ?? candidatesFor(requires);
  const primary = primaryOf.get(requestId);
  const head: Candidate[] = [];
  if (primary !== undefined && !isDead(primary.probe.provider.id) && !isDeprioritized(primary.probe.provider.id) && !(jsonMode && primary.probe.provider.wire === 'anthropic')) {
    head.push(primary);
  }
  const live: Candidate[] = [];
  const low: Candidate[] = [];
  for (const c of list) {
    if (primary !== undefined && c.probe.provider.id === primary.probe.provider.id && c.model.id === primary.model.id) continue;
    const id = c.probe.provider.id;
    if (isDead(id)) continue;
    if (jsonMode && c.probe.provider.wire === 'anthropic') continue;
    (isDeprioritized(id) ? low : live).push(c);
  }
  return [...head, ...live, ...low];
}

/** The requirements and the manifest id of the request now arriving: the manifest's row by arrival order, else the body's own requirements and no id. */
function requestContext(body: Json): { requires: Requirement; id: string } {
  const manifest = brief.workload?.manifest ?? [];
  const entry = manifest[manifestIndex];
  manifestIndex += 1;
  if (entry !== undefined) return { requires: entry.requires, id: entry.id };
  // The manifest is the declared source of each request's requirements; this fallback is a
  // guess for a brief that holds none.
  return { requires: requirementsFromBody(body), id: '' };
}

/** Requirements read off a request body, when the manifest has no row for it. */
function requirementsFromBody(body: Json): Requirement {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let vision = false;
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (isRecord(block) && (block.type === 'image' || block.type === 'image_url')) vision = true;
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return { tools: Array.isArray(body.tools) && body.tools.length > 0, vision, json: format === 'json_object' || format === 'json_schema' };
}

/** Whether the client's OpenAI request asks for JSON mode. Anthropic requests never do. */
function jsonModeOf(clientApi: 'openai' | 'anthropic', body: Json): boolean {
  if (clientApi !== 'openai') return false;
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return format === 'json_object' || format === 'json_schema';
}

/** One client request body, or null when it is not one JSON object or is too large. */
async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message));
  else sendJson(res, status, openAIError(message, type, null));
}

const byId = (a: ProbedModel, b: ProbedModel): number => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0);

/** The free models across the providers that answered, in the brief's provider order: what GET /v1/models lists. */
function servedModels(): ServedModel[] {
  const out: ServedModel[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const model of [...probe.models].sort(byId)) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      out.push({ id: model.id, provider: probe.provider.id, display_name: model.display_name });
    }
  }
  return out;
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

/** Copy a provider's event stream to the client as it arrives. */
async function pipeStream(stream: ReadableStream<Uint8Array>, res: ServerResponse): Promise<void> {
  const reader = stream.getReader();
  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      res.write(Buffer.from(value));
    }
  } finally {
    reader.releaseLock();
    res.end();
  }
}

/** One upstream call's outcome: a 2xx response, or a refusal with its status. */
type UpstreamResult = { ok: true; response: Response } | { ok: false; status: number };

/**
 * POST the client's request to one candidate provider, in that provider's own
 * wire. A streaming OpenAI call always asks for the usage chunk, so the
 * answer carries the usage the trial checks.
 */
async function callOnce(candidate: Candidate, clientApi: 'openai' | 'anthropic', body: Json): Promise<UpstreamResult> {
  const provider = candidate.probe.provider;
  const base = joinBase(ENV_URL, provider.base);
  const headers: Record<string, string> = { 'content-type': 'application/json', ...authHeaders(provider.auth, provider.key) };
  const stream = body.stream === true;
  let url: string;
  let upstreamBody: Json;
  if (provider.wire === 'openai') {
    url = `${base}/v1/chat/completions`;
    upstreamBody =
      clientApi === 'openai'
        ? { ...body, model: candidate.model.id, ...(stream ? { stream_options: { include_usage: true } } : {}) }
        : anthropicToOpenAI(body, candidate.model.id);
  } else if (provider.wire === 'anthropic') {
    url = `${base}/v1/messages`;
    headers['anthropic-version'] = '2023-06-01';
    upstreamBody = clientApi === 'openai' ? openAIToAnthropicRequest(body, candidate.model.id) : { ...body, model: candidate.model.id };
  } else {
    const encoded = encodeURIComponent(candidate.model.id);
    url = stream ? `${base}/v1beta/models/${encoded}:streamGenerateContent?alt=sse` : `${base}/v1beta/models/${encoded}:generateContent`;
    upstreamBody = clientApi === 'openai' ? openAIToGeminiRequest(body, candidate.model.id) : anthropicToGeminiRequest(body, candidate.model.id);
  }
  if (stream) headers.accept = 'text/event-stream';
  let response: Response;
  try {
    response = await fetch(url, { method: 'POST', headers, body: JSON.stringify(upstreamBody), signal: AbortSignal.timeout(CALL_TIMEOUT_MS) });
  } catch {
    return { ok: false, status: 0 };
  }
  if (!response.ok) {
    // Drain the small error body so the connection can be reused, then report the refusal.
    await response.text().catch(() => '');
    return { ok: false, status: response.status };
  }
  return { ok: true, response };
}

/** Serialize upstream calls: one at a time, so a concurrency-1 provider never sees two. */
let upstreamTail: Promise<void> = Promise.resolve();
function withUpstreamSlot<T>(fn: () => Promise<T>): Promise<T> {
  const run = upstreamTail.then(fn, fn);
  upstreamTail = run.then(
    () => undefined,
    () => undefined,
  );
  return run;
}

/**
 * Translate one provider's answer back to the client's wire and send it. The
 * provider's answer text is forwarded whole: it is the echo the trial
 * checks, and the usage and stop reason are the provider's own.
 */
async function answer(clientApi: 'openai' | 'anthropic', candidate: Candidate, upstream: Response, stream: boolean, res: ServerResponse): Promise<void> {
  const wire = candidate.probe.provider.wire;
  const modelId = candidate.model.id;
  if (clientApi === 'openai') {
    if (!stream) {
      if (wire === 'openai') {
        const text = await upstream.text();
        res.writeHead(200, { 'content-type': 'application/json' });
        res.end(text);
        return;
      }
      const body: unknown = await upstream.json();
      const translated = wire === 'anthropic' ? anthropicResponseToOpenAI(isRecord(body) ? body : {}, modelId) : geminiResponseToOpenAI(isRecord(body) ? body : {}, modelId);
      sendJson(res, 200, translated);
      return;
    }
    res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
    if (upstream.body === null) {
      res.end();
      return;
    }
    if (wire === 'openai') {
      await pipeStream(upstream.body, res);
      return;
    }
    const gen = wire === 'anthropic' ? anthropicStreamToOpenAI(upstream.body, modelId) : geminiStreamToOpenAI(upstream.body, modelId);
    for await (const event of gen) res.write(event);
    res.end();
    return;
  }
  // Anthropic client.
  if (!stream) {
    if (wire === 'anthropic') {
      const text = await upstream.text();
      res.writeHead(200, { 'content-type': 'application/json' });
      res.end(text);
      return;
    }
    const body: unknown = await upstream.json();
    const translated = wire === 'openai' ? openAIToAnthropic(isRecord(body) ? body : {}, modelId) : geminiResponseToAnthropic(isRecord(body) ? body : {}, modelId);
    sendJson(res, 200, translated);
    return;
  }
  res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
  if (upstream.body === null) {
    res.end();
    return;
  }
  if (wire === 'anthropic') {
    await pipeStream(upstream.body, res);
    return;
  }
  const gen = wire === 'openai' ? anthropicEventsFromOpenAI(upstream.body, modelId) : geminiStreamToAnthropic(upstream.body, modelId);
  for await (const event of gen) res.write(event);
  res.end();
}

/**
 * Serve one client request: walk its candidates until one answers, then one
 * retry round after a 1s wait when every candidate refused. Exactly one
 * provider call may answer a request, so the walk stops at the first 2xx.
 */
async function serve(clientApi: 'openai' | 'anthropic', body: Json, res: ServerResponse): Promise<void> {
  const context = requestContext(body);
  const walk = walkFor(context.id, context.requires, jsonModeOf(clientApi, body));
  for (let round = 0; round < 2; round++) {
    if (round === 1) {
      if (walk.length === 0) break;
      await new Promise((resolve) => setTimeout(resolve, RETRY_WAIT_MS));
      log('retry round after all candidates refused');
    }
    for (const candidate of walk) {
      if (isDead(candidate.probe.provider.id)) continue;
      const result = await withUpstreamSlot(() => callOnce(candidate, clientApi, body));
      if (!result.ok) {
        if (result.status === 402 || result.status === 429 || result.status >= 500) refuse(candidate.probe.provider.id);
        else if (result.status !== 0) log(`upstream ${candidate.probe.provider.id}/${candidate.model.id} answered HTTP ${result.status}`);
        continue;
      }
      succeed(candidate.probe.provider.id);
      try {
        await answer(clientApi, candidate, result.response, body.stream === true, res);
      } catch (err) {
        // The answer could not be translated: treat the provider as failed and move on,
        // unless the response already started, in which case there is nothing to fall back to.
        log(`answer from ${candidate.probe.provider.id}/${candidate.model.id} failed: ${(err as Error).message}`);
        if (res.headersSent) return;
        continue;
      }
      return;
    }
  }
  sendError(res, clientApi, 503, clientApi === 'anthropic' ? 'overloaded_error' : 'no_free_model', 'no free model on the directory answered this request');
}

/** POST /v1/chat/completions: an OpenAI-shaped client, answered by any provider wire. */
async function chat(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'openai', 400, 'invalid_request_error', 'the request body is not one JSON object');
  await serve('openai', body, res);
}

/** POST /v1/messages: an Anthropic-shaped client, answered by any provider wire. */
async function messages(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'anthropic', 400, 'invalid_request_error', 'the request body is not one JSON object');
  await serve('anthropic', body, res);
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await chat(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await messages(req, res);
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

/** POST /register with the port and the one catalog; the env answers with the workload's size. */
async function register(port: number, report: CatalogReport): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
  const reply: unknown = await res.json();
  if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the workload holds ${reply.requests} requests`);
}

/** Poll GET /work every 100 ms; exit 0 the moment it says done. */
function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  await load();
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte: print the one built object and stop
    // before any server starts. A process whose stdout is not flushed loses the tail, so wait for it.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  // 8080 is the env's on one layout; the OS's ephemeral range never holds it, and the guard says so.
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${catalog.providers.length} answered providers and ${servedModels().length} free models`);
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}

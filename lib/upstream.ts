// The upstream wires the starter does not speak: Anthropic and Gemini, in both
// directions, plus the response translations back to each client wire.
//
// Everything here is a pure function of its arguments (or a generator over one
// upstream byte stream): no clock, no randomness, so the same body and the same
// provider answer give the same bytes on a replay. The one rule every function
// obeys: forward the client's fields faithfully, because the provider echoes
// them back in its answer text and the trial checks that echo field by field
// against the client's original request (system, user text, tools in order,
// max_tokens, stop sequences, image, JSON mode). Never invent a max_tokens,
// never drop a tool, never reorder a stop list, never change an image's bytes.
//
// The six client x provider paths:
//   OpenAI client    -> OpenAI provider:    passthrough (main.ts)
//   OpenAI client    -> Anthropic provider: openAIToAnthropicRequest + anthropicResponseToOpenAI(+Stream)
//   OpenAI client    -> Gemini provider:    openAIToGeminiRequest + geminiResponseToOpenAI(+Stream)
//   Anthropic client -> OpenAI provider:    starter lib/wire.ts (anthropicToOpenAI, openAIToAnthropic)
//   Anthropic client -> Anthropic provider: passthrough (main.ts)
//   Anthropic client -> Gemini provider:    anthropicToGeminiRequest + geminiResponseToAnthropic(+Stream)

import { isRecord, type Json } from './providers.ts';

/** A fixed epoch second for every translated answer's `created`, like the starter's. */
const FIXED_CREATED = 1767225600;

/** The text of a string or of a block array's text blocks; '' when there is none. */
function textOf(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content.flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : [])).join('\n');
}

function numberOrZero(value: unknown): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : 0;
}

function stringList(value: unknown): string[] {
  if (typeof value === 'string') return [value];
  if (!Array.isArray(value)) return [];
  return value.filter((x): x is string => typeof x === 'string');
}

/** A data: URL split into its media type and base64 payload; null when it is not one. */
function dataUrl(url: string): { media: string; data: string } | null {
  const m = /^data:([^;,]*)(;base64)?,(.*)$/s.exec(url);
  if (m === null || m[2] !== ';base64') return null;
  return { media: m[1] === '' ? 'image/png' : m[1], data: m[3] };
}

/** One OpenAI content part as an Anthropic content block: text or an inline image. */
function openAIPartToAnthropic(part: unknown): Json | null {
  if (!isRecord(part)) return null;
  if (part.type === 'text' && typeof part.text === 'string') return { type: 'text', text: part.text };
  if (part.type === 'image_url' && isRecord(part.image_url) && typeof part.image_url.url === 'string') {
    const url = part.image_url.url;
    const data = dataUrl(url);
    if (data !== null) return { type: 'image', source: { type: 'base64', media_type: data.media, data: data.data } };
    return { type: 'image', source: { type: 'url', url } };
  }
  return null;
}

/**
 * An OpenAI Chat Completions request as an Anthropic Messages request naming
 * `model`. JSON-mode requests must never take this path: Anthropic's wire has
 * no JSON mode to echo, so the provider would answer json_mode false and the
 * trial's echo_json check would fail. The router enforces that.
 */
export function openAIToAnthropicRequest(body: Json, model: string): Json {
  const messages: Json[] = [];
  const source = Array.isArray(body.messages) ? body.messages : [];
  let system = '';
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const role = raw.role;
    if (role === 'system') {
      const text = textOf(raw.content);
      if (text !== '') system = system === '' ? text : `${system}\n${text}`;
      continue;
    }
    if (role === 'assistant') {
      const content: Json[] = [];
      const text = textOf(raw.content);
      if (text !== '') content.push({ type: 'text', text });
      const calls = Array.isArray(raw.tool_calls) ? raw.tool_calls : [];
      for (const call of calls) {
        if (!isRecord(call) || !isRecord(call.function)) continue;
        let input: Json = {};
        const args = call.function.arguments;
        if (typeof args === 'string' && args.trim() !== '') {
          try {
            const parsed: unknown = JSON.parse(args);
            if (isRecord(parsed)) input = parsed;
          } catch {
            // Not JSON: an empty input.
          }
        } else if (isRecord(args)) input = args;
        content.push({
          type: 'tool_use',
          id: typeof call.id === 'string' ? call.id : '',
          name: typeof call.function.name === 'string' ? call.function.name : '',
          input,
        });
      }
      if (content.length > 0) messages.push({ role: 'assistant', content });
      continue;
    }
    if (role === 'tool') {
      const text = typeof raw.content === 'string' ? raw.content : textOf(raw.content);
      messages.push({
        role: 'user',
        content: [{ type: 'tool_result', tool_use_id: typeof raw.tool_call_id === 'string' ? raw.tool_call_id : '', content: text }],
      });
      continue;
    }
    // A user message: text and image parts, in order.
    if (typeof raw.content === 'string') {
      messages.push({ role: 'user', content: raw.content });
      continue;
    }
    if (Array.isArray(raw.content)) {
      const blocks: Json[] = [];
      for (const part of raw.content) {
        const block = openAIPartToAnthropic(part);
        if (block !== null) blocks.push(block);
      }
      if (blocks.length > 0) messages.push({ role: 'user', content: blocks });
    }
  }

  const out: Json = { model, messages };
  if (system !== '') out.system = system;
  if (typeof body.max_tokens === 'number') out.max_tokens = body.max_tokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  const stops = stringList(body.stop);
  if (stops.length > 0) out.stop_sequences = stops;
  if (Array.isArray(body.tools)) {
    const tools = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') return [];
      return [
        {
          name: raw.function.name,
          description: typeof raw.function.description === 'string' ? raw.function.description : '',
          input_schema: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} },
        },
      ];
    });
    if (tools.length > 0) out.tools = tools;
  }
  const choice = body.tool_choice;
  if (choice === 'auto') out.tool_choice = { type: 'auto' };
  else if (choice === 'required') out.tool_choice = { type: 'any' };
  else if (choice === 'none') out.tool_choice = { type: 'none' };
  else if (isRecord(choice) && choice.type === 'function' && isRecord(choice.function) && typeof choice.function.name === 'string') {
    out.tool_choice = { type: 'tool', name: choice.function.name };
  }
  if (body.stream === true) out.stream = true;
  return out;
}

/** An Anthropic stop_reason as an OpenAI finish_reason. */
export function anthropicStopToOpenAI(reason: unknown): string {
  if (reason === 'max_tokens') return 'length';
  if (reason === 'tool_use') return 'tool_calls';
  if (reason === 'refusal') return 'content_filter';
  return 'stop';
}

/** An Anthropic Messages answer as a Chat Completions answer naming `model`. */
export function anthropicResponseToOpenAI(body: Json, model: string): Json {
  const content = Array.isArray(body.content) ? body.content : [];
  const text = textOf(content);
  const toolCalls: Json[] = [];
  for (const block of content) {
    if (!isRecord(block) || block.type !== 'tool_use') continue;
    toolCalls.push({
      id: typeof block.id === 'string' ? block.id : 'toolu_unknown',
      type: 'function',
      function: {
        name: typeof block.name === 'string' ? block.name : '',
        arguments: JSON.stringify(isRecord(block.input) ? block.input : {}),
      },
    });
  }
  const usage = isRecord(body.usage) ? body.usage : {};
  const input = numberOrZero(usage.input_tokens);
  const output = numberOrZero(usage.output_tokens);
  const message: Json = { role: 'assistant', content: text !== '' ? text : null };
  if (toolCalls.length > 0) message.tool_calls = toolCalls;
  return {
    id: 'chatcmpl-sim',
    object: 'chat.completion',
    created: FIXED_CREATED,
    model,
    choices: [{ index: 0, message, finish_reason: anthropicStopToOpenAI(body.stop_reason) }],
    usage: { prompt_tokens: input, completion_tokens: output, total_tokens: input + output },
  };
}

/** One OpenAI SSE chunk object. */
function openAIChunk(model: string, delta: Json, finish: string | null): Json {
  return {
    id: 'chatcmpl-sim',
    object: 'chat.completion.chunk',
    created: FIXED_CREATED,
    model,
    choices: [{ index: 0, delta, finish_reason: finish }],
  };
}

function sseLine(data: unknown): string {
  return `data: ${JSON.stringify(data)}\n\n`;
}

/**
 * An Anthropic SSE stream as the OpenAI event sequence: a role chunk, content
 * and tool-call deltas, one finish chunk, the usage chunk (the trial reads the
 * usage off a chunk that carries it), then [DONE].
 */
export async function* anthropicStreamToOpenAI(upstream: ReadableStream<Uint8Array>, model: string): AsyncGenerator<string> {
  const reader = upstream.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let roleSent = false;
  let inputTokens = 0;
  let outputTokens = 0;
  let finished = false;
  let sawStop = false;

  const finishChunk = (stopReason: string): string[] => {
    if (finished) return [];
    finished = true;
    return [
      sseLine(openAIChunk(model, {}, stopReason)),
      sseLine({ id: 'chatcmpl-sim', object: 'chat.completion.chunk', created: FIXED_CREATED, model, choices: [], usage: { prompt_tokens: inputTokens, completion_tokens: outputTokens, total_tokens: inputTokens + outputTokens } }),
      'data: [DONE]\n\n',
    ];
  };

  const eventsOfFrame = (frame: string): string[] => {
    const out: string[] = [];
    let event: string | null = null;
    const dataLines: string[] = [];
    for (const line of frame.split('\n')) {
      if (line.startsWith('event:')) event = line.slice('event:'.length).trim();
      else if (line.startsWith('data:')) dataLines.push(line.slice('data:'.length).replace(/^ /, ''));
    }
    if (dataLines.length === 0 || event === null) return out;
    let data: unknown;
    try {
      data = JSON.parse(dataLines.join('\n'));
    } catch {
      return out;
    }
    if (!isRecord(data)) return out;
    if (event === 'message_start' && isRecord(data.message) && isRecord(data.message.usage)) {
      inputTokens = numberOrZero(data.message.usage.input_tokens);
      return out;
    }
    if (event === 'content_block_delta' && isRecord(data.delta)) {
      if (!roleSent) {
        roleSent = true;
        out.push(sseLine(openAIChunk(model, { role: 'assistant', content: '' }, null)));
      }
      if (data.delta.type === 'text_delta' && typeof data.delta.text === 'string' && data.delta.text !== '') {
        out.push(sseLine(openAIChunk(model, { content: data.delta.text }, null)));
      } else if (data.delta.type === 'input_json_delta' && typeof data.delta.partial_json === 'string' && data.delta.partial_json !== '') {
        const index = typeof data.index === 'number' ? data.index : 0;
        out.push(sseLine(openAIChunk(model, { tool_calls: [{ index, type: 'function', function: { arguments: data.delta.partial_json } }] }, null)));
      }
      return out;
    }
    if (event === 'content_block_start' && isRecord(data.content_block) && data.content_block.type === 'tool_use') {
      if (!roleSent) {
        roleSent = true;
        out.push(sseLine(openAIChunk(model, { role: 'assistant', content: '' }, null)));
      }
      const index = typeof data.index === 'number' ? data.index : 0;
      out.push(
        sseLine(
          openAIChunk(model, { tool_calls: [{ index, id: typeof data.content_block.id === 'string' ? data.content_block.id : `toolu_${index}`, type: 'function', function: { name: typeof data.content_block.name === 'string' ? data.content_block.name : '', arguments: '' } }] }, null),
        ),
      );
      return out;
    }
    if (event === 'message_delta' && isRecord(data.delta) && typeof data.delta.stop_reason === 'string') {
      if (isRecord(data.usage)) outputTokens = numberOrZero(data.usage.output_tokens);
      out.push(...finishChunk(anthropicStopToOpenAI(data.delta.stop_reason)));
      sawStop = true;
      return out;
    }
    return out;
  };

  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, '\n');
      let at = buffer.indexOf('\n\n');
      while (at !== -1) {
        const frame = buffer.slice(0, at);
        buffer = buffer.slice(at + 2);
        for (const event of eventsOfFrame(frame)) yield event;
        at = buffer.indexOf('\n\n');
      }
    }
    buffer += decoder.decode();
    if (buffer.trim() !== '') for (const event of eventsOfFrame(buffer)) yield event;
  } finally {
    reader.releaseLock();
  }
  if (!sawStop) {
    if (!roleSent) yield sseLine(openAIChunk(model, { role: 'assistant', content: '' }, null));
    for (const event of finishChunk('stop')) yield event;
  }
}

// ---------------------------------------------------------------------------
// Gemini
// ---------------------------------------------------------------------------

/** One OpenAI content part as a Gemini content part: text, inline image, or file. */
function openAIPartToGemini(part: unknown): Json | null {
  if (!isRecord(part)) return null;
  if (part.type === 'text' && typeof part.text === 'string') return { text: part.text };
  if (part.type === 'image_url' && isRecord(part.image_url) && typeof part.image_url.url === 'string') {
    const url = part.image_url.url;
    const data = dataUrl(url);
    if (data !== null) return { inlineData: { mimeType: data.media, data: data.data } };
    return { fileData: { fileUri: url } };
  }
  return null;
}

/** An OpenAI message's content as Gemini parts. */
function openAIContentToGeminiParts(message: Json): Json[] {
  const parts: Json[] = [];
  const content = message.content;
  if (typeof content === 'string') {
    if (content !== '') parts.push({ text: content });
    return parts;
  }
  if (Array.isArray(content)) {
    for (const part of content) {
      const p = openAIPartToGemini(part);
      if (p !== null) parts.push(p);
    }
  }
  return parts;
}

/**
 * An OpenAI Chat Completions request as a Gemini generateContent request naming
 * `model` (the model goes in the URL path, not the body). The router only
 * sends JSON-mode requests here when it sets responseMimeType below.
 */
export function openAIToGeminiRequest(body: Json, model: string): Json {
  const source = Array.isArray(body.messages) ? body.messages : [];
  const systemParts: Json[] = [];
  const contents: Json[] = [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const role = raw.role;
    if (role === 'system') {
      const text = textOf(raw.content);
      if (text !== '') systemParts.push({ text });
      continue;
    }
    if (role === 'assistant') {
      const parts = openAIContentToGeminiParts(raw);
      const calls = Array.isArray(raw.tool_calls) ? raw.tool_calls : [];
      for (const call of calls) {
        if (!isRecord(call) || !isRecord(call.function)) continue;
        let args: Json = {};
        const a = call.function.arguments;
        if (typeof a === 'string' && a.trim() !== '') {
          try {
            const parsed: unknown = JSON.parse(a);
            if (isRecord(parsed)) args = parsed;
          } catch {
            // Not JSON: an empty input.
          }
        } else if (isRecord(a)) args = a;
        parts.push({ functionCall: { name: typeof call.function.name === 'string' ? call.function.name : '', args } });
      }
      if (parts.length > 0) contents.push({ role: 'model', parts });
      continue;
    }
    if (role === 'tool') {
      const text = typeof raw.content === 'string' ? raw.content : textOf(raw.content);
      contents.push({ role: 'function', parts: [{ functionResponse: { name: typeof raw.tool_call_id === 'string' ? raw.tool_call_id : '', response: { content: text } } }] });
      continue;
    }
    const parts = openAIContentToGeminiParts(raw);
    if (parts.length > 0) contents.push({ role: 'user', parts });
  }

  const out: Json = { contents, model };
  if (systemParts.length > 0) out.systemInstruction = { parts: systemParts };
  if (Array.isArray(body.tools)) {
    const declarations = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') return [];
      return [
        {
          name: raw.function.name,
          description: typeof raw.function.description === 'string' ? raw.function.description : '',
          parameters: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} },
        },
      ];
    });
    if (declarations.length > 0) out.tools = [{ functionDeclarations: declarations }];
  }
  const config: Json = {};
  if (typeof body.max_tokens === 'number') config.maxOutputTokens = body.max_tokens;
  if (typeof body.temperature === 'number') config.temperature = body.temperature;
  if (typeof body.top_p === 'number') config.topP = body.top_p;
  const stops = stringList(body.stop);
  if (stops.length > 0) config.stopSequences = stops;
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  if (format === 'json_object' || format === 'json_schema') config.responseMimeType = 'application/json';
  if (Object.keys(config).length > 0) out.generationConfig = config;
  return out;
}

/** One Anthropic content block as a Gemini content part. */
function anthropicBlockToGemini(block: unknown): Json | null {
  if (!isRecord(block)) return null;
  if (block.type === 'text' && typeof block.text === 'string') return { text: block.text };
  if (block.type === 'image' && isRecord(block.source)) {
    const source = block.source;
    if (source.type === 'base64' && typeof source.data === 'string') {
      return { inlineData: { mimeType: typeof source.media_type === 'string' ? source.media_type : 'image/png', data: source.data } };
    }
    if (source.type === 'url' && typeof source.url === 'string') return { fileData: { fileUri: source.url } };
  }
  return null;
}

/**
 * An Anthropic Messages request as a Gemini generateContent request naming
 * `model`. Anthropic has no JSON mode, so responseMimeType is never set here.
 */
export function anthropicToGeminiRequest(body: Json, model: string): Json {
  const contents: Json[] = [];
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const assistant = raw.role === 'assistant';
    const content = raw.content;
    if (typeof content === 'string') {
      contents.push({ role: assistant ? 'model' : 'user', parts: [{ text: content }] });
      continue;
    }
    if (!Array.isArray(content)) continue;
    const parts: Json[] = [];
    for (const block of content) {
      if (!isRecord(block)) continue;
      if (!assistant && block.type === 'tool_result') {
        parts.push({
          functionResponse: {
            name: typeof block.tool_use_id === 'string' ? block.tool_use_id : '',
            response: { content: typeof block.content === 'string' ? block.content : textOf(block.content) },
          },
        });
        continue;
      }
      if (assistant && block.type === 'tool_use') {
        parts.push({ functionCall: { name: typeof block.name === 'string' ? block.name : '', args: isRecord(block.input) ? block.input : {} } });
        continue;
      }
      const part = anthropicBlockToGemini(block);
      if (part !== null) parts.push(part);
    }
    if (parts.length > 0) contents.push({ role: assistant ? 'model' : 'user', parts });
  }

  const out: Json = { contents, model };
  const system = textOf(body.system);
  if (system !== '') out.systemInstruction = { parts: [{ text: system }] };
  if (Array.isArray(body.tools)) {
    const declarations = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || typeof raw.name !== 'string') return [];
      return [
        {
          name: raw.name,
          description: typeof raw.description === 'string' ? raw.description : '',
          parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} },
        },
      ];
    });
    if (declarations.length > 0) out.tools = [{ functionDeclarations: declarations }];
  }
  const config: Json = {};
  if (typeof body.max_tokens === 'number') config.maxOutputTokens = body.max_tokens;
  if (typeof body.temperature === 'number') config.temperature = body.temperature;
  if (typeof body.top_p === 'number') config.topP = body.top_p;
  const stops = stringList(body.stop_sequences);
  if (stops.length > 0) config.stopSequences = stops;
  if (Object.keys(config).length > 0) out.generationConfig = config;
  return out;
}

/** The text of a Gemini candidate's parts. */
function geminiText(body: Json): string {
  const candidates = Array.isArray(body.candidates) ? body.candidates : [];
  const first = isRecord(candidates[0]) ? candidates[0] : null;
  if (first === null || !isRecord(first.content)) return '';
  const parts = Array.isArray(first.content.parts) ? first.content.parts : [];
  return parts.flatMap((p) => (isRecord(p) && typeof p.text === 'string' ? [p.text] : [])).join('');
}

/** A Gemini finishReason as an OpenAI finish_reason. */
function geminiStopToOpenAI(reason: unknown): string {
  if (reason === 'MAX_TOKENS') return 'length';
  if (reason === 'SAFETY') return 'content_filter';
  return 'stop';
}

/** A Gemini finishReason as an Anthropic stop_reason. */
function geminiStopToAnthropic(reason: unknown): string {
  if (reason === 'MAX_TOKENS') return 'max_tokens';
  if (reason === 'SAFETY') return 'refusal';
  return 'end_turn';
}

/** Gemini usageMetadata as {input, output} token counts. */
function geminiUsage(body: Json): { input: number; output: number } {
  const meta = isRecord(body.usageMetadata) ? body.usageMetadata : {};
  return { input: numberOrZero(meta.promptTokenCount), output: numberOrZero(meta.candidatesTokenCount) };
}

/** A Gemini generateContent answer as a Chat Completions answer naming `model`. */
export function geminiResponseToOpenAI(body: Json, model: string): Json {
  const candidates = Array.isArray(body.candidates) ? body.candidates : [];
  const first = isRecord(candidates[0]) ? candidates[0] : {};
  const usage = geminiUsage(body);
  return {
    id: 'chatcmpl-sim',
    object: 'chat.completion',
    created: FIXED_CREATED,
    model,
    choices: [{ index: 0, message: { role: 'assistant', content: geminiText(body) }, finish_reason: geminiStopToOpenAI(first.finishReason) }],
    usage: { prompt_tokens: usage.input, completion_tokens: usage.output, total_tokens: usage.input + usage.output },
  };
}

/** A Gemini generateContent answer as an Anthropic Messages answer naming `model`. */
export function geminiResponseToAnthropic(body: Json, model: string): Json {
  const candidates = Array.isArray(body.candidates) ? body.candidates : [];
  const first = isRecord(candidates[0]) ? candidates[0] : {};
  const usage = geminiUsage(body);
  return {
    id: 'msg_sim',
    type: 'message',
    role: 'assistant',
    model,
    content: [{ type: 'text', text: geminiText(body) }],
    stop_reason: geminiStopToAnthropic(first.finishReason),
    stop_sequence: null,
    usage: { input_tokens: usage.input, output_tokens: usage.output },
  };
}

/** One Gemini SSE frame's JSON payloads, in order. */
function geminiFramesOf(text: string): unknown[] {
  const out: unknown[] = [];
  for (const frame of text.replace(/\r\n/g, '\n').split('\n\n')) {
    if (frame.trim() === '') continue;
    const dataLines: string[] = [];
    for (const line of frame.split('\n')) {
      if (line.startsWith('data:')) dataLines.push(line.slice('data:'.length).replace(/^ /, ''));
    }
    if (dataLines.length === 0) continue;
    const payload = dataLines.join('\n').trim();
    if (payload === '[DONE]') continue;
    try {
      out.push(JSON.parse(payload));
    } catch {
      // A frame that is not JSON: skip it.
    }
  }
  return out;
}

/**
 * A Gemini :streamGenerateContent stream as the OpenAI event sequence: a role
 * chunk, content deltas, one finish chunk, the usage chunk, then [DONE].
 */
export async function* geminiStreamToOpenAI(upstream: ReadableStream<Uint8Array>, model: string): AsyncGenerator<string> {
  const reader = upstream.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let roleSent = false;
  let finished = false;

  const finishEvents = (usage: { input: number; output: number }, reason: string): string[] => {
    if (finished) return [];
    finished = true;
    return [
      sseLine(openAIChunk(model, {}, reason)),
      sseLine({ id: 'chatcmpl-sim', object: 'chat.completion.chunk', created: FIXED_CREATED, model, choices: [], usage: { prompt_tokens: usage.input, completion_tokens: usage.output, total_tokens: usage.input + usage.output } }),
      'data: [DONE]\n\n',
    ];
  };

  const eventsOfText = (text: string): string[] => {
    const out: string[] = [];
    for (const frame of geminiFramesOf(text)) {
      if (!isRecord(frame)) continue;
      const text = geminiText(frame);
      if (text !== '') {
        if (!roleSent) {
          roleSent = true;
          out.push(sseLine(openAIChunk(model, { role: 'assistant', content: '' }, null)));
        }
        out.push(sseLine(openAIChunk(model, { content: text }, null)));
      }
      const candidates = Array.isArray(frame.candidates) ? frame.candidates : [];
      const first = isRecord(candidates[0]) ? candidates[0] : null;
      if (first !== null && typeof first.finishReason === 'string') {
        out.push(...finishEvents(geminiUsage(frame), geminiStopToOpenAI(first.finishReason)));
      }
    }
    return out;
  };

  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, '\n');
      let at = buffer.indexOf('\n\n');
      while (at !== -1) {
        const frame = buffer.slice(0, at);
        buffer = buffer.slice(at + 2);
        for (const event of eventsOfText(frame)) yield event;
        at = buffer.indexOf('\n\n');
      }
    }
    buffer += decoder.decode();
    if (buffer.trim() !== '') for (const event of eventsOfText(buffer)) yield event;
  } finally {
    reader.releaseLock();
  }
  if (!finished) {
    if (!roleSent) yield sseLine(openAIChunk(model, { role: 'assistant', content: '' }, null));
    for (const event of finishEvents({ input: 0, output: 0 }, 'stop')) yield event;
  }
}

/** One Anthropic SSE event line. */
function anthropicEvent(event: string, data: unknown): string {
  return `event: ${event}\ndata: ${JSON.stringify(data)}\n\n`;
}

/**
 * A Gemini :streamGenerateContent stream as the Anthropic event sequence:
 * message_start, one text block's start/delta/stop, message_delta with the
 * stop reason and the output tokens, then message_stop.
 */
export async function* geminiStreamToAnthropic(upstream: ReadableStream<Uint8Array>, model: string): AsyncGenerator<string> {
  const reader = upstream.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let started = false;
  let blockOpen = false;
  let finished = false;

  const startEvents = (): string[] => {
    if (started) return [];
    started = true;
    return [
      anthropicEvent('message_start', {
        type: 'message_start',
        message: { id: 'msg_sim', type: 'message', role: 'assistant', model, content: [], stop_reason: null, stop_sequence: null, usage: { input_tokens: 0, output_tokens: 0 } },
      }),
    ];
  };
  const finishEvents = (usage: { input: number; output: number }, reason: string): string[] => {
    if (finished) return [];
    finished = true;
    const out: string[] = [];
    if (blockOpen) {
      out.push(anthropicEvent('content_block_stop', { type: 'content_block_stop', index: 0 }));
      blockOpen = false;
    }
    out.push(
      anthropicEvent('message_delta', { type: 'message_delta', delta: { stop_reason: reason, stop_sequence: null }, usage: { output_tokens: usage.output } }),
      anthropicEvent('message_stop', { type: 'message_stop' }),
    );
    return out;
  };

  const eventsOfText = (text: string): string[] => {
    const out: string[] = [];
    for (const frame of geminiFramesOf(text)) {
      if (!isRecord(frame)) continue;
      const text = geminiText(frame);
      if (text !== '') {
        out.push(...startEvents());
        if (!blockOpen) {
          blockOpen = true;
          out.push(anthropicEvent('content_block_start', { type: 'content_block_start', index: 0, content_block: { type: 'text', text: '' } }));
        }
        out.push(anthropicEvent('content_block_delta', { type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text } }));
      }
      const candidates = Array.isArray(frame.candidates) ? frame.candidates : [];
      const first = isRecord(candidates[0]) ? candidates[0] : null;
      if (first !== null && typeof first.finishReason === 'string') {
        out.push(...startEvents(), ...finishEvents(geminiUsage(frame), geminiStopToAnthropic(first.finishReason)));
      }
    }
    return out;
  };

  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, '\n');
      let at = buffer.indexOf('\n\n');
      while (at !== -1) {
        const frame = buffer.slice(0, at);
        buffer = buffer.slice(at + 2);
        for (const event of eventsOfText(frame)) yield event;
        at = buffer.indexOf('\n\n');
      }
    }
    buffer += decoder.decode();
    if (buffer.trim() !== '') for (const event of eventsOfText(buffer)) yield event;
  } finally {
    reader.releaseLock();
  }
  if (!finished) {
    for (const event of startEvents()) yield event;
    for (const event of finishEvents({ input: 0, output: 0 }, 'end_turn')) yield event;
  }
}

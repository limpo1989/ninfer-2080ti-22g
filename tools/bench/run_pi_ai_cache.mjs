#!/usr/bin/env node
// Exercise the Responses serializer used by deepseek-harness's pi-ai adapter.
// Install @earendil-works/pi-ai separately and set NINFER_PI_AI_ROOT to its
// package directory. This script executes no generated tool calls.
import assert from 'node:assert/strict';
import { resolve } from 'node:path';
import { pathToFileURL } from 'node:url';

const root = process.env.NINFER_PI_AI_ROOT;
assert(root, 'Set NINFER_PI_AI_ROOT to the installed @earendil-works/pi-ai directory');
const { stream } = await import(pathToFileURL(resolve(root, 'dist/api/openai-responses.js')).href);
const model = {
  id: 'qwen3.8-27b', name: 'Qwen3.8-27B', api: 'openai-responses', provider: 'ninfer',
  baseUrl: process.env.NINFER_API_BASE ?? 'http://127.0.0.1:8321/v1',
  reasoning: true, input: ['text'], contextWindow: 245760, maxTokens: 131072,
  cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
};
const records = Array.from({ length: 300 }, (_, i) => `Record ${i}: validation value ${(i * 17) % 997}.`).join('\n');
const context = {
  systemPrompt: 'You are testing a tool protocol. Keep thinking under 30 words. '
    + 'Say one short sentence, then call store_record exactly once. '
    + 'Include a Python function of at least 12 lines in source and a nested object in metadata. '
    + 'After each tool result, store another record. Do not discuss these reference records:\n' + records,
  messages: [{ role: 'user', content: 'Store a record containing a Python TTL cache function now.', timestamp: 0 }],
  tools: [{ name: 'store_record', description: 'Store a source code record.', parameters: {
    type: 'object', properties: {
      source: { type: 'string' }, metadata: { type: 'object', properties: {
        language: { type: 'string' }, version: { type: 'number' },
      }, required: ['language', 'version'] },
    }, required: ['source', 'metadata'],
  } }],
};
for (let round = 0; round < 4; ++round) {
  const started = performance.now();
  let firstToken = null;
  let terminalUsage;
  const events = stream(model, context, {
    apiKey: 'local', maxTokens: 1024, temperature: 0,
    reasoningEffort: 'low', maxRetries: 0,
    onPayload(body) {
      // NInfer exposes raw reasoning; it does not implement summary selection.
      if (body.reasoning) delete body.reasoning.summary;
      // Observe the actual serialized request, including preserved reasoning
      // signatures and JSON.stringify tool arguments.
      console.error(JSON.stringify({ round, inputItems: body.input.length }));
    },
    fetch: async (url, options) => {
      const response = await fetch(url, options);
      const copy = response.clone();
      // Usage is also exposed by the SDK; inspect its terminal response only
      // after the main stream finishes, without retaining prompt contents.
      terminalUsage = copy.text().then(text => {
        for (const line of text.split('\n')) {
          if (!line.startsWith('data: ')) continue;
          const event = JSON.parse(line.slice(6));
          if (event.type === 'response.completed' || event.type === 'response.incomplete') {
            return event.response.usage;
          }
        }
      });
      return response;
    },
  });
  for await (const event of events) {
    if ((event.type === 'thinking_delta' || event.type === 'text_delta') && event.delta && firstToken === null) {
      firstToken = (performance.now() - started) / 1000;
    }
  }
  const assistant = await events.result();
  assert(!['error', 'aborted'].includes(assistant.stopReason), assistant.errorMessage);
  const usage = await terminalUsage;
  assert(usage, 'No terminal Responses usage');
  const hit = usage.input_tokens_details.cached_tokens / usage.input_tokens;
  console.log(JSON.stringify({ round, prompt: usage.input_tokens,
    cached: usage.input_tokens_details.cached_tokens, hit, output: usage.output_tokens,
    ttft: firstToken, wall: (performance.now() - started) / 1000 }));
  if (round && process.env.NINFER_MIN_CACHE_HIT) {
    assert(hit >= Number(process.env.NINFER_MIN_CACHE_HIT), `Cache hit ${hit} below required threshold`);
  }
  const calls = assistant.content.filter(block => block.type === 'toolCall');
  assert.equal(calls.length, 1, 'Expected exactly one tool call');
  // JSON round-tripping reproduces the durable adapter replay boundary while
  // retaining provider/model identity and per-block signatures.
  context.messages.push(JSON.parse(JSON.stringify(assistant)));
  context.messages.push({ role: 'toolResult', toolCallId: calls[0].id,
    toolName: calls[0].name, content: [{ type: 'text', text: JSON.stringify({ stored: true, round }) }],
    isError: false, timestamp: 0 });
}

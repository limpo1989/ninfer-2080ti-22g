#!/usr/bin/env node
// Reconstruct text Responses requests from a deepseek-harness v0 session log.
// Reads committed messages only; never executes the recorded tool calls.
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import { pathToFileURL } from 'node:url';

assert(process.argv[2], 'Supply a deepseek-harness session.jsonl path');
assert(process.env.NINFER_PI_AI_ROOT, 'Set NINFER_PI_AI_ROOT to the installed pi-ai package');
const { convertResponsesMessages, convertResponsesTools } = await import(pathToFileURL(resolve(
  process.env.NINFER_PI_AI_ROOT, 'dist/api/openai-responses-shared.js')).href);
const events = (await readFile(process.argv[2], 'utf8')).trim().split('\n').map(JSON.parse);
const emitRequests = process.argv.includes('--emit-requests');
const context = { messages: [] };
let header;
const model = { id: 'qwen3.8-27b', provider: 'ninfer', api: 'openai-responses', reasoning: false, input: ['text'] };

function assistant(message) {
  const state = message.source.replayState;
  assert.equal(state?.response.kind, 'pi-ai');
  assert.equal(state.response.version, 2);
  assert.equal(state.blocks.length, message.content.length);
  const content = message.content.map((block, i) => {
    const replay = state.blocks[i];
    assert.equal(block.type, replay.type);
    if (block.type === 'text') return { type: 'text', text: block.text, textSignature: replay.textSignature };
    if (block.type === 'reasoning') return { type: 'thinking', thinking: block.text,
      thinkingSignature: replay.thinkingSignature, redacted: replay.redacted };
    assert.equal(block.type, 'tool-call');
    return { type: 'toolCall', id: block.id, name: block.name,
      arguments: JSON.parse(block.arguments), thoughtSignature: replay.thoughtSignature };
  });
  return { ...state.response, role: 'assistant', content, timestamp: 0 };
}

function text(blocks) {
  return blocks.map(block => block.type === 'text' ? block.text
    : block.type === 'tool-result' ? text(block.content) : '').join('');
}

function user(message) {
  const plain = text(message.content.filter(block => block.type !== 'tool-result'));
  const results = message.content.filter(block => block.type === 'tool-result');
  if (plain.length || !results.length) context.messages.push({ role: 'user', content: plain, timestamp: 0 });
  for (const result of results) {
    const call = context.messages.flatMap(item => Array.isArray(item.content) ? item.content : [])
      .find(block => block.type === 'toolCall' && block.id === result.toolCallId);
    context.messages.push({ role: 'toolResult', toolCallId: result.toolCallId,
      toolName: call?.name ?? 'unknown', content: [{ type: 'text', text: text(result.content) || '(no output)' }],
      isError: result.isError ?? false, timestamp: 0 });
  }
}

for (const event of events) {
  if (event.type === 'request/header') {
    header = event.data.header;
    context.systemPrompt = header.system;
    context.tools = header.tools;
    assert.equal(header.config.provider, model.provider);
    assert.equal(header.config.model, model.id);
  } else if (event.type === 'user/message') {
    assert.equal(event.surfaceOp, 'append');
    user(event.data);
  } else if (event.type === 'tool/result') {
    user(event.data.message);
  } else if (event.type === 'assistant/message') {
    assert(header, 'Missing request header');
    const body = { model: model.id,
      input: convertResponsesMessages(model, context, new Set(['openai', 'openai-codex', 'opencode'])),
      tools: convertResponsesTools(context.tools, { supportsStrictMode: false }),
    };
    const expected = event.data.usage.inputTokens + (event.data.usage.cacheReadTokens ?? 0);
    const record = { turn: event.data.turn, step: event.data.step, expected, input_items: body.input.length };
    if (emitRequests) {
      console.log(JSON.stringify({ ...record, body }));
    } else {
      const response = await fetch((process.env.NINFER_API_BASE ?? 'http://127.0.0.1:8321/v1') + '/responses/input_tokens', {
        method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body),
      });
      const result = await response.json();
      assert(response.ok, JSON.stringify(result));
      console.log(JSON.stringify({ ...record, measured: result.input_tokens, matches: result.input_tokens === expected }));
      assert.equal(result.input_tokens, expected, `Request reconstruction differs at ${record.turn}/${record.step}`);
    }
    context.messages.push(assistant(event.data.message));
  }
}

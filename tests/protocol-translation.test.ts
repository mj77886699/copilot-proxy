import test from "node:test";
import assert from "node:assert/strict";

import type { ChatCompletionChunk } from "../src/types/openai.js";
import {
  translateToAnthropic,
  translateToOpenAI,
} from "../src/routes/messages/non-stream-translation.js";
import {
  createAnthropicStreamState,
  finalizeAnthropicStream,
  translateChunkToAnthropicEvents,
} from "../src/routes/messages/stream-translation.js";
import {
  createGeminiStreamState,
  translateGeminiToOpenAI,
  translateOpenAIChunkToGemini,
} from "../src/routes/gemini/gemini-translation.js";
import { prepareCopilotChatPayload } from "../src/services/copilot-completions.js";

function chunk(
  choices: ChatCompletionChunk["choices"],
  usage?: ChatCompletionChunk["usage"]
): ChatCompletionChunk {
  return {
    id: "chatcmpl_test",
    object: "chat.completion.chunk",
    created: 0,
    model: "test-model",
    choices,
    usage,
  };
}

test("Anthropic request forwards thinking without exposing history as visible text", () => {
  const translated = translateToOpenAI({
    model: "claude-sonnet-4-20250514",
    max_tokens: 1024,
    thinking: { type: "enabled", budget_tokens: 4096 },
    messages: [
      {
        role: "assistant",
        content: [
          { type: "thinking", thinking: "private reasoning" },
          { type: "text", text: "visible answer" },
        ],
      },
      { role: "user", content: "continue" },
    ],
  });

  assert.deepEqual(translated.thinking, {
    type: "enabled",
    budget_tokens: 4096,
  });
  assert.equal(translated.reasoning_effort, "medium");
  assert.equal(translated.messages[0].content, "visible answer");
  assert.equal(translated.messages[0].reasoning_content, "private reasoning");
  assert.doesNotMatch(String(translated.messages[0].content), /private reasoning/);
});

test("Anthropic non-streaming response maps hidden reasoning to thinking", () => {
  const translated = translateToAnthropic({
    id: "response",
    object: "chat.completion",
    created: 0,
    model: "claude",
    choices: [
      {
        index: 0,
        message: {
          role: "assistant",
          reasoning_content: "reasoning",
          content: "answer",
        },
        logprobs: null,
        finish_reason: "stop",
      },
    ],
  });

  assert.deepEqual(translated.content, [
    { type: "thinking", thinking: "reasoning" },
    { type: "text", text: "answer" },
  ]);
});

test("Anthropic streaming waits for a choices-empty usage chunk", () => {
  const state = createAnthropicStreamState();
  translateChunkToAnthropicEvents(
    chunk([
      {
        index: 0,
        delta: { content: "hello" },
        finish_reason: null,
        logprobs: null,
      },
    ]),
    state
  );

  const finishEvents = translateChunkToAnthropicEvents(
    chunk([
      {
        index: 0,
        delta: {},
        finish_reason: "stop",
        logprobs: null,
      },
    ]),
    state
  );
  assert.equal(finishEvents.some((event) => event.type === "message_stop"), false);

  const usageEvents = translateChunkToAnthropicEvents(
    chunk([], {
      prompt_tokens: 12,
      completion_tokens: 3,
      total_tokens: 15,
      prompt_tokens_details: { cached_tokens: 2 },
    }),
    state
  );
  const messageDelta = usageEvents.find((event) => event.type === "message_delta");
  assert.deepEqual(messageDelta, {
    type: "message_delta",
    delta: { stop_reason: "end_turn", stop_sequence: null },
    usage: {
      input_tokens: 10,
      output_tokens: 3,
      cache_read_input_tokens: 2,
    },
  });
  assert.equal(usageEvents.at(-1)?.type, "message_stop");
  assert.deepEqual(finalizeAnthropicStream(state), []);
});

test("Anthropic streaming keeps parallel tool block indexes valid", () => {
  const state = createAnthropicStreamState();
  const allEvents = [
    ...translateChunkToAnthropicEvents(
      chunk([
        {
          index: 0,
          delta: {
            tool_calls: [
              {
                index: 0,
                id: "call_a",
                function: { name: "weather", arguments: '{"city"' },
              },
              {
                index: 1,
                id: "call_b",
                function: { name: "time", arguments: '{"zone"' },
              },
            ],
          },
          finish_reason: null,
          logprobs: null,
        },
      ]),
      state
    ),
    ...translateChunkToAnthropicEvents(
      chunk([
        {
          index: 0,
          delta: {
            tool_calls: [
              { index: 1, function: { arguments: ':"UTC"}' } },
              { index: 0, function: { arguments: ':"Shanghai"}' } },
            ],
          },
          finish_reason: null,
          logprobs: null,
        },
      ]),
      state
    ),
    ...translateChunkToAnthropicEvents(
      chunk([
        {
          index: 0,
          delta: {},
          finish_reason: "tool_calls",
          logprobs: null,
        },
      ], {
        prompt_tokens: 5,
        completion_tokens: 7,
        total_tokens: 12,
      }),
      state
    ),
  ];

  const starts = allEvents.filter((event) => event.type === "content_block_start");
  assert.deepEqual(starts.map((event) => event.index), [0, 1]);
  const stopPosition = new Map<number, number>();
  allEvents.forEach((event, index) => {
    if (event.type === "content_block_stop") stopPosition.set(event.index, index);
  });
  allEvents.forEach((event, index) => {
    if (event.type === "content_block_delta") {
      assert.ok(index < (stopPosition.get(event.index) ?? Infinity));
    }
  });
  assert.deepEqual([...stopPosition.keys()], [0, 1]);
});

test("Gemini request maps topK and allowed function names", () => {
  const translated = translateGeminiToOpenAI("gemini", {
    contents: [{ role: "user", parts: [{ text: "hello" }] }],
    generationConfig: { topK: 40 },
    tools: [
      {
        functionDeclarations: [
          { name: "weather" },
          { name: "time" },
        ],
      },
    ],
    toolConfig: {
      functionCallingConfig: {
        mode: "ANY",
        allowedFunctionNames: ["weather"],
      },
    },
  });

  assert.equal(translated.top_k, 40);
  assert.deepEqual(translated.tools?.map((tool) => tool.function.name), ["weather"]);
  assert.deepEqual(translated.tool_choice, {
    type: "function",
    function: { name: "weather" },
  });
});

test("Gemini request preserves parametersJsonSchema used by Claude Desktop", () => {
  const parametersJsonSchema = {
    type: "object",
    properties: {
      file_path: { type: "string" },
    },
    required: ["file_path"],
    additionalProperties: false,
  };
  const translated = translateGeminiToOpenAI("gemini", {
    contents: [{ role: "user", parts: [{ text: "hello" }] }],
    tools: [
      {
        functionDeclarations: [
          {
            name: "Read",
            parametersJsonSchema,
          },
        ],
      },
    ],
  });

  assert.deepEqual(
    translated.tools?.[0].function.parameters,
    parametersJsonSchema
  );
});

test("Gemini content preserves text alongside function responses", () => {
  const translated = translateGeminiToOpenAI("gemini", {
    contents: [
      {
        role: "user",
        parts: [
          {
            functionResponse: {
              id: "call_weather",
              name: "weather",
              response: { temperature: 24 },
            },
          },
          { text: "Please summarize the result." },
        ],
      },
    ],
  });

  assert.equal(translated.messages.length, 2);
  assert.deepEqual(translated.messages[0], {
    role: "tool",
    tool_call_id: "call_weather",
    content: '{"temperature":24}',
  });
  assert.deepEqual(translated.messages[1], {
    role: "user",
    content: "Please summarize the result.",
  });
});

test("Gemini streaming buffers fragmented parallel function arguments", () => {
  const state = createGeminiStreamState();
  const first = translateOpenAIChunkToGemini(
    chunk([
      {
        index: 0,
        delta: {
          tool_calls: [
            {
              index: 0,
              id: "call_weather",
              function: { name: "weather", arguments: '{"city"' },
            },
            {
              index: 1,
              id: "call_time",
              function: { name: "time", arguments: '{"zone"' },
            },
          ],
        },
        finish_reason: null,
        logprobs: null,
      },
    ]),
    "gemini",
    state
  );
  assert.equal(first.candidates?.[0].content, undefined);

  translateOpenAIChunkToGemini(
    chunk([
      {
        index: 0,
        delta: {
          tool_calls: [
            { index: 0, function: { arguments: ':"Shanghai"}' } },
            { index: 1, function: { arguments: ':"Asia/Shanghai"}' } },
          ],
        },
        finish_reason: null,
        logprobs: null,
      },
    ]),
    "gemini",
    state
  );

  const final = translateOpenAIChunkToGemini(
    chunk([
      {
        index: 0,
        delta: {},
        finish_reason: "tool_calls",
        logprobs: null,
      },
    ]),
    "gemini",
    state
  );
  assert.deepEqual(final.candidates?.[0].content?.parts, [
    {
      functionCall: {
        id: "call_weather",
        name: "weather",
        args: { city: "Shanghai" },
      },
    },
    {
      functionCall: {
        id: "call_time",
        name: "time",
        args: { zone: "Asia/Shanghai" },
      },
    },
  ]);

  const usageOnly = translateOpenAIChunkToGemini(
    chunk([], {
      prompt_tokens: 10,
      completion_tokens: 4,
      total_tokens: 14,
    }),
    "gemini",
    state
  );
  assert.equal(usageOnly.candidates, undefined);
  assert.equal(usageOnly.usageMetadata?.totalTokenCount, 14);
});


test("Copilot wire payload omits unsupported protocol-only fields", () => {
  const wirePayload = prepareCopilotChatPayload({
    model: "gemini",
    messages: [{ role: "user", content: "hello" }],
    top_k: 40,
    thinking: { type: "enabled", budget_tokens: 2048 },
    reasoning_effort: "medium",
    stream: true,
  });

  assert.equal(wirePayload.top_k, undefined);
  assert.equal(wirePayload.thinking, undefined);
  assert.equal(wirePayload.reasoning_effort, "medium");
  assert.equal(wirePayload.stream, true);
});

test("Copilot wire payload removes invalid empty Gemini tool selection", () => {
  const wirePayload = prepareCopilotChatPayload({
    model: "gemini",
    messages: [{ role: "user", content: "hello" }],
    tools: [],
    tool_choice: "required",
  });

  assert.equal(wirePayload.tools, undefined);
  assert.equal(wirePayload.tool_choice, undefined);
});

test("Copilot wire payload normalizes empty tool parameter schemas", () => {
  const wirePayload = prepareCopilotChatPayload({
    model: "gemini",
    messages: [{ role: "user", content: "hello" }],
    tools: [
      {
        type: "function",
        function: {
          name: "TaskList",
          parameters: {},
        },
      },
    ],
  });

  assert.deepEqual(wirePayload.tools?.[0].function.parameters, {
    type: "object",
    properties: {},
  });
});


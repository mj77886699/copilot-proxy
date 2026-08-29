import type { ChatCompletionChunk } from "../../types/openai.js";
import type {
  AnthropicStreamEventData,
  AnthropicStreamState,
} from "./anthropic-types.js";
import { mapOpenAIStopReasonToAnthropic } from "./utils.js";

export function createAnthropicStreamState(): AnthropicStreamState {
  return {
    messageStartSent: false,
    nextContentBlockIndex: 0,
    toolCalls: {},
    messageStopped: false,
  };
}

export function translateChunkToAnthropicEvents(
  chunk: ChatCompletionChunk,
  state: AnthropicStreamState
): Array<AnthropicStreamEventData> {
  const events: Array<AnthropicStreamEventData> = [];

  if (chunk.usage) state.lastUsage = chunk.usage;

  const choice = chunk.choices[0];
  if (!choice) {
    if (state.pendingStopReason && !state.messageStopped) {
      events.push(...buildFinalEvents(state));
    }
    return events;
  }

  ensureMessageStart(chunk, state, events);
  const { delta } = choice;

  if (delta.reasoning_content) {
    if (!state.thinkingBlock?.open) {
      const index = state.nextContentBlockIndex++;
      state.thinkingBlock = { anthropicBlockIndex: index, open: true };
      events.push({
        type: "content_block_start",
        index,
        content_block: { type: "thinking", thinking: "" },
      });
    }
    events.push({
      type: "content_block_delta",
      index: state.thinkingBlock.anthropicBlockIndex,
      delta: {
        type: "thinking_delta",
        thinking: delta.reasoning_content,
      },
    });
  }

  if (delta.content) {
    // Reasoning precedes visible text in Anthropic's block model.
    if (state.thinkingBlock?.open) {
      events.push({
        type: "content_block_stop",
        index: state.thinkingBlock.anthropicBlockIndex,
      });
      state.thinkingBlock.open = false;
    }

    if (!state.textBlock?.open) {
      const index = state.nextContentBlockIndex++;
      state.textBlock = { anthropicBlockIndex: index, open: true };
      events.push({
        type: "content_block_start",
        index,
        content_block: { type: "text", text: "" },
      });
    }

    events.push({
      type: "content_block_delta",
      index: state.textBlock.anthropicBlockIndex,
      delta: { type: "text_delta", text: delta.content },
    });
  }

  if (delta.tool_calls) {
    for (const toolCall of delta.tool_calls) {
      const info = (state.toolCalls[toolCall.index] ??= {
        id: "",
        name: "",
        open: false,
        pendingArguments: "",
      });

      if (toolCall.id && !info.id) info.id = toolCall.id;
      if (toolCall.function?.name && !info.name) {
        info.name = toolCall.function.name;
      }

      const argumentFragment = toolCall.function?.arguments ?? "";
      if (info.anthropicBlockIndex === undefined) {
        if (argumentFragment) info.pendingArguments += argumentFragment;

        if (info.id && info.name) {
          info.anthropicBlockIndex = state.nextContentBlockIndex++;
          info.open = true;
          events.push({
            type: "content_block_start",
            index: info.anthropicBlockIndex,
            content_block: {
              type: "tool_use",
              id: info.id,
              name: info.name,
              input: {},
            },
          });

          if (info.pendingArguments) {
            events.push({
              type: "content_block_delta",
              index: info.anthropicBlockIndex,
              delta: {
                type: "input_json_delta",
                partial_json: info.pendingArguments,
              },
            });
            info.pendingArguments = "";
          }
        }
      } else if (argumentFragment && info.open) {
        events.push({
          type: "content_block_delta",
          index: info.anthropicBlockIndex,
          delta: {
            type: "input_json_delta",
            partial_json: argumentFragment,
          },
        });
      }
    }
  }

  if (choice.finish_reason) {
    events.push(...closeOpenBlocks(state));
    state.pendingStopReason = mapOpenAIStopReasonToAnthropic(
      choice.finish_reason
    );

    // Some providers attach usage to the finish chunk. Others send a later
    // choices: [] usage-only chunk; in that case defer message_stop until it
    // arrives (or until the handler finalizes the stream).
    if (chunk.usage) events.push(...buildFinalEvents(state));
  }

  return events;
}

export function finalizeAnthropicStream(
  state: AnthropicStreamState
): Array<AnthropicStreamEventData> {
  if (!state.pendingStopReason || state.messageStopped) return [];
  return [...closeOpenBlocks(state), ...buildFinalEvents(state)];
}

function ensureMessageStart(
  chunk: ChatCompletionChunk,
  state: AnthropicStreamState,
  events: Array<AnthropicStreamEventData>
): void {
  if (state.messageStartSent) return;
  const usage = state.lastUsage;
  const cachedTokens = usage?.prompt_tokens_details?.cached_tokens;

  events.push({
    type: "message_start",
    message: {
      id: chunk.id,
      type: "message",
      role: "assistant",
      content: [],
      model: chunk.model,
      stop_reason: null,
      stop_sequence: null,
      usage: {
        input_tokens: (usage?.prompt_tokens ?? 0) - (cachedTokens ?? 0),
        output_tokens: 0,
        ...(cachedTokens !== undefined
          ? { cache_read_input_tokens: cachedTokens }
          : {}),
      },
    },
  });
  state.messageStartSent = true;
}

function closeOpenBlocks(
  state: AnthropicStreamState
): Array<AnthropicStreamEventData> {
  const openBlocks: Array<{ index: number; close: () => void }> = [];

  if (state.thinkingBlock?.open) {
    openBlocks.push({
      index: state.thinkingBlock.anthropicBlockIndex,
      close: () => {
        if (state.thinkingBlock) state.thinkingBlock.open = false;
      },
    });
  }
  if (state.textBlock?.open) {
    openBlocks.push({
      index: state.textBlock.anthropicBlockIndex,
      close: () => {
        if (state.textBlock) state.textBlock.open = false;
      },
    });
  }
  for (const toolCall of Object.values(state.toolCalls)) {
    if (toolCall.open && toolCall.anthropicBlockIndex !== undefined) {
      openBlocks.push({
        index: toolCall.anthropicBlockIndex,
        close: () => {
          toolCall.open = false;
        },
      });
    }
  }

  return openBlocks
    .sort((a, b) => a.index - b.index)
    .map(({ index, close }) => {
      close();
      return { type: "content_block_stop", index } as const;
    });
}

function buildFinalEvents(
  state: AnthropicStreamState
): Array<AnthropicStreamEventData> {
  if (!state.pendingStopReason || state.messageStopped) return [];

  const usage = state.lastUsage;
  const cachedTokens = usage?.prompt_tokens_details?.cached_tokens;
  state.messageStopped = true;

  return [
    {
      type: "message_delta",
      delta: {
        stop_reason: state.pendingStopReason,
        stop_sequence: null,
      },
      usage: {
        input_tokens: (usage?.prompt_tokens ?? 0) - (cachedTokens ?? 0),
        output_tokens: usage?.completion_tokens ?? 0,
        ...(cachedTokens !== undefined
          ? { cache_read_input_tokens: cachedTokens }
          : {}),
      },
    },
    { type: "message_stop" },
  ];
}

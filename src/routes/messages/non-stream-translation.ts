import type {
  ChatCompletionResponse,
  ChatCompletionsPayload,
  ContentPart,
  Message,
  TextPart,
  Tool,
  ToolCall,
} from "../../types/openai.js";
import type {
  AnthropicAssistantContentBlock,
  AnthropicAssistantMessage,
  AnthropicMessage,
  AnthropicMessagesPayload,
  AnthropicResponse,
  AnthropicTextBlock,
  AnthropicThinkingBlock,
  AnthropicTool,
  AnthropicToolResultBlock,
  AnthropicToolUseBlock,
  AnthropicUserContentBlock,
  AnthropicUserMessage,
} from "./anthropic-types.js";
import { mapOpenAIStopReasonToAnthropic } from "./utils.js";

// === Payload translation (Anthropic -> OpenAI) ===

export function translateToOpenAI(
  payload: AnthropicMessagesPayload
): ChatCompletionsPayload {
  return {
    model: translateModelName(payload.model),
    messages: translateAnthropicMessagesToOpenAI(
      payload.messages,
      payload.system
    ),
    max_tokens: payload.max_tokens,
    stop: payload.stop_sequences,
    stream: payload.stream,
    temperature: payload.temperature,
    top_p: payload.top_p,
    top_k: payload.top_k,
    user: payload.metadata?.user_id,
    tools: translateAnthropicToolsToOpenAI(payload.tools),
    tool_choice: translateAnthropicToolChoiceToOpenAI(payload.tool_choice),
    // Copilot's chat-completions gateway accepts vendor extension fields for
    // Claude models. Keep the original Anthropic setting and also provide the
    // broadly supported OpenAI-compatible reasoning hint.
    thinking: payload.thinking,
    reasoning_effort: mapThinkingBudgetToReasoningEffort(payload.thinking),
  };
}

function translateModelName(model: string): string {
  if (model.startsWith("claude-sonnet-4-")) {
    return model.replace(/^claude-sonnet-4-.*/, "claude-sonnet-4");
  } else if (model.startsWith("claude-opus-4-")) {
    return model.replace(/^claude-opus-4-.*/, "claude-opus-4");
  }
  return model;
}

function mapThinkingBudgetToReasoningEffort(
  thinking: AnthropicMessagesPayload["thinking"]
): ChatCompletionsPayload["reasoning_effort"] {
  if (!thinking) return undefined;
  const budget = thinking.budget_tokens;
  if (budget === undefined) return "medium";
  if (budget <= 2048) return "low";
  if (budget <= 8192) return "medium";
  return "high";
}

function translateAnthropicMessagesToOpenAI(
  anthropicMessages: Array<AnthropicMessage>,
  system: string | Array<AnthropicTextBlock> | undefined
): Array<Message> {
  const systemMessages = handleSystemPrompt(system);
  const otherMessages = anthropicMessages.flatMap((message) =>
    message.role === "user"
      ? handleUserMessage(message)
      : handleAssistantMessage(message)
  );
  return [...systemMessages, ...otherMessages];
}

function handleSystemPrompt(
  system: string | Array<AnthropicTextBlock> | undefined
): Array<Message> {
  if (!system) return [];
  if (typeof system === "string") {
    return [{ role: "system", content: system }];
  }
  const systemText = system.map((block) => block.text).join("\n\n");
  return [{ role: "system", content: systemText }];
}

function handleUserMessage(message: AnthropicUserMessage): Array<Message> {
  const newMessages: Array<Message> = [];

  if (Array.isArray(message.content)) {
    const toolResultBlocks = message.content.filter(
      (block): block is AnthropicToolResultBlock =>
        block.type === "tool_result"
    );
    const otherBlocks = message.content.filter(
      (block) => block.type !== "tool_result"
    );

    for (const block of toolResultBlocks) {
      newMessages.push({
        role: "tool",
        tool_call_id: block.tool_use_id,
        content: mapContent(block.content),
      });
    }

    if (otherBlocks.length > 0) {
      newMessages.push({
        role: "user",
        content: mapContent(otherBlocks),
      });
    }
  } else {
    newMessages.push({
      role: "user",
      content: mapContent(message.content),
    });
  }

  return newMessages;
}

function handleAssistantMessage(
  message: AnthropicAssistantMessage
): Array<Message> {
  if (!Array.isArray(message.content)) {
    return [{ role: "assistant", content: mapContent(message.content) }];
  }

  const toolUseBlocks = message.content.filter(
    (block): block is AnthropicToolUseBlock => block.type === "tool_use"
  );
  const textContent = message.content
    .filter((block): block is AnthropicTextBlock => block.type === "text")
    .map((block) => block.text)
    .join("\n\n");
  const reasoningContent = message.content
    .filter((block): block is AnthropicThinkingBlock => block.type === "thinking")
    .map((block) => block.thinking)
    .join("\n\n");

  return [
    {
      role: "assistant",
      content: textContent || null,
      ...(reasoningContent ? { reasoning_content: reasoningContent } : {}),
      ...(toolUseBlocks.length > 0
        ? {
            tool_calls: toolUseBlocks.map((toolUse) => ({
              id: toolUse.id,
              type: "function" as const,
              function: {
                name: toolUse.name,
                arguments: JSON.stringify(toolUse.input),
              },
            })),
          }
        : {}),
    },
  ];
}

function mapContent(
  content:
    | string
    | Array<AnthropicUserContentBlock | AnthropicAssistantContentBlock>
): string | Array<ContentPart> | null {
  if (typeof content === "string") return content;
  if (!Array.isArray(content)) return null;

  const hasImage = content.some((block) => block.type === "image");
  if (!hasImage) {
    return content
      .filter((block): block is AnthropicTextBlock => block.type === "text")
      .map((block) => block.text)
      .join("\n\n");
  }

  const contentParts: Array<ContentPart> = [];
  for (const block of content) {
    switch (block.type) {
      case "text":
        contentParts.push({ type: "text", text: block.text });
        break;
      case "image":
        contentParts.push({
          type: "image_url",
          image_url: {
            url: `data:${block.source.media_type};base64,${block.source.data}`,
          },
        });
        break;
    }
  }
  return contentParts;
}

function translateAnthropicToolsToOpenAI(
  anthropicTools: Array<AnthropicTool> | undefined
): Array<Tool> | undefined {
  if (!anthropicTools) return undefined;
  return anthropicTools.map((tool) => ({
    type: "function" as const,
    function: {
      name: tool.name,
      description: tool.description,
      parameters: tool.input_schema,
    },
  }));
}

function translateAnthropicToolChoiceToOpenAI(
  anthropicToolChoice: AnthropicMessagesPayload["tool_choice"]
): ChatCompletionsPayload["tool_choice"] {
  if (!anthropicToolChoice) return undefined;
  switch (anthropicToolChoice.type) {
    case "auto":
      return "auto";
    case "any":
      return "required";
    case "tool":
      if (anthropicToolChoice.name) {
        return { type: "function", function: { name: anthropicToolChoice.name } };
      }
      return undefined;
    case "none":
      return "none";
    default:
      return undefined;
  }
}

// === Response translation (OpenAI -> Anthropic) ===

export function translateToAnthropic(
  response: ChatCompletionResponse
): AnthropicResponse {
  const contentBlocks: Array<AnthropicAssistantContentBlock> = [];
  let stopReason: "stop" | "length" | "tool_calls" | "content_filter" | null =
    response.choices[0]?.finish_reason ?? null;

  for (const choice of response.choices) {
    if (choice.message.reasoning_content) {
      contentBlocks.push({
        type: "thinking",
        thinking: choice.message.reasoning_content,
      });
    }
    contentBlocks.push(...getAnthropicTextBlocks(choice.message.content));
    contentBlocks.push(...getAnthropicToolUseBlocks(choice.message.tool_calls));

    if (choice.finish_reason === "tool_calls" || stopReason === "stop") {
      stopReason = choice.finish_reason;
    }
  }

  return {
    id: response.id,
    type: "message",
    role: "assistant",
    model: response.model,
    content: contentBlocks,
    stop_reason: mapOpenAIStopReasonToAnthropic(stopReason),
    stop_sequence: null,
    usage: {
      input_tokens:
        (response.usage?.prompt_tokens ?? 0) -
        (response.usage?.prompt_tokens_details?.cached_tokens ?? 0),
      output_tokens: response.usage?.completion_tokens ?? 0,
      ...(response.usage?.prompt_tokens_details?.cached_tokens !== undefined && {
        cache_read_input_tokens:
          response.usage.prompt_tokens_details.cached_tokens,
      }),
    },
  };
}

function getAnthropicTextBlocks(
  messageContent: Message["content"]
): Array<AnthropicTextBlock> {
  if (typeof messageContent === "string") {
    return messageContent ? [{ type: "text", text: messageContent }] : [];
  }
  if (Array.isArray(messageContent)) {
    return messageContent
      .filter((part): part is TextPart => part.type === "text")
      .map((part) => ({ type: "text", text: part.text }));
  }
  return [];
}

function getAnthropicToolUseBlocks(
  toolCalls: Array<ToolCall> | undefined
): Array<AnthropicToolUseBlock> {
  if (!toolCalls) return [];
  return toolCalls.map((toolCall) => ({
    type: "tool_use",
    id: toolCall.id,
    name: toolCall.function.name,
    input: parseToolArguments(toolCall.function.arguments),
  }));
}

function parseToolArguments(argumentsJson: string): Record<string, unknown> {
  try {
    const parsed: unknown = JSON.parse(argumentsJson);
    return parsed !== null && typeof parsed === "object" && !Array.isArray(parsed)
      ? (parsed as Record<string, unknown>)
      : { value: parsed };
  } catch {
    return {};
  }
}

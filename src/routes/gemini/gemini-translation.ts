import type {
  ChatCompletionsPayload,
  ChatCompletionResponse,
  ChatCompletionChunk,
  Message,
  Tool,
  ContentPart,
} from "../../types/openai.js";
import type {
  GeminiGenerateContentRequest,
  GeminiGenerateContentResponse,
  GeminiContent,
  GeminiPart,
  GeminiStreamState,
  GeminiTool,
} from "./gemini-types.js";

// === Request translation (Gemini -> OpenAI) ===

export function translateGeminiToOpenAI(
  modelId: string,
  request: GeminiGenerateContentRequest
): ChatCompletionsPayload {
  const messages: Message[] = [];

  if (request.systemInstruction) {
    const text = extractTextFromParts(request.systemInstruction.parts);
    if (text) messages.push({ role: "system", content: text });
  }

  if (request.contents) {
    for (const content of request.contents) {
      messages.push(...convertGeminiContentToOpenAI(content));
    }
  }

  const payload: ChatCompletionsPayload = {
    model: modelId,
    messages,
    temperature: request.generationConfig?.temperature,
    top_p: request.generationConfig?.topP,
    top_k: request.generationConfig?.topK,
    max_tokens: request.generationConfig?.maxOutputTokens,
    stop: request.generationConfig?.stopSequences,
    n: request.generationConfig?.candidateCount,
  };

  if (request.tools) {
    payload.tools = translateGeminiToolsToOpenAI(request.tools);
  }

  const functionConfig = request.toolConfig?.functionCallingConfig;
  const allowedNames = functionConfig?.allowedFunctionNames;
  if (allowedNames?.length && payload.tools) {
    const allowed = new Set(allowedNames);
    payload.tools = payload.tools.filter((tool) =>
      allowed.has(tool.function.name)
    );
  }

  if (functionConfig?.mode) {
    if (functionConfig.mode === "AUTO") {
      payload.tool_choice = "auto";
    } else if (functionConfig.mode === "NONE") {
      payload.tool_choice = "none";
    } else if (functionConfig.mode === "ANY") {
      const availableTools = payload.tools ?? [];
      payload.tool_choice =
        availableTools.length === 1
          ? {
              type: "function",
              function: { name: availableTools[0].function.name },
            }
          : "required";
    }
  }

  return payload;
}

function convertGeminiContentToOpenAI(content: GeminiContent): Message[] {
  const messages: Message[] = [];
  const functionResponses = content.parts.filter(
    (part): part is Extract<GeminiPart, { functionResponse: unknown }> =>
      "functionResponse" in part
  );
  const functionCalls = content.parts.filter(
    (part): part is Extract<GeminiPart, { functionCall: unknown }> =>
      "functionCall" in part
  );
  const ordinaryParts = content.parts.filter(
    (part) => !("functionResponse" in part) && !("functionCall" in part)
  );

  // OpenAI represents every function response as its own tool message. Do not
  // return early: Gemini permits text/images alongside those responses.
  for (const part of functionResponses) {
    messages.push({
      role: "tool",
      tool_call_id:
        part.functionResponse.id || part.functionResponse.name,
      content: JSON.stringify(part.functionResponse.response),
    });
  }

  if (functionCalls.length > 0) {
    messages.push({
      role: "assistant",
      content: extractTextFromParts(ordinaryParts) || null,
      tool_calls: functionCalls.map((part, index) => ({
        id: part.functionCall.id || `call_${index}`,
        type: "function" as const,
        function: {
          name: part.functionCall.name,
          arguments: JSON.stringify(part.functionCall.args),
        },
      })),
    });
    return messages;
  }

  if (ordinaryParts.length > 0 || functionResponses.length === 0) {
    const role = content.role === "model" ? "assistant" : "user";
    messages.push({ role, content: convertOrdinaryParts(ordinaryParts) });
  }

  return messages;
}

function convertOrdinaryParts(parts: GeminiPart[]): Message["content"] {
  const hasImage = parts.some((part) => "inlineData" in part);
  if (!hasImage) return extractTextFromParts(parts);

  const contentParts: ContentPart[] = [];
  for (const part of parts) {
    if ("text" in part) {
      contentParts.push({ type: "text", text: part.text });
    } else if ("inlineData" in part) {
      contentParts.push({
        type: "image_url",
        image_url: {
          url: `data:${part.inlineData.mimeType};base64,${part.inlineData.data}`,
        },
      });
    }
  }
  return contentParts;
}

function extractTextFromParts(parts: GeminiPart[]): string {
  return parts
    .filter((part): part is Extract<GeminiPart, { text: string }> =>
      "text" in part
    )
    .map((part) => part.text)
    .join("\n\n");
}

function translateGeminiToolsToOpenAI(tools: GeminiTool[]): Tool[] {
  const result: Tool[] = [];
  for (const tool of tools) {
    for (const fn of tool.functionDeclarations ?? []) {
      result.push({
        type: "function",
        function: {
          name: fn.name,
          description: fn.description,
          // Claude Desktop uses parametersJsonSchema for its built-in tools.
          // Dropping that field produced `parameters: {}` on the Copilot wire,
          // which the Gemini models reject with a plain HTTP 400.
          parameters: fn.parametersJsonSchema ?? fn.parameters ?? {},
        },
      });
    }
  }
  return result;
}

// === Response translation (OpenAI -> Gemini) ===

export function translateOpenAIToGemini(
  response: ChatCompletionResponse,
  modelId: string
): GeminiGenerateContentResponse {
  const candidates = response.choices.map((choice) => {
    const parts: GeminiPart[] = [];

    if (choice.message.content) {
      parts.push({ text: choice.message.content });
    }

    for (const toolCall of choice.message.tool_calls ?? []) {
      parts.push({
        functionCall: {
          id: toolCall.id,
          name: toolCall.function.name,
          args: parseToolArguments(toolCall.function.arguments),
        },
      });
    }

    return {
      content: { parts, role: "model" as const },
      finishReason: mapOpenAIFinishToGemini(choice.finish_reason),
      index: choice.index,
    };
  });

  return {
    candidates,
    usageMetadata: {
      promptTokenCount: response.usage?.prompt_tokens ?? 0,
      candidatesTokenCount: response.usage?.completion_tokens ?? 0,
      totalTokenCount: response.usage?.total_tokens ?? 0,
    },
    modelVersion: modelId,
  };
}

export function createGeminiStreamState(): GeminiStreamState {
  return { toolCalls: {} };
}

export function translateOpenAIChunkToGemini(
  chunk: ChatCompletionChunk,
  modelId: string,
  state: GeminiStreamState = createGeminiStreamState()
): GeminiGenerateContentResponse {
  const result: GeminiGenerateContentResponse = {};
  const choice = chunk.choices[0];

  if (!choice) {
    if (chunk.usage) {
      result.usageMetadata = mapUsage(chunk.usage);
      result.modelVersion = modelId;
    }
    return result;
  }

  const parts: GeminiPart[] = [];
  if (choice.delta.content) parts.push({ text: choice.delta.content });

  for (const toolCall of choice.delta.tool_calls ?? []) {
    const key = `${choice.index}:${toolCall.index}`;
    const buffered = (state.toolCalls[key] ??= {
      choiceIndex: choice.index,
      toolIndex: toolCall.index,
      id: "",
      name: "",
      arguments: "",
      emitted: false,
    });

    if (toolCall.id && !buffered.id) buffered.id = toolCall.id;
    if (toolCall.function?.name && !buffered.name) {
      buffered.name = toolCall.function.name;
    }
    if (toolCall.function?.arguments) {
      buffered.arguments += toolCall.function.arguments;
    }
  }

  // Function arguments are fragmented JSON in OpenAI streams. Buffer every
  // fragment and emit a Gemini functionCall only after the choice finishes.
  if (choice.finish_reason) {
    const completedCalls = Object.values(state.toolCalls)
      .filter(
        (toolCall) =>
          toolCall.choiceIndex === choice.index &&
          !toolCall.emitted &&
          toolCall.name
      )
      .sort((a, b) => a.toolIndex - b.toolIndex);

    for (const toolCall of completedCalls) {
      parts.push({
        functionCall: {
          id: toolCall.id || `call_${toolCall.toolIndex}`,
          name: toolCall.name,
          args: parseToolArguments(toolCall.arguments),
        },
      });
      toolCall.emitted = true;
    }
  }

  result.candidates = [
    {
      ...(parts.length > 0
        ? { content: { parts, role: "model" as const } }
        : {}),
      ...(choice.finish_reason
        ? { finishReason: mapOpenAIFinishToGemini(choice.finish_reason) }
        : {}),
      index: choice.index,
    },
  ];

  if (chunk.usage) {
    result.usageMetadata = mapUsage(chunk.usage);
    result.modelVersion = modelId;
  }

  return result;
}

function parseToolArguments(argumentsJson: string): Record<string, unknown> {
  if (!argumentsJson) return {};
  try {
    const parsed: unknown = JSON.parse(argumentsJson);
    return parsed !== null && typeof parsed === "object" && !Array.isArray(parsed)
      ? (parsed as Record<string, unknown>)
      : { value: parsed };
  } catch {
    return {};
  }
}

function mapUsage(usage: NonNullable<ChatCompletionChunk["usage"]>) {
  return {
    promptTokenCount: usage.prompt_tokens,
    candidatesTokenCount: usage.completion_tokens,
    totalTokenCount: usage.total_tokens,
  };
}

function mapOpenAIFinishToGemini(
  reason: "stop" | "length" | "tool_calls" | "content_filter"
): string {
  const map: Record<string, string> = {
    stop: "STOP",
    length: "MAX_TOKENS",
    tool_calls: "STOP",
    content_filter: "SAFETY",
  };
  return map[reason] || "OTHER";
}

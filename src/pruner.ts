import type { ToolCallIndexer } from "./indexer.js";

/**
 * Filters the `context` event message array.
 * Removes ToolResultMessage entries where toolCallId is in the index.
 * Keeps ALL other messages including AssistantMessages with tool-call blocks.
 */
export function pruneMessages(messages: any[], indexer: ToolCallIndexer): any[] {
  return messages.filter((msg) => {
    // Only remove toolResult messages that have been summarized
    if (msg.role === "toolResult" && indexer.isSummarized(msg.toolCallId)) {
      return false;
    }
    return true;
  });
}

/**
 * Normalizes message sequencing for strict LLM providers (e.g. Google Gemini,
 * Anthropic, Mistral) which reject:
 * 1. User/custom messages directly following a toolResult without a model turn.
 * 2. Consecutive user/custom messages without an intervening model turn.
 */
export function normalizeMessageSequencing(messages: any[]): any[] {
  const out: any[] = [];

  for (let i = 0; i < messages.length; i++) {
    const msg = messages[i];
    const prev = out[out.length - 1];

    // Bridge toolResult -> user/custom with a synthetic assistant response
    if (
      prev &&
      prev.role === "toolResult" &&
      (msg.role === "user" || msg.role === "custom")
    ) {
      out.push({
        role: "assistant",
        content: [{ type: "text", text: "I have processed the tool results." }],
      });
    }

    // Merge consecutive user/custom messages into a single user turn
    const isUserLike = (m: any) =>
      m && (m.role === "user" || m.role === "custom");
    const currentPrev = out[out.length - 1];
    if (isUserLike(currentPrev) && isUserLike(msg)) {
      const getText = (m: any) => {
        if (typeof m.content === "string") return m.content;
        if (Array.isArray(m.content)) {
          return m.content
            .filter((c: any) => c.type === "text")
            .map((c: any) => c.text)
            .join("\n");
        }
        return "";
      };
      const textA = getText(currentPrev);
      const textB = getText(msg);
      const combined = textA && textB ? `${textA}\n\n${textB}` : textA || textB;
      currentPrev.role = "user";
      currentPrev.content = [{ type: "text", text: combined }];
      continue;
    }

    out.push(msg);
  }

  return out;
}

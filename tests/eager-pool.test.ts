import assert from "node:assert/strict";
import { describe, it } from "node:test";
import { EagerSummaryPool } from "../src/eager-pool.js";
import type { CapturedBatch, ContextPruneConfig } from "../src/types.js";
import { DEFAULT_CONFIG } from "../src/types.js";

function makeBatch(
  turnIndex: number,
  toolName = "bash",
  text = "result output",
): CapturedBatch {
  return {
    turnIndex,
    timestamp: Date.now(),
    assistantText: `assistant for turn ${turnIndex}`,
    toolCalls: [
      {
        toolCallId: `tc-${turnIndex}`,
        toolName,
        args: { cmd: "test" },
        resultText: text,
        isError: false,
      },
    ],
  };
}

function makeMockCtx(
  streamHandler?: (batch: CapturedBatch) => Promise<string>,
) {
  return {
    model: { provider: "mock", id: "mock-1" },
    modelRegistry: {
      getApiKeyAndHeaders: async () => ({
        ok: true,
        apiKey: "test",
        headers: {},
      }),
      getProvider: () => ({
        stream: (_model: any, llmContext: any) => {
          const prompt = llmContext.messages[0]?.content?.[0]?.text ?? "";
          return {
            async *[Symbol.asyncIterator]() {
              yield { type: "text_start", partial: { content: [] } };
            },
            result: async () => {
              const text = streamHandler
                ? await streamHandler(llmContext)
                : `Summary for prompt ${prompt.slice(0, 30)}`;
              return {
                content: [{ type: "text", text }],
                stopReason: "stop",
                usage: {
                  input: 10,
                  output: 5,
                  cacheRead: 0,
                  cacheWrite: 0,
                  totalTokens: 15,
                  cost: {
                    input: 0,
                    output: 0,
                    cacheRead: 0,
                    cacheWrite: 0,
                    total: 0,
                  },
                },
              };
            },
          };
        },
      }),
      find: () => undefined,
    },
    ui: { notify: () => {} },
  } as any;
}

describe("EagerSummaryPool", () => {
  it("bounds background concurrency and pumps queue as workers finish", async () => {
    let currentConcurrent = 0;
    let maxObservedConcurrent = 0;
    const completedBatches: number[] = [];

    const ctx = makeMockCtx(async (llmCtx) => {
      currentConcurrent++;
      if (currentConcurrent > maxObservedConcurrent) {
        maxObservedConcurrent = currentConcurrent;
      }
      // Small delay to keep workers concurrently active
      await new Promise((r) => setTimeout(r, 20));
      currentConcurrent--;
      return "summary";
    });

    const pool = new EagerSummaryPool();
    const config: ContextPruneConfig = {
      ...DEFAULT_CONFIG,
      enabled: true,
      eager: true,
      eagerConcurrency: 2,
      eagerMinPendingBatches: 1,
    };

    const b1 = makeBatch(1, "bash", "long output ".repeat(10));
    const b2 = makeBatch(2, "bash", "long output ".repeat(10));
    const b3 = makeBatch(3, "bash", "long output ".repeat(10));

    pool.enqueue([b1, b2, b3], config, ctx);

    // Drain all 3 batches
    const results = await pool.drain([b1, b2, b3], config, ctx);

    assert.equal(results.length, 3);
    assert.ok(results.every((r) => r !== null && r.summaryText === "summary"));
    // Max concurrency must not exceed eagerConcurrency (2)
    assert.ok(
      maxObservedConcurrent <= 2,
      `max concurrent was ${maxObservedConcurrent}, expected <= 2`,
    );
  });

  it("yields instant cache hits for batches already summarized in background", async () => {
    let providerCalls = 0;
    const ctx = makeMockCtx(async () => {
      providerCalls++;
      return "precomputed summary";
    });

    const pool = new EagerSummaryPool();
    const config: ContextPruneConfig = {
      ...DEFAULT_CONFIG,
      enabled: true,
      eager: true,
      eagerConcurrency: 1,
      eagerMinPendingBatches: 1,
    };

    const batch = makeBatch(1, "bash", "long output ".repeat(10));
    pool.enqueue([batch], config, ctx);

    // Give background worker time to complete
    await new Promise((r) => setTimeout(r, 50));
    assert.equal(pool.stats().done, 1, "background job should be done");
    assert.equal(providerCalls, 1);

    // Calling drain now should be an instant cache hit — no additional provider calls
    const start = Date.now();
    const results = await pool.drain([batch], config, ctx);
    const elapsed = Date.now() - start;

    assert.equal(results.length, 1);
    assert.equal(results[0]?.summaryText, "precomputed summary");
    assert.equal(
      providerCalls,
      1,
      "must not re-invoke provider for precomputed batch",
    );
    assert.ok(
      elapsed < 20,
      `drain took ${elapsed}ms, expected near 0ms cache hit`,
    );
  });

  it("awaits in-flight background job rather than spawning duplicate calls", async () => {
    let providerCalls = 0;
    let releaseStream!: () => void;
    const streamGate = new Promise<void>((r) => {
      releaseStream = r;
    });

    const ctx = makeMockCtx(async () => {
      providerCalls++;
      await streamGate;
      return "awaited in-flight summary";
    });

    const pool = new EagerSummaryPool();
    const config: ContextPruneConfig = {
      ...DEFAULT_CONFIG,
      enabled: true,
      eager: true,
      eagerConcurrency: 1,
      eagerMinPendingBatches: 1,
    };

    const batch = makeBatch(1, "bash", "long output ".repeat(10));
    pool.enqueue([batch], config, ctx);

    // Yield tick so job starts running
    await new Promise((r) => setTimeout(r, 5));
    assert.equal(pool.stats().running, 1, "job should be running");

    // Drain while job is still in flight
    const drainPromise = pool.drain([batch], config, ctx);

    // Unblock the stream
    releaseStream();
    const results = await drainPromise;

    assert.equal(results.length, 1);
    assert.equal(results[0]?.summaryText, "awaited in-flight summary");
    assert.equal(
      providerCalls,
      1,
      "in-flight job must be awaited without a duplicate call",
    );
  });

  it("cleans up committed batches on evict", async () => {
    const ctx = makeMockCtx(async () => "summary");
    const pool = new EagerSummaryPool();
    const config: ContextPruneConfig = {
      ...DEFAULT_CONFIG,
      enabled: true,
      eager: true,
      eagerConcurrency: 1,
      eagerMinPendingBatches: 1,
    };

    const b1 = makeBatch(1, "bash", "output 1 ".repeat(10));
    const b2 = makeBatch(2, "bash", "output 2 ".repeat(10));
    pool.enqueue([b1, b2], config, ctx);
    await new Promise((r) => setTimeout(r, 40));

    assert.equal(pool.stats().done, 2);
    pool.evict([b1]);
    // b1 removed, b2 remains
    assert.equal(pool.stats().done, 1);

    pool.abortAll();
    assert.equal(pool.stats().done, 0);
  });
});

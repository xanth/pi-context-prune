import type { ExtensionContext } from "@earendil-works/pi-coding-agent";
import type {
  CapturedBatch,
  ContextPruneConfig,
  SummarizeResult,
} from "./types.js";
import { summarizeBatch } from "./summarizer.js";

interface EagerJob {
  batch: CapturedBatch;
  status: "queued" | "running" | "done" | "failed";
  result?: SummarizeResult | null;
  abortController: AbortController;
  promise: Promise<SummarizeResult | null>;
  resolve: (res: SummarizeResult | null) => void;
}

export interface EagerPoolCallbacks {
  onJobStart?: (batch: CapturedBatch) => void;
  onJobComplete?: (batch: CapturedBatch, result: SummarizeResult | null) => void;
}

/**
 * Manages non-blocking speculative background summarization.
 * Speculative jobs are in-memory only and never committed until flushPending runs.
 */
export class EagerSummaryPool {
  private jobs = new Map<string, EagerJob>();
  private activeCount = 0;

  constructor(private readonly callbacks?: EagerPoolCallbacks) {}

  /**
   * Generates a stable key for a batch based on its tool call IDs.
   * Tool call IDs are unique across the session, making the key immune to
   * differences in 0-based vs 1-based turn index representations.
   */
  private batchKey(batch: CapturedBatch): string {
    const ids = batch.toolCalls
      .map((tc) => tc.toolCallId)
      .filter(Boolean)
      .sort((a, b) => a.localeCompare(b));
    if (ids.length > 0) {
      return ids.join(",");
    }
    return String(batch.turnIndex);
  }

  /**
   * Enqueues batches for speculative background summarization.
   * Only active when config.eager is enabled and batchingMode is "turn".
   */
  enqueue(
    batches: CapturedBatch[],
    config: ContextPruneConfig,
    ctx: ExtensionContext,
  ): void {
    if (!config.enabled || !config.eager || config.batchingMode !== "turn") {
      return;
    }

    for (const batch of batches) {
      const key = this.batchKey(batch);
      if (!this.jobs.has(key)) {
        let resolve!: (res: SummarizeResult | null) => void;
        const promise = new Promise<SummarizeResult | null>((r) => {
          resolve = r;
        });

        this.jobs.set(key, {
          batch,
          status: "queued",
          abortController: new AbortController(),
          promise,
          resolve,
        });
      }
    }

    // Only start speculative execution once the lag threshold is met.
    let eligibleCount = 0;
    for (const j of this.jobs.values()) {
      if (j.status === "queued" || j.status === "running") {
        eligibleCount++;
      }
    }
    if (eligibleCount >= config.eagerMinPendingBatches) {
      this.pump(config, ctx);
    }
  }

  /**
   * Pumps the queue to run queued jobs up to the eagerConcurrency limit.
   */
  private pump(config: ContextPruneConfig, ctx: ExtensionContext): void {
    if (!config.enabled || !config.eager) return;

    const limit = Math.max(1, config.eagerConcurrency);

    for (const [, job] of this.jobs) {
      if (this.activeCount >= limit) break;
      if (job.status !== "queued") continue;

      this.activeCount++;
      job.status = "running";
      this.callbacks?.onJobStart?.(job.batch);

      (async () => {
        try {
          const res = await summarizeBatch(job.batch, config, ctx, {
            signal: job.abortController.signal,
            silent: true, // Speculative jobs never show UI error toasts
          });
          job.result = res;
          job.status = res ? "done" : "failed";
          job.resolve(res);
        } catch {
          job.status = "failed";
          job.resolve(null);
        } finally {
          this.activeCount--;
          this.callbacks?.onJobComplete?.(job.batch, job.result ?? null);
          this.pump(config, ctx);
        }
      })();
    }
  }

  /**
   * Drains summaries for a list of batches when flushPending executes.
   * Reuses pre-computed summaries (0ms), awaits running ones, and falls back
   * to direct summarizeBatch for anything not yet processed.
   */
  async drain(
    batches: CapturedBatch[],
    config: ContextPruneConfig,
    ctx: ExtensionContext,
    options: {
      signal?: AbortSignal;
      onBatchTextProgress?: (
        index: number,
        total: number,
        batch: CapturedBatch,
        receivedChars: number,
      ) => void;
    } = {},
  ): Promise<Array<SummarizeResult | null>> {
    // If any batches need execution, ensure the queue is being pumped
    this.pump(config, ctx);

    return Promise.all(
      batches.map(async (batch, index) => {
        const key = this.batchKey(batch);
        const job = this.jobs.get(key);

        if (job) {
          if (options.signal) {
            if (options.signal.aborted) {
              job.abortController.abort(options.signal.reason);
              throw new Error("drain: aborted while waiting for eager summary");
            }
            options.signal.addEventListener(
              "abort",
              () => job.abortController.abort(options.signal?.reason),
              { once: true },
            );
          }

          const res = await job.promise;
          if (res) {
            options.onBatchTextProgress?.(
              index,
              batches.length,
              batch,
              res.summaryText.length,
            );
            return res;
          }
          // If the speculative job failed (e.g. transient network error),
          // fall through to a fresh, user-visible summarizeBatch call.
        }

        // Fallback: direct on-demand summarizeBatch
        return summarizeBatch(batch, config, ctx, {
          signal: options.signal,
          onTextProgress: (receivedChars) => {
            options.onBatchTextProgress?.(
              index,
              batches.length,
              batch,
              receivedChars,
            );
          },
        });
      }),
    );
  }

  /**
   * Removes committed batches from the eager cache.
   */
  evict(batches: CapturedBatch[]): void {
    for (const batch of batches) {
      const key = this.batchKey(batch);
      const job = this.jobs.get(key);
      if (job) {
        if (job.status === "queued" || job.status === "running") {
          job.abortController.abort();
        }
        this.jobs.delete(key);
      }
    }
  }

  /**
   * Aborts all pending and running jobs and resets the queue.
   */
  abortAll(): void {
    for (const [, job] of this.jobs) {
      job.abortController.abort();
      if (job.status === "queued" || job.status === "running") {
        job.resolve(null);
      }
    }
    this.jobs.clear();
    this.activeCount = 0;
  }

  /**
   * Diagnostic pool status.
   */
  stats(): { queued: number; running: number; done: number; failed: number } {
    let queued = 0;
    let running = 0;
    let done = 0;
    let failed = 0;
    for (const [, job] of this.jobs) {
      if (job.status === "queued") queued++;
      else if (job.status === "running") running++;
      else if (job.status === "done") done++;
      else if (job.status === "failed") failed++;
    }
    return { queued, running, done, failed };
  }
}

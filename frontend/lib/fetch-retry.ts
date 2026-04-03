/**
 * Fetch wrapper with automatic retry on transient network failures.
 *
 * Safari/iOS often drops long-lived cross-origin connections with
 * `TypeError: Load failed` even though the server finished processing.
 * This wrapper retries transparently so the user doesn't have to.
 */
export async function fetchWithRetry(
  input: RequestInfo | URL,
  init: RequestInit & { retries?: number; retryDelay?: number } = {},
): Promise<Response> {
  const { retries = 2, retryDelay = 1500, ...fetchInit } = init;

  let lastError: unknown;
  for (let attempt = 0; attempt <= retries; attempt++) {
    try {
      const res = await fetch(input, fetchInit);
      return res;
    } catch (err) {
      lastError = err;
      // AbortError means intentional cancel — don't retry
      if (err instanceof DOMException && err.name === "AbortError") throw err;
      if (attempt < retries) {
        console.warn(
          `[fetchWithRetry] attempt ${attempt + 1} failed: ${err instanceof Error ? err.message : err}, retrying in ${retryDelay}ms…`,
        );
        await new Promise((r) => setTimeout(r, retryDelay));

        // If the original signal was provided and not aborted, create a
        // fresh AbortController for the retry so the timeout resets.
        // We can't reuse a consumed body though — caller must handle that.
      } else {
        console.error(
          `[fetchWithRetry] all ${retries + 1} attempts failed`,
        );
      }
    }
  }
  throw lastError;
}

/**
 * Build a FormData with a single `file` field.  Returns a new FormData
 * each time so retries don't hit the "body already consumed" problem.
 */
export function makeFormData(blob: Blob, filename: string): FormData {
  const fd = new FormData();
  fd.append("file", blob, filename);
  return fd;
}

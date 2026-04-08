/**
 * Must stay >= Modal Lite worker wall time (see ``MODAL_LITE_FUNCTION_TIMEOUT_S`` in ``modal_app_lite.py``).
 * Shorter client/Edge aborts cause false "timeout" while the backend is still running.
 */
export const LITE_ANALYZE_FETCH_TIMEOUT_MS = 900_000;

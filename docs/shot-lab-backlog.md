# Shot Lab Backlog

Items below are blocked or deferred from the current implementation. Each entry includes the reason and the condition for unblocking.

---

## Blocked — Requires Additional Infrastructure

### 1. Real-time ball tracking & shot tracer from video frames
**Reason**: Requires OpenCV or equivalent WASM/native module running on-device or server-side (Python backend). The current Next.js Edge runtime cannot run heavy CV pipelines. The Python `backend/services/trajectory_service.py` exists but is only wired to the Render-hosted FastAPI pipeline, not to the `/api/lab` Edge route.  
**Condition**: Wire the Python trajectory/blur-speed services into a callable endpoint from Shot Lab (e.g. a `/api/lab/trajectory` proxy to Render), or implement a lightweight WASM-based frame-diff tracker on the client.

### 2. Club/hand trajectory overlay on video
**Reason**: Requires per-frame pose/club tracking data. Current Shot Lab uses single-pass AI (Gemini/Qwen) returning structured JSON — it does not produce frame-level keypoint data.  
**Condition**: Either (a) run MediaPipe pose on the client and pass landmarks to the results page, or (b) add a server-side pose extraction step before AI analysis.

### 3. Spin rate / spin axis estimate
**Reason**: Not reliably estimable from standard phone camera at typical frame rates (30–60 fps). Requires high-speed camera (240+ fps) or multiple camera angles.  
**Condition**: High-speed camera input path, or physics model validated against ground-truth data.

### 4. Angle of Attack (AoA) estimate
**Reason**: Requires precise club-head tracking through the impact zone at high temporal resolution.  
**Condition**: Same as spin rate — high-fps input or validated model.

### 5. Club path / face angle refinement
**Reason**: Current AI provides qualitative assessment only. Quantitative values require frame-level club detection with sub-degree accuracy.  
**Condition**: Dedicated club detection model + high-fps capture.

### 6. Stripe/Hough-based motion-blur speed estimation on Edge
**Reason**: `backend/services/blur_speed_service.py` uses OpenCV which is unavailable in Edge runtime.  
**Condition**: Port to WASM, or proxy to Python backend from Shot Lab routes.

---

## Deferred — Second Phase (No Blocker, Just Scope)

### 7. Indoor net/simulator mode
**Reason**: Scoped out of MVP; different ball-flight physics.  
**Condition**: User demand + separate calibration UX.

### 8. Multi-club history and per-club baseline
**Reason**: Schema supports `club_type` column (added in `0007_lab_v2.sql`) but UI for filtering/selecting club and building baseline is deferred.  
**Condition**: Sufficient data volume per user to be meaningful.

### 9. Multi-camera fusion
**Reason**: Requires two-device coordination protocol.  
**Condition**: Native app or WebRTC pairing flow.

### 10. Personalized baseline / handicap-adjusted feedback
**Reason**: Needs enough session history per user to compute moving averages.  
**Condition**: Trend endpoint already exists; UI for baseline display deferred.

### 11. Simulator export (e.g. E6 Connect / GSPro format)
**Reason**: Third-party protocol integration not in scope.  
**Condition**: Partnership or community demand.

### 12. Recurring subscription billing (Stripe/payment gateway)
**Reason**: Current Pro upgrade is manual/QR/crypto-based (`payment-db.ts`). Stripe recurring billing requires webhook endpoint, customer portal, and plan management UI.  
**Condition**: Product decision on billing provider + webhook infrastructure.

### 13. E2E test suite for Shot Lab
**Reason**: No test framework is currently set up for the frontend. Unit/integration tests for API routes are feasible via curl/fetch but not yet written.  
**Condition**: Add testing framework (Playwright/Vitest) and write tests for: Free quota exceeded → 429, Pro full fields, history retention cutoff, compare 403 for Free, trend 403 for Free, export 403 for Free.

---

## Implementation Notes

### What was completed in this cycle

| Area | Files | Status |
|------|-------|--------|
| Schema v2 | `schema/0007_lab_v2.sql` | Done: `lab_user_entitlements` table, `lab_jobs` columns |
| Shared types | `frontend/lib/lab-types.ts` | Done: all interfaces/types |
| Shared auth/filter | `frontend/lib/lab-auth.ts` | Done: deduped auth, `filterForTier`, `requirePro`, `buildQuota`, `buildFieldsVisibility` |
| Enhanced config | `frontend/lib/lab-config.ts` | Done: trend constants |
| Enhanced DB | `frontend/lib/lab-db.ts` | Done: compare, trend, export queries; entitlements table |
| API: POST /api/lab | `frontend/app/api/lab/route.ts` | Refactored: uses shared modules, saves summary snippet |
| API: GET /api/lab/:id | `frontend/app/api/lab/[id]/route.ts` | Refactored: uses shared modules |
| API: GET /api/lab/history | `frontend/app/api/lab/history/route.ts` | Refactored: uses shared modules |
| API: GET /api/lab/compare | `frontend/app/api/lab/compare/route.ts` | New: Pro-only, two-job diff |
| API: GET /api/lab/trend | `frontend/app/api/lab/trend/route.ts` | New: Pro-only, time-series metrics |
| API: POST /api/lab/export | `frontend/app/api/lab/export/route.ts` | New: Pro-only, full JSON export |
| UI: Shot Lab page | `frontend/app/shot-lab/page.tsx` | Enhanced: all §3.1 modules, tabs, compare, trend, export, timeline, trajectory, preview-then-lock, inline drills, Pro upgrade module |
| Backlog | `docs/shot-lab-backlog.md` | This file |

### Self-check (§13)

- [x] Shot Lab is a **standalone new section**; coexists with existing analysis; lab is a core paid scenario
- [x] Free/Pro **server-side enforced** — `filterForTier` on API, `requirePro` gates on compare/trend/export
- [x] Phone-only constraint met — only camera/mic/screen used
- [x] No hardcoded fake data; estimates labeled with `source: "estimated"` and confidence bars
- [x] Does not break existing features; patch delivery
- [x] D1 + R2; no new external DB stack
- [x] Main flow works; docs include test and rollback info

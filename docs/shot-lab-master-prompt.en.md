# Shot Lab — Integrated Master Prompt (English)

> **Purpose**: Copy-paste this entire document to **Opus / another coding model** as implementation instructions, or hand it to engineers as the **product + technical + monetization** master spec.  
> **Repository**: `stellar-ai`, primary app in **`frontend/`** (Next.js; some APIs on Edge / Cloudflare; existing `frontend/app/api/analyze/route.ts`, `frontend/lib/d1.ts`, `frontend/lib/r2.ts`, `frontend/lib/capture-quality.ts`, `frontend/lib/pose-filters.ts`, `frontend/app/history`, etc.).  
> **Execution principle**: **patch / targeted fix** only; **do not rewrite** the repo; **do not break** existing `/analyze`, `/plus`, `/pro`, `/history` flows; persistence stays **D1 + R2**—no second database stack.

---

## Role instruction (for the AI executor)

You are a senior full-stack + computer vision engineer for this repo. Goal: add **Shot Lab** as a **new, separate product area** (a core paid surface), while **keeping the existing “Analysis” experience** (e.g. `/analyze`, `/api/analyze`, and related pages) **unchanged**—do **not** rename it or make Shot Lab the only analysis entry. Shot Lab must have **its own routes, API namespace (recommended `/api/lab`), and history/quota** (or isolated fields), and may **reuse** shared `frontend/lib/*` utilities only. Also implement a **phone-only** pipeline using **camera + microphone + (optional) gyro/accel when actually available**—**no fake hard-coded metrics**, **no pretending estimates are radar measurements**.

Deliverables must be **buildable, testable, operable**: data model, APIs, **server-side** permission enforcement, UI tiering, quotas/subscription logic, logging and degradation—not marketing fluff.

---

## 1. Shot Lab product definition

### 1.1 Positioning

**Shot Lab** is a **new module alongside** the existing **Analysis** section: it focuses on **phone-vision launch-monitor-style** metrics, tiered monetization, and lab-specific history. It **does not replace** the current analysis product/routes.

**Relationship to existing Analysis (required):**

- **Keep** current analysis entry points, copy, API behavior, and data semantics (unless a separate PR says otherwise).  
- **Add** Shot Lab pages (e.g. `/lab` or `/shot-lab`), jobs, and result models.  
- **Share** library-level code (capture, storage, auth) where appropriate; **separate** business schemas/history tables from legacy analysis to avoid `type`/payload collisions.

Users can:

- Upload or record swing videos  
- View AI analysis  
- View ball speed, tempo, launch parameters, traces, swing issues  
- Receive training guidance  
- Browse history and comparisons (**gated by tier**)  

### 1.2 Experience model: “taste value, then upgrade”

Two-tier system:

- **Free**: fast value, **not** full premium value, natural upgrade path to **Pro**  
- **Pro**: full professional surface; differences must be **obvious** via **unlocked advanced modules**, not spammy popups  

### 1.3 Hard constraints (must hold alongside monetization rules—stricter wins)

1. **No extra hardware**  
2. **No** radar, external sensors, watches, IMU rigs, tripod-specific devices  
3. **Only** phone **camera + microphone**; **gyro/accel** only if exposed by the browser/native shell—otherwise **degrade gracefully**  
4. **iPhone-first**, **Android-compatible** architecture  
5. **Real product**, not a demo: metrics need a **traceable computation chain**, **confidence**, and **failure reasons**  
6. **Patch-based** changes on the existing repo—no rewrite that drops features  

### 1.4 Honesty boundary (must appear in UI + API)

- **Measured-like**: direct estimation from video/audio/geometry/time series  
- **Estimated / Inferred**: model + hybrid physics—**must be labeled**  
- **Not promised in v1**: military-grade radar accuracy; parity with TrackMan/GCQuad; uniform high accuracy across all environments/devices; precise total/side spin or spin axis **without validation**  

---

## 2. User tiers (Free / Pro)

### A. Free

**Goal**: instant value; preserve a strong reason to pay.

**Recommended entitlements (defaults must be documented and configurable server-side):**

1. **Daily analysis quota**: default **3/day** (**enforced server-side**)  
2. **Basic video analysis** (Shot Lab’s **own** pipeline/job flow; Free tier has **truncated output / locked modules**)  
3. **Basic metrics visible**: ball speed, launch angle, launch direction, tempo, **basic** shot tracer (may be shortened/downsampled)  
4. **Basic AI summary** (short—not the full report)  
5. **History**: **last 7 days** or **last 10 items** (pick policy; enforce server-side)  
6. **No** advanced compare  
7. **No** full issue library (Top-3 or subset only)  
8. **No** advanced export  
9. **No** full drill library (preview + locks)  
10. In-result **Pro upgrade prompts** (mostly soft/inline)  

### B. Pro

**Goal**: premium, complete, clearly professional.

**Recommended entitlements:**

1. **Unlimited** or **very high** analysis cap (server-configured)  
2. **Full metric set** (incl. backswing/downswing/top pause, overlays per MVP)  
3. **Full AI report** (structured; consistent with metrics)  
4. **Full swing issue detection** (full issue list)  
5. **Long-term history** (document retention + privacy)  
6. **Swing comparison**  
7. **Trend analytics**  
8. **Multi-session summaries**  
9. **Full drill recommendations**  
10. **Export / share**  
11. **Pro UI affordances** (badge/module styling—tasteful)  
12. **Future**: simulator hooks, club history, personalized baseline (phase 2)  

---

## 3. Shot Lab page structure (Free vs Pro)

### 3.1 Required modules

1. **Header**: title **Shot Lab**, subtitle **击球实验室** (bilingual app: keep Chinese subtitle as specified)  
2. **Upload / record entry**  
3. **Recent analysis cards**  
4. **Current analysis result**  
5. **Metrics section** (Free: partial + previews; Pro: full)  
6. **Trace visualization** (Free: basic tracer; Pro: full tracer + club/hand overlay when available)  
7. **AI diagnostics** (Free: short summary; Pro: full report)  
8. **Drill suggestions** (Free: preview; Pro: full sets)  
9. **History** (Free: 7d/10; Pro: long retention)  
10. **Upgrade Pro block** (inline module—not the only pattern)  

### 3.2 UX rules

- Free must **not** feel broken or “cheap”: use **preview / unlock** cards instead of empty holes  
- Pro value shows up as **deeper dimensions + longer history + advanced modules**  
- Popups only for **hard gates** (quota exhausted); otherwise **inline CTAs, footnotes, light badges**  

---

## 4. Feature tiering table (implementation-grade)

| Feature | Free | Pro | Presentation | Upgrade trigger | Why tiered |
|--------|------|-----|--------------|-----------------|------------|
| Single analysis | ✓ (quota) | ✓ (high/unlimited) | Same entry; block when over quota | quota / advanced click | core hook |
| Daily analyses | default 3 | high/unlimited | show remaining | exhausted | conversion |
| Ball speed | ✓ basic | ✓ + detail/confidence | metric cards | optional deeper breakdown | depth for Pro |
| Launch angle | ✓ | ✓ | metric cards | advanced viz | trust building |
| Launch direction | ✓ | ✓ | metric cards | compare view | trust building |
| Tempo | ✓ | ✓ | metric cards | trends are Pro | rhythm value free |
| Backswing/downswing/top pause | locked/blurred | ✓ full | timeline module | expand timeline | pro depth |
| Shot tracer | ✓ basic | ✓ full | overlay on video | longer trace | visual premium |
| Club/hand trajectory | preview/low | ✓ full | overlay | tap unlock | compute cost |
| Weight shift / shoulder / hip issues | subset / Top-3 context | ✓ full list | issue list | “+N more” | coaching value |
| Top 3 issues | ✓ | ✓ | diagnostics | — | free needs diagnosis |
| Full issue list | ✗ | ✓ | scrollable | expand all | Pro core |
| AI summary | ✓ short | ✓ long | text | “full report” | content depth |
| Full AI report | ✗ | ✓ | structured page | generate full | Pro core |
| Drill recommendations | ✓ 1–2 | ✓ full | drill cards | more drills | training loop |
| History retention | 7d / 10 | long-term | list + locks | open older | data asset |
| Compare swings | ✗ | ✓ | compare page | pick two | advanced |
| Trend analytics | ✗ | ✓ | charts | trends tab | long-term value |
| Export/share | ✗ | ✓ | buttons | export | pro workflow |
| Coach mode / advanced report | ✗ | ✓ | mode toggle | Pro badge | brand separation |

---

## 5. Permissions & billing (must ship in code)

### 5.1 Suggested fields

**User / subscription (extend existing tables; don’t break semantics):**

- `user_id`  
- `plan`: `free` | `pro`  
- `subscription_status`: `active` | `canceled` | `past_due` | `expired`  
- `pro_expires_at` (if applicable)  
- `entitlements_version` (for migrations)  

**Usage (server authoritative):**

- `usage_daily_analysis_count`  
- `usage_daily_analysis_date` (UTC vs user TZ—pick one and document)  
- optional `usage_lifetime_analysis_count`  

### 5.2 Rules

1. **Daily reset** by date key  
2. **Quota decrement** when a job is **accepted / starts processing**; use **idempotent** `analysis_id` to prevent double charge  
3. **Pro expiry**: downgrade to `free` with a **documented policy** for historical Pro data (read-only vs hidden—choose one)  
4. **History retention**: enforce on **read path** + optional **cleanup jobs**—never “hide only in UI”  
5. **API enforcement**: every create-analyze, full-report, compare, trends, export endpoint checks `plan` + `quota`  
6. **Frontend checks**: UX only—**not** security  

### 5.3 Anti-bypass

- **Server-side field filtering** for Free responses (`report_tier: "free"`, truncated payloads)  
- Return **402/403** + stable error codes (`PRO_REQUIRED`, `QUOTA_EXCEEDED`)  
- **No alternate endpoints** that leak Pro fields to Free for the same job  

---

## 6. Upgrade conversion (inside Shot Lab)

1. **Hard prompts**: quota exhausted; compare/trends/export; full issue list; out-of-window history  
2. **Soft prompts**: post-analysis footer “deeper plan”; metric footnotes; drills “+4 more”  
3. **Preview-then-lock**: timeline preview; report teaser; partial issues list  
4. **Hard lock**: export, compare, trends, out-of-window history  
5. **Copy tone**: premium training tool—no shouty ads  
6. **Reason pillars**: deeper analysis, longer history, more professional training loop  

---

## 7. Copy deck (English examples)

**Tone**: premium, professional, restrained—**high-end golf training**, not generic fitness.

1. **Title**: Shot Lab  
2. **Subtitle (Free)**: Professional-grade insight from every swing—using only your phone.  
3. **Subtitle (Pro)**: Full data, long-term history, and deeper training plans—built for serious practice.  
4. **Quota exhausted**: You’ve used today’s included analyses. Come back tomorrow—or continue with Pro.  
5. **Upgrade CTA**: Upgrade to Pro: full reports and long-term progress curves.  
6. **Locked module card**: This module includes the complete movement diagnosis and training sequence. Unlock with Pro.  
7. **History lock**: Older sessions are kept in your Pro history library.  
8. **Compare lock**: Side-by-side comparison and delta breakdown are Pro tools for tracking swing evolution.  

(Chinese copy examples live in `docs/shot-lab-master-prompt.zh.md`.)

---

## 8. Technical scope: MVP / phase 2 / non-commitments

### MVP (must be real + confidence + degradation)

1. Ball speed (estimate)  
2. Vertical launch angle  
3. Horizontal launch direction  
4. Backswing time  
5. Downswing time  
6. Tempo  
7. Swing segmentation (address → finish)  
8. Club/hand trajectory overlay (degrade if unavailable)  
9. Shot tracer  
10. Carry distance **estimate** (must say estimate)  
11. Contact/strike quality score (**do not fake smash factor**)  
12. Session/club history/trends (**Pro full; Free limited**)  
13. AI report (**tiered**)  
14. Drill recommendations (**tiered**)  

### Phase 2

Spin rate/axis estimates, AoA estimate, club path refinement, face angle, indoor net mode, simulator export, multi-ball calibration, personalized baseline, optional multi-camera fusion.

### System architecture (text)

Mobile/Web (Shot Lab) → upload/record → preprocess → swing event detection → 2D pose → hand/club tracking → ball + launch extraction → metric engine → error detection → AI report → history (D1/R2) → calibration/profile.

### Core algorithms (must be actionable)

- **Impact time**: fuse audio peak + pose dynamics + club-head peak motion + ball sudden motion → impact frame + confidence  
- **Ball/launch**: stationary ball; frame diffs; motion-blur streaks; RANSAC/line fit; perspective + calibration; fps + audio alignment  
- **Metrics**: each metric needs **definition, inputs, formula/geometry, failure handling, confidence**—**no fake constant tables as “measurement”**  

### Swing issues (minimum 10)

Weight shift insufficient; shoulder up; hip slide; reverse pivot; early extension; over-the-top; casting; chicken wing; head lift; finish imbalance—each with **keypoints, logic, threshold provenance, confidence, explanation, drill**.

---

## 9. Data model & API (summary—expand to OpenAPI-level during build)

- **Tables**: `lab_sessions`, `lab_shots`, `lab_metrics`, `lab_reports`, `lab_usage_daily`, `user_subscription` (or extend existing)  
- **Examples**: `POST /api/lab/analyze`, `GET /api/lab/jobs/:id`, `GET /api/lab/history`, `GET /api/lab/compare` (Pro), `POST /api/lab/export` (Pro)  
- **Responses**: include `tier`, `fields_visibility`, `quota`  

---

## 10. Phased delivery

### Phase 1 (MVP)

- **New** Shot Lab routes + nav entry **alongside** existing Analysis (**do not** rename legacy analysis to Shot Lab)  
- **Server-side** Free/Pro quotas + response shaping (**lab** endpoints and lab history only)  
- Main path: upload/record → analyze → metrics → report → history (Free limited)  
- Logging, timeouts, retries; feature flag disables **only Shot Lab**, **without** breaking `/analyze`  

### Phase 2

Trends, compare, export, full drill library, calibration/personalization, more estimated metrics.

---

## 11. Phase 1 file touch list (example—verify in repo)

- `frontend/app/lab/**` or `frontend/app/shot-lab/**`  
- `frontend/app/api/lab/**`  
- `frontend/lib/**` (entitlements, quotas, D1 migrations)  
- Minimal nav edits: `frontend/app/layout.tsx`, `frontend/app/page.tsx`  
- Reuse: `capture-quality.ts`, `pose-filters.ts`, `video-store.ts`, `fetch-retry.ts`  
- Avoid expanding misleading semantics in `frontend/app/api/analyze/route.ts` “prediction” fields; put **lab** namespacing on new truth paths  

---

## 12. Test plan & rollback

- **Tests**: E2E for quota exhaustion, Pro full payload, history truncation; API tests for 402/403  
- **Rollback**: feature flag off for Shot Lab routes/APIs; users keep using **legacy Analysis**; reversible migrations  

---

## 13. Self-check

- [ ] Shot Lab is a **separate module** coexisting with legacy Analysis; lab is one paid core surface  
- [ ] Free/Pro enforced **server-side** (not bypassable)  
- [ ] Phone-only constraints satisfied  
- [ ] No fake metrics; estimates labeled  
- [ ] No broken legacy flows; patch delivery  
- [ ] D1+R2 only for persistence additions  
- [ ] Runnable main path; tests + rollback documented  

---

## 14. Engineering deliverable order (produce docs/PRs in this exact sequence)

1. **Shot Lab product definition** (Section 1 + honesty boundary)  
2. **Free / Pro feature matrix** (Section 4; ship a CSV/internal doc mirror)  
3. **Page IA (text diagram)** (Section 3)  
4. **Data model** (user/subscription/usage/session/shot/metrics/report + migrations)  
5. **API permission design** (endpoints, error codes, `tier` + field filtering, idempotency)  
6. **Quota + subscription logic** (daily reset, decrement timing, Pro expiry, retention)  
7. **Upgrade conversion design** (Section 6 + module-level trigger table)  
8. **UI copy deck** (Section 7)  
9. **Phase 1 MVP scope** (Section 10 + Section 11 file list)  
10. **Phase 2 enhancements** (Section 8 phase 2 + Section 10 phase 2)  

---

**End.** Copy this file in full as a single master prompt.

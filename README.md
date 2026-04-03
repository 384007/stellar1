# Stellar AI ⛳
> 紫金动感体育风 · Purple-Gold Sports Theme

> AI-Powered Professional Golf Swing Analysis Platform  
> AI驱动的专业高尔夫挥杆分析平台

## PR visibility notes / PR 可见性说明

Why GitHub (or another host) sometimes **does not show** “Open pull request” / **看不到「创建 / 更新 PR」入口**：

- The **remote must exist** (`origin` or your fork URL). A clone with **no `git remote`** only has local commits — the hosting site never receives them, so there is nothing to open a PR from.
- The **branch must be pushed**. Local-only branches do not appear in the “compare” dropdown until you run `git push`.

**Quick checks / 排查：**

```bash
git remote -v          # expect fetch/push URL; if empty, add: git remote add origin <repo-url>
git status --short     # confirm commits exist
git push -u <remote> <branch>   # e.g. git push -u origin work
```

After push, use the host’s UI to open a PR (base = e.g. `main`, compare = your branch).  
**本地有提交但平台侧看不到 PR 操作** → 优先核对 **remote 是否配置** 与 **是否已 push**。

---

## Architecture / 架构

| Layer | Technology | Deployment |
|-------|-----------|------------|
| Frontend | Next.js 15 (App Router) + TypeScript + Tailwind CSS | Cloudflare Pages |
| Backend API | FastAPI + Python 3.11 | Render (Free Tier) |
| Database | Cloudflare D1 (SQLite-compatible) | Cloudflare |
| Storage | Cloudflare R2 (S3-compatible) | Cloudflare |
| KV Cache | Cloudflare KV | Cloudflare |
| AI Analysis | Google Gemini 2.5 Flash | Google AI |
| Pose Detection | MediaPipe | Python Backend |
| News | SportsData.io / Mock Data | Backend API |

## Features / 功能

### Free Analysis / 免费分析
- Upload swing video (MP4/MOV, up to 100MB)
- 5-dimension scoring (Grip, Stance, Backswing, Downswing, Follow-through)
- Keyframe extraction with phase labels
- Skeleton HUD overlay (4 default joints, expandable to 8)
- Shot distance prediction with animation
- 3 free analyses per day (guest mode)

### PRO Analysis / PRO专业分析
- All free features plus:
- Pro player comparison (Tiger Woods, Rory McIlroy, Shin Ji-ae)
- 7-day personalized training plan
- Advanced HUD with all 8 joints, swing arc trajectory
- Detailed 800-word AI analysis
- Unlimited analyses

### Screen Mode / 屏幕模式
- Real-time camera capture with HUD overlay
- Wake Lock API (screen stays on)
- Fullscreen mode
- Capture and analyze single frames

---

## Deployment Guide / 部署指南

### Prerequisites / 前提条件

- Node.js 18+
- Python 3.11+
- Cloudflare account (free)
- Render account (free)
- Google AI Studio account (free)

### Step 1: Get API Keys / 获取API密钥

#### Google Gemini API Key (FREE)
1. Visit [Google AI Studio](https://aistudio.google.com/apikey)
2. Click "Create API Key"
3. Copy the key

#### Cloudflare Setup
1. Sign up at [Cloudflare Dashboard](https://dash.cloudflare.com)
2. Note your Account ID from the dashboard

### Step 2: Deploy Backend to Render / 部署后端到Render

1. Push the `backend/` folder to a GitHub repository

2. Go to [Render Dashboard](https://dashboard.render.com)

3. Click "New" → "Web Service"

4. Connect your GitHub repo, select the `backend` directory

5. Configure:
   - **Runtime**: Python
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn main:app --host 0.0.0.0 --port $PORT`

6. Add Environment Variables:
   ```
   GEMINI_API_KEY=your-gemini-key
   JWT_SECRET=your-secret-key
   FRONTEND_URL=https://your-app.pages.dev
   ```

7. Deploy and note the URL (e.g., `https://stellar-golf-api.onrender.com`)

### Step 3: Setup Cloudflare D1 Database / 设置D1数据库

```bash
# Install Wrangler CLI
npm install -g wrangler

# Login to Cloudflare
wrangler login

# Create D1 database
wrangler d1 create stellar-golf-db

# Note the database_id and update frontend/wrangler.toml

# Initialize schema
wrangler d1 execute stellar-golf-db --file=./schema/d1_schema.sql
```

### Step 4: Setup Cloudflare R2 Storage / 设置R2存储

```bash
# Create R2 bucket
wrangler r2 bucket create stellar-golf-media

# Enable public access (optional, for direct video URLs)
# Go to Cloudflare Dashboard → R2 → stellar-golf-media → Settings → Public Access
```

### Step 5: Setup Cloudflare KV / 设置KV缓存

```bash
# Create KV namespace
wrangler kv namespace create KV_CACHE

# Note the ID and update frontend/wrangler.toml
```

### Step 6: Deploy Frontend to Cloudflare Pages / 部署前端到Pages

1. Push the `frontend/` folder to a GitHub repository

2. Go to [Cloudflare Pages](https://dash.cloudflare.com/?to=/:account/pages)

3. Click "Create a project" → "Connect to Git"

4. Select your repo, configure:
   - **Framework preset**: Next.js
   - **Build command**: `npx @cloudflare/next-on-pages`
   - **Build output directory**: `.vercel/output/static`

5. Add Environment Variables:
   ```
   GEMINI_API_KEY=your-gemini-key
   NEXT_PUBLIC_BACKEND_URL=https://stellar-golf-api.onrender.com
   NEXT_PUBLIC_APP_URL=https://your-app.pages.dev
   JWT_SECRET=your-secret-key
   CLOUDFLARE_ACCOUNT_ID=your-account-id
   CLOUDFLARE_R2_BUCKET=stellar-golf-media
   ```

6. In **Pages Settings** → **Functions** → **D1 database bindings**:
   - Variable name: `DB`
   - D1 database: `stellar-golf-db`

7. In **Pages Settings** → **Functions** → **R2 bucket bindings**:
   - Variable name: `R2_BUCKET`
   - R2 bucket: `stellar-golf-media`

8. In **Pages Settings** → **Functions** → **KV namespace bindings**:
   - Variable name: `KV_CACHE`
   - KV namespace: your namespace

9. Deploy!

### Alternative: Local Development / 本地开发

```bash
# Backend
cd backend
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
cp .env.example .env
# Fill in .env values
uvicorn main:app --reload --port 8000

# Frontend (in another terminal)
cd frontend
npm install
cp .env.example .env.local
# Fill in .env.local values
npm run dev
```

---

## Project Structure / 项目结构

```
stellar-ai/
├── frontend/                  # Next.js 15 App Router
│   ├── app/
│   │   ├── page.tsx           # Homepage / 首页
│   │   ├── layout.tsx         # Root layout
│   │   ├── globals.css        # Global styles
│   │   ├── login/page.tsx     # Standard login / 普通登录
│   │   ├── pro-login/page.tsx # Pro login / Pro登录
│   │   ├── analyze/page.tsx   # Analysis page / 分析页
│   │   ├── pro/page.tsx       # Pro analysis / Pro分析页
│   │   └── api/               # API routes (proxy layer)
│   ├── components/
│   │   ├── HeroSection.tsx    # Landing hero
│   │   ├── NewsTickerTop.tsx   # Tech tips ticker
│   │   ├── NewsCarousel.tsx    # News image carousel
│   │   ├── HUDOverlay.tsx      # Skeleton HUD canvas
│   │   ├── KeyframeStrip.tsx   # Keyframe timeline
│   │   ├── ProComparison.tsx   # Pro player comparison
│   │   ├── SimAnimation.tsx    # Shot simulation
│   │   ├── UploadZone.tsx      # Video upload
│   │   └── ScreenModeCapture.tsx # Screen mode camera
│   ├── lib/
│   │   ├── auth.ts            # JWT auth helpers
│   │   ├── d1.ts              # D1 database helpers
│   │   ├── r2.ts              # R2 storage helpers
│   │   ├── gemini.ts          # Gemini API client
│   │   └── proData.ts         # Pro player data
│   ├── public/logo.svg        # SVG logo
│   ├── wrangler.toml          # Cloudflare config
│   └── package.json
│
├── backend/                   # FastAPI on Render
│   ├── main.py                # App entry point
│   ├── routers/
│   │   ├── analyze.py         # Lite analysis endpoint
│   │   ├── pro_analyze.py     # Pro analysis endpoint
│   │   ├── pose.py            # MediaPipe pose detection
│   │   ├── news.py            # News API with mock fallback
│   │   └── auth.py            # Authentication
│   ├── services/
│   │   ├── gemini_service.py  # Gemini 2.5 Flash client
│   │   ├── pose_service.py    # MediaPipe skeleton extraction
│   │   ├── keyframe_service.py # Video keyframe extraction
│   │   ├── hud_service.py     # HUD data generation
│   │   └── shot_predictor.py  # Physics-based shot prediction
│   ├── requirements.txt
│   └── render.yaml
│
├── schema/
│   └── d1_schema.sql          # D1 database schema
│
└── README.md
```

## API Endpoints / API接口

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/auth/register` | Register new user |
| POST | `/auth/login` | Standard login |
| POST | `/auth/pro-login` | Pro login with invite code |
| POST | `/auth/guest` | Guest access |
| POST | `/analyze/lite` | Free swing analysis |
| POST | `/analyze/pro` | Pro swing analysis |
| POST | `/pose/from-video` | Extract pose from video |
| POST | `/pose/from-frame` | Extract pose from image |
| GET | `/news` | Golf news feed |
| GET | `/health` | Health check |

## Pro Invite Codes / Pro邀请码

For testing, use these invite codes:
- `STELLAR2024`
- `GOLFPRO2024`
- `PREMIUM2024`

## Tech Stack Details / 技术栈细节

- **Skeleton Detection**: MediaPipe Pose (33 landmarks, 12 golf-relevant joints)
- **AI Analysis**: Gemini 2.5 Flash with structured JSON output
- **Shot Prediction**: Physics-based model using shoulder rotation velocity, X-factor, and launch angle
- **HUD Rendering**: HTML5 Canvas with CSS animations (breathing joints, flow-line connections)
- **News**: SportsData.io API with 30-min KV cache, mock data fallback

## License

MIT License - See LICENSE file for details.

# MediaPipe 自托管：Cloudflare 控制台具体操作（R2 + CORS + Pages）

面向**不用命令行**、只在网页里操作的情况。R2 没有真正的「文件夹」，只有**对象键（Object key）**——键名里带 `/` 看起来像路径，上传时**键名必须一字不差**。

目标键前缀（示例版本 `0.10.33`，与 `frontend/lib/mediapipe-assets.ts` 一致）：

```text
static/mediapipe/tasks-vision/0.10.33/vision_bundle.mjs
static/mediapipe/tasks-vision/0.10.33/wasm/vision_wasm_internal.js
static/mediapipe/tasks-vision/0.10.33/wasm/vision_wasm_internal.wasm
…（wasm 下共 6 个文件）
static/mediapipe/tasks-vision/0.10.33/models/pose_landmarker_full.task
static/mediapipe/tasks-vision/0.10.33/models/pose_landmarker_lite.task
```

打包产物在本地路径：`frontend/build/mediapipe-r2/0.10.33/`（结构与上面键名对应）。

---

## 一、准备文件（没有本地时）

任选其一：

- **GitHub Codespaces**：在浏览器打开仓库 → 终端执行  
  `cd frontend && npm install && node ../tools/mediapipe-pack-for-r2.mjs`  
  然后在文件树里右键 `frontend/build/mediapipe-r2/0.10.33` 打包下载。
- 请能跑 Node 的同事生成该文件夹后 zip 发你。

---

## 二、R2：上传并保持「路径」

1. 浏览器打开 **Cloudflare Dashboard** → 左侧 **R2 对象存储**。
2. 打开桶 **`stellar-golf-media`**（若还没有：点 **创建存储桶**，名称与 Pages 绑定的 R2 一致，见 `frontend/wrangler.toml` 里 `bucket_name`）。
3. 进入该桶 → **对象**（Objects）标签。
4. 点 **上传**（Upload）。

### 方式 0：整包上传整个 `0.10.33` 文件夹（可以，推荐）

**可以**，不必一个一个文件传。只要上传完成后，在 R2 里能看到**同一套目录结构**：

- 某一层下面直接有 `vision_bundle.mjs`
- 同层有 `wasm/`（里面 6 个文件）
- 同层有 `models/`（里面 2 个 `.task`）

**环境变量怎么填：**  
`NEXT_PUBLIC_MEDIAPIPE_CDN_BASE` = **公网根** + **到「这一层」的路径**，**不要**末尾 `/`。  
例如：

| 你上传后 R2 里的对象键长什么样 | `NEXT_PUBLIC_MEDIAPIPE_CDN_BASE` 示例 |
|----------------------------------|--------------------------------------|
| `static/mediapipe/tasks-vision/0.10.33/vision_bundle.mjs` | `https://pub-xxx.r2.dev/static/mediapipe/tasks-vision/0.10.33` |
| `0.10.33/vision_bundle.mjs`（只传了文件夹名这一级） | `https://pub-xxx.r2.dev/0.10.33` |

也就是说：**前缀叫 `static/...` 只是推荐整理方式，不是强制的**；代码只要求「基址 + `/vision_bundle.mjs`」能打开。

**控制台里常见两种整包方式：**

1. **上传文件夹 / 拖入整个 `0.10.33` 目录**  
   若界面可填 **前缀（Prefix）**，填：`static/mediapipe/tasks-vision/0.10.33/`  
   这样键名会变成 `static/mediapipe/tasks-vision/0.10.33/vision_bundle.mjs` 等，和文档推荐一致。

2. **没有前缀选项时**  
   往往会上传成 `0.10.33/vision_bundle.mjs`、`0.10.33/wasm/...`  
   这时把环境变量设为 `https://pub-xxx.r2.dev/0.10.33` 即可（同样无尾斜杠）。

传完后用浏览器打开：  
`你的BASE/vision_bundle.mjs`  
能下载就说明整包路径对了。

**命令行一键整包**（有电脑时）：仓库里的 `bash tools/r2-upload-mediapipe.sh` 会按推荐前缀把 `0.10.33` 下**全部文件**一次性传到 R2。

---

### 方式 A：逐个文件指定「对象键」（最稳）

每上传一个文件时，界面里会有 **对象名称 / Object name / Key**（名称因界面语言略有不同）：

| 本地相对路径（在 `0.10.33/` 下） | 填写的完整对象键（复制粘贴） |
|----------------------------------|------------------------------|
| `vision_bundle.mjs` | `static/mediapipe/tasks-vision/0.10.33/vision_bundle.mjs` |
| `wasm/vision_wasm_internal.js` | `static/mediapipe/tasks-vision/0.10.33/wasm/vision_wasm_internal.js` |
| `wasm/vision_wasm_internal.wasm` | `static/mediapipe/tasks-vision/0.10.33/wasm/vision_wasm_internal.wasm` |
| `wasm/vision_wasm_module_internal.js` | `static/mediapipe/tasks-vision/0.10.33/wasm/vision_wasm_module_internal.js` |
| `wasm/vision_wasm_module_internal.wasm` | `static/mediapipe/tasks-vision/0.10.33/wasm/vision_wasm_module_internal.wasm` |
| `wasm/vision_wasm_nosimd_internal.js` | `static/mediapipe/tasks-vision/0.10.33/wasm/vision_wasm_nosimd_internal.js` |
| `wasm/vision_wasm_nosimd_internal.wasm` | `static/mediapipe/tasks-vision/0.10.33/wasm/vision_wasm_nosimd_internal.wasm` |
| `models/pose_landmarker_full.task` | `static/mediapipe/tasks-vision/0.10.33/models/pose_landmarker_full.task` |
| `models/pose_landmarker_lite.task` | `static/mediapipe/tasks-vision/0.10.33/models/pose_landmarker_lite.task` |

若上传时有 **HTTP 元数据 / Content-Type** 可填，建议：

- `.mjs`、`.js` → `application/javascript`
- `.wasm` → `application/wasm`
- `.task` → `application/octet-stream`

（不填有时也能用，但填了更规范。）

### 方式 B：支持「保留目录结构」的批量上传

若当前界面支持 **上传文件夹** 且可指定**键前缀**，则：

- 前缀填：`static/mediapipe/tasks-vision/0.10.33/`
- 选择本地文件夹 **`0.10.33`** 的内容上传，使网上结构为 `…/vision_bundle.mjs`、`…/wasm/…`、`…/models/…`。

上传完成后，在对象列表里展开核对：**必须能看到** `wasm/` 与 `models/` 下的文件，且键名以 `static/mediapipe/tasks-vision/0.10.33/` 开头。

---

## 三、R2：公网访问（拿到 `https://pub-…r2.dev`）

1. 仍在该 **R2 桶** → **设置**（Settings）。
2. 找到 **公共访问**（Public access）或 **R2.dev 子域**：
   - 按提示 **允许** 公共访问（若首次会分配类似 `pub-xxxxxxxxx.r2.dev` 的地址）。
3. **记下完整公网根 URL**，例如：  
   `https://pub-abc123def456.r2.dev`  
   （下面拼环境变量要用。）

> 若使用 **自定义域名** 绑定到该桶，则把下面所有 `https://pub-….r2.dev` 换成你的域名即可。

---

## 四、R2：CORS（否则浏览器会拦 WASM/模块）

1. 同一桶 → **设置** → **CORS 策略**（CORS Policy）。
2. 粘贴 JSON（把里面的域名改成你的 **真实站点**，可多条）：

```json
[
  {
    "AllowedOrigins": [
      "https://你的项目.pages.dev",
      "https://你的正式域名.com"
    ],
    "AllowedMethods": ["GET", "HEAD"],
    "AllowedHeaders": ["*"],
    "ExposeHeaders": ["ETag", "Content-Length", "Content-Type"],
    "MaxAgeSeconds": 86400
  }
]
```

3. 保存。  
   开发阶段若要本地 `localhost` 调试，可临时加 `"http://127.0.0.1:3000"`（生产可去掉）。

---

## 五、Pages：环境变量 + 重新部署

1. Dashboard → **Workers 和 Pages**（Workers & Pages）→ 选中你的 **Pages 项目**（前端站点）。
2. **设置**（Settings）→ **环境变量**（Environment variables）。
3. **为 Production 添加**（Add variable）：
   - **变量名称**：`NEXT_PUBLIC_MEDIAPIPE_CDN_BASE`
   - **值**（**无末尾斜杠**）：  
     `https://pub-你的子域.r2.dev/static/mediapipe/tasks-vision/0.10.33`  
     即：**公网根** + **`/static/mediapipe/tasks-vision/0.10.33`**（与 R2 里对象键前缀一致，不要多也不要少 `/`）。
4. 保存后进入 **部署**（Deployments）→ **重新部署最近一次的部署**（Retry deployment）或推一个空 commit 触发构建，确保 **新构建** 读到该变量（`NEXT_PUBLIC_*` 在构建时注入）。

---

## 六、自检

1. 浏览器打开（把域名换成你的 R2 公网根）：  
   `https://pub-xxx.r2.dev/static/mediapipe/tasks-vision/0.10.33/vision_bundle.mjs`  
   应能下载/看到脚本内容（不是 404）。
2. 打开你的站点 → 实拍骨架页 → F12 **网络**：`vision_bundle.mjs`、`.wasm` 应对你的 **r2.dev（或自定义域）** 发起请求且状态 200。
3. 若控制台报 **CORS**，回到第四节检查 `AllowedOrigins` 是否包含当前页面域名（含 `https://`，无尾斜杠）。

---

## 七、常见错误

| 现象 | 处理 |
|------|------|
| 404 | 对象键与 `NEXT_PUBLIC_MEDIAPIPE_CDN_BASE` 路径不一致，或漏传 `wasm/`、`models/`。 |
| CORS error | 改 CORS JSON，加上当前访问的完整源（scheme + 域名）。 |
| 改了变量仍不生效 | 必须 **重新触发一次 Pages 构建**，不是只保存变量。 |

英文与脚本说明仍见：`docs/mediapipe-r2-selfhost.md`。

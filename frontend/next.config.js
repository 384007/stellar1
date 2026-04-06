const path = require("path");

/** @type {import('next').NextConfig} */
const nextConfig = {
  images: {
    remotePatterns: [
      { protocol: "https", hostname: "images.unsplash.com" },
      { protocol: "https", hostname: "*.r2.cloudflarestorage.com" },
    ],
  },
  env: {
    NEXT_PUBLIC_BACKEND_URL: process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:8000",
    NEXT_PUBLIC_APP_URL: process.env.NEXT_PUBLIC_APP_URL || "http://localhost:3000",
    // MediaPipe R2 URL is public in the browser anyway. Some hosts only allow "Secret" env names
    // without NEXT_PUBLIC_* — map those here so `next build` can still inline the client bundle.
    NEXT_PUBLIC_MEDIAPIPE_CDN_BASE: (
      process.env.NEXT_PUBLIC_MEDIAPIPE_CDN_BASE ||
      process.env.MEDIAPIPE_CDN_BASE ||
      ""
    ).trim(),
    NEXT_PUBLIC_MEDIAPIPE_ALLOW_FOREIGN_FALLBACK: (
      process.env.NEXT_PUBLIC_MEDIAPIPE_ALLOW_FOREIGN_FALLBACK ||
      process.env.MEDIAPIPE_ALLOW_FOREIGN_FALLBACK ||
      ""
    ).trim(),
  },
  webpack: (config, { webpack, isServer }) => {
    config.resolve.alias["@mediapipe/tasks-vision"] = false;
    config.resolve.alias["@ffmpeg/util"] = path.join(__dirname, "lib/shims/ffmpeg-util.ts");
    // Next 15 edge chunk bootstrap reads `document.baseURI` in Node during `Collecting page data`, which throws.
    // Patch emitted asset so Node build workers (and Cloudflare CI) can load the edge runtime.
    if (isServer) {
      config.plugins.push({
        apply(compiler) {
          compiler.hooks.thisCompilation.tap("PatchEdgeRuntimeWebpackDocument", (compilation) => {
            compilation.hooks.processAssets.tap(
              {
                name: "PatchEdgeRuntimeWebpackDocument",
                stage: webpack.Compilation.PROCESS_ASSETS_STAGE_REPORT,
              },
              () => {
                for (const { name, source } of compilation.getAssets()) {
                  if (!name.includes("edge-runtime-webpack")) continue;
                  const raw = source.source();
                  const s = typeof raw === "string" ? raw : Buffer.from(raw).toString();
                  if (!s.includes("document.baseURI")) continue;
                  const patched = s.replace(
                    /document\.baseURI\|\|self\.location\.href/g,
                    '(typeof document!=="undefined"&&document.baseURI)||(typeof self!=="undefined"&&self.location&&self.location.href)||""',
                  );
                  compilation.updateAsset(name, new webpack.sources.RawSource(patched));
                }
              },
            );
          });
        },
      });
    }
    return config;
  },
};

module.exports = nextConfig;

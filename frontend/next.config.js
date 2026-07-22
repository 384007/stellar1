const path = require("path");

/** @type {import('next').NextConfig} */
const nextConfig = {
  poweredByHeader: false,
  productionBrowserSourceMaps: false,
  images: {
    remotePatterns: [{ protocol: "https", hostname: "images.unsplash.com" }],
  },
  env: {
    // Only values that are intentionally public (site URL). Backend / MediaPipe bases are server-only.
    NEXT_PUBLIC_APP_URL: process.env.NEXT_PUBLIC_APP_URL || "http://localhost:3000",
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

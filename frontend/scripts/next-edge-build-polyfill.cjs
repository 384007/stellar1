/**
 * Optional safety net for `next build`: some Next workers may run without this file,
 * so `next.config.js` also patches `edge-runtime-webpack` output. Keeping both avoids
 * relying on a single mechanism.
 *
 * Next edge bootstrap can read `document.baseURI` in Node during build; this polyfill
 * applies when NODE_OPTIONS preloads this file (see package.json scripts).
 */
/* eslint-disable no-undef */
if (typeof globalThis.document === "undefined") {
  globalThis.document = {
    get baseURI() {
      return "http://next-build.local/";
    },
  };
}
if (typeof globalThis.self === "undefined") {
  globalThis.self = globalThis;
}
if (typeof globalThis.location === "undefined") {
  globalThis.location = { href: "http://next-build.local/" };
}

import { EncryptJWT, jwtDecrypt } from "jose";

export type UploadSessionPayload = {
  file_uri: string;
  mime_type: string;
  file_name: string | null;
  /** 0-based index into ordered Gemini keys on the Edge upload worker */
  gemini_key_index: number;
};

async function dirKeyFromJwtSecret(jwtSecret: string): Promise<Uint8Array> {
  const data = new TextEncoder().encode(jwtSecret || "");
  const hash = await crypto.subtle.digest("SHA-256", data);
  return new Uint8Array(hash);
}

/** Opaque JWE for post-upload Gemini Files reference (browser never sees file_uri / key index). */
export async function sealUploadSession(payload: UploadSessionPayload, jwtSecret: string): Promise<string> {
  const key = await dirKeyFromJwtSecret(jwtSecret);
  return new EncryptJWT({
    file_uri: payload.file_uri,
    mime_type: payload.mime_type,
    file_name: payload.file_name,
    gemini_key_index: payload.gemini_key_index,
  })
    .setProtectedHeader({ alg: "dir", enc: "A256GCM" })
    .setIssuedAt()
    .setExpirationTime("12h")
    .encrypt(key);
}

export async function unsealUploadSession(
  token: string,
  jwtSecret: string,
): Promise<UploadSessionPayload | null> {
  if (!token?.trim() || !jwtSecret) return null;
  try {
    const key = await dirKeyFromJwtSecret(jwtSecret);
    const { payload } = await jwtDecrypt(token, key);
    const file_uri = String(payload.file_uri || "");
    if (!file_uri) return null;
    const ki = payload.gemini_key_index;
    const gemini_key_index =
      typeof ki === "number" && Number.isFinite(ki) ? Math.max(0, Math.floor(ki)) : parseInt(String(ki ?? "0"), 10) || 0;
    return {
      file_uri,
      mime_type: String(payload.mime_type || "video/mp4"),
      file_name: payload.file_name != null ? String(payload.file_name) : null,
      gemini_key_index,
    };
  } catch {
    return null;
  }
}

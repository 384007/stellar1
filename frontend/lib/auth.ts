import { jwtVerify, SignJWT } from "jose";

const _rawSecret = process.env.JWT_SECRET || "";
if (!_rawSecret && typeof window === "undefined") {
  console.warn("[auth] JWT_SECRET not configured");
}
const JWT_SECRET = new TextEncoder().encode(_rawSecret || "");

export interface UserPayload {
  user_id: string;
  email: string;
  is_pro: boolean;
  is_guest: boolean;
}

export async function verifyToken(token: string): Promise<UserPayload | null> {
  try {
    const { payload } = await jwtVerify(token, JWT_SECRET);
    return {
      user_id: payload.user_id as string,
      email: payload.email as string,
      is_pro: payload.is_pro as boolean,
      is_guest: (payload.is_guest as boolean) || false,
    };
  } catch {
    return null;
  }
}

export async function createToken(user: UserPayload): Promise<string> {
  return new SignJWT({
    user_id: user.user_id,
    email: user.email,
    is_pro: user.is_pro,
    is_guest: user.is_guest,
  })
    .setProtectedHeader({ alg: "HS256" })
    .setIssuedAt()
    .setExpirationTime("72h")
    .sign(JWT_SECRET);
}

export function getStoredUser(): UserPayload | null {
  if (typeof window === "undefined") return null;

  const stored = localStorage.getItem("stellar_user");
  if (!stored) return null;

  try {
    return JSON.parse(stored);
  } catch {
    return null;
  }
}

export function getStoredToken(): string | null {
  if (typeof window === "undefined") return null;
  return localStorage.getItem("stellar_token");
}

export function clearAuth(): void {
  if (typeof window === "undefined") return;
  localStorage.removeItem("stellar_token");
  localStorage.removeItem("stellar_user");
}

export function isAuthenticated(): boolean {
  return getStoredToken() !== null;
}

export function isPro(): boolean {
  const user = getStoredUser();
  return user?.is_pro === true;
}

/**
 * Unified upgrade / payment — NEXT_PUBLIC_* inlined at build (set in Cloudflare Pages env).
 */

export function getWechatQrUrl(): string {
  return process.env.NEXT_PUBLIC_PAY_WECHAT_QR_URL || "";
}

export function getAlipayQrUrl(): string {
  return process.env.NEXT_PUBLIC_PAY_ALIPAY_QR_URL || "";
}

export function getUsdcSolAddress(): string {
  return process.env.NEXT_PUBLIC_PAY_SOL_USDC_ADDRESS || "";
}

export function getUsdtSolAddress(): string {
  return process.env.NEXT_PUBLIC_PAY_SOL_USDT_ADDRESS || "";
}

export function getCryptoNetworkLabel(): string {
  return process.env.NEXT_PUBLIC_PAY_CRYPTO_NETWORK_LABEL || "Solana (SPL)";
}

export function getProPriceLabelZh(): string {
  return process.env.NEXT_PUBLIC_PRO_PRICE_LABEL_ZH || "Pro 会员（请按下方方式支付后联系开通）";
}

export function getProPriceLabelEn(): string {
  return process.env.NEXT_PUBLIC_PRO_PRICE_LABEL_EN || "Pro membership — pay below, then we activate your account.";
}

export function getSuggestedUsdcAmount(): string {
  return process.env.NEXT_PUBLIC_PAY_SUGGEST_USDC || "";
}

export function getSuggestedUsdtAmount(): string {
  return process.env.NEXT_PUBLIC_PAY_SUGGEST_USDT || "";
}

export type PaymentChannel = "wechat" | "alipay" | "usdc_sol" | "usdt_sol";

export const PAYMENT_CHANNELS: PaymentChannel[] = ["wechat", "alipay", "usdc_sol", "usdt_sol"];

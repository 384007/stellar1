/**
 * Paste this (plus the JSON from GET /api/admin/finance) into ChatGPT / Cursor / Opus
 * for bookkeeping summaries — no automatic spend on your AI API.
 */

export const FINANCE_AI_USER_PROMPT_ZH = `以下是 Stellar 收款工具的订单与汇总数据（JSON）。请用中文输出：
1) 待确认收款笔数与建议优先处理顺序（按金额/等待时间）
2) 按渠道（微信/支付宝/USDC/USDT）的笔数
3) 本周可记入收入的确认项清单（表格：订单号后8位、渠道、用户标识、备注）
4) 异常项（重复 memo、长时间 pending）
5) 一段给老板的 3 句话 executive summary

数据：
`;

export const FINANCE_AI_SYSTEM_ZH =
  "你是专业的小团队财务助理，只做对账与汇报建议，不编造未出现的订单号；若数据不足要明确说明。";

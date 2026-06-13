#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List

from dotenv import load_dotenv
from tavily import TavilyClient
from openai import OpenAI

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import MarketOrderRequest

load_dotenv()


SYMBOL = os.getenv("SYMBOL", "AAPL")
QUOTE_PER_TRADE = float(os.getenv("QUOTE_PER_TRADE", "100"))
DRY_RUN = os.getenv("DRY_RUN", "true").lower() == "true"

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")
NEBIUS_API_KEY = os.getenv("NEBIUS_API_KEY", "")
NEBIUS_MODEL = os.getenv("NEBIUS_MODEL", "nvidia/Nemotron-3-Ultra-550b-a55b")

ALPACA_API_KEY = os.getenv("ALPACA_API_KEY", "")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY", "")

tavily_client = TavilyClient(api_key=TAVILY_API_KEY)
nebius_client = OpenAI(
    base_url="https://api.tokenfactory.nebius.com/v1/",
    api_key=NEBIUS_API_KEY,
)
alpaca_client = TradingClient(
    api_key=ALPACA_API_KEY,
    secret_key=ALPACA_SECRET_KEY,
    paper=True,
)

@dataclass
class EvidenceItem:
    source: str
    title: str
    url: str
    content: str

def parse_json(text: str) -> Dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = text.replace("```json", "").replace("```", "").strip()
    return json.loads(text)

def tavily_search(query: str, max_items: int = 6) -> List[EvidenceItem]:
    resp = tavily_client.search(query)
    items: List[EvidenceItem] = []
    if isinstance(resp, dict):
        for x in (resp.get("results") or [])[:max_items]:
            items.append(
                EvidenceItem(
                    source="tavily",
                    title=x.get("title", "") or "",
                    url=x.get("url", "") or "",
                    content=x.get("content", "") or x.get("raw_content", "") or "",
                )
            )
    return items

def collect_evidence(symbol: str) -> List[EvidenceItem]:
    news_query = f"{symbol} finance news earnings regulation lawsuit hack macro"
    social_query = f"site:x.com {symbol} Elon Musk CoinDesk WatcherGuru"

    evidence: List[EvidenceItem] = []
    evidence.extend(tavily_search(news_query))
    evidence.extend(tavily_search(social_query))
    return evidence

def analyze_with_nebius(symbol: str, evidence: List[EvidenceItem]) -> Dict[str, Any]:
    payload = [
        {
            "source": e.source,
            "title": e.title,
            "url": e.url,
            "content": e.content[:1200],
        }
        for e in evidence
    ]

    system_prompt = (
        "You are a trading research analyst. "
        "Judge whether evidence is a real bearish catalyst, a temporary washout, or noise. "
        "Return STRICT JSON only."
    )

    user_prompt = f"""
Symbol: {symbol}

Evidence:
{json.dumps(payload, ensure_ascii=False, indent=2)}

Return STRICT JSON with this schema:
{{
  "sentiment": "bullish|bearish|neutral",
  "trade_bias": "long|short|flat",
  "true_negative": true,
  "washout_likely": false,
  "confidence": 0.0,
  "summary": "short paragraph",
  "rationale": ["point 1", "point 2", "point 3"],
  "risk_flags": ["event risk", "thin liquidity"]
}}

Rules:
- Use only the evidence above.
- If evidence is weak or mixed, choose neutral/flat.
- Confidence must be between 0 and 1.
- Output one valid JSON object only.
"""

    resp = nebius_client.chat.completions.create(
        model=NEBIUS_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0,
        max_tokens=2000,
        response_format={"type": "json_object"},
    )

    text = resp.choices[0].message.content or "{}"
    return parse_json(text)

def get_account_summary() -> Dict[str, Any]:
    acct = alpaca_client.get_account()
    return {
        "buying_power": str(acct.buying_power),
        "cash": str(acct.cash),
        "equity": str(acct.equity),
        "status": str(acct.status),
    }

def get_position_qty(symbol: str) -> float:
    try:
        pos = alpaca_client.get_open_position(symbol)
        return float(pos.qty)
    except Exception:
        return 0.0

def get_open_position_or_none(symbol: str):
    try:
        return alpaca_client.get_open_position(symbol)
    except Exception:
        return None

def can_open_short(symbol: str, quote_amount: float) -> tuple[bool, str]:
    account = alpaca_client.get_account()
    equity = float(account.equity)
    buying_power = float(account.buying_power)

    if equity < 2000:
        return False, f"equity too low: {equity:.2f} < 2000"

    # 保守一点：预留 50% 缓冲
    if buying_power < quote_amount * 1.5:
        return False, f"buying power too low: {buying_power:.2f}"

    asset = alpaca_client.get_asset(symbol)
    if not (asset.tradable and asset.shortable and asset.easy_to_borrow):
        return False, "asset not tradable/shortable/ETB"

    return True, ""

def open_short_1_share(symbol: str):
    order = MarketOrderRequest(
        symbol=symbol,
        qty=1,
        side=OrderSide.SELL,
        time_in_force=TimeInForce.DAY,
        position_intent="sell_to_open",
    )
    if DRY_RUN:
        return {"dry_run": True, "action": "SELL_SHORT", "order": str(order)}
    return alpaca_client.submit_order(order)

def close_any_position(symbol: str):
    if DRY_RUN:
        return {"dry_run": True, "action": "CLOSE_POSITION", "symbol": symbol}
    return alpaca_client.close_position(symbol)

def decide_action(analysis: dict, has_position: bool) -> str:
    sentiment = str(analysis.get("sentiment", "neutral")).lower()
    bias = str(analysis.get("trade_bias", "flat")).lower()
    confidence = float(analysis.get("confidence", 0.0) or 0.0)

    # 先保守：先平，再开
    if has_position and sentiment == "bullish" and confidence >= 0.85:
        return "CLOSE"

    if (not has_position) and sentiment == "bearish" and bias == "short" and confidence >= 0.90:
        return "SELL_SHORT"

    if (not has_position) and sentiment == "bullish" and bias == "long" and confidence >= 0.70:
        return "BUY"

    return "HOLD"

def submit_buy(symbol: str, quote_amount: float):
    # 用 market + notional 更适合 paper 和小资金测试
    order = MarketOrderRequest(
        symbol=symbol,
        notional=quote_amount,
        side=OrderSide.BUY,
        time_in_force=TimeInForce.DAY,
    )
    if DRY_RUN:
        return {"dry_run": True, "order": str(order)}
    return alpaca_client.submit_order(order)

def submit_sell_all(symbol: str, qty: float):
    order = MarketOrderRequest(
        symbol=symbol,
        qty=qty,
        side=OrderSide.SELL,
        time_in_force=TimeInForce.DAY,
    )
    if DRY_RUN:
        return {"dry_run": True, "order": str(order)}
    return alpaca_client.submit_order(order)

def run_once():
    evidence = collect_evidence(SYMBOL)
    analysis = analyze_with_nebius(SYMBOL, evidence)

    account = get_account_summary()
    position = get_open_position_or_none(SYMBOL)
    has_position = position is not None

    action = decide_action(analysis, has_position)

    print("\n===== ANALYSIS =====")
    print(json.dumps(analysis, ensure_ascii=False, indent=2))
    print("\n===== ACCOUNT =====")
    print(json.dumps(account, ensure_ascii=False, indent=2))
    print("has_position =", has_position)
    print("decision =", action)

    if action == "SELL_SHORT":
        ok, reason = can_open_short(SYMBOL, QUOTE_PER_TRADE)
        if not ok:
            print("Skip short:", reason)
            return
        result = open_short_1_share(SYMBOL)
        print("\n===== ORDER RESULT =====")
        print(result)

    elif action == "CLOSE":
        result = close_any_position(SYMBOL)
        print("\n===== ORDER RESULT =====")
        print(result)

    elif action == "BUY":
        result = submit_buy(SYMBOL, QUOTE_PER_TRADE)
        print("\n===== ORDER RESULT =====")
        print(result)

    else:
        print("No trade.")

if __name__ == "__main__":
    run_once()

#!/usr/bin/env python3
"""Repair challenge trade snapshots that used client-supplied prices.

The repair is intentionally conservative:
- supported by default for crypto/us-stock challenge trades;
- if quantity is close to the authoritative quote and price is not, treat the
  row as a price/quantity swap;
- otherwise correct price to the authoritative quote and keep quantity.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SERVER_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SERVER_DIR.parents[1]
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

from database import begin_write_transaction, get_db_connection  # noqa: E402
from price_fetcher import get_price_from_market, price_fetch_logging  # noqa: E402


BACKUP_DIR = SERVER_DIR / "data" / "repair_backups"
SUPPORTED_MARKETS = {"crypto", "us-stock"}
PRICE_REL_TOLERANCE = 0.05
SWAP_REL_TOLERANCE = 0.05
MIN_MISMATCH_RATIO = 2.0
DEFAULT_CRYPTO_SYMBOL_CANDIDATES = {
    "BTC",
    "ETH",
    "SOL",
    "BNB",
    "SUI",
    "FET",
    "ADA",
    "XRP",
    "LINK",
    "LTC",
    "AVAX",
    "NEAR",
    "OP",
    "ONDO",
    "TIA",
    "DOT",
    "KAS",
    "ZEC",
}


def utc_now_iso_z() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def row_dict(row: Any) -> dict[str, Any]:
    return dict(row) if row is not None and not isinstance(row, dict) else (row or {})


def as_float(value: Any) -> float:
    try:
        parsed = float(value)
    except Exception:
        return 0.0
    return parsed if math.isfinite(parsed) else 0.0


def relative_diff(a: float, b: float) -> float:
    denominator = max(abs(a), abs(b), 1e-12)
    return abs(a - b) / denominator


def mismatch_ratio(a: float, b: float) -> float:
    if a <= 0 or b <= 0:
        return float("inf")
    return max(a / b, b / a)


def fetch_trade_rows(cursor: Any, challenge_key: str | None, market: str | None, trade_ids: list[int]) -> list[dict[str, Any]]:
    where = ["ct.market IN ('crypto', 'us-stock')"]
    params: list[Any] = []
    if challenge_key:
        where.append("c.challenge_key = ?")
        params.append(challenge_key)
    if market:
        where.append("ct.market = ?")
        params.append(market)
    if trade_ids:
        placeholders = ",".join("?" for _ in trade_ids)
        where.append(f"ct.id IN ({placeholders})")
        params.extend(trade_ids)

    cursor.execute(
        f"""
        SELECT
            ct.*,
            c.challenge_key,
            c.title AS challenge_title,
            a.name AS agent_name
        FROM challenge_trades ct
        JOIN challenges c ON c.id = ct.challenge_id
        JOIN agents a ON a.id = ct.agent_id
        WHERE {' AND '.join(where)}
        ORDER BY c.challenge_key, ct.agent_id, ct.executed_at, ct.id
        """,
        params,
    )
    return [row_dict(row) for row in cursor.fetchall()]


def build_price_repair(row: dict[str, Any], quote: float, repaired_at: str) -> dict[str, Any] | None:
    old_price = as_float(row.get("price"))
    old_quantity = as_float(row.get("quantity"))
    if old_price <= 0 or old_quantity <= 0 or quote <= 0:
        return None
    if relative_diff(old_price, quote) <= PRICE_REL_TOLERANCE:
        return None
    if mismatch_ratio(old_price, quote) < MIN_MISMATCH_RATIO:
        return None

    reason = "server_price_correction"
    new_price = quote
    new_quantity = old_quantity
    if relative_diff(old_quantity, quote) <= SWAP_REL_TOLERANCE:
        reason = "price_quantity_swapped"
        new_quantity = old_price

    return {
        "trade_id": row["id"],
        "challenge_key": row["challenge_key"],
        "agent_id": row["agent_id"],
        "agent_name": row["agent_name"],
        "market": row["market"],
        "symbol": row["symbol"],
        "side": row["side"],
        "executed_at": row["executed_at"],
        "old_price": old_price,
        "old_quantity": old_quantity,
        "new_price": new_price,
        "new_quantity": new_quantity,
        "old_symbol": row["symbol"],
        "new_symbol": row["symbol"],
        "quote": quote,
        "reason": reason,
        "repaired_at": repaired_at,
    }


def build_symbol_repair(
    row: dict[str, Any],
    candidate_symbols: set[str],
    quote_cache: dict[tuple[str, str, str], float | None],
    repaired_at: str,
) -> dict[str, Any] | None:
    if str(row.get("market") or "") != "crypto":
        return None
    old_price = as_float(row.get("price"))
    old_quantity = as_float(row.get("quantity"))
    if old_price <= 0 or old_quantity <= 0:
        return None

    current_symbol = str(row.get("symbol") or "").strip().upper()
    best_symbol = None
    best_quote = 0.0
    best_diff = float("inf")
    with price_fetch_logging(False):
        for symbol in sorted(candidate_symbols):
            if not symbol or symbol == current_symbol:
                continue
            key = ("crypto", symbol, str(row.get("executed_at") or ""))
            if key not in quote_cache:
                try:
                    quote = get_price_from_market(symbol, key[2], "crypto")
                    quote_cache[key] = as_float(quote) if quote is not None else None
                except Exception:
                    quote_cache[key] = None
            quote = quote_cache[key]
            if not quote:
                continue
            diff = relative_diff(old_price, float(quote))
            if diff < best_diff:
                best_symbol = symbol
                best_quote = float(quote)
                best_diff = diff

    if not best_symbol or best_diff > PRICE_REL_TOLERANCE:
        return None
    return {
        "trade_id": row["id"],
        "challenge_key": row["challenge_key"],
        "agent_id": row["agent_id"],
        "agent_name": row["agent_name"],
        "market": row["market"],
        "symbol": row["symbol"],
        "side": row["side"],
        "executed_at": row["executed_at"],
        "old_price": old_price,
        "old_quantity": old_quantity,
        "new_price": best_quote,
        "new_quantity": old_quantity,
        "old_symbol": row["symbol"],
        "new_symbol": best_symbol,
        "quote": best_quote,
        "reason": "invalid_symbol_inferred",
        "repaired_at": repaired_at,
    }


def candidate_symbols_by_challenge(rows: list[dict[str, Any]]) -> dict[str, set[str]]:
    result: dict[str, set[str]] = {}
    for row in rows:
        if row.get("market") != "crypto":
            continue
        challenge_key = str(row.get("challenge_key") or "")
        symbol = str(row.get("symbol") or "").strip().upper()
        if symbol and symbol not in {"ALL", "PORTFOLIO", "STRATEGY"} and not symbol.startswith("0X"):
            result.setdefault(challenge_key, set()).add(symbol)
    for challenge_key in {str(row.get("challenge_key") or "") for row in rows}:
        result.setdefault(challenge_key, set()).update(DEFAULT_CRYPTO_SYMBOL_CANDIDATES)
    return result


def load_trade_event(cursor: Any, trade_id: int) -> dict[str, Any] | None:
    cursor.execute(
        """
        SELECT *
        FROM experiment_events
        WHERE event_type = 'challenge_trade_submitted'
          AND object_type = 'challenge_trade'
          AND object_id = ?
        ORDER BY id DESC
        LIMIT 1
        """,
        (str(trade_id),),
    )
    row = cursor.fetchone()
    return row_dict(row) if row else None


def update_trade_event(cursor: Any, repair: dict[str, Any]) -> None:
    event = load_trade_event(cursor, int(repair["trade_id"]))
    if not event:
        return
    try:
        metadata = json.loads(event.get("metadata_json") or "{}")
        if not isinstance(metadata, dict):
            metadata = {}
    except Exception:
        metadata = {}

    metadata.update(
        {
            "price": repair["new_price"],
            "quantity": repair["new_quantity"],
            "symbol": repair["new_symbol"],
            "repaired_at": repair["repaired_at"],
            "repair_reason": repair["reason"],
            "previous_price": repair["old_price"],
            "previous_quantity": repair["old_quantity"],
            "previous_symbol": repair["old_symbol"],
        }
    )
    cursor.execute(
        "UPDATE experiment_events SET metadata_json = ? WHERE id = ?",
        (json.dumps(metadata, ensure_ascii=True, sort_keys=True), event["id"]),
    )


def write_backup(repairs: list[dict[str, Any]], rows: list[dict[str, Any]], events: list[dict[str, Any]]) -> str:
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = BACKUP_DIR / f"challenge_trade_price_repair_backup_{stamp}.json"
    path.write_text(
        json.dumps(
            {
                "captured_at": utc_now_iso_z(),
                "repairs": repairs,
                "challenge_trades": rows,
                "experiment_events": events,
            },
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
            default=str,
        ),
        encoding="utf-8",
    )
    return str(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="Apply repairs. Defaults to dry-run.")
    parser.add_argument("--challenge-key", help="Limit repair to one challenge key.")
    parser.add_argument("--market", choices=sorted(SUPPORTED_MARKETS), help="Limit repair to one market.")
    parser.add_argument("--trade-id", action="append", type=int, default=[], help="Limit repair to specific trade id; repeatable.")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of scanned rows after filtering.")
    args = parser.parse_args()

    repaired_at = utc_now_iso_z()
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        rows = fetch_trade_rows(cursor, args.challenge_key, args.market, args.trade_id)
        if args.limit and args.limit > 0:
            rows = rows[: args.limit]

        repairs: list[dict[str, Any]] = []
        quote_cache: dict[tuple[str, str, str], float | None] = {}
        candidate_symbols = candidate_symbols_by_challenge(rows)
        with price_fetch_logging(False):
            for row in rows:
                key = (str(row.get("market") or ""), str(row.get("symbol") or ""), str(row.get("executed_at") or ""))
                if key not in quote_cache:
                    try:
                        quote = get_price_from_market(key[1], key[2], key[0])
                        quote_cache[key] = as_float(quote) if quote is not None else None
                    except Exception:
                        quote_cache[key] = None
                quote = quote_cache[key]
                if quote:
                    repair = build_price_repair(row, float(quote), repaired_at)
                else:
                    repair = build_symbol_repair(
                        row,
                        candidate_symbols.get(str(row.get("challenge_key") or ""), set()),
                        quote_cache,
                        repaired_at,
                    )
                if repair:
                    repairs.append(repair)

        events = []
        if repairs:
            ids = [int(repair["trade_id"]) for repair in repairs]
            placeholders = ",".join("?" for _ in ids)
            cursor.execute(
                f"""
                SELECT *
                FROM experiment_events
                WHERE event_type = 'challenge_trade_submitted'
                  AND object_type = 'challenge_trade'
                  AND object_id IN ({placeholders})
                ORDER BY id
                """,
                [str(item) for item in ids],
            )
            events = [row_dict(row) for row in cursor.fetchall()]

        summary = {
            "captured_at": repaired_at,
            "mode": "apply" if args.apply else "dry-run",
            "scanned_trade_count": len(rows),
            "repair_count": len(repairs),
            "reasons": dict(Counter(repair["reason"] for repair in repairs)),
            "repairs": repairs,
        }

        if not args.apply or not repairs:
            print(json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True, default=str))
            return 0

        backup_path = write_backup(repairs, [row for row in rows if int(row["id"]) in {int(r["trade_id"]) for r in repairs}], events)
        conn.commit()
        begin_write_transaction(cursor)
        for repair in repairs:
            cursor.execute(
                """
                UPDATE challenge_trades
                SET symbol = ?, price = ?, quantity = ?
                WHERE id = ?
                """,
                (repair["new_symbol"], repair["new_price"], repair["new_quantity"], repair["trade_id"]),
            )
            update_trade_event(cursor, repair)
        conn.commit()
        summary["backup_path"] = backup_path
        print(json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True, default=str))
        return 0
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())

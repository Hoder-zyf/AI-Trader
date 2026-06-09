import csv
import os
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch


SERVER_DIR = Path(__file__).resolve().parents[1]
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

import database
from challenges import (
    ChallengeError,
    create_challenge_trade,
    get_agent_challenge_portfolio,
    get_challenge_leaderboard,
    create_challenge,
    join_challenge,
    list_challenges,
    settle_challenge,
    settle_due_challenges,
)
from challenge_scoring import score_challenge_results
from research_exports import export_challenge_tables
from routes_shared import utc_now_iso_z


def iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


class ChallengeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        database.DATABASE_URL = ""
        database._SQLITE_DB_PATH = os.path.join(self.tmp.name, "test.db")
        database.init_database()
        self.agent_1 = self._create_agent("agent-1")
        self.agent_2 = self._create_agent("agent-2")
        self.agent_3 = self._create_agent("agent-3")

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _create_agent(self, name: str) -> int:
        conn = database.get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO agents (name, token, points, cash, created_at, updated_at)
            VALUES (?, ?, 0, 100000.0, ?, ?)
            """,
            (name, f"token-{name}", utc_now_iso_z(), utc_now_iso_z()),
        )
        agent_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return agent_id

    def _create_active_challenge(self, **overrides):
        now = datetime.now(timezone.utc)
        payload = {
            "challenge_key": overrides.pop("challenge_key", f"test-{datetime.now().timestamp()}").replace(".", "-"),
            "title": "BTC sprint",
            "market": "crypto",
            "symbol": "BTC",
            "challenge_type": "multi-agent",
            "scoring_method": "return-only",
            "initial_capital": 1000.0,
            "max_position_pct": 100.0,
            "max_drawdown_pct": 20.0,
            "start_at": iso(now - timedelta(minutes=5)),
            "end_at": iso(now + timedelta(hours=1)),
            "rules_json": {"reward_points": {"1": 100, "2": 25}},
        }
        payload.update(overrides)
        return create_challenge(payload, self.agent_1)

    def _submit_challenge_trade(
        self,
        challenge_key: str,
        agent_id: int,
        side: str,
        price: float,
        quantity: float,
        symbol: str = "BTC",
    ):
        with patch("price_fetcher.get_price_from_market", return_value=price):
            return create_challenge_trade(
                challenge_key,
                agent_id=agent_id,
                data={
                    "symbol": symbol,
                    "side": side,
                    "price": price,
                    "quantity": quantity,
                    "executed_at": iso(datetime.now(timezone.utc)),
                },
            )

    def test_create_and_join_challenge_is_idempotent(self):
        challenge = self._create_active_challenge(challenge_key="join-check")

        first = join_challenge(challenge["challenge_key"], self.agent_2)
        second = join_challenge(challenge["challenge_key"], self.agent_2)

        self.assertTrue(first["joined"])
        self.assertFalse(second["joined"])
        self.assertTrue(second["idempotent"])

        conn = database.get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) AS count FROM challenge_participants WHERE challenge_id = ?", (challenge["id"],))
        self.assertEqual(cursor.fetchone()["count"], 1)
        cursor.execute("SELECT event_type FROM experiment_events ORDER BY id")
        self.assertIn("challenge_created", [row["event_type"] for row in cursor.fetchall()])
        conn.close()

    def test_challenges_are_filtered_by_track(self):
        self._create_active_challenge(challenge_key="track-crypto", market="crypto", symbol="BTC")
        self._create_active_challenge(challenge_key="track-stock", market="us-stock", symbol="AAPL")
        self._create_active_challenge(challenge_key="track-polymarket", market="polymarket", symbol="election-market")

        all_tracks = list_challenges(status="active", market="all")
        self.assertEqual(all_tracks["total"], 3)

        stock_track = list_challenges(status="active", market="us-stock")
        self.assertEqual(stock_track["total"], 1)
        self.assertEqual(stock_track["challenges"][0]["challenge_key"], "track-stock")
        self.assertEqual(stock_track["challenges"][0]["market"], "us-stock")

        polymarket_track = list_challenges(status="active", market="polymarket")
        self.assertEqual(polymarket_track["total"], 1)
        self.assertEqual(polymarket_track["challenges"][0]["challenge_key"], "track-polymarket")

    def test_challenge_track_must_be_supported(self):
        with self.assertRaises(ChallengeError):
            self._create_active_challenge(challenge_key="track-forex", market="forex", symbol="EURUSD")

        with self.assertRaises(ChallengeError):
            list_challenges(status="active", market="forex")

    def test_dedicated_challenge_trade_records_isolated_snapshot_and_portfolio(self):
        challenge = self._create_active_challenge(challenge_key="dedicated-trade")
        join_challenge(challenge["challenge_key"], self.agent_2)

        with patch("price_fetcher.get_price_from_market", return_value=100.0):
            result = create_challenge_trade(
                challenge["challenge_key"],
                agent_id=self.agent_2,
                data={
                    "symbol": "BTC",
                    "side": "buy",
                    "price": 1.0,
                    "quantity": 2.0,
                    "executed_at": iso(datetime.now(timezone.utc)),
                },
            )

        self.assertIsNone(result["trade"]["source_signal_id"])
        self.assertEqual(result["trade"]["price"], 100.0)
        self.assertEqual(result["portfolio"]["trade_count"], 1)
        self.assertAlmostEqual(result["portfolio"]["cash"], 800.0)
        self.assertEqual(result["portfolio"]["positions"][0]["quantity"], 2.0)
        conn = database.get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM challenge_trades WHERE source_signal_id IS NULL")
        row = cursor.fetchone()
        self.assertIsNotNone(row)
        self.assertEqual(row["challenge_id"], challenge["id"])
        self.assertEqual(row["agent_id"], self.agent_2)
        cursor.execute("SELECT COUNT(*) AS count FROM positions WHERE agent_id = ?", (self.agent_2,))
        self.assertEqual(cursor.fetchone()["count"], 0)
        cursor.execute("SELECT cash FROM agents WHERE id = ?", (self.agent_2,))
        self.assertEqual(cursor.fetchone()["cash"], 100000.0)
        cursor.execute("SELECT COUNT(*) AS count FROM experiment_events WHERE event_type = 'challenge_trade_submitted'")
        self.assertEqual(cursor.fetchone()["count"], 1)
        conn.close()

        portfolio = get_agent_challenge_portfolio(challenge["challenge_key"], self.agent_2)
        self.assertEqual(portfolio["portfolio"]["trade_count"], 1)

    def test_dedicated_challenge_trade_requires_server_price(self):
        challenge = self._create_active_challenge(challenge_key="dedicated-trade-no-price")
        join_challenge(challenge["challenge_key"], self.agent_2)

        with patch("price_fetcher.get_price_from_market", return_value=None):
            with self.assertRaises(ChallengeError):
                create_challenge_trade(
                    challenge["challenge_key"],
                    agent_id=self.agent_2,
                    data={
                        "symbol": "BTC",
                        "side": "buy",
                        "price": 1.0,
                        "quantity": 2.0,
                        "executed_at": iso(datetime.now(timezone.utc)),
                    },
                )

        conn = database.get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) AS count FROM challenge_trades WHERE challenge_id = ?", (challenge["id"],))
        self.assertEqual(cursor.fetchone()["count"], 0)
        conn.close()

    def test_active_leaderboard_marks_open_positions_to_market(self):
        challenge = self._create_active_challenge(challenge_key="live-mark-leaderboard")
        join_challenge(challenge["challenge_key"], self.agent_2)
        join_challenge(challenge["challenge_key"], self.agent_3)
        self._submit_challenge_trade(challenge["challenge_key"], self.agent_2, "buy", 100.0, 10.0)

        with patch("price_fetcher.get_price_from_market", return_value=90.0) as mock_price:
            result = get_challenge_leaderboard(challenge["challenge_key"])

        self.assertTrue(result["provisional"])
        mock_price.assert_called_once()
        marked_row = next(row for row in result["leaderboard"] if row["agent_id"] == self.agent_2)
        self.assertAlmostEqual(marked_row["return_pct"], -10.0)
        self.assertAlmostEqual(marked_row["max_drawdown"], 10.0)
        self.assertEqual(marked_row["trade_count"], 1)
        self.assertTrue(marked_row["metrics"]["marked_to_market"])
        self.assertEqual(marked_row["metrics"]["live_marks"][0]["price"], 90.0)

    def test_active_leaderboard_ignores_stale_results_and_marks_to_market(self):
        challenge = self._create_active_challenge(challenge_key="live-mark-stale-results")
        join_challenge(challenge["challenge_key"], self.agent_2)
        self._submit_challenge_trade(challenge["challenge_key"], self.agent_2, "buy", 100.0, 10.0)

        conn = database.get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO challenge_results
            (challenge_id, agent_id, return_pct, max_drawdown, risk_adjusted_score,
             quality_score, final_score, rank, metrics_json, settled_at)
            VALUES (?, ?, 0, 0, 0, 0, 0, 1, '{}', ?)
            """,
            (challenge["id"], self.agent_2, utc_now_iso_z()),
        )
        conn.commit()
        conn.close()

        with patch("price_fetcher.get_price_from_market", return_value=120.0):
            result = get_challenge_leaderboard(challenge["challenge_key"])

        self.assertTrue(result["provisional"])
        row = next(item for item in result["leaderboard"] if item["agent_id"] == self.agent_2)
        self.assertAlmostEqual(row["return_pct"], 20.0)
        self.assertAlmostEqual(row["max_drawdown"], 0.0)

    def test_active_portfolio_marks_open_positions_to_market(self):
        challenge = self._create_active_challenge(challenge_key="live-mark-portfolio")
        join_challenge(challenge["challenge_key"], self.agent_2)
        self._submit_challenge_trade(challenge["challenge_key"], self.agent_2, "buy", 100.0, 10.0)

        with patch("price_fetcher.get_price_from_market", return_value=120.0):
            portfolio = get_agent_challenge_portfolio(challenge["challenge_key"], self.agent_2)

        self.assertAlmostEqual(portfolio["portfolio"]["ending_value"], 1200.0)
        self.assertAlmostEqual(portfolio["portfolio"]["return_pct"], 20.0)
        self.assertAlmostEqual(portfolio["portfolio"]["max_drawdown"], 0.0)
        self.assertTrue(portfolio["portfolio"]["marked_to_market"])
        self.assertEqual(portfolio["portfolio"]["live_marks"][0]["price"], 120.0)

    def test_settlement_does_not_use_live_marks_for_open_positions(self):
        challenge = self._create_active_challenge(challenge_key="settle-open-snapshot")
        join_challenge(challenge["challenge_key"], self.agent_2)
        self._submit_challenge_trade(challenge["challenge_key"], self.agent_2, "buy", 100.0, 10.0)

        with patch("price_fetcher.get_price_from_market", return_value=120.0) as mock_price:
            result = settle_challenge(challenge["challenge_key"])

        mock_price.assert_not_called()
        row = next(item for item in result["leaderboard"] if item["agent_id"] == self.agent_2)
        self.assertAlmostEqual(row["return_pct"], 0.0)
        self.assertAlmostEqual(row["max_drawdown"], 0.0)

    def test_due_challenge_settles_return_ranks_rewards_and_exports(self):
        challenge = self._create_active_challenge(challenge_key="settle-return")
        join_challenge(challenge["challenge_key"], self.agent_2)
        join_challenge(challenge["challenge_key"], self.agent_3)
        self._submit_challenge_trade(challenge["challenge_key"], self.agent_2, "buy", 100.0, 10.0)
        self._submit_challenge_trade(challenge["challenge_key"], self.agent_2, "sell", 110.0, 10.0)
        self._submit_challenge_trade(challenge["challenge_key"], self.agent_3, "buy", 100.0, 10.0)
        self._submit_challenge_trade(challenge["challenge_key"], self.agent_3, "sell", 105.0, 10.0)

        conn = database.get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE challenges SET end_at = ? WHERE id = ?",
            (iso(datetime.now(timezone.utc) - timedelta(seconds=1)), challenge["id"]),
        )
        conn.commit()
        conn.close()

        settled = settle_due_challenges()
        self.assertEqual(len(settled), 1)

        leaderboard = settled[0]["leaderboard"]
        self.assertEqual(leaderboard[0]["agent_id"], self.agent_2)
        self.assertEqual(leaderboard[0]["rank"], 1)
        self.assertAlmostEqual(leaderboard[0]["return_pct"], 10.0)
        self.assertEqual(leaderboard[1]["rank"], 2)

        conn = database.get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT points FROM agents WHERE id = ?", (self.agent_2,))
        self.assertEqual(cursor.fetchone()["points"], 100)
        cursor.execute("SELECT points FROM agents WHERE id = ?", (self.agent_3,))
        self.assertEqual(cursor.fetchone()["points"], 25)
        cursor.execute("SELECT event_type FROM experiment_events")
        event_types = {row["event_type"] for row in cursor.fetchall()}
        self.assertTrue({
            "challenge_created",
            "challenge_joined",
            "challenge_trade_submitted",
            "challenge_settled",
            "challenge_reward_granted",
        }.issubset(event_types))
        conn.close()

        export_dir = Path(self.tmp.name) / "exports"
        paths = export_challenge_tables(export_dir, challenge_key=challenge["challenge_key"])
        self.assertIn("challenge_results.csv", paths)
        with open(paths["challenge_results.csv"], newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 2)

    def test_return_only_settlement_records_max_drawdown(self):
        challenge = self._create_active_challenge(challenge_key="return-drawdown")
        join_challenge(challenge["challenge_key"], self.agent_2)
        self._submit_challenge_trade(challenge["challenge_key"], self.agent_2, "buy", 100.0, 10.0)
        self._submit_challenge_trade(challenge["challenge_key"], self.agent_2, "sell", 50.0, 1.0)
        self._submit_challenge_trade(challenge["challenge_key"], self.agent_2, "sell", 120.0, 9.0)

        result = settle_challenge(challenge["challenge_key"])
        row = next(item for item in result["leaderboard"] if item["agent_id"] == self.agent_2)

        self.assertAlmostEqual(row["return_pct"], 13.0)
        self.assertAlmostEqual(row["max_drawdown"], 50.0)
        self.assertAlmostEqual(row["final_score"], 13.0)

        conn = database.get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT cp.max_drawdown AS participant_drawdown,
                   cr.max_drawdown AS result_drawdown
            FROM challenge_participants cp
            JOIN challenge_results cr ON cr.challenge_id = cp.challenge_id AND cr.agent_id = cp.agent_id
            WHERE cp.challenge_id = ? AND cp.agent_id = ?
            """,
            (challenge["id"], self.agent_2),
        )
        stored = cursor.fetchone()
        conn.close()
        self.assertIsNotNone(stored)
        self.assertAlmostEqual(stored["participant_drawdown"], 50.0)
        self.assertAlmostEqual(stored["result_drawdown"], 50.0)

    def test_risk_adjusted_ranking_penalizes_drawdown(self):
        challenge = {
            "id": 1,
            "initial_capital": 1000.0,
            "scoring_method": "risk-adjusted",
            "max_position_pct": 100.0,
            "max_drawdown_pct": 5.0,
            "rules_json": '{"allowed_drawdown": 5, "drawdown_penalty": 1}',
        }
        participants = [
            {"agent_id": 1, "starting_cash": 1000.0, "status": "joined"},
            {"agent_id": 2, "starting_cash": 1000.0, "status": "joined"},
        ]
        trades_by_agent = {
            1: [
                {"id": 1, "market": "crypto", "symbol": "BTC", "side": "buy", "price": 100.0, "quantity": 10, "executed_at": "2026-01-01T00:00:00Z"},
                {"id": 2, "market": "crypto", "symbol": "BTC", "side": "sell", "price": 50.0, "quantity": 1, "executed_at": "2026-01-01T00:01:00Z"},
                {"id": 3, "market": "crypto", "symbol": "BTC", "side": "sell", "price": 160.0, "quantity": 9, "executed_at": "2026-01-01T00:02:00Z"},
            ],
            2: [
                {"id": 4, "market": "crypto", "symbol": "BTC", "side": "buy", "price": 100.0, "quantity": 10, "executed_at": "2026-01-01T00:00:00Z"},
                {"id": 5, "market": "crypto", "symbol": "BTC", "side": "sell", "price": 110.0, "quantity": 10, "executed_at": "2026-01-01T00:01:00Z"},
            ],
        }

        ranked = score_challenge_results(challenge, participants, trades_by_agent)
        rank_by_agent = {row["agent_id"]: row["rank"] for row in ranked}

        self.assertEqual(rank_by_agent[2], 1)
        self.assertEqual(rank_by_agent[1], 2)
        high_drawdown = next(row for row in ranked if row["agent_id"] == 1)
        self.assertAlmostEqual(high_drawdown["return_pct"], 49.0)
        self.assertGreater(high_drawdown["max_drawdown"], 40.0)

    def test_disqualified_agent_gets_no_challenge_reward(self):
        challenge = self._create_active_challenge(
            challenge_key="disqualified-no-reward",
            max_position_pct=50.0,
            rules_json={"reward_points": {"1": 100, "2": 50}},
        )
        join_challenge(challenge["challenge_key"], self.agent_2)
        join_challenge(challenge["challenge_key"], self.agent_3)
        self._submit_challenge_trade(challenge["challenge_key"], self.agent_2, "buy", 100.0, 10.0)
        self._submit_challenge_trade(challenge["challenge_key"], self.agent_3, "buy", 100.0, 1.0)
        result = settle_challenge(challenge["challenge_key"])

        disqualified = next(row for row in result["leaderboard"] if row["agent_id"] == self.agent_2)
        self.assertEqual(disqualified["rank"], None)
        self.assertIn("max_position_pct", disqualified["disqualified_reason"])

        conn = database.get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT points FROM agents WHERE id = ?", (self.agent_2,))
        self.assertEqual(cursor.fetchone()["points"], 0)
        cursor.execute(
            """
            SELECT COUNT(*) AS count
            FROM agent_reward_ledger
            WHERE agent_id = ? AND source_type = 'challenge'
            """,
            (self.agent_2,),
        )
        self.assertEqual(cursor.fetchone()["count"], 0)
        cursor.execute("SELECT COUNT(*) AS count FROM experiment_events WHERE event_type = 'challenge_disqualified'")
        self.assertEqual(cursor.fetchone()["count"], 1)
        conn.close()

    def test_twenty_agent_challenge_settles_with_complete_metrics(self):
        challenge = self._create_active_challenge(
            challenge_key="twenty-agent-active",
            rules_json={"reward_points": {"1": 100, "2": 50, "3": 25}},
        )
        agent_ids = [self._create_agent(f"bulk-agent-{idx}") for idx in range(20)]

        for idx, agent_id in enumerate(agent_ids):
            join_challenge(challenge["challenge_key"], agent_id)
            self._submit_challenge_trade(challenge["challenge_key"], agent_id, "buy", 100.0, 10.0)
            self._submit_challenge_trade(challenge["challenge_key"], agent_id, "sell", 100.0 + idx, 10.0)

        result = settle_challenge(challenge["challenge_key"])
        leaderboard = result["leaderboard"]

        self.assertEqual(len(leaderboard), 20)
        self.assertEqual(leaderboard[0]["agent_id"], agent_ids[-1])
        self.assertEqual([row["rank"] for row in leaderboard], list(range(1, 21)))
        for row in leaderboard:
            self.assertIsNotNone(row["return_pct"])
            self.assertIsNotNone(row["max_drawdown"])
            self.assertEqual(row["trade_count"], 2)


if __name__ == "__main__":
    unittest.main()

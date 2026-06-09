import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient


SERVER_DIR = Path(__file__).resolve().parents[1]
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

import database
from routes import create_app
from utils import hash_password


class AgentLoginTokenStabilityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        database.DATABASE_URL = ""
        database._SQLITE_DB_PATH = os.path.join(self.tmp.name, "test.db")
        database.init_database()
        self.client = TestClient(create_app())

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_agent_login_returns_existing_token_without_rotating(self) -> None:
        register = self.client.post(
            "/api/claw/agents/selfRegister",
            json={"name": "stable-agent", "password": "password123"},
        )
        self.assertEqual(register.status_code, 200, register.text)
        original_token = register.json()["token"]

        first_login = self.client.post(
            "/api/claw/agents/login",
            json={"name": "stable-agent", "password": "password123"},
        )
        second_login = self.client.post(
            "/api/claw/agents/login",
            json={"name": "stable-agent", "password": "password123"},
        )

        self.assertEqual(first_login.status_code, 200, first_login.text)
        self.assertEqual(second_login.status_code, 200, second_login.text)
        self.assertEqual(first_login.json()["token"], original_token)
        self.assertEqual(second_login.json()["token"], original_token)

        me = self.client.get(
            "/api/claw/agents/me",
            headers={"Authorization": f"Bearer {original_token}"},
        )
        self.assertEqual(me.status_code, 200, me.text)
        self.assertIsNone(me.json()["email"])

    def test_agent_registration_stores_normalized_email(self) -> None:
        register = self.client.post(
            "/api/claw/agents/selfRegister",
            json={
                "name": "email-agent",
                "email": "  Trader@Example.COM  ",
                "password": "password123",
            },
        )
        self.assertEqual(register.status_code, 200, register.text)
        self.assertEqual(register.json()["email"], "trader@example.com")
        token = register.json()["token"]

        me = self.client.get(
            "/api/claw/agents/me",
            headers={"Authorization": f"Bearer {token}"},
        )
        self.assertEqual(me.status_code, 200, me.text)
        self.assertEqual(me.json()["email"], "trader@example.com")

        conn = database.get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT email FROM agents WHERE name = ?", ("email-agent",))
        self.assertEqual(cursor.fetchone()["email"], "trader@example.com")
        conn.close()

    def test_registration_tracks_extra_initial_cash_as_deposit(self) -> None:
        register = self.client.post(
            "/api/claw/agents/selfRegister",
            json={
                "name": "funded-agent",
                "password": "password123",
                "initial_balance": 150000,
            },
        )
        self.assertEqual(register.status_code, 200, register.text)
        self.assertEqual(register.json()["deposited"], 50000.0)

        conn = database.get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT cash, deposited FROM agents WHERE name = ?", ("funded-agent",))
        row = cursor.fetchone()
        conn.close()

        self.assertEqual(float(row["cash"]), 150000.0)
        self.assertEqual(float(row["deposited"]), 50000.0)

    def test_registration_marks_initial_positions_to_server_quote(self) -> None:
        with patch("price_fetcher.get_price_from_market", return_value=250.0):
            register = self.client.post(
                "/api/claw/agents/selfRegister",
                json={
                    "name": "position-agent",
                    "password": "password123",
                    "initial_balance": 100000,
                    "positions": [
                        {
                            "market": "us-stock",
                            "symbol": "tsla",
                            "side": "long",
                            "quantity": 2,
                            "entry_price": 1,
                        }
                    ],
                },
            )

        self.assertEqual(register.status_code, 200, register.text)
        self.assertEqual(register.json()["deposited"], 500.0)

        conn = database.get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT id, cash, deposited FROM agents WHERE name = ?", ("position-agent",))
        agent = cursor.fetchone()
        cursor.execute(
            "SELECT symbol, market, quantity, entry_price, current_price FROM positions WHERE agent_id = ?",
            (agent["id"],),
        )
        position = cursor.fetchone()
        conn.close()

        self.assertEqual(float(agent["cash"]), 100000.0)
        self.assertEqual(float(agent["deposited"]), 500.0)
        self.assertEqual(position["symbol"], "TSLA")
        self.assertEqual(position["market"], "us-stock")
        self.assertEqual(float(position["quantity"]), 2.0)
        self.assertEqual(float(position["entry_price"]), 250.0)
        self.assertEqual(float(position["current_price"]), 250.0)

    def test_registration_rejects_initial_position_without_server_quote(self) -> None:
        with patch("price_fetcher.get_price_from_market", return_value=None):
            register = self.client.post(
                "/api/claw/agents/selfRegister",
                json={
                    "name": "bad-position-agent",
                    "password": "password123",
                    "positions": [
                        {
                            "market": "us-stock",
                            "symbol": "tsla",
                            "side": "long",
                            "quantity": 2,
                            "entry_price": 1,
                        }
                    ],
                },
            )

        self.assertEqual(register.status_code, 400, register.text)
        self.assertIn("Unable to fetch current price for TSLA", register.json()["detail"])

        conn = database.get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) AS count FROM agents WHERE name = ?", ("bad-position-agent",))
        self.assertEqual(int(cursor.fetchone()["count"]), 0)
        conn.close()

    def test_agent_identity_defaults_normal_and_can_be_verified_manually(self) -> None:
        register = self.client.post(
            "/api/claw/agents/selfRegister",
            json={"name": "identity-agent", "password": "password123"},
        )
        self.assertEqual(register.status_code, 200, register.text)
        token = register.json()["token"]
        self.assertEqual(register.json()["identity_status"], "normal")
        self.assertFalse(register.json()["is_verified"])

        me = self.client.get(
            "/api/claw/agents/me",
            headers={"Authorization": f"Bearer {token}"},
        )
        self.assertEqual(me.status_code, 200, me.text)
        self.assertEqual(me.json()["identity_status"], "normal")
        self.assertFalse(me.json()["is_verified"])

        conn = database.get_db_connection()
        cursor = conn.cursor()
        cursor.execute("UPDATE agents SET identity_status = 'verified' WHERE name = ?", ("identity-agent",))
        conn.commit()
        conn.close()

        verified_me = self.client.get(
            "/api/claw/agents/me",
            headers={"Authorization": f"Bearer {token}"},
        )
        self.assertEqual(verified_me.status_code, 200, verified_me.text)
        self.assertEqual(verified_me.json()["identity_status"], "verified")
        self.assertTrue(verified_me.json()["is_verified"])

    def test_agent_login_issues_token_only_for_legacy_empty_token(self) -> None:
        conn = database.get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO agents (name, password_hash, token, cash)
            VALUES (?, ?, NULL, 100000.0)
            """,
            ("legacy-empty-token", hash_password("password123")),
        )
        conn.commit()
        conn.close()

        first_login = self.client.post(
            "/api/claw/agents/login",
            json={"name": "legacy-empty-token", "password": "password123"},
        )
        second_login = self.client.post(
            "/api/claw/agents/login",
            json={"name": "legacy-empty-token", "password": "password123"},
        )

        self.assertEqual(first_login.status_code, 200, first_login.text)
        self.assertEqual(second_login.status_code, 200, second_login.text)
        self.assertTrue(first_login.json()["token"])
        self.assertEqual(second_login.json()["token"], first_login.json()["token"])

    def test_new_registration_normalizes_agent_name(self) -> None:
        response = self.client.post(
            "/api/claw/agents/selfRegister",
            json={"name": "  normalized-agent  ", "password": "password123"},
        )
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["name"], "normalized-agent")

        login = self.client.post(
            "/api/claw/agents/login",
            json={"name": "normalized-agent", "password": "password123"},
        )
        self.assertEqual(login.status_code, 200, login.text)

    def test_registration_rejects_normalized_duplicate_of_legacy_spaced_name(self) -> None:
        conn = database.get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO agents (name, password_hash, token, cash)
            VALUES (?, ?, ?, 100000.0)
            """,
            (" legacy-duplicate ", hash_password("password123"), "legacy-token"),
        )
        conn.commit()
        conn.close()

        response = self.client.post(
            "/api/claw/agents/selfRegister",
            json={"name": "legacy-duplicate", "password": "password123"},
        )

        self.assertEqual(response.status_code, 400, response.text)
        self.assertEqual(response.json()["detail"], "Agent name already exists")

    def test_legacy_agent_name_with_spaces_can_still_login_exactly(self) -> None:
        conn = database.get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO agents (name, password_hash, token, cash)
            VALUES (?, ?, ?, 100000.0)
            """,
            (" legacy-spaced ", hash_password("password123"), "legacy-token"),
        )
        conn.commit()
        conn.close()

        exact_login = self.client.post(
            "/api/claw/agents/login",
            json={"name": " legacy-spaced ", "password": "password123"},
        )
        trimmed_login = self.client.post(
            "/api/claw/agents/login",
            json={"name": "legacy-spaced", "password": "password123"},
        )

        self.assertEqual(exact_login.status_code, 200, exact_login.text)
        self.assertEqual(exact_login.json()["token"], "legacy-token")
        self.assertEqual(trimmed_login.status_code, 401, trimmed_login.text)


if __name__ == "__main__":
    unittest.main()

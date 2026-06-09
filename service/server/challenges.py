"""Challenge creation, participation, submission, dedicated trading, and settlement."""

from __future__ import annotations

import json
import math
import re
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from challenge_scoring import score_agent_trades, score_challenge_results
from database import begin_write_transaction, get_db_connection
from experiment_events import record_event
from rewards import grant_agent_reward
from routes_shared import agent_identity_status, agent_is_verified, utc_now_iso_z


class ChallengeError(ValueError):
    pass


class ChallengeNotFound(ChallengeError):
    pass


DEFAULT_CHALLENGE_REWARDS = {'1': 100, '2': 50, '3': 25}
SUPPORTED_SCORING_METHODS = {'return-only', 'risk-adjusted'}
SUPPORTED_CHALLENGE_TRACKS = {'crypto', 'us-stock', 'polymarket'}
AUTHORITATIVE_CHALLENGE_PRICE_MARKETS = {'crypto', 'us-stock'}


def _row_dict(row: Any) -> dict[str, Any]:
    return dict(row) if row is not None and not isinstance(row, dict) else (row or {})


def _model_dump(data: Any) -> dict[str, Any]:
    if isinstance(data, dict):
        return data
    if hasattr(data, 'model_dump'):
        return data.model_dump()
    return dict(data)


def _json_dumps(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def _json_loads(value: Any, default: Any = None) -> Any:
    if value is None or value == '':
        return default
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(value)
    except Exception:
        return default


def _parse_dt(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace('Z', '+00:00')).astimezone(timezone.utc)
    except Exception as exc:
        raise ChallengeError(f'Invalid datetime: {value}') from exc


def _iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace('+00:00', 'Z')


def _live_mark_timestamp(challenge: dict[str, Any]) -> Optional[str]:
    if challenge.get('status') != 'active':
        return None
    try:
        now = datetime.now(timezone.utc)
        start_dt = _parse_dt(challenge.get('start_at'))
        end_dt = _parse_dt(challenge.get('end_at'))
    except Exception:
        return None
    if start_dt and now < start_dt:
        return None
    if end_dt and now >= end_dt:
        return None
    return _iso(now)


def _fetch_live_mark_prices(scored_results: list[dict[str, Any]], mark_timestamp: str) -> dict[tuple[str, str], float]:
    position_keys: set[tuple[str, str]] = set()
    for result in scored_results:
        metrics = result.get('metrics') or {}
        for position in metrics.get('positions') or []:
            market = str(position.get('market') or '').strip()
            symbol = str(position.get('symbol') or '').strip()
            quantity = position.get('quantity')
            try:
                quantity_float = float(quantity)
            except Exception:
                quantity_float = 0.0
            if market and symbol and abs(quantity_float) > 1e-12:
                position_keys.add((market, symbol))

    if not position_keys:
        return {}

    try:
        import price_fetcher
        from price_fetcher import price_fetch_logging
    except Exception:
        return {}

    mark_prices: dict[tuple[str, str], float] = {}
    with price_fetch_logging(False):
        for market, symbol in sorted(position_keys):
            try:
                price = price_fetcher.get_price_from_market(symbol, mark_timestamp, market)
                parsed = float(price) if price is not None else 0.0
            except Exception:
                continue
            if math.isfinite(parsed) and parsed > 0:
                mark_prices[(market, symbol)] = parsed
    return mark_prices


def _fetch_authoritative_challenge_trade_price(market: str, symbol: str, executed_at: str) -> Optional[float]:
    if market not in AUTHORITATIVE_CHALLENGE_PRICE_MARKETS:
        return None

    try:
        import price_fetcher
        from price_fetcher import price_fetch_logging
    except Exception as exc:
        raise ChallengeError('Server price fetcher is unavailable') from exc

    with price_fetch_logging(False):
        try:
            price = price_fetcher.get_price_from_market(symbol, executed_at, market)
        except Exception as exc:
            raise ChallengeError(f'Unable to fetch server price for {symbol}') from exc

    try:
        parsed = float(price) if price is not None else 0.0
    except Exception:
        parsed = 0.0
    if not math.isfinite(parsed) or parsed <= 0:
        raise ChallengeError(f'Unable to fetch server price for {symbol}')
    return parsed


def _score_challenge_results_with_live_marks(
    challenge: dict[str, Any],
    participants: list[dict[str, Any]],
    trades_by_agent: dict[int, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    scored = score_challenge_results(challenge, participants, trades_by_agent)
    mark_timestamp = _live_mark_timestamp(challenge)
    if not mark_timestamp:
        return scored
    mark_prices = _fetch_live_mark_prices(scored, mark_timestamp)
    if not mark_prices:
        return scored
    return score_challenge_results(
        challenge,
        participants,
        trades_by_agent,
        mark_prices=mark_prices,
        mark_timestamp=mark_timestamp,
    )


def _normalize_key(key: Optional[str], title: str) -> str:
    candidate = (key or '').strip().lower()
    if not candidate:
        candidate = re.sub(r'[^a-z0-9]+', '-', title.lower()).strip('-')
        candidate = f'{candidate[:44] or "challenge"}-{uuid.uuid4().hex[:8]}'
    candidate = re.sub(r'[^a-z0-9_\-]+', '-', candidate).strip('-_')
    if not candidate:
        raise ChallengeError('challenge_key is required')
    return candidate[:80]


def _derive_status(start_at: str, end_at: str, requested_status: Optional[str] = None) -> str:
    if requested_status:
        normalized = requested_status.strip().lower()
        if normalized not in {'upcoming', 'active', 'settled', 'canceled'}:
            raise ChallengeError('Unsupported challenge status')
        return normalized
    now = datetime.now(timezone.utc)
    start_dt = _parse_dt(start_at)
    end_dt = _parse_dt(end_at)
    if start_dt and start_dt > now:
        return 'upcoming'
    if end_dt and end_dt <= now:
        return 'active'
    return 'active'


def _normalize_challenge_track(value: Optional[str], *, allow_all: bool = False) -> Optional[str]:
    normalized = (value or '').strip().lower().replace('_', '-')
    if allow_all and (not normalized or normalized == 'all'):
        return None
    if normalized not in SUPPORTED_CHALLENGE_TRACKS:
        raise ChallengeError('Unsupported challenge track')
    return normalized


def _load_challenge(cursor: Any, challenge_key: Optional[str] = None, challenge_id: Optional[int] = None) -> dict[str, Any]:
    if challenge_id is not None:
        cursor.execute("SELECT * FROM challenges WHERE id = ?", (challenge_id,))
    else:
        cursor.execute("SELECT * FROM challenges WHERE challenge_key = ?", (challenge_key,))
    row = cursor.fetchone()
    if not row:
        raise ChallengeNotFound('Challenge not found')
    return _row_dict(row)


def _serialize_challenge(row: Any, participant_count: Optional[int] = None) -> dict[str, Any]:
    data = _row_dict(row)
    if not data:
        return {}
    data['rules'] = _json_loads(data.get('rules_json'), {})
    if participant_count is not None:
        data['participant_count'] = participant_count
    return data


def refresh_challenge_statuses(cursor: Any) -> None:
    now = utc_now_iso_z()
    cursor.execute(
        """
        UPDATE challenges
        SET status = 'active', updated_at = ?
        WHERE status = 'upcoming' AND start_at <= ? AND end_at > ?
        """,
        (now, now, now),
    )


def create_challenge(data: Any, created_by_agent_id: int) -> dict[str, Any]:
    payload = _model_dump(data)
    title = (payload.get('title') or '').strip()
    if not title:
        raise ChallengeError('title is required')

    raw_market = (payload.get('market') or '').strip()
    if not raw_market:
        raise ChallengeError('market is required')
    market = _normalize_challenge_track(raw_market)

    scoring_method = (payload.get('scoring_method') or 'return-only').strip().lower().replace('_', '-')
    if scoring_method not in SUPPORTED_SCORING_METHODS:
        raise ChallengeError('Unsupported scoring_method')

    now_dt = datetime.now(timezone.utc)
    start_at = _iso(_parse_dt(payload.get('start_at')) or now_dt)
    end_at = _iso(_parse_dt(payload.get('end_at')) or (now_dt + timedelta(hours=24)))
    if _parse_dt(end_at) <= _parse_dt(start_at):
        raise ChallengeError('end_at must be after start_at')

    challenge_key = _normalize_key(payload.get('challenge_key'), title)
    status = _derive_status(start_at, end_at, payload.get('status'))
    rules = payload.get('rules_json') or {}
    if isinstance(rules, str):
        rules = _json_loads(rules, {})
    if 'reward_points' not in rules and rules.get('grant_rewards', True):
        rules['reward_points'] = DEFAULT_CHALLENGE_REWARDS

    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        begin_write_transaction(cursor)
        cursor.execute(
            """
            INSERT INTO challenges
            (challenge_key, title, description, market, symbol, challenge_type, status,
             scoring_method, initial_capital, max_position_pct, max_drawdown_pct,
             start_at, end_at, rules_json, experiment_key, created_by_agent_id,
             created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                challenge_key,
                title,
                payload.get('description'),
                market,
                (payload.get('symbol') or '').strip() or None,
                (payload.get('challenge_type') or 'multi-agent').strip(),
                status,
                scoring_method,
                float(payload.get('initial_capital') or 100000.0),
                float(payload.get('max_position_pct') or 100.0),
                float(payload.get('max_drawdown_pct') or 100.0),
                start_at,
                end_at,
                _json_dumps(rules),
                (payload.get('experiment_key') or '').strip() or None,
                created_by_agent_id,
                utc_now_iso_z(),
                utc_now_iso_z(),
            ),
        )
        challenge_id = cursor.lastrowid
        record_event(
            'challenge_created',
            actor_agent_id=created_by_agent_id,
            object_type='challenge',
            object_id=challenge_id,
            market=market,
            experiment_key=(payload.get('experiment_key') or '').strip() or None,
            metadata={'challenge_key': challenge_key, 'scoring_method': scoring_method},
            cursor=cursor,
        )
        conn.commit()
        challenge = _load_challenge(cursor, challenge_id=challenge_id)
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

    return _serialize_challenge(challenge, participant_count=0)


def list_challenges(
    status: Optional[str] = None,
    market: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
) -> dict[str, Any]:
    limit = max(1, min(limit, 200))
    offset = max(0, offset)
    track = _normalize_challenge_track(market, allow_all=True)
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        refresh_challenge_statuses(cursor)
        conn.commit()
        params: list[Any] = []
        conditions: list[str] = []
        if status:
            conditions.append('c.status = ?')
            params.append(status)
        if track:
            conditions.append('c.market = ?')
            params.append(track)
        where = ' AND '.join(conditions) if conditions else '1=1'

        cursor.execute(f"SELECT COUNT(*) AS total FROM challenges c WHERE {where}", params)
        total = cursor.fetchone()['total']
        cursor.execute(
            f"""
            SELECT c.*,
                   (SELECT COUNT(*) FROM challenge_participants cp WHERE cp.challenge_id = c.id) AS participant_count
            FROM challenges c
            WHERE {where}
            ORDER BY c.start_at DESC, c.id DESC
            LIMIT ? OFFSET ?
            """,
            params + [limit, offset],
        )
        rows = [_serialize_challenge(row, row['participant_count']) for row in cursor.fetchall()]
        return {'challenges': rows, 'total': total}
    finally:
        conn.close()


def get_challenge(challenge_key: str) -> dict[str, Any]:
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        refresh_challenge_statuses(cursor)
        conn.commit()
        challenge = _load_challenge(cursor, challenge_key=challenge_key)
        cursor.execute(
            """
            SELECT cp.*, a.name AS agent_name, a.identity_status AS agent_identity_status
            FROM challenge_participants cp
            JOIN agents a ON a.id = cp.agent_id
            WHERE cp.challenge_id = ?
            ORDER BY COALESCE(cp.rank, 999999), cp.joined_at, cp.id
            """,
            (challenge['id'],),
        )
        participants = []
        for row in cursor.fetchall():
            participant = dict(row)
            participant['agent_identity_status'] = agent_identity_status(row)
            participant['agent_is_verified'] = agent_is_verified(row)
            participants.append(participant)
        result = _serialize_challenge(challenge, len(participants))
        result['participants'] = participants
        return result
    finally:
        conn.close()


def _resolve_variant(cursor: Any, experiment_key: Optional[str], agent_id: int, requested_variant: Optional[str]) -> Optional[str]:
    variant_key = (requested_variant or '').strip() or None
    if not experiment_key:
        return variant_key

    cursor.execute(
        """
        SELECT variant_key
        FROM experiment_assignments
        WHERE experiment_key = ? AND unit_type = 'agent' AND unit_id = ?
        """,
        (experiment_key, agent_id),
    )
    row = cursor.fetchone()
    if row:
        return row['variant_key']

    if variant_key:
        cursor.execute(
            """
            INSERT INTO experiment_assignments
            (experiment_key, unit_type, unit_id, variant_key, assignment_reason, metadata_json, created_at)
            VALUES (?, 'agent', ?, ?, 'challenge_join', ?, ?)
            """,
            (experiment_key, agent_id, variant_key, _json_dumps({'source': 'challenge_join'}), utc_now_iso_z()),
        )
    return variant_key


def join_challenge(challenge_key: str, agent_id: int, data: Any = None) -> dict[str, Any]:
    payload = _model_dump(data) if data is not None else {}
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        begin_write_transaction(cursor)
        refresh_challenge_statuses(cursor)
        challenge = _load_challenge(cursor, challenge_key=challenge_key)
        if challenge['status'] not in {'upcoming', 'active'}:
            raise ChallengeError('Challenge is not joinable')

        cursor.execute(
            """
            SELECT cp.*, a.name AS agent_name, a.identity_status AS agent_identity_status
            FROM challenge_participants cp
            JOIN agents a ON a.id = cp.agent_id
            WHERE cp.challenge_id = ? AND cp.agent_id = ?
            """,
            (challenge['id'], agent_id),
        )
        existing = cursor.fetchone()
        if existing:
            participant = dict(existing)
            participant['agent_identity_status'] = agent_identity_status(existing)
            participant['agent_is_verified'] = agent_is_verified(existing)
            conn.commit()
            return {'joined': False, 'idempotent': True, 'participant': participant}

        variant_key = _resolve_variant(cursor, challenge.get('experiment_key'), agent_id, payload.get('variant_key'))
        starting_cash = float(payload.get('starting_cash') or challenge.get('initial_capital') or 100000.0)
        cursor.execute(
            """
            INSERT INTO challenge_participants
            (challenge_id, agent_id, status, variant_key, joined_at, starting_cash)
            VALUES (?, ?, 'joined', ?, ?, ?)
            """,
            (challenge['id'], agent_id, variant_key, utc_now_iso_z(), starting_cash),
        )
        participant_id = cursor.lastrowid
        record_event(
            'challenge_joined',
            actor_agent_id=agent_id,
            object_type='challenge_participant',
            object_id=participant_id,
            market=challenge['market'],
            experiment_key=challenge.get('experiment_key'),
            variant_key=variant_key,
            metadata={'challenge_key': challenge['challenge_key'], 'challenge_id': challenge['id']},
            cursor=cursor,
        )
        conn.commit()

        cursor.execute(
            """
            SELECT cp.*, a.name AS agent_name, a.identity_status AS agent_identity_status
            FROM challenge_participants cp
            JOIN agents a ON a.id = cp.agent_id
            WHERE cp.id = ?
            """,
            (participant_id,),
        )
        participant_row = cursor.fetchone()
        participant = dict(participant_row)
        participant['agent_identity_status'] = agent_identity_status(participant_row)
        participant['agent_is_verified'] = agent_is_verified(participant_row)
        return {'joined': True, 'idempotent': False, 'participant': participant}
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def create_submission(challenge_key: str, agent_id: int, data: Any) -> dict[str, Any]:
    payload = _model_dump(data)
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        begin_write_transaction(cursor)
        challenge = _load_challenge(cursor, challenge_key=challenge_key)
        submission = _create_submission_with_cursor(
            cursor,
            challenge,
            agent_id,
            payload.get('submission_type') or 'manual',
            payload.get('content'),
            payload.get('prediction_json'),
            payload.get('signal_id'),
        )
        conn.commit()
        return submission
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _create_submission_with_cursor(
    cursor: Any,
    challenge: dict[str, Any],
    agent_id: int,
    submission_type: str,
    content: Optional[str],
    prediction_json: Any,
    signal_id: Optional[int] = None,
) -> dict[str, Any]:
    if challenge['status'] not in {'upcoming', 'active'}:
        raise ChallengeError('Challenge is not accepting submissions')

    cursor.execute(
        """
        SELECT *
        FROM challenge_participants
        WHERE challenge_id = ? AND agent_id = ?
        """,
        (challenge['id'], agent_id),
    )
    participant = cursor.fetchone()
    if not participant:
        raise ChallengeError('Agent must join challenge before submitting')

    prediction_text = _json_dumps(prediction_json)
    cursor.execute(
        """
        INSERT INTO challenge_submissions
        (challenge_id, agent_id, signal_id, submission_type, content, prediction_json, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            challenge['id'],
            agent_id,
            signal_id,
            submission_type,
            content,
            prediction_text,
            utc_now_iso_z(),
        ),
    )
    submission_id = cursor.lastrowid
    record_event(
        'challenge_submission_created',
        actor_agent_id=agent_id,
        object_type='challenge_submission',
        object_id=submission_id,
        market=challenge['market'],
        experiment_key=challenge.get('experiment_key'),
        variant_key=participant['variant_key'],
        metadata={
            'challenge_key': challenge['challenge_key'],
            'submission_type': submission_type,
            'signal_id': signal_id,
        },
        cursor=cursor,
    )
    return {
        'id': submission_id,
        'challenge_id': challenge['id'],
        'agent_id': agent_id,
        'signal_id': signal_id,
        'submission_type': submission_type,
        'content': content,
        'prediction_json': prediction_text,
    }


def record_challenge_submission_from_signal(
    cursor: Any,
    *,
    challenge_key: Optional[str],
    agent_id: int,
    signal_id: int,
    submission_type: str,
    content: Optional[str],
    prediction_json: Any = None,
) -> Optional[dict[str, Any]]:
    if not challenge_key:
        return None
    challenge = _load_challenge(cursor, challenge_key=challenge_key)
    return _create_submission_with_cursor(
        cursor,
        challenge,
        agent_id,
        submission_type,
        content,
        prediction_json,
        signal_id,
    )


def _normalize_challenge_trade_symbol(challenge: dict[str, Any], raw_symbol: Any) -> str:
    fixed_symbol = str(challenge.get('symbol') or '').strip()
    requested_symbol = str(raw_symbol or '').strip()
    if fixed_symbol and fixed_symbol.lower() != 'all':
        symbol = fixed_symbol
        if requested_symbol:
            compare_requested = requested_symbol if challenge['market'] == 'polymarket' else requested_symbol.upper()
            compare_fixed = fixed_symbol if challenge['market'] == 'polymarket' else fixed_symbol.upper()
            if compare_requested != compare_fixed:
                raise ChallengeError('Trade symbol does not match challenge symbol')
    else:
        symbol = requested_symbol
    if not symbol:
        raise ChallengeError('symbol is required for challenge trade')
    return symbol if challenge['market'] == 'polymarket' else symbol.upper()


def _serialize_challenge_portfolio(
    challenge: dict[str, Any],
    participant: dict[str, Any],
    trades: list[dict[str, Any]],
    mark_prices: Optional[dict[tuple[str, str], float]] = None,
    mark_timestamp: Optional[str] = None,
) -> dict[str, Any]:
    scored = score_agent_trades(
        challenge,
        participant,
        trades,
        mark_prices=mark_prices,
        mark_timestamp=mark_timestamp,
    )
    metrics = scored.get('metrics') or {}
    ending_value = scored.get('ending_value')
    return {
        'challenge': _serialize_challenge(challenge),
        'participant': participant,
        'portfolio': {
            'starting_cash': scored.get('starting_cash'),
            'cash': metrics.get('cash'),
            'ending_value': ending_value,
            'return_pct': scored.get('return_pct'),
            'max_drawdown': scored.get('max_drawdown'),
            'risk_adjusted_score': scored.get('risk_adjusted_score'),
            'final_score': scored.get('final_score'),
            'trade_count': scored.get('trade_count'),
            'disqualified_reason': scored.get('disqualified_reason'),
            'marked_to_market': metrics.get('marked_to_market') or False,
            'mark_timestamp': metrics.get('mark_timestamp'),
            'live_marks': metrics.get('live_marks') or [],
            'positions': metrics.get('positions') or [],
            'equity_curve': metrics.get('equity_curve') or [],
        },
        'trades': trades,
    }


def get_agent_challenge_portfolio(challenge_key: str, agent_id: int) -> dict[str, Any]:
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        refresh_challenge_statuses(cursor)
        conn.commit()
        challenge = _load_challenge(cursor, challenge_key=challenge_key)
        cursor.execute(
            """
            SELECT cp.*, a.name AS agent_name, a.identity_status AS agent_identity_status
            FROM challenge_participants cp
            JOIN agents a ON a.id = cp.agent_id
            WHERE cp.challenge_id = ? AND cp.agent_id = ?
            """,
            (challenge['id'], agent_id),
        )
        row = cursor.fetchone()
        if not row:
            raise ChallengeError('Agent must join challenge before viewing challenge portfolio')
        participant = dict(row)
        participant['agent_identity_status'] = agent_identity_status(row)
        participant['agent_is_verified'] = agent_is_verified(row)
        cursor.execute(
            """
            SELECT *
            FROM challenge_trades
            WHERE challenge_id = ? AND agent_id = ?
            ORDER BY executed_at, id
            """,
            (challenge['id'], agent_id),
        )
        trades = [dict(trade) for trade in cursor.fetchall()]
        mark_prices: dict[tuple[str, str], float] = {}
        mark_timestamp = _live_mark_timestamp(challenge)
        if mark_timestamp:
            baseline = score_agent_trades(challenge, participant, trades)
            mark_prices = _fetch_live_mark_prices([baseline], mark_timestamp)
        return _serialize_challenge_portfolio(
            challenge,
            participant,
            trades,
            mark_prices=mark_prices,
            mark_timestamp=mark_timestamp,
        )
    finally:
        conn.close()


def create_challenge_trade(challenge_key: str, agent_id: int, data: Any) -> dict[str, Any]:
    payload = _model_dump(data)
    side = str(payload.get('side') or payload.get('action') or '').strip().lower()
    if side not in {'buy', 'sell', 'short', 'cover'}:
        raise ChallengeError('Unsupported challenge trade side')
    try:
        price = float(payload.get('price'))
        quantity = float(payload.get('quantity'))
    except Exception as exc:
        raise ChallengeError('price and quantity are required') from exc
    if price <= 0 or quantity <= 0:
        raise ChallengeError('price and quantity must be positive')

    executed_at = _iso(_parse_dt(payload.get('executed_at')) or datetime.now(timezone.utc))

    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        begin_write_transaction(cursor)
        refresh_challenge_statuses(cursor)
        challenge = _load_challenge(cursor, challenge_key=challenge_key)
        if challenge['status'] != 'active':
            raise ChallengeError('Challenge is not active')
        executed_dt = _parse_dt(executed_at)
        if executed_dt < _parse_dt(challenge['start_at']) or executed_dt > _parse_dt(challenge['end_at']):
            raise ChallengeError('Challenge trade must be inside challenge time window')

        cursor.execute(
            """
            SELECT cp.*, a.name AS agent_name, a.identity_status AS agent_identity_status
            FROM challenge_participants cp
            JOIN agents a ON a.id = cp.agent_id
            WHERE cp.challenge_id = ? AND cp.agent_id = ?
            """,
            (challenge['id'], agent_id),
        )
        row = cursor.fetchone()
        if not row:
            raise ChallengeError('Agent must join challenge before trading')
        participant = dict(row)
        participant['agent_identity_status'] = agent_identity_status(row)
        participant['agent_is_verified'] = agent_is_verified(row)
        if participant.get('status') not in {'joined', 'active'}:
            raise ChallengeError('Challenge participant is not tradeable')

        symbol = _normalize_challenge_trade_symbol(challenge, payload.get('symbol'))
        requested_price = price
        server_price = _fetch_authoritative_challenge_trade_price(challenge['market'], symbol, executed_at)
        if server_price is not None:
            price = server_price
        cursor.execute(
            """
            SELECT *
            FROM challenge_trades
            WHERE challenge_id = ? AND agent_id = ?
            ORDER BY executed_at, id
            """,
            (challenge['id'], agent_id),
        )
        existing_trades = [dict(trade) for trade in cursor.fetchall()]
        proposed_trade = {
            'id': (max([int(trade.get('id') or 0) for trade in existing_trades], default=0) + 1),
            'market': challenge['market'],
            'symbol': symbol,
            'side': side,
            'price': price,
            'quantity': quantity,
            'executed_at': executed_at,
        }
        simulated = score_agent_trades(challenge, participant, [*existing_trades, proposed_trade])
        disqualified_reason = simulated.get('disqualified_reason')
        allowed_rule_disqualifications = {'max_position_pct_exceeded', 'max_drawdown_pct_exceeded'}
        if disqualified_reason and disqualified_reason not in allowed_rule_disqualifications:
            raise ChallengeError(f"Challenge trade rejected: {simulated['disqualified_reason']}")

        cursor.execute(
            """
            INSERT INTO challenge_trades
            (challenge_id, agent_id, source_signal_id, market, symbol, side, price, quantity, executed_at, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                challenge['id'],
                agent_id,
                None,
                challenge['market'],
                symbol,
                side,
                price,
                quantity,
                executed_at,
                utc_now_iso_z(),
            ),
        )
        trade_id = cursor.lastrowid
        content = (payload.get('content') or '').strip() or None
        if content:
            _create_submission_with_cursor(
                cursor,
                challenge,
                agent_id,
                'trade',
                content,
                None,
                None,
            )
        record_event(
            'challenge_trade_submitted',
            actor_agent_id=agent_id,
            object_type='challenge_trade',
            object_id=trade_id,
            market=challenge['market'],
            experiment_key=challenge.get('experiment_key'),
            variant_key=participant.get('variant_key'),
            metadata={
                'challenge_key': challenge['challenge_key'],
                'challenge_id': challenge['id'],
                'symbol': symbol,
                'side': side,
                'price': price,
                'requested_price': requested_price,
                'quantity': quantity,
            },
            cursor=cursor,
        )
        conn.commit()

        cursor.execute(
            """
            SELECT *
            FROM challenge_trades
            WHERE challenge_id = ? AND agent_id = ?
            ORDER BY executed_at, id
            """,
            (challenge['id'], agent_id),
        )
        trades = [dict(trade) for trade in cursor.fetchall()]
        portfolio = _serialize_challenge_portfolio(challenge, participant, trades)
        return {'trade': trades[-1], **portfolio}
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _fetch_participants_and_trades(cursor: Any, challenge_id: int) -> tuple[list[dict[str, Any]], dict[int, list[dict[str, Any]]]]:
    cursor.execute(
        """
        SELECT cp.*, a.name AS agent_name, a.identity_status AS agent_identity_status
        FROM challenge_participants cp
        JOIN agents a ON a.id = cp.agent_id
        WHERE cp.challenge_id = ?
        ORDER BY cp.joined_at, cp.id
        """,
        (challenge_id,),
    )
    participants = []
    for row in cursor.fetchall():
        participant = dict(row)
        participant['agent_identity_status'] = agent_identity_status(row)
        participant['agent_is_verified'] = agent_is_verified(row)
        participants.append(participant)

    cursor.execute(
        """
        SELECT *
        FROM challenge_trades
        WHERE challenge_id = ?
        ORDER BY agent_id, executed_at, id
        """,
        (challenge_id,),
    )
    trades_by_agent: dict[int, list[dict[str, Any]]] = {}
    for row in cursor.fetchall():
        trade = dict(row)
        trades_by_agent.setdefault(trade['agent_id'], []).append(trade)

    return participants, trades_by_agent


def get_challenge_leaderboard(challenge_key: str) -> dict[str, Any]:
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        refresh_challenge_statuses(cursor)
        conn.commit()
        challenge = _load_challenge(cursor, challenge_key=challenge_key)
        cursor.execute(
            """
            SELECT cr.*, a.name AS agent_name, a.identity_status AS agent_identity_status, cp.disqualified_reason, cp.trade_count
            FROM challenge_results cr
            JOIN agents a ON a.id = cr.agent_id
            LEFT JOIN challenge_participants cp ON cp.challenge_id = cr.challenge_id AND cp.agent_id = cr.agent_id
            WHERE cr.challenge_id = ?
            ORDER BY COALESCE(cr.rank, 999999), cr.final_score DESC, cr.id
            """,
            (challenge['id'],),
        )
        result_rows = []
        for row in cursor.fetchall():
            result_row = dict(row)
            result_row['agent_identity_status'] = agent_identity_status(row)
            result_row['agent_is_verified'] = agent_is_verified(row)
            result_rows.append(result_row)
        if result_rows and challenge.get('status') == 'settled':
            return {'challenge': _serialize_challenge(challenge), 'leaderboard': result_rows, 'provisional': False}

        participants, trades_by_agent = _fetch_participants_and_trades(cursor, challenge['id'])
        scored = _score_challenge_results_with_live_marks(challenge, participants, trades_by_agent)
        names = {item['agent_id']: item.get('agent_name') for item in participants}
        identities = {item['agent_id']: item.get('agent_identity_status') for item in participants}
        for item in scored:
            item['agent_name'] = names.get(item['agent_id'])
            item['agent_identity_status'] = agent_identity_status({'identity_status': identities.get(item['agent_id'])})
            item['agent_is_verified'] = item['agent_identity_status'] == 'verified'
            item['metrics_json'] = _json_dumps(item.get('metrics'))
        scored.sort(key=lambda item: (item.get('rank') is None, item.get('rank') or 999999))
        return {'challenge': _serialize_challenge(challenge), 'leaderboard': scored, 'provisional': True}
    finally:
        conn.close()


def _reward_points_for_rank(rules: dict[str, Any], rank: Optional[int]) -> int:
    if not rank or rules.get('grant_rewards') is False:
        return 0
    reward_points = rules.get('reward_points', DEFAULT_CHALLENGE_REWARDS)
    if isinstance(reward_points, list):
        return int(reward_points[rank - 1]) if rank - 1 < len(reward_points) else 0
    if isinstance(reward_points, dict):
        return int(reward_points.get(str(rank), reward_points.get(rank, 0)) or 0)
    return 0


def settle_challenge(challenge_key: str, *, force: bool = False) -> dict[str, Any]:
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        begin_write_transaction(cursor)
        refresh_challenge_statuses(cursor)
        challenge = _load_challenge(cursor, challenge_key=challenge_key)
        if challenge['status'] == 'settled' and not force:
            conn.commit()
            return get_challenge_leaderboard(challenge_key)
        if challenge['status'] == 'canceled':
            raise ChallengeError('Canceled challenge cannot be settled')

        participants, trades_by_agent = _fetch_participants_and_trades(cursor, challenge['id'])
        scored = score_challenge_results(challenge, participants, trades_by_agent)
        participant_by_agent = {item['agent_id']: item for item in participants}
        rules = _json_loads(challenge.get('rules_json'), {}) or {}
        now = utc_now_iso_z()

        if force:
            cursor.execute("DELETE FROM challenge_results WHERE challenge_id = ?", (challenge['id'],))

        for result in scored:
            participant = participant_by_agent[result['agent_id']]
            metrics_json = _json_dumps(result['metrics'])
            status = 'disqualified' if result.get('disqualified_reason') else 'settled'
            cursor.execute(
                """
                UPDATE challenge_participants
                SET status = ?, ending_value = ?, return_pct = ?, max_drawdown = ?,
                    trade_count = ?, rank = ?, disqualified_reason = ?
                WHERE challenge_id = ? AND agent_id = ?
                """,
                (
                    status,
                    result['ending_value'],
                    result['return_pct'],
                    result['max_drawdown'],
                    result['trade_count'],
                    result.get('rank'),
                    result.get('disqualified_reason'),
                    challenge['id'],
                    result['agent_id'],
                ),
            )
            cursor.execute(
                """
                INSERT INTO challenge_results
                (challenge_id, agent_id, return_pct, max_drawdown, risk_adjusted_score,
                 quality_score, final_score, rank, metrics_json, settled_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    challenge['id'],
                    result['agent_id'],
                    result['return_pct'],
                    result['max_drawdown'],
                    result['risk_adjusted_score'],
                    result['quality_score'],
                    result.get('final_score'),
                    result.get('rank'),
                    metrics_json,
                    now,
                ),
            )

            if result.get('disqualified_reason'):
                record_event(
                    'challenge_disqualified',
                    actor_agent_id=result['agent_id'],
                    object_type='challenge_participant',
                    object_id=participant['id'],
                    market=challenge['market'],
                    experiment_key=challenge.get('experiment_key'),
                    variant_key=participant.get('variant_key'),
                    metadata={
                        'challenge_key': challenge['challenge_key'],
                        'reason': result['disqualified_reason'],
                    },
                    cursor=cursor,
                )
                continue

            reward_points = _reward_points_for_rank(rules, result.get('rank'))
            if reward_points > 0:
                grant_agent_reward(
                    result['agent_id'],
                    reward_points,
                    f"challenge_rank_{result['rank']}",
                    source_type='challenge',
                    source_id=challenge['id'],
                    experiment_key=challenge.get('experiment_key'),
                    variant_key=participant.get('variant_key'),
                    metadata={'challenge_key': challenge['challenge_key'], 'rank': result.get('rank')},
                    cursor=cursor,
                )
                record_event(
                    'challenge_reward_granted',
                    actor_agent_id=result['agent_id'],
                    object_type='challenge',
                    object_id=challenge['id'],
                    market=challenge['market'],
                    experiment_key=challenge.get('experiment_key'),
                    variant_key=participant.get('variant_key'),
                    metadata={
                        'challenge_key': challenge['challenge_key'],
                        'rank': result.get('rank'),
                        'points': reward_points,
                    },
                    cursor=cursor,
                )

        cursor.execute(
            """
            UPDATE challenges
            SET status = 'settled', settled_at = ?, updated_at = ?
            WHERE id = ?
            """,
            (now, now, challenge['id']),
        )
        record_event(
            'challenge_settled',
            object_type='challenge',
            object_id=challenge['id'],
            market=challenge['market'],
            experiment_key=challenge.get('experiment_key'),
            metadata={'challenge_key': challenge['challenge_key'], 'participant_count': len(participants)},
            cursor=cursor,
        )
        conn.commit()
        return get_challenge_leaderboard(challenge_key)
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def settle_due_challenges(limit: int = 20) -> list[dict[str, Any]]:
    now = utc_now_iso_z()
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        refresh_challenge_statuses(cursor)
        conn.commit()
        cursor.execute(
            """
            SELECT challenge_key
            FROM challenges
            WHERE status = 'active' AND end_at <= ?
            ORDER BY end_at ASC
            LIMIT ?
            """,
            (now, max(1, min(limit, 100))),
        )
        keys = [row['challenge_key'] for row in cursor.fetchall()]
    finally:
        conn.close()

    settled = []
    for key in keys:
        settled.append(settle_challenge(key))
    return settled


def cancel_challenge(challenge_key: str, agent_id: int) -> dict[str, Any]:
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        begin_write_transaction(cursor)
        challenge = _load_challenge(cursor, challenge_key=challenge_key)
        if challenge.get('created_by_agent_id') and challenge['created_by_agent_id'] != agent_id:
            raise ChallengeError('Only the creator can cancel this challenge')
        if challenge['status'] == 'settled':
            raise ChallengeError('Settled challenge cannot be canceled')
        now = utc_now_iso_z()
        cursor.execute(
            "UPDATE challenges SET status = 'canceled', updated_at = ? WHERE id = ?",
            (now, challenge['id']),
        )
        conn.commit()
        return get_challenge(challenge_key)
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def get_agent_challenges(agent_id: int) -> dict[str, Any]:
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        refresh_challenge_statuses(cursor)
        conn.commit()
        cursor.execute(
            """
            SELECT c.*, cp.status AS participant_status, cp.variant_key, cp.joined_at,
                   cp.return_pct, cp.max_drawdown, cp.trade_count, cp.rank,
                   cp.disqualified_reason,
                   (SELECT COUNT(*) FROM challenge_participants count_cp WHERE count_cp.challenge_id = c.id) AS participant_count
            FROM challenge_participants cp
            JOIN challenges c ON c.id = cp.challenge_id
            WHERE cp.agent_id = ?
            ORDER BY c.status = 'active' DESC, c.start_at DESC, c.id DESC
            """,
            (agent_id,),
        )
        return {'challenges': [_serialize_challenge(row, row['participant_count']) for row in cursor.fetchall()]}
    finally:
        conn.close()


def get_challenge_submissions(challenge_key: str, limit: int = 100, offset: int = 0) -> dict[str, Any]:
    limit = max(1, min(limit, 500))
    offset = max(0, offset)
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        challenge = _load_challenge(cursor, challenge_key=challenge_key)
        cursor.execute(
            "SELECT COUNT(*) AS total FROM challenge_submissions WHERE challenge_id = ?",
            (challenge['id'],),
        )
        total = cursor.fetchone()['total']
        cursor.execute(
            """
            SELECT cs.*, a.name AS agent_name, a.identity_status AS agent_identity_status
            FROM challenge_submissions cs
            JOIN agents a ON a.id = cs.agent_id
            WHERE cs.challenge_id = ?
            ORDER BY cs.created_at DESC, cs.id DESC
            LIMIT ? OFFSET ?
            """,
            (challenge['id'], limit, offset),
        )
        submissions = []
        for row in cursor.fetchall():
            submission = dict(row)
            submission['agent_identity_status'] = agent_identity_status(row)
            submission['agent_is_verified'] = agent_is_verified(row)
            submissions.append(submission)
        return {
            'challenge': _serialize_challenge(challenge),
            'submissions': submissions,
            'total': total,
        }
    finally:
        conn.close()

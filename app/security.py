"""Argon2id password verification, opaque sessions, CSRF, lockout, and audit."""

from __future__ import annotations

import hashlib
import secrets
import time

from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError, VerifyMismatchError

from app.auth_db import connect

SESSION_COOKIE = "prl_session"
IDLE_SECONDS = 30 * 60
ABSOLUTE_SECONDS = 12 * 60 * 60
_hasher = PasswordHasher(time_cost=2, memory_cost=19456, parallelism=1, hash_len=32, salt_len=16)


def hash_password(password: str) -> str:
    return _hasher.hash(password)


def verify_password(password_hash: str, password: str) -> bool:
    try:
        return _hasher.verify(password_hash, password)
    except (VerifyMismatchError, VerificationError, InvalidHashError):
        return False


def audit(actor: str, role: str, action: str, target: str, details: dict | None = None) -> None:
    import json

    with connect() as db:
        db.execute(
            "INSERT INTO audit_log(actor,role,action,target,details_json) VALUES(?,?,?,?,?)",
            (actor, role, action, target, json.dumps(details or {}, separators=(",", ":"))),
        )


def deny(user: dict | None, path: str, method: str, reason: str) -> None:
    audit(
        user["username"] if user else "anonymous",
        user["role"] if user else "anonymous",
        "denied",
        f"{method} {path}",
        {"reason": reason},
    )


def _identity(kind: str, value: str) -> str:
    return hashlib.sha256(f"{kind}:{value}".encode("utf-8")).hexdigest()


def login_locked(username: str, ip: str) -> bool:
    now = time.time()
    ids = (_identity("user", username.casefold()), _identity("ip", ip))
    with connect() as db:
        states = db.execute(
            f"SELECT locked_until FROM login_state WHERE identity_hash IN ({','.join('?' for _ in ids)})", ids
        ).fetchall()
    return any(float(row[0]) > now for row in states)


def failed_login(username: str, ip: str) -> None:
    now = time.time()
    identities = ((_identity("user", username.casefold()), 5), (_identity("ip", ip), 10))
    with connect() as db:
        for identity, limit in identities:
            row = db.execute(
                "SELECT failures,window_start FROM login_state WHERE identity_hash=?", (identity,)
            ).fetchone()
            if not row or now - float(row[1]) > 15 * 60:
                failures, start = 1, now
            else:
                failures, start = int(row[0]) + 1, float(row[1])
            locked_until = now + 60 if failures >= limit else 0
            db.execute(
                "INSERT INTO login_state(identity_hash,failures,window_start,locked_until) VALUES(?,?,?,?) ON CONFLICT(identity_hash) DO UPDATE SET failures=excluded.failures,window_start=excluded.window_start,locked_until=excluded.locked_until",
                (identity, failures, start, locked_until),
            )
    time.sleep(min(0.1 * failures, 0.5))


def clear_login_failures(username: str, ip: str) -> None:
    with connect() as db:
        db.executemany(
            "DELETE FROM login_state WHERE identity_hash=?",
            ((_identity("user", username.casefold()),), (_identity("ip", ip),)),
        )


def create_session(username: str) -> tuple[str, str]:
    token = secrets.token_urlsafe(32)
    csrf = secrets.token_urlsafe(32)
    now = time.time()
    with connect() as db:
        db.execute(
            "INSERT INTO auth_sessions(token_hash,username,csrf_token,created_at,last_seen,expires_at) VALUES(?,?,?,?,?,?)",
            (hashlib.sha256(token.encode()).hexdigest(), username, csrf, now, now, now + ABSOLUTE_SECONDS),
        )
    return token, csrf


def resolve_session(token: str | None) -> tuple[dict | None, str | None]:
    if not token:
        return None, None
    token_hash = hashlib.sha256(token.encode()).hexdigest()
    now = time.time()
    with connect() as db:
        row = db.execute(
            "SELECT s.username,s.csrf_token,s.last_seen,s.expires_at,u.role,u.display_name,u.enabled,u.deleted_at FROM auth_sessions s JOIN auth_users u ON u.username=s.username WHERE s.token_hash=?",
            (token_hash,),
        ).fetchone()
        if not row:
            return None, None
        if not row["enabled"] or row["deleted_at"] or now - row["last_seen"] > IDLE_SECONDS or row["expires_at"] < now:
            db.execute("DELETE FROM auth_sessions WHERE token_hash=?", (token_hash,))
            return None, None
        db.execute("UPDATE auth_sessions SET last_seen=? WHERE token_hash=?", (now, token_hash))
        return {"username": row["username"], "role": row["role"], "display_name": row["display_name"]}, row[
            "csrf_token"
        ]


def revoke_session(token: str | None) -> None:
    if not token:
        return
    with connect() as db:
        db.execute("DELETE FROM auth_sessions WHERE token_hash=?", (hashlib.sha256(token.encode()).hexdigest(),))

"""API-key authentication and tenant-scoped repository authorization.

The authentication boundary is independent of HTTP so it can later be backed by
database-managed API keys or an external identity provider.
"""

from __future__ import annotations

import hashlib
import hmac
from dataclasses import dataclass
from threading import Lock
from typing import Protocol


class AuthenticationError(PermissionError):
    """Raised when a request has no valid credential."""


class AuthorizationError(PermissionError):
    """Raised when a tenant attempts to access another tenant's repository."""


@dataclass(frozen=True)
class TenantPrincipal:
    """Authenticated caller identity used for authorization and audit events."""

    organization_id: str
    key_id: str


def hash_api_key(api_key: str) -> str:
    """Return the SHA-256 representation stored in deployment configuration."""
    return hashlib.sha256(api_key.encode("utf-8")).hexdigest()


class APIKeyAuthenticator:
    """Authenticate a raw API key against configured SHA-256 digests."""

    def __init__(self, key_hashes: dict[str, TenantPrincipal]) -> None:
        self._key_hashes = dict(key_hashes)

    def authenticate(self, api_key: str | None) -> TenantPrincipal:
        if not api_key:
            raise AuthenticationError("Missing X-API-Key header.")
        candidate = hash_api_key(api_key)
        for stored_hash, principal in self._key_hashes.items():
            if hmac.compare_digest(candidate, stored_hash):
                return principal
        raise AuthenticationError("Invalid API key.")


class InMemoryRepositoryOwnership:
    """Repository-to-organization policy store for local development and tests.

    This is intentionally replaceable: production deployments need a durable
    Postgres implementation before running multiple API replicas.
    """

    def __init__(self) -> None:
        self._owners: dict[str, str] = {}
        self._lock = Lock()

    def claim_for_ingestion(self, repo_id: str, organization_id: str) -> bool:
        """Claim a new repository or confirm that the caller already owns it."""
        with self._lock:
            owner = self._owners.get(repo_id)
            if owner is None:
                self._owners[repo_id] = organization_id
                return True
            if owner != organization_id:
                raise AuthorizationError("Repository belongs to another organization.")
            return False

    def require_owner(self, repo_id: str, organization_id: str) -> None:
        with self._lock:
            owner = self._owners.get(repo_id)
        if owner is None:
            raise AuthorizationError("Repository is not registered for this organization.")
        if owner != organization_id:
            raise AuthorizationError("Repository belongs to another organization.")

    def release_if_new(self, repo_id: str, organization_id: str) -> None:
        with self._lock:
            if self._owners.get(repo_id) == organization_id:
                self._owners.pop(repo_id, None)


class RepositoryOwnership(Protocol):
    def claim_for_ingestion(self, repo_id: str, organization_id: str) -> bool: ...
    def require_owner(self, repo_id: str, organization_id: str) -> None: ...
    def release_if_new(self, repo_id: str, organization_id: str) -> None: ...


class PostgresRepositoryOwnership:
    """Durable tenant control plane for Postgres deployments.

    The schema is additive and initialized idempotently. API key hashes are
    bootstrapped from deployment configuration; the application never stores or
    logs raw API keys.
    """

    def __init__(self, database_url: str, api_keys: tuple[tuple[str, str, str], ...]) -> None:
        self._database_url = database_url
        self._api_keys = api_keys
        self._initialized = False
        self._lock = Lock()

    def _connection(self):
        try:
            import psycopg
            return psycopg.connect(self._database_url, autocommit=True)
        except ImportError as error:
            raise RuntimeError("psycopg is required for durable tenancy. Install the postgres extra.") from error

    def _initialize(self) -> None:
        with self._lock:
            if self._initialized:
                return
            with self._connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("CREATE TABLE IF NOT EXISTS tenant_organizations (id TEXT PRIMARY KEY, created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP);")
                    cur.execute("CREATE TABLE IF NOT EXISTS tenant_api_keys (key_id TEXT PRIMARY KEY, key_hash TEXT UNIQUE NOT NULL, organization_id TEXT NOT NULL REFERENCES tenant_organizations(id), created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP, revoked_at TIMESTAMPTZ);")
                    cur.execute("CREATE TABLE IF NOT EXISTS tenant_repositories (repo_id TEXT PRIMARY KEY, organization_id TEXT NOT NULL REFERENCES tenant_organizations(id), created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP);")
                    cur.execute("CREATE INDEX IF NOT EXISTS idx_tenant_repositories_organization ON tenant_repositories(organization_id);")
                    for key_id, key_hash, organization_id in self._api_keys:
                        cur.execute("INSERT INTO tenant_organizations (id) VALUES (%s) ON CONFLICT (id) DO NOTHING;", (organization_id,))
                        cur.execute("INSERT INTO tenant_api_keys (key_id, key_hash, organization_id) VALUES (%s, %s, %s) ON CONFLICT (key_id) DO UPDATE SET key_hash = EXCLUDED.key_hash, organization_id = EXCLUDED.organization_id, revoked_at = NULL;", (key_id, key_hash, organization_id))
            self._initialized = True

    def claim_for_ingestion(self, repo_id: str, organization_id: str) -> bool:
        self._initialize()
        with self._connection() as conn:
            with conn.cursor() as cur:
                cur.execute("INSERT INTO tenant_organizations (id) VALUES (%s) ON CONFLICT (id) DO NOTHING;", (organization_id,))
                cur.execute("INSERT INTO tenant_repositories (repo_id, organization_id) VALUES (%s, %s) ON CONFLICT (repo_id) DO NOTHING RETURNING repo_id;", (repo_id, organization_id))
                created = cur.fetchone() is not None
                cur.execute("SELECT organization_id FROM tenant_repositories WHERE repo_id = %s;", (repo_id,))
                owner = cur.fetchone()[0]
        if owner != organization_id:
            raise AuthorizationError("Repository belongs to another organization.")
        return created

    def require_owner(self, repo_id: str, organization_id: str) -> None:
        self._initialize()
        with self._connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT organization_id FROM tenant_repositories WHERE repo_id = %s;", (repo_id,))
                row = cur.fetchone()
        if row is None:
            raise AuthorizationError("Repository is not registered for this organization.")
        if row[0] != organization_id:
            raise AuthorizationError("Repository belongs to another organization.")

    def release_if_new(self, repo_id: str, organization_id: str) -> None:
        self._initialize()
        with self._connection() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM tenant_repositories WHERE repo_id = %s AND organization_id = %s;", (repo_id, organization_id))

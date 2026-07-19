"""Transient-error retry behavior and auth fallback on the Google client."""
from unittest import mock

import httplib2
import pytest
from google.auth.exceptions import RefreshError
from googleapiclient.errors import HttpError

import resume_tailor.google_api as ga


def _http_error(status: int) -> HttpError:
    return HttpError(resp=httplib2.Response({"status": str(status)}), content=b"err")


def test_is_transient_matches_configured_codes():
    assert ga._is_transient(_http_error(429), (429, 500, 503))
    assert ga._is_transient(_http_error(503), (429, 503))
    assert not ga._is_transient(_http_error(500), (429, 503))   # 500 excluded for writes
    assert not ga._is_transient(_http_error(403), (429, 500, 503))
    assert not ga._is_transient(ValueError("x"), (429, 500, 503))


def test_batch_update_retries_429_then_succeeds():
    client = ga.GoogleAPIClient()
    docs = mock.MagicMock()
    client._docs = docs
    client._drive = mock.MagicMock()          # skip _ensure_auth
    execute = docs.documents.return_value.batchUpdate.return_value.execute
    execute.side_effect = [_http_error(429), {"replies": [{}]}]

    resp = client.batch_update("doc-1", [{"replaceAllText": {}}])

    assert resp == {"replies": [{}]}
    assert execute.call_count == 2


def test_batch_update_does_not_retry_500():
    client = ga.GoogleAPIClient()
    docs = mock.MagicMock()
    client._docs = docs
    client._drive = mock.MagicMock()
    execute = docs.documents.return_value.batchUpdate.return_value.execute
    execute.side_effect = _http_error(500)

    with pytest.raises(HttpError):
        client.batch_update("doc-1", [{"replaceAllText": {}}])
    assert execute.call_count == 1            # a 500 write may have committed — never replay


def test_authenticate_reauthorizes_when_refresh_token_is_dead(tmp_path, monkeypatch):
    """A revoked/expired refresh token must trigger a fresh interactive login,
    not crash with RefreshError. Regression: setup_google_auth.py was unusable
    in exactly the situation it exists for, forcing a manual token.json delete."""
    token_path = tmp_path / "token.json"
    token_path.write_text("{}")                 # exists; loading is mocked below
    secret_path = tmp_path / "client_secret.json"
    secret_path.write_text("{}")                # exists so the flow is reached
    monkeypatch.setattr(ga, "TOKEN_PATH", str(token_path))
    monkeypatch.setattr(ga, "CLIENT_SECRET_PATH", str(secret_path))

    # Saved token loads as expired-with-refresh-token, and refreshing it fails.
    stale = mock.MagicMock(name="stale_creds")
    stale.valid = False
    stale.expired = True
    stale.refresh_token = "dead-refresh-token"
    stale.refresh.side_effect = RefreshError("invalid_grant")
    monkeypatch.setattr(
        ga.Credentials,
        "from_authorized_user_file",
        mock.MagicMock(return_value=stale),
    )

    # The interactive flow yields brand-new credentials.
    fresh = mock.MagicMock(name="fresh_creds")
    fresh.to_json.return_value = '{"token": "fresh"}'
    flow = mock.MagicMock(name="flow")
    flow.run_local_server.return_value = fresh
    from_secrets = mock.MagicMock(return_value=flow)
    monkeypatch.setattr(
        ga.InstalledAppFlow, "from_client_secrets_file", from_secrets
    )

    # Don't build real API services (would hit the network).
    monkeypatch.setattr(ga, "build", mock.MagicMock())

    client = ga.GoogleAPIClient()
    client.authenticate()                       # must NOT raise RefreshError

    stale.refresh.assert_called_once()          # refresh was attempted...
    from_secrets.assert_called_once_with(str(secret_path), ga.OAUTH_SCOPES)
    flow.run_local_server.assert_called_once()  # ...then fell back to browser login
    assert client._creds is fresh               # new creds adopted
    assert token_path.read_text() == '{"token": "fresh"}'   # and persisted


# ---------------------------------------------------------------------------
# Non-interactive authentication (unattended runs). authenticate()'s fallback
# to flow.run_local_server is the documented recovery path for
# scripts/setup_google_auth.py, but on the 02:30 launchd run it blocks forever
# waiting for a browser. allow_interactive=False must surface the failure as
# the taxonomy's systemic exceptions instead of hanging.
# ---------------------------------------------------------------------------
class _StubCreds:
    valid = False
    expired = True
    refresh_token = "rt"

    def refresh(self, request):
        raise RefreshError("invalid_grant")

    def to_json(self):
        return "{}"


def test_authenticate_noninteractive_dead_token_raises(tmp_path, monkeypatch):
    token = tmp_path / "token.json"
    token.write_text("{}")
    monkeypatch.setattr(ga, "TOKEN_PATH", str(token))
    monkeypatch.setattr(
        ga.Credentials, "from_authorized_user_file",
        staticmethod(lambda path, scopes: _StubCreds()),
    )

    def no_browser(*a, **k):
        raise AssertionError("interactive OAuth flow must not start when allow_interactive=False")

    monkeypatch.setattr(
        ga.InstalledAppFlow, "from_client_secrets_file", staticmethod(no_browser))

    with pytest.raises(RefreshError):
        ga.GoogleAPIClient().authenticate(allow_interactive=False)


def test_authenticate_noninteractive_missing_token_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(ga, "TOKEN_PATH", str(tmp_path / "absent.json"))

    def no_browser(*a, **k):
        raise AssertionError("interactive OAuth flow must not start when allow_interactive=False")

    monkeypatch.setattr(
        ga.InstalledAppFlow, "from_client_secrets_file", staticmethod(no_browser))

    with pytest.raises(FileNotFoundError):
        ga.GoogleAPIClient().authenticate(allow_interactive=False)

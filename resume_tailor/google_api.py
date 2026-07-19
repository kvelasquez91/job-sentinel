"""
Google Docs / Drive API client wrapper.

Authentication: OAuth 2.0 Desktop App flow.
First run: opens browser for authorization, saves token.json.
Subsequent runs: refreshes token silently from token.json.

Setup: run scripts/setup_google_auth.py once to create token.json.
"""
import io
import logging
import os
import threading
from typing import Optional

from google.auth.exceptions import RefreshError
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaIoBaseDownload
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from .config import (
    CLIENT_SECRET_PATH,
    TOKEN_PATH,
    OAUTH_SCOPES,
    TAILORED_FOLDER_NAME,
    DOCX_OUTPUT_DIR,
)

logger = logging.getLogger(__name__)

DOCS_URL_TEMPLATE = "https://docs.google.com/document/d/{doc_id}/edit"
GDRIVE_FOLDER_MIME = "application/vnd.google-apps.folder"
DOCX_MIME = (
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
)

# Transient-error retry policy.
# Reads (export) retry 429/500/503. Writes (batchUpdate) retry ONLY 429/503:
# a 500 may mean the server already committed the mutation, and replaying a
# positional deleteContentRange+insertText against a changed doc would corrupt it.
_READ_RETRY_CODES = (429, 500, 503)
_WRITE_RETRY_CODES = (429, 503)


def _is_transient(exc: BaseException, codes: tuple) -> bool:
    if not isinstance(exc, HttpError):
        return False
    status = getattr(getattr(exc, "resp", None), "status", None)
    try:
        return int(status) in codes
    except (TypeError, ValueError):
        return False


def _transient_retry(codes: tuple):
    return retry(
        retry=retry_if_exception(lambda e: _is_transient(e, codes)),
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        reraise=True,
    )


class GoogleAPIClient:
    """Thin wrapper around Google Docs and Drive v3 APIs."""

    def __init__(self):
        self._creds: Optional[Credentials] = None
        self._docs = None
        self._drive = None

    # ------------------------------------------------------------------
    # Authentication
    # ------------------------------------------------------------------

    def authenticate(self, allow_interactive: bool = True) -> None:
        """
        Load credentials from token.json, refreshing if expired.
        If no token exists, run the interactive OAuth flow (opens browser).

        allow_interactive=False is for unattended contexts (the launchd runs):
        instead of falling through to flow.run_local_server — which blocks
        forever waiting for a browser — a dead refresh token re-raises its
        RefreshError and a missing/unusable token raises FileNotFoundError,
        both of which callers already treat as systemic auth failures.
        """
        creds = None

        if os.path.exists(TOKEN_PATH):
            creds = Credentials.from_authorized_user_file(TOKEN_PATH, OAUTH_SCOPES)
            logger.debug("Loaded credentials from %s", TOKEN_PATH)

        if not creds or not creds.valid:
            # Try a silent refresh when we have an expired token with a refresh
            # token. A revoked/expired refresh token raises RefreshError (e.g.
            # invalid_grant) — discard the dead token and fall through to a
            # fresh interactive login rather than crashing, so re-authorization
            # (scripts/setup_google_auth.py) always has a way out.
            if creds and creds.expired and creds.refresh_token:
                logger.info("Refreshing expired Google credentials")
                try:
                    creds.refresh(Request())
                except RefreshError as exc:
                    if not allow_interactive:
                        raise
                    logger.warning(
                        "Stored refresh token is invalid (%s); "
                        "starting fresh browser authorization",
                        exc,
                    )
                    creds = None

            # No usable token (never had one, or the refresh above failed):
            # run the interactive OAuth flow (opens a browser).
            if not creds or not creds.valid:
                if not allow_interactive:
                    raise FileNotFoundError(
                        f"No usable Google token at {TOKEN_PATH} and interactive "
                        "auth is disabled — run scripts/setup_google_auth.py."
                    )
                if not os.path.exists(CLIENT_SECRET_PATH):
                    raise FileNotFoundError(
                        f"Google client secret not found at {CLIENT_SECRET_PATH}.\n"
                        "Download it from Google Cloud Console → APIs & Services → "
                        "Credentials, then place it at that path."
                    )
                logger.info(
                    "Starting OAuth flow — browser will open for authorization"
                )
                flow = InstalledAppFlow.from_client_secrets_file(
                    CLIENT_SECRET_PATH, OAUTH_SCOPES
                )
                creds = flow.run_local_server(port=0)

            # Persist refreshed/new token atomically: concurrent tailor
            # pipelines can refresh at the same moment, and an interleaved
            # plain write would corrupt token.json for every later run.
            os.makedirs(os.path.dirname(TOKEN_PATH), exist_ok=True)
            tmp_path = f"{TOKEN_PATH}.tmp.{os.getpid()}.{threading.get_ident()}"
            with open(tmp_path, "w") as fh:
                fh.write(creds.to_json())
            os.replace(tmp_path, TOKEN_PATH)
            logger.info("Saved credentials to %s", TOKEN_PATH)

        self._creds = creds
        self._docs = build("docs", "v1", credentials=creds, cache_discovery=False)
        self._drive = build("drive", "v3", credentials=creds, cache_discovery=False)
        logger.debug("Google API services initialized")

    def _ensure_auth(self) -> None:
        if self._docs is None or self._drive is None:
            self.authenticate()

    # ------------------------------------------------------------------
    # Drive operations
    # ------------------------------------------------------------------

    def copy_document(self, source_doc_id: str, new_title: str) -> str:
        """
        Copy a Google Doc to a new document with the given title.
        Returns the new document ID.
        """
        self._ensure_auth()
        try:
            result = (
                self._drive.files()
                .copy(fileId=source_doc_id, body={"name": new_title})
                .execute()
            )
            new_id = result["id"]
            logger.info("Copied doc %s → %s ('%s')", source_doc_id, new_id, new_title)
            return new_id
        except HttpError as exc:
            logger.error("Failed to copy document %s: %s", source_doc_id, exc)
            raise

    def _find_folder(self, folder_name: str) -> Optional[str]:
        """Return the Drive folder ID matching folder_name, or None."""
        self._ensure_auth()
        query = (
            f"mimeType='{GDRIVE_FOLDER_MIME}' "
            f"and name='{folder_name}' "
            f"and trashed=false"
        )
        result = (
            self._drive.files()
            .list(q=query, fields="files(id, name)", pageSize=5)
            .execute()
        )
        files = result.get("files", [])
        if files:
            return files[0]["id"]
        return None

    def _create_folder(self, folder_name: str) -> str:
        """Create a Drive folder and return its ID."""
        self._ensure_auth()
        metadata = {
            "name": folder_name,
            "mimeType": GDRIVE_FOLDER_MIME,
        }
        folder = self._drive.files().create(body=metadata, fields="id").execute()
        folder_id = folder["id"]
        logger.info("Created Drive folder '%s' (%s)", folder_name, folder_id)
        return folder_id

    def move_to_folder(
        self,
        file_id: str,
        folder_name: str = TAILORED_FOLDER_NAME,
    ) -> str:
        """
        Move file_id into folder_name (creating the folder if needed).
        Returns the folder ID.
        """
        self._ensure_auth()
        folder_id = self._find_folder(folder_name)
        if not folder_id:
            folder_id = self._create_folder(folder_name)

        # Get current parents so we can remove them
        file_meta = (
            self._drive.files()
            .get(fileId=file_id, fields="parents")
            .execute()
        )
        previous_parents = ",".join(file_meta.get("parents", []))

        self._drive.files().update(
            fileId=file_id,
            addParents=folder_id,
            removeParents=previous_parents,
            fields="id, parents",
        ).execute()
        logger.info("Moved file %s into folder '%s' (%s)", file_id, folder_name, folder_id)
        return folder_id

    @_transient_retry(_READ_RETRY_CODES)
    def export_as_pdf(self, doc_id: str) -> bytes:
        """
        Export a Google Doc as PDF and return the raw bytes.
        Uses Drive API files().export() — does not write to disk.
        """
        self._ensure_auth()
        request = self._drive.files().export_media(
            fileId=doc_id, mimeType="application/pdf"
        )
        buffer = io.BytesIO()
        downloader = MediaIoBaseDownload(buffer, request)
        done = False
        while not done:
            _, done = downloader.next_chunk()
        pdf_bytes = buffer.getvalue()
        logger.info("Exported doc %s as PDF (%d bytes)", doc_id, len(pdf_bytes))
        return pdf_bytes

    def export_as_docx(self, doc_id: str, output_path: str) -> str:
        """
        Export a Google Doc as .docx and write it to output_path.
        Creates parent directories if needed. Returns output_path.
        """
        self._ensure_auth()
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        request = self._drive.files().export_media(
            fileId=doc_id, mimeType=DOCX_MIME
        )
        buffer = io.BytesIO()
        downloader = MediaIoBaseDownload(buffer, request)
        done = False
        while not done:
            _, done = downloader.next_chunk()

        with open(output_path, "wb") as fh:
            fh.write(buffer.getvalue())

        logger.info("Exported doc %s → %s", doc_id, output_path)
        return output_path

    def get_document_url(self, doc_id: str) -> str:
        """Return the Google Docs edit URL for the given document ID."""
        return DOCS_URL_TEMPLATE.format(doc_id=doc_id)

    # ------------------------------------------------------------------
    # Docs operations
    # ------------------------------------------------------------------

    def read_document(self, doc_id: str) -> dict:
        """
        Fetch the full document structure from the Google Docs API.
        Returns the raw document JSON (body, namedStyles, headers, footers, etc.).
        """
        self._ensure_auth()
        try:
            doc = self._docs.documents().get(documentId=doc_id).execute()
            logger.debug("Read document %s ('%s')", doc_id, doc.get("title"))
            return doc
        except HttpError as exc:
            logger.error("Failed to read document %s: %s", doc_id, exc)
            raise

    @_transient_retry(_WRITE_RETRY_CODES)
    def batch_update(self, doc_id: str, requests: list) -> dict:
        """
        Apply a list of batchUpdate request objects to the document.

        IMPORTANT — index shifting:
        When requests use positional indices (insertText, deleteContentRange),
        they MUST be sorted in reverse document order (highest startIndex first)
        so earlier edits don't shift the indices of later ones.
        replaceAllText requests are index-safe and can appear anywhere in the list.

        Returns the raw batchUpdate response.
        """
        self._ensure_auth()
        if not requests:
            logger.debug("batch_update called with empty request list — skipping")
            return {}
        try:
            response = (
                self._docs.documents()
                .batchUpdate(documentId=doc_id, body={"requests": requests})
                .execute()
            )
            logger.info(
                "batchUpdate on %s: %d request(s) applied", doc_id, len(requests)
            )
            return response
        except HttpError as exc:
            logger.error("batchUpdate failed on %s: %s", doc_id, exc)
            raise

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def extract_plain_text(self, doc: dict) -> str:
        """
        Walk a document's body content and return all text as a plain string.
        Useful for ATS checks and LLM prompts.
        """
        parts = []
        for element in doc.get("body", {}).get("content", []):
            paragraph = element.get("paragraph")
            if not paragraph:
                continue
            for el in paragraph.get("elements", []):
                text_run = el.get("textRun")
                if text_run:
                    parts.append(text_run.get("content", ""))
        return "".join(parts)

    def find_text_in_doc(self, doc: dict, search: str) -> list[dict]:
        """
        Return a list of (paragraph_element, startIndex, endIndex) dicts
        for every occurrence of `search` in the document body.
        Useful for surgical edits that need exact indices.
        """
        results = []
        search_lower = search.lower()
        for element in doc.get("body", {}).get("content", []):
            paragraph = element.get("paragraph")
            if not paragraph:
                continue
            for el in paragraph.get("elements", []):
                text_run = el.get("textRun")
                if not text_run:
                    continue
                content = text_run.get("content", "")
                if search_lower in content.lower():
                    results.append(
                        {
                            "element": el,
                            "startIndex": el.get("startIndex"),
                            "endIndex": el.get("endIndex"),
                            "content": content,
                        }
                    )
        return results

    def replace_all_text(self, doc_id: str, old_text: str, new_text: str) -> dict:
        """
        Convenience wrapper: replaceAllText across the entire document.
        Case-sensitive by default.
        """
        return self.batch_update(
            doc_id,
            [
                {
                    "replaceAllText": {
                        "containsText": {"text": old_text, "matchCase": True},
                        "replaceText": new_text,
                    }
                }
            ],
        )

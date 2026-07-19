"""
One-time Google OAuth 2.0 setup script.

Run this once before using the resume tailor:
  python scripts/setup_google_auth.py

Steps:
  1. Load client_secret.json from resume_tailor/config/
  2. Open browser for OAuth authorization
  3. Save token.json to resume_tailor/config/
  4. Verify access: read master resume title, list "Tailored Resumes" folder
  5. Create "Tailored Resumes" folder if it doesn't exist

Prerequisites:
  - Download OAuth 2.0 credentials (Desktop App type) from:
      Google Cloud Console → APIs & Services → Credentials → Create Credentials
  - Save the downloaded file as:
      resume_tailor/config/client_secret.json
  - Enable these APIs in your Google Cloud project:
      Google Docs API, Google Drive API
"""
import os
import sys

# Allow running from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from resume_tailor.config import (
    CLIENT_SECRET_PATH,
    TOKEN_PATH,
    MASTER_DOC_ID,
    TAILORED_FOLDER_NAME,
)
from resume_tailor.google_api import GoogleAPIClient


def main():
    print("=" * 60)
    print("Job Sentinel — Google API Setup")
    print("=" * 60)

    if not os.path.exists(CLIENT_SECRET_PATH):
        print(f"\nERROR: client_secret.json not found at:\n  {CLIENT_SECRET_PATH}")
        print("\nTo fix this:")
        print("  1. Go to Google Cloud Console → APIs & Services → Credentials")
        print("  2. Create an OAuth 2.0 Client ID (type: Desktop App)")
        print("  3. Download the JSON and save it to the path above")
        sys.exit(1)

    print(f"\nClient secret found: {CLIENT_SECRET_PATH}")
    print("Starting OAuth flow — your browser will open for authorization...")
    print("Grant access to: Google Docs (read/write) and Google Drive (file management)")

    client = GoogleAPIClient()
    client.authenticate()
    print(f"\nToken saved to: {TOKEN_PATH}")

    # Verify: read master resume
    print(f"\nVerifying access to master resume (doc ID: {MASTER_DOC_ID})...")
    try:
        doc = client.read_document(MASTER_DOC_ID)
        print(f"  ✓ Master resume found: '{doc.get('title')}'")
    except Exception as exc:
        print(f"  ✗ Could not read master resume: {exc}")
        print("  Check that you shared the document with the correct Google account.")
        sys.exit(1)

    # Verify/create "Tailored Resumes" folder
    print(f"\nChecking for '{TAILORED_FOLDER_NAME}' folder in Drive...")
    folder_id = client._find_folder(TAILORED_FOLDER_NAME)
    if folder_id:
        print(f"  ✓ Folder already exists (ID: {folder_id})")
    else:
        folder_id = client._create_folder(TAILORED_FOLDER_NAME)
        print(f"  ✓ Created folder (ID: {folder_id})")

    print("\n" + "=" * 60)
    print("Setup complete! You can now use the resume tailor.")
    print("=" * 60)


if __name__ == "__main__":
    main()

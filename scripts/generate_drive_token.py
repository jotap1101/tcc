"""Gera o token OAuth do Google Drive (formato 'authorized_user') para o Kaggle.

Executar na própria máquina (ou em um Colab), UMA única vez, autorizando com a
conta principal. O token é salvo em secrets/drive_token.json (pasta local do
repo, git-ignored pelo secrets/.gitignore). O conteúdo também deve ser colado
em um Secret do Kaggle (GDRIVE_TOKEN). Nunca commitar o token.

Uso:
    python scripts/generate_drive_token.py --client-secret caminho/do/client_secret.json
"""

from __future__ import annotations

import argparse
from pathlib import Path

DRIVE_SCOPE = "https://www.googleapis.com/auth/drive"
OUTPUT_PATH = Path(__file__).resolve().parent.parent / "secrets" / "drive_token.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--client-secret",
        type=Path,
        required=True,
        help="JSON de OAuth client (tipo Desktop app) baixado do Google Cloud Console",
    )
    args = parser.parse_args()

    from google_auth_oauthlib.flow import InstalledAppFlow

    flow = InstalledAppFlow.from_client_secrets_file(str(args.client_secret), scopes=[DRIVE_SCOPE])
    creds = flow.run_local_server()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(creds.to_json(), encoding="utf-8")
    print(f"Token salvo em: {OUTPUT_PATH}")
    print("O arquivo está git-ignored (secrets/.gitignore).")
    print("Agora cole o conteúdo desse arquivo em um Secret do Kaggle chamado GDRIVE_TOKEN")


if __name__ == "__main__":
    main()

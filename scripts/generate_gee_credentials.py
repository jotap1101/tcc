"""Gera as credenciais headless do Earth Engine (conta principal) para o Kaggle.

Executar uma única vez (no Colab é mais fácil: fluxo interativo com a conta
principal). O ee cria o arquivo de credenciais em ~/.config/earthengine/credentials;
este script o copia para secrets/earthengine_credentials.json (pasta local do
repo, git-ignored pelo secrets/.gitignore). O conteúdo também deve ser colado
em um Secret do Kaggle (GEE_CREDENTIALS). Nunca commitar o token.

Uso:
    python scripts/generate_gee_credentials.py
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

OUTPUT_PATH = Path(__file__).resolve().parent.parent / "secrets" / "earthengine_credentials.json"


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()

    import ee

    ee.Authenticate()

    credentials_path = Path.home() / ".config" / "earthengine" / "credentials"
    if credentials_path.is_file():
        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(credentials_path, OUTPUT_PATH)
        print(f"Credenciais copiadas para: {OUTPUT_PATH}")
        print("O arquivo está git-ignored (secrets/.gitignore).")
        print("Agora cole o conteúdo em um Secret do Kaggle chamado GEE_CREDENTIALS")
        print("(ou em MyDrive/tcc/secrets/earthengine_credentials.json)")
    else:
        print("Credenciais não encontradas; verifique o fluxo de autenticação.")


if __name__ == "__main__":
    main()

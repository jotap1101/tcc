"""Abstração de plataforma e de armazenamento no Google Drive (raiz tcc/).

Contrato único usado pelos notebooks: detect_platform(), mount_drive(),
ensure_storage_root() e resolve_storage_paths(). A diferença entre Colab
(montagem nativa) e Kaggle (Drive API + cache local) fica isolada aqui.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import tarfile
import urllib.request
from pathlib import Path
from typing import Any

from src.config import get_config

COLAB_DRIVE_ROOT = Path("/content/drive/MyDrive")
KAGGLE_DRIVE_CACHE = Path("/kaggle/working/drive")
REPO_URL = "https://github.com/jotap1101/tcc"


def detect_platform() -> str:
    """Identifica a plataforma de execução: 'colab', 'kaggle' ou 'local'."""
    try:
        import google.colab  # noqa: F401

        return "colab"
    except Exception:
        pass
    if os.getenv("KAGGLE_KERNEL_RUN_TYPE"):
        return "kaggle"
    return "local"


class DriveClient:
    """Cliente da Google Drive API — usado no Kaggle, onde não há mount nativo.

    A credencial é um JSON de OAuth (client_id, client_secret, refresh_token)
    autenticado com a conta principal, lido de GDRIVE_TOKEN ou GDRIVE_TOKEN_FILE.
    """

    def __init__(self) -> None:
        self._service: Any = None

    def _build_service(self) -> Any:
        """Constrói (lazy) o serviço da Drive API com a credencial OAuth do ambiente."""
        if self._service is None:
            from google.oauth2.credentials import Credentials
            from googleapiclient.discovery import build

            token = os.getenv("GDRIVE_TOKEN")
            token_file = os.getenv("GDRIVE_TOKEN_FILE")
            if token is None and token_file:
                token = Path(token_file).read_text(encoding="utf-8")
            if not token:
                raise RuntimeError(
                    "Acesso ao Drive no Kaggle exige credencial OAuth da conta principal "
                    "em GDRIVE_TOKEN (ou GDRIVE_TOKEN_FILE). Persista o token em "
                    "MyDrive/tcc/secrets/ na primeira autenticação."
                )
            if token_file or token.lstrip().startswith("{"):
                creds = Credentials.from_authorized_user_info(json.loads(token))
            else:
                creds = Credentials(token=token)
            self._service = build("drive", "v3", credentials=creds)
        return self._service

    def _folder_id(self, name: str, parent: str) -> str | None:
        """Retorna o ID de uma pasta filha, ou None se ela não existir."""
        res = (
            self._build_service()
            .files()
            .list(
                q=(
                    f"name='{name}' and '{parent}' in parents and mimeType="
                    "'application/vnd.google-apps.folder' and trashed=false"
                ),
                fields="files(id)",
            )
            .execute()
        )
        files = res.get("files", [])
        return files[0]["id"] if files else None

    def ensure_folder(self, remote_path: str) -> None:
        """Cria recursivamente uma pasta (separada por '/') no Drive, se ausente."""
        service = self._build_service()
        parent = "root"
        for part in remote_path.split("/"):
            if not part:
                continue
            folder_id = self._folder_id(part, parent)
            if folder_id is not None:
                parent = folder_id
                continue
            metadata = {
                "name": part,
                "mimeType": "application/vnd.google-apps.folder",
                "parents": [parent],
            }
            created = service.files().create(body=metadata, fields="id").execute()
            parent = created["id"]

    def upload(self, local_path: Path, remote_path: str) -> None:
        """Faz upload de um arquivo local para um caminho remoto no Drive."""
        service = self._build_service()
        parts = remote_path.split("/")
        parent = "root"
        for part in parts[:-1]:
            if not part:
                continue
            folder_id = self._folder_id(part, parent)
            if folder_id is None:
                raise FileNotFoundError(f"Pasta ausente no Drive: {part}")
            parent = folder_id
        metadata = {"name": parts[-1], "parents": [parent]}
        service.files().create(body=metadata, media_body=str(local_path)).execute()

    def download(self, remote_path: str, local_path: Path) -> None:
        """Faz download de um arquivo remoto do Drive para um caminho local."""
        service = self._build_service()
        name = remote_path.split("/")[-1]
        res = (
            service.files()
            .list(q=f"name='{name}' and trashed=false", fields="files(id)")
            .execute()
        )
        files = res.get("files", [])
        if not files:
            raise FileNotFoundError(f"Arquivo não encontrado no Drive: {remote_path}")
        local_path.parent.mkdir(parents=True, exist_ok=True)
        media = service.files().get_media(fileId=files[0]["id"]).execute()
        local_path.write_bytes(media)

    def upload_tree(self, local_dir: Path, remote_prefix: str) -> None:
        """Sobe recursivamente um diretório local para um prefixo remoto no Drive."""
        for path in sorted(local_dir.rglob("*")):
            if not path.is_file():
                continue
            rel = path.relative_to(local_dir).as_posix()
            parent = "/".join(rel.split("/")[:-1])
            if parent:
                self.ensure_folder(f"{remote_prefix}/{parent}")
            self.upload(path, f"{remote_prefix}/{rel}")


def get_drive_client() -> DriveClient:
    """Retorna o cliente do Drive (Kaggle) para operações remotas."""
    return DriveClient()


def mount_drive() -> Path:
    """Monta/acessa o Google Drive e retorna a raiz MyDrive.

    Colab: montagem nativa em /content/drive.
    Kaggle: cache local em /kaggle/working/drive (I/O remoto via Drive API).
    Local: usa a variável DRIVE_ROOT, com fallback em ~/MyDrive (dev only).
    """
    platform = detect_platform()
    if platform == "colab":
        from google.colab import drive

        drive.mount(str(COLAB_DRIVE_ROOT.parent))
        return COLAB_DRIVE_ROOT
    if platform == "kaggle":
        KAGGLE_DRIVE_CACHE.mkdir(parents=True, exist_ok=True)
        return KAGGLE_DRIVE_CACHE
    return Path(os.getenv("DRIVE_ROOT", str(Path.home() / "MyDrive")))


def ensure_storage_root() -> Path:
    """Garante que a raiz tcc/ e todas as subpastas existam (idempotente).

    No Colab/local cria diretórios reais; no Kaggle cria o cache local e as
    pastas remotas via Drive API (best-effort, validado no notebook 00).
    """
    config = get_config()
    root_name = config["storage"]["drive_root"]
    drive_root = mount_drive()
    root = drive_root / root_name
    subfolders = config["storage"]["subfolders"].values()

    if detect_platform() == "kaggle":
        client = get_drive_client()
        client.ensure_folder(root_name)
        for sub in subfolders:
            client.ensure_folder(f"{root_name}/{sub}")
    for sub in subfolders:
        (root / sub).mkdir(parents=True, exist_ok=True)
    return root


def resolve_storage_paths() -> dict[str, Path]:
    """Resolve os caminhos de armazenamento a partir do config.yaml."""
    config = get_config()
    root = ensure_storage_root()
    return {key: root / sub for key, sub in config["storage"]["subfolders"].items()}


def _mirror_repo_root() -> Path:
    """Raiz local do espelho do repositório (MyDrive/tcc/repo) no Colab/local."""
    config = get_config()
    return mount_drive() / config["storage"]["drive_root"] / "repo"


def _mirror_available() -> bool:
    """Indica se o espelho MyDrive/tcc/repo/src existe localmente (Colab/local).

    No Kaggle o espelho é remoto (Drive API) e a leitura de uma árvore inteira
    por API não é usada: lá o notebook sempre obtém o código via bootstrap.
    """
    if detect_platform() == "kaggle":
        return False
    return (_mirror_repo_root() / "src").is_dir()


def _copy_mirror_to_workspace(workspace: Path) -> None:
    """Copia src/ (e data/external/, se houver) do espelho para o workspace."""
    mirror = _mirror_repo_root()
    shutil.copytree(mirror / "src", workspace / "src")
    external = mirror / "data" / "external"
    if external.is_dir():
        shutil.copytree(external, workspace / "data" / "external")


def _seed_repo_mirror(workspace: Path) -> None:
    """Grava src/ e data/external/ no espelho MyDrive/tcc/repo (cria no Drive)."""
    config = get_config()
    root_name = config["storage"]["drive_root"]
    if detect_platform() == "kaggle":
        client = get_drive_client()
        client.upload_tree(workspace / "src", f"{root_name}/repo/src")
        if (workspace / "data" / "external").is_dir():
            client.upload_tree(
                workspace / "data" / "external", f"{root_name}/repo/data/external"
            )
    else:
        mirror = _mirror_repo_root()
        shutil.copytree(workspace / "src", mirror / "src")
        external = workspace / "data" / "external"
        if external.is_dir():
            shutil.copytree(external, mirror / "data" / "external")


def _bootstrap_from_repo(workspace: Path) -> None:
    """Baixa o repositório público e obtém src/ + data/external/ sem espelho.

    Usa o tarball do GitHub (stdlib: urllib + tarfile); sem necessidade de git.
    """
    archive = workspace / "_repo_bootstrap.tar.gz"
    extract_dir = workspace / "_repo_bootstrap"
    try:
        urllib.request.urlretrieve(f"{REPO_URL}/archive/refs/heads/main.tar.gz", archive)
        with tarfile.open(archive, "r:gz") as tf:
            tf.extractall(extract_dir)
        shutil.move(extract_dir / "tcc-main" / "src", workspace / "src")
        external = extract_dir / "tcc-main" / "data" / "external"
        if external.is_dir():
            shutil.move(external, workspace / "data" / "external")
        _seed_repo_mirror(workspace)
    finally:
        archive.unlink(missing_ok=True)
        shutil.rmtree(extract_dir, ignore_errors=True)


def sync_repo_to_workspace(workspace: Path) -> Path:
    """Entrega o código (src/) e os dados externos (data/external/) ao runtime.

    Se o espelho MyDrive/tcc/repo existir, copia dele; se não existir (primeiro
    run), baixa o repositório público e CRIA o espelho no Drive — os próprios
    notebooks geram a estrutura dentro de tcc/. O workspace entra no sys.path.
    """
    for rel in ("src", "data"):
        path = workspace / rel
        if path.exists():
            shutil.rmtree(path)
    if _mirror_available():
        _copy_mirror_to_workspace(workspace)
    else:
        _bootstrap_from_repo(workspace)
    if str(workspace) not in sys.path:
        sys.path.insert(0, str(workspace))
    return workspace / "src"
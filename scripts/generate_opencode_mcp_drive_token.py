"""Gera tokens OAuth do Google Drive para o servidor MCP do opencode (workaround).

O opencode reporta o servidor Drive MCP como conectado sem completar o OAuth
(bug anomalyco/opencode#26195: servidores MCP do Google nao expoem descoberta
OAuth e respondem 200 sem autenticacao). Este script executa o fluxo OAuth do
Google diretamente (PKCE + callback de loopback) e grava os tokens em
~/.local/share/opencode/mcp-auth.json no formato que o opencode espera.

Pre-requisitos:
  - redirect URI http://127.0.0.1:19876/mcp/oauth/callback registrado nas
    Authorized redirect URIs do client OAuth no Google Cloud Console.
  - clientId/clientSecret lidos de opencode.json (mcp.drive.oauth) ou das
    variaveis GOOGLE_DRIVE_MCP_CLIENT_ID / GOOGLE_DRIVE_MCP_CLIENT_SECRET.

Uso:
    python scripts/generate_opencode_mcp_drive_token.py
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import secrets
import threading
import time
import urllib.parse
import urllib.request
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

MCP_SERVER_NAME = "drive"
MCP_SERVER_URL = "https://drivemcp.googleapis.com/mcp/v1"
CALLBACK_HOST = "127.0.0.1"
CALLBACK_PORT = 19876
CALLBACK_PATH = "/mcp/oauth/callback"
REDIRECT_URI = f"http://{CALLBACK_HOST}:{CALLBACK_PORT}{CALLBACK_PATH}"
AUTH_ENDPOINT = "https://accounts.google.com/o/oauth2/v2/auth"
TOKEN_ENDPOINT = "https://oauth2.googleapis.com/token"
SCOPES = [
    "https://www.googleapis.com/auth/drive.readonly",
    "https://www.googleapis.com/auth/drive.file",
]
AUTH_STORE = Path.home() / ".local/share/opencode/mcp-auth.json"
CONFIG_PATH = Path(__file__).resolve().parent.parent / "opencode.json"


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def _sha256_b64url(value: str) -> str:
    return _b64url(hashlib.sha256(value.encode()).digest())


def load_credentials() -> tuple[str, str]:
    client_id = os.environ.get("GOOGLE_DRIVE_MCP_CLIENT_ID")
    client_secret = os.environ.get("GOOGLE_DRIVE_MCP_CLIENT_SECRET")
    if not client_id or not client_secret:
        config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        oauth = config["mcp"][MCP_SERVER_NAME]["oauth"]
        client_id = client_id or oauth["clientId"]
        client_secret = client_secret or oauth["clientSecret"]
    return client_id, client_secret


class CallbackHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path != CALLBACK_PATH:
            self._reply(404, "not found")
            return
        query = urllib.parse.parse_qs(parsed.query)
        if query.get("state") != [self.server.expected_state]:
            self.server.error = "state mismatch"
            self._reply(400, "OAuth state mismatch")
            return
        if "code" in query:
            self.server.authorization_code = query["code"][0]
            self._reply(200, "Autenticação recebida. Pode fechar esta aba.")
        elif "error" in query:
            self.server.error = query["error"][0]
            self._reply(400, f"Erro OAuth: {query['error'][0]}")
        else:
            self._reply(200, "Aguardando callback do Google...")

    def _reply(self, status: int, message: str) -> None:
        body = f"<html><body><p>{message}</p></body></html>".encode()
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args: object) -> None:  # silencia logs do servidor
        pass


class CallbackServer(ThreadingHTTPServer):
    def __init__(self, expected_state: str) -> None:
        super().__init__((CALLBACK_HOST, CALLBACK_PORT), CallbackHandler)
        self.expected_state = expected_state
        self.authorization_code: str | None = None
        self.error: str | None = None


def exchange_code(client_id: str, client_secret: str, code: str, code_verifier: str) -> dict:
    body = urllib.parse.urlencode(
        {
            "code": code,
            "client_id": client_id,
            "client_secret": client_secret,
            "redirect_uri": REDIRECT_URI,
            "grant_type": "authorization_code",
            "code_verifier": code_verifier,
        }
    ).encode()
    request = urllib.request.Request(TOKEN_ENDPOINT, data=body, method="POST")
    request.add_header("Content-Type", "application/x-www-form-urlencoded")
    with urllib.request.urlopen(request) as response:
        return json.loads(response.read())


def store_tokens(token_response: dict) -> None:
    auth = json.loads(AUTH_STORE.read_text(encoding="utf-8")) if AUTH_STORE.exists() else {}
    auth[MCP_SERVER_NAME] = {
        "serverUrl": MCP_SERVER_URL,
        "tokens": {
            "accessToken": token_response["access_token"],
            "expiresAt": int(time.time()) + int(token_response.get("expires_in", 3600)),
            "scope": token_response.get("scope", " ".join(SCOPES)),
        },
    }
    if token_response.get("refresh_token"):
        auth[MCP_SERVER_NAME]["tokens"]["refreshToken"] = token_response["refresh_token"]
    AUTH_STORE.write_text(json.dumps(auth, indent=2), encoding="utf-8")


def main() -> None:
    client_id, client_secret = load_credentials()
    code_verifier = _b64url(secrets.token_bytes(32))
    state = _b64url(secrets.token_bytes(16))

    auth_url = f"{AUTH_ENDPOINT}?" + urllib.parse.urlencode(
        {
            "client_id": client_id,
            "redirect_uri": REDIRECT_URI,
            "response_type": "code",
            "scope": " ".join(SCOPES),
            "access_type": "offline",
            "prompt": "consent",
            "state": state,
            "code_challenge": _sha256_b64url(code_verifier),
            "code_challenge_method": "S256",
        }
    )

    server = CallbackServer(state)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    print(f"Abrindo navegador para autorização...\n{auth_url}")
    webbrowser.open(auth_url)

    deadline = time.time() + 300
    while time.time() < deadline and not server.authorization_code and not server.error:
        time.sleep(0.5)

    if server.error:
        raise SystemExit(f"Erro OAuth: {server.error}")
    if not server.authorization_code:
        raise SystemExit("Tempo esgotado aguardando autorização.")

    token_response = exchange_code(
        client_id, client_secret, server.authorization_code, code_verifier
    )
    store_tokens(token_response)
    server.shutdown()
    print(f"Tokens gravados em {AUTH_STORE}")
    print("Reinicie o opencode e confira com: opencode mcp auth list")


if __name__ == "__main__":
    main()

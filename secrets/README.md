# Tokens e credenciais (Kaggle)

Este diretório **não contém segredos** — contém apenas modelos/documentação. Os
tokens reais NUNCA são versionados (repo público; regra do `AGENTS.md`).

## Por que existem tokens?

O **Colab** autentica com a conta logada de forma nativa (Drive e Earth Engine).
O **Kaggle** não monta o Drive e roda em modo headless — por isso precisa de dois
tokens, obtidos uma única vez com a **conta principal**:

| Variável            | Para quê                                  | Como obter                                   | Onde guardar               |
| ------------------- | ----------------------------------------- | -------------------------------------------- | -------------------------- |
| `GDRIVE_TOKEN`      | Acesso ao `MyDrive/tcc/` via Drive API    | `scripts/generate_drive_token.py`            | Secret do Kaggle (env var) |
| `GDRIVE_TOKEN_FILE` | Alternativa: caminho de um arquivo token  | idem (envie o arquivo como dataset)          | arquivo no Kaggle          |
| `GEE_CREDENTIALS`   | Autenticação headless do Earth Engine     | `scripts/generate_gee_credentials.py`        | Secret do Kaggle (env var) |
| `GEE_PROJECT`       | Projeto Cloud do GEE (`ee.Initialize`)    | ID visto no Earth Engine Code Editor         | Secret do Kaggle (env var) |

## Como usar

1. Gere os tokens na sua máquina (ou num Colab) com os scripts em `scripts/`.
2. **Cópia local (opcional, git-ignored):** salve os tokens aqui em `secrets/`
   (`drive_token.json`, `earthengine_credentials.json`) — este diretório tem um
   `.gitignore` próprio e esses arquivos **não** são versionados.
3. **Kaggle:** cole o conteúdo nos **Secrets** do Kaggle (Settings → Secrets)
   com os nomes acima (`GDRIVE_TOKEN`, `GEE_CREDENTIALS`).
4. O notebook `00` lê dessas variáveis de ambiente; em Colab, com fallback para
   arquivos em `MyDrive/tcc/secrets/` (ex.: `drive_token.json`,
   `earthengine_credentials.json`).

## Regras

- Autorize sempre com a **mesma conta pessoal** — caso contrário a estrutura
  `tcc/` aparecerá em outro Drive/conta.
- Nunca commitar, logar ou expor os tokens. Use apenas variáveis de ambiente.
- O `GDRIVE_TOKEN` também é usado pelo `sync_repo_to_workspace()` e
  `ensure_storage_root()` no Kaggle (ver `src/io.py`).
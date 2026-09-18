"""Valida convenções estruturais de notebooks do projeto (somente stdlib)."""

import json
import re
import sys
from pathlib import Path

NAME_RE = re.compile(r"^\d{2}_[a-z0-9]+(?:_[a-z0-9]+)*\.ipynb$")


def check(path: Path) -> list[str]:
    """Retorna lista de problemas estruturais do notebook; vazia se conforme."""
    problems: list[str] = []
    if not NAME_RE.match(path.name):
        problems.append(f"nome fora do padrão NN_verb_snake_case.ipynb: {path.name}")

    if not path.exists():
        return [f"arquivo não encontrado: {path}"]

    try:
        nb = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return [f"JSON inválido: {exc}"]

    cells = nb.get("cells", [])
    if not cells:
        return ["notebook sem células"]

    prev_type: str | None = None
    for i, cell in enumerate(cells):
        ctype = cell.get("cell_type")
        source = "".join(cell.get("source", []))
        if ctype == "code" and prev_type != "markdown":
            problems.append(f"célula {i}: célula de código sem markdown imediatamente acima")
        if ctype == "code" and not source.strip():
            problems.append(f"célula {i}: célula de código vazia")
        if ctype == "code" and cell.get("outputs"):
            problems.append(
                f"célula {i}: célula de código com outputs versionados "
                "(limpe os outputs antes do commit)"
            )
        prev_type = ctype

    return problems


def main() -> int:
    targets = [Path(p) for p in sys.argv[1:]]
    if not targets:
        notebooks = Path("notebooks")
        targets = sorted(notebooks.glob("*.ipynb")) if notebooks.is_dir() else []

    if not targets:
        print("Nenhum notebook encontrado.")
        return 0

    total = 0
    for t in targets:
        problems = check(t)
        if problems:
            total += len(problems)
            print(f"[FALHA] {t}")
            for p in problems:
                print(f"  - {p}")
        else:
            print(f"[OK] {t}")

    print(f"\n{total} problema(s) encontrado(s).")
    return 1 if total else 0


if __name__ == "__main__":
    raise SystemExit(main())

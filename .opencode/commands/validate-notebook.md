---
description: Valida um notebook contra as convenções do AGENTS.md e do PLAN.md
agent: agents-compliant-reviewer
subtask: true
---

Valide o notebook `$ARGUMENTS` contra as regras de escrita do repositório.

Passos:
1. Leia `AGENTS.md` e `PLAN.md` para ancorar a revisão nas regras vigentes.
2. Inspecione o notebook `$ARGUMENTS`.
3. Reporte uma lista de verificação com: regra violada, localização exata (número da célula ou `arquivo:linha`) e a correção exigida.
4. Ordene por severidade: bloqueadores antes de sugestões.

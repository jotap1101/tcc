import type { Plugin } from "@opencode-ai/plugin";

// Padrões de arquivos de credenciais/segredos que nunca devem ser lidos pelo LLM.
const BLOCKED_READ_PATTERNS: RegExp[] = [
  /(^|\/)\.env(\.|$)/,
  /(^|\/)\.envrc$/,
  /service[-_]account.*\.json$/i,
  /articulate-case-.*\.json$/i,
  /\.(pem|p12|pfx|key)$/i,
];

// Exceções permitidas (ex.: template de variáveis sem valores reais).
const ALLOWLIST: RegExp[] = [/(^|\/)\.env\.example$/];

export const EnvProtection: Plugin = async () => {
  return {
    "tool.execute.before": async (input, output) => {
      // Intercepta apenas a ferramenta de leitura de arquivos.
      if (input.tool !== "read") return;
      const filePath = (output.args as { filePath?: string } | undefined)
        ?.filePath;
      if (!filePath) return;

      const normalized = filePath.replace(/\\/g, "/");
      if (ALLOWLIST.some((re) => re.test(normalized))) return;
      if (BLOCKED_READ_PATTERNS.some((re) => re.test(normalized))) {
        throw new Error(
          "Leitura bloqueada: arquivo de credenciais/segredos. Acesse apenas via variáveis de ambiente.",
        );
      }
    },
  };
};

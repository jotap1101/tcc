import { tool } from "@opencode-ai/plugin";
import path from "path";

// Valida convenções estruturais de notebooks .ipynb do projeto via script Python.
export default tool({
  description:
    "Valida convenções estruturais de notebooks .ipynb (nome NN_verb_snake_case.ipynb, markdown imediatamente acima de cada célula de código, célula de código não vazia e sem outputs versionados). Use ao revisar notebooks do projeto.",
  args: {
    paths: tool.schema
      .array(tool.schema.string())
      .optional()
      .describe(
        "Caminhos dos notebooks a validar; se omitido, valida todos em notebooks/",
      ),
  },
  async execute(args, context) {
    const script = path.join(
      context.worktree,
      ".opencode/tools/scripts/check_notebook.py",
    );
    const targets = args.paths ?? [];

    // Executa o script com argv explícito, evitando problemas de aspas em caminhos.
    const proc = Bun.spawn(["python3", script, ...targets], {
      cwd: context.worktree,
      stdout: "pipe",
      stderr: "pipe",
    });
    const out = await new Response(proc.stdout).text();
    const err = await new Response(proc.stderr).text();
    const code = await proc.exited;
    return (
      [out, err].filter(Boolean).join("\n").trim() ||
      `python3 saiu com código ${code}`
    );
  },
});

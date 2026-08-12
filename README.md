# Trabalho de Conclusão de Curso – Bacharelado em Ciência da Computação (IFSULDEMINAS)

Segmentação Semântica de Lavouras Cafeeiras: Análise Comparativa entre Redes Neurais Convolucionais (CNN) e Vision Transformers (ViT) na Agricultura de Precisão.

O estudo tem como alvo a Região Geográfica Imediata de Guaxupé – MG, Brasil, utilizando imagens Sentinel-2 Nível-2A para avaliar como a transição de um viés convolucional local para a autoatenção global afeta a precisão geométrica e a interpretabilidade sob as restrições do relevo montanhoso.

## Sumário

- [Visão Geral](#visão-geral)
- [Metodologia](#metodologia)
- [Estrutura do Repositório](#estrutura-do-repositório)
- [Pré-requisitos](#pré-requisitos)
- [Execução](#execução)
- [Status](#status)
- [Contribuição](#contribuição)
- [Licença](#licença)

## Visão Geral

- **Tarefa:** segmentação semântica binária (café vs. não-café).
- **Modelos:** U-Net (CNN) e SegFormer (ViT, MiT-b0–b2).
- **Dados:** Sentinel-2 Nível-2A, bandas B2/B3/B4/B8 com resolução espacial de 10 m.
- **Função de perda:** combinação de Dice + Focal + Boundary Loss.
- **Validação:** k-fold espacial (k=5) com métricas em nível de pixel: IoU, F1-Score, Precisão e Revocação.
- **Explicabilidade:** Grad-CAM (U-Net) e Attention Rollout (SegFormer).

## Metodologia

O pipeline abrange aquisição, pré-processamento, treinamento, avaliação e explicabilidade:

1. Aquisição no Google Earth Engine de mosaicos Sentinel-2 e máscaras de referência (MapBiomas / AlphaEarth / super-resolução S2DR, sob avaliação).
2. Mascaramento de nuvens (QA60), normalização e recorte em patches de 512×512 pixels.
3. Particionamento em k-fold espacial para evitar vazamento por autocorrelação espacial.
4. Protocolo de treinamento idêntico para ambas as arquiteturas — apenas o modelo difere.
5. Avaliação de métricas em nível de pixel e geração de mapas de calor de XAI.

O `PLAN.md` é o roteiro oficial, com ordenação das fases, entradas/saídas de cada notebook e um rastreador de progresso.

## Estrutura do Repositório

```bash
tcc/
├── AGENTS.md              # fonte de verdade para regras de código e formatação
├── PLAN.md                # roteiro de implementação
├── pyproject.toml         # uv, ruff, mypy, pytest
├── src/tcc/               # pacote Python compartilhado (lógica reutilizável)
├── tests/                 # pytest apenas sobre src/
├── notebooks/             # estágios .ipynb sequenciais
├── data/                  # dados espectrais e máscaras (ignorado pelo git)
├── models/                # pesos dos modelos (ignorado pelo git)
└── artifacts/             # métricas e figuras (ignorado pelo git)
```

## Pré-requisitos

- Python com PyTorch e Hugging Face Transformers.
- Conta no Google Earth Engine (autenticação via conta de serviço).
- Ambiente de execução com GPU (prioritariamente Kaggle; Google Colab como alternativa).

## Execução

O projeto segue um layout híbrido: os notebooks Jupyter orquestram cada estágio, enquanto a lógica reutilizável reside no pacote `src/tcc/`. Os notebooks devem ser executados em ordem numérica e resolver todos os caminhos de entrada a partir de `src/tcc/config.py`.

O treinamento e o armazenamento de artefatos ocorrem no Kaggle; linting, verificação de tipos e testes rodam no CI (GitHub Actions). Consulte o `AGENTS.md` para as regras operacionais completas.

## Status

Estágio inicial: o código base, as dependências e a configuração ativa de ferramentas ainda estão em construção.

## Contribuição

Siga as convenções definidas no `AGENTS.md`: layout híbrido notebook/pacote, identificadores em en-US, markdown dos notebooks em pt-BR e configuração reprodutível. Credenciais nunca devem ser commitadas.

## Licença

Licenciamento duplo:

- **MIT** — código-fonte.
- **CC BY 4.0** — documentação e conjuntos de dados.

# EvalOral — pacote de reprodutibilidade

Este repositório reúne o código, os dados canônicos, os artefatos de referência e os protocolos de validação usados no trabalho **EvalOral: An Explainable Evolving Neuro-Fuzzy Framework for Oral Image Pattern Discovery and Rule-Aware Visual Auditing**.

Ele foi organizado para que outra pessoa consiga distinguir claramente:

- o que é **resultado computacional rerodável**;
- o que é **snapshot de referência usado no manuscrito**;
- o que é **feedback humano autoral**;
- o que é **protocolo de validação externa ainda independente**;
- e o que pertence às **extensões anatômicas futuras**.

> O EvalOral é código de pesquisa e não é um dispositivo clínico.

> A licença MIT cobre o software do projeto, não relicencia as fotografias de origem. Consulte [`DATA_LICENSE_NOTICE.md`](DATA_LICENSE_NOTICE.md) antes de redistribuir `data/raw/`.

## O que está incluído

- 441 registros canônicos dos módulos-fonte;
- 257 fotografias com conteúdo único por SHA-256 para clustering não supervisionado;
- 121 imagens DAI com rótulos explícitos de vista (56 frontal, 65 oclusal);
- features grayscale + edge de 6.272 dimensões;
- StandardScaler + PCA whitening com 4 PCs no experimento principal;
- neuro-fuzzy evolutiva com regras Gaussianas, relevância de features, coverage e Unim AND/OR/COMP;
- 30 ordens aleatórias por voluntário;
- baseline LinearSVC;
- ablação;
- grade `sigma_init × alpha`;
- análise de merge;
- rule history;
- clustering `k=2...12` e protótipos `k=7`;
- concordância cluster–regra;
- CNN separada + Grad-CAM;
- replay do feedback humano em 40 imagens;
- leave-one-volunteer-out após feedback;
- K-means inicializado por feedback;
- comparação real de memberships antes/depois do feedback;
- regras fuzzy literais;
- kit/protocolo de validação externa;
- kits anatômicos HITL como extensão futura.

## Começo rápido

```bash
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
# Windows: .venv\Scripts\activate
pip install -e ".[all]"
python scripts/check_repo.py
pytest -q
```

Para um teste rápido de ponta a ponta:

```bash
bash scripts/reproduce_smoke.sh
```

Depois, para a reprodução em escala do manuscrito:

```bash
bash scripts/reproduce_core.sh
```

Para também treinar a CNN e gerar novos Grad-CAMs:

```bash
bash scripts/reproduce_all.sh
```

## Um cuidado científico encontrado durante a montagem do pacote

A parte neuro-fuzzy principal reproduz o resultado central do manuscrito com alta fidelidade: mediana de macro-F1 ≈ **0,93435**, accuracy ≈ **0,93494** e **9 regras** finais na mediana.

Entretanto, o rerun pareado do LinearSVC sob o protocolo hoje descrito no paper produz mediana de macro-F1 em torno de **0,939**, enquanto o manuscrito registra **0,964**. Eu não escondi essa diferença e não alterei o código para forçar a coincidência. O ponto está documentado em `docs/REPRODUCIBILITY_NOTES.md` e deve ser conferido pelos autores antes da publicação pública do GitHub.

Isso é exatamente o tipo de detalhe que um bom pacote de reprodutibilidade deve tornar visível.

## Mapa paper → código

Veja [`docs/PAPER_TO_CODE_MAP.md`](docs/PAPER_TO_CODE_MAP.md).

## Limites importantes

- A CNN do Grad-CAM é separada da neuro-fuzzy principal.
- O K-means não cria as regras fuzzy do modelo principal.
- `lateral` entra apenas no experimento de feedback humano.
- os nomes PUFA/trauma dos módulos não são tratados automaticamente como diagnóstico de todas as imagens;
- os kits anatômicos representam continuação da pesquisa, não resultado do paper atual.

## Publicação no GitHub

Antes de tornar público, veja [`docs/GITHUB_PUBLISH_CHECKLIST.md`](docs/GITHUB_PUBLISH_CHECKLIST.md).

## Auditoria do pacote

O estado dos testes, smoke runs e ressalvas de reconstrução histórica está em [`docs/RELEASE_AUDIT.md`](docs/RELEASE_AUDIT.md).

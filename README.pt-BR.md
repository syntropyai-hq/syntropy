<!-- Translation of README.md. Update it in the same change as README.md. -->
<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/media/logo-dark.svg">
    <img src="docs/media/logo-light.svg" alt="Syntropy" width="360">
  </picture>
</p>

<p align="center"><a href="README.md">English</a>&emsp;<a href="README.zh-CN.md">简体中文</a>&emsp;<a href="README.ja.md">日本語</a>&emsp;<a href="README.de.md">Deutsch</a>&emsp;<a href="README.fr.md">Français</a>&emsp;<a href="README.es.md">Español</a>&emsp;<b>Português</b></p>

<p align="center"><sub>Esta é uma tradução. A versão de referência é o <a href="README.md">README em inglês</a>.</sub></p>

<h3 align="center">Rode um modelo grande em todas as GPUs, níveis de memória e máquinas que você tem</h3>

<p align="center">Posicionamento decidido pela medição do seu hardware. Saída que, por padrão, corresponde à referência.<br>
Em Rust, com kernels nativos para CUDA, HIP, Metal, Vulkan e CPU.</p>

<p align="center">
  <a href="#roadmap"><img src="https://img.shields.io/badge/status-pre--alpha%2C_M0a_in_progress-7b5fd0" alt="Status: pré-alfa, M0a em andamento"></a>
  <a href="#licença-e-créditos"><img src="https://img.shields.io/badge/licence-MIT_OR_Apache--2.0-3f6fd1" alt="Licença: MIT OR Apache-2.0"></a>
  <a href="Cargo.toml"><img src="https://img.shields.io/badge/Rust-2024_edition-c2572b?logo=rust&logoColor=white" alt="Rust, edição 2024"></a>
  <a href="PLATFORMS.md"><img src="https://img.shields.io/badge/platforms-Linux%2C_Windows%2C_macOS-2f9a5a" alt="Plataformas: Linux, Windows, macOS"></a>
</p>

<p align="center">
  <a href="#como-funciona"><b>Como&nbsp;funciona</b></a>&emsp;
  <a href="#roadmap"><b>Roadmap</b></a>&emsp;
  <a href="PLATFORMS.md"><b>Plataformas</b></a>&emsp;
  <a href="plan/README.md"><b>O&nbsp;design</b></a>&emsp;
  <a href="CONTRIBUTING.md"><b>Como&nbsp;contribuir</b></a>
</p>

> [!IMPORTANT]
> **O Syntropy está em construção, e ainda não há nada para instalar.** O design completo está em [`plan/`](plan/README.md), e a implementação está em andamento no marco M0a. Tudo nesta página descreve o que foi projetado e planejado, não algo que você possa rodar hoje.

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/media/memory-pool-dark.svg">
    <img src="docs/media/memory-pool-light.svg" width="860" alt="Três máquinas desenhadas na mesma escala. Um PC gamer com 24 GB de VRAM e 64 GB de RAM tem 88 GB, um notebook tem 48 GB de memória unificada e um mini PC tem 128 GB. Nenhuma delas comporta sozinha um modelo de 142 GB. Juntas, têm 264 GB, que comportam o modelo com 122 GB de folga.">
  </picture>
</p>

Um PC gamer com placa de vídeo de 24 GB e 64 GB de RAM, um notebook com 48 GB de memória unificada e um mini PC de 128 GB embaixo da TV não comportam, cada um sozinho, um modelo mixture-of-experts de 235 bilhões de parâmetros, que ocupa 142 GB em 4 bits. Juntos, eles têm 264 GB.

O Syntropy trata essas máquinas como um único computador com vários tipos de memória. Ele mede a velocidade com que cada GPU e cada CPU lê cada nível de memória, escolhe um único posicionamento para o modelo inteiro e mantém a computação junto dos bytes: o dispositivo cuja memória guarda um peso é o dispositivo que computa com ele. Entre máquinas, só as ativações trafegam durante o decode. Dentro de uma máquina, um peso que não está no cache da GPU é computado pela CPU ali mesmo onde está, em vez de ser copiado pelo PCIe a cada token.

## Funcionalidades

Tudo isto está especificado no design, e nada disso foi construído ainda.

### Dentro de uma máquina

- **Todo nível de memória trabalha.** Os pesos e os experts MoE mais quentes ficam em um cache adaptativo na VRAM. O conjunto completo de pesos fica residente na RAM, onde a CPU computa os cache misses ali mesmo. Experts frios e tabelas grandes são lidos do NVMe com direct I/O quando necessário.
- **Experts buscados antes de serem necessários.** A previsão dos experts da próxima camada roda dentro do kernel do roteador, então as leituras de NVMe e de RAM para a próxima camada já estão em andamento enquanto a camada atual é computada.
- **Medido, não tirado de tabela.** Cada máquina testa o que suas GPUs, CPUs, discos e links de rede entregam, usando os kernels reais do Syntropy. O posicionamento é construído a partir dessas medições, não de fichas técnicas.

### Entre máquinas

- **Um único posicionamento para o modelo inteiro.** Um único plano decide quais máquinas participam, qual faixa de camadas cada uma executa, qual nível guarda cada grupo de tensores e qual dispositivo o computa. O plano registra por que cada grupo ficou onde ficou.
- **Um anel de máquinas.** O modelo é dividido em estágios de pipeline que passam os estados ocultos direto para a próxima máquina. Uma única camada só é dividida entre máquinas em links cuja latência medida fica dentro de um limite, como RDMA sobre Thunderbolt 5.
- **Sem configuração na LAN, um token fora dela.** Máquinas na mesma rede se encontram sozinhas. Uma máquina nova é admitida depois que você confirma o mesmo código de seis dígitos nas duas telas. Um notebook longe de casa entra com um token de convite que, por padrão, é de uso único.
- **Perder uma máquina não perde a resposta.** Cada estágio guarda o que enviou recentemente. Quando a máquina de um estágio cai e as outras têm espaço, a geração continua sem que nenhum erro chegue ao cliente.

### Para você e suas ferramentas

- **Exato, a menos que você diga o contrário.** Atalhos que mudam a saída, como pular experts que não estão no cache, são opt-in, têm nome e aparecem listados em toda resposta que afetaram.
- **Decodificação especulativa que preserva a resposta.** Heads MTP ou um modelo de rascunho propõem vários tokens, o modelo verifica todos em uma única passada, e a regra de aceitação preserva a distribuição de saída do próprio modelo.
- **As APIs que suas ferramentas já usam.** APIs no formato da OpenAI e da Anthropic, além de uma nativa, servidas pela máquina que receber a requisição, seja ela qual for.
- **Código nativo de ponta a ponta.** Uma camada de sistema em Rust, kernels nativos para CUDA, HIP, Metal, Vulkan e CPU, e o ggml como fallback por operação, para ampliar a cobertura.

## Uma primeira olhada

Esta é a linha de comando planejada na [RFC-0051](plan/rfcs/RFC-0051-cli-and-dashboard.md). Nenhum destes comandos existe ainda.

```sh
syntropy run                     # faz uma pergunta e recomenda um modelo que caiba
syntropy plan <model> --explain  # mostra o posicionamento e a velocidade prevista sem carregar
syntropy cluster approve <id>    # admite uma máquina da LAN após conferir o código de seis dígitos
syntropy invite create           # token de uso único para um notebook longe de casa
```

Por padrão, cada máquina vai servir as APIs em `http://127.0.0.1:7433`, então os clientes e SDKs da OpenAI e da Anthropic que você já usa vão funcionar sem alterações ([RFC-0050](plan/rfcs/RFC-0050-api-surface.md)):

```sh
curl http://127.0.0.1:7433/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model": "<model>", "stream": true,
       "messages": [{"role": "user", "content": "Por que o decode é limitado pela memória?"}]}'
```

## Roadmap

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/media/milestones-dark.svg">
    <img src="docs/media/milestones-light.svg" width="860" alt="Sete marcos, de M0a a M4. O M0a, o walking skeleton em uma única máquina Linux, está em andamento. O M1a é a primeira versão prévia, o M1b adiciona Windows e Apple silicon, o M2 é o lançamento, com máquinas em uma LAN, o M3 completa as funcionalidades, e o M4 é a 1.0.">
  </picture>
</p>

O design está completo: 25 RFCs, 98 ADRs e 1.411 tarefas em seis áreas do código, cada tarefa com critérios de aceite testáveis e as seções de RFC e os ADRs que implementa. Os marcos não têm datas. A [RFC-0001 §16](plan/rfcs/RFC-0001-system-architecture.md#16-milestones) define o escopo e os critérios de saída de cada um, o [ADR-0020](plan/decisions/ADR-0020-platform-support-tiers.md) define as plataformas que cada um lança, e o [roadmap](plan/implementation/roadmap.md) lista as tarefas de cada marco em ordem de dependência. O progresso é acompanhado no [quadro do projeto](https://github.com/orgs/syntropyai-hq/projects/1).

<details>
<summary><b>O que cada marco entrega</b></summary>
<br>

| Marco | O que entrega |
|---|---|
| **M0a** | Um walking skeleton em uma única máquina Linux: uma família densa (Llama) em Q4_K e Q8_0 na CPU e em uma GPU NVIDIA, chat em streaming no estilo OpenAI passando pelo runner real, com conformidade, o harness de paridade e a verificação contra a referência de quem publica o modelo como gates de merge desde o primeiro commit. Construído e testado, não lançado. |
| **M0b** | O restante do M0 na mesma máquina: uma família MoE (Qwen-MoE, com todos os experts na VRAM), tolerâncias calibradas, seleção certificada de kernels, as primeiras sondagens de hardware e um harness de benchmark vinculado a commits. Não lançado. |
| **M1a** | Níveis de memória em uma única máquina Linux com GPU NVIDIA ou só CPU: os níveis VRAM, RAM e NVMe, o cache de experts com previsão e prefetch, posicionamento em uma única máquina, paginação de KV com cache de prefixos e sessões salvas, e decodificação especulativa. A primeira versão pública, marcada como prévia. |
| **M1b** | Windows com NVIDIA e macOS em Apple silicon, além de só CPU nos dois. O escopo depende do que o M1a medir. |
| **M2** | Várias máquinas em uma LAN: descoberta e pareamento, transportes criptografados, o pipeline em anel, distribuição de pesos peer-to-peer e o primeiro dashboard, além de uma prévia de Vulkan baseada em ggml para GPUs AMD e Intel. Esta versão é o lançamento. |
| **M3** | Funcionalidades completas: links de convite, recuperação após a perda de uma máquina, paralelismo de tensores e de experts em links rápidos, especulação entre máquinas, kernels nativos de HIP e Vulkan, e um modo de throughput para vários usuários. |
| **M4** | 1.0: plataformas Tier 2 e Tier 3, instaladores e atualizações assinados, e gates de regressão de desempenho em hardware de referência. |

</details>

## Onde vai rodar

Cada plataforma tem um tier de suporte ([ADR-0020](plan/decisions/ADR-0020-platform-support-tiers.md)), que descreve o suporte que ela terá na 1.0. O [PLATFORMS.md](PLATFORMS.md) traz a matriz completa, com versões mínimas, backends e o marco em que cada plataforma sai pela primeira vez.

<img src="https://img.shields.io/badge/tier_1-supported-2f9a5a" alt="Tier 1: com suporte"><br>Linux x86_64 com NVIDIA, Windows 11 x86_64 com NVIDIA, macOS 15+ em Apple silicon, e só CPU nos três. Builds pré-compilados e assinados. Paridade e desempenho são testados em hardware real, e uma falha bloqueia o lançamento.

<img src="https://img.shields.io/badge/tier_2-known_gaps-3f6fd1" alt="Tier 2: com suporte e lacunas conhecidas"><br>Linux com AMD ROCm; Vulkan em GPUs AMD, Intel e NVIDIA; AMD Strix Halo; Linux arm64 da classe GB10; placas da classe Raspberry Pi 5 (CPU); NICs RDMA; várias GPUs em uma máquina. Builds pré-compilados, paridade obrigatória, desempenho acompanhado. As lacunas são listadas nas notas de versão.

<img src="https://img.shields.io/badge/tier_3-best_effort-818b98" alt="Tier 3: melhor esforço"><br>Android e iOS como nós; Intel SYCL; CUDA 12 para GPUs Pascal e Volta; WebGPU; Windows 10 22H2; Linux no WSL2 com GPU NVIDIA. Builds publicados como experimentais.

Fora dos planos para a 1.0: macOS em Intel, sistemas operacionais de 32 bits, Windows em arm64 e um nó rodando dentro de um navegador.

## Como funciona?

Gerar um token para um usuário lê uma vez cada peso ativo e o KV cache inteiro. A aritmética por byte é pequena, então a velocidade de decode depende de onde os bytes estão e da velocidade com que o processador ao lado deles consegue lê-los. O Syntropy é projetado em torno disso: a computação vai até os bytes, e os bytes são posicionados por medição.

### Dentro de uma máquina

<p align="center"><img src="docs/media/tiers.svg" width="760" alt="Uma máquina: a VRAM guarda os pesos quentes e um cache de experts e é lida pela GPU; a RAM guarda todos os pesos, e a CPU computa ali os cache misses; o NVMe guarda experts frios e tabelas, com prefetch para a RAM"></p>

- A **VRAM** guarda o que todo token usa (atenção, camadas densas, experts compartilhados, o KV cache) e um cache dos experts MoE mais usados, que se adapta à carga de trabalho.
- A **RAM** guarda o conjunto completo de pesos. Quando um token precisa de um expert que não está no cache da GPU, a CPU o computa ali mesmo, em paralelo com a GPU, em vez de copiá-lo pelo PCIe. Uma CPU de desktop lê DDR5 dual-channel a 76–122 GB/s, cerca de 3–5× o que o PCIe 4.0 x16 entrega à GPU.
- O **NVMe** guarda experts frios e grandes tabelas de consulta, armazenados em extents de experts inteiros e lidos com direct I/O. A previsão da próxima camada inicia essas leituras antes que o roteador as peça.

O gerenciador de níveis do Syntropy controla cada byte residente, então o page cache do sistema operacional nunca decide o que fica na memória. Os pesos só são enviados em streaming para a GPU em lotes grandes, como trechos longos de prompt, e só quando o ponto de equilíbrio medido mostra que copiar compensa mais do que computar onde os pesos já estão.

### Entre máquinas

<p align="center"><img src="docs/media/ring.svg" width="760" alt="Três máquinas em anel, cada uma executando uma faixa de camadas: os estados ocultos passam de uma para a seguinte, e a última amostra o token, devolve-o à primeira e o envia em streaming para a máquina que recebeu a requisição"></p>

O modelo é dividido em estágios por faixa de camadas, um estágio por máquina, e os estágios formam um anel. O primeiro estágio gera o embedding da entrada, e cada estágio passa seu estado oculto diretamente para o seguinte. O último estágio amostra o token, devolve-o pelo anel e o envia em streaming para a máquina que recebeu a requisição. Um pipeline adiciona capacidade, não velocidade a um único fluxo, então o ganho vem de mover bytes para uma memória mais rápida. Uma máquina lenta fica fora do anel quando só acrescentaria um salto.

Dividir uma única camada entre máquinas, com paralelismo de tensores ou de experts, custa uma ida e volta pela rede por camada, por token. O Syntropy só faz isso em links cuja latência medida das operações coletivas é baixa o bastante para compensar. Na prática, isso significa links da classe RDMA. Wi-Fi nunca se qualifica.

Uma máquina é eleita coordenadora e mantém o log de eventos do cluster, que registra os membros e os planos. Ela nunca vê um token e não está no caminho das requisições, então perdê-la não derruba nenhuma requisição.

> [!TIP]
> A explicação mais longa, em linguagem simples, está em [docs/HOW_IT_WORKS.md](docs/HOW_IT_WORKS.md). Os designs completos estão nas RFCs: níveis e residência ([RFC-0020](plan/rfcs/RFC-0020-tier-manager-and-residency.md)), o caminho de I/O dos pesos ([RFC-0021](plan/rfcs/RFC-0021-weight-io-path.md)), o cache de experts ([RFC-0023](plan/rfcs/RFC-0023-moe-expert-cache-prediction-prefetch.md)), posicionamento ([RFC-0011](plan/rfcs/RFC-0011-joint-placement-solver.md)), execução distribuída ([RFC-0032](plan/rfcs/RFC-0032-distributed-execution.md)) e descoberta e convites ([RFC-0030](plan/rfcs/RFC-0030-discovery-membership-invites.md)).

## Exato por padrão

Com as configurações padrão, a saída do Syntropy deve corresponder a um forward pass de referência do mesmo modelo, na mesma quantização, dentro de tolerâncias publicadas ([ADR-0019](plan/decisions/ADR-0019-exactness-contract.md)).

- **A referência** é o executor de referência do Syntropy: acumulação em F32, ordem de redução fixa e pesos desquantizados exatamente a partir do mesmo arquivo. Ele, por sua vez, é verificado contra a implementação de referência de quem publica o modelo, então "exato" cobre tanto a leitura que o Syntropy faz do modelo quanto os seus kernels.
- **O teste** mede a concordância top-1 com teacher forcing e a divergência KL sobre um corpus versionado de chat, código, contexto longo, JSON de tool calls e texto multilíngue. As tolerâncias são definidas por classe de dtype a partir de medições, versionadas e gravadas em cada versão lançada.
- **O que precisa passar** é todo caminho que o planejador pode escolher no modo exato: cada kernel e cada fallback, cada formato de posicionamento, de uma GPU até um anel de máquinas, decodificação especulativa e comum, paginação de KV entre níveis e sessões retomadas. Um caminho que falha é aproximado, seja qual for a intenção original.
- **Atalhos são opt-in e têm nome**, como `expert_skip` (experts ausentes do cache não são computados) ou `kv_lossy` (KV armazenado abaixo da precisão do modelo). Toda resposta informa sua fidelidade e os atalhos que estavam ativos, e uma requisição que pede `exact` nunca é atendida por uma configuração aproximada.

Não se promete saída idêntica byte a byte entre dispositivos diferentes, porque cada hardware arredonda de um jeito dentro da tolerância. Está planejado um modo `deterministic`, mais lento, para saída reproduzível com o mesmo build e os mesmos dispositivos. As tolerâncias e o harness de paridade estão especificados na [RFC-0042](plan/rfcs/RFC-0042-quantization-and-numerical-fidelity.md) e na [RFC-0060](plan/rfcs/RFC-0060-testing-and-correctness.md).

## O design

O [`plan/README.md`](plan/README.md) é o índice e sugere uma ordem de leitura.

| Caminho | O que contém |
|---|---|
| [`plan/00-vision-and-scope.md`](plan/00-vision-and-scope.md) | O problema, os cenários, a matriz de plataformas, os modelos-alvo, os critérios de sucesso SC-1 a SC-32 (todos são metas) e o que significa "funcionalidades completas" |
| [`plan/decisions/`](plan/decisions/) | ADRs. O ADR-0000 traz as diretrizes de produto D1–D8, os ADR-0001 a ADR-0022 formam a base, e os demais são numerados por área |
| [`plan/rfcs/`](plan/rfcs/) | A RFC-0001, com a arquitetura do sistema, e 24 RFCs de domínio para posicionamento, níveis, execução distribuída, computação, produto e qualidade |
| [`plan/implementation/`](plan/implementation/README.md) | As tarefas em seis arquivos de épicos, o roadmap, o relatório de dependências, a rastreabilidade, o registro de riscos e os scripts que verificam tudo isso |
| [`plan/open-questions.md`](plan/open-questions.md) | O que ainda não foi decidido, cada item com a tarefa ou a medição que vai resolvê-lo |
| [`plan/glossary.md`](plan/glossary.md) | Um termo por conceito e os sinônimos que o plano não usa |
| [`plan/research/`](plan/research/) | As evidências: notas sobre inferência com vários dispositivos e com memória em níveis, backends e um modelo de custo de hardware com valores de largura de banda com fonte citada |

Todo número do plano é rotulado como `[target]`, como `[estimate]` com sua derivação, ou como uma medição de hardware com fonte citada. Nenhum é um benchmark do Syntropy, porque ainda não existe nada para medir.

## Como contribuir

Agentes de IA constroem o Syntropy executando as tarefas do plano. Um agente pega uma tarefa, implementa-a em uma branch própria e abre uma única alteração que atende aos critérios de aceite da tarefa e cita as seções de RFC e os ADRs que implementa. O grafo de tarefas decide o que pode começar.

Uma alteração só é mesclada quando passa em todos estes itens ([ADR-0021](plan/decisions/ADR-0021-development-model-ai-agents-open-source.md)):

- build, lint, formatação e testes unitários, de propriedade e de simulador;
- o harness de paridade e os gates de conformidade;
- os gates de regressão de desempenho, quando a alteração mexe em um caminho medido;
- a política de dependências e licenças, a verificação de nomes e a verificação do grafo de tarefas;
- um agente de revisão independente, nunca o que escreveu a alteração, que a confere contra os critérios de aceite da tarefa e as seções de RFC citadas.

O harness de paridade e os gates de conformidade são construídos primeiro, no M0, antes dos kernels e das famílias de modelos que eles verificam. Os mantenedores também aprovam as classes sensíveis de alteração: chaves, assinatura, admissão e o cliente de atualização, configuração de CI e de release, arquivos de instruções dos agentes, novas famílias de modelos, novas dependências diretas e qualquer alteração que um gate não consiga verificar ou que exija dispensar um gate.

Contribuições externas são bem-vindas, com os mesmos gates. O [CONTRIBUTING.md](CONTRIBUTING.md) trata de issues, pull requests, evidências e trabalho assistido por IA, e o [AGENTS.md](AGENTS.md) é o guia de trabalho para os agentes.

## Segurança

Reporte vulnerabilidades de forma privada, não em uma issue pública. O [SECURITY.md](SECURITY.md) traz o canal de reporte, os prazos-alvo de resposta e um resumo do que o Syntropy vai expor na rede: descoberta na LAN, pareamento, links de convite, sua API e seu canal de atualização.

## Licença e créditos

O Syntropy é licenciado sob [MIT](LICENSE-MIT) ou [Apache-2.0](LICENSE-APACHE), à sua escolha. Contribuições são aceitas nos mesmos termos, com um sign-off do Developer Certificate of Origin em todo commit ([ADR-0018](plan/decisions/ADR-0018-licence.md)).

A camada de kernels do Syntropy se apoia no [ggml](https://github.com/ggml-org/ggml), a biblioteca de tensores sob licença MIT, que oferece ampla cobertura de backends e o fallback para toda operação sem kernel nativo. Cada versão vai incluir um arquivo `THIRD-PARTY-NOTICES` gerado, com a lista de todos os componentes que ela contém. O Syntropy não distribui pesos de modelos, e cada modelo mantém sua própria licença.

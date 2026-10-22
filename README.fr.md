<!-- Translation of README.md. Update it in the same change as README.md. -->
<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/media/logo-dark.svg">
    <img src="docs/media/logo-light.svg" alt="Syntropy" width="360">
  </picture>
</p>

<p align="center"><a href="README.md">English</a>&emsp;<a href="README.zh-CN.md">简体中文</a>&emsp;<a href="README.ja.md">日本語</a>&emsp;<a href="README.de.md">Deutsch</a>&emsp;<b>Français</b>&emsp;<a href="README.es.md">Español</a>&emsp;<a href="README.pt-BR.md">Português</a></p>

<p align="center"><sub>Ceci est une traduction. La version de référence est le <a href="README.md">README en anglais</a>.</sub></p>

<h3 align="center">Faites tourner un seul grand modèle sur tous vos GPU, niveaux de mémoire et machines</h3>

<p align="center">Un placement décidé en mesurant votre matériel. Une sortie conforme à la référence par défaut.<br>
Écrit en Rust, avec des kernels natifs pour CUDA, HIP, Metal, Vulkan et le CPU.</p>

<p align="center">
  <a href="#feuille-de-route"><img src="https://img.shields.io/badge/status-pre--alpha%2C_M0a_in_progress-7b5fd0" alt="Statut : pré-alpha, M0a en cours"></a>
  <a href="#licence-et-crédits"><img src="https://img.shields.io/badge/licence-MIT_OR_Apache--2.0-3f6fd1" alt="Licence : MIT OR Apache-2.0"></a>
  <a href="Cargo.toml"><img src="https://img.shields.io/badge/Rust-2024_edition-c2572b?logo=rust&logoColor=white" alt="Rust, édition 2024"></a>
  <a href="PLATFORMS.md"><img src="https://img.shields.io/badge/platforms-Linux%2C_Windows%2C_macOS-2f9a5a" alt="Plateformes : Linux, Windows, macOS"></a>
</p>

<p align="center">
  <a href="#comment-ça-marche-"><b>Comment&nbsp;ça&nbsp;marche</b></a>&emsp;
  <a href="#feuille-de-route"><b>Feuille&nbsp;de&nbsp;route</b></a>&emsp;
  <a href="PLATFORMS.md"><b>Plateformes</b></a>&emsp;
  <a href="plan/README.md"><b>La&nbsp;conception</b></a>&emsp;
  <a href="CONTRIBUTING.md"><b>Contribuer</b></a>
</p>

> [!IMPORTANT]
> **Syntropy est en cours de construction, et il n'y a encore rien à installer.** La conception complète se trouve dans [`plan/`](plan/README.md), et l'implémentation est en cours, au jalon M0a. Tout ce que décrit cette page est conçu et prévu, et rien ne peut encore être exécuté aujourd'hui.

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/media/memory-pool-dark.svg">
    <img src="docs/media/memory-pool-light.svg" width="860" alt="Trois machines dessinées à la même échelle. Un PC de jeu avec 24 Go de VRAM et 64 Go de RAM dispose de 88 Go, un portable de 48 Go de mémoire unifiée et un mini PC de 128 Go. Aucune ne suffit pour un modèle de 142 Go. Ensemble, elles totalisent 264 Go, ce qui contient le modèle avec 122 Go de marge.">
  </picture>
</p>

Un PC de jeu équipé d'une carte graphique de 24 Go et de 64 Go de RAM, un portable doté de 48 Go de mémoire unifiée et un mini PC de 128 Go sous la télé : aucun ne peut contenir à lui seul un modèle MoE (mélange d'experts) de 235 milliards de paramètres, qui pèse 142 Go en 4 bits. À eux trois, ils disposent de 264 Go.

Syntropy les traite comme un seul ordinateur doté de plusieurs types de mémoire. Il mesure à quelle vitesse chaque GPU et chaque CPU lit chaque niveau de mémoire, choisit un placement unique pour tout le modèle et garde le calcul au plus près des octets : le processeur dont la mémoire contient un poids est celui qui calcule avec ce poids. Entre machines, seules les activations circulent pendant le décodage. Au sein d'une machine, un poids absent du cache du GPU n'est pas copié via PCIe à chaque token : le CPU calcule avec lui, là où il réside.

## Fonctionnalités

Tout ce qui suit est spécifié dans la conception, et rien n'est encore implémenté.

### Au sein d'une machine

- **Chaque niveau de mémoire est mis à contribution.** Les poids et les experts MoE les plus sollicités résident dans un cache VRAM adaptatif. L'ensemble des poids reste résident en RAM, où le CPU traite sur place les défauts de cache. Les experts froids et les grandes tables sont lus depuis le NVMe en E/S directes, au moment où on en a besoin.
- **Des experts chargés avant d'être demandés.** La prédiction des experts de la couche suivante s'exécute dans le kernel du routeur : les lectures NVMe et RAM de la couche suivante sont déjà en cours pendant le calcul de la couche courante.
- **Mesuré, pas recopié d'une fiche technique.** Chaque machine sonde ce que ses GPU, ses CPU, ses disques et ses liens réseau délivrent réellement, avec les vrais kernels de Syntropy. Le placement est construit à partir de ces mesures, pas des fiches techniques.

### Entre machines

- **Un placement unique pour tout le modèle.** Un seul plan décide quelles machines participent, quelle plage de couches chacune exécute, quel niveau contient chaque groupe de tenseurs et quel processeur le calcule. Il consigne pourquoi chaque groupe a été placé là où il est.
- **Un anneau de machines.** Le modèle est découpé en étages de pipeline, qui transmettent directement leurs états cachés à la machine suivante. Une même couche n'est répartie sur plusieurs machines que sur des liens dont la latence mesurée respecte un seuil, comme le RDMA sur Thunderbolt 5.
- **Aucune configuration sur le réseau local, un jeton suffit au-dehors.** Les machines d'un même réseau se découvrent mutuellement. Une nouvelle machine est admise une fois que vous avez confirmé le même code à six chiffres sur les deux écrans. Un portable hors de chez vous rejoint le cluster avec un jeton d'invitation, à usage unique par défaut.
- **Perdre une machine ne fait pas perdre la réponse.** Chaque étage conserve ce qu'il a envoyé récemment. Quand la machine d'un étage décroche et que les autres ont la place nécessaire, la génération reprend sans qu'aucune erreur n'atteigne le client.

### Pour vous et vos outils

- **Exact, sauf si vous en décidez autrement.** Les raccourcis qui modifient la sortie, comme sauter les experts absents du cache, s'activent explicitement, portent un nom et sont listés dans chaque réponse qu'ils ont affectée.
- **Un décodage spéculatif qui préserve la réponse.** Des têtes MTP ou un modèle brouillon proposent plusieurs tokens, le modèle les vérifie tous en une seule passe, et la règle d'acceptation préserve la distribution de sortie propre au modèle.
- **Les API que vos outils utilisent déjà.** Des API au format OpenAI et Anthropic, plus une API native, servies par la machine que la requête atteint, quelle qu'elle soit.
- **Du code natif de bout en bout.** Une couche système en Rust, des kernels natifs pour CUDA, HIP, Metal, Vulkan et le CPU, et ggml comme solution de repli, opération par opération, pour élargir la couverture.

## Premier aperçu

Voici la ligne de commande prévue par la [RFC-0051](plan/rfcs/RFC-0051-cli-and-dashboard.md). Aucune de ces commandes n'existe encore.

```sh
syntropy run                     # pose une question, puis recommande un modèle qui tient sur votre matériel
syntropy plan <model> --explain  # prévisualise le placement et la vitesse prévue, sans charger le modèle
syntropy cluster approve <id>    # admet une machine du réseau local après vérification de son code à six chiffres
syntropy invite create           # un jeton à usage unique pour un portable hors de chez vous
```

Par défaut, chaque machine sert les API sur `http://127.0.0.1:7433`, si bien que les clients et SDK OpenAI et Anthropic existants fonctionnent sans modification ([RFC-0050](plan/rfcs/RFC-0050-api-surface.md)) :

```sh
curl http://127.0.0.1:7433/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model": "<model>", "stream": true,
       "messages": [{"role": "user", "content": "Pourquoi le décodage est-il limité par la mémoire ?"}]}'
```

## Feuille de route

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/media/milestones-dark.svg">
    <img src="docs/media/milestones-light.svg" width="860" alt="Sept jalons, de M0a à M4. M0a, le squelette fonctionnel sur une machine Linux, est en cours. M1a est la première préversion, M1b ajoute Windows et Apple silicon, M2 est le lancement, avec des machines sur un réseau local, M3 marque la complétude fonctionnelle, et M4 est la 1.0.">
  </picture>
</p>

La conception est terminée : 25 RFC, 98 ADR et 1 411 tâches réparties sur six domaines du code. Chaque tâche a des critères d'acceptation testables et indique les sections de RFC et les ADR qu'elle implémente. Les jalons ne sont pas datés. La [RFC-0001 §16](plan/rfcs/RFC-0001-system-architecture.md#16-milestones) définit le périmètre et les critères de sortie de chacun, l'[ADR-0020](plan/decisions/ADR-0020-platform-support-tiers.md) les plateformes que chacun livre, et la [feuille de route](plan/implementation/roadmap.md) liste les tâches de chaque jalon dans l'ordre des dépendances. L'avancement est suivi sur le [tableau du projet](https://github.com/orgs/syntropyai-hq/projects/1).

<details>
<summary><b>Ce que livre chaque jalon</b></summary>
<br>

| Jalon | Ce qu'il livre |
|---|---|
| **M0a** | Un squelette fonctionnel (walking skeleton) sur une machine Linux : une famille de modèles denses (Llama) en Q4_K et Q8_0 sur le CPU et un GPU NVIDIA, du chat en streaming au format OpenAI via le vrai runner, avec la conformité, le harnais de parité et la vérification par rapport à la référence de l'éditeur du modèle comme contrôles bloquants pour le merge dès le premier commit. Construit et testé, mais non publié. |
| **M0b** | Le reste de M0 sur la même machine : une famille MoE (Qwen-MoE, tous les experts en VRAM), des tolérances calibrées, une sélection de kernels certifiée, les premières sondes matérielles et un harnais de benchmark lié aux commits. Non publié. |
| **M1a** | Le tiering sur une machine Linux avec un GPU NVIDIA ou le CPU seul : niveaux VRAM, RAM et NVMe, cache d'experts avec prédiction et préchargement, placement sur une seule machine, pagination du KV avec cache de préfixes et sessions sauvegardées, et décodage spéculatif. La première version publique, étiquetée préversion. |
| **M1b** | Windows avec NVIDIA et macOS sur Apple silicon, ainsi que le CPU seul sur les deux. Son périmètre découle de ce que M1a mesure. |
| **M2** | Plusieurs machines sur un réseau local : découverte et appairage, transports chiffrés, pipeline en anneau, distribution des poids en pair-à-pair et premier tableau de bord, plus une préversion Vulkan basée sur ggml pour les GPU AMD et Intel. Cette version marque le lancement. |
| **M3** | Fonctionnellement complet : liens d'invitation, reprise après la perte d'une machine, parallélisme de tenseurs et d'experts sur les liens rapides, spéculation entre machines, kernels HIP et Vulkan natifs, et un mode débit pour plusieurs utilisateurs. |
| **M4** | 1.0 : plateformes Tier 2 et Tier 3, installeurs et mises à jour signés, et contrôles de régression de performance sur du matériel de référence. |

</details>

## Sur quoi Syntropy tournera

Chaque plateforme a un niveau de support, ou tier ([ADR-0020](plan/decisions/ADR-0020-platform-support-tiers.md)), qui décrit son support à la 1.0. [PLATFORMS.md](PLATFORMS.md) contient la matrice complète, avec les versions minimales, les backends et le jalon où chaque plateforme sera livrée pour la première fois.

<img src="https://img.shields.io/badge/tier_1-supported-2f9a5a" alt="Tier 1 : pris en charge"><br>Linux x86_64 avec NVIDIA, Windows 11 x86_64 avec NVIDIA, macOS 15+ sur Apple silicon, et le CPU seul sur les trois. Builds précompilés et signés. La parité et les performances sont testées sur du vrai matériel, et un échec bloque la publication.

<img src="https://img.shields.io/badge/tier_2-known_gaps-3f6fd1" alt="Tier 2 : pris en charge, avec des lacunes connues"><br>Linux avec AMD ROCm ; Vulkan sur les GPU AMD, Intel et NVIDIA ; AMD Strix Halo ; Linux arm64 de classe GB10 ; cartes de classe Raspberry Pi 5 (CPU) ; cartes réseau RDMA ; plusieurs GPU dans une même machine. Builds précompilés, parité exigée, performances suivies. Les lacunes sont listées dans les notes de version.

<img src="https://img.shields.io/badge/tier_3-best_effort-818b98" alt="Tier 3 : au mieux (best effort)"><br>Android et iOS en tant que nœuds ; Intel SYCL ; CUDA 12 pour les GPU Pascal et Volta ; WebGPU ; Windows 10 22H2 ; Linux sous WSL2 avec un GPU NVIDIA. Builds publiés comme expérimentaux.

Non prévus pour la 1.0 : macOS sur Intel, les systèmes d'exploitation 32 bits, Windows sur arm64 et un nœud qui s'exécute dans un navigateur.

## Comment ça marche ?

Générer un token pour un utilisateur demande de lire une fois chaque poids actif et la totalité du cache KV. Le nombre d'opérations par octet est faible, si bien que la vitesse de décodage dépend de l'endroit où se trouvent les octets et de la vitesse à laquelle le processeur voisin peut les lire. Syntropy est conçu autour de ce constat : le calcul va vers les octets, et les octets sont placés d'après des mesures.

### Au sein d'une machine

<p align="center"><img src="docs/media/tiers.svg" width="760" alt="Une machine : la VRAM contient les poids chauds et un cache d'experts, et le GPU la lit ; la RAM contient tous les poids, et le CPU y calcule les défauts de cache ; le NVMe contient les experts froids et les tables, qui sont préchargés en RAM"></p>

- La **VRAM** contient ce que chaque token sollicite (attention, couches denses, experts partagés, cache KV), ainsi qu'un cache des experts MoE les plus utilisés, qui s'adapte à la charge de travail.
- La **RAM** contient l'ensemble des poids. Quand un token a besoin d'un expert absent du cache du GPU, le CPU le calcule sur place, en parallèle du GPU, au lieu de le copier via PCIe. Un CPU de bureau lit la DDR5 double canal à 76–122 Go/s, soit environ 3 à 5 fois ce que le PCIe 4.0 x16 fournit au GPU.
- Le **NVMe** contient les experts froids et les grandes tables de correspondance, stockés sous forme d'extents contenant chacun un expert entier, et lus en E/S directes. La prédiction de la couche suivante lance ces lectures avant que le routeur ne les demande.

Le gestionnaire de niveaux de Syntropy contrôle chaque octet résident, si bien que le cache de pages du système d'exploitation ne décide jamais de ce qui reste en mémoire. Les poids ne sont envoyés en flux vers le GPU que pour les gros lots, comme les longs segments de prompt, et seulement là où le seuil de rentabilité mesuré montre que copier est plus avantageux que calculer sur place.

### Entre machines

<p align="center"><img src="docs/media/ring.svg" width="760" alt="Trois machines en anneau, chacune exécutant une plage de couches : les états cachés passent de l'une à la suivante, la dernière échantillonne le token, le renvoie à la première et le transmet en streaming à la machine qui a reçu la requête"></p>

Le modèle est découpé en étages par plages de couches, un étage par machine, et les étages forment un anneau. Le premier étage calcule l'embedding de l'entrée, et chaque étage transmet directement son état caché au suivant. Le dernier étage échantillonne le token, le renvoie le long de l'anneau et le transmet en streaming à la machine qui a reçu la requête. Un pipeline ajoute de la capacité plutôt que de la vitesse sur un flux unique : le gain vient donc du déplacement des octets vers une mémoire plus rapide. Une machine lente est laissée hors de l'anneau lorsqu'elle n'ajouterait qu'un saut.

Répartir une même couche sur plusieurs machines, par parallélisme de tenseurs ou d'experts, coûte un aller-retour réseau par couche et par token. Syntropy ne le fait que sur les liens dont la latence mesurée des opérations collectives est assez faible pour que ce soit rentable. En pratique, cela veut dire des liens de classe RDMA. Le Wi-Fi n'est jamais retenu.

Une machine est élue coordinateur et tient le journal d'événements du cluster, qui consigne sa composition et ses plans. Elle ne voit jamais passer un token et ne se trouve pas sur le chemin des requêtes : sa perte ne fait donc échouer aucune requête.

> [!TIP]
> L'explication plus longue, en langage courant, se trouve dans [docs/HOW_IT_WORKS.md](docs/HOW_IT_WORKS.md). La conception complète est dans les RFC : niveaux et résidence ([RFC-0020](plan/rfcs/RFC-0020-tier-manager-and-residency.md)), chemin d'E/S des poids ([RFC-0021](plan/rfcs/RFC-0021-weight-io-path.md)), cache d'experts ([RFC-0023](plan/rfcs/RFC-0023-moe-expert-cache-prediction-prefetch.md)), placement ([RFC-0011](plan/rfcs/RFC-0011-joint-placement-solver.md)), exécution distribuée ([RFC-0032](plan/rfcs/RFC-0032-distributed-execution.md)), ainsi que découverte et invitations ([RFC-0030](plan/rfcs/RFC-0030-discovery-membership-invites.md)).

## Exact par défaut

Avec les réglages par défaut, la sortie de Syntropy doit correspondre à une passe avant de référence du même modèle, à la même quantification, dans les tolérances publiées ([ADR-0019](plan/decisions/ADR-0019-exactness-contract.md)).

- **La référence** est l'exécuteur de référence de Syntropy : accumulation en F32, ordre de réduction fixe et poids déquantifiés exactement à partir du même fichier. Il est lui-même vérifié par rapport à l'implémentation de référence de l'éditeur du modèle, si bien que « exact » couvre la lecture du modèle par Syntropy autant que ses kernels.
- **Le test** mesure l'accord top-1 en teacher forcing et la divergence KL sur un corpus versionné de conversations, de code, de contextes longs, de JSON d'appels d'outils et de textes multilingues. Les tolérances sont fixées par classe de dtype à partir de mesures, versionnées, puis inscrites dans chaque version publiée.
- **Ce qui doit passer**, c'est chaque chemin que le planificateur peut choisir en mode exact : chaque kernel et chaque solution de repli, chaque forme de placement, d'un seul GPU à un anneau de machines, le décodage spéculatif comme le décodage classique, la pagination du KV entre niveaux, et les sessions reprises. Un chemin qui échoue est approximatif, quelle qu'ait été l'intention.
- **Les raccourcis s'activent explicitement et portent un nom**, comme `expert_skip` (les experts absents du cache ne sont pas calculés) ou `kv_lossy` (le KV est stocké avec une précision inférieure à celle du modèle). Chaque réponse indique sa fidélité et les raccourcis qui étaient actifs, et une requête qui demande `exact` n'est jamais servie par une configuration approximative.

Une sortie identique à l'octet près sur des processeurs différents n'est pas garantie, car des matériels différents arrondissent différemment, à l'intérieur de la tolérance. Un mode `deterministic`, plus lent, est prévu pour obtenir une sortie reproductible avec le même build et les mêmes processeurs. Les tolérances et le harnais de parité sont spécifiés dans la [RFC-0042](plan/rfcs/RFC-0042-quantization-and-numerical-fidelity.md) et la [RFC-0060](plan/rfcs/RFC-0060-testing-and-correctness.md).

## La conception

[`plan/README.md`](plan/README.md) sert d'index et propose un ordre de lecture.

| Chemin | Contenu |
|---|---|
| [`plan/00-vision-and-scope.md`](plan/00-vision-and-scope.md) | Le problème, les scénarios, la matrice des plateformes, les modèles visés, les critères de succès SC-1 à SC-32 (tous des cibles) et ce que signifie « fonctionnellement complet » |
| [`plan/decisions/`](plan/decisions/) | Les ADR. ADR-0000 contient les directives produit D1–D8, ADR-0001 à ADR-0022 les fondations, et le reste est numéroté par domaine |
| [`plan/rfcs/`](plan/rfcs/) | RFC-0001, l'architecture du système, et 24 RFC de domaine : placement, niveaux de mémoire, exécution distribuée, calcul, produit et qualité |
| [`plan/implementation/`](plan/implementation/README.md) | Les tâches, dans six fichiers d'epics, la feuille de route, le rapport de dépendances, la traçabilité, le registre des risques et les scripts qui les vérifient |
| [`plan/open-questions.md`](plan/open-questions.md) | Ce qui reste à décider, chaque point avec la tâche ou la mesure qui le tranchera |
| [`plan/glossary.md`](plan/glossary.md) | Un terme par concept, et les synonymes que le plan n'utilise pas |
| [`plan/research/`](plan/research/) | Les éléments de preuve : des notes sur l'inférence répartie sur plusieurs processeurs et sur une mémoire hiérarchisée, sur les backends, et un modèle de coût matériel avec des chiffres de bande passante sourcés |

Chaque chiffre du plan est étiqueté comme `[target]` (une cible), comme `[estimate]` (une estimation, avec son calcul) ou comme une mesure matérielle sourcée. Aucun ne provient d'un benchmark de Syntropy, car il n'y a encore rien à mesurer.

## Contribuer

Des agents d'IA construisent Syntropy en exécutant les tâches du plan. Un agent prend une tâche, l'implémente sur sa propre branche et propose une seule modification, qui satisfait les critères d'acceptation de la tâche et cite les sections de RFC et les ADR qu'elle implémente. Le graphe des tâches détermine ce qui peut démarrer.

Une modification n'est fusionnée que si tous ces contrôles passent ([ADR-0021](plan/decisions/ADR-0021-development-model-ai-agents-open-source.md)) :

- build, lint, formatage, et tests unitaires, de propriétés et sur simulateur ;
- le harnais de parité et les contrôles de conformité ;
- les contrôles de régression de performance, lorsque la modification touche un chemin mesuré ;
- la politique de dépendances et de licences, la vérification du nommage et celle du graphe des tâches ;
- un agent de revue indépendant, jamais celui qui a écrit la modification, qui la vérifie au regard des critères d'acceptation de la tâche et des sections de RFC citées.

Le harnais de parité et les contrôles de conformité sont construits en premier, dès M0, avant les kernels et les familles de modèles qu'ils vérifient. Les mainteneurs approuvent en outre les catégories de modifications sensibles : clés, signature, admission et client de mise à jour, configuration de la CI et des releases, fichiers d'instructions des agents, nouvelles familles de modèles, nouvelles dépendances directes, et toute modification qu'aucun contrôle ne peut vérifier ou qui exige une dérogation à un contrôle.

Les contributions extérieures sont les bienvenues, avec les mêmes contrôles. [CONTRIBUTING.md](CONTRIBUTING.md) traite des issues, des pull requests, des éléments de preuve et du travail assisté par IA, et [AGENTS.md](AGENTS.md) est le guide de travail des agents.

## Sécurité

Merci de signaler les vulnérabilités en privé, et non dans une issue publique. [SECURITY.md](SECURITY.md) indique la procédure de signalement, les délais de réponse visés et un résumé de ce que Syntropy exposera sur un réseau : la découverte sur le réseau local, l'appairage, les liens d'invitation, son API et son canal de mise à jour.

## Licence et crédits

Syntropy est distribué sous licence [MIT](LICENSE-MIT) ou [Apache-2.0](LICENSE-APACHE), à votre choix. Les contributions sont acceptées aux mêmes conditions, avec un sign-off Developer Certificate of Origin sur chaque commit ([ADR-0018](plan/decisions/ADR-0018-licence.md)).

La couche de kernels de Syntropy s'appuie sur [ggml](https://github.com/ggml-org/ggml), la bibliothèque de tenseurs sous licence MIT, qui apporte une large couverture de backends et la solution de repli pour toute opération dépourvue de kernel natif. Chaque version publiée inclura un fichier `THIRD-PARTY-NOTICES` généré, qui liste tous les composants qu'elle contient. Syntropy ne distribue aucun poids de modèle, et chaque modèle conserve sa propre licence.

<!-- Translation of README.md. Update it in the same change as README.md. -->
<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/media/logo-dark.svg">
    <img src="docs/media/logo-light.svg" alt="Syntropy" width="360">
  </picture>
</p>

<p align="center"><a href="README.md">English</a>&emsp;<a href="README.zh-CN.md">简体中文</a>&emsp;<a href="README.ja.md">日本語</a>&emsp;<b>Deutsch</b>&emsp;<a href="README.fr.md">Français</a>&emsp;<a href="README.es.md">Español</a>&emsp;<a href="README.pt-BR.md">Português</a></p>

<p align="center"><sub>Dies ist eine Übersetzung. Maßgeblich ist die <a href="README.md">englische README</a>.</sub></p>

<h3 align="center">Ein großes Modell über alle deine GPUs, Speicherebenen und Rechner hinweg ausführen</h3>

<p align="center">Die Platzierung ergibt sich aus Messungen deiner Hardware. Die Ausgabe stimmt standardmäßig mit der Referenz überein.<br>
Rust, mit nativen Kernels für CUDA, HIP, Metal, Vulkan und die CPU.</p>

<p align="center">
  <a href="#roadmap"><img src="https://img.shields.io/badge/status-pre--alpha%2C_M0a_in_progress-7b5fd0" alt="Status: Pre-Alpha, M0a in Arbeit"></a>
  <a href="#lizenz-und-danksagungen"><img src="https://img.shields.io/badge/licence-MIT_OR_Apache--2.0-3f6fd1" alt="Lizenz: MIT OR Apache-2.0"></a>
  <a href="Cargo.toml"><img src="https://img.shields.io/badge/Rust-2024_edition-c2572b?logo=rust&logoColor=white" alt="Rust Edition 2024"></a>
  <a href="PLATFORMS.md"><img src="https://img.shields.io/badge/platforms-Linux%2C_Windows%2C_macOS-2f9a5a" alt="Plattformen: Linux, Windows, macOS"></a>
</p>

<p align="center">
  <a href="#wie-funktioniert-es"><b>So&nbsp;funktioniert&nbsp;es</b></a>&emsp;
  <a href="#roadmap"><b>Roadmap</b></a>&emsp;
  <a href="PLATFORMS.md"><b>Plattformen</b></a>&emsp;
  <a href="plan/README.md"><b>Das&nbsp;Design</b></a>&emsp;
  <a href="CONTRIBUTING.md"><b>Mitwirken</b></a>
</p>

> [!IMPORTANT]
> **Syntropy befindet sich im Aufbau, und es gibt noch nichts zu installieren.** Das vollständige Design liegt in [`plan/`](plan/README.md), die Implementierung ist mit Meilenstein M0a angelaufen. Alles auf dieser Seite beschreibt, was entworfen und geplant ist, nicht etwas, das du heute ausführen kannst.

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/media/memory-pool-dark.svg">
    <img src="docs/media/memory-pool-light.svg" width="860" alt="Drei Rechner im selben Maßstab. Ein Gaming-PC mit 24 GB VRAM und 64 GB RAM hat 88 GB, ein Laptop hat 48 GB Unified Memory und ein Mini-PC 128 GB. Keiner reicht allein für ein Modell mit 142 GB. Zusammen haben sie 264 GB, die das Modell mit 122 GB Reserve aufnehmen.">
  </picture>
</p>

Ein Gaming-PC mit einer 24-GB-Grafikkarte und 64 GB RAM, ein Laptop mit 48 GB Unified Memory und ein Mini-PC mit 128 GB unter dem Fernseher können jeder für sich kein Mixture-of-Experts-Modell mit 235 Milliarden Parametern halten, das bei 4 Bit 142 GB groß ist. Zusammen haben sie 264 GB.

Syntropy behandelt sie als einen Computer mit mehreren Arten von Speicher. Es misst, wie schnell jede GPU und jede CPU aus jeder Speicherebene liest, wählt eine Platzierung für das ganze Modell und hält die Berechnung bei den Bytes: Das Gerät, in dessen Speicher ein Gewicht liegt, rechnet auch damit. Zwischen Rechnern wandern beim Decoding nur Aktivierungen. Innerhalb eines Rechners berechnet die CPU ein Gewicht, das nicht im Cache der GPU liegt, dort, wo es liegt, statt es für jedes Token über PCIe zu kopieren.

## Funktionen

All das ist im Design spezifiziert, und nichts davon ist bisher gebaut.

### Innerhalb eines Rechners

- **Jede Speicherebene arbeitet mit.** Die heißesten Gewichte und MoE-Experten liegen in einem adaptiven VRAM-Cache. Der vollständige Satz an Gewichten bleibt resident im RAM, wo die CPU Cache-Misses direkt vor Ort berechnet. Kalte Experten und große Tabellen werden bei Bedarf per Direct I/O von der NVMe gelesen.
- **Experten werden geholt, bevor sie gebraucht werden.** Die Vorhersage der Experten der nächsten Schicht läuft im Router-Kernel. So laufen die NVMe- und RAM-Lesezugriffe für die nächste Schicht schon, während die aktuelle rechnet.
- **Gemessen, nicht nachgeschlagen.** Jeder Rechner ermittelt mit den echten Kernels von Syntropy, was seine GPUs, CPUs, Laufwerke und Netzwerkverbindungen tatsächlich leisten. Die Platzierung baut auf diesen Messungen auf, nicht auf Datenblättern.

### Über mehrere Rechner hinweg

- **Eine Platzierung für das ganze Modell.** Ein einziger Plan legt fest, welche Rechner mitmachen, welchen Schichtbereich jeder ausführt, welche Speicherebene jede Tensorgruppe hält und welches Gerät sie berechnet. Er hält auch fest, warum jede Gruppe dort gelandet ist, wo sie ist.
- **Ein Ring aus Rechnern.** Das Modell wird in Pipeline-Stufen aufgeteilt, die ihre Hidden States direkt an den nächsten Rechner übergeben. Eine einzelne Schicht wird nur auf Verbindungen wie RDMA über Thunderbolt 5 auf mehrere Rechner verteilt, und nur, wenn deren gemessene Latenz eine Schwelle unterschreitet.
- **Im LAN ohne Einrichtung, außerhalb mit einem Token.** Rechner im selben Netzwerk finden sich gegenseitig. Ein neuer Rechner wird aufgenommen, sobald du auf beiden Bildschirmen denselben sechsstelligen Code bestätigst. Ein Laptop unterwegs tritt mit einem Einladungstoken bei, das standardmäßig nur einmal gilt.
- **Fällt ein Rechner aus, geht die Antwort nicht verloren.** Jede Stufe behält, was sie zuletzt gesendet hat. Fällt der Rechner einer Stufe weg und haben die anderen genug Platz, wird die Generierung fortgesetzt, ohne dass ein Fehler beim Client ankommt.

### Für dich und deine Tools

- **Exakt, solange du nichts anderes sagst.** Shortcuts, die die Ausgabe verändern, etwa das Überspringen von Experten mit Cache-Miss, sind Opt-in, haben einen Namen und werden in jeder Antwort aufgeführt, auf die sie gewirkt haben.
- **Spekulatives Decoding, das die Antwort nicht verändert.** MTP-Heads oder ein Draft-Modell schlagen mehrere Tokens vor, das Modell prüft sie alle in einem Durchlauf, und die Akzeptanzregel erhält die Ausgabeverteilung des Modells selbst.
- **Die APIs, die deine Tools schon nutzen.** APIs im Stil von OpenAI und Anthropic sowie eine native API, bedient von dem Rechner, bei dem die Anfrage ankommt.
- **Durchgehend nativer Code.** Eine Systemschicht in Rust, native Kernels für CUDA, HIP, Metal, Vulkan und die CPU, und ggml als Fallback pro Operation für eine breite Abdeckung.

## Ein erster Blick

Das ist die in [RFC-0051](plan/rfcs/RFC-0051-cli-and-dashboard.md) geplante Kommandozeile. Keiner dieser Befehle existiert bisher.

```sh
syntropy run                     # stellt eine Frage und empfiehlt dann ein passendes Modell
syntropy plan <model> --explain  # zeigt Platzierung und vorhergesagte Geschwindigkeit, ohne zu laden
syntropy cluster approve <id>    # nimmt einen LAN-Rechner auf, wenn sein sechsstelliger Code übereinstimmt
syntropy invite create           # ein Einmal-Token für einen Laptop unterwegs
```

Jeder Rechner wird die APIs standardmäßig unter `http://127.0.0.1:7433` bereitstellen, sodass vorhandene OpenAI- und Anthropic-Clients und -SDKs unverändert funktionieren ([RFC-0050](plan/rfcs/RFC-0050-api-surface.md)):

```sh
curl http://127.0.0.1:7433/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model": "<model>", "stream": true,
       "messages": [{"role": "user", "content": "Warum ist das Decoding speichergebunden?"}]}'
```

## Roadmap

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/media/milestones-dark.svg">
    <img src="docs/media/milestones-light.svg" width="860" alt="Sieben Meilensteine von M0a bis M4. M0a, das Walking Skeleton auf einem Linux-Rechner, ist in Arbeit. M1a ist das erste Preview-Release, M1b ergänzt Windows und Apple Silicon, M2 ist der Launch mit Rechnern im LAN, M3 ist funktionskomplett, und M4 ist 1.0.">
  </picture>
</p>

Das Design ist fertig: 25 RFCs, 98 ADRs und 1.411 Aufgaben in sechs Bereichen des Codes, jede mit testbaren Akzeptanzkriterien und den RFC-Abschnitten und ADRs, die sie umsetzt. Die Meilensteine haben keine Termine. [RFC-0001 §16](plan/rfcs/RFC-0001-system-architecture.md#16-milestones) legt Umfang und Abschlusskriterien jedes Meilensteins fest, [ADR-0020](plan/decisions/ADR-0020-platform-support-tiers.md) die Plattformen, die jeder veröffentlicht, und die [Roadmap](plan/implementation/roadmap.md) listet die Aufgaben jedes Meilensteins in Abhängigkeitsreihenfolge auf. Der Fortschritt wird auf dem [Projekt-Board](https://github.com/orgs/syntropyai-hq/projects/1) verfolgt.

<details>
<summary><b>Was jeder Meilenstein liefert</b></summary>
<br>

| Meilenstein | Was er liefert |
|---|---|
| **M0a** | Ein Walking Skeleton auf einem Linux-Rechner: eine dichte Modellfamilie (Llama) in Q4_K und Q8_0 auf der CPU und einer NVIDIA-GPU, gestreamter Chat im OpenAI-Stil über den echten Runner, und als Merge-Gates ab dem ersten Commit die Konformitätsprüfung, das Parity-Harness und der Abgleich mit der Referenz des Modellherausgebers. Gebaut und getestet, nicht veröffentlicht. |
| **M0b** | Der Rest von M0 auf demselben Rechner: eine MoE-Familie (Qwen-MoE, alle Experten im VRAM), kalibrierte Toleranzen, zertifizierte Kernel-Auswahl, die ersten Hardware-Probes und ein an Commits gebundenes Benchmark-Harness. Nicht veröffentlicht. |
| **M1a** | Tiering auf einem Linux-Rechner mit NVIDIA-GPU oder nur CPU: die Speicherebenen VRAM, RAM und NVMe, der Experten-Cache mit Vorhersage und Prefetch, Platzierung auf einem Rechner, KV-Paging mit Präfix-Cache und gespeicherten Sitzungen sowie spekulatives Decoding. Das erste öffentliche Release, als Preview gekennzeichnet. |
| **M1b** | Windows mit NVIDIA und macOS auf Apple Silicon, dazu auf beiden reiner CPU-Betrieb. Der Umfang richtet sich danach, was M1a misst. |
| **M2** | Mehrere Rechner im LAN: Discovery und Pairing, verschlüsselte Transportwege, die Ring-Pipeline, Peer-to-Peer-Verteilung der Gewichte und das erste Dashboard, dazu eine Vulkan-Preview auf Basis von ggml für GPUs von AMD und Intel. Dieses Release ist der Launch. |
| **M3** | Funktionskomplett: Einladungslinks, Wiederherstellung nach dem Ausfall eines Rechners, Tensor- und Expertenparallelität auf schnellen Verbindungen, Spekulation über Rechner hinweg, native HIP- und Vulkan-Kernels und ein Durchsatzmodus für mehrere Nutzer. |
| **M4** | 1.0: Plattformen in Tier 2 und Tier 3, signierte Installer und Updates sowie Gates für Performance-Regressionen auf Referenzhardware. |

</details>

## Worauf Syntropy laufen wird

Jede Plattform ist einem Support-Tier zugeordnet ([ADR-0020](plan/decisions/ADR-0020-platform-support-tiers.md)), das beschreibt, wie sie in Version 1.0 unterstützt wird. [PLATFORMS.md](PLATFORMS.md) enthält die vollständige Matrix mit Mindestversionen, Backends und dem Meilenstein, mit dem jede Plattform zum ersten Mal ausgeliefert wird.

<img src="https://img.shields.io/badge/tier_1-supported-2f9a5a" alt="Tier 1: unterstützt"><br>Linux x86_64 mit NVIDIA, Windows 11 x86_64 mit NVIDIA, macOS 15+ auf Apple Silicon, und auf allen dreien reiner CPU-Betrieb. Vorkompilierte, signierte Builds. Parität und Performance werden auf echter Hardware getestet, und ein Fehlschlag blockiert das Release.

<img src="https://img.shields.io/badge/tier_2-known_gaps-3f6fd1" alt="Tier 2: unterstützt, mit bekannten Lücken"><br>Linux mit AMD ROCm; Vulkan auf GPUs von AMD, Intel und NVIDIA; AMD Strix Halo; Linux arm64 der GB10-Klasse; Boards der Raspberry-Pi-5-Klasse (CPU); RDMA-NICs; mehrere GPUs in einem Rechner. Vorkompilierte Builds, Parität wird erzwungen, Performance wird verfolgt. Die Lücken stehen in den Release Notes.

<img src="https://img.shields.io/badge/tier_3-best_effort-818b98" alt="Tier 3: Best Effort"><br>Android und iOS als Knoten; Intel SYCL; CUDA 12 für GPUs der Pascal- und Volta-Generation; WebGPU; Windows 10 22H2; Linux unter WSL2 mit NVIDIA-GPU. Builds werden als experimentell veröffentlicht.

Nicht für 1.0 geplant: macOS auf Intel, 32-Bit-Betriebssysteme, Windows auf arm64 und ein Knoten, der im Browser läuft.

## Wie funktioniert es?

Um ein Token für einen Nutzer zu erzeugen, wird jedes aktive Gewicht und der gesamte KV-Cache einmal gelesen. Pro Byte fällt wenig Rechenarbeit an, daher hängt die Decode-Geschwindigkeit davon ab, wo die Bytes liegen und wie schnell der Prozessor neben ihnen sie lesen kann. Darauf ist Syntropy ausgelegt: Die Berechnung wandert zu den Bytes, und die Bytes werden nach Messung platziert.

### Innerhalb eines Rechners

<p align="center"><img src="docs/media/tiers.svg" width="760" alt="Ein Rechner: VRAM hält die heißen Gewichte und einen Experten-Cache und wird von der GPU gelesen; RAM hält alle Gewichte, und die CPU berechnet dort Cache-Misses; NVMe hält kalte Experten und Tabellen, die per Prefetch in den RAM geladen werden"></p>

- **VRAM** hält, was jedes Token berührt (Attention, dichte Schichten, Shared Experts, den KV-Cache), und einen Cache der meistgenutzten MoE-Experten, der sich an die Arbeitslast anpasst.
- **RAM** hält den vollständigen Satz an Gewichten. Braucht ein Token einen Experten, der nicht im Cache der GPU liegt, berechnet ihn die CPU dort, parallel zur GPU, statt ihn über PCIe zu kopieren. Eine Desktop-CPU liest Dual-Channel-DDR5 mit 76–122 GB/s, etwa 3–5× so viel, wie PCIe 4.0 x16 der GPU liefert.
- **NVMe** hält kalte Experten und große Lookup-Tabellen, abgelegt in Extents, die jeweils einen ganzen Experten umfassen, und gelesen per Direct I/O. Die Vorhersage für die nächste Schicht startet diese Lesezugriffe, bevor der Router die Experten anfordert.

Der Tier-Manager von Syntropy verwaltet jedes residente Byte, sodass nie der Page Cache des Betriebssystems entscheidet, was im Speicher bleibt. Gewichte werden nur bei großen Batches, etwa langen Prompt-Abschnitten, zur GPU gestreamt, und nur dort, wo der gemessene Break-even zeigt, dass Kopieren sich gegenüber dem Rechnen vor Ort lohnt.

### Über mehrere Rechner hinweg

<p align="center"><img src="docs/media/ring.svg" width="760" alt="Drei Rechner in einem Ring, jeder führt einen Schichtbereich aus: Hidden States wandern von einem zum nächsten, der letzte sampelt das Token, gibt es an den ersten zurück und streamt es an den Rechner, der die Anfrage erhalten hat"></p>

Das Modell wird nach Schichtbereichen in Stufen geschnitten, eine Stufe pro Rechner, und die Stufen bilden einen Ring. Die erste Stufe berechnet die Embeddings der Eingabe, und jede Stufe gibt ihren Hidden State direkt an die nächste weiter. Die letzte Stufe sampelt das Token, schickt es über den Ring zurück an den Anfang und streamt es an den Rechner, der die Anfrage erhalten hat. Eine Pipeline bringt Kapazität, nicht mehr Geschwindigkeit für einen einzelnen Stream. Der Gewinn entsteht also dadurch, dass Bytes in schnelleren Speicher kommen. Ein langsamer Rechner bleibt außerhalb des Rings, wenn er nur einen zusätzlichen Hop bringen würde.

Eine einzelne Schicht mit Tensor- oder Expertenparallelität auf mehrere Rechner aufzuteilen, kostet pro Schicht und Token einen Netzwerk-Roundtrip. Syntropy tut das nur auf Verbindungen, deren gemessene Latenz für kollektive Operationen niedrig genug ist, damit es sich lohnt. In der Praxis heißt das: Verbindungen der RDMA-Klasse. Wi-Fi kommt nie infrage.

Ein Rechner wird zum Koordinator gewählt und führt das Ereignisprotokoll des Clusters mit Mitgliedschaft und Plänen. Er sieht nie ein Token und liegt nicht im Anfragepfad, daher lässt sein Ausfall keine Anfrage scheitern.

> [!TIP]
> Die ausführlichere Erklärung in einfachen Worten steht in [docs/HOW_IT_WORKS.md](docs/HOW_IT_WORKS.md). Die vollständigen Designs stehen in den RFCs: Speicherebenen und Residenz ([RFC-0020](plan/rfcs/RFC-0020-tier-manager-and-residency.md)), der I/O-Pfad für Gewichte ([RFC-0021](plan/rfcs/RFC-0021-weight-io-path.md)), der Experten-Cache ([RFC-0023](plan/rfcs/RFC-0023-moe-expert-cache-prediction-prefetch.md)), Platzierung ([RFC-0011](plan/rfcs/RFC-0011-joint-placement-solver.md)), verteilte Ausführung ([RFC-0032](plan/rfcs/RFC-0032-distributed-execution.md)) sowie Discovery und Einladungen ([RFC-0030](plan/rfcs/RFC-0030-discovery-membership-invites.md)).

## Standardmäßig exakt

Mit den Standardeinstellungen muss die Ausgabe von Syntropy innerhalb veröffentlichter Toleranzen mit einem Referenz-Forward-Pass desselben Modells bei derselben Quantisierung übereinstimmen ([ADR-0019](plan/decisions/ADR-0019-exactness-contract.md)).

- **Die Referenz** ist der Referenz-Executor von Syntropy: F32-Akkumulation, eine feste Reduktionsreihenfolge und Gewichte, die exakt aus derselben Datei dequantisiert werden. Er wird seinerseits gegen die Referenzimplementierung des Modellherausgebers geprüft. „Exakt“ gilt damit für Syntropys Interpretation des Modells ebenso wie für seine Kernels.
- **Der Test** misst Top-1-Übereinstimmung unter Teacher Forcing und KL-Divergenz über einen versionierten Korpus aus Chat, Code, langem Kontext, Tool-Call-JSON und mehrsprachigem Text. Die Toleranzen werden je dtype-Klasse aus Messungen festgelegt, versioniert und in jedem Release festgeschrieben.
- **Bestehen muss** jeder Pfad, den der Planer im exakten Modus wählen kann: jeder Kernel und jeder Fallback, jede Platzierungsform von einer GPU bis zu einem Ring aus Rechnern, spekulatives und normales Decoding, KV-Paging zwischen Speicherebenen und wieder aufgenommene Sitzungen. Ein Pfad, der durchfällt, ist approximativ, ganz gleich, wofür er gedacht war.
- **Shortcuts sind Opt-in und haben Namen**, etwa `expert_skip` (Experten mit Cache-Miss werden nicht berechnet) oder `kv_lossy` (KV wird unterhalb der Präzision des Modells gespeichert). Jede Antwort weist ihre Genauigkeitsstufe und die aktiven Shortcuts aus, und eine Anfrage, die `exact` verlangt, wird nie von einer approximativen Konfiguration bedient.

Byte-identische Ausgabe auf verschiedenen Geräten wird nicht versprochen, weil unterschiedliche Hardware innerhalb der Toleranz unterschiedlich rundet. Für reproduzierbare Ausgabe auf demselben Build und denselben Geräten ist ein langsamerer `deterministic`-Modus geplant. Toleranzen und Parity-Harness sind in [RFC-0042](plan/rfcs/RFC-0042-quantization-and-numerical-fidelity.md) und [RFC-0060](plan/rfcs/RFC-0060-testing-and-correctness.md) spezifiziert.

## Das Design

[`plan/README.md`](plan/README.md) ist der Index und schlägt eine Lesereihenfolge vor.

| Pfad | Inhalt |
|---|---|
| [`plan/00-vision-and-scope.md`](plan/00-vision-and-scope.md) | Das Problem, die Szenarien, die Plattformmatrix, Zielmodelle, die Erfolgskriterien SC-1 bis SC-32 (durchweg Zielwerte) und was „funktionskomplett“ bedeutet |
| [`plan/decisions/`](plan/decisions/) | ADRs. ADR-0000 enthält die Produktdirektiven D1–D8, ADR-0001 bis ADR-0022 das Fundament, die übrigen sind nach Bereich nummeriert |
| [`plan/rfcs/`](plan/rfcs/) | RFC-0001, die Systemarchitektur, und 24 Fach-RFCs für Platzierung, Speicherebenen, verteilte Ausführung, Berechnung, Produkt und Qualität |
| [`plan/implementation/`](plan/implementation/README.md) | Die Aufgaben in sechs Epic-Dateien, die Roadmap, der Abhängigkeitsbericht, die Rückverfolgbarkeit, das Risikoregister und die Skripte, die sie prüfen |
| [`plan/open-questions.md`](plan/open-questions.md) | Was noch offen ist, jeder Punkt mit der Aufgabe oder Messung, die ihn klärt |
| [`plan/glossary.md`](plan/glossary.md) | Ein Begriff pro Konzept und die Synonyme, die der Plan nicht verwendet |
| [`plan/research/`](plan/research/) | Die Belege: Notizen zu Inferenz über mehrere Geräte und mit mehrstufigem Speicher, zu Backends und ein Hardware-Kostenmodell mit Bandbreitenwerten samt Quellen |

Jede Zahl im Plan ist gekennzeichnet: als `[target]`, als `[estimate]` mit Herleitung oder als zitierte Hardware-Messung. Keine davon ist ein Benchmark von Syntropy, denn es gibt noch nichts zu benchmarken.

## Mitwirken

KI-Agenten bauen Syntropy, indem sie die Aufgaben des Plans ausführen. Ein Agent übernimmt eine Aufgabe, setzt sie auf einem eigenen Branch um und öffnet eine Änderung, die die Akzeptanzkriterien der Aufgabe erfüllt und die RFC-Abschnitte und ADRs nennt, die sie umsetzt. Der Aufgabengraph entscheidet, was beginnen kann.

Eine Änderung wird nur gemergt, wenn all das besteht ([ADR-0021](plan/decisions/ADR-0021-development-model-ai-agents-open-source.md)):

- Build, Lint, Formatierung sowie Unit-, Property- und Simulatortests;
- das Parity-Harness und die Konformitäts-Gates;
- die Gates für Performance-Regressionen, wenn die Änderung einen gemessenen Pfad berührt;
- die Abhängigkeits- und Lizenzrichtlinie, die Namensprüfung und die Prüfung des Aufgabengraphen;
- die Freigabe durch einen unabhängigen Review-Agenten, nie den, der die Änderung geschrieben hat; er prüft sie gegen die Akzeptanzkriterien der Aufgabe und die zitierten RFC-Abschnitte.

Parity-Harness und Konformitäts-Gates werden zuerst gebaut, in M0, noch vor den Kernels und Modellfamilien, die sie prüfen. Maintainer genehmigen außerdem die sensiblen Klassen von Änderungen: Schlüssel, Signierung, die Aufnahme in den Cluster und den Update-Client, CI- und Release-Konfiguration, Instruktionsdateien für Agenten, neue Modellfamilien, neue direkte Abhängigkeiten und jede Änderung, die kein Gate prüfen kann oder für die ein Gate ausgesetzt werden muss.

Beiträge von außen sind willkommen und durchlaufen dieselben Gates. [CONTRIBUTING.md](CONTRIBUTING.md) behandelt Issues, Pull Requests, Nachweise und KI-gestützte Arbeit, und [AGENTS.md](AGENTS.md) ist der Arbeitsleitfaden für Agenten.

## Sicherheit

Bitte melde Sicherheitslücken vertraulich, nicht in einem öffentlichen Issue. [SECURITY.md](SECURITY.md) beschreibt den Meldeweg und die angestrebten Reaktionszeiten und gibt einen Überblick darüber, was Syntropy im Netzwerk erreichbar machen wird: LAN-Discovery, Pairing, Einladungslinks, seine API und seinen Update-Kanal.

## Lizenz und Danksagungen

Syntropy steht nach deiner Wahl unter [MIT](LICENSE-MIT) oder [Apache-2.0](LICENSE-APACHE). Beiträge werden zu denselben Bedingungen angenommen, mit einem Sign-off nach dem Developer Certificate of Origin in jedem Commit ([ADR-0018](plan/decisions/ADR-0018-licence.md)).

Die Kernel-Schicht von Syntropy baut auf [ggml](https://github.com/ggml-org/ggml) auf, der MIT-lizenzierten Tensor-Bibliothek, die eine breite Backend-Abdeckung und den Fallback für jede Operation ohne nativen Kernel liefert. Jedes Release wird eine generierte Datei `THIRD-PARTY-NOTICES` enthalten, die jede enthaltene Komponente auflistet. Syntropy liefert keine Modellgewichte aus, und jedes Modell behält seine eigene Lizenz.

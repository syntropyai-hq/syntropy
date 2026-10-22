<!-- Translation of README.md. Update it in the same change as README.md. -->
<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/media/logo-dark.svg">
    <img src="docs/media/logo-light.svg" alt="Syntropy" width="360">
  </picture>
</p>

<p align="center"><a href="README.md">English</a>&emsp;<a href="README.zh-CN.md">简体中文</a>&emsp;<a href="README.ja.md">日本語</a>&emsp;<a href="README.de.md">Deutsch</a>&emsp;<a href="README.fr.md">Français</a>&emsp;<b>Español</b>&emsp;<a href="README.pt-BR.md">Português</a></p>

<p align="center"><sub>Esta es una traducción. La versión de referencia es el <a href="README.md">README en inglés</a>.</sub></p>

<h3 align="center">Ejecuta un modelo grande en todas tus GPU, niveles de memoria y máquinas</h3>

<p align="center">La ubicación se decide midiendo tu hardware. Por defecto, la salida coincide con la referencia.<br>
Rust, con kernels nativos para CUDA, HIP, Metal, Vulkan y la CPU.</p>

<p align="center">
  <a href="#hoja-de-ruta"><img src="https://img.shields.io/badge/status-pre--alpha%2C_M0a_in_progress-7b5fd0" alt="Estado: prealfa, M0a en curso"></a>
  <a href="#licencia-y-créditos"><img src="https://img.shields.io/badge/licence-MIT_OR_Apache--2.0-3f6fd1" alt="Licencia: MIT OR Apache-2.0"></a>
  <a href="Cargo.toml"><img src="https://img.shields.io/badge/Rust-2024_edition-c2572b?logo=rust&logoColor=white" alt="Rust, edición 2024"></a>
  <a href="PLATFORMS.md"><img src="https://img.shields.io/badge/platforms-Linux%2C_Windows%2C_macOS-2f9a5a" alt="Plataformas: Linux, Windows, macOS"></a>
</p>

<p align="center">
  <a href="#cómo-funciona"><b>Cómo&nbsp;funciona</b></a>&emsp;
  <a href="#hoja-de-ruta"><b>Hoja&nbsp;de&nbsp;ruta</b></a>&emsp;
  <a href="PLATFORMS.md"><b>Plataformas</b></a>&emsp;
  <a href="plan/README.md"><b>El&nbsp;diseño</b></a>&emsp;
  <a href="CONTRIBUTING.md"><b>Contribuir</b></a>
</p>

> [!IMPORTANT]
> **Syntropy está en desarrollo y todavía no hay nada que instalar.** El diseño completo está en [`plan/`](plan/README.md), y la implementación está en marcha en el hito M0a. Todo lo que describe esta página está diseñado y planificado; no es algo que puedas ejecutar hoy.

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/media/memory-pool-dark.svg">
    <img src="docs/media/memory-pool-light.svg" width="860" alt="Tres máquinas dibujadas a la misma escala. Un PC para juegos con 24 GB de VRAM y 64 GB de RAM suma 88 GB, un portátil tiene 48 GB de memoria unificada y un mini PC tiene 128 GB. A ninguna le alcanza para un modelo de 142 GB. Juntas suman 264 GB, en los que el modelo cabe con 122 GB de sobra.">
  </picture>
</p>

Un PC para juegos con una tarjeta gráfica de 24 GB y 64 GB de RAM, un portátil con 48 GB de memoria unificada y un mini PC de 128 GB debajo del televisor no pueden, cada uno por separado, alojar un modelo de mezcla de expertos de 235 mil millones de parámetros, que ocupa 142 GB a 4 bits. Entre los tres suman 264 GB.

Syntropy los trata como una sola máquina con varios tipos de memoria. Mide a qué velocidad lee cada GPU y cada CPU de cada nivel de memoria, elige una única ubicación para todo el modelo y mantiene el cómputo junto a los bytes: el dispositivo en cuya memoria está un peso es el que calcula con él. Entre máquinas, durante la decodificación solo viajan las activaciones. Dentro de una máquina, un peso que no está en la caché de la GPU lo procesa la CPU allí donde reside, en lugar de copiarlo por PCIe en cada token.

## Características

Todo esto está especificado en el diseño, pero todavía no hay nada implementado.

### Dentro de una máquina

- **Todos los niveles de memoria trabajan.** Los pesos y los expertos MoE más calientes están en una caché adaptativa en VRAM. El conjunto completo de pesos permanece residente en RAM, donde la CPU calcula in situ los fallos de caché. Los expertos fríos y las tablas grandes se leen de NVMe con E/S directa cuando hacen falta.
- **Expertos que se cargan antes de necesitarse.** La predicción de los expertos de la capa siguiente se ejecuta dentro del kernel del router, así que las lecturas de NVMe y RAM para la capa siguiente ya están en curso mientras se calcula la actual.
- **Se mide, no se consulta.** Cada máquina mide lo que rinden de verdad sus GPU, CPU, unidades de almacenamiento y enlaces de red, con los kernels reales de Syntropy. La ubicación se construye a partir de esas mediciones, no de fichas técnicas.

### Entre máquinas

- **Una sola ubicación para todo el modelo.** Un único plan decide qué máquinas participan, qué rango de capas ejecuta cada una, qué nivel contiene cada grupo de tensores y qué dispositivo lo calcula. También registra por qué cada grupo acabó donde acabó.
- **Un anillo de máquinas.** El modelo se divide en etapas de pipeline que pasan los estados ocultos directamente a la máquina siguiente. Una misma capa solo se reparte entre máquinas en enlaces cuya latencia medida pasa el control, como RDMA sobre Thunderbolt 5.
- **Sin configuración en la LAN, un token fuera de ella.** Las máquinas de una misma red se encuentran entre sí. Una máquina nueva queda admitida en cuanto confirmas el mismo código de seis dígitos en las dos pantallas. Un portátil fuera de casa se une con un token de invitación que, por defecto, es de un solo uso.
- **Perder una máquina no significa perder la respuesta.** Cada etapa guarda lo que ha enviado recientemente. Si se cae la máquina de una etapa y las demás tienen espacio, la generación se reanuda sin que llegue ningún error al cliente.

### Para ti y tus herramientas

- **Exacto salvo que digas lo contrario.** Los atajos que cambian la salida, como omitir los expertos que no están en la caché, solo se activan si los pides, tienen nombre y figuran en cada respuesta en la que intervinieron.
- **Decodificación especulativa que conserva la respuesta.** Las cabezas MTP o un modelo borrador proponen varios tokens, el modelo los verifica todos en una sola pasada y la regla de aceptación preserva la distribución de salida del propio modelo.
- **Las API que ya usan tus herramientas.** API con el formato de las de OpenAI y Anthropic, más una nativa, servidas desde cualquier máquina a la que llegue la solicitud.
- **Código nativo de principio a fin.** Una capa de sistema en Rust, kernels nativos para CUDA, HIP, Metal, Vulkan y la CPU, y ggml como respaldo por operación para ampliar la cobertura.

## Un primer vistazo

Esta es la línea de comandos prevista en [RFC-0051](plan/rfcs/RFC-0051-cli-and-dashboard.md). Ninguno de estos comandos existe todavía.

```sh
syntropy run                     # hace una pregunta y luego recomienda un modelo que quepa
syntropy plan <model> --explain  # previsualiza la ubicación y la velocidad prevista sin cargar
syntropy cluster approve <id>    # admite una máquina de la LAN tras cotejar su código de seis dígitos
syntropy invite create           # un token de un solo uso para un portátil fuera de casa
```

Por defecto, cada máquina servirá las API en `http://127.0.0.1:7433`, así que los clientes y SDK de OpenAI y Anthropic que ya existen funcionarán sin cambios ([RFC-0050](plan/rfcs/RFC-0050-api-surface.md)):

```sh
curl http://127.0.0.1:7433/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model": "<model>", "stream": true,
       "messages": [{"role": "user", "content": "¿Por qué la decodificación está limitada por la memoria?"}]}'
```

## Hoja de ruta

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/media/milestones-dark.svg">
    <img src="docs/media/milestones-light.svg" width="860" alt="Siete hitos, de M0a a M4. M0a, el esqueleto funcional en una máquina Linux, está en curso. M1a es la primera versión preliminar, M1b añade Windows y Apple silicon, M2 es el lanzamiento con máquinas en una LAN, M3 completa las funcionalidades y M4 es la 1.0.">
  </picture>
</p>

El diseño está completo: 25 RFC, 98 ADR y 1411 tareas repartidas en seis áreas del código, cada tarea con criterios de aceptación verificables y con las secciones de RFC y los ADR que implementa. Los hitos no tienen fechas. [RFC-0001 §16](plan/rfcs/RFC-0001-system-architecture.md#16-milestones) define el alcance y los criterios de salida de cada uno, [ADR-0020](plan/decisions/ADR-0020-platform-support-tiers.md) las plataformas que publica cada uno, y la [hoja de ruta](plan/implementation/roadmap.md) enumera las tareas de cada hito en orden de dependencias. El avance se sigue en el [tablero del proyecto](https://github.com/orgs/syntropyai-hq/projects/1).

<details>
<summary><b>Qué entrega cada hito</b></summary>
<br>

| Hito | Qué entrega |
|---|---|
| **M0a** | Un esqueleto funcional en una máquina Linux: una familia densa (Llama) en Q4_K y Q8_0 sobre la CPU y una GPU NVIDIA, chat en streaming al estilo OpenAI a través del runner real, y la conformidad, el arnés de paridad y la comprobación contra la referencia de quien publica el modelo como controles obligatorios para fusionar desde el primer commit. Se construye y se prueba, pero no se publica. |
| **M0b** | El resto de M0 en la misma máquina: una familia MoE (Qwen-MoE, con todos los expertos en VRAM), tolerancias calibradas, selección certificada de kernels, las primeras sondas de hardware y un arnés de benchmarks vinculado a commits. No se publica. |
| **M1a** | Memoria por niveles en una máquina Linux con una GPU NVIDIA o solo con CPU: niveles VRAM, RAM y NVMe, la caché de expertos con predicción y precarga, ubicación en una sola máquina, paginación del KV con caché de prefijos y sesiones guardadas, y decodificación especulativa. La primera versión pública, etiquetada como preliminar. |
| **M1b** | Windows con NVIDIA y macOS en Apple silicon, además de solo CPU en ambos. Su alcance depende de lo que mida M1a. |
| **M2** | Varias máquinas en una LAN: descubrimiento y emparejamiento, transportes cifrados, el pipeline en anillo, distribución de pesos entre pares y el primer panel de control, además de una versión preliminar de Vulkan basada en ggml para GPU AMD e Intel. Esta versión es el lanzamiento. |
| **M3** | Funcionalidades completas: enlaces de invitación, recuperación tras perder una máquina, paralelismo de tensores y de expertos en enlaces rápidos, especulación entre máquinas, kernels nativos de HIP y Vulkan, y un modo de rendimiento agregado para varios usuarios. |
| **M4** | 1.0: plataformas Tier 2 y Tier 3, instaladores y actualizaciones firmados, y controles de regresión de rendimiento en hardware de referencia. |

</details>

## Dónde funcionará

Cada plataforma tiene un nivel de soporte, o tier ([ADR-0020](plan/decisions/ADR-0020-platform-support-tiers.md)), que describe el soporte que tendrá en la 1.0. [PLATFORMS.md](PLATFORMS.md) contiene la matriz completa, con las versiones mínimas, los backends y el hito en el que cada plataforma se publica por primera vez.

<img src="https://img.shields.io/badge/tier_1-supported-2f9a5a" alt="Tier 1: con soporte"><br>Linux x86_64 con NVIDIA, Windows 11 x86_64 con NVIDIA, macOS 15+ en Apple silicon, y solo CPU en los tres. Binarios precompilados y firmados. La paridad y el rendimiento se prueban en hardware real, y un fallo bloquea la publicación.

<img src="https://img.shields.io/badge/tier_2-known_gaps-3f6fd1" alt="Tier 2: con soporte y limitaciones conocidas"><br>Linux con AMD ROCm; Vulkan en GPU AMD, Intel y NVIDIA; AMD Strix Halo; Linux arm64 de clase GB10; placas de clase Raspberry Pi 5 (CPU); NIC RDMA; varias GPU en una misma máquina. Binarios precompilados, paridad obligatoria y seguimiento del rendimiento. Las limitaciones se enumeran en las notas de la versión.

<img src="https://img.shields.io/badge/tier_3-best_effort-818b98" alt="Tier 3: sin garantías"><br>Android e iOS como nodos; Intel SYCL; CUDA 12 para GPU Pascal y Volta; WebGPU; Windows 10 22H2; Linux bajo WSL2 con una GPU NVIDIA. Binarios publicados como experimentales.

No previsto para la 1.0: macOS en Intel, sistemas operativos de 32 bits, Windows en arm64 y un nodo que se ejecute dentro de un navegador.

## ¿Cómo funciona?

Generar un token para un usuario implica leer una vez todos los pesos activos y toda la caché KV. Se hacen pocas operaciones por byte, así que la velocidad de decodificación depende de dónde están los bytes y de lo rápido que puede leerlos el procesador que tienen al lado. Syntropy está diseñado en torno a eso: el cómputo va hacia los bytes, y los bytes se ubican según mediciones.

### Dentro de una máquina

<p align="center"><img src="docs/media/tiers.svg" width="760" alt="Una máquina: la VRAM contiene los pesos calientes y una caché de expertos, y la lee la GPU; la RAM contiene todos los pesos y la CPU calcula allí los fallos de caché; el NVMe contiene los expertos fríos y las tablas, y su contenido se precarga en la RAM"></p>

- La **VRAM** contiene lo que toca cada token (atención, capas densas, expertos compartidos, la caché KV) y una caché de los expertos MoE más usados que se adapta a la carga de trabajo.
- La **RAM** contiene el conjunto completo de pesos. Cuando un token necesita un experto que no está en la caché de la GPU, la CPU lo calcula allí mismo, a la vez que la GPU, en lugar de copiarlo por PCIe. Una CPU de escritorio lee DDR5 de doble canal a 76–122 GB/s, unas 3–5 veces lo que PCIe 4.0 x16 entrega a la GPU.
- El **NVMe** contiene los expertos fríos y las tablas de consulta grandes, almacenados en extents de expertos completos y leídos con E/S directa. La predicción de la capa siguiente inicia esas lecturas antes de que el router las pida.

El gestor de niveles de Syntropy es el único responsable de cada byte residente, así que la caché de páginas del sistema operativo nunca decide qué permanece en memoria. Los pesos solo se envían en streaming a la GPU para lotes grandes, como fragmentos largos del prompt, y solo donde el punto de equilibrio medido indica que copiar sale mejor que calcular in situ.

### Entre máquinas

<p align="center"><img src="docs/media/ring.svg" width="760" alt="Tres máquinas en anillo, cada una ejecuta un rango de capas: los estados ocultos pasan de una a la siguiente; la última muestrea el token, lo devuelve a la primera y lo envía en streaming a la máquina que recibió la solicitud"></p>

El modelo se divide en etapas por rango de capas, una etapa por máquina, y las etapas forman un anillo. La primera etapa calcula los embeddings de la entrada y cada etapa pasa su estado oculto directamente a la siguiente. La última etapa muestrea el token, lo devuelve por el anillo hasta la primera y lo envía en streaming a la máquina que recibió la solicitud. Un pipeline aporta capacidad, no velocidad en un único flujo, así que la ganancia viene de llevar los bytes a una memoria más rápida. Una máquina lenta se deja fuera del anillo cuando solo añadiría un salto.

Repartir una misma capa entre máquinas, con paralelismo de tensores o de expertos, cuesta un viaje de ida y vuelta por la red por cada capa y cada token. Syntropy solo lo hace en enlaces cuya latencia medida en operaciones colectivas es lo bastante baja como para que compense. En la práctica, eso significa enlaces de clase RDMA. El Wi-Fi nunca cumple el requisito.

Se elige una máquina como coordinador, que mantiene el registro de eventos del clúster con la pertenencia y los planes. El coordinador nunca ve un token y no está en la ruta de las solicitudes, así que perderlo no hace fallar ninguna solicitud.

> [!TIP]
> La explicación más extensa y en lenguaje llano está en [docs/HOW_IT_WORKS.md](docs/HOW_IT_WORKS.md). Los diseños completos están en los RFC: niveles y residencia ([RFC-0020](plan/rfcs/RFC-0020-tier-manager-and-residency.md)), la ruta de E/S de los pesos ([RFC-0021](plan/rfcs/RFC-0021-weight-io-path.md)), la caché de expertos ([RFC-0023](plan/rfcs/RFC-0023-moe-expert-cache-prediction-prefetch.md)), la ubicación ([RFC-0011](plan/rfcs/RFC-0011-joint-placement-solver.md)), la ejecución distribuida ([RFC-0032](plan/rfcs/RFC-0032-distributed-execution.md)), y el descubrimiento y las invitaciones ([RFC-0030](plan/rfcs/RFC-0030-discovery-membership-invites.md)).

## Exacto por defecto

Con la configuración por defecto, la salida de Syntropy debe coincidir con una pasada hacia adelante de referencia del mismo modelo con la misma cuantización, dentro de tolerancias publicadas ([ADR-0019](plan/decisions/ADR-0019-exactness-contract.md)).

- **La referencia** es el ejecutor de referencia de Syntropy: acumulación en F32, un orden de reducción fijo y pesos descuantizados de forma exacta a partir del mismo archivo. A su vez, se contrasta con la implementación de referencia de quien publica el modelo, de modo que «exacto» abarca tanto la interpretación que hace Syntropy del modelo como sus kernels.
- **La prueba** mide la coincidencia top-1 con teacher forcing y la divergencia KL sobre un corpus versionado de chat, código, contexto largo, JSON de llamadas a herramientas y texto multilingüe. Las tolerancias se fijan por clase de dtype a partir de mediciones, se versionan y quedan registradas en cada versión publicada.
- **Lo que tiene que pasar la prueba** es cada ruta que el planificador puede elegir en modo exacto: cada kernel y cada respaldo, cada forma de ubicación desde una GPU hasta un anillo de máquinas, la decodificación especulativa y la normal, la paginación del KV entre niveles y las sesiones reanudadas. Una ruta que falla es aproximada, sea cual sea la intención con la que se diseñó.
- **Los atajos solo se activan a petición y tienen nombre**, como `expert_skip` (no se calculan los expertos que no están en la caché) o `kv_lossy` (el KV se guarda con menos precisión que la del modelo). Cada respuesta indica su fidelidad y los atajos que estaban activos, y una solicitud que pide `exact` nunca la atiende una configuración aproximada.

No se promete una salida idéntica byte a byte entre dispositivos distintos, porque cada hardware redondea de forma diferente dentro de la tolerancia. Está previsto un modo `deterministic`, más lento, para obtener una salida repetible con la misma compilación y los mismos dispositivos. Las tolerancias y el arnés de paridad se especifican en [RFC-0042](plan/rfcs/RFC-0042-quantization-and-numerical-fidelity.md) y [RFC-0060](plan/rfcs/RFC-0060-testing-and-correctness.md).

## El diseño

[`plan/README.md`](plan/README.md) es el índice y propone un orden de lectura.

| Ruta | Qué contiene |
|---|---|
| [`plan/00-vision-and-scope.md`](plan/00-vision-and-scope.md) | El problema, los escenarios, la matriz de plataformas, los modelos objetivo, los criterios de éxito SC-1 a SC-32 (todos son objetivos) y qué significa «funcionalidades completas» |
| [`plan/decisions/`](plan/decisions/) | Los ADR. ADR-0000 recoge las directivas de producto D1–D8, ADR-0001 a ADR-0022 la base, y el resto se numeran por área |
| [`plan/rfcs/`](plan/rfcs/) | RFC-0001, la arquitectura del sistema, y 24 RFC de dominio sobre ubicación, niveles, ejecución distribuida, cómputo, producto y calidad |
| [`plan/implementation/`](plan/implementation/README.md) | Las tareas en seis archivos de épicas, la hoja de ruta, el informe de dependencias, la trazabilidad, el registro de riesgos y los scripts que los comprueban |
| [`plan/open-questions.md`](plan/open-questions.md) | Lo que aún está sin decidir, cada punto con la tarea o la medición que lo resuelve |
| [`plan/glossary.md`](plan/glossary.md) | Un término por concepto y los sinónimos que el plan no usa |
| [`plan/research/`](plan/research/) | La evidencia: notas sobre inferencia multidispositivo y con memoria por niveles, backends, y un modelo de costos de hardware con cifras de ancho de banda citadas |

Cada cifra del plan está etiquetada como `[target]` (objetivo), como `[estimate]` (estimación) con su derivación, o como una medición de hardware citada. Ninguna es un benchmark de Syntropy, porque todavía no hay nada que medir.

## Contribuir

Syntropy lo construyen agentes de IA que ejecutan las tareas del plan. Un agente toma una tarea, la implementa en su propia rama y abre un único cambio que cumple los criterios de aceptación de la tarea y cita las secciones de RFC y los ADR que implementa. El grafo de tareas decide qué puede empezar.

Un cambio solo se fusiona cuando pasa todo lo siguiente ([ADR-0021](plan/decisions/ADR-0021-development-model-ai-agents-open-source.md)):

- compilación, lint y formato, y pruebas unitarias, basadas en propiedades y de simulador;
- el arnés de paridad y los controles de conformidad;
- los controles de regresión de rendimiento, cuando el cambio toca una ruta medida;
- la política de dependencias y licencias, la comprobación de nombres y la comprobación del grafo de tareas;
- un agente de revisión independiente, nunca el que escribió el cambio, que lo contrasta con los criterios de aceptación de la tarea y con las secciones de RFC citadas.

El arnés de paridad y los controles de conformidad se construyen primero, en M0, antes que los kernels y las familias de modelos que comprueban. Los mantenedores aprueban además las clases de cambios sensibles: claves, firma, admisión y el cliente de actualizaciones, la configuración de CI y de publicación, los archivos de instrucciones de los agentes, las familias de modelos nuevas, las dependencias directas nuevas y cualquier cambio que ningún control pueda comprobar o que requiera eximirlo de un control.

Las contribuciones externas son bienvenidas, con los mismos controles. [CONTRIBUTING.md](CONTRIBUTING.md) cubre las issues, los pull requests, la evidencia y el trabajo asistido por IA, y [AGENTS.md](AGENTS.md) es la guía de trabajo para agentes.

## Seguridad

Por favor, informa de las vulnerabilidades en privado y no en una issue pública. [SECURITY.md](SECURITY.md) explica cómo informar, los plazos de respuesta previstos y un resumen de lo que Syntropy expondrá en una red: el descubrimiento en la LAN, el emparejamiento, los enlaces de invitación, su API y su canal de actualizaciones.

## Licencia y créditos

Syntropy se distribuye bajo la licencia [MIT](LICENSE-MIT) o la [Apache-2.0](LICENSE-APACHE), a tu elección. Las contribuciones se aceptan en los mismos términos, con un sign-off del Developer Certificate of Origin en cada commit ([ADR-0018](plan/decisions/ADR-0018-licence.md)).

La capa de kernels de Syntropy se apoya en [ggml](https://github.com/ggml-org/ggml), la biblioteca de tensores con licencia MIT, que aporta una amplia cobertura de backends y el respaldo para cada operación que no tiene kernel nativo. Cada versión incluirá un archivo `THIRD-PARTY-NOTICES` generado que enumera todos los componentes que contiene. Syntropy no distribuye pesos de modelos, y cada modelo conserva su propia licencia.

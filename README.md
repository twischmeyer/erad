

[![CodeFactor](https://www.codefactor.io/repository/github/nlr-distribution-suite/erad/badge)](https://www.codefactor.io/repository/github/nlr-distribution-suite/erad) • [![codecov](https://codecov.io/gh/NLR-Distribution-Suite/erad/graph/badge.svg?token=FtAWhS5svb)](https://codecov.io/gh/NLR-Distribution-Suite/erad) • [![GitHub license](https://img.shields.io/github/license/NREL/erad)](https://github.com/NREL/erad/blob/main/LICENSE.txt) • [![GitHub issues](https://img.shields.io/github/issues/NREL/erad)](https://github.com/NREL/erad/issues) • ![PyPI - Downloads](https://img.shields.io/pypi/dm/NREL-erad) • [![Upload to PyPi](https://github.com/NLR-Distribution-Suite/erad/actions/workflows/publish.yml/badge.svg)](https://github.com/NLR-Distribution-Suite/erad/actions/workflows/publish.yml) • [![deploy-book](https://github.com/NLR-Distribution-Suite/erad/actions/workflows/deploy.yml/badge.svg)](https://github.com/NLR-Distribution-Suite/erad/actions/workflows/deploy.yml) • [![Pytest](https://github.com/NLR-Distribution-Suite/erad/actions/workflows/pull_request_tests.yml/badge.svg)](https://github.com/NLR-Distribution-Suite/erad/actions/workflows/pull_request_tests.yml) • [![DOI](https://joss.theoj.org/papers/10.21105/joss.08782/status.svg)](https://doi.org/10.21105/joss.08782) • ![MCP Server](https://img.shields.io/badge/MCP_Server-enabled-brightgreen) • ![MCP Tools](https://img.shields.io/badge/MCP_Tools-26-blue)

<p align="center"> 
<img src="docs/_static/light.png" width="400" style="display:flex;justify-content:center;">
</p>

# ERAD (<u>E</u>nergy <u>R</u>esilience <u>A</u>nalysis for electric <u>D</u>istribution systems)

[Visit full documentation here.](https://nlr-distribution-suite.github.io/erad/)

Understanding the impact of disaster events on people's ability to access critical services is key to designing appropriate programs to minimize the overall impact. Flooded roads, downed power lines, flooded power substation etc. could impact access to critical services like electricity, food, health and more. The field of disaster modeling is still evolving and so is our understanding of how these events would impact our critical infrastructures such power grid, hospitals, groceries, banks etc.

ERAD is a free, open-source Python toolkit for computing energy resilience measures in the face of hazards like earthquakes and flooding. It uses graph database to store data and perform computation at the household level for a variety of critical services that are connected by power distribution network. It uses asset fragility curves, which are functions that relate hazard severity to survival probability for power system assets including cables, transformers, substations, roof-mounted solar panels, etc. recommended in top literature. Programs like undergrounding, microgrid, and electricity backup units for critical infrastructures may all be evaluated using metrics and compared across different neighborhoods to assess their effects on energy resilience.

ERAD is designed to be used by researchers, students, community stakeholders, distribution utilities to understand and possibly evaluate effectiveness of different post disaster programs to improve energy resilience. It was funded by National Renewable Energy Laboratory (NREL) and made publicly available with open license.

```mermaid
flowchart TD
A([Start])
B[Select entry point<br/>CLI or Python API or MCP tools]
C[Load Asset System<br/>Distribution model from file or cache]
D{Hazard input available?}
E[Load Hazard Model<br/>from JSON or model reference]
F[Create Empty Hazard System]
G[Add Historic Hazard Events<br/>Earthquake, Hurricane, Wildfire]
H[Validate fragility curve set<br/>DEFAULT_CURVES or custom]
I[Run Simulation<br/>asset system id plus hazard system id]
J[Simulation Result ID<br/>asset states and probabilities]
K[Analyze Results<br/>query assets, stats, topology]
L{Need uncertainty scenarios?}
M[Generate Monte Carlo Scenarios<br/>num samples and seed]
N[Tracked Changes Output]
O{Need exports?}
P[Export to SQLite<br/>for downstream analytics]
Q[Export to JSON<br/>asset and hazard systems]
R[Export Tracked Changes<br/>scenario deltas]
S[Clear loaded systems<br/>free memory and reset state]
T([End])

A --> B --> C --> D
D -- Yes --> E --> H
D -- No --> F --> G --> H
H --> I --> J --> K --> L
L -- Yes --> M --> N --> O
L -- No --> O
O -- Yes --> P
O -- Yes --> Q
O -- Yes --> R
O -- No --> S
P --> S
Q --> S
R --> S
S --> T

classDef startEnd fill:#0f172a,color:#ffffff,stroke:#0f172a,stroke-width:2px
classDef intake fill:#e0f2fe,color:#0c4a6e,stroke:#0284c7,stroke-width:1.5px
classDef decision fill:#fff7ed,color:#9a3412,stroke:#f97316,stroke-width:2px
classDef output fill:#f5f3ff,color:#4c1d95,stroke:#8b5cf6,stroke-width:1.5px
classDef cleanup fill:#f1f5f9,color:#334155,stroke:#64748b,stroke-width:1.5px

class A,T startEnd
class B,C,E,F,G,H,I,K,M intake
class D,L,O decision
class J,N,P,Q,R output
class S cleanup
```

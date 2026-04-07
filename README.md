# Flow Problems — Operations Research

A Python project that implements and benchmarks classic **network flow algorithms** on directed graphs.

---

## Features

| Algorithm | Problem solved |
|-----------|----------------|
| **Ford-Fulkerson** (BFS / Edmonds-Karp) | Maximum flow |
| **Push-Relabel** | Maximum flow |
| **Min-Cost Flow** (Bellman-Ford) | Minimum cost flow for a given desired flow |

Additional capabilities:
- Load graphs from text files or generate random ones
- Verbose step-by-step trace logged to `traces/`
- Automated complexity benchmarks saved to `complexité/`

---

## Project Structure

```
.
├── main.py                  # Entry point
├── functions.py             # All algorithms and helpers
├── graphes/                 # Sample graph files (Graphe_1.txt … Graphe_10.txt)
├── traces/                  # Algorithm execution traces (generated)
├── complexité/              # CSV benchmark results (generated)
└── RAP_Recherche_Operationnelle1.pdf  # Project report
```

---

## Requirements

- Python 3.8 or later  
- No external dependencies (standard library only)

---

## Usage

```bash
python main.py
```

You will be presented with an interactive menu:

```
Menu principal:
1. Charger un graphe depuis un fichier
2. Générer un graphe aléatoire
3. Évaluation de la complexité (tests de performance)
4. Quitter
```

### Option 1 — Load a graph from a file

Enter a graph number (e.g. `1` loads `graphes/Graphe_1.txt`), then choose the algorithm to run:

```
1. Ford-Fulkerson
2. Push-Relabel
3. Flot à coût minimal
```

Results are printed to the console and appended to `traces/E2_trace<N>.txt`.

### Option 2 — Generate a random graph

Enter the number of vertices `n`. A random graph is generated and you can run any algorithm on it.

### Option 3 — Complexity benchmarks

Runs all three algorithms on random graphs of sizes `[10, 20, 40, 100, 400, 1000, 4000, 10000]` with 100 iterations each. Execution times are saved to:

```
complexité/complexity_Ford_Fulkerson.csv
complexité/complexity_Push_Relabel.csv
complexité/complexity_MinCostFlow.csv
```

---

## Graph File Format

Graph files use an **adjacency matrix** representation:

```
<n>
<n×n capacity matrix>
<n×n cost matrix>   ← optional, required for Min-Cost Flow
```

**Example** (`graphes/Graphe_1.txt`, 8 nodes):

```
8
0 9 5 7 0 0 0 0
0 0 6 0 6 0 0 0
...
0 1 1 1 0 0 0 0
...
```

- The **first node** (`A`, index 0) is the **source** (`s`).
- The **last node** (index `n-1`) is the **sink** (`t`).
- Intermediate nodes are labelled `a`, `b`, `c`, …

---

## Algorithms

### Ford-Fulkerson (Edmonds-Karp)

Uses **BFS** to find augmenting paths in the residual graph. At each iteration the path with maximum bottleneck capacity is chosen, guaranteeing polynomial time.

**Complexity:** O(V · E²)

### Push-Relabel

Maintains a *height* function and *excess* values per node. Excess flow is pushed from higher to lower nodes; nodes are relabelled when no push is possible.

**Complexity:** O(V² · √E)

### Min-Cost Flow (Bellman-Ford / Successive Shortest Paths)

Finds the cheapest augmenting path using **Bellman-Ford** on the residual graph, then pushes flow along it. Repeats until the desired flow is reached.

**Complexity:** O(V · E · desired_flow)


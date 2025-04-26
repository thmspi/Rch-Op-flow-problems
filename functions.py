#!/usr/bin/env python3
import random
import time
import csv
import os
from collections import deque

# Function to read the file format and load graphs

def read_graph_from_file(filename):
    print(f"Tentative d'ouverture de : {filename}")
    with open(filename, 'r') as f:
        lines = f.readlines()
    lines = [line.strip() for line in lines if line.strip()]
    n = int(lines[0])
    capacity = []
    # Capacity matrix
    for i in range(1, n + 1):
        row = list(map(int, lines[i].split()))
        capacity.append(row)
    cost = None
    # If cost matrix
    if len(lines) >= n + 1 + n:
        cost = []
        for i in range(n + 1, n + 1 + n):
            row = list(map(int, lines[i].split()))
            cost.append(row)
    return n, capacity, cost

# Function to display the matrix (v0)
def print_matrix(matrix, labels, title=""):
    if title:
        print(title)
    header = "    " + " ".join(f"{l:>5}" for l in labels)
    print(header)
    for i, row in enumerate(matrix):
        row_label = labels[i]
        row_str = f"{row_label:>3} " + " ".join(f"{val:>5}" for val in row)
        print(row_str)
    print()

# Functions to switch back and forth between letter and number (for display)
def number_to_letters(k):
    result = ""
    while True:
        result = chr(97 + (k % 26)) + result
        k = k // 26 - 1
        if k < 0:
            break
    return result

def generate_vertex_labels(n):
    labels = []
    for i in range(n):
        if i == 0:
            labels.append("s")
        elif i == n - 1:
            labels.append("t")
        else:
            labels.append(number_to_letters(i - 1))
    return labels

def bfs(residual, source, sink):
    """
    BFS dans le graphe résiduel.
    Retourne une liste représentant le chemin de source à sink s'il existe, sinon None.
    """
    n = len(residual)
    visited = [False] * n
    parent = [-1] * n
    queue = deque()

    queue.append(source)
    visited[source] = True

    while queue:
        u = queue.popleft()

        for v in range(n):
            if not visited[v] and residual[u][v] > 0:
                parent[v] = u
                visited[v] = True
                queue.append(v)
                if v == sink:
                    # Reconstituer le chemin de source à sink
                    path = []
                    current = sink
                    while current != -1:
                        path.insert(0, current)
                        current = parent[current]
                    return path

    return None  # Aucun chemin trouvable
# ------------------------------------------------------------------
# Algorithme ford_fulkerson
# ------------------------------------------------------------------

def ford_fulkerson(capacity, source, sink, labels, log_filename=None, verbose=True):
    n = len(capacity)
    flow = [[0] * n for _ in range(n)]
    max_flow = 0
    iteration = 0
    log_lines = []

    def build_residual_graph():
        residual = [[0] * n for _ in range(n)]
        for u in range(n):
            for v in range(n):
                if capacity[u][v] > flow[u][v]:
                    residual[u][v] = capacity[u][v] - flow[u][v]
                if flow[u][v] > 0:
                    residual[v][u] = flow[u][v]
        return residual

    while True:
        residual = build_residual_graph()
        path = bfs(residual, source, sink)

        if not path:
            break  # plus de chemin augmentant

        # Calcul du flot possible sur ce chemin
        path_flow = float("inf")
        for i in range(len(path) - 1):
            u = path[i]
            v = path[i + 1]
            path_flow = min(path_flow, residual[u][v])

        # Log affichage
        iteration += 1
        path_labels = " -> ".join(labels[i] for i in path)
        log_lines.append(f"Iteration {iteration}: Chemin augmentant: {path_labels} avec flot {path_flow}")
        if verbose:
            print(f"Iteration {iteration}: Chemin augmentant: {path_labels} avec flot {path_flow}")

        # Mise à jour des flots
        for i in range(len(path) - 1):
            u = path[i]
            v = path[i + 1]
            if capacity[u][v] > 0:
                flow[u][v] += path_flow
            else:
                flow[v][u] -= path_flow

        max_flow += path_flow

        if verbose:
            print_matrix(build_residual_graph(), labels, title="Capacités résiduelles")

    log_lines.append(f"Flot maximum: {max_flow}")
    if verbose:
        print(f"Flot maximum: {max_flow}")
    if log_filename:
        with open(log_filename, "w") as f:
            for line in log_lines:
                f.write(line + "\n")
    return max_flow


def push_relabel(capacity, source, sink, labels, log_filename=None, verbose=True):
    """
    Implements the Push-Relabel algorithm for maximum flow calculation.
    The program provides a detailed log for each push and relabel operation.
    """

    # Initialization

    n = len(capacity)
    flow = [[0] * n for _ in range(n)]
    excess = [0] * n
    height = [0] * n
    log_lines = []
    height[source] = n

    # Filling all neighboors of the source 
    for v in range(n):
        # Node saturation for each node from the source
        flow[source][v] = capacity[source][v]
        # Negative flow to build the residual graph (if you need to go from b to a)
        flow[v][source] = -flow[source][v]
        # temporary excess until node are processed
        excess[v] = capacity[source][v]
    excess[source] = 0
    #bliblou

    def push(u, v):
        # Determine how much flow can we really send (depends on b to c)
        send = min(excess[u], capacity[u][v] - flow[u][v])
        # Update flow and excess matrices
        flow[u][v] += send
        flow[v][u] -= send
        excess[u] -= send
        excess[v] += send
        if verbose:
            log_lines.append(f"Pousser {send} de {labels[u]} vers {labels[v]}")
        return send

    def relabel(u):
        min_height = float("inf")
        # We check if there is a neighboor where we can push flow just depending on the height
        for v in range(n):
            if capacity[u][v] - flow[u][v] > 0:
                min_height = min(min_height, height[v])
        old_height = height[u]
        # Update height depending on the lowest neighboord height found
        height[u] = min_height + 1
        if verbose:
            log_lines.append(f"Réétiqueter {labels[u]} de {old_height} à {height[u]}")

    # Iterate over each node except s and t as they won't be relabeled
    vertices = [i for i in range(n) if i != source and i != sink]
    # Loop that go through every node and start again if there's any change
    p = 0
    while p < len(vertices):
        u = vertices[p]
        advanced = False
        # Loop to push flow if it's available after we update the height
        for v in range(n):
            if capacity[u][v] - flow[u][v] > 0 and height[u] == height[v] + 1:
                if excess[u] > 0:
                    push(u, v)
                    advanced = True
        # If not then we have to relabel the node
        if not advanced:
            relabel(u)
        # If there's no excess, nothing change so we increment, else we start over
        if excess[u] == 0:
            p += 1
        else:
            p = 0
        if verbose:
            log_lines.append(f"État de {labels[u]}: hauteur = {height[u]}, excès = {excess[u]}")

    # Delete all remaining excess next to the source
    for u in range(n):
        if u != source and u != sink and excess[u] > 0:
            for v in range(n):
                if capacity[u][v] - flow[u][v] > 0 and height[u] == height[v] + 1:
                    push(u, v)

    max_flow = sum(flow[source][i] for i in range(n))
    if verbose:
        print("Matrice de flot finale:")
        print_matrix(flow, labels, title="Matrice de flot")
        print("Hauteurs:", {labels[i]: height[i] for i in range(n)})
        print("Excès:", {labels[i]: excess[i] for i in range(n)})
        print(f"Flot maximum (Push-Relabel): {max_flow}")
    if log_filename is not None:
        with open(log_filename, "w") as f:
            for line in log_lines:
                f.write(line + "\n")
    return max_flow

def min_cost_flow(capacity, cost, source, sink, desired_flow,labels, log_filename=None, verbose=True):
    n = len(capacity)
    
    residual_cap = [row[:] for row in capacity]
    residual_cost = [row[:] for row in cost]       
    flow   = [[0]*n for _ in range(n)]
    total_flow  = 0
    total_cost  = 0
    INF = float("inf")
    log_lines = []
    iteration = 0

    if verbose:
        print("Début du flot à coût minimal (flot demandé =", desired_flow, ")")

    while total_flow < desired_flow:
        iteration += 1
        
        dist   = [INF]*n
        parent = [-1]*n
        dist[source] = 0
        for _ in range(n-1):
            improved = False
            for u in range(n):
                if dist[u] == INF:           
                    continue
                for v in range(n):
                    if residual_cap[u][v] > 0 and dist[u] + residual_cost[u][v] < dist[v]:
                        dist[v] = dist[u] + residual_cost[u][v]
                        parent[v] = u
                        improved = True
            if not improved:
                break

        
        if dist[sink] == INF:
            raise ValueError("Impossible d’envoyer le flot demandé : réseau saturé")

        
        path = []
        path_flow = INF
        v = sink
        while v != source:
            u = parent[v]
            path.insert(0, v)
            path_flow = min(path_flow, residual_cap[u][v])
            v = u
        path.insert(0, source)

        
        path_flow = min(path_flow, desired_flow - total_flow)

        
        # Toujours construire ces variables, peu importe verbose
        bellman_str = ", ".join(
            f"{labels[i]}:{dist[i] if dist[i]!=INF else 'INF'}" for i in range(n)
        )
        path_labels = " -> ".join(labels[i] for i in path)

        if verbose:
            print(f"Iteration {iteration}: distances -> {bellman_str}")
            print(f"Iteration {iteration}: chemin {path_labels} | flot {path_flow} | coût unitaire {dist[sink]}")

        # Peu importe verbose, ajouter au log
        log_lines.append(f"Iteration {iteration}: chemin {path_labels} | flot {path_flow} | coût unitaire {dist[sink]}")


        
        v = sink
        while v != source:
            u = parent[v]
           
            flow[u][v] += path_flow
            flow[v][u] -= path_flow
            
            residual_cap[u][v] -= path_flow
            residual_cap[v][u] += path_flow
            
            residual_cost[v][u] = -residual_cost[u][v] 
            v = u

        total_flow += path_flow
        total_cost += path_flow * dist[sink]

        if verbose:
            print_matrix(residual_cap, labels, title="Capacités résiduelles après augmentation")

    # Fin 
    if verbose:
        print(f"Flot total envoyé = {total_flow} | Coût total = {total_cost}")

    log_lines.append(f"Flot total = {total_flow}")
    log_lines.append(f"Coût total = {total_cost}")
    if log_filename is not None:
        with open(log_filename, "w") as f:
            f.write("\n".join(log_lines))

    return total_flow, total_cost

def generate_random_graph(n):
    """
    Génère un graphe aléatoire de n sommets,
    avec exactement n²/2 arêtes,
    sans boucle (i ≠ j),
    et avec des capacités et coûts aléatoires.
    """
    capacity = [[0] * n for _ in range(n)]
    cost = [[0] * n for _ in range(n)]
    
    # Liste de toutes les arêtes possibles sans boucle
    edges = [(i, j) for i in range(n) for j in range(n) if i != j]
    
    # Choisir aléatoirement exactement n²/2 arêtes parmi toutes les possibles
    num_edges = (n * n) // 2
    selected_edges = random.sample(edges, num_edges)
    
    # Assigner capacité et coût aléatoire aux arêtes choisies
    for i, j in selected_edges:
        capacity[i][j] = random.randint(1, 20)
        cost[i][j] = random.randint(1, 10)
    
    return capacity, cost


def evaluate_complexity():
    """
    Pour chaque taille n ∈ {10, 20, 40, 100, 400},
    génère un graphe aléatoire,
    puis exécute Ford-Fulkerson, Push-Relabel et Min-Cost-Flow
    sur le même graphe pour 100 itérations.
    """
    sizes = [10, 20, 40, 100]
    #sizes = [10, 20, 40, 100, 400, 1000, 4000, 10000]

    # Préparer un CSV pour chaque algorithme
    files = {
        'Ford-Fulkerson': open("complexity_Ford_Fulkerson.csv", "w", newline=""),
        'Push-Relabel': open("complexity_Push_Relabel.csv", "w", newline=""),
        'MinCostFlow': open("complexity_MinCostFlow.csv", "w", newline=""),
    }
    writers = {name: csv.writer(f) for name, f in files.items()}
    for writer in writers.values():
        writer.writerow(["n", "iteration", "execution_time"])

    for n in sizes:
        print(f"=== Tests pour n = {n} ===")
        for iteration in range(1, 101):
            # 1. Générer un graphe aléatoire
            capacity, cost = generate_random_graph(n)
            labels = generate_vertex_labels(n)
            source = 0
            sink = n - 1

            # 2. Ford-Fulkerson
            start = time.time()
            max_flow_ff = ford_fulkerson(capacity, source, sink, labels, verbose=False)
            end = time.time()
            exec_time_ff = end - start
            writers['Ford-Fulkerson'].writerow([n, iteration, exec_time_ff])

            # 3. Push-Relabel
            start = time.time()
            max_flow_pr = push_relabel(capacity, source, sink, labels, verbose=False)
            end = time.time()
            exec_time_pr = end - start
            writers['Push-Relabel'].writerow([n, iteration, exec_time_pr])

            # 4. Min-Cost Flow
            # Choisir desired_flow = max_flow / 2, arrondi à l'entier supérieur (>=1)
            max_flow = max(max_flow_ff, max_flow_pr)
            desired_flow = max(1, max_flow // 2)
            start = time.time()
            try:
                min_cost_flow(capacity, cost, source, sink, desired_flow, labels, verbose=False)
            except ValueError:
                print(f"\nMinCostFlow impossible pour n={n}, itération={iteration}. Réseau saturé.\n")

            end = time.time()
            exec_time_mcf = end - start
            writers['MinCostFlow'].writerow([n, iteration, exec_time_mcf])

        print(f"=== Terminé pour n = {n} ===\n\n\n")

    # Fermer tous les fichiers
    for f in files.values():
        f.close()


def main():
    while True:
        print("Menu principal:")
        print("1. Charger un graphe depuis un fichier")
        print("2. Générer un graphe aléatoire")
        print("3. Évaluation de la complexité (tests de performance)")
        print("4. Quitter")
        choix = input("Votre choix: ").strip()
        if choix == "1":
            file_number = input("Entrez le numéro du graphe: ").strip()
            filename = f"graphes/Graphe_{file_number}.txt"
            try:
                n, capacity, cost = read_graph_from_file(filename)
            except Exception as e:
                print("Erreur lors de la lecture du fichier:", e)
                continue
            labels = generate_vertex_labels(n)
            print_matrix(capacity, labels, title="Matrice des capacités")
            if cost is not None:
                print_matrix(cost, labels, title="Matrice des coûts")
            else:
                # Si la matrice de coûts n'est pas fournie, on suppose un coût de 1 pour chaque arête existante
                cost = [[1 if capacity[i][j] != 0 else 0 for j in range(n)] for i in range(n)]
            print("Choisissez l'algorithme à utiliser:")
            print("1. Ford-Fulkerson")
            print("2. Push-Relabel")
            print("3. Flot à coût minimal")
            alg_choice = input("Votre choix: ").strip()
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            if alg_choice == "1":
                log_filename = f"E2_trace{file_number}-FF.txt"
                max_flow = ford_fulkerson(capacity, 0, n - 1, labels, log_filename=log_filename)
                print(f"Flot maximum (Ford-Fulkerson): {max_flow}")
            elif alg_choice == "2":
                log_filename = f"E2_trace{file_number}-PR.txt"
                max_flow = push_relabel(capacity, 0, n - 1, labels, log_filename=log_filename)
                print(f"Flot maximum (Push-Relabel): {max_flow}")
            elif alg_choice == "3":
                log_filename = f"E2_trace{file_number}-MF.txt"
                desired_flow = int(input("Entrez le flow désiré: ").strip())
                max_flow, total_cost = min_cost_flow(capacity, cost, 0, n - 1, desired_flow, labels, log_filename=log_filename)
                print(f"Flot maximum (Flot à coût minimal): {max_flow} avec coût total: {total_cost}")
            else:
                print("Choix d'algorithme invalide.")
        elif choix == "2":
            try:
                n = int(input("Entrez le nombre de sommets (n): ").strip())
            except ValueError:
                print("Veuillez entrer un nombre valide.")
                continue
            capacity, cost = generate_random_graph(n)
            labels = generate_vertex_labels(n)
            print_matrix(capacity, labels, title="Matrice des capacités (Graphe aléatoire)")
            print_matrix(cost, labels, title="Matrice des coûts (Graphe aléatoire)")
            print("Choisissez l'algorithme à utiliser:")
            print("1. Ford-Fulkerson")
            print("2. Push-Relabel")
            print("3. Flot à coût minimal")
            alg_choice = input("Votre choix: ").strip()
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            if alg_choice == "1":
                log_filename = f"E2_trace{file_number}-FF.txt"
                max_flow = ford_fulkerson(capacity, 0, n - 1, labels, log_filename=log_filename)
                print(f"Flot maximum (Ford-Fulkerson): {max_flow}")
            elif alg_choice == "2":
                log_filename = f"E2_trace{file_number}-PR.txt"
                max_flow = push_relabel(capacity, 0, n - 1, labels, log_filename=log_filename)
                print(f"Flot maximum (Push-Relabel): {max_flow}")
            elif alg_choice == "3":
                log_filename = f"E2_trace{file_number}-MF.txt"
                desired_flow = int(input("Entrez le flow désiré: ").strip())
                max_flow, total_cost = min_cost_flow(capacity, cost, 0, n - 1, desired_flow, labels, log_filename=log_filename)
                print(f"Flot maximum (Flot à coût minimal): {max_flow} avec coût total: {total_cost}")

            else:
                print("Choix d'algorithme invalide.")
        elif choix == "3":
            evaluate_complexity()
        elif choix == "4":
            print("Au revoir!")
            break
        else:
            print("Choix invalide.")
        print()
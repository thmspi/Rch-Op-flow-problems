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
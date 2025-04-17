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
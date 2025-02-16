import pandas as pd
from graphviz import Digraph

# Load the CSV file
df = pd.read_csv("step_4.csv", keep_default_na=False)

# Create a new directed graph
dot = Digraph(comment="Taxonomy Tree", format="png")

# Dictionaries to store unique nodes
policy_nodes = {}
subtopic1_nodes = {}
subtopic2_nodes = {}

# Iterate through each row in the DataFrame
for idx, row in df.iterrows():
    no_second_node = False
    policy_area = row["policy_area"]
    subtopic_1 = row["subtopic_1"]
    subtopic_2 = row["subtopic_2"]
    cluster_length = row["cluster_length"]

    if subtopic_1 == "":
        subtopic_1 = "Uncategorized"
        no_second_node = True
    elif subtopic_2 == "":
        subtopic_2 = "Uncategorized"

    # Create a unique node for the policy_area if it doesn't exist yet
    if policy_area not in policy_nodes:
        policy_node_id = f"policy_{len(policy_nodes)}"
        policy_nodes[policy_area] = policy_node_id
        dot.node(policy_node_id, policy_area)
    else:
        policy_node_id = policy_nodes[policy_area]

    # Create a unique node for the subtopic_1 under the given policy_area
    key_subtopic1 = (policy_area, subtopic_1)
    if key_subtopic1 not in subtopic1_nodes:
        subtopic1_node_id = f"subtopic1_{len(subtopic1_nodes)}"
        subtopic1_nodes[key_subtopic1] = subtopic1_node_id
        dot.node(subtopic1_node_id, subtopic_1)
        # Connect the policy_area node to the subtopic_1 node
        dot.edge(policy_node_id, subtopic1_node_id)
    else:
        subtopic1_node_id = subtopic1_nodes[key_subtopic1]

    if no_second_node:
        subtopic2_node_id = subtopic1_node_id
    else:
        # Create a unique node for the subtopic_2 under the given subtopic_1
        key_subtopic2 = (policy_area, subtopic_1, subtopic_2)
        if key_subtopic2 not in subtopic2_nodes:
            subtopic2_node_id = f"subtopic2_{len(subtopic2_nodes)}"
            subtopic2_nodes[key_subtopic2] = subtopic2_node_id
            dot.node(subtopic2_node_id, subtopic_2)
            # Connect the subtopic_1 node to the subtopic_2 node
            dot.edge(subtopic1_node_id, subtopic2_node_id)
        else:
            subtopic2_node_id = subtopic2_nodes[key_subtopic2]

    # Create a leaf node for the cluster_length (unique per row)
    cluster_node_id = f"cluster_{idx}"
    cluster_label = f"cluster_length: {cluster_length}"
    dot.node(cluster_node_id, cluster_label)
    # Connect the subtopic_2 node to the cluster_length node
    dot.edge(subtopic2_node_id, cluster_node_id)

parent_node = dot.node("parent", "Parent Node")
for policy_node in policy_nodes.values():
    dot.edge("parent", policy_node)

# Render and view the resulting graph
dot.render("taxonomy_tree", view=False)

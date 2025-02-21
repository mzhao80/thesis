import pandas as pd
from graphviz import Digraph

# Load the CSV file
df = pd.read_csv("step_4.csv")

# Create a new directed graph
dot = Digraph(comment="Taxonomy Tree", format="png")

dot.attr(rankdir="LR")

# Dictionaries to store unique nodes
policy_nodes = {}
subtopic1_nodes = {}
subtopic2_nodes = {}

i = 0
j = 0
lengths = {}

# Iterate through each row in the DataFrame
for idx, row in df.iterrows():
    no_second_node = False
    policy_area = row["policy_area"]
    subtopic_1 = row["subtopic_1"]
    subtopic_2 = row["subtopic_2"]
    cluster_length = row["cluster_length"]

    if subtopic_1 == "Misc.":
        no_second_node = True

    # Create a unique node for the policy_area if it doesn't exist yet
    if policy_area not in policy_nodes:
        policy_node_id = f"policy_{len(policy_nodes)}"
        policy_nodes[policy_area] = policy_node_id
        # get the sum of cluster lengths for rows with policy_area
        lengths[policy_area] = df[df['policy_area'] == policy_area]['cluster_length'].sum()
        dot.node(policy_node_id, policy_area + f" ({lengths[policy_area]}, {lengths[policy_area]*100/df['cluster_length'].sum():.2f}%)")
    else:
        policy_node_id = policy_nodes[policy_area]

    # Create a unique node for the subtopic_1 under the given policy_area
    key_subtopic1 = (policy_area, subtopic_1)
    if key_subtopic1 not in subtopic1_nodes:
        subtopic1_node_id = f"subtopic1_{j}"
        j += 1
        subtopic1_nodes[key_subtopic1] = subtopic1_node_id
        lengths[key_subtopic1] = df[(df['policy_area'] == policy_area) & (df['subtopic_1'] == subtopic_1)]['cluster_length'].sum()
        dot.node(subtopic1_node_id, subtopic_1 + f" ({lengths[key_subtopic1]}, {lengths[key_subtopic1]*100/lengths[policy_area]:.2f}%)")
        # Connect the policy_area node to the subtopic_1 node
        dot.edge(policy_node_id, subtopic1_node_id)
    else:
        subtopic1_node_id = subtopic1_nodes[key_subtopic1]

    if not no_second_node:
        # Create a unique node for the subtopic_2 under the given subtopic_1
        key_subtopic2 = (policy_area, subtopic_1, subtopic_2)
        if key_subtopic2 not in subtopic2_nodes:
            subtopic2_node_id = f"subtopic2_{j}"
            j += 1
            subtopic2_nodes[key_subtopic2] = subtopic2_node_id
            lengths[key_subtopic2] = cluster_length
            dot.node(subtopic2_node_id, subtopic_2 + f" ({lengths[key_subtopic2]}, {lengths[key_subtopic2]*100/lengths[key_subtopic1]:.2f}%)")
            # Connect the subtopic_1 node to the subtopic_2 node
            dot.edge(subtopic1_node_id, subtopic2_node_id)
        else:
            subtopic2_node_id = subtopic2_nodes[key_subtopic2]
    else:
        subtopic2_node_id = subtopic1_node_id

parent_node = dot.node("parent", "Parent Node" + f" ({df['cluster_length'].sum()}, 100%)")
for policy_node in policy_nodes.values():
    dot.edge("parent", policy_node)

# Render and view the resulting graph
dot.render("taxonomy_tree", view=False)

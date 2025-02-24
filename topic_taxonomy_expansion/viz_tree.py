import pandas as pd
from graphviz import Digraph

# Load the CSV file
df = pd.read_csv("step_4.csv")

# Create a new directed graph
dot = Digraph(comment="Taxonomy Tree", format="png")

dot.attr(rankdir="LR")

# Dictionaries to store unique nodes and their counts
policy_nodes = {}
subtopic1_nodes = {}
subtopic2_nodes = {}
policy_counts = {}
subtopic1_counts = {}
subtopic2_counts = {}

i = 0
j = 0
lengths = {}

# First pass: Calculate all counts
for idx, row in df.iterrows():
    policy_area = row["policy_area"]
    subtopic_1 = row["subtopic_1"]
    subtopic_2 = row["subtopic_2"]
    cluster_length = row["cluster_length"]
    
    # Update policy area counts
    policy_counts[policy_area] = policy_counts.get(policy_area, 0) + cluster_length
    
    # Update subtopic1 counts
    key_subtopic1 = (policy_area, subtopic_1)
    subtopic1_counts[key_subtopic1] = subtopic1_counts.get(key_subtopic1, 0) + cluster_length
    
    # Update subtopic2 counts if not Misc.
    if subtopic_1 != "Misc.":
        key_subtopic2 = (policy_area, subtopic_1, subtopic_2)
        subtopic2_counts[key_subtopic2] = subtopic2_counts.get(key_subtopic2, 0) + cluster_length

# Sort policies by count
sorted_policies = sorted(policy_counts.items(), key=lambda x: x[1], reverse=True)

# Create nodes in sorted order
for policy_area, count in sorted_policies:
    policy_node_id = f"policy_{len(policy_nodes)}"
    policy_nodes[policy_area] = policy_node_id
    lengths[policy_area] = count
    dot.node(policy_node_id, policy_area + f" ({count}, {count*100/df['cluster_length'].sum():.2f}%)")
    
    # Sort subtopic1 nodes for this policy
    subtopic1_for_policy = [(key, count) for key, count in subtopic1_counts.items() if key[0] == policy_area]
    sorted_subtopic1 = sorted(subtopic1_for_policy, key=lambda x: x[1], reverse=True)
    
    for (_, subtopic_1), subtopic1_count in sorted_subtopic1:
        key_subtopic1 = (policy_area, subtopic_1)
        subtopic1_node_id = f"subtopic1_{j}"
        j += 1
        subtopic1_nodes[key_subtopic1] = subtopic1_node_id
        lengths[key_subtopic1] = subtopic1_count
        dot.node(subtopic1_node_id, subtopic_1 + f" ({subtopic1_count}, {subtopic1_count*100/lengths[policy_area]:.2f}%)")
        dot.edge(policy_node_id, subtopic1_node_id)
        
        if subtopic_1 != "Misc.":
            # Sort subtopic2 nodes for this subtopic1
            subtopic2_for_subtopic1 = [(key, count) for key, count in subtopic2_counts.items() 
                                     if key[0] == policy_area and key[1] == subtopic_1]
            sorted_subtopic2 = sorted(subtopic2_for_subtopic1, key=lambda x: x[1], reverse=True)
            
            for (_, _, subtopic_2), subtopic2_count in sorted_subtopic2:
                key_subtopic2 = (policy_area, subtopic_1, subtopic_2)
                subtopic2_node_id = f"subtopic2_{j}"
                j += 1
                subtopic2_nodes[key_subtopic2] = subtopic2_node_id
                lengths[key_subtopic2] = subtopic2_count
                dot.node(subtopic2_node_id, subtopic_2 + f" ({subtopic2_count}, {subtopic2_count*100/lengths[key_subtopic1]:.2f}%)")
                dot.edge(subtopic1_node_id, subtopic2_node_id)

parent_node = dot.node("parent", "Parent Node" + f" ({df['cluster_length'].sum()}, 100%)")
for policy_area, _ in sorted_policies:
    dot.edge("parent", policy_nodes[policy_area])

# Render and view the resulting graph
dot.render("taxonomy_tree", view=False)

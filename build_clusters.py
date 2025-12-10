import os
import pandas as pd
from collections import defaultdict, deque

INPUT_FOLDER = "./augmented_csvs"
OUTPUT_FILE = "./final/voter_list.csv"


def load_all_csvs(folder):
    frames = []
    for file in os.listdir(folder):
        if file.endswith(".csv"):
            path = os.path.join(folder, file)
            df = pd.read_csv(path)
            df["source_file"] = file  # helpful for debugging
            frames.append(df)

    if not frames:
        raise ValueError("No CSV files found in folder!")

    return pd.concat(frames, ignore_index=True)


def build_clusters(df):
    import sys
    sys.setrecursionlimit(10000)

    df = df.copy()

    # Normalize helper
    def norm(x):
        if x is None:
            return ""
        return str(x).strip().lower()

    df["cluster_id"] = None

    # Group by section
    sections = df.groupby(["section_no", "section_name"])

    cluster_counter = 1  # Global counter

    for (sec_no, sec_name), group in sections:

        local_indices = list(group.index)
        n = len(local_indices)

        # Adjacency list for hybrid family graph
        adj = {i: [] for i in range(n)}
        local_to_global = {i: local_indices[i] for i in range(n)}

        # Name → indices
        name_map = {}
        for i, global_idx in enumerate(local_indices):
            nm = norm(df.at[global_idx, "name"])
            name_map.setdefault(nm, []).append(i)

        # Build edges based on hybrid logic
        for i, global_idx in enumerate(local_indices):
            my_house = norm(df.at[global_idx, "house_no"])

            father = norm(df.at[global_idx, "father_name"])
            mother = norm(df.at[global_idx, "mother_name"])
            husband = norm(df.at[global_idx, "husband_name"])
            relations = [father, mother, husband]

            # Relationship edges ONLY when house_no matches
            for rel in relations:
                if rel and rel in name_map:
                    for j in name_map[rel]:
                        if i != j:
                            other_global = local_to_global[j]
                            # Hybrid rule: match only if same house_no
                            if norm(df.at[other_global, "house_no"]) == my_house and my_house != "":
                                adj[i].append(j)
                                adj[j].append(i)

        # DFS for connected components
        visited = set()

        def dfs(u, comp):
            visited.add(u)
            comp.append(u)
            for v in adj[u]:
                if v not in visited:
                    dfs(v, comp)

        # Assign clusters
        for i in range(n):
            if i not in visited:
                comp = []
                dfs(i, comp)

                # Isolated voter = cluster -1
                is_isolated = (len(comp) == 1 and len(adj[comp[0]]) == 0)

                if is_isolated:
                    df.at[local_to_global[comp[0]], "cluster_id"] = -1
                else:
                    for j in comp:
                        df.at[local_to_global[j], "cluster_id"] = cluster_counter
                    cluster_counter += 1

    return df




def main():
    print("📥 Loading CSVs from augmented_csvs/ ...")
    df = load_all_csvs(INPUT_FOLDER)
    print(f"✅ Loaded {len(df)} voters")

    print("🔍 Computing family clusters ...")
    df = build_clusters(df)
    print("✅ Cluster IDs assigned")

    print(f"💾 Saving final combined voter list → {OUTPUT_FILE}")
    df.to_csv(OUTPUT_FILE, index=False)

    print("\n🎉 DONE! Your Streamlit app can now read voter_list.csv")


if __name__ == "__main__":
    main()

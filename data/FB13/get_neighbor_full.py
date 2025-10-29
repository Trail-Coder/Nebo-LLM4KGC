import pandas as pd
import networkx as nx
import math
from collections import defaultdict
from typing import Dict, List, Tuple, Iterable, Set

tsv_file = 'train_en_short.tsv'
csv_file = 'entities_neighbors_full_en.csv'
entity_text_file = 'entity2text_capital.txt'
topk_second = 20
betweenness_sample_k = None
random_seed = 42
FUSION_METHOD = 'minmax'

def get_txt(filename: str, position_a: int, position_b: int) -> Dict[str, str]:
    mapping = {}
    try:
        with open(filename, "r", encoding="utf-8") as f:
            for line in f:
                tmp = line.rstrip("\n").split("\t")
                if len(tmp) > max(position_a, position_b):
                    mapping[str(tmp[position_a])] = str(tmp[position_b])
    except FileNotFoundError:
        pass
    return mapping

def safe_name(idx: str, ent2txt: Dict[str, str]) -> str:
    return ent2txt.get(idx, idx)

def build_directed_graph(df: pd.DataFrame) -> nx.DiGraph:
    DG = nx.DiGraph()
    DG.add_nodes_from(df['entity'].astype(str))
    DG.add_nodes_from(df['tail'].astype(str))
    DG.add_edges_from(zip(df['entity'].astype(str), df['tail'].astype(str)))
    return DG

def build_relation_index(df: pd.DataFrame):
    relations_dir = defaultdict(list)
    for _, row in df.iterrows():
        h = str(row['entity']); r = str(row['relation']); t = str(row['tail'])
        relations_dir[(h, t)].append(r)
    return relations_dir

def bridging_centrality_undirected(U: nx.Graph, k: int = None, seed: int = 42) -> Dict[str, float]:
    if k is None:
        btw = nx.betweenness_centrality(U, normalized=True)
    else:
        btw = nx.betweenness_centrality(U, k=k, normalized=True, seed=seed)
    deg = dict(U.degree())
    bc = {}
    for v in U.nodes():
        dv = deg.get(v, 0)
        if dv == 0:
            bc[v] = 0.0
            continue
        neighbor_degrees_sum = sum(deg.get(u, 0) for u in U.neighbors(v))
        bridging_coeff = neighbor_degrees_sum / float(dv)
        bc[v] = btw.get(v, 0.0) * bridging_coeff
    return bc

def adamic_adar_scores(U: nx.Graph, pairs: Iterable[Tuple[str, str]]) -> Dict[Tuple[str, str], float]:
    scores = {}
    for u, v, aa in nx.adamic_adar_index(U, pairs):
        scores[(u, v)] = aa
        scores[(v, u)] = aa
    return scores

def two_hop_targets_directed(DG: nx.DiGraph, s: str) -> Set[str]:
    first = set(DG.successors(s))
    second = set()
    for w in first:
        second.update(DG.successors(w))
    second.discard(s)
    second -= first
    return second

def best_bridge_for(DG: nx.DiGraph, U: nx.Graph, s: str, t: str, bc_undir: Dict[str, float]) -> str:
    commons = set(DG.successors(s)) & set(DG.predecessors(t))
    if not commons:
        return None
    degU = dict(U.degree())
    scored = []
    for w in commons:
        dw = max(2, degU.get(w, 0))
        aa_contrib = 1.0 / math.log(dw) if dw > 1 else 0.0
        scored.append((aa_contrib, bc_undir.get(w, 0.0), w))
    scored.sort(reverse=True)
    return scored[0][2]

def make_1hop_sentences_for(s: str, df: pd.DataFrame, ent2txt: Dict[str, str]) -> List[str]:
    name_s = safe_name(s, ent2txt)
    subset = df[df['entity'] == s]
    sentences = [f"The {rel} of {name_s} is {t}" for rel, t in zip(subset['relation'], subset['tail'])]
    seen, uniq = set(), []
    for sen in sentences:
        if sen not in seen:
            uniq.append(sen); seen.add(sen)
    return uniq

def pick_relation(relations_dir, h, t):
    arr = relations_dir.get((h, t))
    return arr[0] if arr else None

def make_2hop_sentence_pair(s: str, w: str, t: str, relations_dir, ent2txt: Dict[str, str]) -> str:
    r1 = pick_relation(relations_dir, s, w) or "relation"
    r2 = pick_relation(relations_dir, w, t) or "relation"
    sen1 = f"The {r1} of {safe_name(s, ent2txt)} is {safe_name(w, ent2txt)}"
    sen2 = f"The {r2} of {safe_name(w, ent2txt)} is {safe_name(t, ent2txt)}"
    return f"{sen1}; {sen2}."

def _minmax_norm(values: List[float]) -> List[float]:
    if not values:
        return []
    vmin, vmax = min(values), max(values)
    if vmax > vmin:
        return [(v - vmin) / (vmax - vmin) for v in values]
    return [0.5] * len(values)

def _rank_norm(values: List[float]) -> List[float]:
    if not values:
        return []
    uniq = sorted(set(values), reverse=True)
    rank_map = {v: i for i, v in enumerate(uniq)}
    if len(uniq) == 1:
        return [0.5] * len(values)
    max_rank = len(uniq) - 1
    return [(max_rank - rank_map[v]) / max_rank for v in values]

def rank_second_neighbors_directed_equal_weight(
    DG: nx.DiGraph,
    U: nx.Graph,
    s: str,
    bc_undir: Dict[str, float],
    topk: int,
    fusion: str = 'minmax'
) -> List[Tuple[str, float, float, float]]:
    cands = two_hop_targets_directed(DG, s)
    if not cands:
        return []
    pairs = [(s, t) for t in cands]
    aa_map = adamic_adar_scores(U, pairs)
    ts = list(cands)
    aa_vals = [aa_map.get((s, t), 0.0) for t in ts]
    bc_vals = [bc_undir.get(t, 0.0) for t in ts]
    if fusion == 'rank':
        aa_norm = _rank_norm(aa_vals)
        bc_norm = _rank_norm(bc_vals)
    else:
        aa_norm = _minmax_norm(aa_vals)
        bc_norm = _minmax_norm(bc_vals)
    combined = [0.5 * (a + b) for a, b in zip(aa_norm, bc_norm)]
    scored = [(t, aa, bc, c) for t, aa, bc, c in zip(ts, aa_vals, bc_vals, combined)]
    scored.sort(key=lambda x: (x[3], x[1], x[2]), reverse=True)
    return scored[:topk]

def main():
    ent2txt = get_txt(entity_text_file, 0, 1)
    df = pd.read_csv(tsv_file, sep='\t', header=None, names=['entity', 'relation', 'tail'])
    df['entity'] = df['entity'].astype(str)
    df['relation'] = df['relation'].astype(str)
    df['tail'] = df['tail'].astype(str)
    DG = build_directed_graph(df)
    U = DG.to_undirected()
    relations_dir = build_relation_index(df)
    bc_undir = bridging_centrality_undirected(U, k=betweenness_sample_k, seed=random_seed)
    all_nodes = sorted(set(df['entity']).union(set(df['tail'])))
    rows = []
    for s in all_nodes:
        one_hop_sents = make_1hop_sentences_for(s, df, ent2txt)
        one_hop_str = ", ".join(one_hop_sents)
        top2 = rank_second_neighbors_directed_equal_weight(
            DG, U, s, bc_undir, topk_second, fusion=FUSION_METHOD
        )
        two_hop_sents, two_hop_scores = [], []
        for (t, aa, bc_t, comb) in top2:
            w = best_bridge_for(DG, U, s, t, bc_undir)
            if w is None:
                continue
            sent = make_2hop_sentence_pair(s, w, t, relations_dir, ent2txt)
            two_hop_sents.append(sent)
            two_hop_scores.append(
                f"{t}|aa={aa:.6f}|bc_t={bc_t:.6f}|combined={comb:.6f}|via={w}"
            )
        two_hop_str = " || ".join(two_hop_sents)
        scores_str = ", ".join(two_hop_scores)
        rows.append({
            "entity": s,
            "entity_name": safe_name(s, ent2txt),
            "neighbors_1hop_sentences": one_hop_str,
            "neighbors_2hop_topk_sentences": two_hop_str,
            "second_neighbors_scores": scores_str
        })
    out_df = pd.DataFrame(rows, columns=[
        "entity", "entity_name",
        "neighbors_1hop_sentences",
        "neighbors_2hop_topk_sentences",
        "second_neighbors_scores"
    ])
    out_df.to_csv(csv_file, header=True, sep="\t", index=False, encoding='utf-8')
    print(
        f"CSV '{csv_file}' created (DIRECTED KG): 1-hop sentences + top-{topk_second} 2-hop path sentences "
        f"ranked by equal-weight fusion [{FUSION_METHOD}]."
    )

if __name__ == "__main__":
    main()

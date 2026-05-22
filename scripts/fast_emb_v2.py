#!/usr/bin/env python3
"""Fast diverse embedding: sequential per (method, topo), parallel SA within."""
import pickle, numpy as np, sys, glob, time, os
import networkx as nx
import dimod
import dwave_networkx as dnx
import minorminer
from dwave.embedding.pegasus import find_clique_embedding as peg_clique
from dwave.embedding.zephyr import find_clique_embedding as zep_clique
sys.path.insert(0, "src")
from data.generate import (build_physical_hamiltonian, compute_rms_J,
    sa_energy_curve, build_instance, make_grid)

GRID = make_grid()
READS = 100; SWEEPS = 200
TOPO = {'P4':('pegasus',4),'P8':('pegasus',8),'P16':('pegasus',16),'Z4':('zephyr',4)}

def get_hw(t):
    tp,m = TOPO[t]
    return dnx.pegasus_graph(m) if tp=='pegasus' else dnx.zephyr_graph(m)

# Load test instances
print("Loading...")
chunks = []
for c in sorted(glob.glob("data/diverse_sa_mt/chunk_*.pkl")):
    with open(c, "rb") as f: chunks.extend(pickle.load(f))
test = chunks[int(len(chunks)*0.84):]
from collections import defaultdict
by_fam = defaultdict(list)
for i in test: by_fam[i.get('family','?')].append(i)
selected = []
for fam, insts in by_fam.items(): selected.extend(insts[:1000])
print(f"Selected {len(selected)} instances")

configs = [
    ('tuned', ['P4','P8','P16','Z4'], None),
    ('clique', ['P4','P8','P16','Z4'], {'sk_model','weighted_maxcut'}),
    ('clique_init', ['P4','P8','P16','Z4'], {'sk_model','weighted_maxcut'}),
]

for method, topos, fam_filter in configs:
    for topo in topos:
        outdir = f"data/emb_{method}_sa_mt"
        os.makedirs(outdir, exist_ok=True)
        outpath = f"{outdir}/chunk_{topo}.pkl"
        if os.path.exists(outpath):
            print(f"\n[{method}/{topo}] SKIP exists")
            continue

        hw = get_hw(topo)
        pool = selected
        if fam_filter:
            pool = [i for i in selected if i.get('family') in fam_filter and i.get('n_logical',0)<=20]

        print(f"\n[{method}/{topo}] {len(pool)} instances...")
        results = []
        t0 = time.time()

        for idx, inst in enumerate(pool):
            h = inst['h_logical']; J = inst['J_logical']
            n = inst.get('n_logical',0); fam = inst.get('family','')
            seed = 42 + idx

            # Embed
            try:
                if method == 'tuned':
                    bqm = dimod.BinaryQuadraticModel(h, J, 0., dimod.SPIN)
                    src = nx.Graph(); src.add_nodes_from(bqm.variables); src.add_edges_from(bqm.quadratic.keys())
                    emb = minorminer.find_embedding(src, hw, random_seed=seed, timeout=30,
                                                     max_no_improvement=50, tries=10)
                elif method == 'clique':
                    tp,m = TOPO[topo]
                    emb = peg_clique(n,m=m) if tp=='pegasus' else zep_clique(n,m=m)
                    emb = {k:list(v) for k,v in emb.items()} if emb else None
                elif method == 'clique_init':
                    tp,m = TOPO[topo]
                    ce = peg_clique(n,m=m) if tp=='pegasus' else zep_clique(n,m=m)
                    if ce:
                        ce = {k:list(v) for k,v in ce.items()}
                        bqm = dimod.BinaryQuadraticModel(h, J, 0., dimod.SPIN)
                        src = nx.Graph(); src.add_nodes_from(bqm.variables); src.add_edges_from(bqm.quadratic.keys())
                        emb = minorminer.find_embedding(src, hw, random_seed=seed, timeout=30, initial_chains=ce)
                    else: emb = None
            except: emb = None

            if not emb or any(len(c)==0 for c in emb.values()):
                continue

            # Build + SA label
            try:
                h_p, J_p, cem, cef = build_physical_hamiltonian(h, J, emb, hw)
                bqm = dimod.BinaryQuadraticModel(h, J, 0., dimod.SPIN)
                sa = sa_energy_curve(bqm, emb, hw, GRID, n_reads=READS, sweeps=SWEEPS, seed=seed)
                out = build_instance(h, J, emb, hw, cem, cef, h_p, J_p, sa, fam, idx, topo, n)
                out['embedding_method'] = method
                results.append(out)
            except: continue

            if len(results) % 500 == 0 and len(results) > 0:
                with open(outpath, 'wb') as f: pickle.dump(results, f)
                el = time.time()-t0
                print(f"  {len(results)} saved ({el:.0f}s, {el/len(results):.1f}s/inst)")

        with open(outpath, 'wb') as f: pickle.dump(results, f)
        print(f"  [{method}/{topo}] DONE: {len(results)} in {time.time()-t0:.0f}s")

print(f"\n{'='*50}\nALL DONE")
for d in sorted(glob.glob("data/emb_*/chunk_*.pkl")):
    with open(d,'rb') as f: print(f"  {d}: {len(pickle.load(f))}")

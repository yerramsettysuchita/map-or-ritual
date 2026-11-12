import random, numpy as np, networkx as nx, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"], "mathtext.fontset": "stix"})
NAVY, GRAY, INK = "#1f3b73", "#9a9a9a", "#111111"
rng = random.Random(20261108)
G = nx.powerlaw_cluster_graph(26, 2, 0.6, seed=7)
pos = nx.kamada_kawai_layout(G)
R = G.copy()
nx.double_edge_swap(R, nswap=10 * R.number_of_edges(), max_tries=10000, seed=20261108)
assert dict(G.degree()) == dict(R.degree())
fig = plt.figure(figsize=(5.6, 2.8), dpi=100)
fig.patch.set_facecolor("white")
fig.text(0.05, 0.80, "Map or Ritual?", fontsize=22, color=INK, ha="left", va="center")
fig.text(0.05, 0.62, "Null-world tests of code-graph tools\nfor Gemma 4 coding agents",
         fontsize=10.5, color="#333333", ha="left", va="center", linespacing=1.3)
fig.text(0.05, 0.30, "Same tools, same nodes, same degrees.\nOnly the call edges are scrambled.",
         fontsize=8.5, color="#555555", ha="left", va="center", linespacing=1.35, style="italic")
for k, (H, col, lab) in enumerate(((G, NAVY, "real"), (R, GRAY, "rewired"))):
    ax = fig.add_axes([0.53 + k * 0.235, 0.16, 0.22, 0.70])
    ax.set_axis_off()
    nx.draw_networkx_edges(H, pos, ax=ax, edge_color=col, width=0.7, alpha=0.9)
    nx.draw_networkx_nodes(H, pos, ax=ax, node_size=[10 + 6 * H.degree(n) for n in H], node_color="white",
                           edgecolors=INK, linewidths=0.6)
    ax.set_title(lab, fontsize=9, color=INK, pad=2)
fig.savefig("./card_1120x560.png", dpi=200)
fig.savefig("./card_560x280.png", dpi=100)
from PIL import Image
for f in ("card_1120x560", "card_560x280"):
    print(f, Image.open(f"./{f}.png").size)

# Copyright 2026 InnovativeOrbits. Licensed under the Apache License, Version 2.0.
# Map or Ritual? Release notebook, part A (CPU).
# Rebuilds the gold labels and the three graph worlds from the competition
# input and checks them against the hashes fixed before the confirmatory runs.
# Everything is written to /tmp. No competition data or derived data is saved
# to the notebook output. Only counts and hashes are printed.
import os, json, glob, re, collections, hashlib, random, time
import numpy as np

EXPECTED_LABELS = "a0ceb7ebce446eec"
EXPECTED_WORLDS = "5b88c0e967618d8b"
SEED = 20261108
BASE = os.environ.get("MOR_BASE", "/kaggle/input/competitions/gemma-4-developer-agent")
if not os.path.exists(f"{BASE}/tasks.jsonl"):
    hits = glob.glob("/kaggle/input/**/tasks.jsonl", recursive=True)
    assert hits, "Add the competition 'Gemma 4 Developer Agent' as Input first"
    BASE = os.path.dirname(hits[0])
OUT = os.environ.get("MOR_OUT", "/tmp/maporritual")
os.makedirs(OUT, exist_ok=True)
T0 = time.time()

tasks = [json.loads(l) for l in open(f"{BASE}/tasks.jsonl")]
gfiles = {os.path.basename(p)[:-5]: p for p in glob.glob(f"{BASE}/graphs/*.json")}
Efiles = {os.path.basename(p)[:-4]: p for p in glob.glob(f"{BASE}/embeddings/*.npz")}
print("tasks:", len(tasks), "| graphs:", len(gfiles), "| embedding files:", len(Efiles))

def graph_path(t):
    for k, p in gfiles.items():
        if k.endswith(t["base_commit"]):
            return p
    return None

_cache = {}
def load_graph(p):
    if p not in _cache:
        with open(p) as f:
            _cache[p] = json.load(f)
    return _cache[p]

# ---------------- 1. patch parser ----------------
def parse_patch(patch):
    files, cur, old = [], None, None
    for line in patch.splitlines():
        if line.startswith("--- a/") or line.startswith("--- /dev/null"):
            old = line[4:].strip()
        elif line.startswith("+++ b/") or line.startswith("+++ /dev/null"):
            cur = {"old": old, "new": line[4:].strip(), "hunks": []}
            files.append(cur)
        elif line.startswith("@@") and cur is not None:
            m = re.match(r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@(.*)", line)
            cur["hunks"].append({"header": (m.group(5) or "").strip(),
                                 "minus": [], "ctx": []})
        elif cur is not None and cur["hunks"]:
            h = cur["hunks"][-1]
            if line.startswith("-"):
                h["minus"].append(line[1:])
            elif line.startswith(" "):
                h["ctx"].append(line[1:])
    return files

def path_of(f):
    p = f["new"] if f["new"] != "/dev/null" else f["old"]
    return p[2:] if p.startswith(("a/", "b/")) else p

def is_test_path(p):
    pl = p.lower()
    return "test" in pl.split("/")[-1] or "/tests/" in "/" + pl or pl.startswith("tests/")

def candidates_for_file(nodes, fpath):
    mod = fpath[:-3].replace("/", ".")
    parts = mod.split(".")
    for i in range(len(parts)):          # try longest module suffix first
        suf = ".".join(parts[i:])
        c = [n for n in nodes if n["id"] == suf or n["id"].startswith(suf + ".")]
        if c:
            return c, suf
    return [], None

# ---------------- 2. gold labeler v1 ----------------
HDR = re.compile(r"(?:async\s+def|def|class)\s+([A-Za-z_]\w*)")
TOK = re.compile(r"[A-Za-z_][A-Za-z0-9_]{3,}")
GENERIC = {"self", "None", "True", "False", "return", "import",
           "from", "class", "async", "await", "print", "value",
           "name", "data", "type", "list", "dict", "test", "json",
           "text", "file", "path", "args", "kwargs", "result"}

_adj = {}
def adj_of(gp):
    if gp not in _adj:
        a = collections.defaultdict(set)
        for e in load_graph(gp)["edges"]:
            u, v = e["source"], e["target"]
            if u != v:
                a[u].add(v)
                a[v].add(u)
        _adj[gp] = a
    return _adj[gp]

def short(nid):
    return nid.split(".")[-1]

def bfs_dist(adj, sources, targets, max_d=6):
    targets = set(targets)
    if targets & sources:
        return 0
    seen = set(sources)
    frontier = list(sources)
    for d in range(1, max_d + 1):
        nxt = []
        for u in frontier:
            for v in adj.get(u, ()):
                if v in targets:
                    return d
                if v not in seen:
                    seen.add(v)
                    nxt.append(v)
        frontier = nxt
        if not frontier:
            break
    return None

def match_hunk(cands, h):
    probes = [l.strip() for l in h["minus"] + h["ctx"]
              if len(l.strip()) >= 8]
    best, best_s = [], 0
    for n in cands:
        txt = n.get("text") or ""
        s = sum(1 for pr in probes if pr in txt)
        if s > best_s:
            best, best_s = [n], s
        elif s == best_s and s > 0:
            best.append(n)
    if best_s > 0:
        key = lambda n: len(n.get("text") or "")
        return min(best, key=key)["id"], "TEXT"
    m = HDR.search(h["header"])
    if not m:
        if not [l for l in h["minus"] if l.strip()]:
            return None, "ADD_NO_HEADER"
        return None, "MODULE_LEVEL"
    nm = m.group(1)
    hits = [n for n in cands if short(n["id"]) == nm]
    if len(hits) == 1:
        return hits[0]["id"], "HEADER"
    if hits:
        return None, "HEADER_AMBIGUOUS"
    if "async def" in h["header"]:
        return None, "ASYNC_NOT_IN_GRAPH"
    return None, "HEADER_NOT_IN_GRAPH"

def label_task_v1(t):
    nodes = load_graph(graph_path(t))["nodes"]
    gold, how, files = set(), collections.Counter(), []
    for f in parse_patch(t["patch"]):
        p = path_of(f)
        if not p.endswith(".py") or is_test_path(p):
            continue
        files.append(p)
        if f["old"] == "/dev/null":
            how["NEW_FILE"] += 1
            continue
        cands, _ = candidates_for_file(nodes, p)
        if not cands:
            how["NO_MODULE_MATCH"] += 1
            continue
        for h in f["hunks"]:
            nid, tag = match_hunk(cands, h)
            how[tag] += 1
            if nid:
                gold.add(nid)
    return gold, how, files

rows1 = []
for t in tasks:
    gp = graph_path(t)
    adj = adj_of(gp)
    gold, how, files = label_task_v1(t)
    toks = set(TOK.findall(t["problem_statement"]))
    in_issue = any(short(g) in toks for g in gold)
    anchors = set()
    for n in load_graph(gp)["nodes"]:
        s = short(n["id"])
        if s in toks and s not in GENERIC and not s.startswith("__"):
            anchors.add(n["id"])
    hop = None
    if gold and anchors:
        hop = bfs_dist(adj, anchors, gold)
    unc = how["ASYNC_NOT_IN_GRAPH"] + how["HEADER_NOT_IN_GRAPH"]
    gdeg = [len(adj.get(g, ())) for g in gold]
    r = {"id": t["instance_id"], "repo": t["repo"],
         "gold": sorted(gold), "n_files": len(set(files)),
         "how": dict(how), "name_in_issue": in_issue,
         "n_anchors": len(anchors), "hop": hop,
         "gold_isolated": sum(1 for d in gdeg if d == 0),
         "uncovered_hunks": unc,
         "async_uncovered": how["ASYNC_NOT_IN_GRAPH"],
         "broad_refactor": len(set(files)) > 8}
    r["primary"] = (bool(gold) and not in_issue
                    and unc == 0 and not r["broad_refactor"])
    rows1.append(r)

prim = [r for r in rows1 if r["primary"]]
tags = collections.Counter()
for r in rows1:
    tags.update(r["how"])
print("\n=== LABELS ===")
print("labeled tasks:", sum(1 for r in rows1 if r["gold"]), "of", len(rows1))
print("hunk outcomes:", tags.most_common())
print("primary tasks:", len(prim),
      collections.Counter(r["repo"] for r in prim).most_common())
print("primary gold nodes:", sum(len(r["gold"]) for r in prim))
hops = collections.Counter(
    "no_anchor" if r["n_anchors"] == 0 else
    ("unreachable" if r["hop"] is None else r["hop"])
    for r in prim)
print("hop from issue anchors:", sorted(hops.items(), key=lambda kv: str(kv[0])))
blob = json.dumps(rows1, sort_keys=True).encode()
with open(f"{OUT}/labels_v1.json", "wb") as f:
    f.write(blob)
LABELS_HASH = hashlib.sha256(blob).hexdigest()[:16]
print("labels sha256:", LABELS_HASH, "| seconds:", round(time.time() - T0))

# ---------------- 3. GraphNull worlds (W1 real, W2 rewired, W3 permuted) ----------------
def name_seed(name):
    h = hashlib.sha256(f"{SEED}:{name}".encode()).hexdigest()
    return int(h[:12], 16)

def normalize(G):
    pairs, loops = set(), set()
    for e in G["edges"]:
        u, v = e["source"], e["target"]
        if u == v:
            loops.add(u)
        else:
            pairs.add((u, v) if u < v else (v, u))
    return pairs, loops

def swap(pairs, seed, mult=10):
    rng = random.Random(seed)
    E = sorted(pairs)
    S = set(E)
    m = len(E)
    done, tries = 0, 0
    while done < mult * m and tries < mult * m * 50:
        tries += 1
        i = int(rng.random() * m)
        j = int(rng.random() * m)
        if i == j:
            continue
        a, b = E[i]
        c, d = E[j]
        if rng.random() < 0.5:
            c, d = d, c
        if len({a, b, c, d}) < 4:
            continue
        e1 = (a, d) if a < d else (d, a)
        e2 = (c, b) if c < b else (b, c)
        if e1 in S or e2 in S:
            continue
        S.discard(E[i])
        S.discard(E[j])
        S.add(e1)
        S.add(e2)
        E[i], E[j] = e1, e2
        done += 1
    return set(E), done / max(mult * m, 1)

def degrees(pairs):
    d = collections.Counter()
    for u, v in pairs:
        d[u] += 1
        d[v] += 1
    return d

def emit(G, pairs, loops):
    edges = []
    for u, v in sorted(pairs):
        edges.append({"source": u, "target": v, "type": "calls"})
        edges.append({"source": v, "target": u, "type": "calls"})
    for u in sorted(loops):
        edges.append({"source": u, "target": u, "type": "calls"})
    out = {k: v for k, v in G.items() if k not in ("edges", "links")}
    out["edges"] = edges
    return out

def write_json(obj, path):
    blob = json.dumps(obj, sort_keys=True).encode()
    with open(path, "wb") as f:
        f.write(blob)
    return hashlib.sha256(blob).hexdigest()

def file_hash(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()

W = f"{OUT}/worlds"
for w in ("W1", "W2", "W3"):
    for sub in ("graphs", "embeddings"):
        os.makedirs(f"{W}/{w}/{sub}", exist_ok=True)

manifest = {"seed": SEED, "files": {}}
changed, swap_ok, deg_ok, emb_ok = [], [], 0, 0
raw_e, norm_e = 0, 0
names = sorted(gfiles)
for k, name in enumerate(names):
    G = load_graph(gfiles[name])
    pairs, loops = normalize(G)
    raw_e += len(G["edges"])
    norm_e += 2 * len(pairs) + len(loops)
    s = name_seed(name)
    rpairs, frac = swap(pairs, s)
    swap_ok.append(frac)
    deg_ok += degrees(pairs) == degrees(rpairs)
    changed.append(1 - len(pairs & rpairs) / max(len(pairs), 1))

    g1 = f"{W}/W1/graphs/{name}.json"
    g2 = f"{W}/W2/graphs/{name}.json"
    g3 = f"{W}/W3/graphs/{name}.json"
    h1 = write_json(emit(G, pairs, loops), g1)
    h2 = write_json(emit(G, rpairs, loops), g2)
    if not os.path.lexists(g3):
        os.symlink(g1, g3)

    src = Efiles[name]
    for w in ("W1", "W2"):
        dst = f"{W}/{w}/embeddings/{name}.npz"
        if not os.path.lexists(dst):
            os.symlink(src, dst)
    E = np.load(src)
    keys = list(E.keys())
    M = np.stack([E[x] for x in keys])
    perm = np.random.default_rng(s).permutation(len(keys))
    Mp = M[perm]
    dst3 = f"{W}/W3/embeddings/{name}.npz"
    np.savez(dst3, **{x: Mp[i] for i, x in enumerate(keys)})
    E3 = np.load(dst3)
    emb_ok += (sorted(E3.keys()) == sorted(keys)
               and E3[keys[0]].dtype == M.dtype)

    manifest["files"][name] = {"W1_graph": h1, "W2_graph": h2,
                               "emb": file_hash(src),
                               "W3_emb": file_hash(dst3)}
    if k % 25 == 0:
        print(f"  world {k + 1}/{len(names)} | seconds: {round(time.time() - T0)}")

mblob = json.dumps(manifest, sort_keys=True).encode()
with open(f"{W}/manifest.json", "wb") as f:
    f.write(mblob)
WORLDS_HASH = hashlib.sha256(mblob).hexdigest()[:16]

print("\n=== GRAPHNULL WORLDS ===")
print("graphs processed:", len(names))
print("raw edges total:", raw_e, "| normalized:", norm_e)
print("degree sequence preserved (W2 vs W1):", deg_ok, "/", len(names))
print("swaps completed share  min/median:",
      round(min(swap_ok), 3), round(float(np.median(swap_ok)), 3))
print("edges changed by rewiring  min/median:",
      round(min(changed), 3), round(float(np.median(changed)), 3))
print("W3 embeddings same keys and dtype:", emb_ok, "/", len(names))
print("worlds manifest sha256:", WORLDS_HASH)

# ---------------- 4. verdict ----------------
print("\n=== HASH CHECK ===")
print("labels:", LABELS_HASH, "expected", EXPECTED_LABELS,
      "->", "MATCH" if LABELS_HASH == EXPECTED_LABELS else "DIFFERENT")
print("worlds:", WORLDS_HASH, "expected", EXPECTED_WORLDS,
      "->", "MATCH" if WORLDS_HASH == EXPECTED_WORLDS else "DIFFERENT")
print("files saved to notebook output:", len(os.listdir("/kaggle/working")) if os.path.isdir("/kaggle/working") else "n/a")
print("total seconds:", round(time.time() - T0))
print("RELEASE PART A DONE")

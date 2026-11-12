# Copyright 2026 InnovativeOrbits. Licensed under the Apache License, Version 2.0.
# GraphNull held-out check. CPU only. Uses three public repositories that are
# not part of the Gemma 4 Developer Agent competition data. No competition
# data is read or written by this notebook.
import ast, collections, hashlib, io, json, os, random, re, subprocess, tarfile, time
import numpy as np

ON_KAGGLE = os.path.isdir("/kaggle/working")
WORK = "/tmp/heldout" if ON_KAGGLE else "./heldout_work"
RESULTS = "/kaggle/working/heldout_results.json" if ON_KAGGLE else "heldout_results.json"
REPOS = [  # (name, url, pinned commit)
    ("click", "https://github.com/pallets/click.git", "8abb77a8432e03ed14a95f972b44f905279415a7"),
    ("flask", "https://github.com/pallets/flask.git", "49943b5a94413a530761350f04e896f76e8d620e"),
    ("attrs", "https://github.com/python-attrs/attrs.git", "9d9f94a043de7fc4012459853a4ed597a5577855"),
]
SEED = 20261108        # same seed as the main study
N_COMMITS = 400        # most recent non-merge commits scanned per repository
N_NULLS = 20           # rewired graphs per task
N_BOOT = 10000
os.makedirs(WORK, exist_ok=True)

def git(repo, *args, binary=False):
    out = subprocess.run(["git", "-C", repo, *args], capture_output=True, check=True).stdout
    return out if binary else out.decode("utf-8", "replace")

# ---------------- 1. call graph builder (static, approximate) ----------------
SKIP_DIRS = ("tests/", "test/", "docs/", "examples/", "benchmarks/", "bench/")
BUILTIN_METHODS = set()
for _t in (list, dict, str, set, tuple, bytes, int, float, object):
    BUILTIN_METHODS |= set(dir(_t))

def is_source(path):
    p = path.lower()
    if not p.endswith(".py"):
        return False
    if any(("/" + p).find("/" + d) >= 0 for d in SKIP_DIRS):
        return False
    base = p.split("/")[-1]
    return not (base.startswith("test_") or base.endswith("_test.py")
                or base in ("conftest.py", "setup.py", "noxfile.py"))

def module_of(path):
    m = path[:-3].replace("/", ".")
    return m[4:] if m.startswith("src.") else m

DEFS = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)

def index_file(path, text):
    """Return list of (node_id, kind, first_line, last_line, ast_node, class_id)."""
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError):
        return []
    out = []
    def walk(node, prefix, cls):
        for ch in ast.iter_child_nodes(node):
            if isinstance(ch, DEFS):
                nid = prefix + "." + ch.name
                kind = "class" if isinstance(ch, ast.ClassDef) else "function"
                first = min([ch.lineno] + [d.lineno for d in ch.decorator_list])
                out.append((nid, kind, first, ch.end_lineno, ch, cls))
                walk(ch, nid, nid if kind == "class" else cls)
            else:
                walk(ch, prefix, cls)
    walk(tree, module_of(path), None)
    return out

def own_calls(fn):
    """Names called directly inside fn, not inside nested defs."""
    calls = []
    stack = list(ast.iter_child_nodes(fn))
    while stack:
        n = stack.pop()
        if isinstance(n, DEFS):
            continue
        if isinstance(n, ast.Call):
            f = n.func
            if isinstance(f, ast.Name):
                calls.append(("name", f.id))
            elif isinstance(f, ast.Attribute):
                on_self = isinstance(f.value, ast.Name) and f.value.id in ("self", "cls")
                calls.append(("self" if on_self else "attr", f.attr))
        stack.extend(ast.iter_child_nodes(n))
    return calls

def build_graph(files):
    """files: dict path -> source text. Returns (node ids, undirected pair set, spans)."""
    entries, spans = [], {}
    for path in sorted(files):
        rows = index_file(path, files[path])
        spans[path] = [(a, b, nid) for nid, _, a, b, _, _ in rows]
        entries += [(path, r) for r in rows]
    by_short = collections.defaultdict(list)
    for path, (nid, kind, a, b, node, cls) in entries:
        by_short[nid.split(".")[-1]].append((nid, module_of(path)))
    ids = sorted({r[0] for _, r in entries})
    idset = set(ids)
    pairs = set()
    for path, (nid, kind, a, b, node, cls) in entries:
        if kind != "function":
            continue
        mod = module_of(path)
        for how, name in own_calls(node):
            target = None
            if how == "self" and cls and (cls + "." + name) in idset:
                target = cls + "." + name
            else:
                cands = by_short.get(name, [])
                same = [c for c, m in cands if m == mod]
                if how == "name":
                    if len(same) == 1:
                        target = same[0]
                    elif not same and len(cands) == 1:
                        target = cands[0][0]
                elif len(name) >= 4 and name not in BUILTIN_METHODS and len(cands) == 1:
                    target = cands[0][0]
            if target and target != nid:
                pairs.add((nid, target) if nid < target else (target, nid))
    return ids, pairs, spans

# ---------------- 2. GraphNull rewiring (identical to the main study) ----------------
def name_seed(name):
    h = hashlib.sha256(f"{SEED}:{name}".encode()).hexdigest()
    return int(h[:12], 16)

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

def adjacency(pairs):
    a = collections.defaultdict(set)
    for u, v in pairs:
        a[u].add(v)
        a[v].add(u)
    return a

# ---------------- 3. tasks from commit history ----------------
HUNK = re.compile(r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")

def changed_old_lines(repo, parent, commit):
    """dict path -> set of old-side line numbers touched (modified files only)."""
    diff = git(repo, "diff", "-U0", "--no-renames", "--diff-filter=M", parent, commit, "--", "*.py")
    out, cur = collections.defaultdict(set), None
    for line in diff.splitlines():
        if line.startswith("--- a/"):
            cur = line[6:]
        elif line.startswith("@@") and cur:
            m = HUNK.match(line)
            start, n = int(m.group(1)), int(m.group(2) or 1)
            out[cur].update(range(start, start + n) if n else [start])
    return out

def source_at(repo, commit):
    blob = git(repo, "archive", "--format=tar", commit, binary=True)
    files = {}
    with tarfile.open(fileobj=io.BytesIO(blob)) as tf:
        for m in tf.getmembers():
            if m.isfile() and is_source(m.name):
                files[m.name] = tf.extractfile(m).read().decode("utf-8", "replace")
    return files

def gold_nodes(spans, touched):
    gold = set()
    for path, lines in touched.items():
        for ln in lines:
            inner = None
            for a, b, nid in spans.get(path, ()):
                if a <= ln <= b and (inner is None or b - a < inner[0]):
                    inner = (b - a, nid)
            if inner:
                gold.add(inner[1])
    return gold

def recall(adj, gold, hops):
    """Share of edited functions that have another edited function within `hops`."""
    hit = 0
    for g in gold:
        seen, frontier = {g}, {g}
        for _ in range(hops):
            frontier = {v for u in frontier for v in adj.get(u, ())} - seen
            seen |= frontier
        hit += bool((seen - {g}) & gold)
    return hit / len(gold)

def boot_ci(x, n=N_BOOT, seed=SEED):
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, len(x), size=(n, len(x)))].mean(axis=1)
    return float(x.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))

# ---------------- 4. run ----------------
t0 = time.time()
rows, inv = [], collections.defaultdict(list)
for name, url, pin in REPOS:
    repo = f"{WORK}/{name}"
    if not os.path.isdir(repo):
        subprocess.run(["git", "clone", "-q", url, repo], check=True)
    commits = git(repo, "log", "--no-merges", "--format=%H %P", "-n", str(N_COMMITS), pin, "--", "*.py").split("\n")
    kept = 0
    for line in commits:
        parts = line.split()
        if len(parts) != 2:
            continue
        commit, parent = parts
        touched = {p: l for p, l in changed_old_lines(repo, parent, commit).items() if is_source(p)}
        if not touched or len(touched) > 8:
            continue
        ids, pairs, spans = build_graph(source_at(repo, parent))
        gold = gold_nodes(spans, touched)
        if not (2 <= len(gold) <= 10) or len(pairs) < 50:
            continue
        adj1 = adjacency(pairs)
        deg1 = degrees(pairs)
        r1 = {h: recall(adj1, gold, h) for h in (1, 2)}
        r2 = {1: [], 2: []}
        for k in range(N_NULLS):
            rp, frac = swap(pairs, name_seed(f"{name}:{commit}:{k}"))
            inv["degree_preserved"].append(degrees(rp) == deg1)
            inv["swaps_done"].append(frac)
            inv["edges_changed"].append(1 - len(pairs & rp) / len(pairs))
            adj2 = adjacency(rp)
            for h in (1, 2):
                r2[h].append(recall(adj2, gold, h))
        rows.append({"repo": name, "commit": commit[:12], "nodes": len(ids), "edges": len(pairs),
                     "gold": len(gold),
                     "list_size": float(np.mean([deg1[g] for g in gold])),
                     "real_1": r1[1], "null_1": float(np.mean(r2[1])),
                     "real_2": r1[2], "null_2": float(np.mean(r2[2]))})
        kept += 1
    print(f"{name}: scanned {len(commits)} commits, kept {kept} tasks | {round(time.time() - t0)} s")

# ---------------- 5. report ----------------
def fmt(t):
    return f"{100 * t[0]:5.1f}  [{100 * t[1]:5.1f}, {100 * t[2]:5.1f}]"

print("\n=== GRAPHNULL INVARIANTS ON HELD-OUT REPOSITORIES ===")
print("rewired graphs built:", len(inv["degree_preserved"]))
print("degree sequence preserved:", sum(inv["degree_preserved"]), "/", len(inv["degree_preserved"]))
print("swaps completed share, min:", round(min(inv["swaps_done"]), 3))
print("edges changed share, min / median:", round(min(inv["edges_changed"]), 3),
      "/", round(float(np.median(inv["edges_changed"])), 3))

print("\n=== CO-EDIT RECALL: does a neighbor list of one edited function contain another? ===")
print("values are percent, mean with task-bootstrap 95% CI")
groups = [(n, [r for r in rows if r["repo"] == n]) for n, _, _ in REPOS] + [("ALL", rows)]
for gname, rs in groups:
    if not rs:
        continue
    print(f"\n{gname}: tasks {len(rs)} | median nodes {int(np.median([r['nodes'] for r in rs]))}"
          f" | median edges {int(np.median([r['edges'] for r in rs]))}"
          f" | mean neighbor list size {np.mean([r['list_size'] for r in rs]):.1f} (same in both worlds)")
    for h in (1, 2):
        real = [r[f"real_{h}"] for r in rs]
        null = [r[f"null_{h}"] for r in rs]
        d = [a - b for a, b in zip(real, null)]
        print(f"  {h}-hop real   {fmt(boot_ci(real))}")
        print(f"  {h}-hop null   {fmt(boot_ci(null))}")
        print(f"  {h}-hop diff   {fmt(boot_ci(d))}")

out = {"seed": SEED, "repos": REPOS, "n_commits": N_COMMITS, "n_nulls": N_NULLS, "tasks": rows}
blob = json.dumps(out, sort_keys=True).encode()
with open(RESULTS, "wb") as f:
    f.write(blob)
print("\nresults sha256:", hashlib.sha256(blob).hexdigest()[:16])
print("total seconds:", round(time.time() - t0))
print("HELDOUT DONE")

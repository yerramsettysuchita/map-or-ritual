# Copyright 2026 InnovativeOrbits. Licensed under the Apache License, Version 2.0.
# Map or Ritual? Held-out task builder v2 (CPU, internet on, GitHub token).
# Builds SWE-bench style localization tasks from merged pull requests that
# close a GitHub issue, in three public repositories that are not in the
# competition data. The issue text is the problem statement and the
# squash-merge diff gives the gold functions. No competition data is used.
import ast, collections, hashlib, io, json, os, random, re, subprocess, tarfile, time
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=SyntaxWarning)

ON_KAGGLE = os.path.isdir("/kaggle/working")
CLONES = "/tmp/heldout_clones" if ON_KAGGLE else "./heldout_work"
OUT = "/kaggle/working/heldout_issues" if ON_KAGGLE else "./heldout_issues_out"
REPOS = [  # (short name, repo field, url, pinned commit)
    ("click", "pallets/click", "https://github.com/pallets/click.git",
     "8abb77a8432e03ed14a95f972b44f905279415a7"),
    ("flask", "pallets/flask", "https://github.com/pallets/flask.git",
     "49943b5a94413a530761350f04e896f76e8d620e"),
    ("attrs", "python-attrs/attrs", "https://github.com/python-attrs/attrs.git",
     "9d9f94a043de7fc4012459853a4ed597a5577855"),
]
SEED = 20261108
N_PRS = 1500
PER_REPO = 40
EMB_DIM = 256
for d in ("W1/graphs", "W2/graphs", "embeddings", "src"):
    os.makedirs(f"{OUT}/{d}", exist_ok=True)
os.makedirs(CLONES, exist_ok=True)
T0 = time.time()

def git(repo, *args, binary=False):
    out = subprocess.run(["git", "-C", repo, *args], capture_output=True, check=True).stdout
    return out if binary else out.decode("utf-8", "replace")

# ---------------- 1. source filter and call graph builder ----------------
SKIP_DIRS = ("tests/", "test/", "docs/", "examples/", "benchmarks/", "bench/")
BUILTIN_METHODS = set()
for _t in (list, dict, str, set, tuple, bytes, int, float, object):
    BUILTIN_METHODS |= set(dir(_t))
DEFS = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)

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

def index_file(path, text):
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError):
        return []
    lines = text.splitlines(True)
    out = []
    def walk(node, prefix, cls):
        for ch in ast.iter_child_nodes(node):
            if isinstance(ch, DEFS):
                nid = prefix + "." + ch.name
                kind = "class" if isinstance(ch, ast.ClassDef) else "function"
                first = min([ch.lineno] + [d.lineno for d in ch.decorator_list])
                src = "".join(lines[first - 1:ch.end_lineno])
                out.append((nid, kind, first, ch.end_lineno, ch, cls, src))
                walk(ch, nid, nid if kind == "class" else cls)
            else:
                walk(ch, prefix, cls)
    walk(tree, module_of(path), None)
    return out

def own_calls(fn):
    calls, stack = [], list(ast.iter_child_nodes(fn))
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
    entries, spans, text = [], {}, {}
    for path in sorted(files):
        rows = index_file(path, files[path])
        spans[path] = [(a, b, nid) for nid, _, a, b, _, _, _ in rows]
        for r in rows:
            text.setdefault(r[0], r[6])
        entries += [(path, r) for r in rows]
    by_short = collections.defaultdict(list)
    for path, r in entries:
        by_short[r[0].split(".")[-1]].append((r[0], module_of(path)))
    idset = {r[0] for _, r in entries}
    pairs = set()
    for path, (nid, kind, a, b, node, cls, _) in entries:
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
    return text, pairs, spans

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

def emit(text, pairs):
    nodes = [{"name": n, "text": text[n], "id": n} for n in sorted(text)]
    edges = []
    for u, v in sorted(pairs):
        edges.append({"type": "calls", "source": u, "target": v, "key": 0})
        edges.append({"type": "calls", "source": v, "target": u, "key": 0})
    return {"directed": True, "multigraph": True, "graph": {},
            "nodes": nodes, "edges": edges}

# ---------------- 3. embeddings: hashed identifier TF-IDF (lexical, deterministic) ----------------
WORD = re.compile(r"[A-Za-z][a-z0-9]+|[A-Z]+(?![a-z])")
def tokens(nid, src):
    toks = [w.lower() for w in WORD.findall(src) if len(w) >= 3]
    toks += [w.lower() for w in WORD.findall(nid) if len(w) >= 3] * 3
    return toks

def embed(text):
    ids = sorted(text)
    docs = [collections.Counter(tokens(n, text[n])) for n in ids]
    df = collections.Counter(t for d in docs for t in d)
    M = np.zeros((len(ids), EMB_DIM), dtype=np.float32)
    for i, d in enumerate(docs):
        for t, c in d.items():
            h = int(hashlib.md5(t.encode()).hexdigest()[:8], 16)
            sign = 1.0 if (h >> 9) & 1 else -1.0
            M[i, h % EMB_DIM] += sign * (1 + np.log(c)) * np.log((1 + len(ids)) / (1 + df[t]))
        n = np.linalg.norm(M[i])
        if n > 0:
            M[i] /= n
    return ids, M

# ---------------- 4. tasks from merged pull requests that close an issue ----------------
import urllib.request
HUNK = re.compile(r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")
TOK = re.compile(r"[A-Za-z_][A-Za-z0-9_]{3,}")
GENERIC = {"self", "None", "True", "False", "return", "import",
           "from", "class", "async", "await", "print", "value",
           "name", "data", "type", "list", "dict", "test", "json",
           "text", "file", "path", "args", "kwargs", "result"}
QUERY = """
query($owner: String!, $name: String!, $cursor: String) {
  repository(owner: $owner, name: $name) {
    pullRequests(states: MERGED, first: 100, after: $cursor,
                 orderBy: {field: CREATED_AT, direction: DESC}) {
      pageInfo { hasNextPage endCursor }
      nodes {
        number
        mergeCommit { oid }
        closingIssuesReferences(first: 5) {
          nodes { number title body }
        }
      }
    }
  }
}"""

def github_token():
    tok = os.environ.get("GITHUB_TOKEN")
    if not tok:
        try:
            from kaggle_secrets import UserSecretsClient
            tok = UserSecretsClient().get_secret("GITHUB_TOKEN")
        except Exception as e:
            print("GITHUB_TOKEN secret not available:", type(e).__name__)
    assert tok, "Add a Kaggle secret named GITHUB_TOKEN and attach it to this notebook"
    return tok.strip()

def graphql(token, variables):
    body = json.dumps({"query": QUERY, "variables": variables}).encode()
    for attempt in range(5):
        req = urllib.request.Request(
            "https://api.github.com/graphql", data=body,
            headers={"Authorization": "bearer " + token,
                     "Content-Type": "application/json",
                     "User-Agent": "maporritual-heldout-builder"})
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                out = json.loads(r.read())
        except Exception as e:
            print("  GitHub request failed, retry", attempt + 1, type(e).__name__, str(e)[:120])
            time.sleep(5 * (attempt + 1))
            continue
        if out.get("errors"):
            print("  GitHub error, retry", attempt + 1, str(out["errors"])[:200])
            time.sleep(5 * (attempt + 1))
            continue
        return out["data"]
    raise RuntimeError("GitHub GraphQL request failed 5 times")

FETCH = graphql

def merged_prs(token, repo_field, n):
    owner, name = repo_field.split("/")
    prs, cursor = [], None
    while len(prs) < n:
        page = FETCH(token, {"owner": owner, "name": name, "cursor": cursor})["repository"]["pullRequests"]
        prs += page["nodes"]
        if not page["pageInfo"]["hasNextPage"]:
            break
        cursor = page["pageInfo"]["endCursor"]
    return prs[:n]

def issue_text(issues):
    parts = []
    for it in issues:
        body = re.sub(r"<!--.*?-->", "", it.get("body") or "", flags=re.S).strip()
        parts.append(((it.get("title") or "").strip() + "\n\n" + body).strip())
    return "\n\n---\n\n".join(parts).strip()[:6000]

def changed_old_lines(repo, parent, commit):
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

def files_at(repo, commit):
    blob = git(repo, "archive", "--format=tar", commit, binary=True)
    allpy, src = {}, {}
    with tarfile.open(fileobj=io.BytesIO(blob)) as tf:
        for m in tf.getmembers():
            if m.isfile() and m.name.endswith(".py"):
                t = tf.extractfile(m).read().decode("utf-8", "replace")
                allpy[m.name] = t
                if is_source(m.name):
                    src[m.name] = t
    return allpy, src

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

def bfs_dist(adj, sources, targets, max_d=6):
    targets = set(targets)
    if targets & sources:
        return 0
    seen, frontier = set(sources), list(sources)
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

def in_history(repo, commit, pin):
    r = subprocess.run(["git", "-C", repo, "merge-base", "--is-ancestor", commit, pin],
                       capture_output=True)
    return r.returncode == 0

TOKEN = github_token()
tasks, labels, stats = [], [], collections.Counter()
for short, repo_field, url, pin in REPOS:
    repo = f"{CLONES}/{short}"
    if not os.path.isdir(repo):
        subprocess.run(["git", "clone", "-q", url, repo], check=True)
    prs = merged_prs(TOKEN, repo_field, N_PRS)
    stats[f"prs_fetched_{short}"] = len(prs)
    cands = []
    for pr in prs:
        issues = [i for i in (pr.get("closingIssuesReferences") or {}).get("nodes") or [] if i]
        if not issues:
            stats["no_linked_issue"] += 1
            continue
        commit = (pr.get("mergeCommit") or {}).get("oid")
        if not commit or not in_history(repo, commit, pin):
            stats["not_in_pinned_history"] += 1
            continue
        parents = git(repo, "rev-list", "--parents", "-n", "1", commit).split()[1:]
        if len(parents) not in (1, 2):
            stats["octopus_merge"] += 1
            continue
        stats["squash_merge" if len(parents) == 1 else "merge_commit"] += 1
        parent = parents[0]  # the PR's change is the diff from the first parent
        msg = issue_text(issues)
        if len(msg) < 40:
            stats["issue_too_short"] += 1
            continue
        touched = {p: l for p, l in changed_old_lines(repo, parent, commit).items() if is_source(p)}
        if not touched:
            stats["no_source_change"] += 1
            continue
        if len(touched) > 8:
            stats["broad_refactor"] += 1
            continue
        allpy, src = files_at(repo, parent)
        text, pairs, spans = build_graph(src)
        gold = gold_nodes(spans, touched)
        if not (1 <= len(gold) <= 10):
            stats["gold_count_out_of_range"] += 1
            continue
        if len(pairs) < 50:
            stats["graph_too_small"] += 1
            continue
        toks = set(TOK.findall(msg))
        if any(g.split(".")[-1] in toks for g in gold):
            stats["gold_name_in_issue"] += 1
            continue
        cands.append((hashlib.sha256(f"{SEED}:{commit}".encode()).hexdigest(), commit, parent,
                      msg, sorted(gold), pr["number"], [i["number"] for i in issues]))
    cands.sort()
    stats[f"eligible_{short}"] = len(cands)
    for _, commit, parent, msg, gold, prn, inums in cands[:PER_REPO]:
        iid = f"{short}_pr{prn}"
        allpy, src = files_at(repo, parent)
        text, pairs, spans = build_graph(src)
        name = f"{short}_{parent}"
        rp, frac = swap(pairs, name_seed(name))
        assert degrees(rp) == degrees(pairs) and frac == 1.0, (iid, frac)
        if not os.path.exists(f"{OUT}/W1/graphs/{name}.json"):
            with open(f"{OUT}/W1/graphs/{name}.json", "w") as f:
                json.dump(emit(text, pairs), f, sort_keys=True)
            with open(f"{OUT}/W2/graphs/{name}.json", "w") as f:
                json.dump(emit(text, rp), f, sort_keys=True)
            ids, M = embed(text)
            np.savez(f"{OUT}/embeddings/{name}.npz", **{n: M[i] for i, n in enumerate(ids)})
        for p, t in allpy.items():
            dst = f"{OUT}/src/{iid}/{p}"
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            with open(dst, "w") as f:
                f.write(t)
        adj = collections.defaultdict(set)
        for u, v in pairs:
            adj[u].add(v)
            adj[v].add(u)
        toks = set(TOK.findall(msg))
        anchors = {n for n in text if n.split(".")[-1] in toks
                   and n.split(".")[-1] not in GENERIC and not n.split(".")[-1].startswith("__")}
        hop = bfs_dist(adj, anchors, gold) if anchors else None
        tasks.append({"instance_id": iid, "repo": repo_field, "base_commit": parent,
                      "fix_commit": commit, "pull_request": prn, "issues": inums,
                      "problem_statement": msg})
        labels.append({"id": iid, "repo": repo_field, "gold": gold, "primary": True,
                       "hop": hop, "n_anchors": len(anchors),
                       "n_nodes": len(text), "n_edges": len(pairs),
                       "edges_changed": round(1 - len(pairs & rp) / len(pairs), 4)})
    print(f"{short}: merged PRs {len(prs)} | eligible {len(cands)} | kept {min(len(cands), PER_REPO)} | {round(time.time() - T0)} s")

tblob = "\n".join(json.dumps(t, sort_keys=True) for t in tasks).encode() + b"\n"
with open(f"{OUT}/tasks.jsonl", "wb") as f:
    f.write(tblob)
lblob = json.dumps(labels, sort_keys=True).encode()
with open(f"{OUT}/labels_heldout_issues.json", "wb") as f:
    f.write(lblob)
manifest = {"seed": SEED, "repos": [list(r) for r in REPOS], "n_tasks": len(tasks),
            "source": "merged pull requests that close a GitHub issue",
            "tasks_sha256": hashlib.sha256(tblob).hexdigest(),
            "labels_sha256": hashlib.sha256(lblob).hexdigest()}
with open(f"{OUT}/heldout_issues_manifest.json", "w") as f:
    json.dump(manifest, f, sort_keys=True)

print("\n=== HELD-OUT TASK SET (issues) ===")
print("filter counts:", sorted(stats.items()))
print("tasks:", len(tasks), collections.Counter(l["repo"] for l in labels).most_common())
if tasks:
    print("problem statement chars, median:", int(np.median([len(t["problem_statement"]) for t in tasks])))
    print("gold nodes per task, mean:", round(float(np.mean([len(l["gold"]) for l in labels])), 2))
    hops = collections.Counter("no_anchor" if l["n_anchors"] == 0 else
                               ("unreachable" if l["hop"] is None else l["hop"]) for l in labels)
    print("hop from issue anchors:", sorted(hops.items(), key=lambda kv: str(kv[0])))
    print("edges changed by rewiring, min / median:",
          min(l["edges_changed"] for l in labels), float(np.median([l["edges_changed"] for l in labels])))
print("tasks sha256:", manifest["tasks_sha256"][:16])
print("labels sha256:", manifest["labels_sha256"][:16])
print("BUILD DONE | seconds:", round(time.time() - T0))

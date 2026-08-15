"""Docker dataset generator v2 — composition-first.

## Why v2 exists

Measured failure of the v1 dataset (`dataset_docker.py`), from the corrected
evaluation in `docs/EVAL_METHODOLOGY.md`:

    accuracy by flag count in the target command
      0 flags   74.0%      training share 35.4%
      1 flag    47.1%      training share 52.0%
      2 flags    5.0%      training share 12.0%
      3+ flags   0.0%      training share  0.7%

Accuracy collapses exactly where training coverage ends. Two root causes, both
measurable in v1:

1. **No compositional coverage.** 24 distinct flags, but only 17 distinct flag
   *pairs* ever co-occur — and 47 of those occurrences are the trivial `-i`+`-t`
   bundle. `--detach`+`--publish` appears 3 times. `--detach`+`--env` twice. The
   model was asked to compose flags it had never seen composed.

2. **Phrasing is a lexical trigger, not an intent.** Every exec-shell example is
   phrased "open shell in container X" / "run bash in X". On held-out prompts like
   "drop me into a shell on the api container", the model emits `docker exec api
   bash` — right verb, right container, missing `-it`. It learned which words
   precede the flag, not that interactive intent requires it.

## What v2 does differently

* **Flags are composed combinatorially.** Each subcommand declares which flags it
  accepts; examples are built by sampling flag *subsets*, so pair and triple
  coverage is a generation target rather than an accident.
* **Every flag has many independent phrasings** expressing the same intent with
  disjoint vocabulary, so no single token becomes the trigger.
* **Sentence structure varies** — flag mentions are ordered randomly and joined
  with varied connectives, so position is not a cue either.
* **Contrastive pairs** are emitted deliberately: minimally different prompts whose
  correct answers differ by one flag.
* **Uniqueness is enforced**, and the generator refuses to emit a dataset that
  fails its own audit.

Generation is seeded and deterministic.

    python -m nlcli_wizard.dataset_v2 --out data/docker_train_v2.jsonl --n 5000
"""

from __future__ import annotations

import argparse
import collections
import json
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple

# ---------------------------------------------------------------------------
# Entity pools
# ---------------------------------------------------------------------------

IMAGES = [
    "nginx", "redis", "postgres", "mysql", "mongo", "node", "python",
    "ubuntu", "alpine", "busybox", "httpd", "rabbitmq", "elasticsearch",
    "grafana", "prometheus", "caddy", "traefik", "memcached",
]

# Default container port by image — used so port mappings are realistic.
DEFAULT_PORT = {
    "nginx": 80, "httpd": 80, "caddy": 80, "traefik": 80,
    "redis": 6379, "postgres": 5432, "mysql": 3306, "mongo": 27017,
    "node": 3000, "python": 8000, "rabbitmq": 5672, "elasticsearch": 9200,
    "grafana": 3000, "prometheus": 9090, "memcached": 11211,
}

CONTAINER_NAMES = [
    "web", "api", "db", "cache", "worker", "frontend", "backend", "app",
    "proxy", "queue", "search", "auth", "billing", "gateway", "scheduler",
    "notifications", "reports", "ingest",
]

NETWORK_NAMES = [
    "mynet", "appnet", "backend-net", "frontend-net", "db-net", "devnet",
    "staging-net", "prod-net", "internal", "edge",
]

VOLUME_NAMES = [
    "pgdata", "mydata", "app-data", "db-vol", "cache-vol", "uploads",
    "mysql-data", "mongo-data", "redis-data", "logs-vol", "shared",
]

DATA_PATHS = ["/data", "/srv/www", "/opt/app", "/var/data", "/mnt/store", "/home/user/project"]

WORKDIRS = ["/app", "/var/log", "/etc", "/tmp", "/srv", "/usr/src/app", "/home"]

IMAGE_DATA_DIR = {
    "postgres": "/var/lib/postgresql/data",
    "mysql": "/var/lib/mysql",
    "mongo": "/data/db",
    "redis": "/data",
    "nginx": "/usr/share/nginx/html",
    "elasticsearch": "/usr/share/elasticsearch/data",
    "grafana": "/var/lib/grafana",
}

ENV_BY_IMAGE = {
    "postgres": ("POSTGRES_PASSWORD", ["secret", "hunter2", "admin", "pgpass", "s3cret"]),
    "mysql": ("MYSQL_ROOT_PASSWORD", ["admin", "rootpass", "secret", "password123"]),
    "mongo": ("MONGO_INITDB_ROOT_PASSWORD", ["mongopass", "secret", "admin"]),
    "redis": ("REDIS_PASSWORD", ["redispass", "secret", "cachepw"]),
    "node": ("NODE_ENV", ["production", "development", "staging", "test"]),
    "python": ("PYTHONUNBUFFERED", ["1"]),
    "grafana": ("GF_SECURITY_ADMIN_PASSWORD", ["admin", "grafanapw"]),
}

APP_NAMES = ["myapp", "webapp", "api", "backend", "frontend", "service", "worker", "gateway"]
VERSIONS = ["1.0", "2.0", "3.0", "latest", "v1", "v2", "dev", "prod", "1.2.3"]
DOCKERFILES = ["Dockerfile.prod", "Dockerfile.dev", "Dockerfile.test", "Dockerfile.staging", "Dockerfile.ci"]
SERVICES = ["web", "api", "db", "redis", "worker", "queue", "nginx", "scheduler"]
SHELLS = ["bash", "sh"]


# ---------------------------------------------------------------------------
# Flag specifications
# ---------------------------------------------------------------------------

@dataclass
class FlagSpec:
    """One flag: how it renders, and the many ways a human might ask for it.

    `phrasings` is the heart of v2. Each entry is a template rendered with the
    sampled value. They deliberately use disjoint vocabulary so that no single
    word becomes the model's trigger for this flag.
    """

    key: str
    render: Callable[[dict], str]
    phrasings: List[str]
    sample: Callable[[random.Random, dict], dict] = field(default=lambda r, ctx: {})
    weight: float = 1.0


def _port_sample(rng: random.Random, ctx: dict) -> dict:
    img = ctx.get("image", "nginx")
    cport = DEFAULT_PORT.get(img, 8080)
    host = rng.choice([cport, cport, cport + 1, 8080, 3000, 8000, 9000, 5000, 8888])
    return {"host_port": host, "container_port": cport}


GENERIC_ENV = [
    ("LOG_LEVEL", ["debug", "info", "warn", "error"]),
    ("TZ", ["UTC", "Europe/London", "Asia/Kolkata"]),
    ("APP_ENV", ["production", "staging", "development"]),
    ("PORT", ["8080", "3000", "5000"]),
    ("DEBUG", ["1", "0", "true"]),
]


def _env_sample(rng: random.Random, ctx: dict) -> dict:
    """Pick an environment variable that actually belongs to this image.

    v1 of this function fell back to another image's variable when the image was
    unknown, producing training rows like `docker run -e
    MONGO_INITDB_ROOT_PASSWORD=secret caddy`. Syntactically fine, semantically
    nonsense, and not something to train on or show a reviewer.
    """
    img = ctx.get("image", "node")
    if img in ENV_BY_IMAGE:
        name, values = ENV_BY_IMAGE[img]
    else:
        name, values = rng.choice(GENERIC_ENV)
    return {"env_name": name, "env_value": rng.choice(values)}


def _volume_sample(rng: random.Random, ctx: dict) -> dict:
    img = ctx.get("image", "nginx")
    target = IMAGE_DATA_DIR.get(img, "/data")
    if rng.random() < 0.6:
        source = rng.choice(VOLUME_NAMES)
        kind = "named"
    else:
        source = rng.choice(DATA_PATHS)
        kind = "bind"
    return {"vol_source": source, "vol_target": target, "vol_kind": kind}


RUN_FLAGS: List[FlagSpec] = [
    FlagSpec(
        key="detach",
        render=lambda v: "-d",
        phrasings=[
            "in the background", "detached", "in detached mode", "as a daemon",
            "without blocking my terminal", "in the background so i get my prompt back",
            "backgrounded", "so it keeps running after i close the terminal",
            "running quietly in the background", "and dont attach to it",
        ],
        weight=2.0,
    ),
    FlagSpec(
        key="publish",
        render=lambda v: f"-p {v['host_port']}:{v['container_port']}",
        sample=_port_sample,
        phrasings=[
            "on port {host_port}", "exposed on {host_port}",
            "mapping port {host_port} to {container_port}",
            "reachable at localhost:{host_port}",
            "with {host_port} forwarded to {container_port}",
            "bound to {host_port}", "listening on {host_port}",
            "so i can hit it on {host_port}", "published on port {host_port}",
            "port {host_port}",
        ],
        weight=2.0,
    ),
    FlagSpec(
        key="name",
        render=lambda v: f"--name {v['cname']}",
        sample=lambda r, ctx: {"cname": r.choice(CONTAINER_NAMES)},
        phrasings=[
            "named {cname}", "called {cname}", "with the name {cname}",
            "and name it {cname}", "labelled {cname}", "give it the name {cname}",
            "as {cname}", "identified as {cname}", "name {cname}",
        ],
        weight=1.5,
    ),
    FlagSpec(
        key="env",
        render=lambda v: f"-e {v['env_name']}={v['env_value']}",
        sample=_env_sample,
        phrasings=[
            "with {env_name} set to {env_value}", "setting {env_name}={env_value}",
            "with the env var {env_name} as {env_value}",
            "passing {env_name}={env_value}",
            "where {env_name} is {env_value}",
            "with environment {env_name}={env_value}",
            "and set {env_name} to {env_value}",
            "configured with {env_name}={env_value}",
        ],
        weight=1.5,
    ),
    FlagSpec(
        key="volume",
        render=lambda v: f"-v {v['vol_source']}:{v['vol_target']}",
        sample=_volume_sample,
        phrasings=[
            "with {vol_source} mounted at {vol_target}",
            "persisting data to {vol_source}",
            "with the volume {vol_source}",
            "storing its data in {vol_source}",
            "mounting {vol_source} on {vol_target}",
            "backed by {vol_source}",
            "so the data survives in {vol_source}",
            "with {vol_source} attached at {vol_target}",
        ],
        weight=1.2,
    ),
    FlagSpec(
        key="rm",
        render=lambda v: "--rm",
        phrasings=[
            "and remove it when it exits", "as a throwaway",
            "that cleans itself up", "and delete it afterwards",
            "temporarily", "and dont leave the container behind",
            "auto removed on exit", "disposable",
            "and clean up when im done",
        ],
    ),
    FlagSpec(
        key="restart",
        render=lambda v: f"--restart {v['policy']}",
        sample=lambda r, ctx: {"policy": r.choice(["always", "on-failure", "unless-stopped"])},
        phrasings=[
            "with restart policy {policy}", "that restarts {policy}",
            "set to restart {policy}", "with {policy} restart",
            "and keep it restarting {policy}",
            "using the {policy} restart policy",
        ],
    ),
    FlagSpec(
        key="network",
        render=lambda v: f"--network {v['net']}",
        sample=lambda r, ctx: {"net": r.choice(NETWORK_NAMES)},
        phrasings=[
            "on the {net} network", "attached to {net}",
            "connected to the {net} network", "inside {net}",
            "using network {net}", "joined to {net}",
        ],
    ),
    FlagSpec(
        key="interactive",
        render=lambda v: "-it",
        phrasings=[
            "interactively", "with a terminal attached", "so i can type into it",
            "and drop me into it", "with an interactive shell",
            "so i can poke around inside", "attached to my terminal",
            "and give me a prompt", "interactive",
        ],
    ),
    FlagSpec(
        key="memory",
        render=lambda v: f"--memory {v['mem']}",
        sample=lambda r, ctx: {"mem": r.choice(["256m", "512m", "1g", "2g"])},
        phrasings=[
            "limited to {mem} of memory", "capped at {mem} ram",
            "with a {mem} memory limit", "using at most {mem} of memory",
            "with memory limited to {mem}",
        ],
        weight=0.6,
    ),
    FlagSpec(
        key="user",
        render=lambda v: f"-u {v['uid']}",
        sample=lambda r, ctx: {"uid": r.choice(["1000", "1001", "root", "node"])},
        phrasings=[
            "as user {uid}", "running as {uid}", "under the {uid} user",
            "with uid {uid}", "as the {uid} account",
        ],
        weight=0.6,
    ),
]

BUILD_FLAGS: List[FlagSpec] = [
    FlagSpec(
        key="tag",
        render=lambda v: f"-t {v['app']}:{v['ver']}",
        sample=lambda r, ctx: {"app": r.choice(APP_NAMES), "ver": r.choice(VERSIONS)},
        phrasings=[
            "tagged {app}:{ver}", "and tag it {app}:{ver}",
            "called {app} version {ver}", "as {app}:{ver}",
            "naming the image {app}:{ver}", "labelled {app}:{ver}",
            "and call the image {app}:{ver}",
        ],
        weight=2.0,
    ),
    FlagSpec(
        key="file",
        render=lambda v: f"-f {v['df']}",
        sample=lambda r, ctx: {"df": r.choice(DOCKERFILES)},
        phrasings=[
            "using {df}", "from {df}", "with the {df} dockerfile",
            "based on {df}", "reading {df}", "off {df}",
        ],
        weight=1.5,
    ),
    FlagSpec(
        key="nocache",
        render=lambda v: "--no-cache",
        phrasings=[
            "without the cache", "from scratch", "ignoring cached layers",
            "with a clean build", "skipping the cache", "forcing a full rebuild",
            "and dont reuse cached layers",
        ],
        weight=1.5,
    ),
    FlagSpec(
        key="buildarg",
        render=lambda v: f"--build-arg {v['arg']}={v['val']}",
        sample=lambda r, ctx: (lambda a, v: {"arg": a, "val": r.choice(v)})(
            *r.choice([
                ("VERSION", ["1.0", "2.1", "3.0", "1.2.3"]),
                ("NODE_ENV", ["production", "development", "staging"]),
                ("ENV", ["prod", "dev", "staging", "ci"]),
                ("BUILD_ID", ["1042", "2311", "7", "9001"]),
                ("TARGET", ["runtime", "builder", "dev", "prod"]),
            ])
        ),
        phrasings=[
            "passing {arg}={val}", "with build arg {arg}={val}",
            "setting {arg} to {val} at build time",
            "with {arg}={val} as a build argument",
            "and pass {arg}={val} in",
        ],
    ),
]


# ---------------------------------------------------------------------------
# Sentence assembly
# ---------------------------------------------------------------------------

RUN_OPENERS = [
    "run {image}", "start {image}", "launch {image}", "spin up {image}",
    "bring up {image}", "fire up {image}", "create a {image} container",
    "get {image} running", "start up a {image} container", "boot {image}",
    "i need {image}", "give me {image}", "stand up {image}",
    "{image} container", "new {image} container",
]

BUILD_OPENERS = [
    "build the image", "build it", "build this", "build the docker image",
    "make an image", "build from here", "build the current directory",
    "compile this into an image", "build whats here", "create an image",
]

CONNECTORS = [", ", " ", " and ", ", and ", " with it ", "; "]


def _join_phrases(rng: random.Random, opener: str, phrases: List[str]) -> str:
    """Assemble a natural-sounding instruction from an opener and flag phrases.

    Order is randomized and connectives vary, so neither position nor a fixed
    conjunction becomes a cue for a particular flag.
    """
    if not phrases:
        return opener
    phrases = list(phrases)
    rng.shuffle(phrases)

    out = opener
    for i, p in enumerate(phrases):
        if i == len(phrases) - 1 and len(phrases) > 1:
            conn = rng.choice([" and ", ", and ", ", "])
        else:
            conn = rng.choice([", ", " ", " "])
        # Several phrasings already open with a conjunction ("and dont attach to
        # it"). Prepending another produces "and and".
        if p.startswith("and ") and "and" in conn:
            conn = ", "
        out += conn + p
    return " ".join(out.split())


def _dedup_key(prompt: str) -> str:
    """Order/case/punctuation-insensitive key.

    Matches `eval.contamination.prompt_fingerprint` so that anything the audit
    would flag as a near-duplicate is also blocked at generation time.
    """
    import re

    s = re.sub(r"[^\w\s]", " ", prompt.lower())
    return " ".join(sorted(s.split()))


def load_exclusions(paths: Sequence[Path]) -> Set[str]:
    """Prompt fingerprints that generation must avoid (i.e. the test sets)."""
    import re

    prefix = re.compile(r"^Translate to \S+ command:\s*", re.IGNORECASE)
    out: Set[str] = set()
    for path in paths:
        if not path.exists():
            continue
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                raw = row.get("instruction") or row.get("prompt") or ""
                out.add(_dedup_key(prefix.sub("", raw).strip()))
    return out


@dataclass
class Example:
    prompt: str
    command: str
    category: str
    n_flags: int


class DockerDatasetV2:
    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)

    # -- docker run ---------------------------------------------------------

    def gen_run(self, n_flags: int) -> Optional[Example]:
        rng = self.rng
        image = rng.choice(IMAGES)
        ctx = {"image": image}

        pool = [f for f in RUN_FLAGS]
        # Port publishing only makes sense for images that listen on one. Without
        # this, `docker run -p 3000:8080 busybox` enters the training data.
        if image not in DEFAULT_PORT:
            pool = [f for f in pool if f.key != "publish"]
        weights = [f.weight for f in pool]
        chosen: List[FlagSpec] = []
        for _ in range(min(n_flags, len(pool))):
            pick = rng.choices(pool, weights=weights, k=1)[0]
            idx = pool.index(pick)
            pool.pop(idx)
            weights.pop(idx)
            chosen.append(pick)

        # `-it` and `-d` are contradictory; drop detach if both were chosen.
        keys = {f.key for f in chosen}
        if "interactive" in keys and "detach" in keys:
            chosen = [f for f in chosen if f.key != "detach"]

        values: Dict[str, object] = {}
        phrases: List[str] = []
        for spec in chosen:
            v = spec.sample(rng, ctx)
            values.update(v)
            phrases.append(rng.choice(spec.phrasings).format(**{**ctx, **values}))

        # Canonical flag order for the target command (docker's own convention).
        order = ["detach", "interactive", "rm", "publish", "env", "volume",
                 "name", "network", "restart", "memory", "user"]
        chosen.sort(key=lambda f: order.index(f.key) if f.key in order else 99)

        parts = ["docker", "run"]
        for spec in chosen:
            parts.append(spec.render({**ctx, **values}))
        parts.append(image)

        # An interactive run needs a shell to be useful.
        if "interactive" in {f.key for f in chosen}:
            parts.append("bash" if image not in {"alpine", "busybox"} else "sh")

        opener = rng.choice(RUN_OPENERS).format(image=image)
        prompt = _join_phrases(rng, opener, phrases)
        return Example(prompt, " ".join(parts), "run", len(chosen))

    # -- docker build -------------------------------------------------------

    def gen_build(self, n_flags: int) -> Optional[Example]:
        rng = self.rng
        pool = list(BUILD_FLAGS)
        weights = [f.weight for f in pool]
        chosen: List[FlagSpec] = []
        for _ in range(min(n_flags, len(pool))):
            pick = rng.choices(pool, weights=weights, k=1)[0]
            i = pool.index(pick)
            pool.pop(i)
            weights.pop(i)
            chosen.append(pick)

        values: Dict[str, object] = {}
        phrases: List[str] = []
        for spec in chosen:
            values.update(spec.sample(rng, {}))
            phrases.append(rng.choice(spec.phrasings).format(**values))

        order = ["nocache", "file", "buildarg", "tag"]
        chosen.sort(key=lambda f: order.index(f.key) if f.key in order else 99)

        parts = ["docker", "build"]
        for spec in chosen:
            parts.append(spec.render(values))
        parts.append(".")

        opener = rng.choice(BUILD_OPENERS)
        prompt = _join_phrases(rng, opener, phrases)
        return Example(prompt, " ".join(parts), "build", len(chosen))

    # -- docker exec (with explicit interactive/non-interactive contrast) ----

    def gen_exec(self, interactive: Optional[bool] = None) -> Example:
        rng = self.rng
        c = rng.choice(CONTAINER_NAMES)
        if interactive is None:
            interactive = rng.random() < 0.55

        if interactive:
            shell = rng.choice(SHELLS)
            openers = [
                f"drop me into a shell on {c}",
                f"get me a shell in {c}",
                f"i need to poke around inside {c}",
                f"open a terminal in the {c} container",
                f"shell into {c}",
                f"exec into {c}",
                f"attach a terminal to {c}",
                f"give me a prompt inside {c}",
                f"log into the {c} container",
                f"start an interactive session in {c}",
                f"run {shell} in {c} interactively",
                f"connect to {c} and give me a shell",
                f"i want to look around inside {c}",
                f"jump into the {c} container",
            ]
            return Example(rng.choice(openers), f"docker exec -it {c} {shell}", "exec", 1)

        # Non-interactive: run one command and print the output.
        cmd, phrasings = rng.choice([
            ("ls", [f"list the files in {c}", f"show me whats in {c}",
                    f"what files are in the {c} container"]),
            ("df -h", [f"check disk usage inside {c}", f"how much disk does {c} have left",
                       f"show free space in {c}"]),
            ("pwd", [f"what directory is {c} in", f"show the working directory of {c}"]),
            ("ps aux", [f"what processes are running in {c}",
                        f"show the process list inside {c}"]),
            ("env", [f"dump the environment variables in {c}",
                     f"show {c}s environment"]),
        ])
        if rng.random() < 0.35:
            wd = rng.choice(WORKDIRS)
            prompt = rng.choice([
                f"run {cmd} in {wd} on {c}",
                f"check {wd} inside the {c} container",
                f"what's in {wd} on {c}",
            ])
            return Example(prompt, f"docker exec -w {wd} {c} {cmd}", "exec", 1)
        return Example(rng.choice(phrasings), f"docker exec {c} {cmd}", "exec", 0)

    # -- simple / zero-flag commands ---------------------------------------

    def gen_simple(self) -> Example:
        rng = self.rng
        table: List[Tuple[str, List[str], str]] = [
            ("docker ps", ["what containers are running", "list running containers",
                           "show me whats up", "which containers are alive",
                           "whats currently running", "ps"], "ps_images"),
            ("docker ps -a", ["show all containers including stopped",
                              "list every container", "show the dead ones too",
                              "all containers, running or not",
                              "i cant remember the flag for stopped containers"], "ps_images"),
            ("docker ps -q", ["just the container ids", "only ids please",
                              "container ids and nothing else"], "ps_images"),
            ("docker images", ["what images do i have", "list local images",
                               "show my images", "images"], "ps_images"),
            ("docker system df", ["how much disk is docker using",
                                  "whats taking up space",
                                  "docker disk usage"], "system"),
            ("docker system prune", ["clean up unused stuff",
                                     "free up some space",
                                     "prune docker"], "system"),
            ("docker system prune -a", ["clean up everything including images",
                                        "aggressive cleanup, remove unused images too"], "system"),
            ("docker info", ["tell me about this docker install",
                             "docker system info"], "system"),
            ("docker version", ["what docker version is this",
                                "show docker version"], "system"),
            ("docker network ls", ["list networks", "what networks exist",
                                   "show docker networks"], "network"),
            ("docker volume ls", ["list volumes", "what volumes do i have",
                                  "show docker volumes"], "volume"),
            ("docker volume prune", ["delete unused volumes",
                                     "clean up volumes"], "volume"),
            ("docker network prune", ["remove unused networks",
                                      "clean up networks"], "network"),
            ("docker container prune", ["remove stopped containers",
                                        "clean up dead containers"], "system"),
            ("docker image prune", ["remove dangling images",
                                    "clean up untagged images"], "system"),
        ]
        cmd, prompts, cat = rng.choice(table)
        return Example(rng.choice(prompts), cmd, cat, 0)

    # -- parameterised single-entity commands -------------------------------

    def gen_entity(self) -> Example:
        rng = self.rng
        kind = rng.choice(["logs", "stop", "rm", "inspect", "netcreate", "netconn",
                           "volcreate", "volrm", "top", "stats"])
        c = rng.choice(CONTAINER_NAMES)
        if kind == "logs":
            if rng.random() < 0.4:
                return Example(
                    rng.choice([f"follow the logs on {c}", f"tail {c} logs live",
                                f"stream logs from {c}", f"watch {c}s output"]),
                    f"docker logs -f {c}", "ps_images", 1)
            if rng.random() < 0.4:
                n = rng.choice([20, 50, 100, 200])
                return Example(
                    rng.choice([f"last {n} lines of {c} logs",
                                f"show me {n} lines of logs from {c}"]),
                    f"docker logs --tail {n} {c}", "ps_images", 1)
            return Example(
                rng.choice([f"show logs for {c}", f"what has {c} logged",
                            f"logs {c}"]),
                f"docker logs {c}", "ps_images", 0)
        if kind == "stop":
            return Example(
                rng.choice([f"stop {c}", f"shut down the {c} container",
                            f"kill {c}", f"halt {c}", f"take {c} down"]),
                f"docker stop {c}", "system", 0)
        if kind == "rm":
            if rng.random() < 0.4:
                return Example(
                    rng.choice([f"force remove {c}", f"delete {c} even if its running",
                                f"forcibly delete the {c} container"]),
                    f"docker rm -f {c}", "system", 1)
            return Example(
                rng.choice([f"remove the {c} container", f"delete {c}",
                            f"get rid of {c}"]),
                f"docker rm {c}", "system", 0)
        if kind == "inspect":
            return Example(
                rng.choice([f"inspect {c}", f"show me the full config of {c}",
                            f"details on the {c} container"]),
                f"docker inspect {c}", "ps_images", 0)
        if kind == "top":
            return Example(
                rng.choice([f"what processes run in {c}", f"show the processes in {c}"]),
                f"docker top {c}", "ps_images", 0)
        if kind == "stats":
            return Example(
                rng.choice([f"resource usage for {c}", f"how much cpu is {c} using"]),
                f"docker stats {c}", "ps_images", 0)
        if kind == "netcreate":
            n = rng.choice(NETWORK_NAMES)
            return Example(
                rng.choice([f"create a network called {n}", f"make the {n} network",
                            f"new network {n}", f"set up a network named {n}"]),
                f"docker network create {n}", "network", 0)
        if kind == "netconn":
            n = rng.choice(NETWORK_NAMES)
            if rng.random() < 0.5:
                return Example(
                    rng.choice([f"connect {c} to {n}", f"attach the {c} container to {n}",
                                f"put {c} on the {n} network"]),
                    f"docker network connect {n} {c}", "network", 0)
            return Example(
                rng.choice([f"disconnect {c} from {n}", f"take {c} off {n}",
                            f"detach {c} from the {n} network"]),
                f"docker network disconnect {n} {c}", "network", 0)
        if kind == "volcreate":
            v = rng.choice(VOLUME_NAMES)
            return Example(
                rng.choice([f"create a volume named {v}", f"make the {v} volume",
                            f"new volume {v}"]),
                f"docker volume create {v}", "volume", 0)
        v = rng.choice(VOLUME_NAMES)
        return Example(
            rng.choice([f"delete the {v} volume", f"remove volume {v}",
                        f"drop the {v} volume"]),
            f"docker volume rm {v}", "volume", 0)

    # -- docker compose -----------------------------------------------------

    def gen_compose(self) -> Example:
        rng = self.rng
        svc = rng.choice(SERVICES)
        kind = rng.choice(["up", "upd", "down", "downv", "logs", "logsf", "ps",
                           "restart", "build", "buildnc", "scale", "exec",
                           "stop", "start", "pull"])
        # "the stack" has many names; varying it multiplies phrasing coverage for
        # the compose actions that take no service argument, which would otherwise
        # saturate against the dedup filter at a handful of unique prompts.
        stack = rng.choice([
            "the stack", "everything", "all the services", "the compose services",
            "the whole thing", "compose", "the app", "all of it", "the services",
            "the whole stack", "my services",
        ])
        table = {
            "up": ([f"start {stack}", f"bring {stack} up", f"compose up {stack}".strip(),
                    f"get {stack} running", f"launch {stack}", f"boot {stack}",
                    f"fire up {stack}", f"spin up {stack}"],
                   "docker-compose up", 0),
            "upd": ([f"bring {stack} up in the background",
                     f"start {stack} detached",
                     f"launch {stack} without blocking my terminal",
                     f"start {stack} in the background",
                     f"get {stack} running in the background",
                     f"spin up {stack} detached",
                     f"bring {stack} up backgrounded",
                     f"start {stack} as a daemon"],
                    "docker-compose up -d", 1),
            "down": ([f"tear {stack} down", f"stop {stack}", f"bring {stack} down",
                      f"shut {stack} down", f"take {stack} offline",
                      f"kill {stack}", f"stop and remove {stack}"],
                     "docker-compose down", 0),
            "downv": ([f"tear {stack} down and delete the volumes",
                       f"stop {stack} and wipe the data",
                       f"bring {stack} down, volumes included",
                       f"shut {stack} down and remove the volumes",
                       f"take {stack} down and nuke the data"],
                      "docker-compose down -v", 1),
            "logs": ([f"show logs for the {svc} service",
                      f"what has the {svc} service logged",
                      f"logs from the {svc} service",
                      f"print the {svc} service logs",
                      f"i want to see {svc} service output"],
                     f"docker-compose logs {svc}", 0),
            "logsf": ([f"follow the {svc} service logs",
                       f"stream logs from the {svc} service",
                       f"tail the {svc} service logs live",
                       f"watch the {svc} service output",
                       f"keep showing me {svc} service logs as they come"],
                      f"docker-compose logs -f {svc}", 1),
            "ps": ([f"status of {stack}", f"what compose services are up",
                    f"list {stack}", f"show me the state of {stack}",
                    f"which services are running"],
                   "docker-compose ps", 0),
            "restart": ([f"restart {stack}", f"bounce {stack}",
                         f"cycle {stack}", f"restart all the compose services"],
                        "docker-compose restart", 0),
            "build": ([f"build the compose images", f"build {stack}",
                       f"build the service images", f"compile the compose images"],
                      "docker-compose build", 0),
            "buildnc": ([f"rebuild the compose images from scratch",
                         f"build {stack} ignoring the cache",
                         f"rebuild {stack} with no cache",
                         f"force a clean build of the compose images"],
                        "docker-compose build --no-cache", 1),
            "scale": ([f"run {{n}} copies of the {svc} service",
                       f"scale {svc} to {{n}}",
                       f"i want {{n}} {svc} instances",
                       f"give me {{n}} replicas of {svc}",
                       f"bump {svc} up to {{n}} containers",
                       f"run the {svc} service {{n}} times"], None, 2),
            "exec": ([f"shell into the {svc} service",
                      f"get a shell in the {svc} service",
                      f"exec into the {svc} service",
                      f"open a terminal in the {svc} service",
                      f"drop me into the {svc} service"],
                     f"docker-compose exec {svc} bash", 0),
            "stop": ([f"stop {stack} without removing anything",
                      f"halt {stack}", f"pause {stack}",
                      f"stop the compose services but keep them"],
                     "docker-compose stop", 0),
            "start": ([f"start {stack} back up", f"resume {stack}",
                       f"start the stopped compose services"],
                      "docker-compose start", 0),
            "pull": ([f"pull fresh images for {stack}",
                      f"update the compose images",
                      f"fetch the latest images for {stack}"],
                     "docker-compose pull", 0),
        }
        prompts, cmd, nf = table[kind]
        if kind == "scale":
            n = rng.choice([2, 3, 4, 5])
            prompt = rng.choice(prompts).format(n=n)
            return Example(prompt, f"docker-compose up -d --scale {svc}={n}", "compose", 2)
        return Example(rng.choice(prompts), cmd, "compose", nf)

    # -- driver -------------------------------------------------------------

    def generate(self, target: int, exclude: Optional[Set[str]] = None) -> List[Example]:
        """Generate `target` unique examples with deliberate flag-count coverage.

        `exclude` is a set of prompt fingerprints that must never be emitted —
        normally the held-out test set. This is not optional hygiene: the first
        v2 build leaked 12 test prompts verbatim, because the same author wrote
        both the test set and the generator phrasings and naturally reached for
        the same words twice. Human care does not prevent that. A blocklist does.

        Excluding the test set from generation does not leak label information;
        it only enforces the disjointness the evaluation already assumes.
        """
        rng = self.rng
        out: List[Example] = []
        seen: Set[str] = set(exclude or set())

        # Target flag-count mix. Unlike v1 (87% of rows with <=1 flag), multi-flag
        # commands are a first-class part of the distribution.
        # Weights are the *requested* mix. Low-variety generators (simple, compose)
        # saturate against the dedup filter well before their weight is satisfied,
        # so the realised mix is checked in the audit rather than assumed here.
        plan = [
            ("run", 0, 0.02), ("run", 1, 0.05), ("run", 2, 0.07),
            ("run", 3, 0.06), ("run", 4, 0.03), ("run", 5, 0.02),
            ("build", 0, 0.01), ("build", 1, 0.03), ("build", 2, 0.04),
            ("build", 3, 0.02), ("build", 4, 0.01),
            ("exec", -1, 0.16),
            ("compose", -1, 0.20),
            ("entity", -1, 0.22),
            ("simple", -1, 0.06),
        ]
        total_w = sum(w for _, _, w in plan)

        attempts = 0
        max_attempts = target * 80
        while len(out) < target and attempts < max_attempts:
            attempts += 1
            r = rng.random() * total_w
            acc = 0.0
            kind, nf = plan[-1][0], plan[-1][1]
            for k, n, w in plan:
                acc += w
                if r <= acc:
                    kind, nf = k, n
                    break

            if kind == "run":
                ex = self.gen_run(nf)
            elif kind == "build":
                ex = self.gen_build(nf)
            elif kind == "exec":
                ex = self.gen_exec()
            elif kind == "compose":
                ex = self.gen_compose()
            elif kind == "entity":
                ex = self.gen_entity()
            else:
                ex = self.gen_simple()

            if ex is None:
                continue
            key = _dedup_key(ex.prompt)
            if key in seen:
                continue
            seen.add(key)
            out.append(ex)

        rng.shuffle(out)
        return out


# ---------------------------------------------------------------------------
# Audit + output
# ---------------------------------------------------------------------------

def audit(examples: Sequence[Example]) -> Dict[str, object]:
    """The generator grades its own output before it is allowed to be written."""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from eval.normalize import parse_command

    prompts = [e.prompt for e in examples]
    cmds = [e.command for e in examples]

    flag_counts = collections.Counter()
    flag_freq = collections.Counter()
    pairs = collections.Counter()
    for c in cmds:
        fl = sorted({f for f, _ in parse_command(c).flags})
        flag_counts[len(fl)] += 1
        for f in fl:
            flag_freq[f] += 1
        for i in range(len(fl)):
            for j in range(i + 1, len(fl)):
                pairs[(fl[i], fl[j])] += 1

    n = len(examples)
    return {
        "rows": n,
        "unique_prompts": len(set(prompts)),
        "unique_commands": len(set(cmds)),
        "prompt_dup_factor": round(n / max(len(set(prompts)), 1), 3),
        "flag_count_dist": {k: f"{v} ({v / n:.1%})" for k, v in sorted(flag_counts.items())},
        "multi_flag_share": f"{sum(v for k, v in flag_counts.items() if k >= 2) / n:.1%}",
        "distinct_flags": len(flag_freq),
        "distinct_flag_pairs": len(pairs),
        "categories": dict(collections.Counter(e.category for e in examples)),
    }


def to_row(ex: Example) -> dict:
    return {
        "instruction": f"Translate to docker command: {ex.prompt}",
        "input": "",
        "output": f"COMMAND: {ex.command}\n",
        "category": ex.category,
    }


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Generate the v2 Docker dataset.")
    ap.add_argument("--out", type=Path, default=Path("data/docker_train_v2.jsonl"))
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dry-run", action="store_true", help="audit only, do not write")
    ap.add_argument(
        "--exclude",
        type=Path,
        nargs="*",
        default=[Path("data/docker_test_handwritten.jsonl")],
        help="held-out files whose prompts must never be generated (default: the "
             "handwritten test set). Pass nothing to disable, at your peril.",
    )
    args = ap.parse_args(argv)

    exclusions = load_exclusions(args.exclude or [])
    if exclusions:
        print(f"excluding {len(exclusions)} held-out prompt fingerprints from generation")

    gen = DockerDatasetV2(seed=args.seed)
    examples = gen.generate(args.n, exclude=exclusions)
    stats = audit(examples)

    print("=" * 70)
    print("DATASET V2 AUDIT")
    print("=" * 70)
    for k, v in stats.items():
        print(f"  {k:<22} {v}")
    print("=" * 70)

    problems = []
    if stats["prompt_dup_factor"] != 1.0:
        problems.append("duplicate prompts present")
    if stats["distinct_flag_pairs"] < 40:
        problems.append(
            f"only {stats['distinct_flag_pairs']} distinct flag pairs "
            "(v1 had 17; the whole point of v2 is composition coverage)"
        )
    if problems:
        print("REFUSING TO WRITE:")
        for p in problems:
            print(f"  - {p}")
        return 1

    if args.dry_run:
        print("dry run - nothing written")
        return 0

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(to_row(ex)) + "\n")
    print(f"wrote {len(examples)} examples to {args.out}")
    print("\nNext: audit against the held-out set before training on it:")
    print(f"  python -m eval.contamination --train {args.out} "
          "--test data/docker_test_handwritten.jsonl")
    return 0


if __name__ == "__main__":
    sys.exit(main())

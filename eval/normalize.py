"""Structural parsing of CLI commands.

The old harness compared model output to the gold command with `==`. That scores
`docker run -d -p 8080:80 nginx` and `docker run -p 8080:80 -d nginx` as a
mismatch, even though they are the same command. Exact match is a floor, not a
measurement.

This module turns a command string into a structure that can be compared with
order-insensitivity where order genuinely does not matter, and order-sensitivity
where it does (a container's trailing argv, for example, is positional and its
order is load-bearing).

Deliberately Docker-aware but not Docker-only: `ParsedCommand` is generic and the
tool-specific knowledge lives in `FLAG_SPEC`.
"""

from __future__ import annotations

import shlex
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

# ---------------------------------------------------------------------------
# Tool-specific knowledge
# ---------------------------------------------------------------------------

# Flags that consume the following token as their value. Anything not listed is
# treated as a boolean switch. Getting this wrong shifts every later token, so
# this table is the highest-risk part of the module and is unit-tested.
VALUE_FLAGS: Set[str] = {
    # docker run / exec / build
    "-p", "--publish",
    "-e", "--env",
    "-v", "--volume",
    "--name",
    "-w", "--workdir",
    "--network",
    "--restart",
    "-t", "--tag",
    "-f", "--file",
    "--build-arg",
    "-u", "--user",
    "--entrypoint",
    "--memory", "-m",
    "--cpus",
    "--label", "-l",
    # docker ps / logs / images
    "-n",
    "--tail",
    "--since",
    "--filter",
    "--format",
    # docker-compose
    "--scale",
    "--project-name",
}

# Short boolean flags that are commonly bundled (`-it` == `-i -t`).
BUNDLEABLE_SHORT_BOOLS: Set[str] = set("aditqslfPvh")

# Subcommands that take a further subcommand (`docker network create`, etc.).
# Needed so the "command path" is identified correctly.
NESTED_SUBCOMMANDS: Set[str] = {"network", "volume", "system", "container", "image", "compose"}

# Aliases that are the same command spelled differently. `docker compose` (the v2
# plugin) and `docker-compose` (the v1 binary) are interchangeable for our purposes.
COMMAND_ALIASES: Dict[Tuple[str, ...], Tuple[str, ...]] = {
    ("docker", "compose"): ("docker-compose",),
}

# Subcommand pairs that mean the same thing: `docker stop` == `docker container stop`.
SUBCOMMAND_ALIASES: Dict[Tuple[str, ...], Tuple[str, ...]] = {
    ("docker", "container", "stop"): ("docker", "stop"),
    ("docker", "container", "rm"): ("docker", "rm"),
    ("docker", "container", "ls"): ("docker", "ps"),
    ("docker", "image", "ls"): ("docker", "images"),
    ("docker", "container", "logs"): ("docker", "logs"),
    ("docker", "container", "exec"): ("docker", "exec"),
    ("docker", "container", "run"): ("docker", "run"),
}


@dataclass(frozen=True)
class ParsedCommand:
    """A CLI command decomposed into comparable parts."""

    path: Tuple[str, ...]
    """The command path: ('docker', 'run') or ('docker', 'network', 'create')."""

    flags: Tuple[Tuple[str, Optional[str]], ...]
    """(flag, value) pairs, sorted. Value is None for boolean switches."""

    positionals: Tuple[str, ...]
    """Positional arguments in source order — order IS significant here."""

    raw: str = field(default="", compare=False)
    """The original string, retained for error reporting."""

    parse_ok: bool = field(default=True, compare=False)
    """False when the command could not be tokenized (unbalanced quotes, etc.)."""

    @property
    def flag_names(self) -> Set[str]:
        return {f for f, _ in self.flags}

    def __str__(self) -> str:
        return self.raw


def _canonicalize_flag(flag: str) -> str:
    """Map a flag to a canonical spelling so `-p` and `--publish` compare equal."""
    return _SHORT_TO_LONG.get(flag, flag)


# Built from VALUE_FLAGS pairings plus common boolean synonyms.
_SHORT_TO_LONG: Dict[str, str] = {
    "-p": "--publish",
    "-e": "--env",
    "-v": "--volume",
    "-w": "--workdir",
    "-t": "--tag",          # NOTE: only valid for `build`; see _disambiguate_t
    "-f": "--file",
    "-u": "--user",
    "-m": "--memory",
    "-l": "--label",
    "-d": "--detach",
    "-i": "--interactive",
    "-a": "--all",
    "-q": "--quiet",
    "-s": "--size",
    "-n": "--last",
}


_TTY_CONTEXTS = {"run", "exec", "create"}


def _disambiguate_t(path: Sequence[str]) -> Dict[str, str]:
    """`-t` means --tag for `build` but --tty for `run`/`exec`.

    Returns an override map for the current command path. This is exactly the kind
    of detail a naive string comparison hides and a structural one has to face.
    """
    if len(path) >= 2 and path[1] in _TTY_CONTEXTS:
        return {"-t": "--tty"}
    return {}


def _takes_value(flag: str, path: Sequence[str]) -> bool:
    """Whether `flag` consumes the next token, given the command path.

    Context-sensitive because `-t` is a value flag for `docker build` (the tag) but
    a boolean for `docker run` (allocate a TTY). Treating it as a value flag under
    `run` silently eats the image name — `docker run -t nginx` parses as
    `-t=nginx` with no image at all.
    """
    if flag == "-t":
        return not (len(path) >= 2 and path[1] in _TTY_CONTEXTS)
    return flag in VALUE_FLAGS


def _expand_bundled_shorts(token: str, path: Sequence[str]) -> Optional[List[str]]:
    """Expand `-it` into ['-i', '-t'] when every character is a known boolean.

    Returns None if the token is not a safe bundle (e.g. `-p8080`, which is a flag
    with an attached value, or a bundle whose members consume a value).
    """
    if not token.startswith("-") or token.startswith("--") or len(token) <= 2:
        return None
    chars = token[1:]
    if not all(c in BUNDLEABLE_SHORT_BOOLS for c in chars):
        return None
    # A bundle is only unambiguous if none of its members consume a value. This is
    # context-sensitive: `-it` is a valid bundle under `run`/`exec` (where -t is
    # boolean) but `-t` under `build` takes the tag.
    candidates = [f"-{c}" for c in chars]
    if any(_takes_value(c, path) for c in candidates):
        return None
    return candidates


def _apply_aliases(path: Tuple[str, ...]) -> Tuple[str, ...]:
    """Collapse equivalent spellings of the same command path."""
    for prefix, replacement in COMMAND_ALIASES.items():
        if path[: len(prefix)] == prefix:
            path = replacement + path[len(prefix):]
            break
    return SUBCOMMAND_ALIASES.get(path, path)


def _split_path(tokens: List[str]) -> Tuple[Tuple[str, ...], List[str]]:
    """Peel the leading command path off the token list."""
    if not tokens:
        return tuple(), []

    path = [tokens[0]]
    rest = tokens[1:]

    # First subcommand (skip if it's actually a flag, e.g. `docker --version`).
    if rest and not rest[0].startswith("-"):
        path.append(rest[0])
        rest = rest[1:]
        # Second level, for `docker network create` and friends.
        if path[-1] in NESTED_SUBCOMMANDS and rest and not rest[0].startswith("-"):
            path.append(rest[0])
            rest = rest[1:]

    return tuple(path), rest


def parse_command(command: str) -> ParsedCommand:
    """Parse a command string into a comparable structure.

    Never raises. An unparseable command comes back with `parse_ok=False` and empty
    parts, which scores as a miss rather than crashing the eval run.
    """
    command = (command or "").strip()
    if not command:
        return ParsedCommand(tuple(), tuple(), tuple(), raw=command, parse_ok=False)

    try:
        tokens = shlex.split(command)
    except ValueError:
        # Unbalanced quote. Fall back to whitespace splitting so a malformed model
        # output still gets compared rather than aborting the run.
        tokens = command.split()
        return ParsedCommand(
            *_parse_tokens(tokens), raw=command, parse_ok=False
        )

    if not tokens:
        return ParsedCommand(tuple(), tuple(), tuple(), raw=command, parse_ok=False)

    return ParsedCommand(*_parse_tokens(tokens), raw=command, parse_ok=True)


def _parse_tokens(
    tokens: List[str],
) -> Tuple[Tuple[str, ...], Tuple[Tuple[str, Optional[str]], ...], Tuple[str, ...]]:
    path, rest = _split_path(tokens)
    path = _apply_aliases(path)

    t_override = _disambiguate_t(path)

    flags: List[Tuple[str, Optional[str]]] = []
    positionals: List[str] = []

    i = 0
    while i < len(rest):
        token = rest[i]

        if token == "--":
            # Everything after `--` is positional by definition.
            positionals.extend(rest[i + 1:])
            break

        if token.startswith("-") and token != "-":
            # `--flag=value` form
            if token.startswith("--") and "=" in token:
                name, value = token.split("=", 1)
                flags.append((_canonicalize_flag(name), value))
                i += 1
                continue

            bundled = _expand_bundled_shorts(token, path)
            if bundled is not None:
                for b in bundled:
                    canon = t_override.get(b) or _canonicalize_flag(b)
                    flags.append((canon, None))
                i += 1
                continue

            canon = t_override.get(token) or _canonicalize_flag(token)

            if _takes_value(token, path) and i + 1 < len(rest):
                flags.append((canon, rest[i + 1]))
                i += 2
                continue

            flags.append((canon, None))
            i += 1
            continue

        positionals.append(token)
        i += 1

    return path, tuple(sorted(flags)), tuple(positionals)


def normalized_string(command: str) -> str:
    """A canonical string form — useful for grouping and for dedup keys."""
    p = parse_command(command)
    parts = list(p.path)
    for flag, value in p.flags:
        parts.append(flag if value is None else f"{flag}={value}")
    parts.extend(p.positionals)
    return " ".join(parts)

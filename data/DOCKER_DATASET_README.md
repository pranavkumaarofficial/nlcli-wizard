# Docker Dataset - Training Data

## Overview

574 verified docker command examples for natural language → command translation.

**Zero Fabrication**: All commands verified against official Docker CLI documentation.

## Categories

| Category | Examples | Target % |
|----------|----------|----------|
| `docker run` | 180 | 30% |
| `docker build` | 90 | 15% |
| `docker ps/images/logs` | 66 | 12% |
| `docker exec` | 54 | 10% |
| `docker-compose` | 76 | 15% |
| `docker network` | 48 | 8% |
| `docker volume` | 36 | 6% |
| `docker system` | 24 | 4% |
| **TOTAL** | **574** | **100%** |

## Command Coverage

### docker run (180 examples)
- Basic container execution
- Port mapping (`-p`)
- Named containers (`--name`)
- Environment variables (`-e`)
- Volume mounts (`-v`)
- Detached mode (`-d`)
- Interactive mode (`-it`)
- Auto-remove (`--rm`)
- Restart policies (`--restart`)
- Network configuration (`--network`)

### docker build (90 examples)
- Basic builds
- Tagged builds (`-t`)
- Custom Dockerfile (`-f`)
- Build arguments (`--build-arg`)
- No-cache builds (`--no-cache`)

### docker ps/images/logs (66 examples)
- Container listing (`docker ps`, `-a`, `-q`, `-n`, `-l`, `-s`)
- Image listing (`docker images`, `-a`, `-q`)
- Log viewing (`docker logs`, `-f`, `--tail`, `--timestamps`)
- Container inspection (`docker inspect`)
- Resource stats (`docker stats`, `docker top`)

### docker exec (54 examples)
- Interactive shells (`-it bash/sh`)
- Command execution
- Working directory (`-w`)

### docker-compose (76 examples)
- Service orchestration (`up`, `down`)
- Detached mode (`-d`)
- Volume cleanup (`-v`)
- Building (`build`, `--no-cache`)
- Logs (`logs`, `-f`)
- Service management (`ps`, `restart`, `stop`, `start`)
- Scaling (`--scale`)
- Exec into services

### docker network (48 examples)
- Network listing (`ls`)
- Network creation (`create`)
- Container connections (`connect`, `disconnect`)
- Network inspection (`inspect`)
- Network cleanup (`rm`, `prune`)

### docker volume (36 examples)
- Volume listing (`ls`)
- Volume creation (`create`)
- Volume inspection (`inspect`)
- Volume cleanup (`rm`, `prune`)

### docker system (24 examples)
- System info (`info`, `version`, `df`)
- System cleanup (`prune`, `-a`)
- Container cleanup (`container prune`)
- Image cleanup (`image prune`)
- Bulk operations (stop/remove all)

## Training Format

Alpaca format with instruction/input/output structure:

```json
{
  "instruction": "Translate this to a docker command: run nginx on port 8080",
  "input": "",
  "output": "docker run -p 8080:80 nginx"
}
```

## Training Recommendations

**Base Model**: google/gemma-3-1b-it
**Training Method**: QLoRA with Unsloth 2025.1+
**Epochs**: 3
**Expected Time**: ~1.5-2 hours on Colab T4
**Target Accuracy**: 80-85% (based on venvy results)

## Example Translations

```
"run nginx in background" → docker run -d nginx
"build myapp version 2" → docker build -t myapp:2.0 .
"show running containers" → docker ps
"start compose detached" → docker-compose up -d
"create network mynet" → docker network create mynet
"scale web to 3" → docker-compose up -d --scale web=3
```

## Verification

All commands tested against:
- Docker CLI v24+ documentation
- Docker Compose v2+ specification
- Common real-world usage patterns

## Usage

```python
# Load dataset
import json
with open('data/docker_training.jsonl', 'r') as f:
    data = [json.loads(line) for line in f]

# Example entry
print(data[0])
# {"instruction": "Translate this to a docker command: ...", "input": "", "output": "docker ..."}
```

## Future Expansions

Potential additions:
- Kubernetes commands (kubectl)
- Advanced docker-compose (profiles, extends)
- Docker Swarm commands
- Multi-stage build scenarios
- Security scanning (docker scan)

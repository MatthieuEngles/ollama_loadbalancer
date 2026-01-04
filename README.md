# Ollama Load Balancer

An intelligent GPU-aware proxy/orchestrator for Ollama that dynamically manages multiple GPU instances.

*Un proxy/orchestrateur intelligent pour Ollama qui gère dynamiquement plusieurs instances GPU.*

---

## Features / Fonctionnalités

| English | Français |
|---------|----------|
| **100% Ollama API Compatible** - Drop-in replacement on port 11434 | **100% Compatible API Ollama** - Remplacement direct sur le port 11434 |
| **Auto GPU Detection** - Automatically detects all available NVIDIA GPUs | **Détection Auto GPU** - Détecte automatiquement tous les GPUs NVIDIA |
| **Dynamic GPU Allocation** - Auto-allocates 1, 2, or 3 GPUs per model | **Allocation GPU Dynamique** - Alloue automatiquement 1, 2 ou 3 GPUs par modèle |
| **Instance Lifecycle** - Spawns on-demand, cleans up after inactivity | **Cycle de Vie des Instances** - Création à la demande, nettoyage après inactivité |
| **Request Queuing** - Queues when resources unavailable | **File d'Attente** - Met en queue quand les ressources sont indisponibles |
| **Model Reuse** - Reuses instances for same model requests | **Réutilisation** - Réutilise les instances pour le même modèle |
| **Health Monitoring** - Periodic health checks | **Surveillance** - Vérifications de santé périodiques |
| **Streaming Support** - Full support for streaming responses | **Support Streaming** - Support complet des réponses en streaming |
| **System Monitoring** - CPU, RAM usage in status API | **Monitoring Système** - Usage CPU, RAM dans l'API status |
| **Context Tracking** - Track context size per request | **Suivi Contexte** - Suivi de la taille de contexte par requête |

---

## Architecture

```
Client (Standard Ollama API)
        │
        ▼
┌─────────────────────────────────┐
│  Proxy FastAPI (port 11434)     │
│  - Parse request                │
│  - Identify model               │
│  - Check GPU config             │
│  - Allocate available GPUs      │
│  - Spawn or reuse Ollama        │
│  - Proxy request/response       │
└─────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────┐
│  GPU Pool Manager               │
│  - Auto-detect GPUs (nvidia-smi)│
│  - GPU state (free/allocated)   │
│  - Active Ollama instances      │
│  - Dynamic ports (11500+)       │
└─────────────────────────────────┘
        │
   ┌────┴────┬─────────┐
   ▼         ▼         ▼
 GPU 0     GPU 1     GPU N
```

---

## Requirements / Prérequis

- Python 3.10+
- Ollama installed and in PATH / Ollama installé et dans le PATH
- NVIDIA GPUs with CUDA / GPUs NVIDIA avec CUDA
- nvidia-smi (for auto GPU detection) / nvidia-smi (pour la détection auto des GPUs)

---

## Quick Setup (Fresh Machine) / Installation Rapide (Machine Vierge)

**One-liner:**
```bash
git clone https://github.com/MatthieuEngles/ollama_loadbalancer.git && cd ollama_loadbalancer && make setup
```

**Or step by step / Ou étape par étape:**
```bash
# Clone the project / Cloner le projet
git clone https://github.com/MatthieuEngles/ollama_loadbalancer.git
cd ollama_loadbalancer

# (Optional) Edit models to pull on startup / (Optionnel) Modifier les modèles à télécharger
nano startup_models.txt

# Full setup / Installation complète
make setup
```

This script will:
- Install Ollama if not present
- Stop and disable the default `ollama.service` if it exists
- Install the load balancer as a systemd service
- Pull all models listed in `startup_models.txt`

*Ce script va:*
- *Installer Ollama s'il n'est pas présent*
- *Stopper et désactiver le service `ollama.service` par défaut s'il existe*
- *Installer le load balancer comme service systemd*
- *Télécharger tous les modèles listés dans `startup_models.txt`*

---

## Manual Installation / Installation Manuelle

```bash
cd ollama_loadbalancer

# Create virtual environment / Créer l'environnement virtuel
python -m venv venv
source venv/bin/activate

# Install dependencies / Installer les dépendances
pip install -r requirements.txt

# Install as service / Installer comme service
make install

# Edit configuration (optional) / Modifier la configuration (optionnel)
nano config.yaml
```

### Example config.yaml

```yaml
server:
  host: "0.0.0.0"
  port: 11434
  log_level: "INFO"

# GPU Pool Configuration (optional - auto-detected if not specified)
# Configuration GPU Pool (optionnel - auto-détecté si non spécifié)
# gpu_pool:
#   - id: 0
#   - id: 1
#   - id: 2

models:
  "llama3:70b":
    gpu_count: 3
    priority: high
  "llama3:8b":
    gpu_count: 1
    priority: normal
  default:
    gpu_count: 1
    priority: normal

behavior:
  when_busy: "queue"
  queue_timeout: 300
  instance_ttl: 120
```

---

## Usage / Utilisation

### Start / Démarrer

```bash
python main.py
```

### Make Requests / Faire des Requêtes

```bash
# Chat completion
curl http://localhost:11434/api/chat -d '{
  "model": "llama3:8b",
  "messages": [{"role": "user", "content": "Hello!"}]
}'

# Streaming chat
curl http://localhost:11434/api/chat -d '{
  "model": "mistral:7b",
  "messages": [{"role": "user", "content": "Tell me a story"}],
  "stream": true
}'

# Generate completion
curl http://localhost:11434/api/generate -d '{
  "model": "qwen2.5:14b",
  "prompt": "Explain quantum computing"
}'

# Embeddings
curl http://localhost:11434/api/embeddings -d '{
  "model": "nomic-embed-text",
  "prompt": "Hello world"
}'

# Pull a model / Télécharger un modèle
curl http://localhost:11434/api/pull -d '{
  "name": "llama3:8b"
}'

# List models / Lister les modèles
curl http://localhost:11434/api/tags

# List running models / Modèles en cours
curl http://localhost:11434/api/ps
```

### Monitor Status / Surveiller l'État

```bash
# Full system status / État complet
curl http://localhost:11434/api/status

# GPU pool / Pool GPU
curl http://localhost:11434/api/status/gpu

# Running instances / Instances actives
curl http://localhost:11434/api/status/instances

# Request queue / File d'attente
curl http://localhost:11434/api/status/queue
```

---

## API Endpoints

### Ollama Compatible

| Endpoint | Method | Description EN | Description FR |
|----------|--------|----------------|----------------|
| `/api/generate` | POST | Text generation | Génération de texte |
| `/api/chat` | POST | Chat completion | Complétion de chat |
| `/api/embeddings` | POST | Generate embeddings | Générer des embeddings |
| `/api/embed` | POST | Generate embeddings (alt) | Embeddings (alternatif) |
| `/api/tags` | GET | List available models | Lister les modèles |
| `/api/ps` | GET | List running models | Modèles en cours |
| `/api/pull` | POST | Pull a model | Télécharger un modèle |
| `/api/delete` | DELETE | Delete a model | Supprimer un modèle |
| `/api/show` | POST | Show model info | Info sur un modèle |

### Custom Status / État Personnalisé

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/status` | GET | Full system status (CPU, RAM, GPU, instances, queue) |
| `/api/status/gpu` | GET | GPU pool status |
| `/api/status/instances` | GET | Ollama instances with context info |
| `/api/status/queue` | GET | Request queue |
| `/api/queue/{id}` | DELETE | Cancel queued request |
| `/api/admin/clear-queue` | POST | Clear all queued requests |
| `/health` | GET | Health check |
| `/ready` | GET | Readiness check |

### Status Response Example / Exemple de Réponse Status

```json
{
  "status": "running",
  "uptime_seconds": 3600,
  "system": {
    "cpu": {
      "usage_percent": 45.2,
      "cores_logical": 32,
      "cores_physical": 16
    },
    "memory": {
      "total_gb": 128.5,
      "used_gb": 64.3,
      "usage_percent": 50.1
    }
  },
  "gpu_pool": {
    "total": 6,
    "free": 4,
    "allocated": 2
  },
  "instances": {
    "instances": [
      {
        "id": "abc123",
        "model_name": "llama3:8b",
        "gpu_ids": [0],
        "state": "ready",
        "context_length": 8192,
        "model_size": "4.7GB",
        "current_request_context": 2048,
        "last_request_context": 1500,
        "active_requests": 1,
        "request_count": 42
      }
    ]
  },
  "queue": {
    "size": 0,
    "max_size": 20
  },
  "config": {
    "total_gpus": 6,
    "when_busy": "queue"
  }
}
```

---

## Configuration Reference

### Server / Serveur

| Option | Description | Default |
|--------|-------------|---------|
| `host` | Bind address | `0.0.0.0` |
| `port` | Listen port | `11434` |
| `log_level` | DEBUG/INFO/WARNING/ERROR | `INFO` |
| `log_format` | json/text | `json` |

### GPU Pool / Pool GPU

GPUs are **automatically detected** using `nvidia-smi` if `gpu_pool` is not specified.

*Les GPUs sont **automatiquement détectés** via `nvidia-smi` si `gpu_pool` n'est pas spécifié.*

```yaml
# Optional: Manual GPU configuration
# Optionnel: Configuration manuelle des GPUs
gpu_pool:
  - id: 0
  - id: 1
  - id: 2
```

### Models / Modèles

```yaml
models:
  "llama3.3:70b":       # Exact match / Correspondance exacte
    gpu_count: 3
    priority: high

  "llama3:8b":
    gpu_count: 1
    priority: normal

  default:              # Fallback for unlisted models
    gpu_count: 1
    priority: normal
```

### Behavior / Comportement

| Option | Description EN | Description FR | Default |
|--------|----------------|----------------|---------|
| `when_busy` | `queue` or `reject` | `queue` ou `reject` | `queue` |
| `queue_timeout` | Max queue wait (s) | Attente max en queue (s) | `300` |
| `instance_ttl` | Idle shutdown time (s) | Temps avant arrêt (s) | `120` |
| `max_queue_size` | Max queued requests | Requêtes max en queue | `20` |
| `health_check_interval` | Health check interval (s) | Intervalle health check (s) | `10` |
| `startup_timeout` | Max startup time (s) | Temps de démarrage max (s) | `180` |

---

## Systemd Service

Copy the provided service file / Copier le fichier service fourni:

```bash
sudo cp ollama-lb.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable ollama-lb
sudo systemctl start ollama-lb
```

Or create your own `/etc/systemd/system/ollama-lb.service`:

```ini
[Unit]
Description=Ollama Load Balancer
After=network.target

[Service]
Type=simple
User=your-user
WorkingDirectory=/path/to/ollama_loadbalancer
Environment="PATH=/path/to/ollama_loadbalancer/venv/bin:/usr/local/bin:/usr/bin:/bin"
Environment="OLLAMA_MODELS=/home/your-user/.ollama/models"
ExecStart=/path/to/ollama_loadbalancer/venv/bin/python main.py
Restart=always
RestartSec=10

# Logging
StandardOutput=journal
StandardError=journal
SyslogIdentifier=ollama-lb

[Install]
WantedBy=multi-user.target
```

### Service Commands / Commandes Service

```bash
# Start / Démarrer
sudo systemctl start ollama-lb

# Stop / Arrêter
sudo systemctl stop ollama-lb

# Restart / Redémarrer
sudo systemctl restart ollama-lb

# Status / État
sudo systemctl status ollama-lb

# Logs / Journaux
sudo journalctl -u ollama-lb -f
```

---

## Environment Variables / Variables d'Environnement

| Variable | Description | Default |
|----------|-------------|---------|
| `OLLAMA_LB_CONFIG` | Path to config file | `config.yaml` |
| `OLLAMA_MODELS` | Path to models directory | `~/.ollama/models` |

---

## VRAM Calibration / Calibration VRAM

Run the calibration script to measure actual VRAM usage:

*Exécutez le script de calibration pour mesurer l'utilisation réelle de la VRAM:*

```bash
# List installed models / Lister les modèles installés
python calibrate.py --list

# Calibrate all models / Calibrer tous les modèles
python calibrate.py

# Calibrate specific models / Calibrer des modèles spécifiques
python calibrate.py --models "llama3:8b,mistral:7b,gemma3:27b"

# Generate config from calibration / Générer la config
python calibrate.py --output config.yaml
```

---

## Troubleshooting / Dépannage

### Instance fails to start / L'instance ne démarre pas

```bash
# Check logs / Vérifier les logs
sudo journalctl -u ollama-lb -f

# Verify GPUs / Vérifier les GPUs
nvidia-smi

# Check GPU detection / Vérifier la détection GPU
nvidia-smi --query-gpu=index --format=csv,noheader
```

### Permission denied for models / Permission refusée pour les modèles

Set the `OLLAMA_MODELS` environment variable to a writable directory:

*Définir la variable `OLLAMA_MODELS` vers un répertoire accessible en écriture:*

```bash
export OLLAMA_MODELS=/home/your-user/.ollama/models
```

Or in systemd service:

```ini
Environment="OLLAMA_MODELS=/home/your-user/.ollama/models"
```

### Requests timing out / Requêtes en timeout

```yaml
behavior:
  queue_timeout: 600
  startup_timeout: 300
```

### Streaming errors / Erreurs de streaming

The load balancer handles Content-Length header modifications automatically. If you encounter streaming issues, ensure you're using the latest version.

*Le load balancer gère automatiquement les modifications du header Content-Length. En cas de problèmes de streaming, assurez-vous d'utiliser la dernière version.*

---

## Project Structure / Structure du Projet

```
ollama_loadbalancer/
├── main.py              # FastAPI application entry point
├── config.py            # Configuration loading and auto GPU detection
├── gpu_pool.py          # GPU allocation management
├── ollama_manager.py    # Ollama instance lifecycle and model info
├── proxy.py             # Request proxying and context tracking
├── request_queue.py     # Request queuing system
├── calibrate.py         # VRAM calibration script
├── config.yaml          # Configuration file
├── startup_models.txt   # Models to pull on setup
├── ollama-lb.service    # Systemd service file
├── requirements.txt     # Python dependencies
├── Makefile             # Build and management commands
├── scripts/
│   └── setup.sh         # Full setup script for fresh machines
└── docs/
    ├── api_formats.md   # Ollama vs OpenAI API documentation
    └── gpu_inference_guide.md
```

---

## Security TODO / Sécurité TODO

> **Warning / Attention**: This is a development version. Security features need to be implemented before production use.
>
> *Ceci est une version de développement. Les fonctionnalités de sécurité doivent être implémentées avant la production.*

### To Implement / À implémenter

- [ ] **Authentication** - API key or JWT authentication for all endpoints
  - *Authentification par clé API ou JWT pour tous les endpoints*

- [ ] **Rate Limiting** - Per-client request rate limiting to prevent abuse
  - *Limitation du débit par client pour éviter les abus*

- [ ] **Input Validation** - Sanitize model names and prompts to prevent injection
  - *Valider les noms de modèles et prompts pour éviter les injections*

- [ ] **TLS/HTTPS** - Enable HTTPS with proper certificates
  - *Activer HTTPS avec des certificats valides*

- [ ] **Network Isolation** - Bind to localhost or use firewall rules
  - *Isolation réseau - écouter sur localhost ou utiliser des règles firewall*

- [ ] **Admin Endpoints Protection** - Separate auth for `/api/admin/*` routes
  - *Protection des endpoints admin avec authentification séparée*

- [ ] **Request Size Limits** - Limit prompt/context size to prevent DoS
  - *Limiter la taille des prompts/contexte pour éviter les DoS*

- [ ] **Audit Logging** - Log all requests with client IP and model used
  - *Journalisation de toutes les requêtes avec IP client et modèle utilisé*

- [ ] **Model Allowlist** - Restrict which models can be loaded
  - *Liste blanche des modèles autorisés à charger*

- [ ] **Resource Quotas** - Per-user GPU time and request limits
  - *Quotas par utilisateur pour le temps GPU et les requêtes*

### Quick Security Hardening / Sécurisation rapide

```bash
# Bind to localhost only (edit config.yaml)
# Écouter uniquement sur localhost
server:
  host: "127.0.0.1"

# Use reverse proxy with auth (nginx example)
# Utiliser un reverse proxy avec auth (exemple nginx)
location /api/ {
    auth_basic "Ollama API";
    auth_basic_user_file /etc/nginx/.htpasswd;
    proxy_pass http://127.0.0.1:11434;
}

# Firewall rules (allow only local network)
# Règles firewall (autoriser uniquement le réseau local)
sudo ufw allow from 192.168.1.0/24 to any port 11434
sudo ufw deny 11434
```

---

## License / Licence

MIT

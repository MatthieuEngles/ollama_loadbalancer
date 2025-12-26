# Ollama Load Balancer

An intelligent GPU-aware proxy/orchestrator for Ollama that dynamically manages multiple GPU instances.

*Un proxy/orchestrateur intelligent pour Ollama qui gère dynamiquement plusieurs instances GPU.*

---

## Features / Fonctionnalités

| English | Français |
|---------|----------|
| **100% Ollama API Compatible** - Drop-in replacement on port 11434 | **100% Compatible API Ollama** - Remplacement direct sur le port 11434 |
| **Dynamic GPU Allocation** - Auto-allocates 1, 2, or 3 GPUs per model | **Allocation GPU Dynamique** - Alloue automatiquement 1, 2 ou 3 GPUs par modèle |
| **Instance Lifecycle** - Spawns on-demand, cleans up after inactivity | **Cycle de Vie des Instances** - Création à la demande, nettoyage après inactivité |
| **Request Queuing** - Queues when resources unavailable | **File d'Attente** - Met en queue quand les ressources sont indisponibles |
| **Model Reuse** - Reuses instances for same model requests | **Réutilisation** - Réutilise les instances pour le même modèle |
| **Health Monitoring** - Periodic health checks | **Surveillance** - Vérifications de santé périodiques |
| **Streaming Support** - Full support for streaming responses | **Support Streaming** - Support complet des réponses en streaming |

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
│  - GPU state (free/allocated)   │
│  - Active Ollama instances      │
│  - Dynamic ports (11500+)       │
└─────────────────────────────────┘
        │
   ┌────┴────┬─────────┐
   ▼         ▼         ▼
 GPU 0     GPU 1     GPU 2
```

---

## Requirements / Prérequis

- Python 3.10+
- Ollama installed and in PATH / Ollama installé et dans le PATH
- NVIDIA GPUs with CUDA / GPUs NVIDIA avec CUDA

---

## Installation

```bash
# Clone the project / Cloner le projet
cd ollama_loadbalancer

# Create virtual environment / Créer l'environnement virtuel
python -m venv venv
source venv/bin/activate

# Install dependencies / Installer les dépendances
pip install -r requirements.txt

# Edit configuration / Modifier la configuration
nano config.yaml
```

### Example config.yaml

```yaml
server:
  host: "0.0.0.0"
  port: 11434
  log_level: "INFO"

gpu_pool:
  - id: 0
  - id: 1
  - id: 2

models:
  - pattern: "*:70b"
    gpu_count: 3
  - pattern: "*:32b"
    gpu_count: 2
  - pattern: "*:27b"
    gpu_count: 2
  - pattern: "*"
    gpu_count: 1

behavior:
  when_busy: "queue"
  queue_timeout: 300
  instance_ttl: 300
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
| `/api/status` | GET | Full system status |
| `/api/status/gpu` | GET | GPU pool status |
| `/api/status/instances` | GET | Ollama instances |
| `/api/status/queue` | GET | Request queue |
| `/api/queue/{id}` | DELETE | Cancel queued request |
| `/health` | GET | Health check |
| `/ready` | GET | Readiness check |

---

## Configuration Reference

### Server / Serveur

| Option | Description | Default |
|--------|-------------|---------|
| `host` | Bind address | `0.0.0.0` |
| `port` | Listen port | `11434` |
| `log_level` | DEBUG/INFO/WARNING/ERROR | `INFO` |
| `log_format` | json/text | `json` |

### Models / Modèles

```yaml
models:
  - pattern: "llama3.3:70b"   # Exact match / Correspondance exacte
    gpu_count: 3
    priority: 10

  - pattern: "*:70b"          # Wildcard / Joker
    gpu_count: 3

  - pattern: "*"              # Default / Par défaut
    gpu_count: 1
```

### Behavior / Comportement

| Option | Description EN | Description FR | Default |
|--------|----------------|----------------|---------|
| `when_busy` | `queue` or `reject` | `queue` ou `reject` | `queue` |
| `queue_timeout` | Max queue wait (s) | Attente max en queue (s) | `300` |
| `instance_ttl` | Idle shutdown time (s) | Temps avant arrêt (s) | `300` |
| `max_queue_size` | Max queued requests | Requêtes max en queue | `100` |
| `startup_timeout` | Max startup time (s) | Temps de démarrage max (s) | `120` |

---

## Systemd Service

Create / Créer `/etc/systemd/system/ollama-lb.service`:

```ini
[Unit]
Description=Ollama Load Balancer
After=network.target

[Service]
Type=simple
User=your-user
WorkingDirectory=/path/to/ollama_loadbalancer
ExecStart=/path/to/venv/bin/python main.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable ollama-lb
sudo systemctl start ollama-lb
```

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
journalctl -u ollama-lb -f

# Verify GPUs / Vérifier les GPUs
nvidia-smi
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
├── config.py            # Configuration loading and validation
├── gpu_pool.py          # GPU allocation management
├── ollama_manager.py    # Ollama instance lifecycle
├── proxy.py             # Request proxying logic
├── request_queue.py     # Request queuing system
├── calibrate.py         # VRAM calibration script
├── config.yaml          # Configuration file
└── requirements.txt     # Python dependencies
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

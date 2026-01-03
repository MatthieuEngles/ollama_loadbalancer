# API Formats : Ollama vs OpenAI

## Vue d'ensemble

Le load balancer supporte **deux formats d'API** :
- **API Ollama native** : `/api/*`
- **API OpenAI-compatible** : `/v1/*`

Ollama lui-même supporte les deux formats, donc le load balancer proxy les requêtes directement.

---

## Comparaison des endpoints

| Fonction | Ollama API | OpenAI API |
|----------|------------|------------|
| Chat | `/api/chat` | `/v1/chat/completions` |
| Completion | `/api/generate` | `/v1/completions` |
| Embeddings | `/api/embed` ou `/api/embeddings` | `/v1/embeddings` |
| Liste modèles | `/api/tags` | `/v1/models` |

---

## Différences de format

### Chat Request

**Ollama (`/api/chat`)** :
```json
{
  "model": "gemma3:12b",
  "messages": [
    {"role": "user", "content": "Hello"}
  ],
  "stream": false
}
```

**OpenAI (`/v1/chat/completions`)** :
```json
{
  "model": "gemma3:12b",
  "messages": [
    {"role": "user", "content": "Hello"}
  ],
  "stream": false,
  "temperature": 0.7,
  "max_tokens": 1000
}
```

### Chat Response

**Ollama** :
```json
{
  "model": "gemma3:12b",
  "message": {
    "role": "assistant",
    "content": "Hello! How can I help?"
  },
  "done": true,
  "total_duration": 1234567890
}
```

**OpenAI** :
```json
{
  "id": "chatcmpl-123",
  "object": "chat.completion",
  "created": 1704067200,
  "model": "gemma3:12b",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Hello! How can I help?"
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 10,
    "completion_tokens": 8,
    "total_tokens": 18
  }
}
```

### Embeddings

**Ollama (`/api/embed`)** :
```json
{
  "model": "nomic-embed-text",
  "input": "Hello world"
}
```

**OpenAI (`/v1/embeddings`)** :
```json
{
  "model": "nomic-embed-text",
  "input": "Hello world"
}
```
*(Format identique pour la requête)*

---

## Détection automatique dans le Load Balancer

Le load balancer détecte automatiquement le format via le **path de la requête** :

```python
# proxy.py
OPENAI_ENDPOINTS = {
    "/v1/chat/completions",
    "/v1/completions",
    "/v1/embeddings",
}

MODEL_ENDPOINTS = {
    "/api/generate",
    "/api/chat",
    "/api/embeddings",
    "/api/embed",
}
```

### Routage dans main.py

```python
# Ollama native
@app.post("/api/chat")
async def ollama_chat(request: Request):
    return await proxy.handle_model_request(request, "/api/chat")

# OpenAI compatible
@app.post("/v1/chat/completions")
async def openai_chat_completions(request: Request):
    return await proxy.handle_openai_request(request, "/v1/chat/completions")
```

---

## Fonctionnement interne

1. **Requête arrive** sur `/api/chat` ou `/v1/chat/completions`
2. **Extraction du modèle** : même clé `model` dans les deux formats
3. **Allocation GPU** : identique pour les deux
4. **Proxy vers Ollama** : Ollama gère nativement les deux formats
5. **Réponse** : renvoyée telle quelle au client

**Pas de transformation nécessaire** - Ollama traduit automatiquement entre les formats.

---

## Compatibilité clients

### Clients OpenAI SDK

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="not-needed"  # Ollama n'utilise pas d'API key
)

response = client.chat.completions.create(
    model="gemma3:12b",
    messages=[{"role": "user", "content": "Hello"}]
)
```

### Clients Ollama natifs

```python
import ollama

response = ollama.chat(
    model="gemma3:12b",
    messages=[{"role": "user", "content": "Hello"}]
)
```

### curl Ollama

```bash
curl http://localhost:11434/api/chat -d '{
  "model": "gemma3:12b",
  "messages": [{"role": "user", "content": "Hello"}],
  "stream": false
}'
```

### curl OpenAI

```bash
curl http://localhost:11434/v1/chat/completions -d '{
  "model": "gemma3:12b",
  "messages": [{"role": "user", "content": "Hello"}],
  "stream": false
}'
```

---

## Paramètres spécifiques

### Paramètres OpenAI (traduits par Ollama)

| OpenAI | Ollama équivalent |
|--------|-------------------|
| `max_tokens` | `num_predict` |
| `temperature` | `temperature` |
| `top_p` | `top_p` |
| `stop` | `stop` |
| `frequency_penalty` | `frequency_penalty` |
| `presence_penalty` | `presence_penalty` |

### Paramètres Ollama uniquement

- `num_ctx` : taille du contexte
- `num_gpu` : nombre de layers GPU (géré par le load balancer)
- `keep_alive` : durée de rétention en mémoire
- `format` : format de sortie (json)

---

## Fonctionnalités avancées supportées

Le load balancer supporte **toutes les fonctionnalités** des modèles Ollama :

### Tools / Function Calling

Permet au modèle d'appeler des fonctions définies par l'utilisateur.

**Ollama (`/api/chat`)** :
```json
{
  "model": "qwen2.5:14b",
  "messages": [{"role": "user", "content": "What's the weather in Paris?"}],
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "Get current weather for a location",
        "parameters": {
          "type": "object",
          "properties": {
            "location": {"type": "string", "description": "City name"}
          },
          "required": ["location"]
        }
      }
    }
  ],
  "stream": false
}
```

**OpenAI (`/v1/chat/completions`)** :
```json
{
  "model": "qwen2.5:14b",
  "messages": [{"role": "user", "content": "What's the weather in Paris?"}],
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "Get current weather for a location",
        "parameters": {
          "type": "object",
          "properties": {
            "location": {"type": "string"}
          },
          "required": ["location"]
        }
      }
    }
  ]
}
```

**Réponse avec tool call** :
```json
{
  "message": {
    "role": "assistant",
    "content": "",
    "tool_calls": [
      {
        "function": {
          "name": "get_weather",
          "arguments": "{\"location\": \"Paris\"}"
        }
      }
    ]
  }
}
```

**Modèles supportant tools** : `qwen2.5`, `llama3.1`, `mistral`, `command-r`, etc.

---

### Vision (Images)

Permet d'envoyer des images au modèle pour analyse.

**Ollama (`/api/chat`)** :
```json
{
  "model": "llava:13b",
  "messages": [
    {
      "role": "user",
      "content": "What's in this image?",
      "images": ["base64_encoded_image_data"]
    }
  ],
  "stream": false
}
```

**OpenAI (`/v1/chat/completions`)** :
```json
{
  "model": "llava:13b",
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "text", "text": "What's in this image?"},
        {
          "type": "image_url",
          "image_url": {
            "url": "data:image/jpeg;base64,..."
          }
        }
      ]
    }
  ]
}
```

**Modèles vision** : `llava`, `llava-llama3`, `bakllava`, `moondream`, `gemma3` (avec vision)

---

### Think / Reasoning (Chain of Thought)

Modèles avec raisonnement explicite avant la réponse.

**Ollama (`/api/chat`)** :
```json
{
  "model": "deepseek-r1:14b",
  "messages": [{"role": "user", "content": "Solve: 2x + 5 = 13"}],
  "stream": false
}
```

**Réponse avec thinking** :
```json
{
  "message": {
    "role": "assistant",
    "content": "<think>\nI need to solve for x.\n2x + 5 = 13\n2x = 13 - 5\n2x = 8\nx = 4\n</think>\n\nThe solution is x = 4."
  }
}
```

**Modèles thinking** : `deepseek-r1`, `qwq`, modèles avec tag `:thinking`

---

### Embeddings

Génération de vecteurs d'embeddings pour la recherche sémantique.

**Ollama (`/api/embed`)** :
```json
{
  "model": "nomic-embed-text",
  "input": ["Hello world", "Goodbye world"]
}
```

**OpenAI (`/v1/embeddings`)** :
```json
{
  "model": "nomic-embed-text",
  "input": ["Hello world", "Goodbye world"]
}
```

**Réponse** :
```json
{
  "embeddings": [
    [0.123, -0.456, 0.789, ...],
    [0.321, -0.654, 0.987, ...]
  ]
}
```

**Modèles embeddings** : `nomic-embed-text`, `mxbai-embed-large`, `all-minilm`, `snowflake-arctic-embed`

---

### Structured Output (JSON Mode)

Force le modèle à répondre en JSON valide.

**Ollama** :
```json
{
  "model": "gemma3:12b",
  "messages": [{"role": "user", "content": "List 3 colors"}],
  "format": "json",
  "stream": false
}
```

**OpenAI** :
```json
{
  "model": "gemma3:12b",
  "messages": [{"role": "user", "content": "List 3 colors"}],
  "response_format": {"type": "json_object"}
}
```

---

## Matrice de compatibilité

| Fonctionnalité | Ollama API | OpenAI API | Modèles requis |
|----------------|------------|------------|----------------|
| Chat | `/api/chat` | `/v1/chat/completions` | Tous |
| Completion | `/api/generate` | `/v1/completions` | Tous |
| Streaming | `stream: true` | `stream: true` | Tous |
| Tools | `tools: [...]` | `tools: [...]` | qwen2.5, llama3.1, mistral |
| Vision | `images: [...]` | `content: [{type: image_url}]` | llava, gemma3, moondream |
| Thinking | Automatique | Automatique | deepseek-r1, qwq |
| Embeddings | `/api/embed` | `/v1/embeddings` | nomic-embed-text, mxbai |
| JSON mode | `format: json` | `response_format: {type: json_object}` | Tous |

---

## Notes importantes

1. **API key** : Ollama ne requiert pas d'API key, utiliser une valeur quelconque avec les SDK OpenAI

2. **Streaming** : Les deux formats supportent le streaming (`stream: true`)

3. **Modèles** : Utiliser les noms de modèles Ollama (ex: `gemma3:12b`, pas `gpt-4`)

4. **Proxy transparent** : Le load balancer transmet toutes ces fonctionnalités sans transformation, Ollama gère la compatibilité

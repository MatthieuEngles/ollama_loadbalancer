# Guide GPU et Inférence LLM

## Quantization GGUF

### Types de quantization

| Symbole | Bits | VRAM | Qualité | Description |
|---------|------|------|---------|-------------|
| **fp16** | 16 | 100% | Référence | Full precision |
| **q8_0** | 8 | ~50% | Excellente | 8-bit uniforme |
| **q6_K** | 6 | ~40% | Très bonne | 6-bit K-quant |
| **q5_K_M** | 5 | ~35% | Bonne | 5-bit K-quant medium |
| **q5_K_S** | 5 | ~35% | Bonne- | 5-bit K-quant small |
| **q5_0/q5_1** | 5 | ~35% | Correcte | 5-bit legacy |
| **q4_K_M** | 4 | ~30% | Correcte | 4-bit K-quant medium |
| **q4_K_S** | 4 | ~30% | Acceptable | 4-bit K-quant small |
| **q4_0/q4_1** | 4 | ~30% | Acceptable- | 4-bit legacy |
| **q3_K_L** | 3 | ~25% | Dégradée | 3-bit K-quant large |
| **q3_K_M** | 3 | ~25% | Dégradée | 3-bit K-quant medium |
| **q3_K_S** | 3 | ~25% | Mauvaise | 3-bit K-quant small |
| **q2_K** | 2 | ~20% | Très dégradée | 2-bit K-quant |

### Suffixes

- **K** = K-quantization (meilleure qualité à bits égaux)
- **_M** = Medium - équilibre taille/qualité
- **_S** = Small - plus compact, moins précis
- **_L** = Large - plus gros, meilleure qualité
- **_0/_1** = Anciennes méthodes (legacy)

### Exemple : Llama 3.1 70B

| Variante | Taille fichier | VRAM estimée |
|----------|---------------|--------------|
| 70b-instruct-fp16 | ~140 GB | ~140 GB |
| 70b-instruct-q8_0 | ~70 GB | ~74 GB |
| 70b-instruct-q4_K_M | ~40 GB | ~44 GB |
| 70b-instruct-q4_0 | ~38 GB | ~42 GB |
| 70b-instruct-q2_K | ~26 GB | ~30 GB |

---

## Impact sur la vitesse d'inférence

### Le vrai saut de paradigme

Le gros gain de vitesse est **entre q8 et q4**, pas entre q4 et q3 :

```
fp16 → q8  : 2x plus rapide (significatif)
q8   → q4  : 2x plus rapide (significatif)
q4   → q3  : 1.2x plus rapide (marginal, qualité dégradée)
q3   → q2  : 1.1x plus rapide (marginal, qualité très dégradée)
```

### Pourquoi ?

L'inférence LLM est **memory-bound** : la vitesse dépend de la quantité de données lues depuis la VRAM, pas de la puissance de calcul.

| Quant | Bits | Taille relative | Vitesse relative |
|-------|------|-----------------|------------------|
| fp16 | 16 | 100% | 1x |
| q8_0 | 8 | 50% | **~2x** |
| q6_K | 6 | 37% | ~2.6x |
| q4_K_M | 4 | 25% | **~4x** |
| q3_K | 3 | 19% | ~4.5x |
| q2_K | 2 | 12% | ~5x |

### NVFP4 sur Blackwell (GB10)

Les GPU Blackwell ont des **Tensor Cores FP4 natifs** :
- q4 sur Blackwell = accélération hardware native
- q4 sur Ampere (A4000) = déquantization software → calcul fp16

### Recommandation

**q4_K_M = meilleur compromis vitesse/qualité**

Sous q4, la perte de qualité dépasse le gain de vitesse.

---

## Pourquoi l'inférence est memory-bound

### Le problème fondamental

À chaque token généré, le GPU doit :
1. **Lire** tous les poids du modèle depuis la VRAM
2. **Calculer** une multiplication matricielle
3. **Écrire** le résultat (le prochain token)

```
Llama 70B en fp16 = 140 GB de poids
Pour générer 1 token → lire 140 GB
Pour générer 100 tokens → lire 140 GB × 100 = 14 TB
```

### Le ratio compute/memory

| Opération | Temps théorique (A4000) |
|-----------|-------------------------|
| Lire 140 GB @ 448 GB/s | **312 ms** |
| Calculer la multiplication | ~5 ms |

Le GPU passe **98% du temps à attendre les données** et 2% à calculer.

### Visualisation

```
Timeline génération d'un token (70B fp16 sur A4000):

Mémoire:  [████████████████████████████████████████] 312ms
Calcul:   [██] 5ms

Total: ~317ms par token = ~3 tokens/seconde
```

### Avec quantization q4

```
Llama 70B en q4 = 35 GB de poids

Mémoire:  [██████████] 78ms  (4x moins à lire)
Calcul:   [██] 5ms (pareil)

Total: ~83ms par token = ~12 tokens/seconde
```

---

## CUDA Cores : quand ça compte ?

### Sous-utilisation massive en inférence

| GPU | Capacité calcul | Utilisation réelle (inférence) |
|-----|-----------------|-------------------------------|
| A4000 | 153 TFLOPS (fp16) | ~5-10% |
| A100 | 312 TFLOPS (fp16) | ~5-10% |

Les cores passent leur temps à **attendre** les données.

### Bande passante vs CUDA cores

| GPU | CUDA Cores | Bande passante | Ratio (GB/s par 1000 cores) |
|-----|------------|----------------|----------------------------|
| A4000 | 6,144 | 448 GB/s | 73 |
| RTX 4090 | 16,384 | 1,008 GB/s | 62 |
| A100 | 6,912 | 2,039 GB/s | **295** |
| H100 | 16,896 | 3,350 GB/s | **198** |

Les A100/H100 ont **plus de bande passante par core** → optimisés pour l'inférence.

### Le point de bascule (batch size)

```
Temps total = max(temps_mémoire, temps_calcul)

Inférence batch=1:
  temps_mémoire = 100ms (lire les poids)
  temps_calcul  = 5ms   (matmul)
  → Total = 100ms (memory-bound)

Inférence batch=32:
  temps_mémoire = 100ms (mêmes poids, une seule lecture)
  temps_calcul  = 50ms  (32x plus de calcul)
  → Total = 100ms (encore memory-bound, mais moins)

Inférence batch=256:
  temps_mémoire = 100ms
  temps_calcul  = 200ms
  → Total = 200ms (compute-bound!)
```

| GPU | Batch size pour devenir compute-bound |
|-----|--------------------------------------|
| A4000 | ~64-128 |
| A100 | ~256-512 |

**Problème** : En inférence interactive (chat), batch = 1 ou 2.

### Où les CUDA cores aident vraiment

| Phase | Bottleneck | CUDA cores utiles ? |
|-------|------------|---------------------|
| Prompt processing (prefill) | Mixte | Oui, modérément (~50-70% GPU) |
| Génération (decode) | Mémoire | Non (~5-10% GPU) |
| Multi-users batch | Mixte/Compute | Oui |

---

## Training vs Inference

| | Training | Inference |
|--|----------|-----------|
| Batch size | Grand (32-512) | Petit (1-8) |
| Réutilisation des poids | Haute | Très basse |
| Bottleneck | Compute-bound | **Memory-bound** |
| Utilisation GPU | 80-95% | 5-15% |

En training, les mêmes poids servent pour tout le batch → amortissement.
En inference, tu génères token par token → relecture complète à chaque fois.

---

## Recommandations pour 3x A4000 (48 GB total)

### Par taille de modèle

| Modèle | Quantization | GPUs | Tokens/s estimés |
|--------|--------------|------|------------------|
| ≤7B | q8_0 | 1 | ~80-100 |
| 8-13B | q8_0 ou q4_K_M | 1 | ~40-60 |
| 14-30B | q4_K_M | 1-2 | ~20-40 |
| 32-34B | q4_K_M | 2 | ~15-25 |
| 70B | q4_K_M | 3 | ~8-12 |
| 70B | q2_K | 2 | ~10-15 (qualité dégradée) |

### Calcul rapide VRAM

```
VRAM ≈ Paramètres × Bits / 8 + overhead (10-20%)

Exemple 70B q4:
  70B × 4 bits / 8 = 35 GB
  + 20% overhead = 42 GB
  → Tient sur 3x A4000 (48 GB)
```

### Calcul rapide tokens/s

```
Tokens/s ≈ Bande_passante / Taille_modèle

Exemple 70B q4 sur 3x A4000:
  Bande passante: 3 × 448 = 1344 GB/s (théorique)
  Bande passante effective: ~800 GB/s (multi-GPU overhead)
  Taille modèle: 35 GB

  → 800 / 35 ≈ 23 tokens/s (théorique max)
  → Réel: ~10-12 tokens/s (overhead divers)
```

---

## Résumé

1. **Quantization q4_K_M** = sweet spot (4x moins de VRAM, 4x plus rapide, qualité acceptable)
2. **Bande passante mémoire** > CUDA cores pour l'inférence
3. **Plus de VRAM** permet des modèles plus gros, pas forcément plus rapides
4. **Multi-GPU** aide surtout pour faire tenir le modèle, gain de vitesse limité (~1.5-2x pour 2 GPUs)

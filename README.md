# Agent RH interne

Solution minimale, réaliste et démontrable en 2 jours pour répondre à des questions RH internes avec:

- RAG simple sur documents RH
- filtrage par rôle
- contexte utilisateur
- déploiement Docker

## Choix d'architecture

- `FastAPI` pour exposer l'API et servir une petite interface web.
- `RAG lexical` sur fichiers Markdown locaux: pas de vector DB, pas d'embeddings payants.
- `OpenRouter` par défaut pour la génération finale avec `openrouter/free`. `Gemini API` reste disponible comme fournisseur alternatif.
- `Contrôle d'accès` au niveau des documents et des sources selon le rôle utilisateur (`employee`, `manager`, `hr`).
- `Contexte employé` injecté depuis `data/users.json`.

## Workflow agentique

1. Détecter l'intention et la catégorie RH.
2. Charger le profil utilisateur.
3. Filtrer les documents autorisés pour son rôle.
4. Retrouver les passages pertinents.
5. Enrichir avec les données personnelles utiles.
6. Générer une réponse structurée avec sources.

## Structure

- `app/`: API, moteur agentique, récupération des documents, UI
- `knowledge_base/`: documents RH simulés
- `data/`: profils employés
- `tests/`: cas de tests
- `report/`: rapport académique et PDF
- `scripts/`: génération du PDF

## Lancer localement

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --reload
```

Puis ouvrir `http://localhost:8000`.

## Lancer avec Docker

```bash
cp .env.example .env
docker compose up --build
```

## API

- `GET /health`
- `GET /users`
- `POST /chat`

Exemple:

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"user_id":"e001","question":"Combien de jours de congés me reste-t-il ?"}'
```

## Tests

```bash
pytest
```

Les tests tournent sans clé API et vérifient:

- récupération RAG
- filtrage par rôle
- contexte utilisateur
- réponses attendues sur les cas fournis

## Générer le PDF

```bash
python3 scripts/generate_pdf_report.py
```

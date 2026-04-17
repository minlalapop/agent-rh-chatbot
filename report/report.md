# Rapport Projet - Agent RH interne

## 1. Objectif

Construire en moins de 2 jours un chatbot RH interne simple, gratuit et démontrable, capable de:

- répondre à des questions RH fréquentes
- faire de la recherche documentaire RH
- adapter la réponse au profil employé
- filtrer les informations selon le rôle
- être déployé avec Docker

## 2. Contraintes respectées

- 100% gratuit
- aucun modèle local
- aucun fine-tuning
- architecture simple
- outils existants
- déploiement Docker

## 3. Architecture finale

Le système est composé d'un seul service applicatif FastAPI.

Pipeline:

1. L'utilisateur choisit un profil simulé et pose une question.
2. L'agent détecte l'intention RH.
3. Le profil utilisateur est chargé depuis `data/users.json`.
4. Les documents RH sont filtrés selon le rôle.
5. Les passages pertinents sont récupérés par recherche lexicale.
6. La réponse est générée avec:
   - Gemini API si une clé gratuite est fournie
   - sinon un mode démo déterministe
7. Les sources et les étapes du workflow sont renvoyées.

## 4. Workflow agentique

Le workflow suit exactement:

`intent -> access -> retrieval -> response`

Détail:

- Intent: classifier la question en `contrat`, `paie`, `conges`, `organisation`, ou `general`
- Access: charger le rôle (`employee`, `manager`, `hr`) et filtrer les documents autorisés
- Retrieval: chercher les paragraphes RH les plus utiles
- Response: assembler le contexte profil + documents + réponse finale

## 5. Implémentation RAG

Le RAG reste volontairement minimal:

- documents RH en Markdown
- métadonnées simples (`allowed_roles`, `tags`)
- chunking par sections
- scoring lexical avec pondération par recouvrement et tag

Pourquoi ce choix:

- aucun coût
- pas de vector DB
- pas d'embeddings payants
- suffisant pour une démonstration académique

## 6. Gestion des accès

La gestion des accès repose sur les métadonnées des documents:

- documents salariés visibles par `employee`, `manager`, `hr`
- documents managers visibles par `manager`, `hr`
- documents RH internes visibles par `hr` seulement

Effet démontré:

- un salarié voit la procédure générale de correction de paie
- un RH voit aussi la procédure interne détaillée

## 7. Stack technologique

- FastAPI: API et interface simple
- Python standard + requests: logique de workflow et appel API
- Markdown: base documentaire RH
- pytest: tests
- Docker + docker-compose: déploiement
- Gemini API gratuit: génération cloud optionnelle

## 8. Plan d'implémentation sur 2 jours

### Jour 1

- créer les documents RH et profils utilisateurs
- coder le moteur `intent -> access -> retrieval -> response`
- exposer l'API FastAPI
- ajouter une petite interface web

### Jour 2

- écrire les cas de test
- ajouter Dockerfile et docker-compose
- rédiger le rapport
- générer le PDF
- faire une démonstration finale

## 9. Cas de tests couverts

Les tests couvrent notamment:

- téléchargement du contrat
- période d'essai
- N+1
- erreur de bulletin
- solde de congés
- congé sans solde
- passage à temps partiel
- fin de CDD
- validation manager

## 10. Erreurs à éviter

- ajouter une base vectorielle inutile
- utiliser des services payants
- faire du fine-tuning
- multiplier les microservices
- mélanger logique RH et logique de sécurité
- oublier les cas sans clé API

## 11. Conclusion

Cette solution est adaptée à un étudiant car elle est:

- rapide à construire
- simple à expliquer
- réaliste pour une démo
- compatible avec les contraintes académiques

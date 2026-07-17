# Tafatafa AI — Système RAG local

Clone simplifié de NotebookLM, développé en local avec Streamlit, LangChain, ChromaDB et Ollama.
Projet réalisé dans le cadre du TP ERP / M1 MIAGE.

## Fonctionnalités

- 📁 Upload et indexation de documents (PDF, TXT, Markdown)
- 🔍 **Mode recherche sémantique** : retrouve les passages les plus pertinents sans génération LLM
- 🤖 **Mode RAG** : réponses générées par un LLM local (Ollama), strictement basées sur le contexte des documents
- 💬 **Mode chat libre** : discussion avec le LLM sans document indexé
- 📚 Transparence des sources : affichage des passages utilisés pour chaque réponse RAG
- 🗂️ Historique de discussions multiples, persistant entre les sessions
- 🗑️ Création et suppression de discussions

## Prérequis

- Python 3.10+
- [Ollama](https://ollama.com) installé et lancé en local
- Le modèle `qwen2.5-coder:3b` téléchargé

### Installation d'Ollama et du modèle

```bash
# Installer Ollama (voir https://ollama.com pour ton OS)

# Télécharger le modèle utilisé par l'application
ollama pull qwen2.5-coder:3b

# Démarrer le serveur Ollama (si non lancé automatiquement)
ollama serve
```

## Installation du projet

1. Cloner le dépôt et se placer dans le dossier du projet :

```bash
git clone <url-du-depot>
cd <nom-du-dossier>
```

2. Créer un environnement virtuel (recommandé) :

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

3. Installer les dépendances :

```bash
pip install streamlit pillow langchain-community langchain-text-splitters langchain-huggingface langchain-chroma langchain-ollama langchain-core
```

> 💡 Tu peux aussi générer un fichier `requirements.txt` avec `pip freeze > requirements.txt` une fois l'installation testée, puis le committer pour que d'autres puissent faire `pip install -r requirements.txt`.

4. Vérifier que le dossier `assets/` contient bien le logo `logoTenaIzy.png` référencé dans le code.

## Lancement de l'application

Assure-toi qu'Ollama tourne en arrière-plan, puis :

```bash
streamlit run app.py
```

L'application s'ouvre automatiquement dans le navigateur (par défaut sur `http://localhost:8501`).

## Utilisation

1. **Indexer des documents** : dans la barre latérale, uploade un ou plusieurs fichiers (PDF, TXT, MD) puis clique sur **Indexer les documents**.
2. **Choisir un mode** :
   - Toggle **désactivé** + documents indexés → recherche sémantique (affiche les passages les plus proches de la question)
   - Toggle **activé** + documents indexés → mode RAG complet (réponse générée par le LLM à partir du contexte)
   - Toggle **activé** + aucun document → chat libre avec le LLM
3. **Gérer les discussions** : utilise **➕ Nouveau chat** pour démarrer une nouvelle conversation, clique sur un chat existant dans la liste pour y revenir, ou **🗑️** pour le supprimer.

## Structure du projet

```
.
├── app.py              # Application principale (tout le code y est regroupé)
├── assets/
│   └── logoTenaIzy.png # Logo affiché dans l'interface
├── chroma_db/          # Base vectorielle générée (créée automatiquement, ignorée par Git)
├── chats.json          # Historique des discussions (créé automatiquement, ignoré par Git)
└── README.md
```

## Architecture technique (pipeline RAG)

```
Documents (PDF/TXT/MD)
   ↓  load_documents()
Documents LangChain
   ↓  split_documents()
Chunks (700 caractères, overlap 100)
   ↓  create_vector_store()
Base vectorielle ChromaDB (embeddings HuggingFace all-MiniLM-L6-v2)
   ↓  similarity_search_with_score()
Chunks pertinents + scores
   ↓  format_context() + build_rag_prompt()
Prompt structuré (contexte + question)
   ↓  get_llm() → ChatOllama (qwen2.5-coder:3b)
Réponse générée, contrainte au contexte fourni
```

## Choix techniques principaux

| Choix | Justification |
|---|---|
| **Un seul fichier `app.py`** | Conforme au sujet du TP |
| **ChromaDB avec `persist_directory`** | Base vectorielle légère, locale, sans serveur externe à gérer |
| **`delete_collection()` avant réindexation** | Évite les doublons ; plus sûr que supprimer les fichiers directement (évite les erreurs de verrouillage sous Windows) |
| **`all-MiniLM-L6-v2`** | Modèle d'embeddings léger et rapide, suffisant pour un usage local |
| **`qwen2.5-coder:3b` via Ollama** | Modèle local léger (3B paramètres), confidentiel (aucune donnée envoyée à un tiers), gratuit |
| **`temperature=0.1`** | Réponses factuelles et déterministes, adaptées à un pipeline RAG |
| **Stockage JSON pour l'historique** | Simplicité pour un TP, pas besoin de base de données relationnelle |

## Limites connues

- Les sources affichées dans l'historique ne sont conservées que pour la session en cours (non persistées dans `chats.json`)
- Pas de confirmation avant suppression d'une discussion
- La réindexation remplace entièrement la base vectorielle existante (pas d'ajout incrémental de documents)

## Auteur

Herilala — M1 MIAGE
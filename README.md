# Local RAG System

Projet de TP - Master 1 MIAGE

## Description

Application RAG (Retrieval-Augmented Generation) développée avec :

- Streamlit
- LangChain
- ChromaDB
- Ollama
- HuggingFace Embeddings

Le projet permet :

- d'importer des documents PDF, TXT et Markdown ;
- de les indexer dans une base vectorielle ;
- d'effectuer une recherche sémantique ;
- d'interagir avec les documents via un modèle LLM local.

## Lancement

```bash
streamlit run app.py
```
import streamlit as st
from PIL import Image
from langchain_community.document_loaders import (
    PyMuPDFLoader,
    TextLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
import tempfile
import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
import json
import uuid
from datetime import datetime


#===========================
# Gestion de la persistance des chats
#===========================
CHATS_FILE = "chats.json"

def load_all_chats():
    """
    Charge tous les chats sauvegardés depuis le disque.
    Retourne un dictionnaire vide si le fichier n'existe pas encore.
    """
    if os.path.exists(CHATS_FILE):
        with open(CHATS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_all_chats(chats):
    """
    Sauvegarde tous les chats sur le disque.
    """
    with open(CHATS_FILE, "w", encoding="utf-8") as f:
        json.dump(chats, f, ensure_ascii=False, indent=2)

def create_new_chat():
    """
    Crée un nouveau chat vide et le définit comme chat actif.
    """
    chat_id = str(uuid.uuid4())

    st.session_state.all_chats[chat_id] = {
        "title": "Nouvelle discussion",
        "created_at": datetime.now().isoformat(),
        "messages": []
    }

    st.session_state.current_chat_id = chat_id
    st.session_state.messages = []

    save_all_chats(st.session_state.all_chats)

def switch_chat(chat_id):
    """
    Bascule vers un chat existant et recharge ses messages.
    """
    st.session_state.current_chat_id = chat_id
    st.session_state.messages = st.session_state.all_chats[chat_id]["messages"]

def delete_chat(chat_id):
    """
    Supprime un chat. Si c'était le chat actif, bascule vers
    un autre chat existant, ou en crée un nouveau si c'était le dernier.
    """
    was_current = (chat_id == st.session_state.current_chat_id)

    del st.session_state.all_chats[chat_id]
    save_all_chats(st.session_state.all_chats)

    if was_current:
        if st.session_state.all_chats:
            # Bascule vers le chat le plus récent restant
            latest_id = max(
                st.session_state.all_chats,
                key=lambda cid: st.session_state.all_chats[cid]["created_at"]
            )
            switch_chat(latest_id)
        else:
            # Plus aucun chat : on en recrée un vide
            create_new_chat()

def update_current_chat_title(first_message):
    """
    Donne un titre au chat basé sur le premier message posé,
    tant qu'il porte encore le titre par défaut.
    """
    chat = st.session_state.all_chats[st.session_state.current_chat_id]

    if chat["title"] == "Nouvelle discussion":
        # Tronque à 40 caractères pour un titre lisible dans la sidebar
        chat["title"] = first_message[:40] + ("..." if len(first_message) > 40 else "")

#===========================
# Construction du prompt RAG
#===========================
def build_rag_prompt():
    """
    Crée le template de prompt pour le pipeline RAG.
    Le contexte et la question sont injectés dynamiquement.
    """
    system_prompt = """Tu es un assistant qui répond aux questions UNIQUEMENT à partir du contexte fourni ci-dessous.

Règles strictes :
- Si l'information demandée n'est pas présente dans le contexte, réponds exactement : "Je ne trouve pas cette information dans les documents fournis."
- Ne jamais inventer ou compléter avec des connaissances externes.
- Sois concis et précis.
- Si possible, appuie-toi sur des éléments exacts du contexte (chiffres, noms, dates).

Contexte :
{context}
"""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{question}")
    ])

    return prompt

#===========================
# Formatage du contexte pour le prompt
#===========================
def format_context(results):
    """
    Transforme les documents retrouvés en une chaîne de texte
    utilisable dans le prompt.
    results : liste de tuples (Document, score) issus de similarity_search_with_score
    """
    context_parts = []

    for doc, score in results:
        source = doc.metadata.get("source", "inconnu")
        page = doc.metadata.get("page", "?")
        context_parts.append(
            f"[Source: {source}, page {page}]\n{doc.page_content}"
        )

    return "\n\n---\n\n".join(context_parts)

#===========================
# Connexion au modèle Ollama
#===========================
def get_llm():
    """
    Instancie le modèle Ollama local.
    """
    llm = ChatOllama(
        model="qwen2.5-coder:3b",
        temperature=0.1  # réponses factuelles, peu de "créativité"
    )
    return llm

#===========================
# Fonction pour charger les documents
#===========================
def load_documents(uploaded_files):
    documents = []

    for uploaded_file in uploaded_files:

        suffix = os.path.splitext(uploaded_file.name)[1]

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:

            tmp_file.write(uploaded_file.getbuffer())

            tmp_path = tmp_file.name

        if suffix.lower() == ".pdf":
            loader = PyMuPDFLoader(tmp_path)

        else:
            loader = TextLoader(tmp_path, encoding="utf-8")

        docs = loader.load()

        for doc in docs:
            doc.metadata["source"] = uploaded_file.name

        documents.extend(docs)

    return documents

#===========================
# Creation de la base vectorielle avec Chroma
#===========================
def create_vector_store(chunks):
    """
    Crée la base vectorielle ChromaDB à partir des chunks.
    Supprime l'ancienne collection via l'API Chroma (pas via le système de fichiers)
    pour éviter les erreurs de verrouillage sous Windows.
    """
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    persist_dir = "chroma_db"

    # Si une base existe déjà en session, on vide sa collection
    # avant de recréer — évite les doublons sans toucher aux fichiers directement
    if st.session_state.get("vector_store") is not None:
        try:
            st.session_state.vector_store.delete_collection()
        except Exception:
            pass  # la collection n'existait peut-être pas encore

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_dir
    )

    st.session_state.vector_store = vector_store

    return vector_store

#===========================
# Fonction pour charger les documents
#===========================
def split_documents(documents):

    text_splitter = RecursiveCharacterTextSplitter(

        chunk_size=700,
        chunk_overlap=100,

        length_function=len,

        separators=[
            "\n\n",
            "\n",
            ". ",
            "? ",
            "! ",
            "; ",
            ", ",
            " ",
            ""
        ]
    )

    chunks = text_splitter.split_documents(documents)

    with st.spinner("Création des embeddings..."):

        create_vector_store(chunks)

    st.success("Base vectorielle créée avec succès !")

    return chunks

#===========================
# Chargement du logo
#===========================
logo = Image.open("assets/logoTenaIzy.png")

# ==========================
# Configuration de la page
# ==========================
st.set_page_config(
    page_title="Local RAG System",
    page_icon=logo,
    layout="wide"
)


# ==========================
# Initialisation de la session
# ==========================
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "all_chats" not in st.session_state:
    st.session_state.all_chats = load_all_chats()

if "current_chat_id" not in st.session_state:
    if st.session_state.all_chats:
        # Reprend le chat le plus récent au lancement
        latest_id = max(
            st.session_state.all_chats,
            key=lambda cid: st.session_state.all_chats[cid]["created_at"]
        )
        st.session_state.current_chat_id = latest_id
        st.session_state.messages = st.session_state.all_chats[latest_id]["messages"]
    else:
        create_new_chat()

# ==========================
# Titre
# ==========================
st.title("Tafatafa AI")
st.caption("Votre assistant IA local")

# ==========================
# Barre latérale
# ==========================
with st.sidebar:

    st.header("Discussions")

    if st.button("➕ Nouveau chat", use_container_width=True):
        create_new_chat()
        st.rerun()

    st.divider()

    # Liste des chats existants, du plus récent au plus ancien
    sorted_chats = sorted(
        st.session_state.all_chats.items(),
        key=lambda item: item[1]["created_at"],
        reverse=True
    )

    for chat_id, chat_data in sorted_chats:
        is_active = (chat_id == st.session_state.current_chat_id)

        label = ("🟢 " if is_active else "") + chat_data["title"]

        col1, col2 = st.columns([5, 1])

        with col1:
            if st.button(label, key=f"chat_{chat_id}", use_container_width=True):
                switch_chat(chat_id)
                st.rerun()

        with col2:
            if st.button("🗑️", key=f"delete_{chat_id}"):
                delete_chat(chat_id)
                st.rerun()

    st.divider()

    uploaded_files = st.file_uploader(
        "Choisissez vos documents",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True
    )

    index_button = st.button("Indexer les documents")
    if index_button:

        if not uploaded_files:
            st.warning("Veuillez sélectionner au moins un document.")

        else:

            with st.spinner("Lecture des documents..."):

                documents = load_documents(uploaded_files)
                chunks = split_documents(documents)

            st.success(f"{len(documents)} document(s) chargé(s).")
            st.success(f"{len(chunks)} chunk(s) créés.")

            with st.expander("🔍 Détails du traitement (debug)"):
                for i, chunk in enumerate(chunks):
                    st.write(
                        f"Chunk {i+1} : {len(chunk.page_content)} caractères "
                        f"(page {chunk.metadata.get('page')})"
                    )

                for doc in documents:
                    st.write("------")
                    st.write(doc.metadata)
                    st.write(doc.page_content[:500])

    use_llm = st.toggle("Activer le LLM", value=False)

# ==========================
# Historique de discussion
# ==========================
if "messages" not in st.session_state:
    st.session_state.messages = []

# Affichage de l'historique
for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Zone de saisie
question = st.chat_input("Posez votre question...")
if question:

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    # Mise à jour du titre au premier message + sauvegarde
    if len(st.session_state.messages) == 1:
        update_current_chat_title(question)

    st.session_state.all_chats[st.session_state.current_chat_id]["messages"] = st.session_state.messages
    save_all_chats(st.session_state.all_chats)

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):

        # Cas 1 : aucun document indexé
        if st.session_state.vector_store is None:

            if use_llm:
                # Chat libre : LLM sans RAG, sans contexte documentaire
                st.info("Mode chat libre (sans document)")

                # with st.spinner("Génération de la réponse..."):

                #     llm = get_llm()
                #     response = llm.invoke(question)

                with st.spinner("Génération de la réponse..."):
                    try:
                        llm = get_llm()
                        response = llm.invoke(question)
                        answer = response.content
                    except Exception as e:
                        answer = (
                            "⚠️ Impossible de contacter Ollama. "
                            "Vérifie qu'il est bien lancé (`ollama serve`) "
                            "et que le modèle `qwen2.5-coder:3b` est installé."
                        )
                        st.error(f"Erreur technique : {e}")

                answer = response.content

                st.markdown(answer)

                st.session_state.messages.append(
                    {"role": "assistant", "content": answer}
                )
                st.session_state.all_chats[st.session_state.current_chat_id]["messages"] = st.session_state.messages
                save_all_chats(st.session_state.all_chats)

            else:
                st.warning(
                    "Aucun document indexé. Activez le LLM pour discuter librement, "
                    "ou indexez un document pour la recherche sémantique / le RAG."
                )

        # Cas 2 : des documents sont indexés
        else:

            # Mode recherche sémantique
            if not use_llm:

                st.info("Mode recherche sémantique")

                results = st.session_state.vector_store.similarity_search_with_score(
                    question,
                    k=3
                )

                for i, (doc, score) in enumerate(results):
                    st.write(f"#### Résultat {i+1}")
                    st.write(f"Score : {score}")
                    st.write(f"Source : {doc.metadata.get('source')}")
                    st.write(doc.page_content)
                    st.divider()

                # Sauvegarde d'un résumé textuel des résultats comme "réponse"
                summary = "\n\n".join(
                    f"**{doc.metadata.get('source')}** (score {score:.3f}) : {doc.page_content[:200]}..."
                    for doc, score in results
                )
                st.session_state.messages.append(
                    {"role": "assistant", "content": summary}
                )
                st.session_state.all_chats[st.session_state.current_chat_id]["messages"] = st.session_state.messages
                save_all_chats(st.session_state.all_chats)

            # Mode RAG (LLM + documents)
            else:

                st.info("Mode RAG")

                with st.spinner("Recherche des passages pertinents..."):

                    results = st.session_state.vector_store.similarity_search_with_score(
                        question,
                        k=3
                    )
                    context = format_context(results)

                # with st.spinner("Génération de la réponse..."):

                #     llm = get_llm()
                #     prompt = build_rag_prompt()
                #     chain = prompt | llm

                #     response = chain.invoke({
                #         "context": context,
                #         "question": question
                #     })

                with st.spinner("Génération de la réponse..."):
                    try:
                        llm = get_llm()
                        prompt = build_rag_prompt()
                        chain = prompt | llm

                        response = chain.invoke({
                            "context": context,
                            "question": question
                        })
                        answer = response.content
                    except Exception as e:
                        answer = (
                            "⚠️ Impossible de contacter Ollama. "
                            "Vérifie qu'il est bien lancé (`ollama serve`)."
                        )
                        st.error(f"Erreur technique : {e}")

                answer = response.content

                st.markdown(answer)

                # Transparence : affichage des sources utilisées
                with st.expander("📚 Sources utilisées"):
                    for i, (doc, score) in enumerate(results):
                        source = doc.metadata.get("source", "inconnu")
                        page = doc.metadata.get("page", "?")

                        st.markdown(f"**{i+1}. {source}** — page {page} · score {score:.3f}")
                        st.caption(doc.page_content[:300] + ("..." if len(doc.page_content) > 300 else ""))
                        st.divider()

                st.session_state.messages.append(
                    {"role": "assistant", "content": answer}
                )
                st.session_state.all_chats[st.session_state.current_chat_id]["messages"] = st.session_state.messages
                save_all_chats(st.session_state.all_chats)
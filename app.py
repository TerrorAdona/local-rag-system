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
    Repart d'une base propre à chaque indexation pour éviter les doublons.
    """
    import shutil

    persist_dir = "chroma_db"

    # Supprime l'ancienne base si elle existe, pour éviter l'accumulation de doublons
    if os.path.exists(persist_dir):
        shutil.rmtree(persist_dir)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

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

# ==========================
# Titre
# ==========================
st.title("Tafatafa AI")
st.caption("Votre assistant IA local")

# ==========================
# Barre latérale
# ==========================
with st.sidebar:

    st.header("Configuration")

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
            
            # for i, chunk in enumerate(chunks[:3]):

            #     st.write(f"### Chunk {i + 1}")

            #     st.write(chunk.metadata)

            #     st.write(chunk.page_content)

            #     st.divider()

            for i, chunk in enumerate(chunks):
                st.write(
                    f"Chunk {i+1} : "
                    f"{len(chunk.page_content)} caractères "
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

    with st.chat_message("user"):
        st.markdown(question)


    with st.chat_message("assistant"):

        # Vérification qu'une base existe
        if st.session_state.vector_store is None:

            st.warning(
                "Veuillez d'abord indexer un document."
            )

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

            # Mode RAG (LLM activé)
            else:

                st.info("Mode RAG")

                with st.spinner("Recherche des passages pertinents..."):

                    results = st.session_state.vector_store.similarity_search_with_score(
                        question,
                        k=3
                    )

                    context = format_context(results)

                with st.spinner("Génération de la réponse..."):

                    llm = get_llm()
                    prompt = build_rag_prompt()

                    # Le prompt est "rempli" puis envoyé au modèle
                    chain = prompt | llm

                    response = chain.invoke({
                        "context": context,
                        "question": question
                    })

                answer = response.content

                st.markdown(answer)

                # Sauvegarde dans l'historique du chat
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer
                    }
                )
            # Mode recherche sémantique
            # if not use_llm:

            #     st.info("Mode recherche sémantique")

            #     # results = st.session_state.vector_store.similarity_search(
            #     #     question,
            #     #     k=3
            #     # )
                
            #     results = st.session_state.vector_store.similarity_search_with_score(
            #         question,
            #         k=3
            #     )

            #     # results = [
            #     #     doc for doc, score in results
            #     #     if score < 1.2
            #     # ]

            #     for i, (doc, score) in enumerate(results):

            #         st.write(f"#### Résultat {i+1}")

            #         st.write(
            #             f"Score : {score}"
            #         )

            #         st.write(
            #             f"Source : {doc.metadata.get('source')}"
            #         )

            #         st.write(doc.page_content)

            #         st.divider()

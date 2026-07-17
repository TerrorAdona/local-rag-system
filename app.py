import streamlit as st
from PIL import Image
from langchain_community.document_loaders import (
    PyMuPDFLoader,
    TextLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
import tempfile
import os


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
# Fonction pour charger les documents
#===========================
def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        length_function=len,
        separators=[
            "\n\n",
            "\n",
            " ",
            ""
        ]
    )

    chunks = text_splitter.split_documents(documents)

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
        st.markdown("Je suis prêt, mais aucun document n'est encore indexé.")
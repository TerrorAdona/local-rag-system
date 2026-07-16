import streamlit as st
from PIL import Image

#===========================
# Chargement du logo
#===========================
logo = Image.open("assets/logoTenaIzy.png")
gl = Image.open("assets/logo.png")

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

    st.image(gl, width=50)
    st.header("Configuration")

    uploaded_files = st.file_uploader(
        "Choisissez vos documents",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True
    )

    index_button = st.button("Indexer les documents")

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
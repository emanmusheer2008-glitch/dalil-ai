import streamlit as st

st.set_page_config(
    page_title="Dalil AI",
    page_icon="🇸🇦",
    layout="wide"
)

# ---------- HEADER ----------
st.title("🇸🇦 Dalil AI")
st.subheader("Your AI Guide to Saudi Government & Public Services")

st.write(
    "Ask questions in **Arabic or English** and find relevant information "
    "from official Saudi government sources."
)

st.divider()

# ---------- SIDEBAR ----------
with st.sidebar:
    st.header("Dalil AI")
    
    page = st.radio(
        "Navigate",
        [
            "💬 Ask Dalil",
            "🔎 Explore Services",
            "📚 Knowledge Base",
            "📊 AI Evaluation"
        ]
    )

    st.divider()

    st.caption(
        "Dalil AI is an independent educational AI project. "
        "Always verify important information with the linked official source."
    )


# ---------- ASK DALIL ----------
if page == "💬 Ask Dalil":

    st.header("Ask Dalil")

    st.write(
        "Ask about Saudi government services, procedures, "
        "documents, education, business, visas, transportation and more."
    )

    question = st.text_input(
        "What would you like to know?",
        placeholder="Example: How can I renew my driving licence?"
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.caption("🎓 Education")

    with col2:
        st.caption("🏢 Business")

    with col3:
        st.caption("🚗 Transportation")

    col4, col5, col6 = st.columns(3)

    with col4:
        st.caption("🛂 Visas & Residents")

    with col5:
        st.caption("🏥 Health")

    with col6:
        st.caption("📄 Personal Documents")

    if st.button("🔍 Ask Dalil", type="primary"):

        if question.strip():

            st.info(
                "Dalil's AI knowledge base will be connected in the next step."
            )

        else:
            st.warning("Please enter a question.")


# ---------- EXPLORE ----------
elif page == "🔎 Explore Services":

    st.header("🔎 Explore Saudi Government Services")

    st.write(
        "Browse and search government services contained "
        "in the Dalil knowledge base."
    )

    st.info("Government service dataset will appear here.")


# ---------- KNOWLEDGE BASE ----------
elif page == "📚 Knowledge Base":

    st.header("📚 Dalil Knowledge Base")

    col1, col2, col3 = st.columns(3)

    col1.metric("Official Services", "0")
    col2.metric("Government Agencies", "0")
    col3.metric("Languages", "Arabic + English")

    st.write(
        "Dalil uses information collected from official Saudi government sources."
    )

    st.info("We will populate the knowledge base in the next step.")


# ---------- EVALUATION ----------
elif page == "📊 AI Evaluation":

    st.header("📊 AI Evaluation Dashboard")

    st.write(
        "This dashboard will measure how accurately Dalil retrieves "
        "and supports answers using official information."
    )

    col1, col2, col3 = st.columns(3)

    col1.metric("Questions Tested", "0")
    col2.metric("Retrieval Accuracy", "—")
    col3.metric("Source Accuracy", "—")

    st.info("Real evaluation results will appear after testing.")
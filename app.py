import streamlit as st
import numpy as np
import pickle
import plotly.graph_objects as go
from sklearn.decomposition import PCA

st.set_page_config(
    page_title="CTI Embedding Explorer",
    page_icon="🛡️",
    layout="wide"
)

@st.cache_resource
def load_assets():
    matrix = np.load("cbow_embedding_matrix.npy")
    with open("cbow_metadata.pkl", "rb") as f:
        meta = pickle.load(f)
    return matrix, meta["word_to_ix"], meta["ix_to_word"], meta["vocab"]

try:
    embedding_matrix, word_to_ix, ix_to_word, vocab = load_assets()
except Exception as e:
    st.error(f"خطأ أثناء قراءة الملفات: {e}")
    st.stop()

def get_similar(query, top_n=6):
    query = query.lower().strip()
    if query not in word_to_ix:
        return None

    target_idx = word_to_ix[query]
    target_vec = embedding_matrix[target_idx]

    dot_products = np.dot(embedding_matrix, target_vec)
    norms = np.linalg.norm(embedding_matrix, axis=1) * np.linalg.norm(target_vec)
    similarities = dot_products / (norms + 1e-9)

    best_indices = np.argsort(similarities)[::-1]

    results = []
    for idx in best_indices:
        token = ix_to_word[idx]
        if token != query:
            results.append((token, float(similarities[idx]), idx))
        if len(results) == top_n:
            break

    return results, target_idx

st.title("🛡️ Cyber Threat Intelligence (CTI) Embedding Explorer")
st.markdown("استكشاف الفضاء الدلالي لنموذج **CBOW** المدرّب على مقالات التهديدات السيبرانية.")

m1, m2, m3 = st.columns(3)
m1.metric("حجم القاموس", f"{len(vocab):,} كلمة")
m2.metric("أبعاد المتجه", f"{embedding_matrix.shape[1]}")
m3.metric("إجمالي العينات", f"{embedding_matrix.shape[0]:,}")

st.divider()

st.sidebar.header("⚙️ البحث والاستعلام")
search_word = st.sidebar.text_input("المصطلح المستهدف:", value="actor").strip().lower()
k_words = st.sidebar.slider("عدد الكلمات (Top-K):", min_value=3, max_value=20, value=6)

col_left, col_right = st.columns([1, 2])

query_output = get_similar(search_word, top_n=k_words)

with col_left:
    st.subheader("🔍 أقرب المصطلحات")
    if query_output is None:
        st.warning(f"الكلمة '{search_word}' غير موجودة بالقاموس.")
    else:
        neighbors, _ = query_output
        table_data = [{"المصطلح": w, "التشابه": f"{s:.4f}"} for w, s, _ in neighbors]
        st.table(table_data)

with col_right:
    st.subheader("📊 إسقاط الفضاء المتجهي (2D PCA)")
    if query_output is not None:
        neighbors, target_idx = query_output

        indices = [target_idx] + [idx for _, _, idx in neighbors]
        labels = [search_word] + [w for w, _, _ in neighbors]

        vectors = embedding_matrix[indices]
        pca = PCA(n_components=2, random_state=42)
        reduced = pca.fit_transform(vectors)

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=reduced[1:, 0],
            y=reduced[1:, 1],
            mode='markers+text',
            text=labels[1:],
            textposition="top center",
            marker=dict(size=14, color='#1f77b4'),
            name="المصطلحات المشابهة"
        ))

        fig.add_trace(go.Scatter(
            x=[reduced[0, 0]],
            y=[reduced[0, 1]],
            mode='markers+text',
            text=[f"★ {labels[0]}"],
            textposition="bottom center",
            marker=dict(size=20, color='#d62728', symbol='diamond'),
            name="الكلمة المستهدفة"
        ))

        fig.update_layout(
            height=460,
            margin=dict(l=20, r=20, t=30, b=20),
            xaxis=dict(title="PC 1"),
            yaxis=dict(title="PC 2"),
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)

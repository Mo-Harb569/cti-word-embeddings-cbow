# 🛡️ Cyber Threat Intelligence (CTI) Word Embedding Explorer (CBOW)

An end-to-end NLP and representation learning project that builds a domain-specific **Continuous Bag of Words (CBOW)** embedding model from scratch using **TensorFlow/Keras**, specifically tailored for Cyber Threat Intelligence (CTI) textual data scraped from real-world threat feeds.

---

## 📌 Problem Statement & Motivation
General-purpose language representations (e.g., standard Word2Vec trained on Wikipedia or Google News) often fail to capture the specialized semantics of cybersecurity:
- **"Actor"** is typically associated with performing arts rather than a **Threat Actor (APT)**.
- **"Injection"** clusters with medical terminology instead of code execution vulnerabilities (e.g., SQLi, Command Injection).
- **"Port"** relates to maritime transport rather than networking and attack surfaces.

This project addresses this semantic gap by developing a custom CTI corpus, engineering a clean vocabulary, training a domain-specific vector space, and deploying an interactive inspection tool.

---

## ⚙️ Architecture & Engineering Pipeline

The Hacker News (Scraping & Pagination)
│
▼
NLP Preprocessing (NLTK, Stopwords, Frequency Filtering)
│
▼
Sliding Window Context Extraction (Context = 2, Target = 1)
│
▼
CBOW Neural Network (Embedding -> GlobalAveragePooling1D -> Softmax)
│
▼
Vector Space Extraction & Cosine Similarity Analysis
│
▼
Interactive 2D Visualization (PCA + Streamlit + Plotly)

1. **Targeted Web Scraper:** Automated multi-page traversal collecting 60+ full threat intelligence articles via `requests` and `BeautifulSoup`.
2. **Preprocessing Pipeline:** Tokenization, stopword removal, and vocabulary pruning ($\text{frequency} \ge 3$) yielding a clean vocabulary of **3,338 unique domain tokens**.
3. **CBOW Model Design:** Replaced naive flattening layers with **`GlobalAveragePooling1D`** to enforce position-invariant representation averaging, preventing severe overfitting and scaling parameter efficiency.
4. **Interactive Dashboard:** Built with **Streamlit** and **Plotly**, projecting the 16-dimensional embedding space into 2D using **Principal Component Analysis (PCA)** with real-time Cosine Similarity search.

---

## 📊 Semantic Discovery (Sample Results)

The model successfully learned morphological and tactical relationships without any manual labeling:

| Target Word | Top Neighbors | Cosine Similarity | Contextual Relationship |
| :--- | :--- | :--- | :--- |
| **actor** | `actors` | 0.8449 | Plural inflection / Morphology |
| | `inject` | 0.7958 | Adversary tactical action |
| **access** | `modification` | 0.7439 | Post-exploitation impact |
| | `theft` | 0.6992 | Exfiltration / Impact |
| **malware** | `risk` | 0.6842 | Security threat classification |
| | `websocket` | 0.6103 | C2 communication protocol |

---

## 🚀 How to Run Locally

```bash
# 1. Clone the repository
git clone [https://github.com/Mo-Harb569/cti-word-embeddings-cbow.git](https://github.com/Mo-Harb569/cti-word-embeddings-cbow.git)
cd cti-word-embeddings-cbow

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch Streamlit
streamlit run app.py


3. انزل لأسفل الصفحة واضغط **Commit changes**.

#### ب) ضبط قسم الـ About (على يمين الشاشة):
1. اضغط على أيقونة الترس (⚙️) بجانب كلمة **About** في أقصى اليمين.
2. في خانة **Description** ضع:
   `Domain-specific CBOW word embedding model and interactive Streamlit explorer for Cyber Threat Intelligence (CTI).`
3. في خانة **Topics** أضف وسوم مثل: `nlp`, `word2vec`, `cbow`, `cybersecurity`, `threat-intelligence`, `streamlit`, `tensorflow`.
4. اضغط **Save changes**.

---

### الخطوة 2: النشر على LinkedIn

رابط المستودع الخاص بك أصبح جاهزاً رسمياً:
`[https://github.com/Mo-Harb569/cti-word-embeddings-cbow](https://github.com/Mo-Harb569/cti-word-embeddings-cbow)`

1. افتح حسابك على **LinkedIn** وأنشئ منشوراً جديداً.
2. أرفق صورة للواجهة (أو فيديو قصير 15-20 ثانية أثناء استعراض البحث بالواجهة).
3. الصق النص التالي:

```text
🛡️ Building a Domain-Specific Word Embedding for Cyber Threat Intelligence (CTI) from Scratch

General-purpose NLP embeddings often struggle with contextual domain nuances:
- "Actor" gets clustered with theater or cinema instead of Threat Actors.
- "Injection" relates to healthcare rather than malicious payload delivery.
- "Port" aligns with maritime logistics rather than network interfaces.

To address this domain gap, I engineered and trained an end-to-end Continuous Bag of Words (CBOW) embedding model from scratch using TensorFlow/Keras on specialized threat intelligence data:

Key Engineering Milestones:
1. Data Pipeline: Implemented a resilient multi-page scraper collecting full-length threat reports, followed by an NLTK cleaning pipeline and frequency pruning (vocab size: 3,338 terms).
2. Architecture Optimization: Utilized GlobalAveragePooling1D over context vectors to enforce position-invariant representation averaging, preventing overfitting and ensuring clean convergence.
3. Semantic Evaluation: Evaluated the embedding matrix using Cosine Similarity, observing strong semantic clustering (e.g., 'actor' aligning with 'inject', and 'access' grouping with 'theft' and 'modification').
4. Interactive Dashboard: Deployed a Streamlit application integrating 2D PCA projection via Plotly for real-time vector space exploration.

🔗 Full repository and code on GitHub:
https://github.com/Mo-Harb569/cti-word-embeddings-cbow

#MachineLearning #NLP #DeepLearning #CyberSecurity #DataScience #TensorFlow #Streamlit #ArtificialIntelligence

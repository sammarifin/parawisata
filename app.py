import streamlit as st
import pandas as pd
import re
import json
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

# Load kamus sentimen
try:
    with open("sentiment_lexicon.json", "r", encoding='utf-8') as f:
        kamus_sentimen = json.load(f)
except FileNotFoundError:
    st.error("❌ File kamus 'sentiment_lexicon.json' tidak ditemukan.")
    st.stop()

# Pembersihan teks
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text

# Fungsi sentimen berdasarkan kamus
def get_sentiment_kamus(text):
    skor = 0
    for kata in text.split():
        skor += kamus_sentimen.get(kata, 0)
    if skor > 0:
        return "positive"
    elif skor < 0:
        return "negative"
    else:
        return "neutral"

# UI Streamlit
st.title("🔍 Analisis Sentimen Ulasan Pariwisata")

uploaded_file = st.file_uploader("📤 Upload File CSV", type="csv")

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Gagal membaca file CSV: {e}")
        st.stop()

    if 'Review Text' not in df.columns:
        st.error("Kolom 'Review Text' tidak ditemukan dalam file CSV.")
        st.stop()

    # Preprocessing dan analisis
    df['clean_review'] = df['Review Text'].astype(str).apply(clean_text)
    df['sentiment'] = df['clean_review'].apply(get_sentiment_kamus)

    st.subheader("📄 Data + Sentimen")
    st.write(df[['Review Text', 'sentiment']])

    # Distribusi sentimen
    st.subheader("📊 Distribusi Sentimen")
    sentiment_counts = df['sentiment'].value_counts()
    st.bar_chart(sentiment_counts)

    # Input manual
    st.subheader("📝 Coba Analisis Kalimat Sendiri")
    user_input = st.text_input("Masukkan kalimat:")
    if user_input:
        cleaned = clean_text(user_input)
        result = get_sentiment_kamus(cleaned)
        st.success(f"Sentimen: **{result}**")

    # Visualisasi Top Kata
    st.subheader("💡 Top Kata per Sentimen")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    for i, label in enumerate(["positive", "neutral", "negative"]):
        text = ' '.join(df[df['sentiment'] == label]['clean_review'])
        words = text.split()
        word_counts = Counter(words).most_common(10)
        if word_counts:
            words, counts = zip(*word_counts)
            sns.barplot(x=list(counts), y=list(words), ax=axes[i], palette="viridis")
            axes[i].set_title(label.capitalize())
            axes[i].set_xlabel("Frekuensi")
            axes[i].tick_params(axis='y', labelsize=10)

    st.pyplot(fig)




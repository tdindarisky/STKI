



import pandas as pd
import numpy as np
import joblib
from plotly import express as px
import nltk
import streamlit as st
import nltk
from nltk import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
from wordcloud import WordCloud


# Memuat model Naive Bayes
model = joblib.load("model/model_naive_bayes.sav")

# Fungsi untuk melakukan prediksi sentimen
def predict_sentiment(sentiment_text):
    # Gunakan model Naive Bayes yang telah Anda muat dari file pickle
    predicted_sentiment = model.predict([sentiment_text])
    return predicted_sentiment[0]

#Judul aplikasi
st.title("Analisis Sentimen Twitter Tentang Kasus Kopi Sianida")

#Side bar
st.sidebar.title("Analisis Sentimen Tentang Kasus Kopi Sianida")
st.sidebar.write("Selamat datang di Aplikasi Analisis Sentimen!")

app_mode = st.sidebar.selectbox("Pilih Mode Aplikasi", ("🏠 Home","📈 Data Analysis","🤷 Sentiment Analysis"))
if app_mode == "🏠 Home":
    from PIL import Image
    image = Image.open('Twitter_logo.jpg')
    st.subheader("Mengapa Sentimen masyarakat tentang suatu kasus penting?")
    # Poin a
    st.markdown("1. Mengidentifikasi Isu yang Mendesak")
    st.markdown("Dengan memantau sentimen publik, pihak institusi dapat mengidentifikasi apa saja isu-isu yang mendesak yang perlu segera diatasi.")

    st.markdown("2. Mengukur Dampak Kebijakan dan Keputusan")
    st.markdown("Melakukan pemantauan sentimen publik membantu pihak institusi untuk menilai apakah kebijakan dan keputusan yang mereka buat akan diterima oleh masyarakat.")

    st.markdown("3. Mengukur Kinerja Suatu Instansi Yang Terkait Kepada Kasus")
    st.markdown("Pemantauan sentimen publik membantu institusi untuk mengukur kinerja mereka dari pandangan publik."
                "Umumnya, sentimen publik memberikan informasi penting tentang sejauh mana institusi berhasil memenuhi tujuannya dan memuaskan masyarakatnya.")
    
elif app_mode == "📈 Data Analysis":
    st.sidebar.write("Anda berada di Mode Data Analysis.")
    
    sentiment_twitter=pd.read_csv("data/Twitter_Data.csv")

    st.subheader('Data Sentimen Dari X atau Twitter')
    st.write(sentiment_twitter)

  

    st.subheader('Distribusi Sentimen')
    percent_val = 100 * sentiment_twitter['sentiment'].value_counts() / len(sentiment_twitter)
    st.bar_chart(percent_val)

    st.subheader('Word Cloud dari Sentiment Twitter')
    word_cloud_text = ''.join(sentiment_twitter['clean_text'])
    wordcloud = WordCloud(max_font_size=100, max_words=100, background_color="white",
                         scale=10, width=800, height=400).generate(word_cloud_text)

    st.image(wordcloud.to_array())

elif app_mode == "🤷 Sentiment Analysis":
    st.sidebar.write("Anda berada di Mode Analisis Sentimen.")

    review_input = st.text_area("Masukkan kalimat:", "")

    if st.button("Prediksi"):
        if review_input:
            # Lakukan prediksi sentimen
            sentiment = predict_sentiment(review_input)
            
            # Tampilkan hasil prediksi
            if sentiment == 1:
                st.markdown('<p style="color: green; font-size: 20px;">Sentimen Positif 😃</p>', unsafe_allow_html=True)
            else:
                st.markdown('<p style="color: red; font-size: 20px;">Sentimen Negatif 😞</p>', unsafe_allow_html=True)
        else:
            st.warning("Masukkan kalimat terlebih dahulu.")

    st.sidebar.markdown("---")



import streamlit as st
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import torch
import nltk
import re
from nltk.corpus import stopwords

tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
model = RobertaForSequenceClassification.from_pretrained('roberta-base')

nltk.download('stopwords')
stop_words = set(stopwords.words('english')) - {'not'}

replacement_dict = {
    "not good": "bad",
    "don't like": "dislike"
}

def clean_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text)  
    text = text.lower()  
    words = text.split()
    for old, new in replacement_dict.items():     
        text = text.replace(old, new)
    text = ' '.join(word for word in words if word not in stop_words)    
    return " ".join(words)

def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1).item()
    
    sentiment = "Negative" if predicted_class == 0 else "Positive"
    
    return sentiment

st.title("Sharhlarni ijobiy yoki salbiyligini aniqlash")
st.write("Bu dastur Amazon Fine Food sharhlari uchun sentimentni (ijobiy yoki salbiy) aniqlashga yordam beradi.")


    # Foydalanuvchi matn kiritishi uchun qism
user_input = st.text_area("Sharhingizni yozing:", "")
    
if st.button("Natijani ko'rish"):
    if user_input:
            # Matnni tozalash
        cleaned_input = clean_text(user_input)
            
            # Sentimentni bashorat qilish
        sentiment = predict_sentiment(cleaned_input)
            
            # Natijani chiqarish
        st.subheader(f"**Natija:** {sentiment}")
    else:
        st.error("Iltimos, sharh matnini kiriting!")

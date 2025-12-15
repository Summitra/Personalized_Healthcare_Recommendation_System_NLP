import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer  # Tokenizer is class used to convert text into numbers
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

@st.cache_data
def load_dataset():
    np.random.seed(42)
    symptoms_list = [
        "fever, cough, sore throat", "headache, nausea, vomiting",
        "fatigue, dizziness, weakness", "skin rash, itching, redness",
        "abdominal pain, diarrhea, bloating", "joint pain, stiffness, swelling",
        "chest pain, shortness of breath", "anxiety, insomnia, stress",
        "back pain, muscle ache", "cold, sneezing, runny nose",
        "fever, body ache, chills", "dizziness, blurred vision"
    ]
    medicines = [
        "Paracetamol", "Ibuprofen", "Cough Syrup", "Antihistamine",
        "Antacid", "Pain Reliever", "Vitamin Supplements",
        "Anti-inflammatory", "Antibiotic", "Decongestant",
        "Anti-viral", "Eye Drops"
    ]
    data = {
        "age": np.random.randint(10, 80, 1200),
        "gender": np.random.choice(["male", "female"], 1200),
        "symptoms": np.random.choice(symptoms_list, 1200),
        "medicine": np.random.choice(medicines, 1200)
    }
    df = pd.DataFrame(data)
    return df

df = load_dataset()


#  Prepare NLP Data
@st.cache_resource
def train_model(df):
    texts = df["symptoms"].astype(str).values
    labels = df["medicine"].values

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(labels)
    num_classes = len(label_encoder.classes_)

    tokenizer = Tokenizer(num_words=5000, oov_token="<OOV>")
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    X = pad_sequences(sequences, maxlen=10, padding='post', truncating='post')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Build Sequential model
    model = Sequential([
        Embedding(input_dim=5000, output_dim=64, input_length=10),
        LSTM(64, return_sequences=False),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=5, batch_size=32, validation_data=(X_test, y_test), verbose=0)

    return model, tokenizer, label_encoder

model, tokenizer, label_encoder = train_model(df)


#  Streamlit UI
st.set_page_config(page_title="Deep Learning Healthcare Recommender", layout="wide")
st.title("🧬Personalized Healthcare Recommendation System")

st.sidebar.title("Personalized Recommender App")
page = st.sidebar.radio("Navigate", ["🏠 Home", "📊 Explore Dataset", "💊 Get Recommendation"])



# Home Page

if page == "🏠 Home":
    st.subheader("Welcome!")
    st.markdown("""
    This app demonstrates a **Deep Learning NLP approach** for healthcare recommendations.
    - Model: **Keras Sequential (Embedding + LSTM + Dense)**
    - Dataset: Synthetic (1200 samples)
    - Task: Predict medicine from user-entered symptoms
    """)
    st.image("medical.png", width=250)

# Explore Dataset
elif page == "📊 Explore Dataset":
    st.subheader("Dataset Overview")
    st.dataframe(df.head())
    st.markdown("### Dataset Summary")
    st.write(df.describe(include='all'))

    st.markdown("### Gender Distribution")
    st.bar_chart(df["gender"].value_counts())

    st.markdown("### Common Symptoms")
    st.bar_chart(df["symptoms"].value_counts().head(10))


#  Recommendation Section (NLP Model)

elif page == "💊 Get Recommendation":
    st.subheader("Get Personalized Recommendation using Deep Learning")
    age = st.slider("Age", 1, 100, 30)
    gender = st.radio("Gender", ["male", "female"])
    symptoms_input = st.text_area("Enter your symptoms:", placeholder="e.g. fever, cough, sore throat")

    if st.button("Recommend"):
        if symptoms_input.strip() == "":
            st.warning("Please enter symptoms before predicting.")
        else:
            # Preprocess input
            seq = tokenizer.texts_to_sequences([symptoms_input])
            padded = pad_sequences(seq, maxlen=10, padding='post', truncating='post')

            # Predict
            preds = model.predict(padded)
            top_indices = preds[0].argsort()[-5:][::-1]
            top_meds = label_encoder.inverse_transform(top_indices)
            top_scores = preds[0][top_indices]

            result_df = pd.DataFrame({
                "Medicine": top_meds,
                "Confidence": np.round(top_scores * 100, 2)
            })

            st.markdown("### 💊 Top 5 Recommended Medicines")
            st.dataframe(result_df)
            st.success(f"🏥 Most likely recommendation: **{top_meds[0]}**")

      

#  Download Dataset Section

st.markdown("---")
st.subheader("📥 Download Dataset")
csv = df.to_csv(index=False).encode('utf-8')
st.download_button(
    label="Download Healthcare Dataset",
    data=csv,
    file_name="healthcare_recommendation_dataset.csv",
    mime="text/csv"
)

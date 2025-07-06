import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import pickle
import os
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch

# Set up the app
st.set_page_config(page_title="Ecommerce Text Classifier", layout="wide")

# App title and description
st.title("🛍️ Ecommerce Text Classifier")
st.markdown("""
This app classifies user messages as ecommerce-related or not using machine learning.
""")

# Load final_balanced_all.csv dataset
@st.cache_data
def load_dataset():
    try:
        # Try to load from local file first
        if os.path.exists("final_balanced_all.csv"):
            df = pd.read_csv("final_balanced_all.csv")
            
            # Ensure required columns exist (case insensitive)
            df.columns = df.columns.str.lower()
            required_columns = {'text', 'intent', 'lang'}  # Note: 'intent' might be misspelled in your data
            
            # Check for common column name variations
            if not required_columns.issubset(df.columns):
                # Try common alternatives
                if 'intent' not in df.columns and 'category' in df.columns:
                    df = df.rename(columns={'category': 'intent'})
                if 'text' not in df.columns and 'message' in df.columns:
                    df = df.rename(columns={'message': 'text'})
                if 'lang' not in df.columns and 'language' in df.columns:
                    df = df.rename(columns={'language': 'lang'})
            
            # Verify we have the required columns now
            required_columns = {'text', 'intent', 'lang'}
            if not required_columns.issubset(df.columns):
                missing = required_columns - set(df.columns)
                st.warning(f"Missing columns in dataset: {missing}. Using sample data instead.")
                return create_sample_data()
            
            return df
        
        # If not found, use sample data
        st.warning("final_balanced_all.csv not found. Using sample data instead.")
        return create_sample_data()
    except Exception as e:
        st.error(f"Error loading dataset: {str(e)}")
        return create_sample_data()

def create_sample_data():
    return pd.DataFrame({
        'text': [
            "help can need what checking in cases ask for refunds",
            "cannot of registration error notify",
            "Is there a tutorial or guide?",
            "do know to need what payment options dont list ur",
            "to you what try check payment methods accept",
            "to would like check your policy money back",
            "what's the weather today",
            "tell me a joke",
            "how to make pizza at home",
            "when is the next holiday"
        ],
        'intent': [
            "ecommerce",
            "ecommerce",
            "ecommerce",
            "ecommerce",
            "ecommerce",
            "ecommerce",
            "other",
            "other",
            "other",
            "other"
        ],
        'lang': [
            "en", "en", "en", "en", "en", "en", "en", "en", "en", "en"
        ]
    })

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = None
if 'vectorizer' not in st.session_state:
    st.session_state.vectorizer = None
if 'data' not in st.session_state:
    st.session_state.data = load_dataset()
if 'use_bert' not in st.session_state:
    st.session_state.use_bert = False
if 'bert_model' not in st.session_state:
    st.session_state.bert_model = None
if 'bert_tokenizer' not in st.session_state:
    st.session_state.bert_tokenizer = None

# Function to train logistic regression model
def train_logistic_regression(data):
    try:
        # Check if we have both classes in the data
        if len(data['intent'].unique()) < 2:
            return False, "Need both 'ecommerce' and 'other' examples in the dataset"
        
        # Vectorize text
        vectorizer = TfidfVectorizer(max_features=1000)
        X = vectorizer.fit_transform(data['text'])
        y = data['intent']
        
        # Train model
        model = LogisticRegression(max_iter=1000)
        model.fit(X, y)
        
        # Store in session state
        st.session_state.model = model
        st.session_state.vectorizer = vectorizer
        
        # Evaluate
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True)
        
        return True, report
    except Exception as e:
        return False, str(e)

# Function to load BERT model
def load_bert_model():
    try:
        model_name = "distilbert-base-uncased"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
        
        # Simple fine-tuning (in a real app, you'd want to properly fine-tune on your data)
        st.session_state.bert_tokenizer = tokenizer
        st.session_state.bert_model = model
        
        return True, "BERT model loaded successfully"
    except Exception as e:
        return False, str(e)

# Function to predict with BERT
def predict_with_bert(text):
    try:
        if st.session_state.bert_model is None or st.session_state.bert_tokenizer is None:
            return "Error: Model not loaded"
            
        inputs = st.session_state.bert_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        outputs = st.session_state.bert_model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        pred = torch.argmax(probs).item()
        
        return "ecommerce" if pred == 1 else "other"
    except Exception as e:
        return f"Error: {str(e)}"

# Function to predict with logistic regression
def predict_with_lr(text):
    try:
        if st.session_state.model is None or st.session_state.vectorizer is None:
            return "Error: Model not trained"
            
        vectorized = st.session_state.vectorizer.transform([text])
        prediction = st.session_state.model.predict(vectorized)
        return prediction[0]
    except Exception as e:
        return f"Error: {str(e)}"

# Sidebar for controls
with st.sidebar:
    st.header("Controls")
    
    # Model selection
    model_type = st.radio(
        "Select model type:",
        ("Logistic Regression", "BERT (Demo)"),
        index=0
    )
    
    st.session_state.use_bert = model_type == "BERT (Demo)"
    
    if st.session_state.use_bert:
        if st.button("Load BERT Model"):
            with st.spinner("Loading BERT model..."):
                success, message = load_bert_model()
                if success:
                    st.success(message)
                else:
                    st.error(f"Failed to load BERT model: {message}")
    else:
        if st.button("Train Logistic Regression Model"):
            with st.spinner("Training model..."):
                success, result = train_logistic_regression(st.session_state.data)
                if success:
                    st.success("Model trained successfully!")
                    st.json(result)
                else:
                    st.error(f"Training failed: {result}")

    # Data management
    st.header("Data Management")
    uploaded_file = st.file_uploader("Upload additional data (CSV)", type=["csv"])
    if uploaded_file is not None:
        try:
            new_data = pd.read_csv(uploaded_file)
            required_columns = {'text', 'intent', 'lang'}
            
            # Handle column name variations
            new_data.columns = new_data.columns.str.lower()
            if not required_columns.issubset(new_data.columns):
                if 'intent' not in new_data.columns and 'category' in new_data.columns:
                    new_data = new_data.rename(columns={'category': 'intent'})
                if 'text' not in new_data.columns and 'message' in new_data.columns:
                    new_data = new_data.rename(columns={'message': 'text'})
                if 'lang' not in new_data.columns and 'language' in new_data.columns:
                    new_data = new_data.rename(columns={'language': 'lang'})
            
            if not required_columns.issubset(new_data.columns):
                st.error(f"Uploaded file must contain these columns: {required_columns}")
            else:
                st.session_state.data = pd.concat([st.session_state.data, new_data], ignore_index=True)
                st.success(f"Added {len(new_data)} new samples. Total samples: {len(st.session_state.data)}")
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")

# Main content area
tab1, tab2, tab3 = st.tabs(["Classifier", "Data Viewer", "Add Samples"])

with tab1:
    st.header("Text Classification")
    
    user_input = st.text_area("Enter text to classify:", "How can I return my purchase?")
    
    if st.button("Classify Text"):
        if st.session_state.use_bert:
            if st.session_state.bert_model is None:
                st.error("Please load the BERT model first using the sidebar controls")
            else:
                with st.spinner("Classifying with BERT..."):
                    prediction = predict_with_bert(user_input)
                    st.success(f"Prediction: {prediction}")
        else:
            if st.session_state.model is None:
                st.error("Please train the logistic regression model first using the sidebar controls")
            else:
                with st.spinner("Classifying..."):
                    prediction = predict_with_lr(user_input)
                    st.success(f"Prediction: {prediction}")

with tab2:
    st.header("Current Training Data")
    st.write(f"Loaded {len(st.session_state.data)} samples from final_balanced_all.csv")
    st.dataframe(st.session_state.data)
    
    st.download_button(
        label="Download current data as CSV",
        data=st.session_state.data.to_csv(index=False).encode('utf-8'),
        file_name='ecommerce_classification_data.csv',
        mime='text/csv'
    )

with tab3:
    st.header("Add New Samples")
    
    with st.form("add_sample_form"):
        text = st.text_input("Text")
        intent = st.selectbox("Intent", ["ecommerce", "other"])
        lang = st.selectbox("Language", ["en", "es", "fr", "de", "it"])
        
        submitted = st.form_submit_button("Add Sample")
        if submitted:
            if text.strip() == "":
                st.error("Please enter some text")
            else:
                new_sample = pd.DataFrame({
                    'text': [text],
                    'intent': [intent],
                    'lang': [lang]
                })
                st.session_state.data = pd.concat([st.session_state.data, new_sample], ignore_index=True)
                st.success("Sample added successfully!")
                st.dataframe(new_sample)

# Footer
st.markdown("---")
st.markdown("""
**Note**: This app uses data from final_balanced_all.csv. For production use:
1. Ensure your dataset has balanced classes
2. Consider adding more model options and evaluation metrics
3. Implement proper model persistence
""")
#https://huggingface.co/spaces/efecelik/sentiment-analysis
import streamlit as st
from transformers import pipeline


# Load the sentiment analysis pipeline
@st.cache_resource
def load_model():
    return pipeline("sentiment-analysis")


def main():
    st.title("Sentiment Analysis App")

    # Create text input
    user_input = st.text_area("Enter text for sentiment analysis:")

    # Analyze button
    if st.button("Analyze Sentiment"):
        if user_input:
            # Load model
            sentiment_model = load_model()

            # Perform sentiment analysis
            result = sentiment_model(user_input)[0]

            # Display results
            st.write("Sentiment:", result['label'])
            st.write("Confidence Score:", f"{result['score']:.2%}")
        else:
            st.warning("Please enter some text to analyze.")


if __name__ == "__main__":
    main()
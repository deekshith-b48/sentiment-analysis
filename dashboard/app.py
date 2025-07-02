# dashboard/app.py
"""
Streamlit Dashboard for Interactive Sentiment Analysis.
"""
import streamlit as st
import requests # To call the FastAPI backend
import pandas as pd
import plotly.express as px
import json

# Configuration for the FastAPI backend URL
# Assumes the FastAPI app is running on localhost:8000
# If deployed, this URL will need to change.
API_URL = "http://localhost:8000/predict/"

st.set_page_config(layout="wide", page_title="Sentiment Analysis Dashboard")

# --- Helper Functions ---
def call_sentiment_api(text_input):
    """Calls the sentiment analysis API."""
    payload = {"text": text_input}
    try:
        response = requests.post(API_URL, json=payload, timeout=10) # Added timeout
        response.raise_for_status() # Raise an exception for HTTP errors (4xx or 5xx)
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error connecting to API: {e}")
        return None
    except json.JSONDecodeError:
        st.error(f"Error decoding API response. Response text: {response.text}")
        return None


# --- Page Title ---
st.title("✨ Interactive Sentiment Analysis Dashboard")
st.markdown("""
Welcome to the Sentiment Analysis Dashboard!
Enter text to analyze its sentiment, or upload a CSV file for batch processing.
This dashboard interacts with a backend API to provide sentiment predictions.
""")

# --- Sidebar for Configuration (Future Use) ---
st.sidebar.header("⚙️ Configuration")
# sensitivity_level = st.sidebar.slider("Sentiment Sensitivity", 0.0, 1.0, 0.5, 0.1) # Placeholder
# domain_choice = st.sidebar.selectbox("Select Domain", ["General", "Product Reviews", "Social Media"]) # Placeholder

# --- Main Application ---

# Tabbed Interface
tab1, tab2 = st.tabs(["📝 Single Text Analysis", "📂 Batch Analysis (CSV)"])

with tab1:
    st.header("Analyze Single Text Input")
    user_text = st.text_area("Enter text here:", "I love Streamlit, it's so easy to build dashboards!", height=100)

    if st.button("Analyze Sentiment", key="single_text_button"):
        if user_text:
            with st.spinner("Analyzing..."):
                api_response = call_sentiment_api(user_text)

            if api_response:
                st.subheader("Sentiment Analysis Result:")

                sentiment = api_response.get("sentiment", "N/A").capitalize()
                confidence = api_response.get("confidence_score")
                explanation = api_response.get("explanation") # For LIME/SHAP

                # Display sentiment with color coding
                if sentiment == "Positive":
                    st.markdown(f"**Sentiment:** <span style='color:green; font-size: 1.2em;'>{sentiment}</span>", unsafe_allow_html=True)
                elif sentiment == "Negative":
                    st.markdown(f"**Sentiment:** <span style='color:red; font-size: 1.2em;'>{sentiment}</span>", unsafe_allow_html=True)
                elif sentiment == "Neutral": # Assuming neutral is a possibility
                    st.markdown(f"**Sentiment:** <span style='color:blue; font-size: 1.2em;'>{sentiment}</span>", unsafe_allow_html=True)
                else:
                    st.markdown(f"**Sentiment:** {sentiment}", unsafe_allow_html=True)

                if confidence is not None:
                    st.write(f"**Confidence:** {confidence:.2f}")

                if explanation:
                    st.subheader("Prediction Explanation (Keywords):")
                    # This part will need to be structured based on how LIME/SHAP output is formatted
                    st.json(explanation)

                st.markdown("---")
                st.write("Raw API Response:")
                st.json(api_response)
        else:
            st.warning("Please enter some text to analyze.")

with tab2:
    st.header("Batch Sentiment Analysis from CSV")
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            st.write("Uploaded CSV preview:")
            st.dataframe(df.head())

            text_column = st.selectbox("Select the column containing text data:", df.columns)

            if text_column and st.button("Analyze Batch", key="batch_button"):
                if text_column not in df.columns:
                    st.error(f"Column '{text_column}' not found in the CSV.")
                else:
                    results = []
                    progress_bar = st.progress(0)
                    total_rows = len(df)

                    with st.spinner(f"Analyzing {total_rows} texts... This may take a while."):
                        for i, row in df.iterrows():
                            text_to_analyze = str(row[text_column])
                            if text_to_analyze:
                                api_response = call_sentiment_api(text_to_analyze)
                                if api_response:
                                    results.append({
                                        "original_text": text_to_analyze,
                                        "sentiment": api_response.get("sentiment"),
                                        "confidence": api_response.get("confidence_score")
                                    })
                                else:
                                    results.append({
                                        "original_text": text_to_analyze,
                                        "sentiment": "Error",
                                        "confidence": None
                                    })
                            else:
                                results.append({
                                    "original_text": "",
                                    "sentiment": "Empty",
                                    "confidence": None
                                })
                            progress_bar.progress((i + 1) / total_rows)

                    results_df = pd.DataFrame(results)
                    st.subheader("Batch Analysis Results:")
                    st.dataframe(results_df)

                    # Sentiment Distribution Plot
                    if not results_df.empty and 'sentiment' in results_df.columns:
                        st.subheader("Sentiment Distribution")
                        sentiment_counts = results_df['sentiment'].value_counts().reset_index()
                        sentiment_counts.columns = ['sentiment', 'count']

                        fig = px.bar(sentiment_counts, x='sentiment', y='count',
                                     color='sentiment', title="Distribution of Sentiments",
                                     labels={'sentiment':'Sentiment', 'count':'Number of Texts'})
                        st.plotly_chart(fig, use_container_width=True)

                        fig_pie = px.pie(sentiment_counts, names='sentiment', values='count',
                                         title="Proportion of Sentiments", hole=0.3)
                        st.plotly_chart(fig_pie, use_container_width=True)


                    # Download results
                    csv_export = results_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Download Results as CSV",
                        data=csv_export,
                        file_name="sentiment_analysis_results.csv",
                        mime="text/csv",
                    )
        except Exception as e:
            st.error(f"An error occurred while processing the CSV file: {e}")


# --- Footer ---
st.markdown("---")
st.markdown("Sentiment Analysis System - Alpha Version")
st.markdown("To run the dashboard: `streamlit run dashboard/app.py`")
st.markdown("Ensure the FastAPI backend is running: `python api/app.py`")


if __name__ == '__main__':
    # This block is not strictly necessary as Streamlit apps are run with `streamlit run app.py`
    # but can be useful for direct execution if needed (though not standard for Streamlit).
    st.info("To run this dashboard, use the command: `streamlit run dashboard/app.py`")
    st.info("Make sure the FastAPI backend (api/app.py) is running on http://localhost:8000.")

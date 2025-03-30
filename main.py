import streamlit as st
import PyPDF2
import json
import requests
from io import BytesIO
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Watsonx.ai API Configuration from environment variables
IBM_CLOUD_API_KEY = os.getenv("IBM_CLOUD_API_KEY")
WATSONX_PROJECT_ID = os.getenv("WATSONX_PROJECT_ID")
WATSONX_URL = os.getenv("WATSONX_URL", "https://us-south.ml.cloud.ibm.com/ml/v1/text/generation?version=2023-06-29")

GRANITE_MODELS = {
    "8B Instruct": "ibm/granite-3-8b-instruct",
}

# IAM access token
def get_iam_token(api_key):
    headers = {
        "Content-Type": "application/x-www-form-urlencoded",
        "Accept": "application/json"
    }
    data = {
        "grant_type": "urn:ibm:params:oauth:grant-type:apikey",
        "apikey": api_key
    }
    
    try:
        response = requests.post(
            "https://iam.cloud.ibm.com/identity/token",
            headers=headers,
            data=data,
            timeout=30
        )
        response.raise_for_status()
        return response.json().get("access_token")
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to get IAM token: {str(e)}")
        return None

# Extract text from a PDF
def extract_text_from_pdf(pdf_file):
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text.strip()
    except Exception as e:
        st.error(f"Error extracting text from PDF: {str(e)}")
        return None

# Analyze document with Watsonx.ai
def analyze_document_with_watsonx(text, access_token, model_id):
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Authorization": f"Bearer {access_token}"
    }
    
    payload = {
        "model_id": model_id,
        "project_id": WATSONX_PROJECT_ID,
        "input": text,
        "parameters": {
            "decoding_method": "sample",
            "max_new_tokens": 1000,
            "min_new_tokens": 100,
            "stop_sequences": [],
            "repetition_penalty": 1.1,
            "temperature": 0.7,
            "top_k": 50,
            "top_p": 0.9
        }
    }
    
    try:
        response = requests.post(WATSONX_URL, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"error": f"API request failed: {str(e)}"}

# Streamlit UI
def main():
    st.set_page_config(page_title="AI Document Compliance", layout="wide")
    st.title("📄 Analyze Contracts & Mitigate Risks with watsonx.ai: ")

    # Verify required environment variables
    if not all([IBM_CLOUD_API_KEY, WATSONX_PROJECT_ID]):
        st.error("Missing required environment variables. Please check your .env file.")
        return

    # Model selection dropdown
    selected_model = st.sidebar.selectbox(
        "Select Granite Model",
        list(GRANITE_MODELS.keys()),
        index=0  # Default to 8B Instruct
    )

    uploaded_file = st.file_uploader("Upload a PDF document", type=["pdf"])

    if uploaded_file:
        st.subheader("Extracted Text")
        extracted_text = extract_text_from_pdf(uploaded_file)
        
        if extracted_text is None:
            return
            
        st.text_area("Document Content", extracted_text, height=300, label_visibility="collapsed")

        if st.button("Analyze with Watsonx.ai"):
            with st.spinner("Authenticating and analyzing document..."):
                # Get fresh IAM token
                access_token = get_iam_token(IBM_CLOUD_API_KEY)
                
                if not access_token:
                    return
                
                # Create a comprehensive prompt for compliance analysis
                prompt = f"""Examine thoroughly this business paper and send a reply with the following:

1. EXECUTIVE SUMMARY:

- Clear outline of document's aim and content

2. COMPLIANCE RISK ASSESSMENT:

- Possible regulatory compliance problems (GDPR, HIPAA, etc.)
- Contractual responsibility risks
- Protection of personal information issues

3. ACTIONABLE RECOMMENDATIONS:

- Special measures to minimize existing risks
- Proposals about which field should be made the principle one

Document Content:
{extracted_text}

Format your response with clear Markdown headings (##) for each section.
"""
                
                result = analyze_document_with_watsonx(
                    prompt, 
                    access_token, 
                    GRANITE_MODELS[selected_model]
                )
                
                if "error" in result:
                    st.error(result["error"])
                else:
                    try:
                        # Parse and display the formatted response
                        generated_text = result.get("results", [{}])[0].get("generated_text", "No response generated")
                        
                        st.subheader("🔍 Comprehensive Analysis")
                        st.markdown(generated_text)
                        
                        # Show raw API response in expander for debugging
                        with st.expander("Show API Response Details"):
                            st.json(result)
                            
                    except Exception as e:
                        st.error(f"Error processing response: {str(e)}")
                        st.json(result)  # Show raw response for debugging

    st.sidebar.info("""
    **How to use:**
    1. Upload a PDF document
    2. Click 'Analyze with Watsonx.ai'
    3. Review compliance analysis
    """)

if __name__ == "__main__":
    main()
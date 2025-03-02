import streamlit as st
from openai import OpenAI
from elevenlabs.client import ElevenLabs
from stability_sdk import client
import stability_sdk.interfaces.gooseai.generation.generation_pb2 as generation
import pandas as pd
from datetime import datetime
from transformers import pipeline

# --- Configuration ---
st.set_page_config(
    page_title="AfriLearn AI Tutor",
    page_icon="🌍",
    layout="wide"
)

# --- Load API Keys from secrets.toml ---
AIML_API_KEY = st.secrets["AIML_API_KEY"]
ELEVENLABS_API_KEY = st.secrets["ELEVENLABS_API_KEY"]
STABILITY_KEY = st.secrets["STABILITY_KEY"]
ADMIN_PASS = st.secrets["ADMIN_PASS"]

# --- Initialize APIs ---
@st.cache_resource
def initialize_apis():
    base_url = "https://api.aimlapi.com/v1"  # AIML API endpoint
    api = OpenAI(api_key=AIML_API_KEY, base_url=base_url)
    elevenlabs_client = ElevenLabs(api_key=ELEVENLABS_API_KEY)
    stability_client = client.StabilityInference(
        key=STABILITY_KEY,
        engine="stable-diffusion-xl-1024-v1-0"
    )
    return api, elevenlabs_client, stability_client

api, elevenlabs_client, stability_client = initialize_apis()

# --- Language Setup ---
LANGUAGES = {
    "English": "en",
    "French": "fr",
    "Swahili": "sw"
}

# --- Initialize Session State ---
if "language" not in st.session_state:
    st.session_state.language = "English"  # Default language

if "offline_mode" not in st.session_state:
    st.session_state.offline_mode = False  # Default offline mode

if "messages" not in st.session_state:
    st.session_state.messages = []

# --- UI Translations ---
TRANSLATIONS = {
    "English": {
        "chat_header": "🤖 Chat with the AI Tutor",
        "chat_placeholder": "Ask your question...",
        "enable_voice": "Enable Voice",
        "select_voice": "🎤 Select Voice",
    },
    "French": {
        "chat_header": "🤖 Discutez avec le tuteur IA",
        "chat_placeholder": "Posez votre question...",
        "enable_voice": "Activer la voix",
        "select_voice": "🎤 Sélectionnez une voix",
    },
    "Swahili": {
        "chat_header": "🤖 Zungumza na Mwalimu wa AI",
        "chat_placeholder": "Uliza swali lako...",
        "enable_voice": "Washa Sauti",
        "select_voice": "🎤 Chagua sauti",
    }
}

# Get translations for the selected language
translations = TRANSLATIONS.get(st.session_state.language, TRANSLATIONS["English"])

# --- Language Selection ---
LANGUAGE = st.sidebar.selectbox(
    "🌍 Choose Language / Choisir la langue / Chagua lugha",
    list(LANGUAGES.keys()),
    index=list(LANGUAGES.keys()).index(st.session_state.language)  # Set default from session state
)

# Update session state when language changes
if LANGUAGE != st.session_state.language:
    st.session_state.language = LANGUAGE
    st.session_state.messages = []  # Clear chat history to reflect new language
    st.rerun()  # Rerun the app to apply language changes

# --- Voice Selection ---
VOICE_MAPPING = {
    "English": ["George", "Alice"],  # Voices for English
    "French": ["Georges", "Alice"],  # Voices for French
    "Swahili": ["George", "Alice"]   # Voices for Swahili
}

# Get voices for the selected language
voices = VOICE_MAPPING.get(st.session_state.language, ["Emily"])  # Default to English voices

# Add voice selection dropdown
selected_voice = st.sidebar.selectbox(
    translations["select_voice"],
    voices
)

# --- System Prompt for AI Tutor ---
SYSTEM_PROMPT = f"""
You are a friendly tutor for African students. Follow these rules:
1. Respond in {st.session_state.language}
2. Use examples from local culture (markets, farming, traditions)
3. Keep answers under 3 sentences
4. For math/science, use simple terms
"""

# --- Main Navigation Tabs ---
tabs = ["Home", "About Me", "Tutoring Request", "Career Guidance"]
selected_tab = st.sidebar.radio("Go to / Aller à / Nenda kwa", tabs)

# --- Offline Mode Setup ---
@st.cache_resource
def load_offline_model():
    # Load the castorini/afriberta_base model
    return pipeline("text-generation", model="castorini/afriberta_base")

# --- Offline Mode Checkbox ---
st.session_state.offline_mode = st.sidebar.checkbox(
    "Enable Offline Mode",
    value=st.session_state.offline_mode  # Preserve state
)

# --- Footer in Sidebar ---
st.sidebar.markdown("---")
st.sidebar.write("Developed with ❤️ by Motsim")
st.sidebar.write("Powered by **AI for Connectivity Hackathon II**")

# --- Home Tab ---
if selected_tab == "Home":
    st.header(translations["chat_header"])
    
    # Initialize chat history
    if not st.session_state.messages:
        st.session_state.messages = [{
            "role": "assistant", 
            "content": {
                "English": "Jambo! I'm your AI tutor. Ask me anything about school subjects!",
                "French": "Bonjour! Je suis votre tuteur IA. Posez-moi des questions sur les sujets scolaires!",
                "Swahili": "Habari! Mimi ni mwalimu wako wa AI. Niulize chochote kuhusu masomo!"
            }.get(st.session_state.language, "Jambo! I'm your AI tutor. Ask me anything about school subjects!")
        }]

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # Chat input
    if prompt := st.chat_input(translations["chat_placeholder"]):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Generate response using AIML API or offline model
        if st.session_state.offline_mode:
            # Use offline model
            try:
                offline_model = load_offline_model()
                ai_response = offline_model(prompt, max_length=50)[0]['generated_text']
            except Exception as e:
                ai_response = f"⚠️ Error: {str(e)}"
        else:
            # Use AIML API
            try:
                completion = api.chat.completions.create(
                    model="mistralai/Mistral-7B-Instruct-v0.2",  # Use the correct model name
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.7,
                    max_tokens=256,
                )
                ai_response = completion.choices[0].message.content
            except Exception as e:
                ai_response = f"⚠️ Error: {str(e)}"
        
        st.session_state.messages.append({"role": "assistant", "content": ai_response})
        st.rerun()

    # --- Text-to-Speech Feature with Language Support ---
    if st.checkbox(translations["enable_voice"], key="voice_toggle"):
        if st.session_state.messages[-1]["role"] == "assistant":
            try:
                # Generate audio for the text in the selected language
                audio_generator = elevenlabs_client.generate(
                    text=st.session_state.messages[-1]["content"],
                    voice=selected_voice,  # Use the selected voice
                    model="eleven_multilingual_v2"  # Use the multilingual model
                )
                
                # Convert generator to bytes
                audio_bytes = b"".join(audio_generator)
                
                # Play audio
                st.audio(audio_bytes, format="audio/mp3")
            except Exception as e:
                st.error(f"⚠️ Error generating audio: {str(e)}")

# --- About Me Tab ---
elif selected_tab == "About Us":
    st.header("About Me 👨‍💻")
    st.write("Meet the AfriLearn AI Team We are a diverse team of professionals  and students specializing in AI,   software engineering, pedagogy, cybersecurity, and machine learning. With expertise spanning   data science, low-latency networks, education, and secure systems, we are committed to developing AfriLearn—an AI-powered tutoring platform designed to provide equitable, personalized education across Africa. Our platform integrates multilingual AI chatbots, culturally relevant visual lessons, and mentorship tracking to enhance learning experiences, even in low-connectivity regions. Together, we aim to bridge the digital divide and empower students through innovative, accessible education.")

# --- Tutoring Request Tab ---
elif selected_tab == "Tutoring Request":
    st.header("📝 Tutoring Request Form")
    st.write("Fill out the form below to request a tutor.")

    MENTORSHIP_DB = "mentorship_requests.csv"

    def load_requests():
        try:
            return pd.read_csv(MENTORSHIP_DB)
        except FileNotFoundError:
            return pd.DataFrame(columns=[
                "timestamp", "name", "country", "interests", 
                "contact", "status", "notes"
            ])

    def save_request(request):
        df = load_requests()
        new_row = pd.DataFrame([request])
        df = pd.concat([df, new_row], ignore_index=True)
        df.to_csv(MENTORSHIP_DB, index=False)

    with st.form("mentor_form"):
        col1, col2 = st.columns(2)
        with col1:
            name = st.text_input("Full Name*", help="Required field")
            country = st.selectbox("Country*", [
                "Nigeria", "Kenya", "Ghana", 
                "South Africa", "Other"
            ])
        with col2:
            contact = st.text_input("Email/Phone*", 
                placeholder="e.g. +2348123456789")
            education_level = st.selectbox("Education Level", [
                "Primary School", "Secondary School", 
                "University", "Graduate"
            ])
        
        interests = st.multiselect("Areas of Need*", [
            "Math", "Science", "Coding", 
            "University Applications", "Career Advice"
        ])
        
        preferred_comms = st.radio("Preferred Communication", [
            "WhatsApp", "SMS", "Email"
        ], horizontal=True)
        
        if st.form_submit_button("Submit Request"):
            if not name or not contact or not interests:
                st.error("Please fill required fields (*)")
            else:
                request = {
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "name": name,
                    "country": country,
                    "contact": contact,
                    "interests": ", ".join(interests),
                    "education_level": education_level,
                    "preferred_comms": preferred_comms,
                    "status": "Pending",
                    "notes": ""
                }
                save_request(request)
                st.success("Request submitted! Mentor will contact you within 48hrs.")

    # Admin Panel
    if st.sidebar.text_input("Admin Password", type="password") == ADMIN_PASS:
        st.sidebar.success("Admin Mode")
        
        st.header("🔒 Mentor Request Dashboard")
        df = load_requests()
        
        if not df.empty:
            # Filtering
            status_filter = st.multiselect("Filter Status", 
                options=df["status"].unique(), default=["Pending"])
            filtered_df = df[df["status"].isin(status_filter)]
            
            # Display table
            st.dataframe(
                filtered_df.sort_values("timestamp", ascending=False),
                column_config={
                    "timestamp": "Date",
                    "contact": "Contact Info",
                    "preferred_comms": "Preferred Channel"
                },
                use_container_width=True
            )
            
            # Update status
            selected_id = st.number_input("Enter Request ID to Update", 
                min_value=0, max_value=len(df)-1)
            new_status = st.selectbox("New Status", 
                ["Pending", "Contacted", "Resolved"])
            new_notes = st.text_area("Add Notes")
            
            if st.button("Update Request"):
                df.at[selected_id, "status"] = new_status
                df.at[selected_id, "notes"] = new_notes
                df.to_csv(MENTORSHIP_DB, index=False)
                st.success("Request updated!")
        else:
            st.info("No mentorship requests yet")

# --- Career Guidance Tab ---
elif selected_tab == "Career Guidance":
    st.header("🚀 Career Guidance")
    st.write("Explore career options and connect with mentors.")

    CAREER_DB = "career_requests.csv"

    def load_career_requests():
        try:
            return pd.read_csv(CAREER_DB)
        except FileNotFoundError:
            return pd.DataFrame(columns=[
                "timestamp", "name", "email", "career_interest"
            ])

    def save_career_request(request):
        df = load_career_requests()
        new_row = pd.DataFrame([request])
        df = pd.concat([df, new_row], ignore_index=True)
        df.to_csv(CAREER_DB, index=False)

    with st.form("career_form"):
        name = st.text_input("Your Name*")
        email = st.text_input("Your Email*")
        career_interest = st.selectbox("Career Interest", [
            "Engineering", "Medicine", "Business", "Technology", "Arts"
        ])
        
        if st.form_submit_button("Submit"):
            if not name or not email:
                st.error("Please fill required fields (*)")
            else:
                request = {
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "name": name,
                    "email": email,
                    "career_interest": career_interest
                }
                save_career_request(request)
                st.success("Thank you! A mentor will contact you soon.")

    # Admin Panel
    if st.sidebar.text_input("Admin Password", type="password") == ADMIN_PASS:
        st.sidebar.success("Admin Mode")
        
        st.header("🔒 Career Guidance Dashboard")
        df = load_career_requests()
        
        if not df.empty:
            st.dataframe(
                df.sort_values("timestamp", ascending=False),
                column_config={
                    "timestamp": "Date",
                    "email": "Email",
                    "career_interest": "Career Interest"
                },
                use_container_width=True
            )
        else:
            st.info("No career guidance requests yet")

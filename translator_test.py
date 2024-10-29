import streamlit as st
from streamlit_option_menu import option_menu
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from processor import IndicProcessor
import speech_recognition as sr
from gtts import gTTS
import os

# Load model and tokenizer
model_name = "ai4bharat/indictrans2-indic-indic-1B"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name, trust_remote_code=True)
ip = IndicProcessor(inference=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model.to(DEVICE)


# Language options
languages = {
    "Assamese": "asm_Beng", "Bengali": "ben_Beng", "Bodo": "brx_Deva", "Dogri": "doi_Deva", 
    "English": "eng_Latn", "Gujarati": "guj_Gujr", "Hindi": "hin_Deva", "Kannada": "kan_Knda", 
    "Kashmiri (Arabic)": "kas_Arab", "Kashmiri (Devanagari)": "kas_Deva", "Konkani": "gom_Deva", 
    "Maithili": "mai_Deva", "Malayalam": "mal_Mlym", "Marathi": "mar_Deva", "Manipuri (Bengali)": "mni_Beng",
    "Manipuri (Meitei)": "mni_Mtei", "Nepali": "npi_Deva", "Odia": "ory_Orya", "Punjabi": "pan_Guru", 
    "Sanskrit": "san_Deva", "Santali": "sat_Olck", "Sindhi (Arabic)": "snd_Arab", "Sindhi (Devanagari)": "snd_Deva", 
    "Tamil": "tam_Taml", "Telugu": "tel_Telu", "Urdu": "urd_Arab"
}
LANGUAGES = {
                'English': 'en', 'Hindi': 'hi', 'Tamil': 'ta', 'Telugu': 'te',
                'Marathi': 'mr', 'Bengali': 'bn', 'Gujarati': 'gu',
                'Kannada': 'kn', 'Malayalam': 'ml', 'Punjabi': 'pa', 'Urdu': 'ur'
            }

# Streamlit app configuration
st.set_page_config(page_title="IndicTrans2 Translator", page_icon="🌐", layout="centered")
st.title("🌐 IndicTrans2 Translator with Speech Recognition")

if "input_text" not in st.session_state:
    st.session_state.input_text = ""
    
# Sidebar
with st.sidebar:
    selected = option_menu(
        "Menu", ["Translate", "About"],
        icons=["translate", "info-circle"],
        menu_icon="menu-app", default_index=0
    )

# About Section
if selected == "About":
    st.header("About IndicTrans2 Translator")
    st.write("""
        This app allows translations across 22 Indic languages supported by the AI4Bharat IndicTrans2 model.
        The model is designed to translate sentences with high accuracy, capturing linguistic nuances.
        Developed for ease of use, this app provides a seamless interface for translations.
    """)
    st.write("Created with ❤️ using Streamlit.")
    
# Translation Section
if selected == "Translate":
    st.subheader("🌐 Select Languages and Enter Text or Speak")

    # Language selection
    src_lang = st.selectbox("Choose source language:", options=list(languages.keys()))
    tgt_lang = st.selectbox("Choose target language:", options=list(languages.keys()))
    print(src_lang)
    print(tgt_lang)
    
    if st.button("Speak"):
        r = sr.Recognizer()
        with sr.Microphone() as source:
            st.info("🎙️ Listening... Speak now.")
            r.adjust_for_ambient_noise(source, duration=1)
            r.energy_threshold = 200
            audio = r.listen(source,timeout=15, phrase_time_limit=15)
            st.success("🔄 Processing...")
            try:
                input_text = r.recognize_google(audio,language=LANGUAGES[src_lang])
                st.write(f"Recognized Speech: {input_text}")
                st.session_state.input_text = input_text
            except sr.UnknownValueError:
                st.error("Could not understand audio")
            except sr.RequestError:
                st.error("Could not request results; check your network connection")

    
    # Translation
    input_text = st.text_area("Enter sentences to translate:",value=st.session_state.input_text)
    print(input_text)
    input_sentences = [sentence for sentence in input_text.split('\n')] if input_text else []
    print(input_sentences)
    if st.button("Translate"):
        if not input_sentences:
            st.warning("Please enter or speak text to translate.")
        else:
            batch = ip.preprocess_batch(
                input_sentences,
                src_lang=languages[src_lang],
                tgt_lang=languages[tgt_lang],
            )

            # Tokenize and generate translations
            inputs = tokenizer(
                batch,
                truncation=True,
                padding="longest",
                return_tensors="pt",
                return_attention_mask=True,
            ).to(DEVICE)

            with torch.no_grad():
                generated_tokens = model.generate(
                    **inputs,
                    use_cache=True,
                    min_length=0,
                    max_length=256,
                    num_beams=5,
                    num_return_sequences=1,
                )

            with tokenizer.as_target_tokenizer():
                generated_tokens = tokenizer.batch_decode(
                    generated_tokens.detach().cpu().tolist(),
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True,
                )

            translations = ip.postprocess_batch(generated_tokens, lang=languages[tgt_lang])
            
            # Display and Audio Playback
            result = "\n".join(f"**{src_lang}:** _{input_sentence}_  \n**{tgt_lang}:** _{translation}_" for input_sentence, translation in zip(input_sentences, translations))
            st.markdown(f"<div class='result-box'>{result}</div>", unsafe_allow_html=True)

            # Join all translated sentences into a single text
            translated_text = " ".join(translations)

            # Check if the target language is supported by gTTS
            if tgt_lang in LANGUAGES:
                lang_code = LANGUAGES[tgt_lang]
                tts = gTTS(translated_text, lang=lang_code)
                tts.save("translation_audio.mp3")
                
                # Show Play Button
                st.audio("translation_audio.mp3", format="audio/mp3")
            else:
                # Message if language not supported
                st.info(f"Audio playback is not available for {tgt_lang}.")
# 🌐 IndicTrans2 Translator with Speech-to-Text and Text-to-Speech Support

Welcome to **IndicTrans2 Translator**, an interactive Streamlit app that leverages AI-based language translation for Indian languages. This app enables you to translate spoken or typed input into various Indian languages, complete with text-to-speech playback for translated outputs. It's powered by IndicTrans2 and gTTS, providing a streamlined and user-friendly experience for real-time multilingual communication.

## 🚀 Features

- **Real-Time Speech-to-Text Recognition**: Speak in your preferred language, and the app will recognize your input.
- **AI-Powered Translation**: Translate text from one Indian language to another with high accuracy.
- **Text-to-Speech Playback**: Listen to translations in the target language.
- **Streamlined UI**: Built with Streamlit for a simple, interactive interface.

## 📋 Supported Languages

This app supports multiple Indian languages, including:

- English (en)
- Hindi (hi)
- Tamil (ta)
- Telugu (te)
- Marathi (mr)
- Bengali (bn)
- Gujarati (gu)
- Kannada (kn)
- Malayalam (ml)
- Punjabi (pa)
- Urdu (ur)

## 📂 Project Structure

```bash
├── translator_test.py        # Main app file    
└── README.md 
└── requirements.txt       
```       

## ⚙️ Installation & Setup

- Clone the repo:
```bash
git clone https://github.com/770navyasharma/IndicTransTranslator.git
cd IndicTrans2-Translator
```

- Install dependencies:
```bash
pip install -r requirements.txt
```

- Run the app:
```bash
streamlit run translator_test.py
```

## 📝 Usage Instructions
- **Select Languages**: Choose your source and target languages from the dropdown.
- **Speak or Type**: Click the Speak button to input your text via speech, or manually type text into the input box.
- **Translate**: Hit the Translate button to generate a translation.
- **Listen**: If supported, click Play to hear the translated text.

## 🧑‍💻 Tech Stack
- **Streamlit** - Interactive UI
- **Transformers** - IndicTrans2 translation model
- **SpeechRecognition** - Real-time speech-to-text input
- **gTTS** - Text-to-speech playback

## 🤝 Contributing
Contributions, issues, and feature requests are welcome! Feel free to check out the issues page if you'd like to contribute.



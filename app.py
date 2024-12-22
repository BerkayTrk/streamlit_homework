import streamlit as st
from transformers import pipeline

# ------------------------------
# Load Whisper Model
# ------------------------------
def load_whisper_model():
    """
    Load the Whisper model for audio transcription.
    """
    # TODO
    whisper = pipeline("automatic-speech-recognition", model="openai/whisper-tiny", return_timestamps=True)
    return whisper


# ------------------------------
# Load NER Model
# ------------------------------
def load_ner_model():
    """
    Load the Named Entity Recognition (NER) model pipeline.
    """
    # TODO
    ner = pipeline("token-classification", model="dslim/bert-base-NER", aggregation_strategy="simple")
    return ner


# ------------------------------
# Transcription Logic
# ------------------------------
def transcribe_audio(uploaded_file):
    """
    Transcribe audio into text using the Whisper model.
    Args:
        uploaded_file: Audio file uploaded by the user.
    Returns:
        str: Transcribed text from the audio file.
    """
    # TODO
    if uploaded_file is None:
        return "Waiting for the file..."

    whisper = load_whisper_model()
    output = whisper(uploaded_file.getvalue())
    output = output["text"]
    return output


# ------------------------------
# Entity Extraction
# ------------------------------
def extract_entities(text, ner_pipeline):
    """
    Extract entities from transcribed text using the NER model.
    Args:
        text (str): Transcribed text.
        ner_pipeline: NER pipeline loaded from Hugging Face.
    Returns:
        dict: Grouped entities (ORGs, LOCs, PERs).
    """
    # TODO
    if text == "Waiting for the file...":
        text = "Waiting for the file..."
        return [["Waiting for the file..."], ["Waiting for the file..."], ["Waiting for the file..."]]
    ner_results = ner_pipeline(text)
    ner_per = [i["word"] for i in ner_results if i["entity_group"] == "PER"]
    ner_loc = [i["word"] for i in ner_results if i["entity_group"] == "LOC"]
    ner_org = [i["word"] for i in ner_results if i["entity_group"] == "ORG"]

    ner_per, ner_loc, ner_org = list(set(ner_per)), list(set(ner_loc)), list(set(ner_org))

    output = [ner_org, ner_loc, ner_per]
    return output


# ------------------------------
# Main Streamlit Application
# ------------------------------
def main():
    st.title("Meeting Transcription and Entity Extraction")

    # You must replace below
    STUDENT_NAME = "Berkay Türk"
    STUDENT_ID = "150220320"
    st.write(f"**{STUDENT_ID} - {STUDENT_NAME}**")
    # TODO
    # Fill here to create the streamlit application by using the functions you filled
    st.write("## Upload your Audio File down Below")
    audio_file = st.file_uploader(label="Audio File", type=["wav"])
    if audio_file:
        st.info("Transcribing the audio file. This may take a minute.")
        with st.spinner("Transcription in progress..."):
            transcription = transcribe_audio(uploaded_file=audio_file)
    st.write("### Transcription:")
    st.write(transcription)

    if transcription:
        st.info("Extracting the entities. This may take a minute.")
        with st.spinner("Extraction in progress..."):
            entities = extract_entities(transcription, load_ner_model())
    st.write("### Entities: ")
    col1, col2, col3 = st.columns(3)

    col1.write("#### Organizations:")
    for i in entities[0]:
        col1.write("- "+i)

    col2.write("#### Locations:")
    for i in entities[1]:
        col2.write("- "+i)

    col3.write("#### Names:")
    for i in entities[2]:
        col3.write("- "+i)


if __name__ == "__main__":
    main()

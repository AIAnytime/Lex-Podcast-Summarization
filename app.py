import streamlit as st
from pytube import YouTube
from pathlib import Path
import shutil
import os
from whisper_jax import FlaxWhisperPipline
import jax.numpy as jnp
import openai
from langchain import PromptTemplate, LLMChain
from langchain.llms import AzureOpenAI

os.environ['OPENAI_API_KEY'] = ""
os.environ['OPENAI_API_BASE'] = ""
os.environ['OPENAI_API_TYPE'] = "azure"
os.environ['OPENAI_API_VERSION'] = "2023-05-15"

openai.api_key = os.getenv('OPENAI_API_KEY')
openai.api_base = os.getenv('OPENAI_API_BASE')
openai.api_type = os.getenv('OPENAI_API_TYPE')
openai.api_version = os.getenv('OPENAI_API_VERSION')

prompt_template = '''I want you to act as a digital marketing expert.
Write a content for LinkedIn post with key findings and pointers from the context. \n
Context: {context}'''

prompt = PromptTemplate(
    input_variables = ['context'],
    template = prompt_template
)

def save_video(url, video_filename):
    youtubeObject = YouTube(url)
    youtubeObject = youtubeObject.streams.get_highest_resolution()
    try:
        youtubeObject.download()
    except:
        print("An error has occurred")
    print("Download is completed successfully")
    return video_filename

def save_audio(url):
    yt = YouTube(url)
    video = yt.streams.filter(only_audio=True).first()
    out_file = video.download()
    base, ext = os.path.splitext(out_file)
    file_name = base + '.mp3'
    try:
        os.rename(out_file, file_name)
    except WindowsError:
        os.remove(file_name)
        os.rename(out_file, file_name)
    audio_filename = Path(file_name).stem+'.mp3'
    video_filename = save_video(url, Path(file_name).stem+'.mp4')
    print(yt.title + " Has been successfully downloaded")
    return yt.title, audio_filename, video_filename

#Loading the whisper model
@st.cache_resource
def load_model():
    # instantiate pipeline
    pipeline = FlaxWhisperPipline("openai/whisper-base")
    return pipeline

#transcription function using whisper jax
def transcription(audio_file):
    model = load_model()
    outputs = model(audio_file,  task="transcribe", return_timestamps=True)
    return outputs

#llm pipeline
def llm_pipeline():
    llm = AzureOpenAI(
        deployment_name="aianytime-gpt35",
        model_name = "gpt-35-turbo",
        temperature = 0.7,
        max_tokens = 1000
    )
    return llm

st.set_page_config(layout="wide")

def main():

    st.markdown("<h1 style='text-align: center; color: white;'>Lex Podcast Summarization 🦜🎥📄 </h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center; color: white;'>Built using Whisper Jax and Azure OpenAI <a href='https://github.com/AIAnytime'> (By AI Anytime with ❤️) </a></h3>", unsafe_allow_html=True)

    st.markdown("<h2 style='text-align: center; color:green;'>Enter the clip URL👇</h2>", unsafe_allow_html=True)
    url =  st.text_input('Enter URL of YouTube video:')

    if url is not None:
        if st.button("Submit"):
            col1, col2, col3 = st.columns([1,1,1])
            with col1:
                st.subheader("Video Preview")
                video_title, audio_filename, video_filename = save_audio(url)
                st.video(video_filename)
            with col2:
                st.subheader("Transcript Below") 
                print(audio_filename)
                transcript_result = transcription(audio_filename)
                st.success(transcript_result)
            with col3:
                st.subheader("💡 LinkedIn Post Content") 
                transcript_text = transcript_result['text']
                llm = llm_pipeline()
                chain1 = LLMChain(llm=llm,prompt=prompt)
                result = chain1.run(transcript_text)
                st.write(result)
                st.code(result)

if __name__ == "__main__":
    main()
            

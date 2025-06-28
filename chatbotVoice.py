import soundcard as sc
from sqlalchemy import create_engine
import io
import os
from langchain_openai import ChatOpenAI
import soundfile as sf # Import the soundfile library
from openai import OpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_community.chat_message_histories import (
    SQLChatMessageHistory,
)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory

from dotenv import load_dotenv
from elevenlabs.client import ElevenLabs
from elevenlabs import play

load_dotenv()
elevenlabsAPI = os.getenv('ELEVENLABS_API_KEY')
elevenLabsClient = ElevenLabs(api_key=elevenlabsAPI
                              )

client = OpenAI()
sample_rate = 44100
duration = 5  
recording = sc.default_microphone().record(numframes=int(duration * sample_rate), samplerate=sample_rate)

print("Recording complete. Shape:", recording.shape)


memory_buffer = io.BytesIO()
sf.write(memory_buffer, recording, sample_rate, format='wav')
memory_buffer.seek(0)
memory_buffer.name="audio_input.wav"


transcript = client.audio.transcriptions.create(
    model="whisper-1",
    file=memory_buffer
)

print(transcript.text)
human_template = f"{{userInput}}"
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful personal assiant. Keep you repoonses concise and short"),
    MessagesPlaceholder(variable_name="history"),
    ("human", human_template),
])



llm = ChatOpenAI(model="gpt-4o-mini")
chain = prompt | llm
engine = create_engine("sqlite:///db.sqlite")
chain_with_history = RunnableWithMessageHistory(
    chain,
    lambda session_id: SQLChatMessageHistory(session_id=session_id, connection=engine),
    input_messages_key="userInput",
    history_messages_key="history",
)
reponse = chain_with_history.invoke({"userInput": transcript.text}, config={"configurable": {"session_id": "21312423456896456"}},)
print(reponse.content)
audio = elevenLabsClient.text_to_speech.convert(
    text=reponse.content,
    voice_id="flHkNRp1BlvT73UL6gyz",
    model_id="eleven_flash_v2_5",
    output_format="mp3_44100_128",
)
play(audio)

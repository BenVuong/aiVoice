import soundcard as sc
from sqlalchemy import create_engine
import io
import gradio as gr
import os
from langchain_openai import ChatOpenAI
import soundfile as sf
from openai import OpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import (
    SQLChatMessageHistory,
)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from dotenv import load_dotenv
from elevenlabs.client import ElevenLabs
from elevenlabs import play
from langchain.agents import create_openai_functions_agent, AgentExecutor
from composio_langchain import ComposioToolSet

load_dotenv()
elevenlabsAPI = os.getenv('ELEVENLABS_API_KEY')
elevenLabsClient = ElevenLabs(api_key=elevenlabsAPI
                             )

client = OpenAI()
sample_rate = 44100
duration = 5  # This duration might not be directly used for recording, but good to keep if you have other uses for it.

human_template = f"{{userInput}}"
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful personal assistant. Keep your responses concise and short. Reply in a snarky and funny manner"),
    MessagesPlaceholder(variable_name="history"),
    ("human", human_template),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])
composio_toolset = ComposioToolSet(api_key=os.getenv("COMPOSIO_API_KEY"))
tools = composio_toolset.get_tools(actions=['GOOGLETASKS_CREATE_TASK_LIST', 'GOOGLETASKS_GET_TASK_LIST','GOOGLETASKS_INSERT_TASK'])

llm = ChatOpenAI(model="gpt-4o-mini")
agent = create_openai_functions_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, return_intermediate_steps=True,verbose=True)
chain = prompt | llm
engine = create_engine("sqlite:///db.sqlite")
chain_with_history = RunnableWithMessageHistory(
    agent_executor,
    lambda session_id: SQLChatMessageHistory(session_id=session_id, connection=engine),
    input_messages_key="userInput",
    history_messages_key="history",
)

def chatWithVoice(audio):
    if audio is None:
        return "", "", None # Handle cases where no audio is recorded (e.g., if user releases immediately)

    wavIO = io.BytesIO()
    sf.write(wavIO, audio[1], sample_rate, format='wav')
    wavIO.seek(0)
    wavIO.name = "user_audio.wav"

    transcript = client.audio.transcriptions.create(
        model="whisper-1",
        file=wavIO
    )

    response = chain_with_history.invoke({"userInput": transcript.text}, config={"configurable": {"session_id": "21312423456896456"}},)

    audioOut = elevenLabsClient.text_to_speech.convert(
        text=response["output"],
        voice_id="flHkNRp1BlvT73UL6gyz",
        model_id="eleven_flash_v2_5",
        output_format="mp3_44100_128",
    )

    output_path = "output/bot_response.mp3"
    audio_bytes = b"".join(audioOut)

    # Save to file
    with open(output_path, "wb") as f:
        f.write(audio_bytes)
    return transcript.text, response["output"], output_path

with gr.Blocks() as demo:
    gr.Markdown("## Voice Assistant Chabot")
    with gr.Row():
        # Changed the type to "filepath" for better handling with Whisper and Eleven Labs
        # Added sources=["microphone"] and streaming=False for typical push-to-talk
        mic = gr.Audio(type="numpy", label="Push to Talk", streaming=False, sources=["microphone"])
        
        # Remove the explicit "Send" button as it will be triggered by stop_recording
        # btn = gr.Button("Send") 
        
        transcript_text = gr.Textbox(label="You said")
        response_text = gr.Textbox(label="Bot says")
        voice_output = gr.Audio(label="Bot Voice", type="filepath", autoplay=True)

    # Attach the chatWithVoice function to the `stop_recording` event of the microphone
    mic.stop_recording(
        chatWithVoice,
        inputs=[mic],
        outputs=[transcript_text, response_text, voice_output]
    )

demo.launch()
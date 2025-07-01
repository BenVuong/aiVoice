import soundcard as sc
from storeLongTermMemory import archiveOldMessages
from sqlalchemy import create_engine
import io
import gradio as gr
import os

from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
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
from langchain.tools import tool

from browser_use.llm import ChatOpenAI as browserChat
from browser_use import Agent
import asyncio

load_dotenv()
elevenlabsAPI = os.getenv('ELEVENLABS_API_KEY')
elevenLabsClient = ElevenLabs(api_key=elevenlabsAPI
                             )
qdrant_client = QdrantClient(url=os.getenv("QDRANT_URL"),api_key=os.getenv("QDRANT_API_KEY")) 
client = OpenAI()
sample_rate = 44100
duration = 5  # This duration might not be directly used for recording, but good to keep if you have other uses for it.
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
qdrant_client = QdrantClient(url=os.getenv("QDRANT_URL"),api_key=os.getenv("QDRANT_API_KEY")) 
vectorStore = QdrantVectorStore(client=qdrant_client, collection_name="LongTermMemory",embedding=embeddings)
human_template = f"{{userInput}}"
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful personal assistant. Keep your responses concise and short. Reply in a snarky and funny manner
                As you chat with the user your old messages will be stored in Qdrant vector database as long term memory
                If the user asks you to complete a task by opening up a browswer use the useBrowserSearch tool and pass in the user's requested task as the parameter
     """),
    MessagesPlaceholder(variable_name="history"),
    ("human", human_template),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

@tool
def longTermMemorySearch(query: str) -> str:
    """
    Use this tool to recall longterm memory by querying for past messages"""
    results = vectorStore.similarity_search_with_score(query, k=3, score_threshold=0.25)
    return(results)

async def browser(task: str):
    agent = Agent(
        task=task,
        llm=browserChat(model="gpt-4.1"),
    )
    result = await agent.run()
     # Display all the thinking/reasoning

    
    # print("\n=== FINAL RESULT ===")
    # print(result.final_result())
    return(result.final_result())

@tool
def useBrowserSearch(instruction: str) -> str:
   """Use this tool to up an internet browser to complete a task. Just input the task in as the instruction argument"""
   results =  asyncio.run(browser(instruction))
   return(results)



composio_toolset = ComposioToolSet(api_key=os.getenv("COMPOSIO_API_KEY"))
composioTools = composio_toolset.get_tools(actions=['GOOGLETASKS_CREATE_TASK_LIST', 'GOOGLETASKS_GET_TASK_LIST','GOOGLETASKS_INSERT_TASK'])
tools = composioTools +[useBrowserSearch] 
llm = ChatOpenAI(model="gpt-4.1")
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
    chatMessages = SQLChatMessageHistory(session_id="21312423456896456", connection=engine).get_messages()
    if(len(chatMessages)>=4):
        archiveOldMessages(2, "21312423456896456")
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
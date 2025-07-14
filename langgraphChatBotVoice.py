import json
import asyncio
from typing import List, Literal, Optional
import gradio as gr
import os
import uuid
import sqlite3
import whisper
import torchaudio as ta
from chatterbox.tts import ChatterboxTTS
from composio_langchain import ComposioToolSet
import soundfile as sf
import tiktoken
from langchain_core.documents import Document
from langchain_core.messages import get_buffer_string, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PayloadSchemaType
from browser_use.llm import ChatOpenAI as browserChat
from browser_use import Agent
import configparser
configFile = configparser.ConfigParser()
configFile.read('config/config.ini')
whisperModel = whisper.load_model("tiny")
qdrantClient = QdrantClient(url=os.getenv("QDRANT_URL"), api_key=os.getenv("QDRANT_API_KEY"))

if not qdrantClient.collection_exists("LongTermMemory"):
    qdrantClient.create_collection(
        collection_name="LongTermMemory",
        vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
    )

qdrantClient.create_payload_index(
    collection_name="LongTermMemory",
    field_name="metadata.user_id",
    field_schema=PayloadSchemaType.KEYWORD,
)

recall_vector_store = QdrantVectorStore(client=qdrantClient, collection_name="LongTermMemory",
                                        embedding=OpenAIEmbeddings(model="text-embedding-3-small"))


def get_user_id(config: RunnableConfig) -> str:
    user_id = config["configurable"].get("user_id")
    if user_id is None:
        raise ValueError("User ID needs to be provided to save a memory.")
    return user_id


async def browser(task: str):
    agent = Agent(
        task=task,
        llm=browserChat(model="gpt-4o"), # Adjusted model name for compatibility
    )
    result = await agent.run()
    return result.final_result()


@tool
def useBrowserSearch(instruction: str) -> str:
    """Use this tool to up an internet browser to complete a task. Just input the task in as the instruction argument"""
    results = asyncio.run(browser(instruction))
    return results


@tool
def save_recall_memory(memory: str, config: RunnableConfig) -> str:
    """Save memory to vectorstore for later semantic retrieval."""
    user_id = get_user_id(config)
    document = Document(
        page_content=memory,
        metadata={"user_id": user_id, "id": str(uuid.uuid4())}
    )
    recall_vector_store.add_documents([document])
    return f"Memory saved: '{memory}'"


@tool
def search_recall_memories(query: str, config: RunnableConfig) -> List[str]:
    """Search for relevant memories."""
    user_id = get_user_id(config)
    from qdrant_client.models import FieldCondition, Filter, MatchValue

    _filter = Filter(
        must=[
            FieldCondition(
                key="metadata.user_id",
                match=MatchValue(value=user_id)
            )
        ]
    )
    documents = recall_vector_store.similarity_search(
        query, k=3, filter=_filter
    )
    return [document.page_content for document in documents]

composio_toolset = ComposioToolSet(os.getenv("COMPOSIO_API_KEY"))

composioTools = composio_toolset.get_tools(actions=['GOOGLECALENDAR_CREATE_EVENT', 'GOOGLETASKS_LIST_TASK_LISTS','GOOGLETASKS_CREATE_TASK_LIST','GOOGLETASKS_GET_TASK_LIST', 'GOOGLETASKS_INSERT_TASK', 'GOOGLETASKS_DELETE_TASK', 'GOOGLECALENDAR_GET_CALENDAR'])
tools = [save_recall_memory, search_recall_memories, useBrowserSearch]


class State(MessagesState):
    recall_memories: List[str]


prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant with advanced long-term memory"
            " capabilities. Powered by a stateless LLM, you must rely on"
            " external memory to store information between conversations."
            " Utilize the available memory tools to store and retrieve"
            " important details that will help you better attend to the user's"
            " needs and understand their context.\n\n"
            " If the user asks you to complete a task by opening up a browswer use the useBrowserSearch tool and pass in the user's requested task as the parameter "
            "Memory Usage Guidelines:\n"
            "1. Actively use memory tools to build a comprehensive understanding of the user.\n"
            "2. Make informed suppositions and extrapolations based on stored memories.\n"
            "3. Regularly reflect on past interactions to identify patterns and preferences.\n"
            "4. Update your mental model of the user with each new piece of information.\n"
            "5. Cross-reference new information with existing memories for consistency.\n"
            "6. Prioritize storing emotional context and personal values alongside facts.\n"
            "7. Use memory to anticipate needs and tailor responses to the user's style.\n"
            "8. Recognize and acknowledge changes in the user's situation or perspectives over time.\n"
            "9. Leverage memories to provide personalized examples and analogies.\n"
            "10. Recall past challenges or successes to inform current problem-solving.\n\n"
            "## Recall Memories\n"
            "Recall memories are contextually retrieved based on the current conversation:\n{recall_memories}\n\n"
            "## Instructions\n"
            "Engage with the user naturally, as a trusted colleague or friend."
            "Respond in a snarky manner"
            " There's no need to explicitly mention your memory capabilities."
            " Instead, seamlessly incorporate your understanding of the user"
            " into your responses. Be attentive to subtle cues and underlying"
            " emotions. Adapt your communication style to match the user's"
            " preferences and current emotional state. Use tools to persist"
            " information you want to retain in the next conversation. If you"
            " do call tools, all text preceding the tool call is an internal"
            " message. Respond AFTER calling the tool, once you have"
            " confirmation that the tool completed successfully.\n\n",
        ),
        ("placeholder", "{messages}"),
    ]
)

model = ChatOpenAI(model_name="gpt-4o")
model_with_tools = model.bind_tools(tools)
tokenizer = tiktoken.encoding_for_model("gpt-4o")


def agent(state: State) -> State:
    """Process the current state and generate a response using the LLM."""
    bound = prompt | model_with_tools
    recall_str = (
        "<recall_memory>\n" + "\n".join(state["recall_memories"]) + "\n</recall_memory>"
    )
    prediction = bound.invoke(
        {
            "messages": state["messages"],
            "recall_memories": recall_str,
        }
    )
    return {
        "messages": [prediction],
    }


def load_memories(state: State, config: RunnableConfig) -> State:
    """Load memories for the current conversation."""
    convo_str = get_buffer_string(state["messages"])
    convo_str = tokenizer.decode(tokenizer.encode(convo_str)[:2048])
    recall_memories = search_recall_memories.invoke(convo_str, config)
    return {
        "recall_memories": recall_memories,
    }


def route_tools(state: State) -> Literal["tools", "__end__"]:
    """Determine whether to use tools or end the conversation based on the last message."""
    msg = state["messages"][-1]
    if msg.tool_calls:
        return "tools"
    return END


# Create the graph and add nodes
builder = StateGraph(State)
builder.add_node("load_memories", load_memories)
builder.add_node("agent", agent)
builder.add_node("tools", ToolNode(tools))

# Add edges to the graph
builder.add_edge(START, "load_memories")
builder.add_edge("load_memories", "agent")
builder.add_conditional_edges("agent", route_tools)
builder.add_edge("tools", "agent")


memory = SqliteSaver(conn=sqlite3.connect("db/checkpoints.sqlite", check_same_thread=False))
graph = builder.compile(checkpointer=memory)



def chatWithVoice(audio, history):
    """
    Function to integrate the LangGraph chatbot with the Gradio interface.
    """
    if audio is None:
        return "", "", None # Handle cases where no audio is recorded (e.g., if user releases immediately)
    sample_rate = 44100
    sf.write("input/input.wav", audio[1], sample_rate, format='wav')
    transcript = whisperModel.transcribe("input/input.wav")
    config = {"configurable": {"user_id": "1", "thread_id": "1"}}
    
    final_response = ""
    
    # Stream the response from the graph
    for chunk in graph.stream({"messages": [("user", transcript["text"])]}, config=config, stream_mode="values"):
        # The final AI response is in the 'agent' node's output
        if isinstance(chunk["messages"][-1], AIMessage) and not chunk["messages"][-1].tool_calls:
            final_response = chunk["messages"][-1].content
            
    model = ChatterboxTTS.from_pretrained(device="cuda")
    output_path = "output/bot_response.wav"
    wav = model.generate(final_response,audio_prompt_path=configFile['VoiceSetting']['voice'])
    ta.save(output_path, wav, model.sr)
    

    return transcript["text"], final_response, output_path

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
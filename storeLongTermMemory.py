from sqlalchemy import create_engine, text
import json
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langchain_community.chat_message_histories import (
    SQLChatMessageHistory
)
import os
from dotenv import load_dotenv
load_dotenv()

qdrantClient = QdrantClient(url=os.getenv("QDRANT_URL"),api_key=os.getenv("QDRANT_API_KEY"))

if not qdrantClient.collection_exists("LongTermMemory"):
    qdrantClient.create_collection(
        collection_name="LongTermMemory",
        vectors_config=VectorParams(size=1536,distance=Distance.COSINE),
    )

vectorStore = QdrantVectorStore(client=qdrantClient, collection_name="LongTermMemory", embedding=OpenAIEmbeddings(model="text-embedding-3-small"))

engine = create_engine("sqlite:///db.sqlite")
chatHistory = SQLChatMessageHistory(session_id="21312423456896456", connection=engine)

TABLE_NAME = "message_store"

def archiveOldMessages(numOfOldestMessages: int, sessionId: str):
    with engine.begin() as connection:
        # 1) pull id + message
        query = text(f"""
            SELECT id, message 
              FROM {TABLE_NAME}
             WHERE session_id = :session_id
             ORDER BY id ASC
             LIMIT :limit
        """)
        result = connection.execute(query, {"session_id": sessionId, "limit": numOfOldestMessages})
        rows = result.fetchall()

        if not rows:
            return

        # 2) build your docs and collect ids
        messagesToStore = []
        ids_to_delete = []
        for row in rows:
            row_id, messageJsonString = row
            messageDict = json.loads(messageJsonString)
            messageType = messageDict.get("type")
            messageContent = messageDict.get("data", {}).get("content")
            messagesToStore.append(
                Document(page_content=messageContent, metadata={"Message-Type": messageType, "id": row_id})
            )
            ids_to_delete.append(row_id)

        # 3) push to Qdrant
        vectorStore.add_documents(messagesToStore)

        # 4) delete from SQLite
        #    note: since these are ints, f‑string interpolation is safe
        id_list_str = ",".join(str(i) for i in ids_to_delete)
        delete_query = text(f"""
            DELETE FROM {TABLE_NAME}
             WHERE session_id = :session_id
               AND id IN ({id_list_str})
        """)
        connection.execute(delete_query, {"session_id": sessionId})
                


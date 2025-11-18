import os
import chromadb
from chromadb.utils import embedding_functions

# Match your existing DB path logic
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PERSIST_DIR = os.path.join(BASE_DIR, "db")
print("DB folder:", PERSIST_DIR)

chroma_client = chromadb.PersistentClient(path=PERSIST_DIR)

openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=os.environ.get("CHROMA_OPENAI_API_KEY"),
    model_name="text-embedding-3-large",
)

collection = chroma_client.get_or_create_collection(
    name="bible_commentaries",
    embedding_function=openai_ef,
)

print("Chunks in collection:", collection.count())

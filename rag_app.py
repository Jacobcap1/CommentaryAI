import os
from typing import List, Dict

import chromadb
from chromadb.utils import embedding_functions
from openai import OpenAI

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

##Connect to same ChromaDB

PERSIST_DIR = os.path.join(BASE_DIR,"db")

print("rag_app BASEDIR =", BASE_DIR)
print("rag_app PERSIST =", PERSIST_DIR)
os.makedirs(PERSIST_DIR,exist_ok=True)


chroma_client = chromadb.PersistentClient(path=PERSIST_DIR)

openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=os.environ.get("KEY"),
    model_name = "text-embedding-3-large",
    )
collection = chroma_client.get_or_create_collection( # create collection in database
    name = "bible_commentaries",
    embedding_function=openai_ef,
)
#OpenAI client for calling the chat model
client = OpenAI() # create instance of Openai client

#Retrieve relevant chunks from Chroma

def retrieve_context(question: str, k: int = 5) -> List[Dict]:
    #Use Chroma to find top-k most relevant commentary chunks
    #return a list of {content, metadata} dicts.

    result = collection.query( # searches database to find relevant information
        query_texts=[question],#the question is what we search with
        n_results = k,
    )
    docs = result.get("documents",[[]])[0]## list of lists of text chunks
    metas = result.get("metadatas",[[]])[0]## list of lists of metatdata dicts

    context_chunks = []

    for doc, meta in zip(docs,metas):
        context_chunks.append(
            {
                "content":doc,
                "metadata":meta,
            }
        )

    return context_chunks #list of context chunks to the caller

def build_context_block(chunks:List[Dict]) ->str:
#Build a single big string from retrieved chunks, with labels.
# #This is what we feed to LLM as context

    lines = [] # list of strings
    for i, ch in enumerate(chunks, start = 1):
        meta = ch["metadata"]
        commentary = meta.get("commentary", "Unknown Commentary")
        page = meta.get("page","?")
        source_file = meta.get("source_file","unknown.pdf")

        # this creates a label for this chunk
        # could be like [1] macarthur_romans , p.34 (macarthurromans.pdf)
        header = f"[{i}] {commentary}, p.{page}({source_file})"

        lines.append(header) # add to list
        lines.append(ch["content"]) #add content beneath that header
        lines.append("") #blank line between chunks # add blank space
    return "\n".join(lines) ##One big block of text: headers and chunks ==this is the context

def call_llm(question: str, context_block:str) ->str:
    #ask Openai model to answer using only provided commentary excerpts

    messages = [
        {
            "role": "system",
            "content":(
                "You are a Bible commentary assistant. "
                "You will be given excerpts from Bible commentaries. "
                "Use ONLY these excerpts to answer the user's question. "
                "If the excerpts do not contain enough information, say: "
                "'I couldn’t find a clear answer to that in the loaded commentaries.' "
                "Do not invent or hallucinate. "
                "At the end of your answer, include a section titled 'Sources' "
                "and list which commentaries and pages you used."
            ),

        },
        {
            "role":"user",
            "content": (
                f"Question: {question}\n\n"
                f"Relevant excerpts from commentaries: \n\n{context_block}"

            ),
        },
    ]

    response = client.chat.completions.create(
        model = "gpt-4o",
        messages = messages,
        temperature = .2,
    )

    return response.choices[0].message.content


def answer_question(question: str,k: int=5)->Dict:
    """
       Full pipeline:
         - retrieve top-k chunks from Chroma
         - build context
         - call LLM
         - return answer + structured sources
       """
    chunks = retrieve_context(question, k=k)
    context_block = build_context_block(chunks)
    answer = call_llm(question,context_block)

    sources = [] ## this section pulls out metadata from relevant chunks
    for ch in chunks:
        meta = ch["metadata"]
        sources.append(
            {
                "commentary":meta.get("commentary"),
                "page": meta.get("page"),
                "source_file":meta.get("source_file"),

            }
        )
    return {
        "answer":answer,
        "sources":sources,
    }

if __name__ == "__main__":
    print ("Bible commentary assistant")
    print("Type a question, or just press enter to quit.\n")

    while True:
        q = input("Your question: ").strip()
        if not q:
            print("goodbye")
            break

        result = answer_question(q,k=5)

        print("\n=== ANSWER ===")
        print(result["answer"])
        print("\n=== SOURCES ===")
        for s in result["sources"]:
            print(f"- {s['commentary']} (page {s['page']}, file {s['source_file']})")
        print("\n" + "=" * 40 + "\n")

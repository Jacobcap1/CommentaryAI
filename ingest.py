import os # working with file paths and variables
import uuid # Unique Ids for each chunk
import glob #finding all pdfs in a folder

import pdfplumber # extracts text from pdf files
import chromadb # chroma vector db
from chromadb import EmbeddingFunction
from chromadb.utils import embedding_functions #gives access to OpenAI embedding wrapper

import time

from openai import OpenAI
openai_client = OpenAI(api_key=os.environ.get("KEY"))

def embed_with_retry(texts, model="text-embedding-3-large",max_retries = 10):
    #retry embedding requests when hitting rate limits
    for attempt in range (max_retries):
        try:
            response = openai_client.embeddings.create(
                model=model,
                input=texts
            )
            return [item.embedding for item in response.data]

        except Exception as e:
            if "rate" in str(e).lower():
                wait = (2** attempt) *.25 #exponential backoff
                print("f\nRate limit hit. Waiting {wait:.2f} seconds...")
                time.sleep(wait)
            else:
                raise e
        raise RuntimeError("max retries exceeded for OpenAI embedding calls")

# this is where ingest.py lives
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

#This is where the embedding algorithm will store vectors
PERSIST_DIR = os.path.join(BASE_DIR,"db")

os.makedirs(PERSIST_DIR,exist_ok=True)

# Connect to Chroma Database
chroma_client = chromadb.PersistentClient(path=PERSIST_DIR)

#Set up OPENAI embeddings (using my key "KEY" env var)
##openai_ef = embedding_functions.OpenAIEmbeddingFunction(
 ##   api_key=os.environ.get("KEY"), #load API key from environment
 ##   model_name="text-embedding-3-large", #Embedding model you want to use
 ##   )
class RetryableOpenAIEmbedding(EmbeddingFunction):
    def __call__(self,input):
        return embed_with_retry(input)

openai_ef = RetryableOpenAIEmbedding()


#Get collection where commentaries live
collection = chroma_client.get_or_create_collection(
    name="bible_commentaries", #collection name (like table from database)
    embedding_function=openai_ef, #Embedding function that chroma should use
)

def extract_pages(pdf_path):
    """return list of dicts: {page,text} for each page"""
    pages = [] # where we store page text
    with pdfplumber.open(pdf_path) as pdf: # open pdf file
        for i, page in enumerate(pdf.pages, start=1): #Loop through each page
            text = page.extract_text() #Extract text from page
            if text:#only keep pages that contain text
                pages.append({"page": i, "text": text})
    return pages #return a list of page dictionaries


def chunk_text(text,chunk_size=800, overlap=200):
    """Simple word based chunking test"""
    words = text.split() #split into words
    chunks = [] # where chunks will be stored
    start = 0 #  starting index for the chunk

    while start < len(words): # keep chunking until we run out of words
        end = start + chunk_size # Ending index for this chunk
        chunk_words = words[start:end] # extract this chunk of words
        if not chunk_words: #if empty, stop
            break
        chunks.append(" ".join(chunk_words)) # join words back into string
        start += chunk_size - overlap # move forward with overlap included
    return chunks # return list of chunks

def ingest_commentary(pdf_path, commentary_name): # process and insert a single commentary PDF into Chroma
    print(f"Ingesting: {commentary_name} from {pdf_path}")

    pages = extract_pages(pdf_path) # extract all page text from PDF

    ids = [] # unique ids for chroma docs
    docs = [] # chunk text bodies
    metas =[] # metadata for each chunk

    for page_info in pages: #loop through each extracted page
        page_num = page_info["page"] #Page Number
        text = page_info["text"] #The text on that page

        chunks = chunk_text(text) # chunk the page into smaller sections

        for idx, chunk in enumerate(chunks):
            doc_id = str(uuid.uuid4()) # generate unique id for the chunk
            ids.append(doc_id) # add id to list
            docs.append(chunk) # add chunk text
            metas.append(#add metadata for the chunk
                {
                    "commentary": commentary_name,
                    "page": page_num,
                    "chunk_index": idx,
                    "source_file": os.path.basename(pdf_path),
                }
            )
            print(f"Total chunks for {commentary_name}before adding:", len(docs))
    if docs: ## only write to chroma if we have docs to store

        BATCH_SIZE = 50
        for i in range(0, len(docs), BATCH_SIZE):
            batch_docs = docs[i:i+BATCH_SIZE]
            batch_ids = ids[i:i+BATCH_SIZE]
            batch_metas = metas[i:i+BATCH_SIZE]

            collection.add(
                ids = batch_ids, # unique chunk ids
                documents = batch_docs, # chunk content
                metadatas = batch_metas, # metadata for each chunk

            )

        print(f"Added {len(docs)} chunks for {commentary_name}")


def main():

    pdf_files = glob.glob(os.path.join("commentaries", "*pdf"))# find all PDF files inside "Commentaries" folder

    if not pdf_files:
        print ("No PDFs found in ./commentaries")
        return

    print("Final collection chunk count:", collection.count())
    for pdf in pdf_files: # process pdfs
        #use filename (without .pdf) as commentary name
        commentary_name = os.path.splitext(os.path.basename(pdf))[0] # remove .pdf extension
        ingest_commentary(pdf,commentary_name) # ingest pdf

    print("Done ingesting all commentaries")

# when this file is run directly, execute main();
if __name__ == "__main__":
    main()

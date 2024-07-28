import pickle
from langchain_chroma import Chroma
from langchain_ai21 import AI21Embeddings

data = ""

with open("SISWA_900_150.cache", "rb") as file_bytes_reader:
    data = pickle.loads(file_bytes_reader.read())

embedding_function = AI21Embeddings(api_key="xlDvCjQch5NcZyokpRheEvc3l8QfEU4j")

db = Chroma.from_documents(data[0:5], embedding_function, persist_directory="./chromadb")

query = "Unsur-Unsur Berita"

docs = db.similarity_search(query)

print(docs[0].page_content)

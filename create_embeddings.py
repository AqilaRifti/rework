import pickle
from langchain_chroma import Chroma
from langchain_fireworks import FireworksEmbeddings

# Load data from pickle file
with open("SISWA_900_150.cache", "rb") as file_bytes_reader:
    data = pickle.loads(file_bytes_reader.read())

# Initialize the embedding function
embedding_function = FireworksEmbeddings(model="nomic-ai/nomic-embed-text-v1.5", fireworks_api_key="fpbRTHn1iqBjdjGEvbkGxozyvvwMwYzxDpumZKH2TUQuNjbt")

# Define batch size
batch_size = 256

# Function to batch documents
def batch_documents(documents, batch_size):
    for i in range(0, len(documents), batch_size):
        yield documents[i:i + batch_size]

# Load existing database or create a new one
try:
    db = Chroma.load(persist_directory="./siswadb")
except:
    db = Chroma(embedding_function=embedding_function, persist_directory="./siswadb")

# Process documents in batches
for batch in batch_documents(data, batch_size):
    db.add_documents(batch)  # Add documents to the existing collection


# Optionally, print a message indicating completion
print("Documents added to Chroma database successfully.")

exit()
import pickle
from langchain_chroma import Chroma
from langchain_fireworks import FireworksEmbeddings
data = ""

with open("SISWA_900_150.cache", "rb") as file_bytes_reader:
    data = pickle.loads(file_bytes_reader.read())

#embedding_function = AI21Embeddings(api_key="xlDvCjQch5NcZyokpRheEvc3l8QfEU4j", batch_size=len(data))
embedding_function = FireworksEmbeddings(model="nomic-ai/nomic-embed-text-v1.5", fireworks_api_key="fpbRTHn1iqBjdjGEvbkGxozyvvwMwYzxDpumZKH2TUQuNjbt")


db = Chroma.from_documents(data, embedding_function, persist_directory="./siswadb")

from plugins import *

class Memory():

    def __init__(self):
        from llama_index.core import Settings
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding
        Settings.embed_model = HuggingFaceEmbedding()
        Settings.llm = None
        
        from llama_index.core import SimpleDirectoryReader
        self.documents =  SimpleDirectoryReader("./plugins/retrieve/priv").load_data()
        
        from llama_index.core import VectorStoreIndex
        self.vector_index = VectorStoreIndex.from_documents(self.documents)
        
        self.query_engine = self.vector_index.as_query_engine()
        
    def add(self, new_knowledge: str) -> None | MemoryError:
        self.vector_index.insert(new_knowledge)

    def query(self, query: str) -> str | MemoryError:
        logger.warning(f"querying keyword {query}")
        return str(self.query_engine.query(query))
    

from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
from config import config
import time

class VectorStore:
    def __init__(self):
        self.pc = Pinecone(api_key=config.PINECONE_API_KEY)
        self.index_name = config.PINECONE_INDEX_NAME
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self._initialize_index()
    
    def _initialize_index(self):
        """Initialize Pinecone index if it doesn't exist"""
        if self.index_name not in self.pc.list_indexes().names():
            self.pc.create_index(
                name=self.index_name,
                dimension=384,
                metric='cosine',
                spec=ServerlessSpec(
                    cloud='aws',
                    region='us-east-1'
                )
            )
            while not self.pc.describe_index(self.index_name).status['ready']:
                time.sleep(1)
        
        self.index = self.pc.Index(self.index_name)
    
    def add_knowledge(self, text: str, metadata: dict = None):
        """Add knowledge to the vector database"""
        embedding = self.embedding_model.encode(text).tolist()
        vector_id = str(hash(text))
        
        self.index.upsert(
            vectors=[{
                'id': vector_id,
                'values': embedding,
                'metadata': {'text': text, **(metadata or {})}
            }]
        )
        return vector_id
    
    def search(self, query: str, top_k: int = 3):
        """Search for relevant knowledge"""
        query_embedding = self.embedding_model.encode(query).tolist()
        results = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )
        return results
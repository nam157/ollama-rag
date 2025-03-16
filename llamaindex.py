import nest_asyncio
nest_asyncio.apply()

from llama_parse import LlamaParse
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.llms.ollama import Ollama
from llama_index.core.node_parser import MarkdownElementNodeParser
from llama_index.embeddings.ollama import OllamaEmbedding
import pandas as pd

class DataParser:
    def __init__(self, api_key, file_path):
        self.api_key = api_key
        self.file_path = file_path
        self.parser = LlamaParse(api_key=self.api_key, result_type='markdown')

    def parse(self):
        return self.parser.load_data(self.file_path)

class NodeProcessor:
    def __init__(self, llm_model='llama2', num_workers=4):
        self.llm = Ollama(model=llm_model)
        self.node_parser = MarkdownElementNodeParser(llm=self.llm, num_workers=num_workers)

    def process_nodes(self, documents):
        base_nodes, objects = self.node_parser.get_nodes_and_objects(documents)
        return base_nodes, objects
    
class IndexBuilder:
    def __init__(self, llm_model='llama2', embed_model_name='llama2'):
        self.llm = Ollama(model=llm_model)
        self.embed_model = OllamaEmbedding(model_name=embed_model_name)

    def build_index(self, nodes):
        return VectorStoreIndex(nodes=nodes, llm=self.llm, embed_model=self.embed_model)

class QueryEngine:
    def __init__(self, index, similarity_top_k=5, llm = None):
        self.query_engine = index.as_query_engine(similarity_top_k=similarity_top_k, llm = llm)

    def query(self, question):
        return self.query_engine.query(question)

def main():
    # Danh sách các file Excel cần xử lý
    excel_files = [
        '/home/namnh1/rag-llm-chatbot/BC AI/BC Nhân sự.xlsx',
        '/home/namnh1/rag-llm-chatbot/BC AI/BC Tiến độ sản xuất.xlsx',
        '/home/namnh1/rag-llm-chatbot/BC AI/bc.xlsx'
    ]
    
    # Khởi tạo và parse dữ liệu
    parser = DataParser(api_key='llx-JUKgXwAJ4IrHHB1IFyDKh34IwHwPHiVPfyPbkXFIKTfkVhdS', file_path=excel_files)
    documents = parser.parse()
    

    # Xử lý nodes
    processor = NodeProcessor()
    base_nodes, objects = processor.process_nodes(documents)

    # Xây dựng chỉ mục
    index_builder = IndexBuilder()
    index = index_builder.build_index(base_nodes + objects)

    # Truy vấn
    query_engine = QueryEngine(index, similarity_top_k=5, llm =  index_builder.llm)
    response = query_engine.query("Nam Thắng tổng công nợ bao nhiêu?")

    # Hiển thị kết quả
    from IPython.display import Markdown, display
    print(f"{response}")
    
if __name__ == "__main__":
    main()
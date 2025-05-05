import os
import json
import re
import logging
import hashlib
import warnings
import chromadb
from chromadb.config import Settings

from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader, PyMuPDFLoader
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    RecursiveJsonSplitter,
    HTMLSemanticPreservingSplitter,
)
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_community.document_loaders import UnstructuredExcelLoader
from langchain_community.vectorstores.utils import filter_complex_metadata

from .fintastic import FinDataExtractor 


warnings.filterwarnings("ignore", module="pypdf")
logging.getLogger("pypdf").setLevel(logging.ERROR)

os.environ["USER_AGENT"] = "ABFRL-RAG/1.0"

class FinVecStore:
    def __init__(self):
        self.current_dir = os.path.dirname(os.path.abspath(__file__))
        self.docs_dir = os.path.join(self.current_dir, "docs")
        self.pdf_paths = [
            os.path.join(self.docs_dir, name) for name in [
                # "ABFRL_01052025170724_SE_intimation-appointment.pdf",
                # "ABFRL_16042025193034_SE.pdf",
                # "ABFRL_02042025233743_SE.pdf"
            ]
        ]
        self.excel_paths = [
            os.path.join(self.docs_dir, name) for name in [
                # "ABFRL_financials.xlsx"
            ]
        ]
        self.web_urls = [
            # "https://www.abfrl.com/about-us/",
            ]
        self.screener_urls = [
                         "https://www.screener.in/company/ABFRL/"
                         ]
        self.chroma_client = chromadb.PersistentClient(path=os.path.join(self.current_dir, "db"),
                                                       settings=Settings(anonymized_telemetry=False))
        self.embedding = OpenAIEmbeddings()
        self.collection_name = "abfrl-pdfs"
        self.logger = logging.getLogger(self.__class__.__name__)

    def _get_content_hash(self, content: str) -> str:
        return hashlib.md5(content.encode("utf-8")).hexdigest()

    def _already_exists(self, doc_hash: str, hashes_seen: set) -> bool:
        return doc_hash in hashes_seen
    
    def _clean_content(self, text: str) -> str:
        """ Clean text data of any unnecessary characters"""
        # Fix broken up characters like "M\na\nr\n" → "Mar"
        text = re.sub(r'(\\n)+', '\n', text)  # Unescape repeated `\n` sequences
        text = text.encode("utf-8").decode("unicode_escape")  # Convert escaped sequences
        text = re.sub(r'\n+', '\n', text)  # Collapse multiple newlines
        text = re.sub(r'[ \t]+', ' ', text)  # Collapse excessive spaces/tabs
        text = text.replace('\u00a0', ' ')  # Replace non-breaking spaces
        lines = text.splitlines()
        lines = [line.strip() for line in lines if line.strip()]
        return ' '.join(lines)
    
    def _convert_screener_json_to_documents(self, json_data):
        """
        Converts a nested Screener JSON dictionary into a list of LangChain Document objects.
        """
        docs = []
        for section, content in json_data.items():
            if isinstance(content, str):
                text = f"{section}: {content}"
            elif isinstance(content, list):
                text = f"{section}:\n" + "\n\n".join(
                    "\n".join(f"{k}: {v}" for k, v in item.items()) if isinstance(item, dict) else str(item)
                    for item in content
                )
            elif isinstance(content, dict):
                text = f"{section}:\n" + "\n".join(f"{k}: {v}" for k, v in content.items())
            else:
                continue

            docs.append(Document(page_content=self._clean_content(text), metadata={"section": section}))
        return docs

    def _split_and_dedup_docs(self, docs_or_json, splitter, seen_hashes, all_chunks, source_type="generic"):
        """
        Handles splitting logic depending on the splitter type (e.g. text, HTML, JSON),
        and deduplicates based on page content.
        """
        if isinstance(splitter, RecursiveJsonSplitter):
            # Handle structured JSON input (not a list of Documents)
            json_chunks = splitter.create_documents(texts=[docs_or_json])  # docs_or_json is dict
            for chunk in json_chunks:
                text = chunk.page_content.strip()
                if len(text) < 20 or not any(c.isalnum() for c in text):
                    continue
                doc_hash = self._get_content_hash(text)
                if not self._already_exists(doc_hash, seen_hashes):
                    chunk.metadata["source_type"] = source_type
                    all_chunks.append(chunk)
                    seen_hashes.add(doc_hash)

        else:
            # Handle Document-based splitters
            for chunk in splitter.split_documents(docs_or_json):
                text = chunk.page_content.strip()
                if len(text) < 20 or not any(c.isalnum() for c in text):
                    continue
                doc_hash = self._get_content_hash(text)
                if not self._already_exists(doc_hash, seen_hashes):
                    chunk.metadata["source_type"] = source_type
                    all_chunks.append(chunk)
                    seen_hashes.add(doc_hash)

    def generate_vectorstore(self):
        all_chunks = []
        seen_hashes = set()

        # --- Web Pages ---
        self.logger.info("Loading documents from web...")
        for doc in WebBaseLoader(self.web_urls).lazy_load():
            doc.page_content = self._clean_content(doc.page_content)
            splitter = HTMLSemanticPreservingSplitter(chunk_size=1000, 
                                                      min_split_chars=500,
                                                      length_function=len,
                                                      separators=["\n\n", "\n", " "],
                                                      is_separator_regex=False)
            self._split_and_dedup_docs([doc], splitter, seen_hashes, all_chunks, source_type="web")

        # --- PDFs ---
        self.logger.info("Loading PDF documents...")
        for file in self.pdf_paths:
            pdf_docs = PyMuPDFLoader(file).load()
            for doc in pdf_docs:
                doc.page_content = self._clean_content(doc.page_content)
            splitter = RecursiveCharacterTextSplitter(chunk_size=800,
                                                      chunk_overlap=100,
                                                      length_function=len,
                                                      separators=["\n\n", "\n", " "],
                                                      is_separator_regex=False)
            self._split_and_dedup_docs(pdf_docs, splitter, seen_hashes, all_chunks, source_type="pdf")

        # --- Excel Files ---
        self.logger.info("Loading excel spreadsheets...")
        for file in self.excel_paths:
            excel_docs = UnstructuredExcelLoader(file, mode="elements").load()
            splitter = RecursiveCharacterTextSplitter(chunk_size=500,
                                                      chunk_overlap=50,
                                                      length_function=len,
                                                      separators=["\n\n", "\n", " "],
                                                      is_separator_regex=False)
            self._split_and_dedup_docs(excel_docs, splitter, seen_hashes, all_chunks, source_type="excel")

        # --- Screener JSON ---
        self.logger.info("Extracting Screener data dynamically...")
        for url in self.screener_urls:
            try:
                extractor = FinDataExtractor(url=url, use_browser=False)
                json_data = extractor.extract_all()
                # Use one of these ways to get the splitter
                
                # This is to use json.dumps to convert into plain text
                # json_docs = json.dumps(json_data, indent=2)
                # splitter = RecursiveJsonSplitter(max_chunk_size=800)
                
                # This is to use RecursiveCharacterTextSplitter
                json_docs = self._convert_screener_json_to_documents(json_data)
                # Clean each JSON doc before splitting
                for doc in json_docs:
                    doc.page_content = self._clean_content(doc.page_content)
                splitter = RecursiveCharacterTextSplitter(chunk_size=500,
                                                      chunk_overlap=50,
                                                      length_function=len,
                                                      separators=["\n\n", "\n", " "],
                                                      is_separator_regex=False)
                
                self._split_and_dedup_docs(json_docs, splitter, seen_hashes, all_chunks, source_type="screener")
            except Exception as e:
                self.logger.warning(f"[!] Failed to extract from Screener URL: {url} — {e}")

        self.logger.info("Total unique chunks prepared: %d", len(all_chunks))

        filtered_chunks = filter_complex_metadata(all_chunks)
        self.logger.info("Total chunks before filtering: %d", len(all_chunks))
        self.logger.info("Filtered chunks retained: %d", len(filtered_chunks))

        if not filtered_chunks:
            self.logger.warning("No valid chunks found for embedding. Vectorstore will not be created.")
            return None

        self.logger.info("Creating vectorstore with Chroma...")

        vectorstore = Chroma.from_documents(
            documents=filtered_chunks,
            embedding=self.embedding,
            collection_name=self.collection_name,
            client=self.chroma_client,
        )

        self.logger.info("✅ Vectorstore created successfully.")
        self._log_vectorstore_metadata(vectorstore)

        return vectorstore


    def _log_vectorstore_metadata(self, vectorstore):
        retriever = vectorstore.as_retriever()
        try:
            # Use a dummy query to get some documents
            sample_docs = retriever.invoke("ABFRL")[:5]
            total_tokens = sum(len(doc.page_content.split()) for doc in sample_docs)
            self.logger.info("Sample documents fetched: %d", len(sample_docs))
            self.logger.info("Estimated total tokens in sample: %d", total_tokens)
        except Exception as e:
            self.logger.warning("Metadata estimation failed: %s", str(e))

    def query_vectorstore(self, query: str, top_k: int = 5) -> str:
        vectorstore = Chroma(
            client=self.chroma_client,
            collection_name=self.collection_name,
            embedding_function=self.embedding,
        )
        results = vectorstore.similarity_search(query, k=top_k)
        return "\n\n".join(doc.page_content for doc in results)

    @staticmethod
    def query_existing_vectorstore(client, query: str, collection_name: str, top_k: int = 5) -> str:
        embedding = OpenAIEmbeddings()
        vectorstore = Chroma(
            client=client,
            collection_name=collection_name,
            embedding_function=embedding,
        )
        results = vectorstore.similarity_search(query, k=top_k)
        # for i, doc in enumerate(results):
        #     print(f"\n--- Document {i} ---")
        #     print("Content:", repr(doc.page_content))
        #     print("Metadata:", doc.metadata)
        return "\n\n".join(doc.page_content for doc in results)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    store = FinVecStore()
    store.generate_vectorstore()

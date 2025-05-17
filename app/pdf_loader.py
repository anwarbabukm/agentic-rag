import logging
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def load_and_chunk_pdf(pdf_path: str, chunk_size: int = 1000, chunk_overlap: int = 200):
    logger.info(f"Loading PDF from path: {pdf_path}")
    
    try:
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        logger.info(f"Loaded {len(documents)} document(s) from {pdf_path}")
    except Exception as e:
        logger.error(f"Error loading PDF {pdf_path}: {e}")
        raise

    logger.info(f"Splitting documents into chunks (chunk_size={chunk_size}, overlap={chunk_overlap})")
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(documents)
    logger.info(f"Generated {len(chunks)} chunk(s) from {pdf_path}")

    return chunks
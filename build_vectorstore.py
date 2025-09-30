import os
import shutil
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

DATA_PATH = "./data"
VECTORSTORE_PATH = "./vectorstore"

def build_vectorstore():
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY가 .env 파일에 설정되지 않았습니다.")

    if os.path.exists(VECTORSTORE_PATH):
        print(f"경고: 기존 '{VECTORSTORE_PATH}' 폴더를 삭제하고 다시 생성합니다.")
        shutil.rmtree(VECTORSTORE_PATH)
    
    print(f"'{DATA_PATH}' 폴더에서 문서를 로드합니다...")
    loader = DirectoryLoader(
        DATA_PATH,
        glob="**/*",
        loader_cls=lambda path: PyPDFLoader(path) if path.lower().endswith(".pdf") else TextLoader(path, encoding='utf-8'),
        show_progress=True,
        use_multithreading=True
    )
    documents = loader.load()
    if not documents:
        print(f"'{DATA_PATH}' 폴더에 처리할 파일이 없습니다. 스크립트를 종료합니다.")
        return

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = text_splitter.split_documents(documents)
    print(f"총 {len(docs)}개의 문서 조각으로 분할되었습니다.")

    # [수정됨] 더 가벼운 'small' 모델로 변경하여 메모리 사용량을 줄입니다.
    embedding_model = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY, model="text-embedding-3-small")

    print("문서를 임베딩하고 벡터 저장소를 생성합니다. 시간이 걸릴 수 있습니다...")
    vectorstore = FAISS.from_documents(
        documents=docs, 
        embedding=embedding_model
    )
    
    # FAISS 벡터스토어 저장
    os.makedirs(VECTORSTORE_PATH, exist_ok=True)
    vectorstore.save_local(VECTORSTORE_PATH)
    
    print(f"벡터 저장소 생성이 완료되었습니다. '{VECTORSTORE_PATH}'에 저장되었습니다.")

if __name__ == '__main__':
    build_vectorstore()

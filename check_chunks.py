import os
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# build_vectorstore.py와 동일한 설정을 사용합니다.
DATA_PATH = "./data"
OUTPUT_FILE = "chunks_preview.txt" # 결과를 저장할 파일 이름

def check_document_chunks():
    """
    data 폴더의 문서를 로드하고 텍스트 조각으로 나눈 뒤,
    그 결과를 파일로 저장하여 직접 확인할 수 있게 합니다.
    """
    print(f"'{DATA_PATH}' 폴더에서 문서를 로드합니다...")
    loader = DirectoryLoader(
        DATA_PATH,
        glob="**/*.pdf", # PDF 파일만 대상으로 확인
        loader_cls=PyPDFLoader,
        show_progress=True,
        use_multithreading=True
    )
    documents = loader.load()
    if not documents:
        print(f"'{DATA_PATH}' 폴더에 확인할 PDF 파일이 없습니다.")
        return

    print("문서를 텍스트 조각으로 분할합니다...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    docs = text_splitter.split_documents(documents)
    
    print(f"총 {len(docs)}개의 문서 조각을 생성했습니다.")
    
    # 확인을 위해 처음 10개의 조각만 파일에 저장합니다.
    num_chunks_to_preview = 10
    
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(f"--- 총 {len(docs)}개의 조각 중 처음 {num_chunks_to_preview}개 미리보기 ---\n\n")
        
        for i, doc in enumerate(docs[:num_chunks_to_preview]):
            f.write(f"=============== Chunk {i+1} ===============\n")
            f.write(f"출처(Source): {doc.metadata.get('source', 'N/A')}\n")
            f.write(f"페이지(Page): {doc.metadata.get('page', 'N/A')}\n")
            f.write("----------------- 내용 -----------------\n")
            f.write(doc.page_content)
            f.write("\n\n")

    print(f"'{OUTPUT_FILE}' 파일에 미리보기 결과가 저장되었습니다.")
    print("파일을 열어 텍스트가 깨지거나 이상하게 분할되지 않았는지 확인해주세요.")

if __name__ == '__main__':
    check_document_chunks()

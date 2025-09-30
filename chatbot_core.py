import os
import pickle
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

VECTORSTORE_PATH = "./vectorstore"
DATA_PATH = "./data"

def get_embedding_model(api_key):
    """OpenAI 임베딩 모델을 가져옵니다."""
    return OpenAIEmbeddings(openai_api_key=api_key, model="text-embedding-3-small")

def build_vectorstore(embedding_model):
    """TXT 파일만 임베딩하여 벡터 저장소를 생성합니다."""
    from langchain_community.document_loaders import DirectoryLoader, TextLoader
    from langchain.text_splitter import RecursiveCharacterTextSplitter

    print("새로운 벡터 저장소를 생성합니다.")
    loader = DirectoryLoader(
        DATA_PATH,
        glob="**/*.txt",
        loader_cls=TextLoader,
        show_progress=True,
        loader_kwargs={'encoding': 'utf-8'}
    )
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = text_splitter.split_documents(documents)
    print(f"총 {len(docs)}개의 문서 조각으로 분할되었습니다.")
    
    # FAISS 벡터스토어 생성
    vectorstore = FAISS.from_documents(
        documents=docs,
        embedding=embedding_model
    )
    
    # 벡터스토어 저장
    os.makedirs(VECTORSTORE_PATH, exist_ok=True)
    vectorstore.save_local(VECTORSTORE_PATH)
    print("벡터 저장소 생성이 완료되었습니다.")
    return vectorstore

def get_vectorstore(embedding_model):
    """디스크에 저장된 벡터 저장소를 불러옵니다."""
    faiss_index_path = os.path.join(VECTORSTORE_PATH, "index.faiss")
    faiss_pkl_path = os.path.join(VECTORSTORE_PATH, "index.pkl")
    
    if not (os.path.exists(faiss_index_path) and os.path.exists(faiss_pkl_path)):
        raise FileNotFoundError(
            f"'{VECTORSTORE_PATH}' 벡터 저장소를 찾을 수 없습니다. "
            f"먼저 'python build_vectorstore.py'를 실행하여 벡터 저장소를 생성해주세요."
        )
    print("기존 벡터 저장소를 로드합니다.")
    return FAISS.load_local(VECTORSTORE_PATH, embedding_model, allow_dangerous_deserialization=True)

def create_sample_vectorstore(embedding_model):
    """샘플 데이터로 벡터 저장소를 생성합니다 (Streamlit Cloud용)."""
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_core.documents import Document
    
    # 샘플 금융 규제 데이터
    sample_texts = [
        """개인정보보호법 제15조 (개인정보의 수집·이용)
① 개인정보처리자는 다음 각 호의 어느 하나에 해당하는 경우에는 개인정보를 수집할 수 있으며 그 수집 목적의 범위에서 이용할 수 있다.
1. 정보주체의 동의를 받은 경우
2. 법률에 특별한 규정이 있거나 법령상 의무를 준수하기 위하여 불가피한 경우
3. 공공기관이 법령 등에서 정하는 소관 업무의 수행을 위하여 불가피한 경우""",
        
        """전자금융거래법 제21조 (전자금융거래 기록의 생성 및 보존)
① 전자금융업자 및 전자금융거래를 하는 금융기관은 전자금융거래에 관한 기록을 생성하여야 하며, 전자금융거래가 종료된 후 5년간 보존하여야 한다.
② 전자금융업자 및 전자금융거래를 하는 금융기관은 전자금융거래 기록을 위조·변조하거나 그 기록을 훼손 또는 누설하여서는 아니 된다.""",
        
        """정보통신망 이용촉진 및 정보보호 등에 관한 법률 제28조 (개인정보의 수집 제한 등)
① 정보통신서비스 제공자는 이용자의 개인정보를 수집하는 때에는 다음 각 호의 사항을 이용자에게 알리고 동의를 받아야 한다.
1. 개인정보의 수집·이용 목적
2. 수집하는 개인정보의 항목
3. 개인정보의 보유·이용 기간
4. 동의를 거부할 권리가 있다는 사실 및 동의 거부에 따른 불이익이 있는 경우에는 그 불이익의 내용"""
    ]
    
    # 문서 객체 생성
    documents = []
    for i, text in enumerate(sample_texts):
        doc = Document(
            page_content=text,
            metadata={"source": f"sample_doc_{i+1}.txt", "page": i+1}
        )
        documents.append(doc)
    
    # 텍스트 분할
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = text_splitter.split_documents(documents)
    
    # FAISS 벡터스토어 생성 (메모리에만 저장)
    vectorstore = FAISS.from_documents(documents=docs, embedding=embedding_model)
    
    print(f"샘플 벡터 저장소 생성 완료: {len(docs)}개 문서 조각")
    return vectorstore

def create_rag_chain(vectorstore, api_key):
    """RAG 체인을 생성합니다."""
    llm = ChatOpenAI(openai_api_key=api_key, model_name="gpt-4o", temperature=0)
    prompt = ChatPromptTemplate.from_template("""
    당신은 금융감독원에서 20년간 근무한 시니어 금융 규제 전문가입니다. 
    금융보안, 개인정보보호, 전자금융거래법, 자본시장법 등 모든 금융 규제에 정통하며, 
    금융회사들의 컴플라이언스 업무를 직접 지도해온 실무 전문가입니다.
    
    === 답변 원칙 ===
    
    1. **전문성과 정확성**
       - 금융 규제 전문가로서 정확하고 신뢰할 수 있는 정보만 제공
       - 법령 조항, 시행령, 고시 등을 정확히 인용
       - 불확실한 정보는 명확히 구분하여 안내
    
    2. **실무 적용성**
       - 이론적 설명과 함께 실무에서 어떻게 적용하는지 구체적 가이드 제공
       - 금융회사 입장에서 실제 준수해야 할 사항들을 명확히 설명
       - 위반 시 제재 수준과 리스크를 구체적으로 안내
    
    3. **맞춤형 조언**
       - 질문자의 상황(금융회사 규모, 업종 등)을 고려한 맞춤형 답변
       - 단계별 실행 방안과 체크포인트 제시
       - 관련 부서(법무팀, IT팀, 컴플라이언스팀 등)별 역할 안내
    
    === 답변 구조 ===
    
    **📋 핵심 답변**
    [질문에 대한 직접적이고 명확한 답변]
    
    **📖 법적 근거**
    [관련 법령, 조항, 고시 등을 정확히 명시]
    출처: [문서명, 페이지/조항] (Context에서 참조한 경우)
    
    **⚖️ 실무 적용**
    [실제 금융회사에서 어떻게 적용해야 하는지 구체적 가이드]
    
    **⚠️ 주의사항**
    [준수하지 않을 경우의 리스크와 제재 수준]
    
    **💡 권장사항**
    [모범 사례와 추가 개선 방안]
    
    === Context 활용 지침 ===
    
    제공된 Context: {context}
    
    1. Context에 관련 정보가 있는 경우:
       - Context의 정보를 최우선으로 활용
       - 출처를 명확히 표기: [출처: 파일명, 페이지/섹션]
       - Context의 내용을 바탕으로 실무적 해석과 적용 방안 제시
    
    2. Context에 충분한 정보가 없는 경우:
       - 금융 규제 전문가로서의 기존 지식 활용
       - "일반적인 금융 규제 관점에서..." 라고 명시
       - 추가 확인이 필요한 부분은 "관련 부서에 재확인 권장" 안내
    
    3. Context와 기존 지식이 상충하는 경우:
       - Context의 정보를 우선하되 "최신 규정 변경사항 확인 필요" 안내
    
    === 질문 분석 ===
    
    사용자 질문: {input}
    
    질문 유형 판단:
    - 법령 해석: 정확한 조항과 해석 제공
    - 실무 적용: 구체적 실행 방안 제시  
    - 리스크 문의: 제재 수준과 대응 방안 안내
    - 모범 사례: 업계 우수 사례와 권장사항 제시
    
    위의 구조에 따라 전문적이고 실무적인 답변을 제공해주세요.
    모든 답변은 한국어로 작성하며, 금융회사 실무진이 바로 활용할 수 있는 수준으로 구체적으로 작성해주세요.
    """)
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    return retrieval_chain
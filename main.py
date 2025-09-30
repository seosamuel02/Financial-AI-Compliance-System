import streamlit as st
import os
import plotly.graph_objects as go
import pandas as pd
from dotenv import load_dotenv
from chatbot_core import get_embedding_model, get_vectorstore, build_vectorstore, create_rag_chain, create_sample_vectorstore
from langchain_community.document_loaders import PyPDFLoader
from multi_agent_system import MultiAgentAnalysisSystem, create_intelligent_router

def _render_score_card(card):
    """점수 카드 렌더링 함수"""
    score = card["score"]
    category = card["category"]
    grade = card["grade"]
    reason = card["reason"]
    
    # 점수에 따른 색상 설정
    if score >= 90:
        color = "#16a34a"  # 녹색
        bg_color = "#f0fdf4"
    elif score >= 80:
        color = "#3b82f6"  # 파란색
        bg_color = "#eff6ff"
    elif score >= 70:
        color = "#f59e0b"  # 주황색
        bg_color = "#fffbeb"
    else:
        color = "#dc2626"  # 빨간색
        bg_color = "#fef2f2"
    
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, {bg_color} 0%, white 100%);
        border: 1px solid {color};
        border-radius: 12px;
        padding: 1.5rem;
        margin: 0.5rem 0;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
        transition: transform 0.2s;
    ">
        <h2 style="color: {color}; margin: 0; font-size: 2.5rem; font-weight: 700;">
            {score}
        </h2>
        <h4 style="color: {color}; margin: 0.3rem 0; font-size: 1.1rem;">
            {grade}
        </h4>
        <p style="color: #1f2937; margin: 0.5rem 0; font-weight: 600; font-size: 1rem;">
            {category}
        </p>
        <p style="color: #6b7280; margin: 0; font-size: 0.85rem; line-height: 1.4;">
            {reason[:60]}{"..." if len(reason) > 60 else ""}
        </p>
    </div>
    """, unsafe_allow_html=True)

# --- 환경변수 로드 ---
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

st.set_page_config(
    page_title="FSEC AI - 금융보안 규제 분석 시스템",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 커스텀 CSS 스타일
st.markdown("""
<style>
    /* 메인 헤더 스타일 */
    .main-header {
        background: linear-gradient(90deg, #1e3a8a 0%, #3b82f6 50%, #06b6d4 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    /* 사이드바 스타일 */
    .css-1d391kg {
        background: linear-gradient(180deg, #f8fafc 0%, #e2e8f0 100%);
    }
    
    /* 메트릭 카드 스타일 */
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #3b82f6;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        margin: 1rem 0;
    }
    
    /* 결과 박스 스타일 */
    .result-box {
        background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
        border: 1px solid #0ea5e9;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    
    /* 위험도 표시 */
    .risk-high { color: #dc2626; font-weight: bold; }
    .risk-medium { color: #f59e0b; font-weight: bold; }
    .risk-low { color: #16a34a; font-weight: bold; }
    
    /* 버튼 스타일 */
    .stButton > button {
        background: linear-gradient(90deg, #3b82f6 0%, #1d4ed8 100%);
        color: white;
        border: none;
        padding: 0.5rem 2rem;
        border-radius: 25px;
        font-weight: 600;
        transition: all 0.3s;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(59, 130, 246, 0.3);
    }
    
    /* 파일 업로더 스타일 */
    .stFileUploader {
        border: 2px dashed #3b82f6;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        background: #f8fafc;
    }
    
    /* 채팅 메시지 스타일 */
    .stChatMessage {
        border-radius: 15px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    
    /* 섹션 헤더 */
    .section-header {
        background: linear-gradient(90deg, #1e40af 0%, #3b82f6 100%);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        text-align: center;
        font-size: 1.2rem;
        font-weight: 600;
    }
    
    /* 정보 카드 */
    .info-card {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
    }
    
    /* 스피너 커스터마이징 */
    .stSpinner {
        color: #3b82f6;
    }
</style>
""", unsafe_allow_html=True)

# --- 메인 헤더 ---
st.markdown("""
<div class="main-header">
    <h1>🏛️ FSEC AI</h1>
    <h3>Financial Security Regulation Compliance System</h3>
    <p style="margin: 0; opacity: 0.9;">AI 기반 금융보안 규제 준수 분석 플랫폼</p>
</div>
""", unsafe_allow_html=True)

# --- 사이드바 ---
st.sidebar.markdown("""
<div style="text-align: center; padding: 1rem;">
    <h2 style="color: #1e40af; margin-bottom: 0;">🏛️ FSEC AI</h2>
    <p style="color: #64748b; font-size: 0.9rem; margin: 0;">금융보안 규제 시스템</p>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")

page = st.sidebar.radio(
    "📋 **기능 선택**",
    ["🤖 금융 보안 규제 QA 챗봇", "🔒 보안 적정성 평가", "🚀 AI 멀티에이전트 분석"],
    index=0
)

st.sidebar.markdown("---")

# 시스템 상태 표시
if OPENAI_API_KEY:
    st.sidebar.success("🟢 OpenAI API 연결됨")
else:
    st.sidebar.error("🔴 OpenAI API 설정 필요")

if TAVILY_API_KEY:
    st.sidebar.success("🟢 웹 검색 기능 활성화")
else:
    st.sidebar.warning("🟡 웹 검색 기능 비활성화")

st.sidebar.markdown("---")

st.sidebar.markdown("""
<div class="info-card">
    <h4 style="color: #1e40af; margin-top: 0;">💡 시스템 정보</h4>
    <ul style="margin: 0; padding-left: 1rem;">
        <li>🤖 AI 모델: GPT-4o</li>
        <li>📚 벡터 DB: FAISS</li>
        <li>🔍 웹 검색: Tavily</li>
        <li>🛡️ 보안: 로컬 처리</li>
    </ul>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("""
<div style="text-align: center; padding: 1rem; color: #64748b; font-size: 0.8rem;">
    © 2024 FSEC AI System<br>
    Financial Security Compliance
</div>
""", unsafe_allow_html=True)

@st.cache_resource
def initialize_chatbot(api_key):
    if not api_key:
        return None, None
    embedding_model = get_embedding_model(api_key)
    try:
        vectorstore = get_vectorstore(embedding_model)
        st.success("✅ 기존 벡터 저장소를 로드했습니다.")
    except FileNotFoundError:
        try:
            vectorstore = build_vectorstore(embedding_model)
            st.success("✅ 새로운 벡터 저장소를 생성했습니다.")
        except Exception as e:
            st.warning(f"⚠️ 벡터 저장소 생성 오류: {str(e)}")
            st.info("📝 샘플 데이터로 벡터 저장소를 생성합니다.")
            vectorstore = create_sample_vectorstore(embedding_model)
    except Exception as e:
        st.warning(f"⚠️ 벡터 저장소 로드 오류: {str(e)}")
        st.info("📝 샘플 데이터로 벡터 저장소를 생성합니다.")
        vectorstore = create_sample_vectorstore(embedding_model)
    
    rag_chain = create_rag_chain(vectorstore, api_key)
    return rag_chain, embedding_model

if OPENAI_API_KEY:
    rag_chain, embedding_model = initialize_chatbot(OPENAI_API_KEY)
    # 멀티에이전트 시스템 초기화
    multi_agent_system = MultiAgentAnalysisSystem(OPENAI_API_KEY, TAVILY_API_KEY)
    intelligent_router = create_intelligent_router(OPENAI_API_KEY)
else:
    rag_chain, embedding_model = None, None
    multi_agent_system = None
    intelligent_router = None


def search_additional_info(query, api_key):
    """Tavily API를 사용하여 추가 정보를 검색합니다."""
    try:
        from tavily import TavilyClient
        
        if not api_key:
            return "Tavily API 키가 설정되지 않았습니다."
        
        tavily = TavilyClient(api_key=api_key)
        
        # 금융 규제 관련 검색 쿼리 생성
        search_query = f"금융 규제 보안 가이드라인 {query} 금융위원회 금융보안원"
        
        search_result = tavily.search(
            query=search_query,
            search_depth="advanced",
            max_results=3,
            include_domains=["fss.or.kr", "fsc.go.kr", "fsec.or.kr"]  # 금융 관련 공식 사이트
        )
        
        additional_info = ""
        if search_result.get("results"):
            additional_info = "\n\n=== 추가 검색 정보 ===\n"
            for i, result in enumerate(search_result["results"][:3]):
                additional_info += f"\n**[웹 검색 {i+1}] {result.get('title', '')}**\n"
                additional_info += f"출처: {result.get('url', '')}\n"
                additional_info += f"내용: {result.get('content', '')[:300]}...\n"
        
        return additional_info
        
    except Exception as e:
        return f"웹 검색 중 오류 발생: {str(e)}"

def security_assessment_content(content, embedding_model, is_pdf=False, uploaded_file=None):
    from langchain_community.vectorstores import FAISS
    from langchain_core.documents import Document
    from langchain.text_splitter import RecursiveCharacterTextSplitter

    if is_pdf and uploaded_file:
        temp_dir = "temp"
        os.makedirs(temp_dir, exist_ok=True)
        temp_path = os.path.join(temp_dir, uploaded_file.name)
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        loader = PyPDFLoader(temp_path)
        documents = loader.load()
    elif content:
        documents = [Document(page_content=content, metadata={"source": "입력 텍스트"})]
    else:
        return "평가할 텍스트 또는 PDF 파일을 입력/업로드해주세요.", ""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
    docs = text_splitter.split_documents(documents)
    docs = docs[:30]
    vectorstore = FAISS.from_documents(docs, embedding=embedding_model)
    rag_chain_file = create_rag_chain(vectorstore, OPENAI_API_KEY)

    # 문서 내용에서 키워드 추출
    doc_keywords = ""
    if documents:
        sample_content = documents[0].page_content[:500]
        doc_keywords = sample_content
    
    # Tavily로 추가 정보 검색
    additional_info = search_additional_info(doc_keywords, TAVILY_API_KEY)
    
    prompt = f"""
    당신은 금융 보안 담당자입니다.
    입력된 문서의 내용을 바탕으로 보안 적정성 평가 체크리스트와 평가 의견을 작성해주세요.

    체크리스트에는 인증, 무결성, 개인정보보호, 로그관리, 외부위탁 등 주요 항목을 포함해주세요.
    각 항목별로 문서 근거(페이지/스니펫)도 반드시 명시하세요.
    
    추가 참고 정보:
    {additional_info}
    """
    response = rag_chain_file.invoke({"input": prompt})
    answer = response.get('answer', '평가 결과를 생성하지 못했습니다.')
    sources = ""
    if response.get("context"):
        for i, doc in enumerate(response["context"]):
            page = doc.metadata.get('page', '')
            snippet = doc.page_content[:200].replace('\n', ' ')
            sources += f"**[출처 {i+1}] (Page: {page})**\n"
            sources += f"> {snippet}...\n\n"
    return answer, sources

if rag_chain:
    if page == "🤖 금융 보안 규제 QA 챗봇":
        st.markdown("""
        <div class="section-header">
            🤖 금융 보안 규제 QA 챗봇
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-card">
            <p style="margin: 0; color: #64748b;">
                💡 금융보안원, 금융위원회 규제 문서를 기반으로 정확한 답변을 제공합니다.<br>
                📚 현재 <strong>{}</strong>개의 문서 조각이 벡터 데이터베이스에 저장되어 있습니다.
            </p>
        </div>
        """.format("규제 문서"), unsafe_allow_html=True)

        if "messages" not in st.session_state:
            st.session_state.messages = [{"role": "assistant", "content": "안녕하세요! 🏛️ FSEC AI 금융 보안 규제 챗봇입니다.\n\n궁금한 금융 규제 사항이 있으시면 언제든 질문해주세요! 💼"}]

        # 채팅 인터페이스
        st.markdown('<div style="margin: 1rem 0;"></div>', unsafe_allow_html=True)
        
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                if message["role"] == "assistant":
                    st.markdown(f"""
                    <div class="result-box">
                        {message["content"]}
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(message["content"])
                    
                if "source" in message and message["source"]:
                    with st.expander("📚 답변 근거 확인하기"):
                        st.markdown(message["source"])

        if prompt := st.chat_input("💬 금융 규제에 대해 질문해주세요... (예: 개인정보보호 가이드라인은?)"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            with st.chat_message("assistant"):
                with st.spinner("🔍 지능형 라우팅으로 최적 분석 중..."):
                    try:
                        # 지능형 라우팅으로 질문 분류
                        route = intelligent_router(prompt) if intelligent_router else "qa_chatbot"
                        
                        if route == "multi_agent" and multi_agent_system:
                            # 멀티에이전트 분석
                            st.info("🚀 복합 분석이 감지되어 AI 멀티에이전트 시스템으로 전환합니다.")
                            analysis_result = multi_agent_system.analyze_document(prompt)
                            answer = analysis_result.get("final_report", "멀티에이전트 분석을 완료했습니다.")
                        else:
                            # 기본 RAG 체인 사용
                            response = rag_chain.invoke({"input": prompt})
                            answer = response.get('answer', '답변을 생성하지 못했습니다.')
                        
                        st.markdown(f"""
                        <div class="result-box">
                            {answer}
                        </div>
                        """, unsafe_allow_html=True)
                        source_docs = ""
                        if route != "multi_agent" and 'response' in locals() and response.get("context"):
                            for i, doc in enumerate(response["context"]):
                                source_name = os.path.basename(doc.metadata.get('source', ''))
                                page = doc.metadata.get('page', '')
                                snippet = doc.page_content[:200].replace('\n', ' ')
                                source_docs += f"**[출처 {i+1}] {source_name} (Page: {page})**\n"
                                source_docs += f"> {snippet}...\n\n"
                        
                        if source_docs:
                            with st.expander("📚 답변 근거 및 출처 확인하기"):
                                st.markdown(source_docs)
                        elif route == "multi_agent":
                            with st.expander("🤖 멀티에이전트 분석 세부사항"):
                                if 'analysis_result' in locals():
                                    st.json({
                                        "문서분류": analysis_result.get("document_type", ""),
                                        "분석단계": analysis_result.get("current_step", ""),
                                        "위험평가": analysis_result.get("risk_assessment", {}),
                                        "준수점수": analysis_result.get("compliance_score", {})
                                    })
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": answer,
                            "source": source_docs if source_docs else ""
                        })
                    except Exception as e:
                        st.error(f"답변 생성 중 오류 발생: {e}")

    elif page == "🔒 보안 적정성 평가":
        st.markdown("""
        <div class="section-header">
            🔒 금융 상품/서비스 보안 적정성 평가
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-card">
            <p style="margin: 0; color: #64748b;">
                🛡️ 금융 상품 및 서비스의 보안 적정성을 종합 평가합니다.<br>
                📋 인증, 암호화, 개인정보보호, 로그관리 등 주요 보안 항목을 체크합니다.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # 입력 방법 선택
        st.markdown('<div style="margin: 2rem 0;"></div>', unsafe_allow_html=True)
        st.markdown("### 📝 평가 대상 입력")
        input_method = st.radio(
            "입력 방법을 선택하세요:",
            ["PDF 파일 업로드", "텍스트 직접 입력"],
            horizontal=True
        )
        
        uploaded_file = None
        input_text = ""
        
        if input_method == "PDF 파일 업로드":
            uploaded_file = st.file_uploader(
                "📄 보안성 평가할 PDF 파일을 업로드하세요",
                type=["pdf"],
                help="금융 상품 설명서, 서비스 약관, 보안 정책 등을 업로드하세요"
            )
        else:
            input_text = st.text_area(
                "✏️ 보안성 평가할 텍스트를 직접 입력하세요",
                height=200,
                placeholder="금융 상품이나 서비스의 보안 관련 내용을 입력하세요...",
                help="상품 설명, 보안 정책, 개인정보 처리 방침 등"
            )

        if uploaded_file or input_text.strip():
            with st.spinner("🔍 보안 적정성 평가 중입니다..."):
                try:
                    if uploaded_file:
                        assessment, assessment_sources = security_assessment_content(
                            None, embedding_model, is_pdf=True, uploaded_file=uploaded_file
                        )
                    else:
                        assessment, assessment_sources = security_assessment_content(
                            input_text.strip(), embedding_model, is_pdf=False
                        )
                    
                    st.markdown("""
                    <div class="section-header">
                        ⚖️ 보안 적정성 평가 결과
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown(f"""
                    <div class="result-box">
                        {assessment}
                    </div>
                    """, unsafe_allow_html=True)
                    if assessment_sources:
                        with st.expander("📚 평가 근거 및 참조 문서"):
                            st.markdown(assessment_sources)
                        
                        # Tavily 검색 결과가 있다면 추가 정보 표시
                        if TAVILY_API_KEY:
                            with st.expander("🌐 웹에서 검색된 최신 규제 정보"):
                                if uploaded_file:
                                    # PDF 내용에서 키워드 추출해서 검색
                                    temp_dir = "temp"
                                    os.makedirs(temp_dir, exist_ok=True)
                                    temp_path = os.path.join(temp_dir, uploaded_file.name)
                                    with open(temp_path, "wb") as f:
                                        f.write(uploaded_file.getbuffer())
                                    loader = PyPDFLoader(temp_path)
                                    docs = loader.load()
                                    if docs:
                                        search_keywords = docs[0].page_content[:200]
                                        web_info = search_additional_info(search_keywords, TAVILY_API_KEY)
                                        st.markdown(web_info)
                                else:
                                    web_info = search_additional_info(input_text.strip()[:200], TAVILY_API_KEY)
                                    st.markdown(web_info)
                except Exception as e:
                    st.error(f"평가 생성 중 오류 발생: {e}")
        else:
            st.markdown("""
            <div class="info-card">
                <h4 style="color: #1e40af; margin-top: 0;">📤 평가 시작하기</h4>
                <p style="margin: 0; color: #64748b;">
                    위에서 입력 방법을 선택하고 평가할 내용을 입력해주세요.
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            # 평가 가능한 항목 안내
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("""
                <div class="info-card">
                    <h4 style="color: #16a34a; margin-top: 0;">📋 평가 항목</h4>
                    <ul style="margin: 0; padding-left: 1rem; color: #374151;">
                        <li>🔐 인증 및 접근제어</li>
                        <li>🛡️ 데이터 보안 및 암호화</li>
                        <li>👤 개인정보보호</li>
                        <li>📊 로그 관리 및 모니터링</li>
                        <li>🏢 외부위탁 관리</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            with col2:
                st.markdown("""
                <div class="info-card">
                    <h4 style="color: #dc2626; margin-top: 0;">📄 추천 업로드 파일</h4>
                    <ul style="margin: 0; padding-left: 1rem; color: #374151;">
                        <li>금융상품 설명서</li>
                        <li>서비스 이용약관</li>
                        <li>개인정보 처리방침</li>
                        <li>보안정책 문서</li>
                        <li>시스템 구성도</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)

    elif page == "🚀 AI 멀티에이전트 분석":
        st.markdown("""
        <div class="section-header">
            🚀 AI 멀티에이전트 분석 시스템
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-card">
            <p style="margin: 0; color: #64748b;">
                🤖 6개의 전문 AI 에이전트가 협업하여 종합적인 규제 분석을 수행합니다.<br>
                📊 문서 분류 → 1차 분석 → 위험도 평가 → 웹 검색 → 준수 점수 → 최종 보고서
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        if multi_agent_system:
            # 분석 방법 선택
            st.markdown("### 🔍 분석 방법 선택")
            analysis_method = st.radio(
                "분석할 방법을 선택하세요:",
                ["📄 텍스트 직접 입력", "📋 PDF 문서 업로드"],
                horizontal=True
            )
            
            content_to_analyze = ""
            
            if analysis_method == "📄 텍스트 직접 입력":
                content_to_analyze = st.text_area(
                    "✏️ 분석할 내용을 입력하세요",
                    height=200,
                    placeholder="금융 상품 설명, 보안 정책, 개인정보 처리방침 등을 입력하세요...",
                    help="상세할수록 더 정확한 분석이 가능합니다"
                )
            else:
                uploaded_file = st.file_uploader(
                    "📄 분석할 PDF 파일을 업로드하세요",
                    type=["pdf"],
                    help="금융 상품 설명서, 보안 정책, 개인정보 처리방침 등"
                )
                
                if uploaded_file:
                    # PDF 내용 추출
                    temp_dir = "temp"
                    os.makedirs(temp_dir, exist_ok=True)
                    temp_path = os.path.join(temp_dir, uploaded_file.name)
                    with open(temp_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    from langchain_community.document_loaders import PyPDFLoader
                    loader = PyPDFLoader(temp_path)
                    documents = loader.load()
                    content_to_analyze = "\n".join([doc.page_content for doc in documents[:5]])  # 처음 5페이지만
            
            if content_to_analyze.strip():
                if st.button("🚀 멀티에이전트 분석 시작", type="primary"):
                    with st.spinner("🤖 AI 에이전트들이 협업하여 분석 중입니다..."):
                        
                        # 진행 상태 표시
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        step_info = st.empty()
                        
                        # 에이전트 단계별 진행 표시
                        steps = [
                            "🔍 문서 분류 에이전트 작동 중...",
                            "📊 1차 분석 에이전트 분석 중...",
                            "⚠️ 위험도 평가 에이전트 평가 중...",
                            "🌐 웹 검색 에이전트 정보 수집 중...",
                            "📈 점수 계산 에이전트 점수 산출 중...",
                            "📝 보고서 생성 에이전트 최종 보고서 작성 중..."
                        ]
                        
                        # 분석 실행
                        import time
                        analysis_result = None
                        
                        for i, step in enumerate(steps):
                            progress = int((i + 1) / len(steps) * 85)  # 85%까지만
                            progress_bar.progress(progress)
                            step_info.info(step)
                            time.sleep(0.3)  # 시각적 효과
                        
                        # 실제 분석 실행
                        analysis_result = multi_agent_system.analyze_document(content_to_analyze)
                        
                        # 완료 처리
                        progress_bar.progress(100)
                        step_info.success("🎉 6개 AI 에이전트 협업 분석 완료!")
                        status_text.success("✅ 종합 분석 보고서가 생성되었습니다!")
                        
                        # 완료 효과
                        time.sleep(0.5)
                        progress_bar.empty()
                        step_info.empty()
                        
                        # 결과 표시
                        if analysis_result.get("error_message"):
                            st.error(f"⚠️ 분석 중 오류 발생: {analysis_result['error_message']}")
                        else:
                            # 최종 보고서
                            st.markdown("""
                            <div class="section-header">
                                📋 종합 분석 보고서
                            </div>
                            """, unsafe_allow_html=True)
                            
                            st.markdown(f"""
                            <div class="result-box">
                                {analysis_result.get('final_report', '보고서 생성 실패')}
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # 전체 점수 하이라이트
                            if analysis_result.get("compliance_score", {}).get("전체점수"):
                                overall_data = analysis_result["compliance_score"]["전체점수"]
                                overall_score = overall_data.get("점수", 0)
                                overall_grade = overall_data.get("등급", "미평가")
                                
                                # 점수에 따른 색상 설정
                                if overall_score >= 90:
                                    score_color = "#16a34a"  # 녹색
                                    bg_color = "#f0fdf4"
                                elif overall_score >= 80:
                                    score_color = "#3b82f6"  # 파란색
                                    bg_color = "#eff6ff"
                                elif overall_score >= 70:
                                    score_color = "#f59e0b"  # 주황색
                                    bg_color = "#fffbeb"
                                else:
                                    score_color = "#dc2626"  # 빨간색
                                    bg_color = "#fef2f2"
                                
                                st.markdown(f"""
                                <div style="
                                    background: linear-gradient(135deg, {bg_color} 0%, white 100%);
                                    border: 2px solid {score_color};
                                    border-radius: 15px;
                                    padding: 2rem;
                                    text-align: center;
                                    margin: 2rem 0;
                                    box-shadow: 0 8px 25px rgba(0,0,0,0.1);
                                ">
                                    <h1 style="color: {score_color}; margin: 0; font-size: 3rem; font-weight: 800;">
                                        {overall_score}점
                                    </h1>
                                    <h3 style="color: {score_color}; margin: 0.5rem 0; font-size: 1.5rem;">
                                        등급: {overall_grade}
                                    </h3>
                                    <p style="color: #64748b; margin: 0; font-size: 1.1rem;">
                                        전체 규제 준수 점수
                                    </p>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            # 카테고리별 점수 카드
                            if analysis_result.get("compliance_score"):
                                st.markdown("""
                                <div class="section-header" style="margin-top: 2rem;">
                                    📊 카테고리별 준수 점수
                                </div>
                                """, unsafe_allow_html=True)
                                
                                # 점수 데이터 준비
                                score_cards = []
                                for category, data in analysis_result["compliance_score"].items():
                                    if category != "전체점수" and isinstance(data, dict) and "점수" in data:
                                        score_cards.append({
                                            "category": category,
                                            "score": data["점수"],
                                            "grade": data.get("등급", ""),
                                            "reason": data.get("사유", "")
                                        })
                                
                                # 2x2 그리드로 카드 배치
                                if len(score_cards) >= 4:
                                    cols = st.columns(2)
                                    for i, card in enumerate(score_cards):
                                        with cols[i % 2]:
                                            _render_score_card(card)
                                else:
                                    cols = st.columns(len(score_cards))
                                    for i, card in enumerate(score_cards):
                                        with cols[i]:
                                            _render_score_card(card)
                            
                            # 점수 차트 표시 (개선된 디자인)
                            if analysis_result.get("compliance_score"):
                                st.markdown("""
                                <div class="section-header" style="margin-top: 2rem;">
                                    📈 준수 점수 시각화
                                </div>
                                """, unsafe_allow_html=True)
                                
                                # 차트 탭 생성
                                tab1, tab2 = st.tabs(["📊 막대 차트", "🎯 레이더 차트"])
                                
                                with tab1:
                                    try:
                                        chart = multi_agent_system.create_score_chart(analysis_result["compliance_score"])
                                        st.plotly_chart(chart, use_container_width=True)
                                    except Exception as e:
                                        st.warning(f"막대 차트 생성 중 오류: {str(e)}")
                                
                                with tab2:
                                    try:
                                        radar_chart = multi_agent_system.create_radar_chart(analysis_result["compliance_score"])
                                        st.plotly_chart(radar_chart, use_container_width=True)
                                    except Exception as e:
                                        st.warning(f"레이더 차트 생성 중 오류: {str(e)}")
                            
                            # 문서 분류 및 분석 요약 (간단히)
                            col1, col2 = st.columns(2)
                            with col1:
                                st.markdown("""
                                <div class="info-card">
                                    <h4 style="color: #1e40af; margin-top: 0;">📑 분석 정보</h4>
                                    <p style="margin: 0; color: #374151;">
                                        <strong>문서 유형:</strong> {}<br>
                                        <strong>분석 일시:</strong> {}<br>
                                        <strong>분석 에이전트:</strong> 6개 AI 협업
                                    </p>
                                </div>
                                """.format(
                                    analysis_result.get('document_type', 'Unknown'),
                                    pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")
                                ), unsafe_allow_html=True)
                            
                            with col2:
                                # 주요 발견사항 요약
                                primary_analysis = analysis_result.get("primary_analysis", {})
                                risk_count = len(primary_analysis.get("위험요소", []))
                                regulation_count = len(primary_analysis.get("규제관련사항", []))
                                
                                st.markdown(f"""
                                <div class="info-card">
                                    <h4 style="color: #dc2626; margin-top: 0;">🔍 발견사항 요약</h4>
                                    <p style="margin: 0; color: #374151;">
                                        <strong>위험요소:</strong> {risk_count}개 발견<br>
                                        <strong>규제사항:</strong> {regulation_count}개 확인<br>
                                        <strong>웹검색:</strong> 최신 정보 반영
                                    </p>
                                </div>
                                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="info-card">
                    <h4 style="color: #1e40af; margin-top: 0;">🚀 멀티에이전트 분석 시작하기</h4>
                    <p style="margin: 0; color: #64748b;">
                        위에서 분석 방법을 선택하고 내용을 입력한 후 분석을 시작하세요.
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
                # 시스템 아키텍처 설명
                st.markdown("### 🔧 시스템 아키텍처")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("""
                    <div class="info-card">
                        <h4 style="color: #16a34a; margin-top: 0;">🤖 AI 에이전트</h4>
                        <ul style="margin: 0; padding-left: 1rem; color: #374151;">
                            <li>📋 문서 분류 에이전트</li>
                            <li>🔍 1차 분석 에이전트</li>
                            <li>⚠️ 위험도 평가 에이전트</li>
                            <li>🌐 웹 검색 에이전트</li>
                            <li>📊 점수 계산 에이전트</li>
                            <li>📝 보고서 생성 에이전트</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown("""
                    <div class="info-card">
                        <h4 style="color: #dc2626; margin-top: 0;">📊 분석 항목</h4>
                        <ul style="margin: 0; padding-left: 1rem; color: #374151;">
                            <li>개인정보보호</li>
                            <li>데이터보안</li>
                            <li>접근제어</li>
                            <li>규제준수</li>
                            <li>전체위험도</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    st.markdown("""
                    <div class="info-card">
                        <h4 style="color: #f59e0b; margin-top: 0;">📈 결과물</h4>
                        <ul style="margin: 0; padding-left: 1rem; color: #374151;">
                            <li>종합 분석 보고서</li>
                            <li>점수 시각화 차트</li>
                            <li>위험도 평가 결과</li>
                            <li>개선 권장사항</li>
                            <li>최신 규제 정보</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.error("멀티에이전트 시스템이 초기화되지 않았습니다. API 키를 확인해주세요.")

else:
    st.markdown("""
    <div class="section-header">
        ⚠️ API 키 설정이 필요합니다
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-card">
        <h4 style="color: #dc2626; margin-top: 0;">🔧 설정 방법</h4>
        <ol style="color: #374151; margin: 0;">
            <li>프로젝트 폴더에 <code>.env</code> 파일을 생성하세요</li>
            <li>파일에 다음과 같이 작성하세요:</li>
        </ol>
        <div style="background: #f1f5f9; padding: 1rem; border-radius: 8px; margin: 1rem 0; font-family: monospace;">
            OPENAI_API_KEY=your_api_key_here<br>
            TAVILY_API_KEY=your_tavily_api_key_here  # 웹 검색 기능용 (선택)
        </div>
        <ol start="3" style="color: #374151; margin: 0;">
            <li>앱을 다시 시작하세요</li>
        </ol>
        
        <div style="margin-top: 1rem; padding: 1rem; background: #fef3c7; border-radius: 8px; border-left: 4px solid #f59e0b;">
            <p style="margin: 0; color: #92400e;">
                💡 <strong>참고:</strong> Tavily API는 선택사항입니다. OPENAI_API_KEY만 있어도 기본 기능은 모두 사용 가능합니다.
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)
"""
멀티에이전트 규제 분석 시스템
LangGraph를 사용한 고도화된 분석 워크플로우
"""

import os
from typing import Dict, List, Any, TypedDict
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
import json
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime


class AnalysisState(TypedDict):
    """분석 상태를 관리하는 클래스"""
    input_content: str
    document_type: str
    primary_analysis: Dict[str, Any]
    risk_assessment: Dict[str, Any]
    web_search_results: Dict[str, Any]
    compliance_score: Dict[str, Any]
    final_report: str
    current_step: str
    error_message: str


class MultiAgentAnalysisSystem:
    """멀티에이전트 분석 시스템"""
    
    def __init__(self, openai_api_key: str, tavily_api_key: str = None):
        self.openai_api_key = openai_api_key
        self.tavily_api_key = tavily_api_key
        self.llm = ChatOpenAI(
            openai_api_key=openai_api_key,
            model_name="gpt-4o",
            temperature=0
        )
        self.workflow = self._create_workflow()
    
    def _create_workflow(self) -> StateGraph:
        """워크플로우 생성"""
        workflow = StateGraph(AnalysisState)
        
        # 노드 추가
        workflow.add_node("document_classifier", self._classify_document)
        workflow.add_node("primary_analyzer", self._primary_analysis)
        workflow.add_node("risk_assessor", self._assess_risk)
        workflow.add_node("web_searcher", self._search_web_info)
        workflow.add_node("compliance_scorer", self._calculate_compliance_score)
        workflow.add_node("report_generator", self._generate_final_report)
        
        # 워크플로우 정의
        workflow.set_entry_point("document_classifier")
        workflow.add_edge("document_classifier", "primary_analyzer")
        workflow.add_edge("primary_analyzer", "risk_assessor")
        workflow.add_edge("risk_assessor", "web_searcher")
        workflow.add_edge("web_searcher", "compliance_scorer")
        workflow.add_edge("compliance_scorer", "report_generator")
        workflow.add_edge("report_generator", END)
        
        return workflow.compile()
    
    def _classify_document(self, state: AnalysisState) -> AnalysisState:
        """문서 분류 에이전트 - 고도화된 프롬프트"""
        try:
            prompt = ChatPromptTemplate.from_template("""
            당신은 금융 규제 전문가로서 20년간 금융감독원에서 근무한 경험이 있습니다.
            다음 문서를 정확히 분류하고, 분류 근거를 함께 제시해주세요.
            
            === 분석 대상 문서 ===
            {content}
            
            === 문서 분류 기준 ===
            1. 금융상품설명서
               - 투자위험, 수익구조, 상품특성 설명 포함
               - 금융투자업법, 자본시장법 관련 용어 사용
               - 투자자 보호, 투자권유, 적합성 원칙 언급
            
            2. 서비스약관
               - 서비스 이용조건, 권리의무, 책임한계 규정
               - 약관의 변경, 해지, 분쟁해결 절차 포함
               - 소비자보호법, 전자상거래법 관련 내용
            
            3. 개인정보처리방침
               - 개인정보 수집/이용/제공/파기 절차
               - 개인정보보호법, 신용정보법 준수사항
               - 정보주체 권리, 동의철회, 손해배상 명시
            
            4. 보안정책
               - 정보보안 관리체계, 접근통제, 암호화
               - 보안사고 대응, 취약점 관리, 보안교육
               - 정보보호관리체계(ISMS), ISO27001 관련
            
            5. 시스템구성도
               - 시스템 아키텍처, 네트워크 구성
               - 서버/DB 구조, 보안장비 배치
               - 기술적 보안조치, 인프라 설명
            
            6. 기타
               - 위 카테고리에 해당하지 않는 문서
            
            === 출력 형식 ===
            분류번호: [1-6]
            분류명: [해당 문서 유형]
            신뢰도: [1-10점]
            근거: [분류한 주요 근거 3가지]
            
            반드시 이 형식으로만 답변하세요.
            """)
            
            response = self.llm.invoke(
                prompt.format_messages(content=state["input_content"][:2000])
            )
            
            # 고도화된 응답 파싱
            response_text = response.content.strip()
            lines = response_text.split('\n')
            
            doc_type = "기타"
            confidence = 5
            
            for line in lines:
                if "분류번호:" in line:
                    try:
                        num = line.split(":")[1].strip()
                        doc_types = {
                            "1": "금융상품설명서",
                            "2": "서비스약관", 
                            "3": "개인정보처리방침",
                            "4": "보안정책",
                            "5": "시스템구성도",
                            "6": "기타"
                        }
                        doc_type = doc_types.get(num, "기타")
                    except:
                        pass
                elif "신뢰도:" in line:
                    try:
                        confidence = int(line.split(":")[1].strip().split("점")[0])
                    except:
                        pass
            
            state["document_type"] = doc_type
            state["current_step"] = f"문서 분류 완료 (신뢰도: {confidence}/10)"
            
        except Exception as e:
            state["error_message"] = f"문서 분류 오류: {str(e)}"
            state["document_type"] = "기타"
        
        return state
    
    def _primary_analysis(self, state: AnalysisState) -> AnalysisState:
        """1차 분석 에이전트 - 전문가 수준 분석"""
        try:
            prompt = ChatPromptTemplate.from_template("""
            당신은 금융감독원 검사국에서 15년간 근무한 시니어 금융검사관입니다.
            {doc_type} 문서에 대해 금융 규제 관점에서 심층 분석을 수행하세요.
            
            === 분석 대상 문서 ===
            {content}
            
            === 분석 가이드라인 ===
            
            1. 주요내용 분석 시:
               - 문서의 핵심 목적과 적용범위 명확히 파악
               - 이해관계자(고객, 기업, 규제기관) 관점에서 중요도 평가
               - 비즈니스 임팩트와 규제 리스크 동시 고려
            
            2. 규제관련사항 식별 시:
               - 금융위원회 고시, 금감원 규정 위반 가능성
               - 금융소비자보호법, 개인정보보호법, 신용정보법 준수사항
               - 전자금융거래법, 자본시장법 관련 의무사항
               - 국제 규제(바젤III, GDPR 등) 영향도
            
            3. 보안요소 검토 시:
               - 정보보호관리체계(ISMS-P) 인증 요구사항
               - 암호화, 접근통제, 로그관리 등 기술적 조치
               - 물리적/관리적 보안조치 적정성
               - 보안사고 대응체계 구축 현황
            
            4. 개인정보 처리 검토 시:
               - 수집/이용/제공/파기 각 단계별 적법성
               - 동의 획득 절차와 고지사항 충족성
               - 개인정보 영향평가 대상 여부 판단
               - 정보주체 권리 보장 메커니즘
            
            5. 위험요소 평가 시:
               - 규제 위반으로 인한 제재 위험 (과태료, 영업정지 등)
               - 평판 리스크와 고객 신뢰도 손상 가능성
               - 시스템 장애나 보안사고 발생 시 파급효과
               - 경쟁사 대비 컴플라이언스 수준 격차
            
            === 출력 형식 (JSON) ===
            {{
                "주요내용": {{
                    "목적": "문서의 핵심 목적",
                    "적용범위": "적용 대상과 범위",
                    "핵심조항": ["중요한 조항 3-5개"]
                }},
                "규제관련사항": {{
                    "준수법령": ["관련 법령명"],
                    "규제요구사항": ["구체적 요구사항"],
                    "컴플라이언스이슈": ["발견된 이슈"],
                    "개선필요사항": ["개선이 필요한 부분"]
                }},
                "보안요소": {{
                    "기술적조치": ["암호화, 접근통제 등"],
                    "관리적조치": ["정책, 절차, 교육 등"],
                    "물리적조치": ["시설보안, 출입통제 등"],
                    "보안수준평가": "상/중/하"
                }},
                "개인정보": {{
                    "처리현황": ["수집/이용/제공/파기 현황"],
                    "법적근거": ["처리 법적 근거"],
                    "권리보장": ["정보주체 권리 보장 현황"],
                    "위험도": "상/중/하"
                }},
                "위험요소": {{
                    "규제위험": ["규제 위반 가능성"],
                    "운영위험": ["시스템/프로세스 리스크"],
                    "평판위험": ["이미지 손상 요소"],
                    "우선순위": ["즉시해결/단기개선/중장기과제"]
                }}
            }}
            
            반드시 JSON 형식으로만 답변하고, 각 항목은 구체적이고 실무적으로 작성하세요.
            """)
            
            response = self.llm.invoke(
                prompt.format_messages(
                    doc_type=state["document_type"],
                    content=state["input_content"][:2000]
                )
            )
            
            try:
                analysis_result = json.loads(response.content)
            except:
                # JSON 파싱 실패 시 기본 구조 제공
                analysis_result = {
                    "주요내용": response.content[:500],
                    "규제관련사항": ["분석 중 오류 발생"],
                    "보안요소": ["분석 중 오류 발생"],
                    "개인정보": ["분석 중 오류 발생"],
                    "위험요소": ["분석 중 오류 발생"]
                }
            
            state["primary_analysis"] = analysis_result
            state["current_step"] = "1차 분석 완료"
            
        except Exception as e:
            state["error_message"] = f"1차 분석 오류: {str(e)}"
            state["primary_analysis"] = {"오류": "분석 실패"}
        
        return state
    
    def _assess_risk(self, state: AnalysisState) -> AnalysisState:
        """위험도 평가 에이전트 - 정량적 리스크 모델링"""
        try:
            prompt = ChatPromptTemplate.from_template("""
            당신은 Big4 회계법인의 리스크 어드바이저리 파트너로서 10년간 금융회사 리스크 관리를 전담했습니다.
            다음 분석 결과를 바탕으로 정량적 위험도 평가를 실시하세요.
            
            === 평가 대상 ===
            문서 유형: {doc_type}
            1차 분석 결과: {analysis}
            
            === 위험도 평가 방법론 ===
            
            각 영역별로 다음 기준에 따라 1-10점으로 평가하세요:
            - 1-2점: 모범사례 수준 (업계 상위 10%)
            - 3-4점: 우수 수준 (규제 요구사항 완벽 충족)
            - 5-6점: 적정 수준 (기본 요구사항 충족)
            - 7-8점: 미흡 수준 (일부 개선 필요)
            - 9-10점: 위험 수준 (즉시 조치 필요)
            
            === 평가 영역별 세부 기준 ===
            
            1. 개인정보보호 (GDPR, 개인정보보호법 기준)
               - 수집/이용 목적의 명확성과 최소수집 원칙 준수
               - 동의 획득 절차의 적법성 (명시적/선택적 동의)
               - 개인정보 처리위탁 관리 체계
               - 정보주체 권리 행사 절차 구비
               - 개인정보 유출 시 대응체계
               평가항목: 법적 근거, 동의 체계, 처리 위탁, 권리 보장, 사고 대응
            
            2. 데이터보안 (ISMS-P, ISO27001 기준)
               - 암호화 적용 범위와 강도 (전송/저장)
               - 데이터 분류 체계와 보호 조치
               - 백업 및 복구 체계
               - 데이터 생명주기 관리
               - 클라우드/외부 보관 시 보안 조치
               평가항목: 암호화, 분류 체계, 백업/복구, 생명주기, 외부 보관
            
            3. 접근제어 (최소권한 원칙)
               - 사용자 인증 체계 (다중 인증 포함)
               - 권한 부여 및 관리 절차
               - 관리자 계정 보안 조치
               - 접근 로그 모니터링 체계
               - 권한 정기 검토 프로세스
               평가항목: 인증 체계, 권한 관리, 관리자 보안, 로그 관리, 정기 검토
            
            4. 규제준수 (금융 규제 전반)
               - 관련 법령 식별 완성도
               - 규제 요구사항 이행 수준
               - 내부 통제 체계 구축
               - 컴플라이언스 모니터링 체계
               - 규제 변화 대응 체계
               평가항목: 법령 준수, 요구사항 이행, 내부 통제, 모니터링, 변화 대응
            
            === 출력 형식 (JSON) ===
            {{
                "개인정보보호": {{
                    "점수": [1-10],
                    "등급": "모범/우수/적정/미흡/위험",
                    "사유": "구체적 평가 근거 (법령 조항 포함)",
                    "주요이슈": ["발견된 주요 이슈 2-3개"],
                    "개선방안": ["즉시 개선 방안 2-3개"]
                }},
                "데이터보안": {{
                    "점수": [1-10],
                    "등급": "모범/우수/적정/미흡/위험",
                    "사유": "구체적 평가 근거 (보안 표준 포함)",
                    "주요이슈": ["발견된 주요 이슈 2-3개"],
                    "개선방안": ["즉시 개선 방안 2-3개"]
                }},
                "접근제어": {{
                    "점수": [1-10],
                    "등급": "모범/우수/적정/미흡/위험",
                    "사유": "구체적 평가 근거",
                    "주요이슈": ["발견된 주요 이슈 2-3개"],
                    "개선방안": ["즉시 개선 방안 2-3개"]
                }},
                "규제준수": {{
                    "점수": [1-10],
                    "등급": "모범/우수/적정/미흡/위험",
                    "사유": "구체적 평가 근거 (관련 법령 명시)",
                    "주요이슈": ["발견된 주요 이슈 2-3개"],
                    "개선방안": ["즉시 개선 방안 2-3개"]
                }},
                "전체위험도": {{
                    "점수": [1-10],
                    "등급": "모범/우수/적정/미흡/위험",
                    "종합의견": "전체적인 위험 수준에 대한 종합 의견",
                    "우선개선과제": ["가장 시급한 개선 과제 3개"],
                    "예상제재": ["위험도별 예상 제재 수준"]
                }}
            }}
            
            반드시 JSON 형식으로 답변하고, 모든 평가는 구체적 근거와 함께 제시하세요.
            """)
            
            response = self.llm.invoke(
                prompt.format_messages(
                    doc_type=state["document_type"],
                    analysis=str(state["primary_analysis"])
                )
            )
            
            try:
                risk_result = json.loads(response.content)
            except:
                # 기본 위험도 설정
                risk_result = {
                    "개인정보보호": {"점수": 5, "사유": "분석 중 오류"},
                    "데이터보안": {"점수": 5, "사유": "분석 중 오류"},
                    "접근제어": {"점수": 5, "사유": "분석 중 오류"},
                    "규제준수": {"점수": 5, "사유": "분석 중 오류"},
                    "전체위험도": {"점수": 5, "등급": "보통"}
                }
            
            state["risk_assessment"] = risk_result
            state["current_step"] = "위험도 평가 완료"
            
        except Exception as e:
            state["error_message"] = f"위험도 평가 오류: {str(e)}"
            state["risk_assessment"] = {"오류": "평가 실패"}
        
        return state
    
    def _search_web_info(self, state: AnalysisState) -> AnalysisState:
        """웹 검색 에이전트"""
        try:
            if not self.tavily_api_key:
                state["web_search_results"] = {
                    "결과": "웹 검색 기능이 비활성화됨",
                    "관련규제": ["Tavily API 키 필요"]
                }
                state["current_step"] = "웹 검색 건너뜀"
                return state
            
            from tavily import TavilyClient
            tavily = TavilyClient(api_key=self.tavily_api_key)
            
            # 검색 쿼리 생성
            doc_type = state["document_type"]
            search_query = f"금융 규제 {doc_type} 가이드라인 2024 금융위원회 금융보안원"
            
            search_result = tavily.search(
                query=search_query,
                search_depth="basic",
                max_results=3,
                include_domains=["fss.or.kr", "fsc.go.kr", "fsec.or.kr"]
            )
            
            web_info = {
                "검색쿼리": search_query,
                "결과수": len(search_result.get("results", [])),
                "관련규제": []
            }
            
            for result in search_result.get("results", [])[:3]:
                web_info["관련규제"].append({
                    "제목": result.get("title", ""),
                    "URL": result.get("url", ""),
                    "내용": result.get("content", "")[:200]
                })
            
            state["web_search_results"] = web_info
            state["current_step"] = "웹 검색 완료"
            
        except Exception as e:
            state["error_message"] = f"웹 검색 오류: {str(e)}"
            state["web_search_results"] = {"오류": "검색 실패"}
        
        return state
    
    def _calculate_compliance_score(self, state: AnalysisState) -> AnalysisState:
        """규제 준수 점수 계산 에이전트"""
        try:
            # 위험도 점수를 준수 점수로 변환 (위험도가 낮을수록 준수도가 높음)
            risk_scores = state.get("risk_assessment", {})
            
            compliance_scores = {}
            total_score = 0
            count = 0
            
            for category, data in risk_scores.items():
                if category != "전체위험도" and isinstance(data, dict) and "점수" in data:
                    # 위험도 점수(1-10)를 준수 점수(1-100)로 변환
                    risk_score = data["점수"]
                    compliance_score = max(10, 110 - (risk_score * 10))
                    compliance_scores[category] = {
                        "점수": compliance_score,
                        "등급": self._get_grade(compliance_score),
                        "사유": data.get("사유", "")
                    }
                    total_score += compliance_score
                    count += 1
            
            # 전체 점수 계산
            if count > 0:
                overall_score = total_score / count
                compliance_scores["전체점수"] = {
                    "점수": round(overall_score, 1),
                    "등급": self._get_grade(overall_score),
                    "백분율": f"{round(overall_score)}%"
                }
            
            state["compliance_score"] = compliance_scores
            state["current_step"] = "준수 점수 계산 완료"
            
        except Exception as e:
            state["error_message"] = f"점수 계산 오류: {str(e)}"
            state["compliance_score"] = {"오류": "계산 실패"}
        
        return state
    
    def _get_grade(self, score: float) -> str:
        """점수를 등급으로 변환"""
        if score >= 90:
            return "우수"
        elif score >= 80:
            return "양호"
        elif score >= 70:
            return "보통"
        elif score >= 60:
            return "미흡"
        else:
            return "부족"
    
    def _generate_final_report(self, state: AnalysisState) -> AnalysisState:
        """최종 보고서 생성 에이전트 - 금융감독 전문 리포트"""
        try:
            prompt = ChatPromptTemplate.from_template("""
            당신은 금융감독원 전자검사팀에서 10년간 근무한 수석 검사관입니다.
            다음 분석 결과를 바탕으로 임원진과 이사회에 제출할 수준의 전문적인 종합 보고서를 작성하세요.
            
            === 분석 데이터 ===
            문서 유형: {doc_type}
            1차 분석: {primary_analysis}
            위험도 평가: {risk_assessment}
            웹 검색 결과: {web_search}
            준수 점수: {compliance_score}
            
            === 보고서 작성 지침 ===
            
            1. 임원진 관점 고려사항:
               - 규제 위반 시 발생 가능한 과태료 및 제재 조치
               - 기업 이미지와 고객 신뢰도에 미치는 영향
               - 경쟁사 대비 컴플라이언스 수준과 차별화 요소
               - 투자 우선순위와 예산 배정을 위한 구체적 근거
            
            2. 이사회 보고 수준:
               - 정량적 지표 중심의 객관적 평가
               - 법적 리스크의 재무적 영향 분석
               - 단계별 개선 로드맵과 예상 소요 기간
               - 업계 모범사례 및 벤치마킹 결과
            
            3. 실무진 액션플랜:
               - 즉시 조치 가능한 단기 개선과제 (1-3개월)
               - 시스템 개선이 필요한 중기 과제 (3-12개월)
               - 정책/프로세스 고도화 장기 과제 (1-2년)
               - 각 과제별 담당 부서와 예상 비용
            
            === 보고서 구조 (Executive Summary 스타일) ===
            
            ## 🏛️ 규제 준수 종합 분석 보고서
            
            ### 📋 Executive Summary
            **문서명**: {doc_type} 규제 준수 분석
            **분석일시**: [현재 시간]
            **전체 준수 점수**: [점수]/100점 ([등급])
            **종합 의견**: [전체적인 준수 수준에 대한 한 줄 요약]
            
            ### 🎯 주요 발견사항 (Key Findings)
            **1. 핵심 준수 이슈**
            - [가장 중요한 컴플라이언스 이슈 2-3개]
            
            **2. 규제 위험도 평가**
            - 개인정보보호: [점수]점 ([등급]) - [주요 이슈]
            - 데이터보안: [점수]점 ([등급]) - [주요 이슈]
            - 접근제어: [점수]점 ([등급]) - [주요 이슈]
            - 규제준수: [점수]점 ([등급]) - [주요 이슈]
            
            **3. 규제 환경 변화**
            - [최신 규제 동향과 우리 기업에 미치는 영향]
            
            ### ⚠️ 위험 요소 및 영향 분석 (Risk Assessment)
            **1. 즉시 조치 필요 (Critical)**
            - [9-10점 위험 요소들] → 예상 제재: [구체적 제재 수준]
            
            **2. 단기 개선 필요 (High)**
            - [7-8점 위험 요소들] → 예상 영향: [비즈니스 영향도]
            
            **3. 중장기 관리 필요 (Medium)**
            - [5-6점 위험 요소들] → 관리 방향: [개선 방향성]
            
            ### 📋 Action Plan (실행 계획)
            **1. 즉시 실행 (1-30일)**
            - [ ] [구체적 액션 아이템 1] - 담당: [부서] - 비용: [예상 비용]
            - [ ] [구체적 액션 아이템 2] - 담당: [부서] - 비용: [예상 비용]
            
            **2. 단기 실행 (1-6개월)**
            - [ ] [구체적 액션 아이템 1] - 담당: [부서] - 비용: [예상 비용]
            - [ ] [구체적 액션 아이템 2] - 담당: [부서] - 비용: [예상 비용]
            
            **3. 중장기 실행 (6-24개월)**
            - [ ] [구체적 액션 아이템 1] - 담당: [부서] - 비용: [예상 비용]
            - [ ] [구체적 액션 아이템 2] - 담당: [부서] - 비용: [예상 비용]
            
            ### 💰 투자 우선순위 및 예산 가이드
            **1. 필수 투자 (ROI: 즉시)**
            - [규제 위반 방지를 위한 필수 투자 항목들]
            
            **2. 권장 투자 (ROI: 6-12개월)**
            - [경쟁 우위 확보를 위한 권장 투자 항목들]
            
            **3. 선택 투자 (ROI: 12-24개월)**
            - [미래 대비를 위한 선택적 투자 항목들]
            
            ### 🔄 지속적 모니터링 체계
            **1. 정기 점검 주기**
            - 월간: [월간 점검 항목]
            - 분기: [분기 점검 항목]
            - 연간: [연간 점검 항목]
            
            **2. KPI 및 성과지표**
            - [측정 가능한 성과지표 3-5개]
            
            ### 📚 참고 규제 및 가이드라인
            - [관련 법령 및 규정 목록]
            - [업계 모범사례 및 벤치마크]
            - [최신 규제 동향 정보]
            
            ---
            **보고서 작성**: 금융감독 AI 시스템
            **검토 필요**: 컴플라이언스팀, 법무팀, 정보보호팀
            **승인 라인**: 부서장 → 임원진 → 이사회
            
            === 작성 시 주의사항 ===
            - 모든 위험도 점수는 구체적 수치로 명시
            - 예상 제재나 비용은 현실적 범위로 제시
            - 액션 아이템은 실행 가능한 수준으로 구체화
            - 담당 부서는 일반적인 조직 구조 기준으로 제시
            - 법령명과 조항은 정확히 명시
            - 전문 용어 사용 시 약어 설명 포함
            """)
            
            response = self.llm.invoke(
                prompt.format_messages(
                    doc_type=state["document_type"],
                    primary_analysis=str(state["primary_analysis"]),
                    risk_assessment=str(state["risk_assessment"]),
                    web_search=str(state["web_search_results"]),
                    compliance_score=str(state["compliance_score"])
                )
            )
            
            # 현재 시간과 동적 데이터 치환
            current_time = datetime.now().strftime("%Y년 %m월 %d일 %H:%M:%S")
            final_report = response.content.replace("[현재 시간]", current_time)
            
            # 전체 점수 정보 추가
            if "compliance_score" in state and "전체점수" in state["compliance_score"]:
                total_score_data = state["compliance_score"]["전체점수"]
                score = total_score_data.get("점수", "미측정")
                grade = total_score_data.get("등급", "미평가")
                final_report = final_report.replace("[점수]/100점 ([등급])", f"{score}/100점 ({grade})")
                final_report = final_report.replace("[점수]", str(score))
                final_report = final_report.replace("[등급]", grade)
            
            state["final_report"] = final_report
            state["current_step"] = "전문 보고서 생성 완료"
            
        except Exception as e:
            state["error_message"] = f"보고서 생성 오류: {str(e)}"
            state["final_report"] = f"## ⚠️ 보고서 생성 오류\n\n보고서 생성 중 오류가 발생했습니다: {str(e)}\n\n기본 분석 결과를 확인해주세요."
        
        return state
    
    def analyze_document(self, content: str) -> AnalysisState:
        """문서 분석 실행"""
        initial_state = AnalysisState(
            input_content=content,
            document_type="",
            primary_analysis={},
            risk_assessment={},
            web_search_results={},
            compliance_score={},
            final_report="",
            current_step="시작",
            error_message=""
        )
        
        result = self.workflow.invoke(initial_state)
        return result
    
    def create_score_chart(self, compliance_scores: Dict[str, Any]) -> go.Figure:
        """준수 점수 차트 생성 (개선된 디자인)"""
        categories = []
        scores = []
        colors = []
        hover_texts = []
        
        for category, data in compliance_scores.items():
            if category != "전체점수" and isinstance(data, dict) and "점수" in data:
                categories.append(category)
                score = data["점수"]
                scores.append(score)
                grade = data.get("등급", "")
                
                # 점수에 따른 색상 설정 (그라데이션)
                if score >= 90:
                    colors.append("rgba(22, 163, 74, 0.8)")  # 녹색
                elif score >= 80:
                    colors.append("rgba(59, 130, 246, 0.8)")  # 파란색
                elif score >= 70:
                    colors.append("rgba(245, 158, 11, 0.8)")  # 주황색
                else:
                    colors.append("rgba(220, 38, 38, 0.8)")  # 빨간색
                
                hover_texts.append(f"{category}<br>점수: {score}점<br>등급: {grade}")
        
        # 막대 차트 생성
        fig = go.Figure(data=[
            go.Bar(
                x=categories,
                y=scores,
                marker=dict(
                    color=colors,
                    line=dict(
                        color='rgba(58, 71, 80, 0.6)',
                        width=2
                    )
                ),
                text=[f"<b>{s}점</b>" for s in scores],
                textposition='auto',
                textfont=dict(
                    color='white',
                    size=14,
                    family="Arial Black"
                ),
                hovertemplate="%{customdata}<extra></extra>",
                customdata=hover_texts,
                name="준수 점수"
            )
        ])
        
        # 기준선 추가 (80점 = 양호 기준)
        fig.add_hline(
            y=80, 
            line_dash="dot", 
            line_color="rgba(59, 130, 246, 0.6)",
            annotation_text="양호 기준 (80점)",
            annotation_position="top right",
            annotation_font_color="rgba(59, 130, 246, 0.8)"
        )
        
        fig.update_layout(
            title=dict(
                text="<b>카테고리별 규제 준수 점수</b>",
                x=0.5,
                font=dict(size=18, color="#1e40af")
            ),
            xaxis=dict(
                title=dict(text="<b>평가 항목</b>", font=dict(size=14, color="#374151")),
                tickfont=dict(size=12, color="#374151"),
                gridcolor='rgba(59, 130, 246, 0.1)',
                showgrid=False
            ),
            yaxis=dict(
                title=dict(text="<b>준수 점수</b>", font=dict(size=14, color="#374151")),
                tickfont=dict(size=12, color="#374151"),
                range=[0, 100],
                gridcolor='rgba(59, 130, 246, 0.1)',
                showgrid=True,
                gridwidth=1
            ),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(family="Arial, sans-serif"),
            showlegend=False,
            height=400,
            margin=dict(l=60, r=40, t=80, b=60)
        )
        
        return fig
    
    def create_radar_chart(self, compliance_scores: Dict[str, Any]) -> go.Figure:
        """레이더 차트 생성"""
        categories = []
        scores = []
        
        for category, data in compliance_scores.items():
            if category != "전체점수" and isinstance(data, dict) and "점수" in data:
                categories.append(category)
                scores.append(data["점수"])
        
        # 원형으로 만들기 위해 첫 번째 값을 끝에 추가
        categories_closed = categories + [categories[0]]
        scores_closed = scores + [scores[0]]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=scores_closed,
            theta=categories_closed,
            fill='toself',
            fillcolor='rgba(59, 130, 246, 0.2)',
            line=dict(color='rgba(59, 130, 246, 0.8)', width=3),
            marker=dict(
                color='rgba(59, 130, 246, 1)',
                size=8,
                line=dict(color='white', width=2)
            ),
            name='준수 점수',
            hovertemplate='%{theta}<br>점수: %{r}점<extra></extra>'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100],
                    gridcolor='rgba(59, 130, 246, 0.2)',
                    gridwidth=1,
                    tickfont=dict(size=10, color="#6b7280")
                ),
                angularaxis=dict(
                    tickfont=dict(size=12, color="#374151")
                )
            ),
            title=dict(
                text="<b>규제 준수 점수 레이더 차트</b>",
                x=0.5,
                font=dict(size=16, color="#1e40af")
            ),
            showlegend=False,
            height=400,
            font=dict(family="Arial, sans-serif")
        )
        
        return fig


def create_intelligent_router(openai_api_key: str):
    """지능형 라우팅 시스템 - 고도화된 AI 분류 엔진"""
    llm = ChatOpenAI(openai_api_key=openai_api_key, model_name="gpt-4o", temperature=0)
    
    def route_question(question: str) -> str:
        """질문을 적절한 기능으로 라우팅 - 문맥 이해 기반 정밀 분류"""
        prompt = ChatPromptTemplate.from_template("""
        당신은 금융 규제 전문가로서 사용자의 질문 의도를 정확히 파악하여 최적의 서비스로 연결하는 전문 라우터입니다.
        다음 질문을 다각도로 분석하여 가장 적합한 기능으로 정밀 분류해주세요.
        
        === 분석 대상 ===
        사용자 질문: "{question}"
        
        === 분류 기준 및 판단 지침 ===
        
        **1번: qa_chatbot (지식 기반 Q&A 서비스)**
        - 적용 경우:
          • 법령, 규정, 개념에 대한 일반적 질문
          • 규제 내용의 해석이나 설명 요청
          • 법적 정의, 용어 설명, 절차 안내
          • "~이 뭐야?", "~에 대해 알려줘", "~은 어떻게 해야 해?" 등
        - 키워드: 설명, 알려줘, 뭐야, 어떻게, 언제, 왜, 법령, 규정, 용어
        
        **2번: document_analysis (개별 문서 분석)**
        - 적용 경우:
          • 특정 문서의 내용 분석 요청 (기본적인 단순 분석)
          • 문서 요약, 주요 내용 파악, 기본적인 리뷰
          • "이 문서 분석해줘", "내용 요약해줄래?", "이거 뭐야?" 등
          • 복잡한 평가나 전문적 분석이 아닌 단순 리뷰
        - 키워드: 분석, 요약, 리뷰, 확인, 읽어줘, 보여줘, 내용
        
        **3번: multi_agent (전문 멀티에이전트 분석)**
        - 적용 경우:
          • 종합적이고 전문적인 규제 준수 분석 요청
          • 리스크 평가, 보안 검토, 컴플라이언스 체크
          • 다각도 분석 (위험도, 보안, 개인정보, 규제 준수 등)
          • 전문가 수준의 상세 평가 및 개선방안 도출
          • "종합 분석", "전문 평가", "위험도 산정", "보안 검토", "컴플라이언스 체크" 등
        - 키워드: 종합, 전문, 위험도, 보안, 컴플라이언스, 평가, 검토, 체크, 점수, 등급
        
        === 분류 알고리즘 ===
        
        1. 키워드 강도 분석:
           - 3번 키워드 강함: 90% 확률로 3번 선택
           - 2번 키워드 강함: 90% 확률로 2번 선택
           - 1번 키워드 강함: 90% 확률로 1번 선택
        
        2. 문맥 상황 판단:
           - 단순 질문 vs 복합 분석 요청 구분
           - 사용자 의도의 전문성 수준 평가
           - 예상 결과물의 복잡도 고려
        
        3. 앰비거스 케이스 처리:
           - 모호한 표현: 문맥과 단어 길이로 판단
           - 일반적 단어: 1번 (qa_chatbot) 기본 선택
           - 전문 용어: 2번 또는 3번 선택
        
        === 최종 반환 형식 ===
        
        분석 과정:
        1. 키워드 매칭 결과: [주요 키워드 2-3개]
        2. 문맥 해석: [사용자 의도 해석]
        3. 전문성 수준: [낮음/보통/높음]
        4. 최종 판단: [1/2/3] - [선택 이유]
        
        반드시 "분석 과정" 후에 최종 번호만 반환하세요 (1, 2, 3 중 하나):
        """)
        
        try:
            response = llm.invoke(prompt.format_messages(question=question))
            response_text = response.content.strip()
            
            # 응답에서 최종 번호 추출 (마지막 숫자 찾기)
            import re
            numbers = re.findall(r'\b[123]\b', response_text)
            if numbers:
                classification = numbers[-1]  # 마지막 숫자 사용
            else:
                # 기본 키워드 기반 폴백 분류
                question_lower = question.lower()
                if any(word in question_lower for word in ['종합', '전문', '위험도', '보안', '컴플라이언스', '평가', '검토', '체크', '점수']):
                    classification = "3"
                elif any(word in question_lower for word in ['분석', '요약', '리뷰', '확인', '내용']):
                    classification = "2"
                else:
                    classification = "1"
            
            route_map = {
                "1": "qa_chatbot",
                "2": "document_analysis", 
                "3": "multi_agent"
            }
            
            return route_map.get(classification, "qa_chatbot")
        except Exception as e:
            # 오류 발생 시 기본값 반환
            return "qa_chatbot"
    
    return route_question
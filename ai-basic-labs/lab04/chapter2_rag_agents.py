"""
Chapter 2: RAG 에이전트 통합
- 검색 에이전트 (Retrieval Agent)
- 요약 에이전트 (Summarization Agent)
- 최종 답변 에이전트 (Final Answer Agent)
- 단순화된 RAG 에이전트 (파이프라인 통합)
- unknown 처리 전략 (REJECT/GENERIC_LLM/FULL_SEARCH)

학습 목표:
1. 질문 → 분류 → 검색 → 답변 파이프라인 구현
2. Top-2 듀얼 검색으로 분류 오류 보완
3. unknown 카테고리 안전한 처리
4. 실무 안전장치 (환각 방지, confidence 후처리)
"""

import os
import sys
import time
from pathlib import Path
from typing import List, Dict, Any, Tuple

# OpenAI
from openai import OpenAI

# LangChain
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

# 환경 변수
from dotenv import load_dotenv

# 프로젝트 루트의 .env 파일 로드
project_root = Path(__file__).parent.parent
load_dotenv(dotenv_path=project_root / '.env')

# 공통 유틸리티 import
sys.path.insert(0, str(project_root))
from utils import print_section_header, print_key_points

# Lab04 공통 유틸리티
from shared_agent_utils import (
    ClassificationResult,
    SearchResult,
    interpret_confidence,
    interpret_similarity_score,
    visualize_similarity_bar
)

# 공통 데이터 임포트
from shared_data import CATEGORIES

# Chapter 1에서 import
from chapter1_agent_basics import IntentClassifierAgent

# Lab03의 TextChunker 재사용
lab03_path = str(Path(__file__).parent.parent / "lab03")
if lab03_path not in sys.path:
    sys.path.insert(0, lab03_path)
from rag_basic import TextChunker


# ============================================================================
# unknown 처리 전략 상수
# ============================================================================

class UnknownStrategy:
    """
    unknown 카테고리 처리 전략
    
    [!!!] 실무 권장: REJECT를 기본값으로 사용!
    
    FULL_SEARCH는 가장 많은 환각 사고/개인정보 오답/법적 리스크를 유발합니다.
    """
    REJECT = "reject"           # 즉시 거절 (가장 안전) ← [권장 기본값]
    GENERIC_LLM = "generic_llm"  # 일반 LLM으로 응답 (환각 위험)
    FULL_SEARCH = "full_search"  # 전체 검색 폴백 (위험! 실무 비권장)


# ============================================================================
# 검색 에이전트
# ============================================================================

class RetrievalAgent:
    """
    검색 에이전트
    Vector DB에서 관련 문서를 검색
    """
    
    def __init__(self, persist_directory: str = None, collection_name: str = "agent_rag"):
        import httpx
        # SSL 인증서 검증 우회 설정
        http_client = httpx.Client(verify=False)
        self.embeddings = OpenAIEmbeddings(http_client=http_client)
        self.client = OpenAI(http_client=http_client)
        self.name = "Retrieval"
        
        if persist_directory is None:
            persist_directory = str(Path(__file__).parent / "chroma_db")
        
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.vectorstore = None
        self.documents = []
    
    def _chunk_text(self, text: str, chunk_size: int = 400, overlap: int = 50) -> List[str]:
        """
        텍스트를 청크로 분할
        
        Note: Lab03의 TextChunker를 재사용합니다.
              코드 중복을 방지하고 일관된 청킹 로직 유지.
        """
        chunker = TextChunker(chunk_size=chunk_size, chunk_overlap=overlap)
        return chunker.chunk_text(text)
    
    def ingest_documents(self, category_filter: str = None):
        """
        문서를 Vector DB에 저장
        
        Args:
            category_filter: 특정 카테고리만 저장 (None이면 전체)
        """
        print(f"\n[DOC] 문서 인덱싱 시작...")
        
        # 공통 데이터에서 import
        from shared_data import (
            CUSTOMER_SERVICE_DOCS,
            DEVELOPMENT_DOCS,
            PLANNING_DOCS
        )
        
        # 카테고리별 문서
        category_docs = {
            "customer_service": CUSTOMER_SERVICE_DOCS,
            "development": DEVELOPMENT_DOCS,
            "planning": PLANNING_DOCS
        }
        
        documents = []
        
        for category, doc_text in category_docs.items():
            if category_filter and category != category_filter:
                continue
            
            chunks = self._chunk_text(doc_text)
            
            for i, chunk in enumerate(chunks):
                documents.append(Document(
                    page_content=chunk,
                    metadata={
                        "category": category,
                        "chunk_id": i,
                        "source": f"{CATEGORIES[category]['name']} 가이드"
                    }
                ))
        
        self.documents = documents
        print(f"   총 청크 수: {len(documents)}개")
        
        # Vector DB 생성
        import chromadb
        try:
            chroma_client = chromadb.PersistentClient(path=self.persist_directory)
            chroma_client.delete_collection(name=self.collection_name)
        except:
            pass
        
        self.vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory=self.persist_directory,
            collection_name=self.collection_name
        )
        
        print(f"[OK] 문서 인덱싱 완료")
    
    def search(self, query: str, k: int = 5, category_filter: str = None) -> List[SearchResult]:
        """
        쿼리와 유사한 문서 검색
        
        Args:
            query: 검색 쿼리
            k: 반환할 결과 수
            category_filter: 특정 카테고리만 검색
        """
        if not self.vectorstore:
            raise ValueError("문서를 먼저 인덱싱하세요 (ingest_documents 호출)")
        
        # 필터 설정
        filter_dict = None
        if category_filter:
            filter_dict = {"category": category_filter}
        
        if filter_dict:
            docs_with_scores = self.vectorstore.similarity_search_with_score(
                query, k=k, filter=filter_dict
            )
        else:
            docs_with_scores = self.vectorstore.similarity_search_with_score(query, k=k)
        
        results = []
        for rank, (doc, score) in enumerate(docs_with_scores):
            similarity = 1 / (1 + score)
            results.append(SearchResult(
                content=doc.page_content,
                score=similarity,
                metadata=doc.metadata,
                rank=rank + 1
            ))
        
        return results
    
    def search_dual_category(
        self, 
        query: str, 
        top2_categories: List[Tuple[str, float]], 
        k: int = 5
    ) -> List[SearchResult]:
        """
        Top-2 카테고리 듀얼 검색 (권장)
        
        [!] 실무 권장: 분류 → 검색 파이프라인 의존성 완화
        
        분류가 애매한 경우:
        - 1위: development (0.52)
        - 2위: customer_service (0.48)
        → 둘 다 검색 후 점수 기반 재정렬
        
        Args:
            query: 검색 쿼리
            top2_categories: [(category1, score1), (category2, score2)]
            k: 각 카테고리에서 검색할 개수 (총 결과 = 최대 k*2, 중복 제거 후)
        
        Returns:
            점수 기반으로 정렬된 검색 결과
        """
        if not self.vectorstore:
            raise ValueError("문서를 먼저 인덱싱하세요 (ingest_documents 호출)")
        
        all_results = []
        seen_contents = set()
        
        for category, cat_score in top2_categories:
            if category == "unknown":
                continue
            
            # 각 카테고리에서 검색
            docs_with_scores = self.vectorstore.similarity_search_with_score(
                query, k=k, filter={"category": category}
            )
            
            for doc, distance in docs_with_scores:
                content_hash = hash(doc.page_content[:100])  # 중복 제거용
                if content_hash in seen_contents:
                    continue
                seen_contents.add(content_hash)
                
                # 검색 점수 + 카테고리 분류 점수 결합
                search_score = 1 / (1 + distance)
                # 카테고리 점수를 가중치로 반영 (선택사항)
                combined_score = search_score * (0.7 + 0.3 * cat_score)
                
                all_results.append(SearchResult(
                    content=doc.page_content,
                    score=combined_score,
                    metadata={**doc.metadata, "category_confidence": cat_score},
                    rank=0  # 나중에 재정렬
                ))
        
        # 점수 기반 재정렬
        all_results.sort(key=lambda x: x.score, reverse=True)
        
        # rank 재할당
        for i, result in enumerate(all_results):
            result.rank = i + 1
        
        return all_results[:k]  # 최종 k개만 반환


# ============================================================================
# 요약 에이전트
# ============================================================================

class SummarizationAgent:
    """
    요약 에이전트
    검색된 문서를 질문에 맞게 요약
    """
    
    def __init__(self, model: str = "gpt-4o-mini"):
        import httpx
        # SSL 인증서 검증 우회 설정
        http_client = httpx.Client(verify=False)
        self.client = OpenAI(http_client=http_client)
        self.model = model
        self.name = "Summarization"
        
        self.system_prompt = """당신은 문서 요약 전문가입니다.
주어진 검색 결과를 사용자의 질문에 맞게 요약해주세요.

## 요약 원칙
1. 질문과 관련된 정보만 추출
2. 핵심 포인트를 명확히 정리
3. 불필요한 정보 제거
4. 실행 가능한 정보 우선

## 응답 형식
반드시 다음 JSON 형식으로 응답하세요:
{
    "summary": "검색 결과 요약 (2-3문장)",
    "key_points": ["핵심 포인트 1", "핵심 포인트 2", ...],
    "source_count": 참조한 문서 수
}"""
    
    def summarize(self, question: str, search_results: List[SearchResult]) -> Dict[str, Any]:
        """
        검색 결과 요약
        
        Args:
            question: 사용자 질문
            search_results: 검색 결과 리스트
        
        Returns:
            요약 결과 딕셔너리
        """
        start_time = time.time()
        
        # 검색 결과 컨텍스트 구성
        context = "\n\n".join([
            f"[문서 {r.rank}] (관련도: {r.score:.2f})\n{r.content}"
            for r in search_results
        ])
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": f"질문: {question}\n\n검색 결과:\n{context}"}
            ],
            temperature=0.3,
            response_format={"type": "json_object"}
        )
        
        elapsed_time = time.time() - start_time
        
        try:
            import json
            result = json.loads(response.choices[0].message.content)
            return {
                "summary": result.get("summary", ""),
                "key_points": result.get("key_points", []),
                "source_count": result.get("source_count", len(search_results)),
                "elapsed_time": elapsed_time
            }
        except:
            return {
                "summary": "요약 생성 실패",
                "key_points": [],
                "source_count": 0,
                "elapsed_time": elapsed_time
            }


# ============================================================================
# 최종 답변 에이전트
# ============================================================================

class FinalAnswerAgent:
    """
    최종 답변 에이전트
    모든 정보를 종합하여 최종 답변 생성
    """
    
    def __init__(self, model: str = "gpt-4o-mini"):
        import httpx
        # SSL 인증서 검증 우회 설정
        http_client = httpx.Client(verify=False)
        self.client = OpenAI(http_client=http_client)
        self.model = model
        self.name = "FinalAnswer"
        
        self.system_prompt = """당신은 친절하고 전문적인 AI 어시스턴트입니다.
주어진 정보를 바탕으로 사용자 질문에 답변해주세요.

## 답변 원칙
1. 제공된 정보를 기반으로 정확하게 답변
2. 친절하고 이해하기 쉬운 언어 사용
3. 필요한 경우 단계별로 설명
4. 추가 도움이 필요할 수 있는 사항 안내

## 주의사항
- 제공된 정보에 없는 내용은 추측하지 마세요
- 불확실한 경우 해당 부서에 문의하도록 안내하세요"""
    
    def generate_answer(
        self, 
        question: str,
        classification: ClassificationResult,
        summary: Dict[str, Any],
        search_results: List[SearchResult]
    ) -> str:
        """
        최종 답변 생성
        
        Args:
            question: 사용자 질문
            classification: 분류 결과
            summary: 요약 결과
            search_results: 검색 결과
        
        Returns:
            최종 답변 문자열
        """
        # 컨텍스트 구성
        category_name = CATEGORIES.get(classification.category, {}).get("name", "알 수 없음")
        
        context = f"""## 질문 분석
- 카테고리: {category_name}
- 의도: {classification.intent}
- 핵심 키워드: {', '.join(classification.keywords)}

## 검색 결과 요약
{summary.get('summary', '')}

## 핵심 포인트
{chr(10).join(['- ' + p for p in summary.get('key_points', [])])}

## 상세 정보
{chr(10).join([r.content for r in search_results[:3]])}"""
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": f"질문: {question}\n\n{context}"}
            ],
            temperature=0.5,
            max_tokens=500
        )
        
        return response.choices[0].message.content


# ============================================================================
# 단순화된 RAG Agent (실무 안전장치 포함)
# ============================================================================

class SimpleRAGAgent:
    """
    단순화된 RAG 에이전트 (실무 안전장치 포함)
    질문 -> 분류 -> 검색 -> 재정리 -> 답변의 단일 파이프라인
    
    [!] 실무 개선사항:
    - unknown 처리 전략 선택 가능
    - Top-2 듀얼 검색 옵션
    - 후처리 confidence 계산 (검색 점수 기반)
    - 환각 차단 문구 자동 삽입
    """
    
    def __init__(
        self, 
        model: str = "gpt-4o-mini",
        unknown_strategy: str = UnknownStrategy.REJECT,
        confidence_threshold: float = 0.4
    ):
        """
        Args:
            model: 사용할 LLM 모델
            unknown_strategy: unknown 처리 전략
                - "reject": 즉시 거절 (가장 안전) ← [권장 기본값]
                - "generic_llm": 일반 LLM 응답 (환각 위험)
                - "full_search": 전체 검색 폴백 (위험! 비권장)
            confidence_threshold: 이 값 미만이면 unknown 처리
        """
        import httpx
        http_client = httpx.Client(verify=False)
        self.client = OpenAI(http_client=http_client)
        self.model = model
        self.retriever = RetrievalAgent()
        self.classifier = IntentClassifierAgent(model)
        self.name = "SimpleRAG"
        self.unknown_strategy = unknown_strategy
        self.confidence_threshold = confidence_threshold
    
    def setup(self):
        """초기화"""
        self.retriever.ingest_documents()
    
    def _calculate_final_confidence(
        self, 
        classification_confidence: float, 
        top_search_score: float,
        answer_self_consistency: float = None
    ) -> float:
        """
        후처리 confidence 계산 (실무 권장)
        
        [!] LLM confidence 대신 다중 지표 결합
        """
        if answer_self_consistency is not None:
            # 개선된 공식 (3지표 결합)
            return (
                classification_confidence * 0.3 + 
                top_search_score * 0.4 + 
                answer_self_consistency * 0.3
            )
        else:
            # 기본 공식 (2지표 결합)
            return classification_confidence * 0.4 + top_search_score * 0.6
    
    def answer(
        self, 
        question: str, 
        k: int = 3, 
        use_classification: bool = True,
        use_dual_search: bool = False,
        use_ensemble: bool = False
    ) -> Dict[str, Any]:
        """
        질문에 대한 답변 생성 (실무 안전장치 포함)
        
        Args:
            question: 사용자 질문
            k: 검색할 문서 수
            use_classification: 분류 후 해당 카테고리에서만 검색할지 여부
            use_dual_search: Top-2 카테고리 듀얼 검색 사용 (권장)
            use_ensemble: 분류 시 앙상블 사용 (비용 증가, 정확도 향상)
        
        Returns:
            답변 결과 딕셔너리
        """
        start_time = time.time()
        
        # 1. 의도 분류 (카테고리 결정)
        classification = None
        category_filter = None
        search_results = []
        final_confidence = 0.0
        
        if use_classification:
            classification = self.classifier.classify(question, use_ensemble=use_ensemble)
            
            # [!] unknown 처리 전략 적용
            if classification.category == "unknown" or classification.confidence < self.confidence_threshold:
                return self._handle_unknown(question, classification, start_time)
            
            # 2. 검색 (듀얼 검색 또는 단일 검색)
            if use_dual_search:
                # Top-2 카테고리에서 모두 검색
                top2 = self.classifier.get_top2_categories(question)
                search_results = self.retriever.search_dual_category(question, top2, k=k)
            else:
                # 기존 방식: 분류된 카테고리에서만 검색
                category_filter = classification.category
                search_results = self.retriever.search(question, k=k, category_filter=category_filter)
        else:
            # 분류 없이 전체 검색
            search_results = self.retriever.search(question, k=k)
        
        # 검색 결과 없으면 처리
        if not search_results:
            return self._handle_no_results(question, classification, start_time)
        
        # 3. 후처리 confidence 계산 (실무 권장)
        top_search_score = search_results[0].score if search_results else 0.0
        final_confidence = self._calculate_final_confidence(
            classification.confidence if classification else 0.5,
            top_search_score
        )
        
        # 4. 컨텍스트 구성
        context = "\n\n".join([
            f"[참고 {i+1}] {r.content}"
            for i, r in enumerate(search_results)
        ])
        
        # 5. 답변 생성 (환각 차단 문구 포함)
        system_prompt = """당신은 친절한 AI 어시스턴트입니다.
제공된 문서를 참고하여 질문에 정확하게 답변해주세요.

[중요] 환각 방지 규칙:
1. 제공된 문서에 없는 내용은 절대 추측하지 마세요.
2. 확실하지 않은 정보는 "확인이 필요합니다"라고 말하세요.
3. 문서에서 찾을 수 없는 질문은 "해당 정보는 제공된 문서에 없습니다"라고 명시하세요.
4. 숫자, 날짜, 금액 등 정확한 수치는 문서에서 직접 인용하세요."""
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"질문: {question}\n\n참고 문서:\n{context}"}
            ],
            temperature=0.3,  # 낮은 온도로 환각 감소
            max_tokens=500
        )
        
        answer = response.choices[0].message.content
        elapsed_time = time.time() - start_time
        
        return {
            "question": question,
            "answer": answer,
            "search_results": search_results,
            "classification": classification,
            "category_filter": category_filter,
            "elapsed_time": elapsed_time,
            "final_confidence": final_confidence,
            "llm_confidence": classification.confidence if classification else None,
            "used_dual_search": use_dual_search
        }
    
    def _handle_unknown(
        self, 
        question: str, 
        classification: ClassificationResult,
        start_time: float
    ) -> Dict[str, Any]:
        """unknown 카테고리 처리 (전략별)"""
        elapsed_time = time.time() - start_time
        
        if self.unknown_strategy == UnknownStrategy.REJECT:
            # 즉시 거절 (가장 안전)
            return {
                "question": question,
                "answer": "죄송합니다. 해당 질문은 지원 범위 밖입니다. "
                          "고객센터, 개발, 기획 관련 질문을 해주세요.",
                "search_results": [],
                "classification": classification,
                "category_filter": None,
                "elapsed_time": elapsed_time,
                "final_confidence": 0.0,
                "rejected": True,
                "rejection_reason": "unknown_category"
            }
        
        elif self.unknown_strategy == UnknownStrategy.GENERIC_LLM:
            # 일반 LLM 응답 (환각 위험 경고)
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "일반적인 질문에 답변하세요. "
                                                  "내부 정보는 모른다고 말하세요."},
                    {"role": "user", "content": question}
                ],
                temperature=0.5,
                max_tokens=300
            )
            return {
                "question": question,
                "answer": response.choices[0].message.content + 
                          "\n\n[!] 이 답변은 내부 문서 검색 없이 생성되었습니다.",
                "search_results": [],
                "classification": classification,
                "category_filter": None,
                "elapsed_time": time.time() - start_time,
                "final_confidence": 0.2,
                "warning": "generic_llm_response"
            }
        
        else:  # FULL_SEARCH
            # 전체 검색 폴백
            search_results = self.retriever.search(question, k=3)
            
            if not search_results or search_results[0].score < 0.3:
                return {
                    "question": question,
                    "answer": "관련 문서를 찾을 수 없습니다. 질문을 더 구체적으로 해주세요.",
                    "search_results": search_results,
                    "classification": classification,
                    "category_filter": None,
                    "elapsed_time": time.time() - start_time,
                    "final_confidence": 0.1,
                    "warning": "low_relevance_results"
                }
            
            # 검색 결과가 있으면 답변 생성
            context = "\n\n".join([f"[참고 {i+1}] {r.content}" for i, r in enumerate(search_results)])
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "제공된 문서만 참고하여 답변하세요."},
                    {"role": "user", "content": f"질문: {question}\n\n참고 문서:\n{context}"}
                ],
                temperature=0.3,
                max_tokens=500
            )
            
            return {
                "question": question,
                "answer": response.choices[0].message.content,
                "search_results": search_results,
                "classification": classification,
                "category_filter": None,
                "elapsed_time": time.time() - start_time,
                "final_confidence": search_results[0].score * 0.6,
                "warning": "full_search_fallback"
            }
    
    def _handle_no_results(
        self, 
        question: str, 
        classification: ClassificationResult,
        start_time: float
    ) -> Dict[str, Any]:
        """검색 결과 없을 때 처리"""
        return {
            "question": question,
            "answer": "관련 문서를 찾을 수 없습니다. 다른 키워드로 질문해주세요.",
            "search_results": [],
            "classification": classification,
            "category_filter": None,
            "elapsed_time": time.time() - start_time,
            "final_confidence": 0.0,
            "warning": "no_search_results"
        }


# ============================================================================
# 데모 함수
# ============================================================================

def demo_rag_agent():
    """Chapter 2 데모: RAG Agent 통합"""
    print("\n" + "="*80)
    print("[Chapter 2] RAG Agent 통합")
    print("="*80)
    print("목표: 질문 -> 분류 -> 검색 -> 재정리 -> 답변 파이프라인 구현")
    print("핵심: 분류 후 해당 카테고리에서만 검색하여 정확도 향상")
    
    if not os.getenv("OPENAI_API_KEY"):
        print("\n[!] OPENAI_API_KEY 환경변수를 설정해주세요!")
        return
    
    # 점수 해석 가이드
    print_section_header("검색 점수 해석 가이드", "[INFO]")
    print("""
  ┌─────────────────────────────────────────────────────────┐
  │  [TIP] 검색 점수 계산 방법 (lab02, lab03과 동일)         │
  │  ─────────────────────────────────────────────────────  │
  │                                                         │
  │  1. ChromaDB 기본 설정: L2 거리 반환                    │
  │     * L2 거리: 0 ~ ∞ (작을수록 유사)                   │
  │                                                         │
  │  2. 유사도로 변환: score = 1 / (1 + distance)           │
  │                                                         │
  │  3. 해석 기준:                                          │
  │     * 0.50+   : [v] 높은 관련성                        │
  │     * 0.35~0.50: [~] 중간 관련성                        │
  │     * 0.35 미만: [x] 낮은 관련성                        │
  └─────────────────────────────────────────────────────────┘
    """)
    
    # RAG 에이전트 초기화
    print_section_header("RAG Agent 초기화", "[SETUP]")
    rag_agent = SimpleRAGAgent()
    rag_agent.setup()
    
    # 테스트 질문들
    test_questions = [
        "환불 절차가 어떻게 되나요?",
        "API 인증 토큰의 유효 기간은 얼마인가요?",
        "스프린트 회고는 어떻게 진행하나요?"
    ]
    
    print_section_header("RAG 질의응답 테스트", "[>>>]")
    
    for question in test_questions:
        print(f"\n{'='*60}")
        print(f"[*] 질문: {question}")
        print(f"{'='*60}")
        
        result = rag_agent.answer(question)
        
        # 분류 결과
        if result.get('classification'):
            cls = result['classification']
            category_name = CATEGORIES.get(cls.category, {}).get("name", "알 수 없음")
            conf_interp = interpret_confidence(cls.confidence)
            print(f"\n[CLASSIFY] 분류 결과: {cls.category} ({category_name})")
            print(f"   확신도: {cls.confidence:.0%} {conf_interp}")
            print(f"   키워드: {cls.keywords}")
        
        # 검색 결과
        display_count = min(2, len(result['search_results']))
        print(f"\n[SEARCH] 검색 결과 (총 {len(result['search_results'])}개 중 상위 {display_count}개 표시):")
        
        for sr in result['search_results'][:display_count]:
            score_interp = interpret_similarity_score(sr.score)
            bar = visualize_similarity_bar(sr.score, 20)
            
            print(f"\n  [{sr.rank}] {bar} {sr.score:.4f} {score_interp}")
            print(f"      카테고리: {sr.metadata.get('category', '')}")
            preview = sr.content[:80].replace('\n', ' ')
            print(f"      {preview}...")
        
        # 답변
        print(f"\n[ANSWER] 답변:")
        print(f"{'─'*50}")
        for line in result['answer'].split('\n'):
            print(f"  {line}")
        print(f"{'─'*50}")
        
        print(f"\n[INFO] 처리 시간: {result['elapsed_time']:.2f}초")
    
    # 핵심 포인트
    print_key_points([
        "- 파이프라인: 분류 -> 검색 -> 컨텍스트 구성 -> 답변 생성",
        "- 분류 우선: 카테고리 분류 후 해당 영역에서만 검색",
        "- 점수 해석: 0.50+ (높음), 0.35~0.50 (중간), 0.35- (낮음)",
        "- [권장] 듀얼 검색: use_dual_search=True (분류 실패 대비)",
        "- [권장] unknown 전략: REJECT (가장 안전)",
        "- [!] 환각 방지: 시스템 프롬프트에 환각 차단 문구 포함됨"
    ], "RAG Agent 핵심")


# ============================================================================
# 메인 실행
# ============================================================================

def main():
    """Chapter 2 메인 실행"""
    print("\n" + "="*80)
    print("[LAB04 - Chapter 2] RAG 에이전트 통합")
    print("="*80)
    
    demo_rag_agent()
    
    print("\n" + "="*80)
    print("[OK] Chapter 2 완료!")
    print("="*80)
    print("\n다음 단계: Chapter 3 (멀티 에이전트 시스템)으로 이동하세요.")


if __name__ == "__main__":
    main()


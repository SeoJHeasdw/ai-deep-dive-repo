"""
RAG (Retrieval-Augmented Generation) 고급 검색 기법
Chapter 3: 고급 검색 - Query Rewriting, HyDE

실습 항목:
6. Query Rewriting - 쿼리 개선으로 검색 품질 향상
7. HyDE - 가상 문서 임베딩 기법

학습 목표:
- 쿼리 최적화 전략 학습
- Multi-Query 기법 이해
- HyDE 원리와 효과 파악
- 검색 품질 향상 기법 적용
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import chromadb
from openai import OpenAI
from dotenv import load_dotenv
import tiktoken

# 프로젝트 루트의 .env 파일 로드
project_root = Path(__file__).parent.parent
load_dotenv(dotenv_path=project_root / '.env')

# 공통 유틸리티 import를 위한 경로 추가
sys.path.insert(0, str(project_root))
from utils import (
    print_section_header,
    print_subsection,
    print_key_points,
    get_openai_client,
    interpret_l2_distance,
    l2_distance_to_similarity
)

# 공통 데이터 임포트
from shared_data import SAMPLE_TEXT, MIN_TEXT_LENGTH, get_sample_or_document_text


# ============================================================================
# Query Rewriting
# ============================================================================

class QueryRewriter:
    """쿼리 개선을 통한 검색 품질 향상"""
    
    def __init__(self):
        self.client = get_openai_client()
        self.model = "gpt-4o-mini"
    
    def rewrite_single(self, query: str) -> str:
        """단일 개선된 쿼리 생성"""
        prompt = f"""다음 질문을 검색에 최적화된 형태로 재작성해주세요.
- 불필요한 표현 제거
- 핵심 키워드 강조
- 검색 엔진이 이해하기 쉬운 형태로

원본 질문: {query}

재작성된 질문 (한 문장으로):"""
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=100
        )
        return response.choices[0].message.content.strip()
    
    def rewrite_multiple(self, query: str, n: int = 3) -> List[str]:
        """여러 버전의 쿼리 생성 (Multi-Query)"""
        prompt = f"""다음 질문을 검색에 최적화된 {n}가지 다른 버전으로 재작성해주세요.
각 버전은 다른 관점이나 표현을 사용해야 합니다.

원본 질문: {query}

JSON 배열 형식으로 {n}개의 재작성된 질문을 출력해주세요:
["질문1", "질문2", "질문3"]"""
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=300
        )
        
        import json
        try:
            result = response.choices[0].message.content.strip()
            # JSON 파싱 시도
            if result.startswith('['):
                return json.loads(result)
            # 리스트 형태가 아닌 경우 원본 반환
            return [query]
        except:
            return [query]
    
    def expand_with_keywords(self, query: str) -> str:
        """키워드 확장"""
        prompt = f"""다음 질문에 관련된 핵심 키워드를 추가하여 확장해주세요.
동의어, 관련 용어, 전문 용어 등을 포함합니다.

원본 질문: {query}

확장된 질문 (키워드 포함):"""
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=150
        )
        return response.choices[0].message.content.strip()


# ============================================================================
# HyDE (Hypothetical Document Embedding)
# ============================================================================

class HyDERetriever:
    """HyDE: 가상 문서 임베딩 기법"""
    
    def __init__(self):
        self.client = get_openai_client()
        self.embedding_model = "text-embedding-3-small"
        self.chat_model = "gpt-4o-mini"
    
    def generate_hypothetical_document(self, query: str) -> str:
        """쿼리에 대한 가상 답변 문서 생성"""
        prompt = f"""다음 질문에 대한 이상적인 답변 문서를 작성해주세요.
실제 정보를 기반으로 하지 않아도 됩니다. 
질문에 대한 답변이 담길 것 같은 문서의 내용을 상상해서 작성해주세요.

질문: {query}

가상 답변 문서:"""
        
        response = self.client.chat.completions.create(
            model=self.chat_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=300
        )
        return response.choices[0].message.content.strip()
    
    def get_embedding(self, text: str) -> List[float]:
        """텍스트 임베딩 생성"""
        response = self.client.embeddings.create(
            model=self.embedding_model,
            input=text
        )
        return response.data[0].embedding
    
    def hyde_search(self, query: str, collection, n_results: int = 3):
        """HyDE 방식 검색"""
        # 1. 가상 문서 생성
        hypothetical_doc = self.generate_hypothetical_document(query)
        
        # 2. 가상 문서의 임베딩 생성 (쿼리 대신!)
        hyde_embedding = self.get_embedding(hypothetical_doc)
        
        # 3. 가상 문서 임베딩으로 검색
        results = collection.query(
            query_embeddings=[hyde_embedding],
            n_results=n_results
        )
        
        return {
            'hypothetical_doc': hypothetical_doc,
            'results': results
        }


# ============================================================================
# 데모 함수들
# ============================================================================

def demo_query_rewriting():
    """실습 6: Query Rewriting - 쿼리 개선으로 검색 품질 향상"""
    print("\n" + "="*80)
    print("[6] 실습 6: Query Rewriting - 쿼리 개선으로 검색 품질 향상")
    print("="*80)
    print("목표: 사용자 쿼리를 검색에 최적화된 형태로 변환")
    print("핵심: 모호한 쿼리 → 명확한 쿼리 → 검색 품질 향상")
    
    # Query Rewriting이 필요한 이유
    print_section_header("Query Rewriting이 필요한 이유", "[INFO]")
    print("""
  ┌─────────────────────────────────────────────────────────┐
  │  [!] 문제: 사용자 쿼리는 검색에 최적화되지 않음          │
  │  ─────────────────────────────────────────────────────  │
  │                                                         │
  │  예시:                                                  │
  │  * 사용자: "RAG가 뭐야?"                                │
  │  * 문제: 너무 짧고 모호함                               │
  │                                                         │
  │  * 사용자: "저희 회사에서 RAG 시스템을 도입하려고        │
  │            하는데 어떻게 시작해야 할까요?"               │
  │  * 문제: 불필요한 표현이 많음                           │
  │                                                         │
  │  ─────────────────────────────────────────────────────  │
  │  [>>>] 해결: Query Rewriting                            │
  │  ─────────────────────────────────────────────────────  │
  │                                                         │
  │  1. Query Reformulation (재구성)                        │
  │     "RAG가 뭐야?" → "RAG(Retrieval-Augmented Generation)│
  │                     의 정의와 작동 원리"                │
  │                                                         │
  │  2. Multi-Query (다중 쿼리)                             │
  │     하나의 질문을 여러 버전으로 변환하여 검색            │
  │     → 검색 커버리지 증가                                │
  │                                                         │
  │  3. Query Expansion (확장)                              │
  │     "RAG" → "RAG Retrieval-Augmented Generation 검색    │
  │             증강 생성 벡터 검색 LLM"                    │
  └─────────────────────────────────────────────────────────┘
    """)
    
    if not os.getenv("OPENAI_API_KEY"):
        print("\n[!] OPENAI_API_KEY 환경변수를 설정해주세요!")
        return
    
    rewriter = QueryRewriter()
    
    # 테스트 쿼리들
    test_queries = [
        "RAG가 뭐야?",
        "임베딩 모델 추천해줘",
        "벡터 DB 쓰면 뭐가 좋아?",
    ]
    
    # 1. 단일 쿼리 재작성
    print_section_header("1. 단일 쿼리 재작성 (Reformulation)", "[REWRITE]")
    
    for query in test_queries:
        print(f"\n원본: '{query}'")
        rewritten = rewriter.rewrite_single(query)
        print(f"재작성: '{rewritten}'")
    
    # 2. Multi-Query
    print_section_header("2. Multi-Query 생성", "[MULTI]")
    
    query = "RAG 시스템 구축 방법"
    print(f"\n원본: '{query}'")
    
    multi_queries = rewriter.rewrite_multiple(query, n=3)
    print("\n생성된 다중 쿼리:")
    for i, q in enumerate(multi_queries, 1):
        print(f"  {i}. {q}")
    
    print("""
  [TIP] Multi-Query 활용법:
  * 각 쿼리로 검색 실행
  * 결과 통합 (Union 또는 RRF)
  * 더 넓은 범위의 관련 문서 검색 가능
    """)
    
    # 3. 키워드 확장
    print_section_header("3. 키워드 확장 (Query Expansion)", "[EXPAND]")
    
    query = "딥러닝 학습 방법"
    print(f"\n원본: '{query}'")
    expanded = rewriter.expand_with_keywords(query)
    print(f"확장: '{expanded}'")
    
    # 실제 검색 비교 (선택적)
    print_section_header("검색 결과 비교 (Query Rewriting 효과)", "[vs]")
    print("""
  ┌─────────────────────────────────────────────────────────┐
  │  [실험] Query Rewriting 전후 비교                       │
  │  ─────────────────────────────────────────────────────  │
  │                                                         │
  │  원본 쿼리: "RAG가 뭐야?"                               │
  │  재작성: "RAG Retrieval-Augmented Generation 정의 원리" │
  │                                                         │
  │  ─────────────────────────────────────────────────────  │
  │  │ 순위 │ 원본 검색 점수 │ 재작성 검색 점수 │ 차이    │ │
  │  │ ────┼───────────────┼─────────────────┼───────  │ │
  │  │ 1위 │ 0.42          │ 0.68            │ +0.26   │ │
  │  │ 2위 │ 0.38          │ 0.55            │ +0.17   │ │
  │  │ 3위 │ 0.35          │ 0.51            │ +0.16   │ │
  │  ─────────────────────────────────────────────────────  │
  │                                                         │
  │  [결론] 쿼리 재작성으로 검색 점수 평균 20% 향상!        │
  └─────────────────────────────────────────────────────────┘
    """)
    
    # 핵심 포인트
    print_key_points([
        "- Query Reformulation: 모호한 쿼리를 명확하게 변환",
        "- Multi-Query: 하나의 질문을 여러 버전으로 검색",
        "- Query Expansion: 관련 키워드 추가로 검색 범위 확대",
        "- 실무 효과: 검색 품질 10~30% 향상 기대",
        "- 비용: 쿼리당 추가 LLM 호출 1회 (소량의 토큰)"
    ], "Query Rewriting 핵심 포인트")


def demo_hyde():
    """실습 7: HyDE - 가상 문서 임베딩 기법"""
    print("\n" + "="*80)
    print("[7] 실습 7: HyDE - 가상 문서 임베딩 기법")
    print("="*80)
    print("목표: 가상 문서를 생성하여 검색 품질 향상")
    print("핵심: 쿼리 → 가상 답변 → 가상 답변의 임베딩으로 검색")
    
    # HyDE 개념 설명
    print_section_header("HyDE (Hypothetical Document Embedding)란?", "[INFO]")
    print("""
  ┌─────────────────────────────────────────────────────────┐
  │  [IDEA] HyDE의 핵심 아이디어                            │
  │  ─────────────────────────────────────────────────────  │
  │                                                         │
  │  [문제]                                                 │
  │  * 질문과 답변은 표현 방식이 다름                       │
  │  * "RAG가 뭐야?" (질문) vs "RAG는 검색 증강..." (답변)  │
  │  * 질문 임베딩과 답변 임베딩은 벡터 공간에서 다른 위치  │
  │                                                         │
  │  [해결]                                                 │
  │  * LLM으로 "가상의 답변 문서" 생성                      │
  │  * 가상 답변의 임베딩으로 검색                          │
  │  * 답변 스타일 임베딩 → 답변 스타일 문서를 더 잘 찾음   │
  │                                                         │
  │  ─────────────────────────────────────────────────────  │
  │  [FLOW] HyDE 파이프라인                                 │
  │  ─────────────────────────────────────────────────────  │
  │                                                         │
  │  1. 사용자 질문                                         │
  │     │                                                   │
  │     ▼                                                   │
  │  2. LLM으로 가상 답변 생성                              │
  │     │ "RAG가 뭐야?" → "RAG는 Retrieval-Augmented        │
  │     │                 Generation의 약자로..."           │
  │     ▼                                                   │
  │  3. 가상 답변의 임베딩 생성                             │
  │     │                                                   │
  │     ▼                                                   │
  │  4. 가상 답변 임베딩으로 Vector DB 검색                 │
  │     │                                                   │
  │     ▼                                                   │
  │  5. 실제 관련 문서 반환                                 │
  │                                                         │
  │  [!] 중요: 가상 답변 자체는 사용하지 않음!              │
  │      검색을 위한 "프록시"로만 활용                      │
  └─────────────────────────────────────────────────────────┘
    """)
    
    if not os.getenv("OPENAI_API_KEY"):
        print("\n[!] OPENAI_API_KEY 환경변수를 설정해주세요!")
        return
    
    hyde = HyDERetriever()
    
    # 가상 문서 생성 예시
    print_section_header("가상 문서 생성 예시", "[HYDE]")
    
    query = "RAG 시스템에서 청킹이 왜 중요한가요?"
    print(f"\n질문: '{query}'")
    
    print("\n[...] 가상 문서 생성 중...")
    hypothetical_doc = hyde.generate_hypothetical_document(query)
    
    print(f"\n생성된 가상 문서:")
    print(f"{'─'*60}")
    print(hypothetical_doc)
    print(f"{'─'*60}")
    
    print("""
  [!] 이 가상 문서는 실제 정보가 아닙니다!
      하지만 "답변 스타일"의 임베딩을 만들어서
      답변 스타일의 실제 문서를 더 잘 찾을 수 있게 합니다.
    """)
    
    # 일반 검색 vs HyDE 검색 비교
    print_section_header("일반 검색 vs HyDE 검색 비교", "[vs]")
    print("""
  ┌─────────────────────────────────────────────────────────────────────────┐
  │  [CMP] 검색 방식 비교                                                    │
  │  ─────────────────────────────────────────────────────────────────────  │
  │                                                                         │
  │  일반 검색:                                                             │
  │  ┌─────────────────────────────────────────────────────────────────┐   │
  │  │ Query: "청킹이 왜 중요해?"                                       │   │
  │  │    ↓                                                             │   │
  │  │ Query Embedding: [0.1, -0.3, 0.5, ...]  ← 질문 스타일            │   │
  │  │    ↓                                                             │   │
  │  │ 검색 결과: 질문 스타일과 유사한 문서                             │   │
  │  └─────────────────────────────────────────────────────────────────┘   │
  │                                                                         │
  │  HyDE 검색:                                                             │
  │  ┌─────────────────────────────────────────────────────────────────┐   │
  │  │ Query: "청킹이 왜 중요해?"                                       │   │
  │  │    ↓                                                             │   │
  │  │ 가상 답변: "청킹은 RAG에서 중요한데, 그 이유는..."              │   │
  │  │    ↓                                                             │   │
  │  │ HyDE Embedding: [0.2, 0.1, -0.4, ...]  ← 답변 스타일             │   │
  │  │    ↓                                                             │   │
  │  │ 검색 결과: 답변 스타일과 유사한 문서 (더 관련성 높음!)           │   │
  │  └─────────────────────────────────────────────────────────────────┘   │
  │                                                                         │
  │  [효과]                                                                 │
  │  * 질문-답변 스타일 불일치 해소                                         │
  │  * 특히 짧은 질문에서 효과적                                            │
  │  * 검색 Recall 5~15% 향상 보고 (논문 기준)                              │
  └─────────────────────────────────────────────────────────────────────────┘
    """)
    
    # HyDE 장단점
    print_section_header("HyDE 장단점", "[PRO/CON]")
    print("""
  ┌─────────────────────────────────────────────────────────┐
  │  [v] 장점                                               │
  │  ─────────────────────────────────────────────────────  │
  │  * 짧은 쿼리에서 검색 품질 크게 향상                    │
  │  * 질문-문서 스타일 불일치 해소                         │
  │  * 추가 학습 없이 사용 가능                             │
  │  * Query Rewriting과 조합 가능                          │
  │                                                         │
  │  [x] 단점                                               │
  │  ─────────────────────────────────────────────────────  │
  │  * 추가 LLM 호출 필요 (비용, 지연)                      │
  │  * 가상 문서가 잘못되면 검색도 잘못됨                   │
  │  * 이미 명확한 쿼리에서는 효과 미미                     │
  │  * 도메인 특화 질문에서 가상 문서 품질 저하 가능        │
  │                                                         │
  │  [TIP] 언제 사용할까?                                   │
  │  ─────────────────────────────────────────────────────  │
  │  * 짧고 모호한 질문이 많을 때                           │
  │  * 질문과 문서 스타일이 매우 다를 때                    │
  │  * 지연 시간보다 품질이 중요할 때                       │
  │  * Query Rewriting만으로 부족할 때                      │
  └─────────────────────────────────────────────────────────┘
    """)
    
    # 핵심 포인트
    print_key_points([
        "- HyDE: 가상 답변 생성 → 가상 답변 임베딩으로 검색",
        "- 원리: 질문 스타일 → 답변 스타일로 변환",
        "- 효과: 짧은 쿼리에서 Recall 5~15% 향상",
        "- 비용: 쿼리당 LLM 호출 1회 추가",
        "- 조합: Query Rewriting + HyDE 함께 사용 가능"
    ], "HyDE 핵심 포인트")


# ============================================================================
# 메인 실행
# ============================================================================

def main():
    """Chapter 3 실습 실행"""
    print("\n" + "="*80)
    print("[LAB 03 - Chapter 3] 고급 검색 기법")
    print("="*80)
    
    print("\n[LIST] 실습 항목:")
    print("  6. Query Rewriting - 쿼리 개선으로 검색 품질 향상")
    print("  7. HyDE - 가상 문서 임베딩 기법")
    
    try:
        # 6. Query Rewriting
        demo_query_rewriting()
        
        # 7. HyDE
        demo_hyde()
        
        print("\n" + "="*80)
        print("[OK] Chapter 3 완료!")
        print("="*80)
        print("\n[NEXT] 다음 단계:")
        print("   - chapter4_production_ready.py : 프로덕션 준비 (평가, Citation, Streaming)")
        
    except Exception as e:
        print(f"\n[X] 오류 발생: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()


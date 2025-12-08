"""
Advanced RAG - Chapter 2: Re-ranking
Cross-Encoder를 사용한 검색 결과 재정렬

실습 항목:
2. Re-ranking 적용 (Cross-Encoder)
   - 초기 검색 → Cross-Encoder 재정렬
   - Precision 향상
   - Before/After 비교

학습 목표:
- Re-ranking의 필요성 이해
- Cross-Encoder vs Bi-Encoder 차이
- Precision 향상 효과 측정
- 실무 적용 가이드
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass
import math

# Sentence Transformers (Cross-Encoder)
from sentence_transformers import CrossEncoder

# 기본 라이브러리
from dotenv import load_dotenv

# 프로젝트 루트의 .env 파일 로드
project_root = Path(__file__).parent.parent
load_dotenv(dotenv_path=project_root / '.env')

# 공통 유틸리티 import
sys.path.insert(0, str(project_root))
from shared_data import SAMPLE_TEXT, SAMPLE_TEXT_EN

# Chapter 1에서 필요한 클래스 import
from lab03.advanced_chapter1_hybrid_search import (
    SearchResult,
    DocumentProcessor,
    HybridRetriever,
    print_search_result
)

from langchain_openai import OpenAIEmbeddings


@dataclass
class SearchResult:
    """검색 결과 데이터 클래스"""
    content: str
    score: float
    metadata: Dict[str, Any]
    rank: int


class Reranker:
    """Re-ranking 모델 (Cross-Encoder)"""
    
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """
        Args:
            model_name: 리랭킹 모델 이름
            기본값: cross-encoder/ms-marco-MiniLM-L-6-v2 (경량, ~80MB)
            대안: BAAI/bge-reranker-base (고성능, ~500MB)
        """
        print(f"[...] Re-ranker 모델 로딩 중... ({model_name})")
        
        # SSL 인증서 검증 우회 설정
        import ssl
        import warnings
        
        try:
            _create_unverified_https_context = ssl._create_unverified_context
        except AttributeError:
            pass
        else:
            ssl._create_default_https_context = _create_unverified_https_context
        
        os.environ['CURL_CA_BUNDLE'] = ''
        os.environ['REQUESTS_CA_BUNDLE'] = ''
        
        warnings.filterwarnings('ignore', message='Unverified HTTPS request')
        
        try:
            import urllib3
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        except:
            pass
        
        self.model = CrossEncoder(model_name)
        print("[OK] Re-ranker 준비 완료")
    
    def rerank(
        self,
        query: str,
        results: List[SearchResult],
        top_k: int = 5
    ) -> List[SearchResult]:
        """
        검색 결과 재순위화
        
        Args:
            query: 검색 쿼리
            results: 초기 검색 결과
            top_k: 반환할 상위 결과 수
        """
        if not results:
            return []
        
        # 쿼리-문서 쌍 생성
        pairs = [[query, result.content] for result in results]
        
        # Cross-Encoder 점수 계산 (로짓: -∞ ~ +∞)
        raw_scores = self.model.predict(pairs)
        
        # 로짓을 확률로 변환 (sigmoid: 0~1)
        normalized_scores = [1 / (1 + math.exp(-s)) for s in raw_scores]
        
        # 재정렬
        reranked = []
        for result, norm_score, raw_score in zip(results, normalized_scores, raw_scores):
            reranked.append(SearchResult(
                content=result.content,
                score=float(norm_score),
                metadata={**result.metadata, "raw_rerank_score": float(raw_score)},
                rank=0
            ))
        
        reranked.sort(key=lambda x: x.score, reverse=True)
        
        for rank, result in enumerate(reranked[:top_k]):
            result.rank = rank + 1
        
        return reranked[:top_k]


def run_single_query_test(
    query: str,
    retriever: 'HybridRetriever',
    reranker: 'Reranker',
    expected_keywords: List[str] = None
) -> Dict[str, Any]:
    """단일 쿼리에 대한 Re-ranking 테스트 실행"""
    
    print(f"\n[*] 쿼리: '{query}'")
    if expected_keywords:
        print(f"    기대 키워드: {', '.join(expected_keywords)}")
    
    # Before: 하이브리드 검색만
    print(f"\n{'─'*60}")
    print("[BEFORE] 하이브리드 검색 (Re-ranking 없음)")
    print(f"{'─'*60}")
    results_before = retriever.hybrid_search(query, k=10, alpha=0.5)
    
    print(f"검색 결과: {len(results_before)}개")
    for i, result in enumerate(results_before[:3], 1):
        print_search_result(result, i, show_full=(i==1))
    
    # After: Re-ranking 적용
    print(f"\n{'─'*60}")
    print("[AFTER] Re-ranking 적용 (Cross-Encoder)")
    print(f"{'─'*60}")
    results_after = reranker.rerank(query, results_before, top_k=5)
    
    print(f"Re-ranked 결과: {len(results_after)}개")
    for i, result in enumerate(results_after[:3], 1):
        print_search_result(result, i, show_full=(i==1))
    
    # 키워드 매칭 분석 (정답 포함 여부)
    def count_keyword_matches(content: str, keywords: List[str]) -> int:
        if not keywords:
            return 0
        return sum(1 for kw in keywords if kw.lower() in content.lower())
    
    before_top1_matches = count_keyword_matches(results_before[0].content, expected_keywords) if expected_keywords else 0
    after_top1_matches = count_keyword_matches(results_after[0].content, expected_keywords) if expected_keywords else 0
    
    before_chunk = results_before[0].metadata.get('chunk_id', -1)
    after_chunk = results_after[0].metadata.get('chunk_id', -1)
    
    return {
        "query": query,
        "before_chunk": before_chunk,
        "after_chunk": after_chunk,
        "before_score": results_before[0].score,
        "after_score": results_after[0].score,
        "raw_rerank_score": results_after[0].metadata.get('raw_rerank_score', 0),
        "before_matches": before_top1_matches,
        "after_matches": after_top1_matches,
        "expected_keywords": expected_keywords or [],
        "ranking_changed": before_chunk != after_chunk,
        "improved": after_top1_matches > before_top1_matches,
        "degraded": after_top1_matches < before_top1_matches
    }


def experiment_reranking():
    """실습 2: Re-ranking 효과 측정 - 영어 vs 한국어 비교"""
    print("\n" + "="*80)
    print("[2] 실습 2: Re-ranking - 언어별 성공/실패 케이스 비교")
    print("="*80)
    print("목표: ms-marco 모델이 영어에서는 효과적이고, 한국어에서는 비효과적임을 이해")
    
    # Re-ranker 초기화 (공통 사용)
    reranker = Reranker()
    
    processor = DocumentProcessor(chunk_size=500, chunk_overlap=50)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    # ================================================================
    # 케이스 1: 영어 문서 + 영어 쿼리 (성공 케이스)
    # ================================================================
    print("\n" + "="*80)
    print("[케이스 1] ✅ 성공 케이스: 영어 문서 + 영어 쿼리")
    print("="*80)
    print("이유: ms-marco 모델은 영어 웹 검색 데이터로 학습됨")
    
    print(f"\n[DOC] 영어 문서 처리 중...")
    docs_en = processor.create_chunks(SAMPLE_TEXT_EN, metadata={"source": "AI_Guide_EN"})
    print(f"  - 생성된 청크: {len(docs_en)}개")
    
    retriever_en = HybridRetriever(
        documents=docs_en,
        embeddings=embeddings,
        collection_name="rerank_en"
    )
    
    result_en = run_single_query_test(
        query="How does RAG prevent hallucination in language models?",
        retriever=retriever_en,
        reranker=reranker,
        expected_keywords=["Hallucination", "RAG", "Preventing", "facts"]
    )
    
    # ================================================================
    # 케이스 2: 한국어 문서 + 한국어 쿼리 (실패 케이스)
    # ================================================================
    print("\n" + "="*80)
    print("[케이스 2] ❌ 실패 케이스: 한국어 문서 + 한국어 쿼리")
    print("="*80)
    print("이유: ms-marco 모델은 한국어를 학습하지 않음 (언어 불일치)")
    
    print(f"\n[DOC] 한국어 문서 처리 중...")
    docs_ko = processor.create_chunks(SAMPLE_TEXT, metadata={"source": "AI_가이드_KO"})
    print(f"  - 생성된 청크: {len(docs_ko)}개")
    
    retriever_ko = HybridRetriever(
        documents=docs_ko,
        embeddings=embeddings,
        collection_name="rerank_ko"
    )
    
    result_ko = run_single_query_test(
        query="RAG에서 환각(Hallucination)을 방지하는 방법은?",
        retriever=retriever_ko,
        reranker=reranker,
        expected_keywords=["환각", "Hallucination", "방지", "RAG"]
    )
    
    # ================================================================
    # 종합 분석
    # ================================================================
    print("\n" + "="*80)
    print("[종합 분석] 언어별 Re-ranking 효과 비교")
    print("="*80)
    
    # 영어 케이스 분석
    print(f"\n{'─'*60}")
    print("케이스 1: ✅ 영어 (ms-marco 학습 언어)")
    print(f"{'─'*60}")
    print(f"  쿼리: {result_en['query']}")
    print(f"  기대 키워드: {', '.join(result_en['expected_keywords'])}")
    print(f"\n  Before 1위: 청크 #{result_en['before_chunk'] + 1}")
    print(f"    - 키워드 매칭: {result_en['before_matches']}/{len(result_en['expected_keywords'])}개")
    print(f"  After 1위:  청크 #{result_en['after_chunk'] + 1}")
    print(f"    - 키워드 매칭: {result_en['after_matches']}/{len(result_en['expected_keywords'])}개")
    print(f"    - Cross-Encoder 로짓: {result_en['raw_rerank_score']:.4f}")
    
    if result_en['improved']:
        print(f"\n  ✅ 결과: Re-ranking으로 정확도 향상!")
    elif result_en['degraded']:
        print(f"\n  ⚠️ 결과: Re-ranking으로 정확도 하락")
    elif result_en['before_matches'] == result_en['after_matches'] and result_en['before_matches'] > 0:
        print(f"\n  ✅ 결과: 정확도 유지 (이미 좋은 결과)")
    else:
        print(f"\n  ➡️ 결과: 정확도 변화 없음")
    
    # 한국어 케이스 분석
    print(f"\n{'─'*60}")
    print("케이스 2: ❌ 한국어 (ms-marco 미학습 언어)")
    print(f"{'─'*60}")
    print(f"  쿼리: {result_ko['query']}")
    print(f"  기대 키워드: {', '.join(result_ko['expected_keywords'])}")
    print(f"\n  Before 1위: 청크 #{result_ko['before_chunk'] + 1}")
    print(f"    - 키워드 매칭: {result_ko['before_matches']}/{len(result_ko['expected_keywords'])}개")
    print(f"  After 1위:  청크 #{result_ko['after_chunk'] + 1}")
    print(f"    - 키워드 매칭: {result_ko['after_matches']}/{len(result_ko['expected_keywords'])}개")
    print(f"    - Cross-Encoder 로짓: {result_ko['raw_rerank_score']:.4f}")
    
    if result_ko['improved']:
        print(f"\n  ✅ 결과: Re-ranking으로 정확도 향상!")
    elif result_ko['degraded']:
        print(f"\n  ❌ 결과: Re-ranking으로 오히려 정확도 하락!")
    else:
        print(f"\n  ➡️ 결과: 정확도 변화 없음")
    
    # ================================================================
    # 핵심 인사이트
    # ================================================================
    print("\n" + "="*80)
    print("[핵심 인사이트] 모델과 데이터 언어의 일치가 중요!")
    print("="*80)
    
    print("""
  ┌─────────────────────────────────────────────────────────────────┐
  │  [현재 사용 모델]                                               │
  │  cross-encoder/ms-marco-MiniLM-L-6-v2                           │
  │  → MS MARCO 영어 웹 검색 데이터셋으로 학습됨                     │
  │  → 영어 문서에서만 효과적!                                      │
  └─────────────────────────────────────────────────────────────────┘
  
  ┌─────────────────────────────────────────────────────────────────┐
  │  ✅ 영어에서 Re-ranking이 효과적인 이유                          │
  │  ─────────────────────────────────────────────────────────────  │
  │  • 모델이 영어로 학습됨 → 영어 문서의 관련성을 정확히 판단       │
  │  • Query-Document 간의 의미적 관계를 제대로 이해                 │
  │  • Cross-Encoder의 Attention이 영어 토큰에 최적화됨             │
  └─────────────────────────────────────────────────────────────────┘
  
  ┌─────────────────────────────────────────────────────────────────┐
  │  ❌ 한국어에서 Re-ranking이 실패하는 이유                        │
  │  ─────────────────────────────────────────────────────────────  │
  │  • 모델이 한국어를 학습하지 않음 → 한국어 토큰을 이해 못함       │
  │  • 의미적 관련성 판단 불가 → 무작위에 가까운 점수 부여           │
  │  • 오히려 잘못된 문서를 1위로 올릴 수 있음!                      │
  └─────────────────────────────────────────────────────────────────┘
    """)
    
    # ================================================================
    # 실무 가이드
    # ================================================================
    print("\n" + "="*80)
    print("[실무 가이드] 언어별 Re-ranker 모델 선택")
    print("="*80)
    
    print("""
  ┌─────────────────────────────────────────────────────────────────┐
  │  [언어별 권장 모델]                                             │
  │  ─────────────────────────────────────────────────────────────  │
  │                                                                 │
  │  📌 영어 문서                                                    │
  │     → cross-encoder/ms-marco-MiniLM-L-6-v2 (~80MB) ✅ 현재 사용  │
  │     → cross-encoder/ms-marco-MiniLM-L-12-v2 (~130MB)            │
  │                                                                 │
  │  📌 한국어 문서                                                  │
  │     → Dongjin-kr/ko-reranker (~500MB)                           │
  │     → BAAI/bge-reranker-v2-m3 (다국어, ~1.5GB)                  │
  │                                                                 │
  │  📌 다국어 문서                                                  │
  │     → BAAI/bge-reranker-v2-m3                                   │
  │     → mmarco-mMiniLMv2-L12-H384-v1                              │
  └─────────────────────────────────────────────────────────────────┘
  
  ┌─────────────────────────────────────────────────────────────────┐
  │  [핵심 교훈]                                                    │
  │  ─────────────────────────────────────────────────────────────  │
  │                                                                 │
  │  1️⃣  Re-ranking 모델의 학습 언어와 문서 언어를 일치시켜라!       │
  │                                                                 │
  │  2️⃣  모델 점수가 높다고 정확한 게 아니다!                        │
  │     → 한국어에서도 높은 점수가 나오지만 틀린 결과                │
  │     → 반드시 키워드 매칭 등으로 실제 정확도 검증 필요            │
  │                                                                 │
  │  3️⃣  Re-ranking은 만능이 아니다!                                │
  │     → 모델-데이터 불일치 시 오히려 성능 저하                     │
  │     → 도입 전 반드시 A/B 테스트 수행                             │
  └─────────────────────────────────────────────────────────────────┘
  
  ┌─────────────────────────────────────────────────────────────────┐
  │  [권장 파이프라인]                                              │
  │  ─────────────────────────────────────────────────────────────  │
  │                                                                 │
  │  1단계: 문서 언어 감지                                          │
  │  2단계: 언어에 맞는 Re-ranker 모델 선택                         │
  │  3단계: Bi-Encoder로 후보 추출 (Recall 확보)                    │
  │  4단계: Cross-Encoder로 재정렬 (Precision 향상)                 │
  │  5단계: A/B 테스트로 효과 검증                                  │
  │                                                                 │
  │  💡 핵심: 모델과 데이터의 언어가 일치해야 Re-ranking 효과 있음!  │
  └─────────────────────────────────────────────────────────────────┘
    """)


def main():
    """Chapter 2 실행"""
    print("\n" + "="*80)
    print("[Advanced RAG - Chapter 2] Re-ranking")
    print("="*80)
    
    print("\n[LIST] 실습 항목:")
    print("  2. Re-ranking 적용 (Cross-Encoder)")
    
    try:
        experiment_reranking()
        
        print("\n" + "="*80)
        print("[OK] Chapter 2 완료!")
        print("="*80)
        print("\n[NEXT] 다음 단계:")
        print("   - advanced_chapter3_advanced_patterns.py : 고급 패턴")
        
    except Exception as e:
        print(f"\n[X] 오류 발생: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()


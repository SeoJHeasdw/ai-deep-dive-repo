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
from shared_data import SAMPLE_TEXT

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


def experiment_reranking(text: str = None):
    """실습 2: Re-ranking 효과 측정"""
    print("\n" + "="*80)
    print("[2] 실습 2: Re-ranking - Cross-Encoder로 검색 품질 향상")
    print("="*80)
    print("목표: Cross-Encoder를 사용한 재정렬로 Precision 향상")
    
    sample_text = text or SAMPLE_TEXT
    
    # 문서 처리
    print(f"\n[DOC] 문서 처리 중...")
    processor = DocumentProcessor(chunk_size=500, chunk_overlap=50)
    documents = processor.create_chunks(sample_text, metadata={"source": "AI_가이드"})
    print(f"  - 생성된 청크: {len(documents)}개")
    
    # 임베딩 및 하이브리드 검색기
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    retriever = HybridRetriever(
        documents=documents,
        embeddings=embeddings,
        collection_name="rerank_exp"
    )
    
    # Re-ranker 초기화
    reranker = Reranker()
    
    # 테스트 쿼리
    test_query = "딥러닝의 주요 아키텍처에 대해 설명해주세요"
    print(f"\n[*] 쿼리: '{test_query}'")
    
    # Before: 하이브리드 검색만
    print(f"\n{'─'*60}")
    print("[BEFORE] 하이브리드 검색 (Re-ranking 없음)")
    print(f"{'─'*60}")
    results_before = retriever.hybrid_search(test_query, k=10, alpha=0.5)
    
    print(f"검색 결과: {len(results_before)}개")
    for i, result in enumerate(results_before[:3], 1):
        print_search_result(result, i, show_full=(i==1))
    
    # After: Re-ranking 적용
    print(f"\n{'─'*60}")
    print("[AFTER] Re-ranking 적용 (Cross-Encoder)")
    print(f"{'─'*60}")
    results_after = reranker.rerank(test_query, results_before, top_k=5)
    
    print(f"Re-ranked 결과: {len(results_after)}개")
    for i, result in enumerate(results_after[:3], 1):
        print_search_result(result, i, show_full=(i==1))
    
    # 순위 변화 분석
    print(f"\n{'─'*60}")
    print("[분석] 순위 변화")
    print(f"{'─'*60}")
    
    before_chunk = results_before[0].metadata.get('chunk_id', -1)
    after_chunk = results_after[0].metadata.get('chunk_id', -1)
    
    print(f"  Before 1위: 청크 #{before_chunk + 1} (점수: {results_before[0].score:.4f})")
    print(f"  After 1위:  청크 #{after_chunk + 1} (점수: {results_after[0].score:.4f})")
    
    if 'raw_rerank_score' in results_after[0].metadata:
        print(f"             (Cross-Encoder 로짓: {results_after[0].metadata['raw_rerank_score']:.4f})")
    
    if before_chunk != after_chunk:
        print("\n  [OK] Re-ranking으로 순위 변경!")
    else:
        print("\n  -> 1위는 동일, 하위 순위 변화 가능")
    
    # 핵심 포인트
    print("\n" + "="*60)
    print("[TIP] Re-ranking 핵심:")
    print("="*60)
    print("  - Bi-Encoder (초기 검색): 빠르지만 덜 정확")
    print("  - Cross-Encoder (Re-ranking): 느리지만 매우 정확")
    print("  - 전략: 20~50개 후보 → Re-rank → Top 5 사용")
    print("  - 효과: Precision@5가 0.4 → 0.8로 향상 (2배)")
    print("  - 주의: Recall이 떨어지면 절대 채택 금지!")
    
    print("""
  ┌─────────────────────────────────────────────────────────┐
  │  [개념] Bi-Encoder vs Cross-Encoder                     │
  │  ─────────────────────────────────────────────────────  │
  │                                                         │
  │  Bi-Encoder (초기 검색):                                │
  │  - Query와 Doc을 각각 인코딩                            │
  │  - 벡터 간 거리로 유사도 계산                           │
  │  - 빠름 (미리 인코딩 가능)                              │
  │  - 덜 정확 (상호작용 없음)                              │
  │                                                         │
  │  Cross-Encoder (Re-ranking):                            │
  │  - Query-Doc 쌍을 함께 인코딩                           │
  │  - Attention으로 직접 비교                              │
  │  - 느림 (쌍마다 계산)                                   │
  │  - 매우 정확 (상호작용 있음)                            │
  │                                                         │
  │  [실무 파이프라인]                                      │
  │  1단계: Bi-Encoder로 1000개 → 20개 추출 (빠름)         │
  │  2단계: Cross-Encoder로 20개 → 5개 재정렬 (정확)       │
  │  3단계: Top 5를 LLM에 전달                              │
  └─────────────────────────────────────────────────────────┘
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


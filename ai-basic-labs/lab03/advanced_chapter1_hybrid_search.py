"""
Advanced RAG - Chapter 1: 하이브리드 검색
Sparse (BM25) + Dense (Vector) 검색 결합

실습 항목:
1. Sparse + Dense 하이브리드 검색
   - BM25 (키워드 매칭)
   - Vector DB (의미적 유사도)
   - 가중 결합 (alpha 파라미터)

학습 목표:
- Sparse와 Dense 검색의 장단점 이해
- 하이브리드 검색의 원리 파악
- 한글 토큰화 문제 인식
- 검색 방법별 결과 비교
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

# LangChain
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

# Sparse 검색 (BM25)
from rank_bm25 import BM25Okapi

# 기본 라이브러리
import chromadb
from dotenv import load_dotenv
import numpy as np
import pdfplumber

# 프로젝트 루트의 .env 파일 로드
project_root = Path(__file__).parent.parent
load_dotenv(dotenv_path=project_root / '.env')

# 공통 유틸리티 import를 위한 경로 추가
sys.path.insert(0, str(project_root))

# 공통 데이터 임포트
from shared_data import SAMPLE_TEXT, MIN_TEXT_LENGTH, get_sample_or_document_text


@dataclass
class SearchResult:
    """검색 결과 데이터 클래스"""
    content: str
    score: float
    metadata: Dict[str, Any]
    rank: int


class DocumentProcessor:
    """문서 처리 및 청킹"""
    
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def load_pdf(self, file_path: str) -> str:
        """PDF 파일 로드"""
        text = ""
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        return text
    
    def chunk_text(self, text: str) -> List[str]:
        """텍스트를 청크로 분할"""
        chunks = []
        start = 0
        text_length = len(text)
        
        while start < text_length:
            end = start + self.chunk_size
            
            if end >= text_length:
                chunk = text[start:].strip()
                if chunk:
                    chunks.append(chunk)
                break
            
            # 문장 경계 찾기
            best_end = -1
            
            # 단락 끝
            double_newline = text.rfind('\n\n', start, end + 50)
            if double_newline != -1:
                best_end = double_newline + 2
            
            # 문장 끝
            if best_end == -1:
                for i in range(end, max(start, end - 100), -1):
                    if i < text_length - 1 and text[i] == '.' and text[i+1] == '\n':
                        best_end = i + 2
                        break
            
            # 마침표 + 공백
            if best_end == -1:
                period_space = text.rfind('. ', start, end + 30)
                if period_space != -1:
                    best_end = period_space + 2
            
            # 줄바꿈
            if best_end == -1:
                newline = text.rfind('\n', start, end + 20)
                if newline != -1:
                    best_end = newline + 1
            
            # 공백
            if best_end == -1:
                space = text.rfind(' ', start, end)
                if space != -1 and space > start + self.chunk_size // 2:
                    best_end = space + 1
            
            # 강제로 자르기
            if best_end == -1:
                best_end = end
            
            chunk = text[start:best_end].strip()
            if chunk:
                chunks.append(chunk)
            
            next_start = best_end - self.chunk_overlap
            if next_start <= start:
                next_start = best_end
            
            start = next_start
        
        return chunks
    
    def create_chunks(self, text: str, metadata: Optional[Dict] = None) -> List[Document]:
        """텍스트를 청크로 분할하여 Document 객체 생성"""
        chunks = self.chunk_text(text)
        
        documents = []
        for i, chunk in enumerate(chunks):
            doc_metadata = metadata.copy() if metadata else {}
            doc_metadata.update({
                "chunk_id": i,
                "chunk_size": len(chunk),
                "total_chunks": len(chunks)
            })
            documents.append(Document(page_content=chunk, metadata=doc_metadata))
        
        return documents


class HybridRetriever:
    """Sparse (BM25) + Dense (Vector) 하이브리드 검색"""
    
    # 한글 조사 패턴 (간단 버전)
    KOREAN_PARTICLES = [
        '이란', '이란?', '란', '란?', '은', '는', '이', '가', '을', '를',
        '의', '에', '에서', '으로', '로', '와', '과', '도', '만', '까지',
        '부터', '이다', '입니다', '인가', '인가?', '인지', '하는', '되는'
    ]
    
    @staticmethod
    def tokenize_korean(text: str) -> List[str]:
        """
        간단한 한글 토큰화 (교육용)
        
        ⚠️ 한계:
        - 실제 형태소 분석이 아님 (단순 규칙 기반 조사 제거)
        - 복잡한 어미 처리 불가
        
        🔧 실무 권장:
        - KoNLPy (Mecab, Okt, Komoran 등) 형태소 분석기 사용
        """
        import re
        
        # 구두점을 공백으로
        text = re.sub(r'[.,!?;:()"\'\[\]{}]', ' ', text)
        
        # 공백으로 분리
        tokens = text.split()
        
        # 조사 제거 시도
        cleaned_tokens = []
        for token in tokens:
            cleaned = token
            for particle in sorted(HybridRetriever.KOREAN_PARTICLES, key=len, reverse=True):
                if cleaned.endswith(particle) and len(cleaned) > len(particle):
                    cleaned = cleaned[:-len(particle)]
                    break
            if cleaned:
                cleaned_tokens.append(cleaned)
        
        return cleaned_tokens
    
    def __init__(
        self,
        documents: List[Document],
        embeddings: OpenAIEmbeddings,
        persist_directory: str = "./chroma_db",
        collection_name: str = "hybrid_search"
    ):
        self.documents = documents
        self.embeddings = embeddings
        
        # Dense 검색: Vector DB (Chroma)
        print(f"Dense 검색 준비 중... (컬렉션: {collection_name})")
        
        # 기존 컬렉션 삭제
        try:
            chroma_client = chromadb.PersistentClient(path=persist_directory)
            chroma_client.delete_collection(name=collection_name)
        except:
            pass
        
        self.vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            persist_directory=persist_directory,
            collection_name=collection_name
        )
        
        # Sparse 검색: BM25
        print("Sparse 검색 준비 중... (BM25 + 한글 토큰화)")
        self.corpus = [doc.page_content for doc in documents]
        self.tokenized_corpus = [self.tokenize_korean(doc) for doc in self.corpus]
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        
        print(f"하이브리드 검색 준비 완료 (문서 수: {len(documents)})")
    
    def sparse_search(self, query: str, k: int = 10) -> List[SearchResult]:
        """BM25 Sparse 검색"""
        tokenized_query = self.tokenize_korean(query)
        scores = self.bm25.get_scores(tokenized_query)
        
        k = min(k, len(self.documents))
        top_indices = np.argsort(scores)[::-1]
        
        results = []
        rank = 1
        for idx in top_indices:
            score = float(scores[idx])
            
            if score == 0:
                continue
            
            results.append(SearchResult(
                content=self.documents[idx].page_content,
                score=score,
                metadata={**self.documents[idx].metadata, "matched_tokens": tokenized_query},
                rank=rank
            ))
            rank += 1
            
            if len(results) >= k:
                break
        
        if not results:
            results.append(SearchResult(
                content=f"[키워드 '{' '.join(tokenized_query)}' 매칭 없음]",
                score=0.0,
                metadata={"no_match": True},
                rank=1
            ))
        
        return results
    
    def dense_search(self, query: str, k: int = 10) -> List[SearchResult]:
        """Vector DB Dense 검색"""
        k = min(k, len(self.documents))
        docs_with_scores = self.vectorstore.similarity_search_with_score(query, k=k)
        
        results = []
        for rank, (doc, score) in enumerate(docs_with_scores):
            # L2 거리를 0~1 점수로 변환
            distance_score = 1 / (1 + score)
            results.append(SearchResult(
                content=doc.page_content,
                score=float(distance_score),
                metadata={**doc.metadata, "raw_distance": float(score)},
                rank=rank + 1
            ))
        
        return results
    
    def hybrid_search(
        self,
        query: str,
        k: int = 10,
        alpha: float = 0.5
    ) -> List[SearchResult]:
        """
        하이브리드 검색 (Sparse + Dense)
        
        Args:
            query: 검색 쿼리
            k: 반환할 결과 수
            alpha: Dense 가중치 (0~1, 1-alpha가 Sparse 가중치)
        """
        # 두 검색 실행
        sparse_results = self.sparse_search(query, k=k*2)
        dense_results = self.dense_search(query, k=k*2)
        
        # 정규화 (0~1 범위로)
        def normalize_scores(results: List[SearchResult]) -> List[SearchResult]:
            if not results or results[0].metadata.get("no_match"):
                return results
            
            scores = [r.score for r in results]
            min_score = min(scores)
            max_score = max(scores)
            
            if max_score == min_score:
                for r in results:
                    r.score = 1.0
                return results
            
            for r in results:
                r.score = (r.score - min_score) / (max_score - min_score)
            
            return results
        
        sparse_results = normalize_scores(sparse_results)
        dense_results = normalize_scores(dense_results)
        
        # 결합
        combined_scores = {}
        
        for result in sparse_results:
            if result.metadata.get("no_match"):
                continue
            content = result.content
            combined_scores[content] = {
                "sparse": result.score * (1 - alpha),
                "dense": 0.0,
                "metadata": result.metadata
            }
        
        for result in dense_results:
            content = result.content
            if content in combined_scores:
                combined_scores[content]["dense"] = result.score * alpha
            else:
                combined_scores[content] = {
                    "sparse": 0.0,
                    "dense": result.score * alpha,
                    "metadata": result.metadata
                }
        
        # 최종 점수 계산 및 정렬
        final_results = []
        for content, scores in combined_scores.items():
            final_score = scores["sparse"] + scores["dense"]
            final_results.append(SearchResult(
                content=content,
                score=final_score,
                metadata={
                    **scores["metadata"],
                    "sparse_score": scores["sparse"],
                    "dense_score": scores["dense"]
                },
                rank=0
            ))
        
        final_results.sort(key=lambda x: x.score, reverse=True)
        
        for rank, result in enumerate(final_results[:k], 1):
            result.rank = rank
        
        return final_results[:k]


def format_chunk(content: str, indent: str = "      ") -> str:
    """청크 내용을 보기 좋게 포맷팅"""
    lines = content.strip().split('\n')
    formatted_lines = []
    for line in lines:
        line = line.strip()
        if line:
            formatted_lines.append(f"{indent}{line}")
    return '\n'.join(formatted_lines)


def print_search_result(result: SearchResult, index: int, show_full: bool = False):
    """검색 결과 출력"""
    print(f"\n  [{index}] 점수: {result.score:.4f} | 청크 #{result.metadata.get('chunk_id', -1) + 1}")
    
    if show_full:
        print(f"  {'─'*50}")
        lines = result.content.strip().split('\n')
        for line in lines[:10]:
            print(f"      {line}")
        if len(lines) > 10:
            print(f"      ... ({len(lines) - 10}줄 더 있음)")
        print(f"  {'─'*50}")
    else:
        preview = result.content.replace('\n', ' ')[:100]
        print(f"      {preview}...")


def experiment_hybrid_search(text: str = None):
    """실습 1: 하이브리드 검색"""
    print("\n" + "="*80)
    print("[1] 실습 1: Sparse + Dense 하이브리드 검색")
    print("="*80)
    print("목표: BM25(키워드) + Vector(의미)를 결합하여 검색 품질 향상")
    
    sample_text = text or SAMPLE_TEXT
    
    # 문서 처리
    print(f"\n[DOC] 문서 처리 중...")
    processor = DocumentProcessor(chunk_size=500, chunk_overlap=50)
    documents = processor.create_chunks(sample_text, metadata={"source": "AI_가이드"})
    print(f"  - 생성된 청크: {len(documents)}개")
    
    # 임베딩 모델
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    # 하이브리드 검색기 초기화
    retriever = HybridRetriever(
        documents=documents,
        embeddings=embeddings,
        collection_name="hybrid_exp"
    )
    
    # 테스트 쿼리
    test_query = "딥러닝의 주요 아키텍처는 무엇인가요?"
    print(f"\n[*] 쿼리: '{test_query}'")
    
    # 1. Sparse 검색만
    print(f"\n{'─'*60}")
    print("[>] Sparse 검색 (BM25)")
    print(f"{'─'*60}")
    sparse_results = retriever.sparse_search(test_query, k=3)
    for i, result in enumerate(sparse_results, 1):
        print_search_result(result, i, show_full=(i==1))
    
    # 2. Dense 검색만
    print(f"\n{'─'*60}")
    print("[>] Dense 검색 (Vector DB)")
    print(f"{'─'*60}")
    dense_results = retriever.dense_search(test_query, k=3)
    for i, result in enumerate(dense_results, 1):
        print_search_result(result, i, show_full=(i==1))
    
    # 3. 하이브리드 검색
    print(f"\n{'─'*60}")
    print("[>] 하이브리드 검색 (alpha=0.5)")
    print(f"{'─'*60}")
    hybrid_results = retriever.hybrid_search(test_query, k=3, alpha=0.5)
    for i, result in enumerate(hybrid_results, 1):
        print_search_result(result, i, show_full=(i==1))
    
    # 핵심 포인트
    print("\n" + "="*60)
    print("[TIP] 하이브리드 검색 핵심:")
    print("="*60)
    print("  - Sparse (BM25): 키워드 매칭 (정확한 용어 검색에 강함)")
    print("  - Dense (Vector): 의미적 유사도 (동의어, 유사 표현에 강함)")
    print("  - Hybrid: 두 방법의 장점 결합")
    print("  - alpha: Dense 가중치 (0.5 = 50:50)")


def main():
    """Chapter 1 실행"""
    print("\n" + "="*80)
    print("[Advanced RAG - Chapter 1] 하이브리드 검색")
    print("="*80)
    
    print("\n[LIST] 실습 항목:")
    print("  1. Sparse + Dense 하이브리드 검색")
    
    try:
        experiment_hybrid_search()
        
        print("\n" + "="*80)
        print("[OK] Chapter 1 완료!")
        print("="*80)
        print("\n[NEXT] 다음 단계:")
        print("   - advanced_chapter2_reranking.py : Re-ranking (Cross-Encoder)")
        
    except Exception as e:
        print(f"\n[X] 오류 발생: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()


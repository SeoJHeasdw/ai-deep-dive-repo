"""
Advanced RAG - Chapter 3: 고급 검색 패턴
Multi-hop, Chunk Size, Context Window 관리

실습 항목:
3. Multi-hop 검색 (다단계 추론)
4. Chunk Size 최적화
5. Context Window 관리

학습 목표:
- Multi-hop 검색으로 복잡한 질문 처리
- 최적 Chunk Size 결정 방법
- Context Window 관리 전략
- 실무 적용 가이드
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
import tiktoken
import time

# LangChain
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from openai import OpenAI

# 기본 라이브러리
from dotenv import load_dotenv

# 프로젝트 루트의 .env 파일 로드
project_root = Path(__file__).parent.parent
load_dotenv(dotenv_path=project_root / '.env')

# 공통 유틸리티 import
sys.path.insert(0, str(project_root))
from shared_data import SAMPLE_TEXT

# 이전 Chapter에서 import
from lab03.advanced_chapter1_hybrid_search import (
    SearchResult,
    DocumentProcessor,
    HybridRetriever,
    print_search_result,
    format_chunk
)
from lab03.advanced_chapter2_reranking import Reranker


@dataclass
class SearchResult:
    """검색 결과 데이터 클래스"""
    content: str
    score: float
    metadata: Dict[str, Any]
    rank: int


class MultiHopRetriever:
    """Multi-hop 질의: 두 단계 검색으로 답 찾기"""
    
    def __init__(
        self,
        retriever: HybridRetriever,
        reranker: Optional[Reranker] = None,
        llm: Optional[ChatOpenAI] = None
    ):
        """
        Args:
            retriever: 하이브리드 검색기
            reranker: 리랭커 (선택)
            llm: LLM 모델 (쿼리 분해용)
        """
        self.retriever = retriever
        self.reranker = reranker
        
        if llm is None:
            import httpx
            http_client = httpx.Client(verify=False)
            self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, http_client=http_client)
        else:
            self.llm = llm
        
        self.decompose_prompt = ChatPromptTemplate.from_template(
            """다음 질문을 두 개의 하위 질문으로 분해하세요.
첫 번째 질문의 답변을 바탕으로 두 번째 질문에 답할 수 있어야 합니다.

원본 질문: {question}

다음 형식으로 답변하세요:
1. [첫 번째 하위 질문]
2. [두 번째 하위 질문]

하위 질문:"""
        )
    
    def decompose_query(self, query: str) -> List[str]:
        """복잡한 쿼리를 하위 쿼리로 분해"""
        print(f"\n[>>>] 쿼리 분해 중: {query}")
        
        chain = self.decompose_prompt | self.llm
        response = chain.invoke({"question": query})
        
        lines = response.content.strip().split('\n')
        sub_queries = []
        for line in lines:
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith('-')):
                sub_query = line.split('.', 1)[-1].strip() if '.' in line else line[1:].strip()
                sub_queries.append(sub_query)
        
        print(f"  -> 하위 질문 1: {sub_queries[0] if len(sub_queries) > 0 else 'N/A'}")
        print(f"  -> 하위 질문 2: {sub_queries[1] if len(sub_queries) > 1 else 'N/A'}")
        
        return sub_queries
    
    def multi_hop_search(
        self,
        query: str,
        k_per_hop: int = 5,
        use_reranker: bool = True
    ) -> Tuple[List[SearchResult], Dict[str, Any]]:
        """
        Multi-hop 검색 수행
        
        Args:
            query: 원본 쿼리
            k_per_hop: 각 홉당 검색 결과 수
            use_reranker: 리랭커 사용 여부
        
        Returns:
            (최종 검색 결과, 메타데이터)
        """
        metadata = {"sub_queries": [], "hop_results": []}
        
        # 1단계: 쿼리 분해
        sub_queries = self.decompose_query(query)
        metadata["sub_queries"] = sub_queries
        
        if len(sub_queries) < 2:
            print("[!] 쿼리 분해 실패, 일반 검색으로 대체")
            results = self.retriever.hybrid_search(query, k=k_per_hop)
            if use_reranker and self.reranker:
                results = self.reranker.rerank(query, results, top_k=k_per_hop)
            return results, metadata
        
        # 2단계: 첫 번째 홉
        print(f"\n[>>>] Hop 1 검색 중...")
        hop1_results = self.retriever.hybrid_search(sub_queries[0], k=k_per_hop)
        if use_reranker and self.reranker:
            hop1_results = self.reranker.rerank(sub_queries[0], hop1_results, top_k=k_per_hop)
        
        metadata["hop_results"].append({
            "query": sub_queries[0],
            "num_results": len(hop1_results)
        })
        
        # 3단계: 두 번째 홉 (첫 홉 결과를 컨텍스트로 사용)
        print(f"[>>>] Hop 2 검색 중...")
        hop1_context = "\n\n".join([r.content[:200] + "..." for r in hop1_results[:3]])
        enhanced_query = f"{sub_queries[1]}\n\n참고 정보:\n{hop1_context}"
        
        hop2_results = self.retriever.hybrid_search(enhanced_query, k=k_per_hop)
        if use_reranker and self.reranker:
            hop2_results = self.reranker.rerank(sub_queries[1], hop2_results, top_k=k_per_hop)
        
        metadata["hop_results"].append({
            "query": sub_queries[1],
            "num_results": len(hop2_results)
        })
        
        # 4단계: 결과 병합 (중복 제거)
        combined = {}
        for result in hop1_results + hop2_results:
            if result.content not in combined:
                combined[result.content] = result
            else:
                if result.score > combined[result.content].score:
                    combined[result.content] = result
        
        final_results = sorted(combined.values(), key=lambda x: x.score, reverse=True)[:k_per_hop]
        
        for rank, result in enumerate(final_results):
            result.rank = rank + 1
        
        print(f"[OK] Multi-hop 검색 완료 (최종 결과: {len(final_results)}개)")
        
        return final_results, metadata


class ContextWindowManager:
    """컨텍스트 윈도우 관리"""
    
    def __init__(self, model: str = "gpt-4o-mini", max_tokens: int = 4096):
        """
        Args:
            model: 사용할 모델 이름
            max_tokens: 최대 토큰 수
        """
        self.model = model
        self.max_tokens = max_tokens
        self.encoding = tiktoken.encoding_for_model(model)
    
    def count_tokens(self, text: str) -> int:
        """텍스트의 토큰 수 계산"""
        return len(self.encoding.encode(text))
    
    def fit_context(
        self,
        query: str,
        results: List[SearchResult],
        system_prompt: str = "",
        reserve_tokens: int = 1000
    ) -> Tuple[List[SearchResult], Dict[str, int]]:
        """
        컨텍스트 윈도우에 맞게 결과 조정
        
        Args:
            query: 쿼리
            results: 검색 결과
            system_prompt: 시스템 프롬프트
            reserve_tokens: 응답 생성을 위한 예약 토큰
        
        Returns:
            (조정된 결과, 토큰 통계)
        """
        system_tokens = self.count_tokens(system_prompt)
        query_tokens = self.count_tokens(query)
        fixed_tokens = system_tokens + query_tokens + reserve_tokens
        
        available_tokens = self.max_tokens - fixed_tokens
        
        print(f"\n[INFO] 컨텍스트 윈도우 관리:")
        print(f"  - 최대 토큰: {self.max_tokens}")
        print(f"  - 시스템 프롬프트: {system_tokens} 토큰")
        print(f"  - 쿼리: {query_tokens} 토큰")
        print(f"  - 예약 (응답용): {reserve_tokens} 토큰")
        print(f"  - 사용 가능: {available_tokens} 토큰")
        
        fitted_results = []
        used_tokens = 0
        
        for result in results:
            result_tokens = self.count_tokens(result.content)
            
            if used_tokens + result_tokens <= available_tokens:
                fitted_results.append(result)
                used_tokens += result_tokens
            else:
                remaining_tokens = available_tokens - used_tokens
                if remaining_tokens > 100:
                    tokens = self.encoding.encode(result.content)
                    truncated_tokens = tokens[:remaining_tokens]
                    truncated_content = self.encoding.decode(truncated_tokens)
                    
                    fitted_results.append(SearchResult(
                        content=truncated_content + "...",
                        score=result.score,
                        metadata={**result.metadata, "truncated": True},
                        rank=result.rank
                    ))
                    used_tokens += remaining_tokens
                break
        
        stats = {
            "total_tokens": self.max_tokens,
            "system_tokens": system_tokens,
            "query_tokens": query_tokens,
            "reserve_tokens": reserve_tokens,
            "available_tokens": available_tokens,
            "used_tokens": used_tokens,
            "num_results": len(fitted_results),
            "num_truncated": sum(1 for r in fitted_results if r.metadata.get("truncated", False))
        }
        
        print(f"  - 사용된 토큰: {used_tokens}")
        print(f"  - 포함된 결과: {len(fitted_results)}개")
        print(f"  - 잘린 결과: {stats['num_truncated']}개")
        
        return fitted_results, stats


class AdvancedRAGSystem:
    """고급 RAG 시스템 통합"""
    
    def __init__(
        self,
        chunk_size: int = 1024,
        model: str = "gpt-4o-mini",
        use_reranker: bool = True
    ):
        """
        Args:
            chunk_size: 청크 크기
            model: LLM 모델
            use_reranker: 리랭커 사용 여부
        """
        self.chunk_size = chunk_size
        self.model = model
        self.use_reranker = use_reranker
        
        import httpx
        http_client = httpx.Client(verify=False)
        
        self.doc_processor = DocumentProcessor(chunk_size=chunk_size)
        self.embeddings = OpenAIEmbeddings(http_client=http_client)
        self.llm = ChatOpenAI(model=model, temperature=0, http_client=http_client)
        self.client = OpenAI(http_client=http_client)
        
        self.retriever = None
        self.reranker = None
        self.multi_hop_retriever = None
        self.context_manager = ContextWindowManager(model=model)
        
        print(f"Advanced RAG System 초기화 완료 (chunk_size={chunk_size})")
    
    def ingest_text(
        self,
        text: str,
        source_name: str,
        collection_name: Optional[str] = None
    ):
        """텍스트 수집 및 인덱싱"""
        print(f"\n[TEXT] 텍스트 수집 중: {source_name}")
        
        documents = self.doc_processor.create_chunks(
            text, 
            metadata={"source": source_name}
        )
        print(f"  - 생성된 청크: {len(documents)}개")
        
        self.retriever = HybridRetriever(
            documents=documents,
            embeddings=self.embeddings,
            collection_name=collection_name or "advanced_rag"
        )
        
        if self.use_reranker:
            self.reranker = Reranker()
        
        self.multi_hop_retriever = MultiHopRetriever(
            retriever=self.retriever,
            reranker=self.reranker,
            llm=self.llm
        )
        
        print("[OK] 인덱싱 완료")
    
    def search(
        self,
        query: str,
        method: str = "hybrid",
        k: int = 5,
        use_reranker: bool = False
    ) -> List[SearchResult]:
        """검색 수행"""
        results = self.retriever.hybrid_search(query, k=k, alpha=0.5)
        
        if use_reranker and self.reranker:
            results = self.reranker.rerank(query, results, top_k=k)
        
        return results
    
    def multi_hop_search(
        self,
        query: str,
        k: int = 5,
        use_reranker: bool = False
    ) -> Tuple[List[SearchResult], Dict[str, Any]]:
        """Multi-hop 검색"""
        return self.multi_hop_retriever.multi_hop_search(
            query, 
            k_per_hop=k, 
            use_reranker=use_reranker
        )
    
    def generate_answer(
        self,
        query: str,
        results: List[SearchResult],
        use_context_window: bool = False
    ) -> Dict[str, Any]:
        """답변 생성"""
        if use_context_window:
            results, stats = self.context_manager.fit_context(query, results)
        
        context = "\n\n".join([r.content for r in results])
        
        prompt = f"""다음 문서를 참고하여 질문에 답하세요.

질문: {query}

참고 문서:
{context}

답변:"""
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        
        return {
            "answer": response.choices[0].message.content,
            "num_results": len(results)
        }


def experiment_multi_hop(text: str = None):
    """실습 3: Multi-hop 검색"""
    print("\n" + "="*80)
    print("[3] 실습 3: Multi-hop 검색 (다단계 추론)")
    print("="*80)
    print("목표: 복잡한 질문을 하위 질문으로 분해하고 단계적으로 검색")
    
    sample_text = text or SAMPLE_TEXT
    
    rag = AdvancedRAGSystem(chunk_size=400, use_reranker=False)
    rag.doc_processor = DocumentProcessor(chunk_size=400, chunk_overlap=50)
    rag.ingest_text(sample_text, source_name="AI_가이드", collection_name="multihop_exp")
    
    complex_query = "딥러닝 모델의 종류와 각각이 어떤 실무 분야에 적용되는지 설명해주세요"
    
    print(f"\n[*] 복잡한 쿼리: '{complex_query}'")
    print(f"총 청크 수: {len(rag.retriever.documents)}개")
    
    # 일반 검색
    print(f"\n{'─'*60}")
    print("[>] 일반 Hybrid 검색")
    print(f"{'─'*60}")
    
    simple_results = rag.search(complex_query, method="hybrid", k=3)
    print(f"검색 결과: {len(simple_results)}개")
    print_search_result(simple_results[0], 1, show_full=True)
    
    # Multi-hop 검색
    print(f"\n{'─'*60}")
    print("[>] Multi-hop 검색")
    print(f"{'─'*60}")
    
    results, metadata = rag.multi_hop_search(complex_query, k=3, use_reranker=False)
    
    print(f"\n[INFO] Multi-hop 과정:")
    print(f"  - 원본 쿼리 -> {len(metadata['sub_queries'])}개 하위 질문으로 분해")
    for i, sq in enumerate(metadata.get('sub_queries', []), 1):
        print(f"    {i}. {sq}")
    
    print(f"\n  - Hop 1: 첫 번째 질문으로 검색")
    print(f"  - Hop 2: Hop 1 결과를 참고하여 두 번째 질문 검색")
    print(f"  - 최종 결과: 두 홉의 결과 병합 ({len(results)}개)")
    
    print(f"\n상위 결과:")
    for i, result in enumerate(results[:2], 1):
        print_search_result(result, i, show_full=True)
    
    print(f"\n[TIP] Multi-hop 핵심:")
    print("  - 복잡한 질문을 단순한 하위 질문으로 분해")
    print("  - 첫 검색 결과를 두 번째 검색의 컨텍스트로 활용")
    print("  - 적합: 두 개념 이상 연결 필요")
    print("  - 주의: 단순 질문에는 오히려 비효율적")


def experiment_chunk_size(text: str = None):
    """실습 4: Chunk Size 최적화"""
    print("\n" + "="*80)
    print("[4] 실습 4: Chunk Size 최적화")
    print("="*80)
    print("목표: 다양한 청크 크기로 실험하여 최적값 찾기")
    
    sample_text = text or SAMPLE_TEXT
    test_query = "강화 학습이란 무엇인가?"
    
    chunk_sizes = [256, 512, 1024]
    results_comparison = []
    
    for chunk_size in chunk_sizes:
        print(f"\n{'─'*60}")
        print(f"[*] Chunk Size: {chunk_size}자")
        print(f"{'─'*60}")
        
        # RAG 시스템 초기화
        rag = AdvancedRAGSystem(chunk_size=chunk_size, use_reranker=False)
        rag.ingest_text(
            sample_text, 
            source_name="AI_가이드",
            collection_name=f"chunk_{chunk_size}"
        )
        
        # 검색
        start_time = time.time()
        search_results = rag.search(test_query, method="hybrid", k=3)
        
        # 답변 생성
        answer_data = rag.generate_answer(test_query, search_results)
        elapsed = time.time() - start_time
        
        # 토큰 계산
        used_tokens = sum(
            rag.context_manager.count_tokens(r.content) 
            for r in search_results
        )
        
        results_comparison.append({
            'chunk_size': chunk_size,
            'num_chunks': len(rag.retriever.documents),
            'used_tokens': used_tokens,
            'elapsed_time': elapsed,
            'answer': answer_data['answer']
        })
        
        print(f"  - 생성된 청크: {len(rag.retriever.documents)}개")
        print(f"  - 사용 토큰: {used_tokens}")
        print(f"  - 소요 시간: {elapsed:.2f}초")
        print(f"  - 답변: {answer_data['answer'][:100]}...")
    
    # 비교 표
    print(f"\n{'─'*60}")
    print("[분석] Chunk Size별 비교")
    print(f"{'─'*60}")
    
    print(f"\n{'청크 크기':<10} {'청크 수':<8} {'토큰':<8} {'시간':<10} {'답변 길이':<8}")
    print("─" * 60)
    for result in results_comparison:
        print(f"{result['chunk_size']:<10} {result['num_chunks']:<8} {result['used_tokens']:<8} {result['elapsed_time']:.2f}초{'':<4} {len(result['answer'])}자")
    
    print(f"\n[TIP] Chunk Size 가이드:")
    print("  - 작은 청크 (256~512): 정밀 검색, 더 많은 청크 필요")
    print("  - 중간 청크 (512~1024): 균형 잡힌 선택 (일반 권장)")
    print("  - 큰 청크 (1024+): 넓은 컨텍스트, 노이즈 증가 가능")
    print("  - 권장: 도메인별 실험으로 결정")


def experiment_context_window(text: str = None):
    """실습 5: Context Window 관리"""
    print("\n" + "="*80)
    print("[5] 실습 5: Context Window 관리")
    print("="*80)
    print("목표: 컨텍스트 윈도우를 고려한 결과 조정")
    
    sample_text = text or SAMPLE_TEXT
    
    # RAG 시스템 초기화
    rag = AdvancedRAGSystem(chunk_size=600, use_reranker=False)
    rag.ingest_text(
        sample_text, 
        source_name="AI_가이드",
        collection_name="context_exp"
    )
    
    test_query = "딥러닝의 주요 기법들을 설명해주세요"
    
    print(f"\n[*] 쿼리: '{test_query}'")
    
    # 많은 결과 검색 (10개)
    print(f"\n{'─'*60}")
    print("[>] 10개 검색 결과 (Context Window 관리 없이)")
    print(f"{'─'*60}")
    
    results = rag.search(test_query, method="hybrid", k=10)
    print(f"검색 결과: {len(results)}개")
    
    total_tokens = sum(
        rag.context_manager.count_tokens(r.content) 
        for r in results
    )
    print(f"총 토큰: {total_tokens}")
    
    # Context Window 관리 적용
    print(f"\n{'─'*60}")
    print("[>] Context Window 관리 적용")
    print(f"{'─'*60}")
    
    fitted_results, stats = rag.context_manager.fit_context(
        query=test_query,
        results=results,
        system_prompt="당신은 AI 전문가입니다.",
        reserve_tokens=1000
    )
    
    print(f"\n[분석] 조정 결과:")
    print(f"  - 원본 결과: {len(results)}개")
    print(f"  - 조정 후: {len(fitted_results)}개")
    print(f"  - 잘린 결과: {stats['num_truncated']}개")
    print(f"  - 토큰 절약: {total_tokens - stats['used_tokens']}")
    
    print(f"\n[TIP] Context Window 관리:")
    print("  - 토큰 제한을 초과하지 않도록 결과 조정")
    print("  - 우선순위: 상위 결과 → 하위 결과 순으로 포함")
    print("  - 부분 포함: 마지막 결과는 잘라서 포함 가능")
    print("  - 실무: 모델별 토큰 제한 고려 필수")


def main():
    """Chapter 3 실행"""
    print("\n" + "="*80)
    print("[Advanced RAG - Chapter 3] 고급 검색 패턴")
    print("="*80)
    
    print("\n[LIST] 실습 항목:")
    print("  3. Multi-hop 검색")
    print("  4. Chunk Size 최적화")
    print("  5. Context Window 관리")
    
    try:
        experiment_multi_hop()
        experiment_chunk_size()
        experiment_context_window()
        
        print("\n" + "="*80)
        print("[OK] Chapter 3 완료!")
        print("="*80)
        print("\n[NEXT] Advanced 시리즈 완료! 수고하셨습니다.")
        
    except Exception as e:
        print(f"\n[X] 오류 발생: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()


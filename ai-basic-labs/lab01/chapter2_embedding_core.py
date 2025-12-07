"""
[Chapter 2] 임베딩의 핵심 사이클 ⭐
- 실습 3: OpenAI 임베딩 생성
- 실습 4: 코사인 유사도 계산
- 실습 5: 간단한 검색 엔진 구현

학습 목표:
• 텍스트를 벡터로 변환하는 방법 (임베딩)
• 벡터 간 유사성을 측정하는 방법 (코사인 유사도)
• 의미 기반 검색 시스템 구축 (RAG의 Retrieval 부분)

실행:
  python chapter2_embedding_core.py
"""

import os
import sys
from pathlib import Path
from typing import List, Tuple
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv

# 프로젝트 루트의 .env 파일 로드
project_root = Path(__file__).parent.parent
load_dotenv(dotenv_path=project_root / '.env')

# 공통 유틸리티 import
sys.path.insert(0, str(project_root))
from utils import (
    print_section_header, 
    print_subsection, 
    print_key_points, 
    visualize_similarity_bar,
    cosine_similarity,
    cosine_similarity_normalized,
    is_normalized,
    interpret_cosine_similarity,
    get_openai_client,
)


# ============================================================================
# 실습 3: OpenAI 임베딩 생성
# ============================================================================

class EmbeddingGenerator:
    """OpenAI 임베딩 생성기"""
    
    def __init__(self, api_key: str = None):
        self.client = get_openai_client(api_key)
        self.model = "text-embedding-3-small"
    
    def get_embedding(self, text: str) -> List[float]:
        """단일 텍스트의 임베딩 생성"""
        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"\n[!] 임베딩 생성 실패: {e}")
            print(f"[TIP] 확인 사항:")
            print(f"     1. OPENAI_API_KEY가 올바른지 확인")
            print(f"     2. 네트워크 연결 상태 확인")
            raise
    
    def get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """여러 텍스트의 임베딩을 배치로 생성"""
        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=texts
            )
            return [data.embedding for data in response.data]
        except Exception as e:
            print(f"\n[!] 배치 임베딩 생성 실패: {e}")
            raise


def demo_embeddings():
    """실습 3: OpenAI 임베딩 생성"""
    print("\n" + "="*80)
    print("[3] 실습 3: OpenAI 임베딩 생성")
    print("="*80)
    print("목표: 텍스트가 어떻게 숫자 벡터로 변환되는지 이해")
    print("핵심: 의미가 비슷한 텍스트 -> 비슷한 벡터 -> 가까운 거리")
    
    # 임베딩이란?
    print_section_header("임베딩(Embedding)이란?", "[INFO]")
    print("""
  ┌─────────────────────────────────────────────────────────┐
  │  [TIP] 임베딩의 개념                                     │
  │  ─────────────────────────────────────────────────────  │
  │  • 텍스트 -> 고정 길이 숫자 벡터로 변환                  │
  │  • 예: "고양이" -> [0.1, -0.3, 0.5, ..., 0.2] (1536차원) │
  │                                                         │
  │  왜 벡터로 변환하는가?                                   │
  │  • 컴퓨터는 숫자만 연산 가능                             │
  │  • 벡터 공간에서 의미적 유사성 측정 가능                 │
  │  • "왕 - 남자 + 여자 = 여왕" 같은 연산 가능             │
  │                                                         │
  │  OpenAI 임베딩 모델:                                     │
  │  • text-embedding-3-small: 1536차원, 저렴, 빠름          │
  │  • text-embedding-3-large: 3072차원, 고성능              │
  │                                                         │
  │  [TIP] 대부분의 RAG 시스템은 small로 충분!               │
  └─────────────────────────────────────────────────────────┘
    """)
    
    # API 키 확인
    if not os.getenv("OPENAI_API_KEY"):
        print("\n[!] OPENAI_API_KEY 환경변수를 설정해주세요!")
        return
    
    generator = EmbeddingGenerator()
    
    # 단일 임베딩
    print_section_header("단일 텍스트 임베딩", "[DOC]")
    
    text = "Artificial intelligence is transforming the world."
    embedding = generator.get_embedding(text)
    
    print(f"\n텍스트: '{text}'")
    print(f"\n임베딩 결과:")
    print(f"  • 벡터 차원: {len(embedding)}")
    print(f"  • 처음 5개 값: {[round(v, 4) for v in embedding[:5]]}")
    print(f"  • 마지막 5개 값: {[round(v, 4) for v in embedding[-5:]]}")
    print(f"  • 값의 범위: [{min(embedding):.4f}, {max(embedding):.4f}]")
    
    # 값 분포 시각화
    print(f"\n  값 분포 시각화:")
    bins = [0, 0, 0, 0, 0]
    for v in embedding:
        if v < -0.05:
            bins[0] += 1
        elif v < 0:
            bins[1] += 1
        elif v < 0.05:
            bins[2] += 1
        elif v < 0.1:
            bins[3] += 1
        else:
            bins[4] += 1
    
    labels = ["< -0.05", "-0.05~0", "0~0.05", "0.05~0.1", "> 0.1"]
    max_bin = max(bins)
    for label, count in zip(labels, bins):
        bar_len = int(count / max_bin * 30)
        print(f"    {label:>10}: {'#' * bar_len} ({count})")
    
    # L2 노름 계산
    l2_norm = np.sqrt(sum(v**2 for v in embedding))
    
    print(f"""
  [!] OpenAI 임베딩 특성:
     • L2 정규화됨: 벡터 크기(L2 노름) = {l2_norm:.4f} (≈ 1.0)
     • 대부분 값이 -0.1 ~ 0.1 사이에 분포
     • 코사인 유사도 계산에 최적화된 형태
     • 정규화 덕분에 내적(dot product)만으로 유사도 계산 가능""")
    
    # 배치 임베딩
    print_section_header("배치 임베딩 (효율적인 방법)", "[BATCH]")
    
    texts = [
        "I love machine learning.",
        "Deep learning is a subset of AI.",
        "Python is a great programming language."
    ]
    
    print("\n[DOC] 임베딩 생성 코드:")
    print("  ┌─────────────────────────────────────────────────────")
    print("  │ # 비효율적: 개별 호출 (3번 API 호출)")
    print("  │ for text in texts:")
    print("  │     response = client.embeddings.create(input=text)")
    print("  │")
    print("  │ # 효율적: 배치 호출 (1번 API 호출)")
    print("  │ response = client.embeddings.create(input=texts)")
    print("  └─────────────────────────────────────────────────────")
    
    embeddings = generator.get_embeddings_batch(texts)
    
    print(f"\n배치 임베딩 결과:")
    for i, (text, emb) in enumerate(zip(texts, embeddings)):
        print(f"  {i+1}. '{text}'")
        print(f"     차원: {len(emb)}, 처음 5개: {[round(v, 4) for v in emb[:5]]}")
    
    # 핵심 포인트
    print_key_points([
        "- 임베딩: 텍스트 -> 고차원 벡터 (의미를 숫자로 인코딩)",
        "- text-embedding-3-small: 1536차원, 대부분의 RAG에 충분",
        "- 배치 처리: 여러 텍스트를 한 번에 -> API 호출 최소화",
        "- 비용: small ~$0.02/1M토큰",
        "- 용도: 유사도 검색, 클러스터링, RAG"
    ], "임베딩 핵심 포인트")


# ============================================================================
# 실습 4: 코사인 유사도 계산
# ============================================================================

def one_to_many_similarity(query_embedding: List[float], 
                          document_embeddings: List[List[float]]) -> List[float]:
    """1:N 유사도 계산 (하나의 쿼리와 여러 문서)"""
    similarities = []
    for doc_emb in document_embeddings:
        sim = cosine_similarity(query_embedding, doc_emb)
        similarities.append(sim)
    return similarities


def many_to_many_similarity(embeddings1: List[List[float]], 
                           embeddings2: List[List[float]]) -> np.ndarray:
    """N:M 유사도 계산 (여러 쿼리와 여러 문서)"""
    matrix = np.zeros((len(embeddings1), len(embeddings2)))
    
    for i, emb1 in enumerate(embeddings1):
        for j, emb2 in enumerate(embeddings2):
            matrix[i][j] = cosine_similarity(emb1, emb2)
    
    return matrix


def demo_similarity():
    """실습 4: 코사인 유사도 계산"""
    print("\n" + "="*80)
    print("[4] 실습 4: 코사인 유사도 계산")
    print("="*80)
    print("목표: 벡터 간 유사성을 측정하는 방법 이해")
    print("핵심: 코사인 유사도 = 벡터 방향의 유사성 (크기 무관)")
    
    # 코사인 유사도란?
    print_section_header("코사인 유사도란?", "[INFO]")
    print("""
  ┌─────────────────────────────────────────────────────────┐
  │  [TIP] 코사인 유사도 공식                                │
  │  ─────────────────────────────────────────────────────  │
  │                                                         │
  │              A · B           Σ(Aᵢ × Bᵢ)                 │
  │   cos θ = ───────────  =  ─────────────────            │
  │            |A| × |B|      √(ΣAᵢ²) × √(ΣBᵢ²)            │
  │                                                         │
  │  값의 범위 (-1 ~ +1):                                    │
  │  • +1 : 완전히 같은 방향 (매우 유사)                     │
  │  •  0 : 직각 (관련 없음)                                │
  │  • -1 : 반대 방향 (거의 없음)                           │
  │                                                         │
  │  [TIP] 실무 해석 기준:                                   │
  │  • 0.8+ : 매우 유사 (거의 같은 의미)                     │
  │  • 0.6~0.8 : 관련 있음                                  │
  │  • 0.4~0.6 : 약간 관련                                  │
  │  • 0.4 미만 : 다른 주제                                 │
  └─────────────────────────────────────────────────────────┘
    """)
    
    if not os.getenv("OPENAI_API_KEY"):
        print("\n[!] OPENAI_API_KEY 환경변수를 설정해주세요!")
        return
    
    generator = EmbeddingGenerator()
    
    # 문장 준비
    sentences = [
        "I love programming in Python.",
        "Python is my favorite programming language.",
        "I enjoy cooking Italian food.",
        "Machine learning is fascinating.",
    ]
    
    # 임베딩 생성
    embeddings = generator.get_embeddings_batch(sentences)
    
    # 1:N 유사도 계산
    print_section_header("1:N 유사도 계산", "[>>>]")
    
    query = "I like coding with Python."
    query_embedding = generator.get_embedding(query)
    
    print(f"\n쿼리: '{query}'")
    print(f"\n각 문장과의 코사인 유사도:")
    print(f"{'─'*60}")
    
    similarities = one_to_many_similarity(query_embedding, embeddings)
    
    # 결과를 유사도 순으로 정렬
    sorted_results = sorted(zip(sentences, similarities), key=lambda x: x[1], reverse=True)
    
    for sentence, sim in sorted_results:
        bar = visualize_similarity_bar(sim, 30)
        interpretation = interpret_cosine_similarity(sim)
        print(f"\n  {bar} {sim:.4f} {interpretation}")
        print(f"  '{sentence}'")
    
    # 가장 유사한 문장
    most_similar_idx = np.argmax(similarities)
    print(f"\n[#1] 가장 유사한 문장: '{sentences[most_similar_idx]}'")
    print(f"     유사도: {similarities[most_similar_idx]:.4f}")
    
    # 정규화된 벡터 내적 = 코사인 유사도
    print_section_header("정규화된 벡터: 내적 = 코사인 유사도", "[MATH]")
    
    print("""
  ┌─────────────────────────────────────────────────────────┐
  │  [TIP] OpenAI 임베딩의 비밀                              │
  │  ─────────────────────────────────────────────────────  │
  │  OpenAI 임베딩은 L2 정규화되어 있습니다.                  │
  │  즉, ||A|| = ||B|| = 1.0                                │
  │                                                         │
  │  따라서:                                                │
  │       A · B                                             │
  │   ─────────  =  A · B  (내적만으로 충분!)                │
  │    1 × 1                                                │
  │                                                         │
  │  결론: 나눗셈 생략 → 더 빠른 계산!                       │
  └─────────────────────────────────────────────────────────┘
    """)
    
    # 실제 검증
    print("[실험] 정규화 여부 확인:")
    print(f"{'─'*60}")
    
    query_norm = np.linalg.norm(query_embedding)
    print(f"\n  쿼리 벡터 L2 노름: {query_norm:.6f}")
    print(f"  정규화 여부: {'[v] 정규화됨' if is_normalized(query_embedding) else '[x] 정규화 안됨'}")
    
    # 내적 vs 코사인 유사도 비교
    sim_full = cosine_similarity(query_embedding, embeddings[0])
    sim_dot = cosine_similarity_normalized(query_embedding, embeddings[0])
    
    print(f"\n  계산 방법 비교:")
    print(f"    전체 공식: {sim_full:.10f}")
    print(f"    내적만:    {sim_dot:.10f}")
    print(f"    차이:      {abs(sim_full - sim_dot):.2e}")
    
    if abs(sim_full - sim_dot) < 1e-6:
        print(f"\n  [v] 결과가 동일! 내적만으로 충분합니다.")
    
    # N:M 유사도 계산
    print_section_header("N:M 유사도 행렬", "[INFO]")
    
    queries = [
        "Programming languages",
        "Food and cooking"
    ]
    query_embeddings = generator.get_embeddings_batch(queries)
    
    similarity_matrix = many_to_many_similarity(query_embeddings, embeddings)
    
    print("\n유사도 행렬:")
    print(f"{'─'*80}")
    
    # 헤더 출력
    print(f"{'쿼리 \\ 문서':<20}", end="")
    for i in range(len(sentences)):
        print(f"Doc{i+1:2d}  ", end="")
    print()
    print(f"{'─'*80}")
    
    # 각 쿼리별 유사도
    for i, query in enumerate(queries):
        print(f"{query:<20}", end="")
        for j in range(len(sentences)):
            score = similarity_matrix[i][j]
            if score >= 0.5:
                print(f"[{score:.4f}]", end="")
            else:
                print(f" {score:.4f} ", end="")
        print()
    
    print(f"{'─'*80}")
    
    # 문서 목록
    print("\n문서 목록:")
    for i, sentence in enumerate(sentences):
        print(f"  Doc{i+1:2d}: {sentence}")
    
    # 핵심 포인트
    print_key_points([
        "- 코사인 유사도: 벡터 방향의 유사성 (-1 ~ 1)",
        "- 해석: 0.8+ (매우 유사), 0.6~0.8 (관련), 0.4~0.6 (약간 관련)",
        "- 1:N 검색: 쿼리 vs 모든 문서 -> 가장 유사한 문서 찾기",
        "- OpenAI 임베딩: 정규화되어 있어 내적만으로 유사도 계산 가능",
        "- Vector DB는 이 계산을 최적화"
    ], "유사도 계산 핵심 포인트")


# ============================================================================
# 실습 5: 간단한 검색 엔진
# ============================================================================

class SimpleSearchEngine:
    """간단한 의미 기반 검색 엔진"""
    
    def __init__(self, api_key: str = None):
        self.generator = EmbeddingGenerator(api_key)
        self.documents: List[str] = []
        self.embeddings: np.ndarray = None
    
    def add_documents(self, documents: List[str]):
        """문서들을 검색 엔진에 추가"""
        print(f"\n{len(documents)}개의 문서를 인덱싱 중...")
        self.documents.extend(documents)
        new_embeddings = self.generator.get_embeddings_batch(documents)
        
        new_embeddings_array = np.array(new_embeddings)
        if self.embeddings is None:
            self.embeddings = new_embeddings_array
        else:
            self.embeddings = np.vstack([self.embeddings, new_embeddings_array])
        
        print(f"인덱싱 완료! ({self.embeddings.shape[0]}개 문서 × {self.embeddings.shape[1]}차원)")
    
    def search(self, query: str, top_k: int = 3) -> List[Tuple[str, float]]:
        """쿼리와 가장 유사한 문서 검색"""
        if not self.documents or self.embeddings is None:
            return []
        
        # 쿼리 임베딩 생성
        query_embedding = np.array(self.generator.get_embedding(query))
        
        # numpy 벡터화 연산으로 유사도 계산
        # OpenAI 임베딩은 정규화되어 있으므로 dot product만 사용
        similarities = np.dot(self.embeddings, query_embedding)
        
        # 상위 k개 결과
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            results.append((self.documents[idx], float(similarities[idx])))
        
        return results
    
    def print_search_results(self, query: str, top_k: int = 3):
        """검색 결과를 보기 좋게 출력"""
        print(f"\n[>>>] 검색 쿼리: '{query}'")
        print("─" * 60)
        
        results = self.search(query, top_k)
        
        if not results:
            print("검색 결과가 없습니다.")
            return
        
        print(f"\n상위 {len(results)}개 결과:\n")
        for i, (doc, score) in enumerate(results, 1):
            bar = visualize_similarity_bar(score, 25)
            
            if score >= 0.8:
                interpretation = "[v] 매우 유사"
            elif score >= 0.6:
                interpretation = "[~] 관련 있음"
            elif score >= 0.4:
                interpretation = "[o] 약간 관련"
            else:
                interpretation = "[x] 다른 주제"
            
            print(f"[{i}] {bar} {score:.4f} {interpretation}")
            print(f"    {doc}\n")


def demo_search_engine():
    """실습 5: 간단한 검색 엔진"""
    print("\n" + "="*80)
    print("[5] 실습 5: 간단한 검색 엔진 (의미 기반)")
    print("="*80)
    print("목표: 임베딩 기반 검색 시스템의 작동 원리 이해")
    print("핵심: 문서 인덱싱 -> 쿼리 임베딩 -> 유사도 검색 -> 순위 정렬")
    
    # 검색 엔진 구조
    print_section_header("의미 기반 검색 엔진 구조", "[ARCH]")
    print("""
  ┌─────────────────────────────────────────────────────────┐
  │  [STEP 1] 인덱싱 단계 (오프라인)                         │
  │  ─────────────────────────────────────────────────────  │
  │  문서들 -> 임베딩 생성 -> 벡터 저장                       │
  │                                                         │
  │  [STEP 2] 검색 단계 (온라인)                             │
  │  ─────────────────────────────────────────────────────  │
  │  1. 쿼리 입력                                           │
  │  2. 쿼리 임베딩 생성                                    │
  │  3. 저장된 벡터들과 유사도 계산                          │
  │  4. 상위 k개 결과 반환                                  │
  │                                                         │
  │  [!] 이것이 RAG의 "Retrieval" 부분!                      │
  └─────────────────────────────────────────────────────────┘
    """)
    
    if not os.getenv("OPENAI_API_KEY"):
        print("\n[!] OPENAI_API_KEY 환경변수를 설정해주세요!")
        return
    
    # 검색 엔진 초기화
    search_engine = SimpleSearchEngine()
    
    # 샘플 문서 추가
    print_section_header("문서 인덱싱", "[LIST]")
    
    documents = [
        "Python is a high-level programming language known for its simplicity.",
        "Machine learning is a subset of artificial intelligence.",
        "Deep learning uses neural networks with multiple layers.",
        "Natural language processing helps computers understand human language.",
        "Computer vision enables machines to interpret visual information.",
        "Data science involves extracting insights from data.",
        "JavaScript is commonly used for web development.",
        "SQL is used for managing relational databases.",
        "Cloud computing provides on-demand computing resources.",
        "Cybersecurity protects systems from digital attacks.",
        "The weather is beautiful today with clear skies.",
        "I love eating pizza and pasta for dinner.",
        "Exercise and healthy eating are important for wellness.",
        "Traveling to new places broadens your perspective.",
        "Reading books is a great way to learn new things.",
    ]
    
    print("\n인덱싱할 문서:")
    for i, doc in enumerate(documents[:5], 1):
        print(f"  {i}. {doc}")
    print(f"  ... ({len(documents) - 5}개 더)")
    
    search_engine.add_documents(documents)
    print(f"\n[OK] 총 {len(documents)}개 문서 인덱싱 완료")
    
    # 다양한 쿼리로 검색
    print_section_header("검색 테스트", "[>>>]")
    
    queries = [
        "What is AI and machine learning?",
        "Tell me about programming languages",
        "How can I stay healthy?",
        "I want to learn about databases",
    ]
    
    for query in queries:
        search_engine.print_search_results(query, top_k=3)
    
    # 키워드 검색 vs 의미 검색
    print_section_header("키워드 검색 vs 의미 검색", "[vs]")
    print("""
  ┌─────────────────────────────────────────────────────────┐
  │  [CMP] 비교                                              │
  │  ─────────────────────────────────────────────────────  │
  │  키워드 검색 (BM25)         │ 의미 검색 (임베딩)          │
  │  ───────────────────────────┼────────────────────────── │
  │  "Python" 검색 시           │ "Python" 검색 시           │
  │  -> "Python" 포함 문서만    │ -> 프로그래밍 관련 문서도  │
  │                             │    (JavaScript, SQL 등)    │
  │  ───────────────────────────┼────────────────────────── │
  │  장점: 빠름, 정확한 키워드  │ 장점: 동의어, 유사 개념    │
  │  단점: 동의어 못 찾음       │ 단점: 임베딩 비용 필요     │
  │  ───────────────────────────┼────────────────────────── │
  │  [TIP] 실무: Hybrid 검색 사용 (lab03에서 학습)          │
  └─────────────────────────────────────────────────────────┘
    """)
    
    # 현재 방식의 한계
    print_section_header("현재 방식의 한계", "[!]")
    print(f"""
  ┌─────────────────────────────────────────────────────────┐
  │  [!] 선형 검색의 시간 복잡도: O(n)                       │
  │  ─────────────────────────────────────────────────────  │
  │  현재 인덱싱된 문서: {len(search_engine.documents)}개                              │
  │                                                         │
  │  문서 수에 따른 검색 시간:                               │
  │  • 1,000개    → ~10ms   (실시간 가능)                   │
  │  • 10,000개   → ~100ms  (약간 지연)                     │
  │  • 100,000개  → ~1초    (느림)                          │
  │  • 1,000,000개 → ~10초   (실시간 불가!)                 │
  │                                                         │
  │  [>>>] Vector DB의 해결책:                              │
  │  • ANN 알고리즘 사용 → 시간 복잡도 O(log n)             │
  │  • 1,000,000개 → ~10ms (1000배 빠름!)                   │
  │                                                         │
  │  → lab02에서 ChromaDB로 학습                            │
  └─────────────────────────────────────────────────────────┘
    """)
    
    # 핵심 포인트
    print_key_points([
        "- 의미 검색: 키워드가 달라도 의미가 비슷하면 검색됨",
        "- 인덱싱: 문서를 임베딩으로 변환하여 저장 (1회)",
        "- 검색: 쿼리 임베딩 -> 유사도 계산 -> 순위 정렬",
        "- 한계: O(n) 복잡도 -> Vector DB(O(log n))로 해결",
        "- 발전: RAG = 검색 + LLM 답변 생성 (lab03)"
    ], "검색 엔진 핵심 포인트")


# ============================================================================
# 메인 실행
# ============================================================================

def main():
    """챕터 2 실행"""
    print("\n" + "="*80)
    print("[Chapter 2] 임베딩의 핵심 사이클 ⭐")
    print("="*80)
    print("\n학습 목표:")
    print("  • 텍스트를 벡터로 변환하는 방법 (임베딩)")
    print("  • 벡터 간 유사성을 측정하는 방법 (코사인 유사도)")
    print("  • 의미 기반 검색 시스템 구축 (RAG의 핵심)")
    
    # API 키 확인
    if not os.getenv("OPENAI_API_KEY"):
        print("\n[!] OPENAI_API_KEY가 설정되지 않았습니다!")
        print("    프로젝트 루트의 .env 파일에 키를 추가해주세요.")
        return
    
    try:
        # 실습 3: 임베딩
        demo_embeddings()
        
        # 실습 4: 유사도
        demo_similarity()
        
        # 실습 5: 검색 엔진
        demo_search_engine()
        
        # 완료 메시지
        print("\n" + "="*80)
        print("[OK] Chapter 2 완료!")
        print("="*80)
        
        print("\n[요약]")
        print("  • 임베딩: 텍스트 -> 벡터 (의미를 숫자로)")
        print("  • 코사인 유사도: 벡터 간 유사성 측정")
        print("  • 검색: 생성 -> 비교 -> 순위 정렬")
        print("  • 이것이 RAG의 Retrieval 부분!")
        
        print("\n[다음 단계]")
        print("  • Chapter 3: 임베딩 시각화, 모델 비교, 한글 처리")
        print("  • python chapter3_deep_dive.py")
        print("  • 또는 lab02: Vector DB로 대용량 검색 학습")
        
    except Exception as e:
        print(f"\n[X] 오류 발생: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()


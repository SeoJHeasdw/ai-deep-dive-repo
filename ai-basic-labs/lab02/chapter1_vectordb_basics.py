"""
[Chapter 1] Vector DB 기초
- 실습 1: 임베딩(Embedding) 이해하기
- 실습 2: Vector DB 기본 작업 (Add/Query)
- 실습 3: 거리와 유사도 이해하기

학습 목표:
• Lab01에서 배운 임베딩을 Vector DB와 연결
• ChromaDB에 벡터 저장/검색하는 기본 흐름 이해
• L2 거리와 코사인 유사도의 차이 파악
• 거리 스코어 해석 방법 학습

실행:
  python chapter1_vectordb_basics.py

[!] Windows + Python 3.13 사용자:
    ChromaDB 호환 이슈가 있을 수 있습니다.
    Python 3.11 또는 3.12 권장
"""

import os
import sys
import platform
from pathlib import Path
from typing import List, Dict, Any
import chromadb
from openai import OpenAI
from dotenv import load_dotenv
import numpy as np

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
    interpret_cosine_similarity,
    interpret_l2_distance,
    get_openai_client,
)

# 유사도 막대 그래프 너비
SIMILARITY_BAR_WIDTH = 30


def truncate_text(text: str, max_len: int = 50) -> str:
    """텍스트를 지정 길이로 자르고 ... 추가"""
    if len(text) <= max_len:
        return text
    return text[:max_len] + "..."


# ============================================================================
# 임베딩 생성기
# ============================================================================

class EmbeddingGenerator:
    """OpenAI 임베딩 생성기"""
    
    def __init__(self, api_key: str = None):
        """
        임베딩 생성기 초기화
        
        임베딩이란?
        - 텍스트를 고정 길이의 숫자 벡터로 변환하는 것
        - 의미적으로 유사한 텍스트는 유사한 벡터를 갖음
        """
        self.client = get_openai_client(api_key)
        self.model = "text-embedding-3-small"
        self.dimensions = 1536
    
    def get_embedding(self, text: str) -> List[float]:
        """단일 텍스트의 임베딩 생성"""
        response = self.client.embeddings.create(
            model=self.model,
            input=text
        )
        return response.data[0].embedding
    
    def get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """여러 텍스트의 임베딩을 배치로 생성"""
        response = self.client.embeddings.create(
            model=self.model,
            input=texts
        )
        return [data.embedding for data in response.data]


# ============================================================================
# ChromaDB 관리자
# ============================================================================

class ChromaDBManager:
    """ChromaDB 관리 클래스"""
    
    def __init__(self, persist_directory: str = None):
        """
        ChromaDB 클라이언트 초기화
        
        Vector DB란?
        - 임베딩 벡터를 효율적으로 저장하고 검색하는 데이터베이스
        - 일반 DB: 정확한 값 매칭 (WHERE id = 123)
        - Vector DB: 유사도 기반 검색 (가장 비슷한 벡터 찾기)
        """
        if persist_directory is None:
            persist_directory = str(Path(__file__).parent / "chroma_db")
        
        self.persist_directory = persist_directory
        self.client = chromadb.PersistentClient(path=persist_directory)
        
        print(f"[OK] ChromaDB 초기화 완료")
        print(f"   저장 위치: {persist_directory}")
    
    def create_collection(self, name: str, reset: bool = False, 
                          distance_fn: str = "l2") -> chromadb.Collection:
        """
        컬렉션 생성 또는 가져오기
        
        컬렉션 = 벡터들을 그룹화하는 단위 (테이블과 유사)
        
        Args:
            name: 컬렉션 이름
            reset: True면 기존 컬렉션 삭제 후 재생성
            distance_fn: 거리 함수 ("l2", "cosine", "ip")
                - "l2": 유클리드 거리 (기본값, 0에 가까울수록 유사)
                - "cosine": 코사인 거리 (0에 가까울수록 유사)
                - "ip": 내적의 음수
        """
        if reset:
            try:
                self.client.delete_collection(name=name)
                print(f"   기존 컬렉션 '{name}' 삭제됨")
            except:
                pass
        
        metadata = {
            "description": "Vector database collection",
            "hnsw:space": distance_fn
        }
        
        collection = self.client.get_or_create_collection(
            name=name,
            metadata=metadata
        )
        
        print(f"   컬렉션 '{name}' 준비 완료 (문서 수: {collection.count()}, 거리: {distance_fn})")
        return collection


# ============================================================================
# 실습 1: 임베딩 기초 (Lab01 복습)
# ============================================================================

def demo_embedding_basics():
    """실습 1: 임베딩(Embedding) 이해하기"""
    print("\n" + "="*80)
    print("[1] 실습 1: 임베딩(Embedding) 이해하기")
    print("="*80)
    print("목표: 텍스트가 어떻게 숫자 벡터로 변환되는지 이해")
    print("핵심: 의미가 비슷한 텍스트 -> 비슷한 벡터 -> 가까운 거리")
    
    if not os.getenv("OPENAI_API_KEY"):
        print("\n[!] OPENAI_API_KEY 환경변수를 설정해주세요!")
        return
    
    embedder = EmbeddingGenerator()
    
    # 임베딩 생성 기본
    print_section_header("임베딩 생성 기본", "[INFO]")
    
    text = "Python is a programming language"
    embedding = embedder.get_embedding(text)
    
    print(f"\n입력 텍스트: '{text}'")
    print(f"\n임베딩 결과:")
    print(f"  • 벡터 차원: {len(embedding)}")
    print(f"  • 처음 5개 값: {[round(v, 4) for v in embedding[:5]]}")
    print(f"  • 마지막 5개 값: {[round(v, 4) for v in embedding[-5:]]}")
    print(f"  • 값의 범위: [{min(embedding):.4f}, {max(embedding):.4f}]")
    
    print(f"""
  ┌─────────────────────────────────────────────────────────┐
  │  [TIP] 벡터 값의 의미                                    │
  │  ─────────────────────────────────────────────────────  │
  │  * 각 차원은 텍스트의 특정 "의미적 특징"을 나타냄        │
  │  * 실제로는 사람이 해석하기 어려운 추상적 특징           │
  │  * 중요한 것: 비슷한 의미 = 비슷한 벡터 패턴            │
  │  * 개별 값보다 전체 벡터의 "방향"이 의미를 결정          │
  └─────────────────────────────────────────────────────────┘
    """)
    
    # 의미적 유사성 데모
    print_section_header("의미적 유사성 테스트", "[*]")
    
    texts = [
        "I love programming in Python",      # 프로그래밍
        "Python is great for coding",        # 프로그래밍 (유사)
        "I enjoy cooking Italian food",      # 요리 (다름)
        "The weather is sunny today",        # 날씨 (완전 다름)
    ]
    
    print("\n비교할 텍스트:")
    for i, t in enumerate(texts, 1):
        print(f"  {i}. {t}")
    
    embeddings = embedder.get_embeddings_batch(texts)
    
    print_subsection("첫 번째 텍스트와의 코사인 유사도")
    
    base_embedding = embeddings[0]
    print(f"\n기준: '{texts[0]}'")
    print()
    
    for i, (text, emb) in enumerate(zip(texts[1:], embeddings[1:]), 2):
        similarity = cosine_similarity(base_embedding, emb)
        bar = visualize_similarity_bar(similarity, SIMILARITY_BAR_WIDTH)
        interpretation = interpret_cosine_similarity(similarity)
        
        print(f"  vs '{text}'")
        print(f"     {bar} {similarity:.4f} {interpretation}")
        print()
    
    # 코사인 유사도 vs L2 거리 설명
    print_section_header("코사인 유사도 vs L2 거리", "[vs]")
    print("""
  ┌─────────────────────────────────────────────────────────┐
  │  [CMP] 두 가지 유사도 측정 방법                          │
  │  ─────────────────────────────────────────────────────  │
  │                                                         │
  │  1. 코사인 유사도 (Cosine Similarity)                   │
  │     * 벡터의 "방향"만 비교                              │
  │     * 값 범위: -1 ~ 1 (1이 가장 유사)                   │
  │     * Lab01에서 직접 계산한 방식                        │
  │                                                         │
  │  2. L2 거리 (Euclidean Distance)                       │
  │     * 벡터 간의 "직선 거리"                             │
  │     * 값 범위: 0 ~ ∞ (0이 가장 유사)                    │
  │     * ChromaDB의 기본 설정                              │
  │     * 이 Lab에서 사용할 방식                            │
  │                                                         │
  │  [!] OpenAI 임베딩은 L2 정규화되어 있어서               │
  │      두 방식 모두 비슷한 결과를 제공합니다              │
  └─────────────────────────────────────────────────────────┘
    """)
    
    # ChromaDB 거리 함수 설정
    print_subsection("ChromaDB 거리 함수 설정 방법")
    print("""
  [CODE] 컬렉션 생성 시 거리 함수 지정:
  ┌─────────────────────────────────────────────────────
  │ # L2 거리 (기본값)
  │ collection = client.create_collection(
  │     name="my_collection",
  │     metadata={"hnsw:space": "l2"}
  │ )
  │
  │ # 코사인 거리 (Lab01과 동일)
  │ collection = client.create_collection(
  │     name="my_collection",
  │     metadata={"hnsw:space": "cosine"}
  │ )
  └─────────────────────────────────────────────────────

  [!] Lab01(cosine) vs Lab02(L2) 혼동 방지:
  
  OpenAI 임베딩은 L2 정규화되어 있어서
  cosine, inner product, L2는 대부분 동일한 "순위" 결과를 줍니다.
  
  즉, "어떤 거리를 써도 1등은 같다!" (스코어 수치만 다름)
    """)
    
    # 핵심 포인트
    print_key_points([
        "- 임베딩: 텍스트 -> 고정 길이 숫자 벡터 (1536차원)",
        "- 의미가 비슷한 텍스트 -> 벡터 공간에서 가까이 위치",
        "- 코사인 유사도: 방향 비교 (-1 ~ 1)",
        "- L2 거리: 직선 거리 (0 ~ ∞), ChromaDB 기본값",
        "- OpenAI 임베딩은 정규화되어 있어 둘 다 유사한 결과"
    ])


# ============================================================================
# 실습 2: Vector DB 기본 작업
# ============================================================================

def demo_basic_operations():
    """실습 2: Vector DB 기본 작업"""
    print("\n" + "="*80)
    print("[2] 실습 2: Vector DB 기본 작업")
    print("="*80)
    print("목표: ChromaDB에 벡터 저장하고 검색하는 기본 흐름 이해")
    print("핵심: 문서 추가(Add) -> 검색(Query) -> 결과 해석")
    
    if not os.getenv("OPENAI_API_KEY"):
        print("\n[!] OPENAI_API_KEY 환경변수를 설정해주세요!")
        return
    
    # DB 초기화
    print_section_header("데이터베이스 초기화", "[DB]")
    
    db = ChromaDBManager()
    collection = db.create_collection("demo_basic", reset=True)
    
    # 임베딩 생성기
    embedder = EmbeddingGenerator()
    
    # 샘플 문서 추가
    print_section_header("문서 추가 (Add)", "[DOC]")
    
    documents = [
        "Python is a high-level programming language known for its simplicity.",
        "Machine learning is a subset of artificial intelligence that learns from data.",
        "ChromaDB is an open-source vector database for AI applications.",
        "Deep learning uses neural networks with multiple layers.",
        "JavaScript is widely used for web development.",
    ]
    
    print(f"\n추가할 문서 ({len(documents)}개):")
    for i, doc in enumerate(documents, 1):
        print(f"  {i}. {truncate_text(doc, 50)}")
    
    # 배치 임베딩 생성
    print("\n[...] 임베딩 생성 중...")
    embeddings = embedder.get_embeddings_batch(documents)
    
    # ChromaDB에 추가
    collection.add(
        documents=documents,
        embeddings=embeddings,
        ids=[f"doc_{i}" for i in range(len(documents))],
        metadatas=[{"source": "demo", "index": i} for i in range(len(documents))]
    )
    
    print(f"[OK] {len(documents)}개 문서 추가 완료")
    print(f"   컬렉션 크기: {collection.count()}개")
    
    # 검색 테스트
    print_section_header("유사도 검색 (Query)", "[>>>]")
    
    query = "What is a vector database?"
    print(f"\n쿼리: '{query}'")
    
    # 쿼리 임베딩 생성
    query_embedding = embedder.get_embedding(query)
    
    # 검색 실행
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3
    )
    
    print(f"\n상위 {len(results['documents'][0])}개 결과:")
    print(f"{'─'*60}")
    
    for i, (doc, distance, metadata) in enumerate(zip(
        results['documents'][0],
        results['distances'][0],
        results['metadatas'][0]
    ), 1):
        similarity = 1 / (1 + distance)
        bar = visualize_similarity_bar(similarity, SIMILARITY_BAR_WIDTH)
        interpretation = interpret_l2_distance(distance)
        
        print(f"\n[{i}위] 거리: {distance:.4f} | 유사도: {similarity:.4f} {interpretation}")
        print(f"     {bar}")
        print(f"     문서: {doc}")
        print(f"     메타: {metadata}")
    
    # 거리 스코어 해석 가이드
    print_section_header("거리 스코어 해석 가이드", "[INFO]")
    print("""
  ┌─────────────────────────────────────────────────────────┐
  │  [TIP] L2 거리 스코어 해석 (이 실습 데이터 기준 예시)     │
  │  ─────────────────────────────────────────────────────  │
  │  * OpenAI 임베딩 + L2 거리 참고 범위:                    │
  │                                                         │
  │    거리 0.0 ~ 0.5  ->  매우 높은 관련성                  │
  │    거리 0.5 ~ 1.0  ->  높은 관련성                       │
  │    거리 1.0 ~ 1.5  ->  중간 관련성                       │
  │    거리 1.5 ~ 2.0  ->  낮은 관련성                       │
  │    거리 2.0 이상   ->  거의 무관                         │
  │                                                         │
  │  ⚠️ [!] 중요: 이 수치는 "이 실습 데이터셋" 기준 예시!    │
  │  ─────────────────────────────────────────────────────  │
  │  실무에서 L2 거리 스케일은 다음에 따라 완전히 달라집니다: │
  │  * 데이터 분포 (문서들이 얼마나 다양한지)                │
  │  * 문서 길이 (긴 문서 vs 짧은 문서)                     │
  │  * 임베딩 모델 (small vs large)                         │
  │                                                         │
  │  → 반드시 자신의 데이터셋으로 분석 후 임계값 결정!       │
  └─────────────────────────────────────────────────────────┘
    """)
    
    # 핵심 포인트
    print_key_points([
        "- Vector DB: 임베딩 벡터를 저장하고 유사도로 검색",
        "- Collection: 벡터들의 그룹 (일반 DB의 테이블)",
        "- Add: 문서 + 임베딩 + 메타데이터 저장",
        "- Query: 쿼리 임베딩과 가장 가까운 벡터 검색",
        "- 거리 해석: 절대 수치보다 상대 순위가 중요!",
        "- L2 거리 임계값은 데이터셋마다 다름"
    ])


# ============================================================================
# 실습 3: 거리와 유사도 이해하기
# ============================================================================

def demo_distance_scores():
    """실습 3: 거리와 유사도 이해하기"""
    print("\n" + "="*80)
    print("[3] 실습 3: 거리와 유사도 이해하기")
    print("="*80)
    print("목표: ChromaDB의 거리 스코어가 무엇을 의미하는지 이해")
    print("핵심: 거리 DOWN = 유사도 UP = 더 관련성 높음")
    
    if not os.getenv("OPENAI_API_KEY"):
        print("\n[!] OPENAI_API_KEY 환경변수를 설정해주세요!")
        return
    
    # 초기화
    db = ChromaDBManager()
    collection = db.create_collection("demo_scores", reset=True)
    embedder = EmbeddingGenerator()
    
    # 다양한 관련성을 가진 문서들
    print_section_header("테스트 문서 준비", "[LIST]")
    
    documents = [
        "Python programming is fun and easy to learn.",
        "I write Python code every day for data analysis.",
        "Machine learning with Python is powerful.",
        "Italian pasta is my favorite food.",
        "The stock market showed gains today.",
    ]
    
    print("\n테스트 문서:")
    for i, doc in enumerate(documents, 1):
        print(f"  {i}. {doc}")
    
    # 임베딩 및 저장
    embeddings = embedder.get_embeddings_batch(documents)
    collection.add(
        documents=documents,
        embeddings=embeddings,
        ids=[f"doc_{i}" for i in range(len(documents))]
    )
    
    # 검색 및 스코어 분석
    print_section_header("거리 스코어 분석", "[DATA]")
    
    query = "Python programming language"
    query_embedding = embedder.get_embedding(query)
    
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=len(documents)
    )
    
    print(f"\n쿼리: '{query}'")
    print("""
  ┌─────────────────────────────────────────────────────────┐
  │  [INFO] 해석 기준 (⚠️ 이 실습 데이터 한정 예시)          │
  │  ─────────────────────────────────────────────────────  │
  │  * [v] 높음: L2 거리 < 1.0 - 주제 일치                  │
  │  * [~] 중간: L2 거리 1.0~1.8 - 관련성 있음              │
  │  * [x] 낮음: L2 거리 > 1.8 - 다른 주제                  │
  │                                                         │
  │  ⚠️ 절대 일반화 금지! 데이터셋마다 다름!                 │
  └─────────────────────────────────────────────────────────┘
    """)
    
    print(f"{'─'*70}")
    print(f"{'순위':<4} {'L2 거리':<10} {'유사도':<10} {'해석':<12} 문서")
    print(f"{'─'*70}")
    
    for i, (doc, distance) in enumerate(zip(
        results['documents'][0],
        results['distances'][0]
    ), 1):
        similarity = 1 / (1 + distance)
        interpretation = interpret_l2_distance(distance)
        bar = visualize_similarity_bar(similarity, SIMILARITY_BAR_WIDTH)
        
        print(f"{i:<4} {distance:<10.4f} {similarity:<10.4f} {interpretation:<12} {truncate_text(doc, 35)}")
        print(f"     {bar}")
        print()
    
    # 거리 메트릭 설명
    print_section_header("거리 메트릭 이해하기", "[CALC]")
    print("""
  ┌─────────────────────────────────────────────────────────┐
  │  L2 거리 (Euclidean Distance)                          │
  │  ─────────────────────────────────────────────────────  │
  │  * 두 벡터 간의 직선 거리                               │
  │  * 공식: sqrt(sum((a[i] - b[i])^2))                    │
  │  * 값 범위: 0 ~ infinity                               │
  │  * 0에 가까울수록 유사                                  │
  └─────────────────────────────────────────────────────────┘
  
  [!] 실무 RAG에서는 0.35~0.55 구간의 문서들도
      상위 컨텍스트로 매우 자주 사용됩니다!
  
  * 절대 점수보다 "다른 결과 대비 상대적으로 높은가"가 핵심
    """)
    
    # 핵심 포인트
    print_key_points([
        "- L2 거리: 0이 가장 가까움 (거리 DOWN = 유사도 UP)",
        "- 실무: 유사도 0.35~0.55도 충분히 유용한 결과!",
        "- 핵심: 절대 점수보다 상대 순위가 중요",
        "- 거리 임계값은 데이터셋마다 다르므로 직접 분석 필요"
    ])


# ============================================================================
# 메인 실행
# ============================================================================

def main():
    """챕터 1 실행"""
    print("\n" + "="*80)
    print("[Chapter 1] Vector DB 기초")
    print("="*80)
    
    # Python 버전 체크
    python_version = sys.version_info
    is_windows = platform.system() == "Windows"
    
    print(f"\n[INFO] 실행 환경:")
    print(f"   Python: {python_version.major}.{python_version.minor}.{python_version.micro}")
    print(f"   OS: {platform.system()}")
    
    # Windows + Python 3.13 경고
    if is_windows and python_version.minor >= 13:
        print(f"""
[!] Python 3.13은 ChromaDB가 아직 공식 지원하지 않는 버전입니다.
    Python 3.11 또는 3.12 권장

    계속하려면 Enter 키를 누르세요...
""")
        input()
    
    print("\n학습 목표:")
    print("  • Lab01 임베딩 개념 복습")
    print("  • ChromaDB 기본 사용법 (Add/Query)")
    print("  • L2 거리와 코사인 유사도 차이 이해")
    print("  • 거리 스코어 해석 방법")
    
    try:
        # 실습 1: 임베딩 기초
        demo_embedding_basics()
        
        # 실습 2: 기본 작업
        demo_basic_operations()
        
        # 실습 3: 거리 이해
        demo_distance_scores()
        
        # 완료 메시지
        print("\n" + "="*80)
        print("[OK] Chapter 1 완료!")
        print("="*80)
        
        print("\n[요약]")
        print("  • 임베딩: 텍스트를 1536차원 벡터로 변환")
        print("  • Vector DB: 벡터를 저장하고 유사도로 검색")
        print("  • L2 거리: ChromaDB 기본 거리 함수 (0 = 가장 유사)")
        print("  • 핵심: 절대 점수보다 상대 순위가 중요!")
        
        print("\n[다음 단계]")
        print("  python chapter2_practical_usage.py")
        print("  → 메타데이터 필터링과 실전 문서 관리 시스템")
        
    except Exception as e:
        print(f"\n[X] 오류 발생: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()


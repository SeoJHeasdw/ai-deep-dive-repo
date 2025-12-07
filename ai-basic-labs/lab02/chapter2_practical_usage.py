"""
[Chapter 2] 실전 활용 ⭐
- 실습 4: 메타데이터 필터링 (조건부 검색)
- 실습 5: 문서 관리 시스템 (실전 예제)

학습 목표:
• 메타데이터를 활용한 정밀 검색 구현
• DocumentManager 클래스 패턴 학습 (실무 활용)
• 배치 처리로 효율적인 문서 추가
• 다양한 검색 시나리오 구현

핵심:
이 챕터의 코드는 실무에서 그대로 사용할 수 있는 패턴입니다!

실행:
  python chapter2_practical_usage.py
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import chromadb
from dotenv import load_dotenv

# 프로젝트 루트의 .env 파일 로드
project_root = Path(__file__).parent.parent
load_dotenv(dotenv_path=project_root / '.env')

# 공통 유틸리티 import
sys.path.insert(0, str(project_root))
from utils import (
    print_section_header,
    print_key_points,
    visualize_similarity_bar,
    interpret_l2_distance,
    get_openai_client,
)

SIMILARITY_BAR_WIDTH = 30


def truncate_text(text: str, max_len: int = 50) -> str:
    """텍스트를 지정 길이로 자르고 ... 추가"""
    if len(text) <= max_len:
        return text
    return text[:max_len] + "..."


# ============================================================================
# 데이터 클래스
# ============================================================================

@dataclass
class SearchResult:
    """검색 결과 데이터 클래스"""
    content: str
    distance: float
    similarity: float
    metadata: Dict[str, Any]
    rank: int


# ============================================================================
# 임베딩 생성기 & ChromaDB 관리자 (재사용)
# ============================================================================

class EmbeddingGenerator:
    """OpenAI 임베딩 생성기"""
    
    def __init__(self, api_key: str = None):
        self.client = get_openai_client(api_key)
        self.model = "text-embedding-3-small"
    
    def get_embedding(self, text: str) -> List[float]:
        """단일 텍스트의 임베딩 생성"""
        response = self.client.embeddings.create(model=self.model, input=text)
        return response.data[0].embedding
    
    def get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """배치 임베딩 생성"""
        response = self.client.embeddings.create(model=self.model, input=texts)
        return [data.embedding for data in response.data]


class ChromaDBManager:
    """ChromaDB 관리 클래스"""
    
    def __init__(self, persist_directory: str = None):
        if persist_directory is None:
            persist_directory = str(Path(__file__).parent / "chroma_db")
        
        self.persist_directory = persist_directory
        self.client = chromadb.PersistentClient(path=persist_directory)
        print(f"[OK] ChromaDB 초기화: {persist_directory}")
    
    def create_collection(self, name: str, reset: bool = False) -> chromadb.Collection:
        """컬렉션 생성 또는 가져오기"""
        if reset:
            try:
                self.client.delete_collection(name=name)
            except:
                pass
        
        collection = self.client.get_or_create_collection(name=name)
        print(f"   컬렉션 '{name}' 준비 (문서: {collection.count()}개)")
        return collection


# ============================================================================
# 문서 관리자 (실전 패턴)
# ============================================================================

class DocumentManager:
    """문서 관리 시스템 - 실무에서 사용할 수 있는 패턴"""
    
    def __init__(self, collection_name: str = "documents", reset: bool = False):
        """
        Args:
            collection_name: 컬렉션 이름
            reset: True면 기존 데이터 삭제 후 시작
        """
        self.db = ChromaDBManager()
        self.collection = self.db.create_collection(collection_name, reset=reset)
        self.embedder = EmbeddingGenerator()
    
    def add_document(self, text: str, metadata: Dict[str, Any]) -> str:
        """단일 문서 추가"""
        doc_id = f"doc_{self.collection.count()}"
        embedding = self.embedder.get_embedding(text)
        
        self.collection.add(
            documents=[text],
            embeddings=[embedding],
            ids=[doc_id],
            metadatas=[metadata]
        )
        
        return doc_id
    
    def add_documents_batch(self, texts: List[str], metadatas: List[Dict[str, Any]]) -> List[str]:
        """여러 문서 일괄 추가 (배치 임베딩으로 효율적)"""
        start_idx = self.collection.count()
        doc_ids = [f"doc_{start_idx + i}" for i in range(len(texts))]
        embeddings = self.embedder.get_embeddings_batch(texts)
        
        self.collection.add(
            documents=texts,
            embeddings=embeddings,
            ids=doc_ids,
            metadatas=metadatas
        )
        
        return doc_ids
    
    def search(self, query: str, n_results: int = 5, 
               where: Optional[Dict] = None) -> List[SearchResult]:
        """문서 검색"""
        query_embedding = self.embedder.get_embedding(query)
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where
        )
        
        search_results = []
        for i, (doc, dist, meta) in enumerate(zip(
            results['documents'][0],
            results['distances'][0],
            results['metadatas'][0]
        )):
            similarity = 1 / (1 + dist)
            
            search_results.append(SearchResult(
                content=doc,
                distance=dist,
                similarity=similarity,
                metadata=meta,
                rank=i + 1
            ))
        
        return search_results
    
    def get_stats(self) -> Dict:
        """컬렉션 통계"""
        return {
            'total_documents': self.collection.count(),
            'collection_name': self.collection.name
        }


# ============================================================================
# 실습 4: 메타데이터 필터링
# ============================================================================

def demo_metadata_filtering():
    """실습 4: 메타데이터 필터링"""
    print("\n" + "="*80)
    print("[4] 실습 4: 메타데이터 필터링")
    print("="*80)
    print("목표: 벡터 검색에 조건을 추가하여 정밀한 검색 수행")
    print("핵심: 의미 검색 + 메타데이터 필터 = 더 정확한 결과")
    
    if not os.getenv("OPENAI_API_KEY"):
        print("\n[!] OPENAI_API_KEY 환경변수를 설정해주세요!")
        return
    
    # 초기화
    db = ChromaDBManager()
    collection = db.create_collection("demo_metadata", reset=True)
    embedder = EmbeddingGenerator()
    
    # 메타데이터가 있는 문서들
    print_section_header("메타데이터가 있는 문서 추가", "[LIST]")
    
    documents = [
        {"text": "Python basics for beginners", "metadata": {"language": "python", "level": "beginner", "year": 2024}},
        {"text": "Advanced Python programming", "metadata": {"language": "python", "level": "advanced", "year": 2024}},
        {"text": "JavaScript for web developers", "metadata": {"language": "javascript", "level": "intermediate", "year": 2023}},
        {"text": "React framework tutorial", "metadata": {"language": "javascript", "level": "intermediate", "year": 2024}},
        {"text": "Python for data science", "metadata": {"language": "python", "level": "intermediate", "year": 2023}},
        {"text": "Java programming fundamentals", "metadata": {"language": "java", "level": "beginner", "year": 2023}},
    ]
    
    print("\n추가된 문서:")
    print(f"{'─'*70}")
    for i, doc in enumerate(documents, 1):
        meta = doc["metadata"]
        print(f"  {i}. {doc['text']}")
        print(f"     +-- {meta}")
    
    # 문서 추가
    texts = [d["text"] for d in documents]
    metadatas = [d["metadata"] for d in documents]
    embeddings = embedder.get_embeddings_batch(texts)
    
    collection.add(
        documents=texts,
        embeddings=embeddings,
        ids=[f"doc_{i}" for i in range(len(documents))],
        metadatas=metadatas
    )
    
    # 쿼리 준비
    query = "programming tutorial"
    query_embedding = embedder.get_embedding(query)
    
    # 1. 필터 없이 검색
    print_section_header("필터 없이 검색", "[>>>]")
    print(f"\n쿼리: '{query}'")
    
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3
    )
    
    print("\n결과:")
    for i, (doc, meta, dist) in enumerate(zip(
        results['documents'][0],
        results['metadatas'][0],
        results['distances'][0]
    ), 1):
        similarity = 1 / (1 + dist)
        print(f"  [{i}] {doc}")
        print(f"      메타: {meta} | 유사도: {similarity:.4f}")
    
    # 2. 단일 필터
    print_section_header("필터: language='python'", "[FILTER]")
    
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3,
        where={"language": "python"}
    )
    
    print("\n결과 (Python만):")
    for i, (doc, meta, dist) in enumerate(zip(
        results['documents'][0],
        results['metadatas'][0],
        results['distances'][0]
    ), 1):
        similarity = 1 / (1 + dist)
        print(f"  [{i}] {doc}")
        print(f"      메타: {meta} | 유사도: {similarity:.4f}")
    
    # 3. 복합 필터 ($and)
    print_section_header("필터: language='python' AND level='beginner'", "[*]")
    
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3,
        where={
            "$and": [
                {"language": "python"},
                {"level": "beginner"}
            ]
        }
    )
    
    print("\n결과 (Python + 초급):")
    for i, (doc, meta, dist) in enumerate(zip(
        results['documents'][0],
        results['metadatas'][0],
        results['distances'][0]
    ), 1):
        similarity = 1 / (1 + dist)
        print(f"  [{i}] {doc}")
        print(f"      메타: {meta} | 유사도: {similarity:.4f}")
    
    # 4. 비교 연산자
    print_section_header("필터: year >= 2024", "[DATE]")
    
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3,
        where={"year": {"$gte": 2024}}
    )
    
    print("\n결과 (2024년 이후):")
    for i, (doc, meta, dist) in enumerate(zip(
        results['documents'][0],
        results['metadatas'][0],
        results['distances'][0]
    ), 1):
        similarity = 1 / (1 + dist)
        print(f"  [{i}] {doc}")
        print(f"      메타: {meta} | 유사도: {similarity:.4f}")
    
    # 필터 연산자 설명
    print_section_header("지원하는 필터 연산자", "[REF]")
    print("""
  ┌─────────────────────────────────────────────────────────┐
  │  비교 연산자                                            │
  │  ─────────────────────────────────────────────────────  │
  │  * $eq   : 같음          {"field": {"$eq": "value"}}   │
  │  * $ne   : 같지 않음      {"field": {"$ne": "value"}}   │
  │  * $gt   : 크다          {"field": {"$gt": 10}}        │
  │  * $gte  : 크거나 같다    {"field": {"$gte": 10}}       │
  │  * $lt   : 작다          {"field": {"$lt": 10}}        │
  │  * $lte  : 작거나 같다    {"field": {"$lte": 10}}       │
  │  * $in   : 포함          {"field": {"$in": ["a","b"]}} │
  └─────────────────────────────────────────────────────────┘
  
  ┌─────────────────────────────────────────────────────────┐
  │  논리 연산자                                            │
  │  ─────────────────────────────────────────────────────  │
  │  * $and  : 모두 만족      {"$and": [조건1, 조건2]}      │
  │  * $or   : 하나만 만족    {"$or": [조건1, 조건2]}       │
  └─────────────────────────────────────────────────────────┘
    """)
    
    # 핵심 포인트
    print_key_points([
        "- 메타데이터: 문서와 함께 저장되는 구조화된 정보",
        "- 벡터 검색 + 필터 = 의미 검색 + 조건 검색 결합",
        "- 실무 활용: 날짜, 카테고리, 사용자별 필터링",
        "- where 파라미터로 DB 레벨에서 필터링 (효율적)",
        "- 주의: 필터가 너무 제한적이면 결과 없을 수 있음"
    ])


# ============================================================================
# 실습 5: 문서 관리 시스템
# ============================================================================

def demo_document_manager():
    """실습 5: 실전 예제 - 문서 관리 시스템"""
    print("\n" + "="*80)
    print("[5] 실습 5: 실전 예제 - 문서 관리 시스템")
    print("="*80)
    print("목표: 실제 애플리케이션에서 Vector DB를 활용하는 패턴 익히기")
    print("핵심: 클래스 설계, 배치 처리, 검색 결과 가공")
    
    if not os.getenv("OPENAI_API_KEY"):
        print("\n[!] OPENAI_API_KEY 환경변수를 설정해주세요!")
        return
    
    # 문서 관리자 초기화
    print_section_header("DocumentManager 초기화", "[SETUP]")
    
    doc_manager = DocumentManager("knowledge_base", reset=True)
    
    # 샘플 문서 추가
    print_section_header("지식 베이스 구축", "[LIST]")
    
    knowledge_base = [
        {
            "text": "Python은 읽기 쉬운 문법과 강력한 라이브러리로 데이터 과학과 웹 개발에 널리 사용됩니다.",
            "metadata": {"topic": "programming", "language": "python", "difficulty": "beginner"}
        },
        {
            "text": "머신러닝은 데이터에서 패턴을 학습하여 예측하는 인공지능의 한 분야입니다.",
            "metadata": {"topic": "ai", "subtopic": "machine_learning", "difficulty": "intermediate"}
        },
        {
            "text": "딥러닝은 다층 신경망을 사용하여 복잡한 패턴을 학습하는 머신러닝의 하위 분야입니다.",
            "metadata": {"topic": "ai", "subtopic": "deep_learning", "difficulty": "advanced"}
        },
        {
            "text": "React는 Facebook에서 만든 사용자 인터페이스 구축을 위한 JavaScript 라이브러리입니다.",
            "metadata": {"topic": "programming", "language": "javascript", "difficulty": "intermediate"}
        },
        {
            "text": "SQL은 관계형 데이터베이스에서 데이터를 관리하고 조회하는 표준 언어입니다.",
            "metadata": {"topic": "database", "language": "sql", "difficulty": "beginner"}
        },
        {
            "text": "Vector DB는 고차원 벡터를 효율적으로 저장하고 유사도 검색을 수행하는 데이터베이스입니다.",
            "metadata": {"topic": "database", "subtopic": "vector_db", "difficulty": "intermediate"}
        },
    ]
    
    print(f"\n추가할 문서: {len(knowledge_base)}개")
    
    # 배치 추가
    texts = [d["text"] for d in knowledge_base]
    metadatas = [d["metadata"] for d in knowledge_base]
    
    doc_ids = doc_manager.add_documents_batch(texts, metadatas)
    
    print(f"[OK] 문서 추가 완료")
    print(f"   통계: {doc_manager.get_stats()}")
    
    # 다양한 검색 시나리오
    search_scenarios = [
        {
            "name": "기본 검색",
            "query": "프로그래밍을 배우고 싶어요",
            "filter": None
        },
        {
            "name": "AI 관련 검색",
            "query": "인공지능의 학습 방법은?",
            "filter": {"topic": "ai"}
        },
        {
            "name": "초급자용 콘텐츠",
            "query": "쉽게 시작할 수 있는 언어",
            "filter": {"difficulty": "beginner"}
        },
    ]
    
    for scenario in search_scenarios:
        print_section_header(f"검색: {scenario['name']}", "[>>>]")
        print(f"\n쿼리: '{scenario['query']}'")
        if scenario['filter']:
            print(f"필터: {scenario['filter']}")
        
        results = doc_manager.search(
            query=scenario['query'],
            n_results=3,
            where=scenario['filter']
        )
        
        print(f"\n결과:")
        for result in results:
            bar = visualize_similarity_bar(result.similarity, SIMILARITY_BAR_WIDTH)
            interpretation = interpret_l2_distance(result.distance)
            print(f"\n  [{result.rank}위] 유사도: {result.similarity:.4f} {interpretation}")
            print(f"       {bar}")
            print(f"       [DOC] {truncate_text(result.content, 60)}")
            print(f"       [TAG] {result.metadata}")
    
    # 실무 패턴 설명
    print_section_header("실무에서 사용할 수 있는 패턴", "[TIP]")
    print("""
  ┌─────────────────────────────────────────────────────────┐
  │  [PATTERN] DocumentManager 클래스 활용                  │
  │  ─────────────────────────────────────────────────────  │
  │                                                         │
  │  이 코드는 실무에서 그대로 사용 가능합니다:             │
  │                                                         │
  │  1. 초기화                                              │
  │     manager = DocumentManager("my_docs", reset=False)   │
  │                                                         │
  │  2. 문서 추가 (단건)                                    │
  │     doc_id = manager.add_document(                      │
  │         text="내용",                                    │
  │         metadata={"category": "tech"}                   │
  │     )                                                   │
  │                                                         │
  │  3. 문서 추가 (배치) ← 효율적!                          │
  │     doc_ids = manager.add_documents_batch(              │
  │         texts=[...],                                    │
  │         metadatas=[...]                                 │
  │     )                                                   │
  │                                                         │
  │  4. 검색 (필터 포함)                                    │
  │     results = manager.search(                           │
  │         query="질문",                                   │
  │         n_results=5,                                    │
  │         where={"category": "tech"}                      │
  │     )                                                   │
  │                                                         │
  │  5. 결과 사용                                           │
  │     for r in results:                                   │
  │         print(r.content, r.similarity, r.metadata)      │
  │                                                         │
  │  [TIP] 확장 아이디어:                                   │
  │  * update_document(doc_id, new_text)                    │
  │  * delete_document(doc_id)                              │
  │  * export_to_json() / import_from_json()                │
  └─────────────────────────────────────────────────────────┘
    """)
    
    # 핵심 포인트
    print_key_points([
        "- 클래스 캡슐화: 재사용성과 유지보수성 향상",
        "- 배치 임베딩: API 호출 최소화 (100배 효율)",
        "- 메타데이터 설계: 다양한 검색 시나리오 지원",
        "- SearchResult 클래스: 결과 가공 편리",
        "- 실무 적용: 이 코드를 프로젝트에 바로 사용 가능!"
    ])


# ============================================================================
# 메인 실행
# ============================================================================

def main():
    """챕터 2 실행"""
    print("\n" + "="*80)
    print("[Chapter 2] 실전 활용 ⭐")
    print("="*80)
    
    print("\n학습 목표:")
    print("  • 메타데이터 필터링으로 정밀 검색")
    print("  • DocumentManager 패턴 익히기")
    print("  • 배치 처리로 효율성 극대화")
    print("  • 실무에 바로 적용 가능한 코드 학습")
    
    if not os.getenv("OPENAI_API_KEY"):
        print("\n[!] OPENAI_API_KEY가 설정되지 않았습니다!")
        return
    
    try:
        # 실습 4: 메타데이터 필터링
        demo_metadata_filtering()
        
        # 실습 5: 문서 관리 시스템
        demo_document_manager()
        
        # 완료 메시지
        print("\n" + "="*80)
        print("[OK] Chapter 2 완료!")
        print("="*80)
        
        print("\n[요약]")
        print("  • 메타데이터 필터: 의미 검색 + 조건 검색")
        print("  • DocumentManager: 실무 패턴 클래스")
        print("  • 배치 처리: 효율적인 대량 문서 추가")
        print("  • 이 코드를 프로젝트에 바로 활용하세요!")
        
        print("\n[다음 단계]")
        print("  • Chapter 3: ANN 알고리즘, 대용량 처리 전략")
        print("  • python chapter3_advanced_topics.py")
        print("  • 또는 lab03: RAG 시스템 구축")
        
    except Exception as e:
        print(f"\n[X] 오류 발생: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()


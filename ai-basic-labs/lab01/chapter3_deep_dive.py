"""
[Chapter 3] 깊이 이해하기
- 실습 6: 임베딩 시각화 (t-SNE)
- 실습 7: 오픈소스 임베딩 모델 (Sentence Transformers)
- 실습 8: 임베딩 모델 비교 (small vs large)
- 실습 9: 한글-영어 임베딩 비교 (다국어 정렬)

학습 목표:
• 고차원 벡터를 2D로 시각화하여 이해
• OpenAI 외 오픈소스 대안 탐색
• 모델 선택 시 고려사항 (품질, 비용, 속도)
• 한글 RAG 구축 시 주의사항

실행:
  python chapter3_deep_dive.py
"""

import os
import sys
from pathlib import Path
from typing import List
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
    print_key_points,
    visualize_similarity_bar,
    cosine_similarity,
    interpret_cosine_similarity,
    get_openai_client,
)


# ============================================================================
# 임베딩 생성기 (재사용)
# ============================================================================

class EmbeddingGenerator:
    """OpenAI 임베딩 생성기"""
    
    def __init__(self, api_key: str = None):
        self.client = get_openai_client(api_key)
        self.model = "text-embedding-3-small"
    
    def get_embedding(self, text: str) -> List[float]:
        """단일 텍스트의 임베딩 생성"""
        try:
            response = self.client.embeddings.create(model=self.model, input=text)
            return response.data[0].embedding
        except Exception as e:
            print(f"\n[!] 임베딩 생성 실패: {e}")
            raise
    
    def get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """배치 임베딩 생성"""
        try:
            response = self.client.embeddings.create(model=self.model, input=texts)
            return [data.embedding for data in response.data]
        except Exception as e:
            print(f"\n[!] 배치 임베딩 실패: {e}")
            raise


# ============================================================================
# 실습 6: 임베딩 시각화
# ============================================================================

def demo_embedding_visualization():
    """실습 6: 임베딩 시각화 - t-SNE로 벡터 공간 이해하기"""
    print("\n" + "="*80)
    print("[6] 실습 6: 임베딩 시각화 - t-SNE로 벡터 공간 이해하기")
    print("="*80)
    print("목표: 고차원 임베딩을 2D로 시각화하여 의미적 클러스터 확인")
    print("핵심: 비슷한 의미의 텍스트는 시각화에서도 가까이 모임")
    
    # t-SNE 개념
    print_section_header("차원 축소란?", "[INFO]")
    print("""
  ┌─────────────────────────────────────────────────────────┐
  │  [!] 문제: 1536차원을 어떻게 이해할까?                   │
  │  ─────────────────────────────────────────────────────  │
  │  • 임베딩: [0.1, -0.3, ..., 0.2] ← 1536개 숫자          │
  │  • 사람이 직접 해석 불가능 → 2D 시각화 필요!            │
  │                                                         │
  │  [ALGO] 대표적인 차원 축소 알고리즘                      │
  │  ─────────────────────────────────────────────────────  │
  │  1. t-SNE: 가까운 점 관계 보존, 클러스터 시각화 최적     │
  │  2. UMAP: t-SNE보다 빠름, 대용량에 적합                 │
  │  3. PCA: 가장 빠름, 선형 변환                           │
  │                                                         │
  │  [TIP] 클러스터 확인에는 t-SNE/UMAP 추천                │
  └─────────────────────────────────────────────────────────┘
    """)
    
    if not os.getenv("OPENAI_API_KEY"):
        print("\n[!] OPENAI_API_KEY 환경변수를 설정해주세요!")
        return
    
    # 라이브러리 확인
    try:
        from sklearn.manifold import TSNE
        tsne_available = True
    except ImportError:
        tsne_available = False
        print("\n[!] scikit-learn이 설치되지 않았습니다.")
        print("   설치: pip install scikit-learn")
    
    try:
        import matplotlib.pyplot as plt
        matplotlib_available = True
    except ImportError:
        matplotlib_available = False
        print("\n[!] matplotlib이 설치되지 않았습니다.")
        print("   설치: pip install matplotlib")
    
    # 샘플 텍스트 (카테고리별)
    texts_by_category = {
        "프로그래밍": [
            "Python is a great programming language",
            "JavaScript is used for web development",
            "Java is popular for enterprise applications",
            "C++ is used for system programming",
        ],
        "음식": [
            "Pizza is my favorite Italian food",
            "Sushi is a traditional Japanese dish",
            "Tacos are delicious Mexican food",
            "Pasta with tomato sauce is amazing",
        ],
        "스포츠": [
            "Soccer is the most popular sport worldwide",
            "Basketball requires great athleticism",
            "Tennis is an individual sport",
            "Swimming is excellent exercise",
        ],
    }
    
    # 텍스트와 라벨 준비
    all_texts = []
    labels = []
    for category, texts in texts_by_category.items():
        all_texts.extend(texts)
        labels.extend([category] * len(texts))
    
    print(f"\n총 {len(all_texts)}개 텍스트 (3개 카테고리)")
    
    # 임베딩 생성
    generator = EmbeddingGenerator()
    print("\n[...] 임베딩 생성 중...")
    embeddings = generator.get_embeddings_batch(all_texts)
    embeddings_array = np.array(embeddings)
    print(f"[OK] 임베딩 완료: {embeddings_array.shape}")
    
    if tsne_available and matplotlib_available:
        # t-SNE 실행
        print("\n[...] t-SNE 차원 축소 중...")
        tsne = TSNE(n_components=2, perplexity=5, random_state=42, max_iter=1000)
        embeddings_2d = tsne.fit_transform(embeddings_array)
        print(f"[OK] t-SNE 완료: {embeddings_2d.shape}")
        
        # ASCII 시각화
        print_section_header("ASCII 산점도", "[CHART]")
        
        # 좌표 정규화
        x_min, x_max = embeddings_2d[:, 0].min(), embeddings_2d[:, 0].max()
        y_min, y_max = embeddings_2d[:, 1].min(), embeddings_2d[:, 1].max()
        
        # 그리드 생성
        grid_width, grid_height = 60, 20
        grid = [[' ' for _ in range(grid_width)] for _ in range(grid_height)]
        
        # 점 배치
        category_symbols = {"프로그래밍": 'P', "음식": 'F', "스포츠": 'S'}
        for i, label in enumerate(labels):
            x_norm = int((embeddings_2d[i, 0] - x_min) / (x_max - x_min + 1e-10) * (grid_width - 1))
            y_norm = int((embeddings_2d[i, 1] - y_min) / (y_max - y_min + 1e-10) * (grid_height - 1))
            y_norm = grid_height - 1 - y_norm
            grid[y_norm][x_norm] = category_symbols[label]
        
        # 출력
        print("\n  +" + "-" * grid_width + "+")
        for row in grid:
            print("  |" + "".join(row) + "|")
        print("  +" + "-" * grid_width + "+")
        print(f"  범례: P=프로그래밍, F=음식, S=스포츠")
        
        # 차트 파일 저장
        try:
            import platform
            system = platform.system()
            
            if system == "Windows":
                plt.rcParams['font.family'] = 'Malgun Gothic'
            elif system == "Darwin":
                plt.rcParams['font.family'] = 'AppleGothic'
            else:
                plt.rcParams['font.family'] = 'NanumGothic'
            
            plt.rcParams['axes.unicode_minus'] = False
            
            plt.figure(figsize=(10, 8))
            
            color_map = {"프로그래밍": "blue", "음식": "green", "스포츠": "red"}
            
            for category in texts_by_category.keys():
                mask = [l == category for l in labels]
                indices = [i for i, m in enumerate(mask) if m]
                plt.scatter(
                    embeddings_2d[indices, 0],
                    embeddings_2d[indices, 1],
                    c=color_map[category],
                    label=category,
                    s=100,
                    alpha=0.7
                )
            
            plt.title("t-SNE 임베딩 시각화")
            plt.xlabel("차원 1")
            plt.ylabel("차원 2")
            plt.legend()
            plt.tight_layout()
            
            output_path = Path(__file__).parent / "embedding_tsne_demo.png"
            plt.savefig(output_path, dpi=150)
            plt.close()
            
            print(f"\n[OK] 차트 저장: {output_path}")
            
        except Exception as e:
            print(f"\n[!] 차트 저장 실패: {e}")
    
    # 핵심 포인트
    print_key_points([
        "- 차원 축소: 1536차원 → 2D로 시각화",
        "- t-SNE: 클러스터 시각화에 최적",
        "- 해석 주의: 시각화 거리 ≠ 실제 유사도",
        "- 활용: 데이터 품질 확인, 이상치 탐지"
    ], "시각화 핵심 포인트")


# ============================================================================
# 실습 7: 오픈소스 임베딩 모델
# ============================================================================

def demo_sentence_transformers():
    """실습 7: Sentence Transformers 소개"""
    print("\n" + "="*80)
    print("[7] 실습 7: 오픈소스 임베딩 모델 - Sentence Transformers")
    print("="*80)
    print("목표: OpenAI 외 무료 오픈소스 대안 탐색")
    print("핵심: 비용 절감, 오프라인 사용, 커스터마이징")
    
    # Sentence Transformers 소개
    print_section_header("Sentence Transformers란?", "[INFO]")
    print("""
  ┌─────────────────────────────────────────────────────────┐
  │  [LIB] Sentence Transformers                            │
  │  ─────────────────────────────────────────────────────  │
  │  • Hugging Face 기반 문장 임베딩 라이브러리             │
  │  • 수백 개의 사전 훈련된 모델 제공                      │
  │  • MIT 라이선스 (상업적 사용 가능)                      │
  │  • 로컬 실행 → API 비용 없음!                          │
  │                                                         │
  │  [설치]                                                 │
  │  pip install sentence-transformers                      │
  └─────────────────────────────────────────────────────────┘
    """)
    
    # OpenAI vs Sentence Transformers
    print_section_header("OpenAI vs Sentence Transformers", "[vs]")
    print("""
  ┌─────────────────────────────────────────────────────────┐
  │  항목         │ OpenAI            │ Sentence Transformers│
  │  ────────────┼───────────────────┼──────────────────── │
  │  비용        │ $0.02/1M 토큰     │ 무료 (로컬)          │
  │  속도        │ 네트워크 지연     │ GPU시 빠름           │
  │  품질        │ 최상급            │ 모델에 따라 다름     │
  │  오프라인    │ ✗ 불가            │ ✓ 가능              │
  │  파인튜닝    │ ✗ 불가            │ ✓ 가능              │
  │                                                         │
  │  [TIP] OpenAI: 최고 품질, 빠른 시작                      │
  │       ST: 비용 절감, 오프라인, 커스터마이징              │
  └─────────────────────────────────────────────────────────┘
    """)
    
    # 실제 사용 예시
    print_section_header("Sentence Transformers 사용", "[CODE]")
    
    try:
        from sentence_transformers import SentenceTransformer
        st_available = True
    except ImportError:
        st_available = False
        print("\n[!] sentence-transformers가 설치되지 않았습니다.")
        print("   설치: pip install sentence-transformers")
    
    if st_available:
        print("\n[...] 모델 로딩 중...")
        
        try:
            model = SentenceTransformer('all-MiniLM-L6-v2')
            
            sentences = [
                "Python is a programming language",
                "Java is also a programming language",
                "I love eating pizza",
            ]
            
            print(f"[OK] 모델 로드 완료: all-MiniLM-L6-v2")
            
            # 임베딩 생성
            embeddings = model.encode(sentences)
            
            print(f"\n임베딩 결과:")
            print(f"  Shape: {embeddings.shape}")
            print(f"  차원: {embeddings.shape[1]}")
            
            # 유사도 계산
            sim_1_2 = cosine_similarity(embeddings[0].tolist(), embeddings[1].tolist())
            sim_1_3 = cosine_similarity(embeddings[0].tolist(), embeddings[2].tolist())
            
            print(f"\n코사인 유사도:")
            print(f"  Python vs Java: {sim_1_2:.4f} (프로그래밍)")
            print(f"  Python vs Pizza: {sim_1_3:.4f} (다른 주제)")
            
        except Exception as e:
            print(f"\n[!] 실행 오류: {e}")
    
    # 핵심 포인트
    print_key_points([
        "- Sentence Transformers: 무료 오픈소스 라이브러리",
        "- all-MiniLM-L6-v2: 가볍고 빠름 (384차원)",
        "- OpenAI vs ST: 품질 vs 비용/커스터마이징",
        "- 파인튜닝 가능 (도메인 특화)",
        "- 다국어: paraphrase-multilingual 모델"
    ], "Sentence Transformers 핵심 포인트")


# ============================================================================
# 실습 8: 임베딩 모델 비교
# ============================================================================

def demo_embedding_model_comparison():
    """실습 8: 임베딩 모델 비교"""
    print("\n" + "="*80)
    print("[8] 실습 8: 임베딩 모델 비교 - small vs large")
    print("="*80)
    print("목표: 모델 선택 시 고려사항 이해")
    print("핵심: 품질, 비용, 속도의 Trade-off")
    
    # OpenAI 모델 비교
    print_section_header("OpenAI 임베딩 모델 비교", "[CMP]")
    print("""
  ┌─────────────────────────────────────────────────────────┐
  │  모델                    │ 차원  │ 가격(/1M토큰) │ 특징  │
  │  ───────────────────────┼──────┼──────────────┼────── │
  │  text-embedding-3-small │ 1536 │ $0.02        │ 가성비│
  │  text-embedding-3-large │ 3072 │ $0.13        │ 고품질│
  │  text-embedding-ada-002 │ 1536 │ $0.10        │ 레거시│
  └─────────────────────────────────────────────────────────┘
    """)
    
    # 비용 계산
    print_section_header("비용 계산 예시", "[CALC]")
    print("""
  ┌─────────────────────────────────────────────────────────┐
  │  [시나리오] RAG 시스템 운영                              │
  │  • 문서 10,000개 (각 500 토큰) = 5M 토큰                │
  │  • 월간 쿼리 30,000개 (각 50 토큰) = 1.5M 토큰          │
  │                                                         │
  │  초기 인덱싱:                                            │
  │  • small: $0.10  │  large: $0.65                       │
  │                                                         │
  │  월간 비용:                                             │
  │  • small: $0.03  │  large: $0.20                       │
  │                                                         │
  │  [결론] Small로도 충분! Large는 정밀도 중요시만 사용     │
  └─────────────────────────────────────────────────────────┘
    """)
    
    # 실제 비교
    if os.getenv("OPENAI_API_KEY"):
        print_section_header("실제 비교 실험", "[EXP]")
        
        try:
            client = get_openai_client()
            
            test_texts = [
                "What is machine learning?",
                "Machine learning is a subset of AI",
                "I love eating pizza",
            ]
            
            # Small 모델
            response_small = client.embeddings.create(
                model="text-embedding-3-small",
                input=test_texts
            )
            embeddings_small = [d.embedding for d in response_small.data]
            
            # Large 모델
            response_large = client.embeddings.create(
                model="text-embedding-3-large",
                input=test_texts
            )
            embeddings_large = [d.embedding for d in response_large.data]
            
            # 유사도 비교
            sim_small = cosine_similarity(embeddings_small[0], embeddings_small[1])
            sim_large = cosine_similarity(embeddings_large[0], embeddings_large[1])
            
            print(f"\n유사도 비교 (텍스트1 vs 텍스트2):")
            print(f"  Small: {sim_small:.4f}")
            print(f"  Large: {sim_large:.4f}")
            print(f"\n차원:")
            print(f"  Small: {len(embeddings_small[0])}")
            print(f"  Large: {len(embeddings_large[0])}")
            
        except Exception as e:
            print(f"\n[!] 비교 실패: {e}")
    
    # 모델 선택 가이드
    print_section_header("모델 선택 가이드", "[GUIDE]")
    print("""
  ┌─────────────────────────────────────────────────────────┐
  │  Q1. 비용이 가장 중요한가?                              │
  │   ├─ YES → Q2. GPU 있는가?                              │
  │   │         ├─ YES → Sentence Transformers              │
  │   │         └─ NO → text-embedding-3-small              │
  │   └─ NO → Q3. 최고 품질 필요?                           │
  │            ├─ YES → text-embedding-3-large              │
  │            └─ NO → text-embedding-3-small               │
  │                                                         │
  │  [TIP] 대부분의 RAG는 small로 충분합니다!               │
  └─────────────────────────────────────────────────────────┘
    """)
    
    # 핵심 포인트
    print_key_points([
        "- small: 가성비 최고, 대부분 충분",
        "- large: 법률/의료 등 정밀도 중요시만",
        "- 비용: small $0.02 vs large $0.13 (6.5배)",
        "- 품질 차이: 실무에서 체감 어려움",
        "- 오픈소스: bge-large, e5-large도 고품질"
    ], "모델 비교 핵심 포인트")


# ============================================================================
# 실습 9: 한글-영어 임베딩 비교
# ============================================================================

def demo_korean_english_comparison():
    """실습 9: 한글-영어 임베딩 비교"""
    print("\n" + "="*80)
    print("[9] 실습 9: 한글-영어 임베딩 비교 - 다국어 정렬")
    print("="*80)
    print("목표: 한글과 영어가 같은 의미일 때 임베딩이 얼마나 유사한지 확인")
    print("핵심: 다국어 임베딩의 Cross-lingual Alignment 품질")
    
    # 다국어 정렬
    print_section_header("다국어 의미 정렬이란?", "[INFO]")
    print("""
  ┌─────────────────────────────────────────────────────────┐
  │  [문제] 한글 질문으로 영어 문서를 검색할 수 있을까?      │
  │                                                         │
  │  이상적인 다국어 임베딩:                                 │
  │  • "파이썬은 프로그래밍 언어다"                         │
  │  • "Python is a programming language"                  │
  │  → 두 벡터가 가까워야 함!                               │
  │                                                         │
  │  [중요성]                                               │
  │  • 한글 RAG에서 영어 기술 문서 검색                     │
  │  • 다국어 챗봇 구현                                     │
  │  • 번역 없이 교차 언어 검색                             │
  └─────────────────────────────────────────────────────────┘
    """)
    
    if not os.getenv("OPENAI_API_KEY"):
        print("\n[!] OPENAI_API_KEY를 설정해주세요!")
        return
    
    # 테스트 문장 쌍
    test_pairs = [
        {
            "korean": "파이썬은 프로그래밍 언어입니다",
            "english": "Python is a programming language",
            "category": "프로그래밍"
        },
        {
            "korean": "나는 피자를 좋아합니다",
            "english": "I love eating pizza",
            "category": "음식"
        },
        {
            "korean": "머신러닝은 인공지능의 한 분야입니다",
            "english": "Machine learning is a subset of AI",
            "category": "AI"
        },
    ]
    
    print_section_header("테스트 문장 쌍", "[DATA]")
    for pair in test_pairs:
        print(f"\n  {pair['category']}:")
        print(f"    KO: {pair['korean']}")
        print(f"    EN: {pair['english']}")
    
    # OpenAI 임베딩 테스트
    print_section_header("OpenAI 다국어 정렬 테스트", "[TEST]")
    
    generator = EmbeddingGenerator()
    
    all_texts = []
    for pair in test_pairs:
        all_texts.extend([pair['korean'], pair['english']])
    
    print("\n[...] 임베딩 생성 중...")
    embeddings = generator.get_embeddings_batch(all_texts)
    print(f"[OK] {len(embeddings)}개 임베딩 완료")
    
    # 유사도 분석
    print(f"\n[분석] 한글-영어 쌍별 유사도:")
    print(f"{'─'*60}")
    
    same_meaning_sims = []
    for i, pair in enumerate(test_pairs):
        ko_emb = embeddings[i * 2]
        en_emb = embeddings[i * 2 + 1]
        sim = cosine_similarity(ko_emb, en_emb)
        same_meaning_sims.append(sim)
        
        bar = visualize_similarity_bar(sim, 20)
        interpretation = interpret_cosine_similarity(sim)
        print(f"\n{pair['category']}: {bar} {sim:.4f}")
        print(f"  {interpretation}")
    
    avg_sim = np.mean(same_meaning_sims)
    print(f"\n평균 유사도: {avg_sim:.4f}")
    
    if avg_sim > 0.8:
        print(f"→ [v] 뛰어난 다국어 정렬! 한글 RAG에 적합")
    elif avg_sim > 0.6:
        print(f"→ [~] 양호한 정렬")
    else:
        print(f"→ [x] 다국어 정렬 약함")
    
    # 실무 가이드
    print_section_header("한글 RAG 실무 가이드", "[GUIDE]")
    print("""
  ┌─────────────────────────────────────────────────────────┐
  │  [시나리오별 권장 모델]                                  │
  │                                                         │
  │  1. 한글 문서 + 한글 질문                                │
  │     → OpenAI text-embedding-3-small (권장)              │
  │                                                         │
  │  2. 영어 문서 + 한글 질문 (또는 반대)                   │
  │     → OpenAI (다국어 정렬 우수)                         │
  │     → paraphrase-multilingual-MiniLM (비용 절감)        │
  │                                                         │
  │  3. 한글 전용 + 비용 최소화                             │
  │     → paraphrase-multilingual-MiniLM (무료)             │
  │     → KoSimCSE (한글 특화)                              │
  │                                                         │
  │  [주의] 영어 중심 모델은 한글 성능 저하!                 │
  └─────────────────────────────────────────────────────────┘
    """)
    
    # 핵심 포인트
    print_key_points([
        "- 다국어 정렬: 같은 의미의 다른 언어가 벡터 공간에서 가까운 정도",
        "- OpenAI: 다국어 정렬 우수, 한글 RAG에 안전",
        "- multilingual 모델: 명시적 다국어 학습 필요",
        "- 영어 중심 모델: 한글 성능 저하 가능",
        "- 실무: 반드시 한글 테스트로 검증 후 선택!"
    ], "한글-영어 비교 핵심 포인트")


# ============================================================================
# 메인 실행
# ============================================================================

def main():
    """챕터 3 실행"""
    print("\n" + "="*80)
    print("[Chapter 3] 깊이 이해하기")
    print("="*80)
    print("\n학습 목표:")
    print("  • 임베딩 벡터를 시각화하여 이해")
    print("  • 오픈소스 대안 탐색 (비용 절감)")
    print("  • 모델 선택 기준 파악 (품질 vs 비용)")
    print("  • 한글 RAG 구축 시 주의사항")
    
    # API 키 확인
    api_key_available = bool(os.getenv("OPENAI_API_KEY"))
    if not api_key_available:
        print("\n[!] OPENAI_API_KEY가 없습니다.")
        print("    일부 실습은 개념 설명만 제공됩니다.")
    
    try:
        # 실습 6: 시각화
        if api_key_available:
            demo_embedding_visualization()
        else:
            print("\n[SKIP] 실습 6: API 키 필요")
        
        # 실습 7: Sentence Transformers
        demo_sentence_transformers()
        
        # 실습 8: 모델 비교
        demo_embedding_model_comparison()
        
        # 실습 9: 한글-영어 비교
        if api_key_available:
            demo_korean_english_comparison()
        else:
            print("\n[SKIP] 실습 9: API 키 필요")
        
        # 완료 메시지
        print("\n" + "="*80)
        print("[OK] Chapter 3 완료!")
        print("="*80)
        
        print("\n[전체 요약]")
        print("  • 시각화: t-SNE로 클러스터 확인")
        print("  • 오픈소스: Sentence Transformers로 비용 절감")
        print("  • 모델 선택: 대부분 small로 충분")
        print("  • 한글: OpenAI가 가장 안전한 선택")
        
        print("\n[다음 단계]")
        print("  • lab02: Vector DB (ChromaDB)로 대용량 검색")
        print("  • lab03: RAG 시스템 구축 (검색 + LLM)")
        print("  • lab04: Agent 시스템 (고급)")
        
    except Exception as e:
        print(f"\n[X] 오류 발생: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()


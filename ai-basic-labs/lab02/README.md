# Lab 02: Vector Database (ChromaDB)

임베딩 벡터를 효율적으로 저장하고 검색하는 Vector DB의 원리와 실전 활용법을 학습합니다.

## 📚 학습 구조

이 실습은 **3개 챕터**로 구성되어 있으며, 순서대로 진행하는 것을 권장합니다.

```
Chapter 1 (기초) → Chapter 2 (실전) → Chapter 3 (심화)
   30분             25분             40분
```

### 📖 Chapter 1: Vector DB 기초

**파일:** `chapter1_vectordb_basics.py`

Lab01에서 배운 임베딩을 Vector DB와 연결하고, 기본 CRUD 작업을 익힙니다.

**실습 내용:**
- 실습 1: 임베딩 기초 (Lab01 복습)
  - 텍스트 → 벡터 변환
  - 의미적 유사성 확인
  - 코사인 유사도 vs L2 거리
  
- 실습 2: Vector DB 기본 작업
  - ChromaDB 초기화
  - 문서 추가 (Add)
  - 유사도 검색 (Query)
  
- 실습 3: 거리와 유사도 이해하기
  - L2 거리 스코어 해석
  - 거리 DOWN = 유사도 UP
  - 데이터셋별 임계값 차이

**실행:**
```bash
python chapter1_vectordb_basics.py
```

**필요사항:**
- `OPENAI_API_KEY` 환경변수
- ChromaDB 설치
- ⚠️ Windows + Python 3.13: 호환 이슈 가능, Python 3.11 권장

---

### 🎯 Chapter 2: 실전 활용 ⭐ (메인)

**파일:** `chapter2_practical_usage.py`

**이 챕터가 가장 중요합니다!** 실무에서 바로 사용할 수 있는 패턴을 학습합니다.

**실습 내용:**
- 실습 4: 메타데이터 필터링
  - 벡터 검색 + 조건 필터링
  - $and, $or, $gte 등 연산자
  - 카테고리, 날짜, 레벨별 필터
  
- 실습 5: 문서 관리 시스템 (실전 예제)
  - DocumentManager 클래스 패턴
  - 배치 임베딩으로 효율화
  - SearchResult 데이터 클래스
  - 다양한 검색 시나리오 구현

**실행:**
```bash
python chapter2_practical_usage.py
```

**핵심 가치:**
이 챕터의 `DocumentManager` 클래스는 실무 프로젝트에 그대로 활용할 수 있는 패턴입니다!

---

### 🔬 Chapter 3: 심화 이론 - 확장성과 최적화

**파일:** `chapter3_advanced_topics.py`

Vector DB가 어떻게 빠르게 작동하는지, 100만 건 이상 데이터를 어떻게 처리하는지 학습합니다.

**실습 내용:**
- 실습 6: ANN 인덱스 알고리즘 이해하기
  - HNSW (계층 그래프, ChromaDB 기본)
  - IVF (클러스터 기반)
  - PQ (벡터 압축)
  - 알고리즘 비교 및 선택 가이드
  
- 실습 7: 대용량 데이터 처리 전략
  - 배치 처리로 API 호출 100배 감소
  - 샤딩 (카테고리/시간 기반 분산)
  - 캐싱 (임베딩/결과 캐시)
  - 하드웨어 사양 가이드

**실행:**
```bash
python chapter3_advanced_topics.py
```

**특징:**
- 이론 중심이므로 **API 키 없이도 학습 가능**
- 실무 스케일업 시 필수 지식

---

## 🚀 빠른 시작

### 1. 환경 설정

```bash
# 필수 라이브러리 설치
pip install -r ../requirements.txt

# ChromaDB 설치
pip install chromadb

# .env 파일에 API 키 추가 (프로젝트 루트)
OPENAI_API_KEY=sk-...
```

### 2. ⚠️ Windows + Python 3.13 사용자

ChromaDB가 Python 3.13을 아직 공식 지원하지 않습니다.

**권장 해결책:**
```bash
# Python 3.11 또는 3.12로 다운그레이드
pyenv install 3.11.9
pyenv local 3.11.9

# 또는 conda 사용
conda create -n lab02 python=3.11
conda activate lab02
```

**대안:**
```bash
# 호환 버전 강제 설치 (시도해볼 가치 있음)
pip install chromadb==0.4.22 hnswlib==0.8.0 --force-reinstall
```

### 3. 순서대로 학습

```bash
# Chapter 1: 기초 (30분)
python chapter1_vectordb_basics.py

# Chapter 2: 실전 (25분) ⭐
python chapter2_practical_usage.py

# Chapter 3: 심화 (40분)
python chapter3_advanced_topics.py
```

---

## 📊 학습 흐름도

```
┌─────────────────────────────────────────────────────────┐
│  Chapter 1: 기초                                         │
│  ───────────────────────────────────────────────────── │
│  • Lab01 임베딩 복습                                    │
│  • ChromaDB 설치 및 초기화                              │
│  • Add/Query 기본 흐름                                  │
│  • L2 거리 vs 코사인 유사도                             │
└─────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│  Chapter 2: 실전 ⭐ (가장 중요!)                         │
│  ───────────────────────────────────────────────────── │
│  • 메타데이터 필터링 (조건부 검색)                      │
│  • DocumentManager 클래스 패턴                          │
│  • 배치 처리로 효율화                                   │
│  • 실무에 바로 쓸 수 있는 코드!                         │
└─────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│  Chapter 3: 심화                                         │
│  ───────────────────────────────────────────────────── │
│  • ANN 알고리즘 (HNSW, IVF, PQ)                         │
│  • 왜 Vector DB가 빠른가?                               │
│  • 100만 건 이상 처리 전략                              │
│  • 샤딩, 캐싱, 배치 처리                                │
└─────────────────────────────────────────────────────────┘
```

---

## 💡 핵심 개념 요약

### Vector DB란?
- 임베딩 벡터를 효율적으로 저장하고 검색하는 데이터베이스
- 일반 DB: 정확한 값 매칭 (`WHERE id = 123`)
- Vector DB: 유사도 기반 검색 (가장 비슷한 벡터 찾기)

### 컬렉션 (Collection)
- 벡터들의 그룹 (일반 DB의 테이블과 유사)
- 각 컬렉션마다 독립적인 인덱스

### L2 거리 (Euclidean Distance)
- 두 벡터 간의 직선 거리
- 값 범위: 0 ~ ∞ (0이 가장 유사)
- ChromaDB 기본 설정
- **중요:** 절대 수치보다 상대 순위가 중요!

### 메타데이터 필터링
- 벡터 검색 + 조건 검색 결합
- 예: `where={"category": "tech", "year": {"$gte": 2024}}`
- DB 레벨에서 필터링하므로 효율적

### ANN (Approximate Nearest Neighbor)
- 근사 최근접 이웃 검색
- 95~99% 정확도로 1000배 빠른 검색
- HNSW, IVF, PQ 등 다양한 알고리즘

---

## ⚠️ 주의사항

### 1. 거리 스코어 해석

```
⚠️ 중요: L2 거리 임계값은 데이터셋마다 다릅니다!

이 실습 예시:
- 거리 0.0 ~ 0.5: 매우 높은 관련성
- 거리 0.5 ~ 1.0: 높은 관련성
- 거리 1.0 ~ 1.5: 중간 관련성
- 거리 1.5 이상: 낮은 관련성

⚠️ 이 수치를 절대 일반화하지 마세요!
→ 자신의 데이터셋으로 분석 후 임계값 결정
```

### 2. Python 버전
- **Python 3.11 또는 3.12 권장**
- Python 3.13: ChromaDB 호환 이슈
- 설치 실패 시 WSL2 고려

### 3. 메모리 사용량
```
1536차원 벡터 메모리 계산:
- 벡터: 1536 × 4바이트 = 6KB/벡터
- HNSW 인덱스: 벡터 메모리 × 3배
- 100만 벡터 ≈ 18~30GB RAM 필요
```

---

## 🎓 학습 목표 체크리스트

### Chapter 1 완료 후
- [ ] ChromaDB 초기화 및 컬렉션 생성 이해
- [ ] 문서 추가 (Add) 및 검색 (Query) 기본 흐름 파악
- [ ] L2 거리와 코사인 유사도의 차이 이해
- [ ] 거리 스코어 해석 방법 숙지

### Chapter 2 완료 후 ⭐
- [ ] 메타데이터 필터링으로 조건부 검색 구현 가능
- [ ] DocumentManager 클래스 패턴 이해 및 활용 가능
- [ ] 배치 처리로 효율적인 문서 추가 가능
- [ ] SearchResult 데이터 클래스 활용 가능
- [ ] **이 코드를 실무에 적용할 수 있음!**

### Chapter 3 완료 후
- [ ] ANN 알고리즘의 필요성 이해
- [ ] HNSW, IVF, PQ 알고리즘 차이 파악
- [ ] HNSW 파라미터 튜닝 방법 숙지
- [ ] 대용량 데이터 처리 전략 이해
- [ ] 메모리 계산 및 하드웨어 사양 결정 가능

---

## 📁 파일 구조

```
lab02/
├── README.md                        # 이 파일
├── chapter1_vectordb_basics.py      # 챕터 1: 기초
├── chapter2_practical_usage.py      # 챕터 2: 실전 ⭐
├── chapter3_advanced_topics.py      # 챕터 3: 심화
└── chroma_db/                       # ChromaDB 데이터 (자동 생성)
```

---

## 🔗 다음 단계

### Lab 03: RAG 시스템
- Vector DB + LLM = 검색 증강 생성
- 실전 AI 애플리케이션 구축
- Hybrid 검색 (BM25 + 벡터)

### Lab 04: Agent 시스템
- Function Calling
- Multi-Agent 협업
- 고급 AI 애플리케이션

---

## 📚 참고 자료

- [ChromaDB Documentation](https://docs.trychroma.com/)
- [HNSW 논문](https://arxiv.org/abs/1603.09320)
- [OpenAI Embeddings Guide](https://platform.openai.com/docs/guides/embeddings)
- [MTEB Leaderboard](https://huggingface.co/spaces/mteb/leaderboard)

---

## 💬 트러블슈팅

### ChromaDB 설치 실패 (Windows)
```bash
# 해결 1: Python 3.11로 다운그레이드
pyenv install 3.11.9

# 해결 2: 호환 버전 설치
pip install chromadb==0.4.22 hnswlib==0.8.0

# 해결 3: WSL2 사용
wsl --install
```

### 메모리 부족 오류
```python
# 배치 크기 줄이기
BATCH_SIZE = 50  # 기본 100에서 감소
```

### 검색 결과가 이상함
```python
# 데이터 초기화 후 재시작
doc_manager = DocumentManager("my_collection", reset=True)
```

---

**Happy Learning! 🚀**

이 Lab을 마치면 실무에서 Vector DB를 활용할 수 있는 능력을 갖추게 됩니다!

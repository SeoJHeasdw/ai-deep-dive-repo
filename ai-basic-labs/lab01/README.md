# Lab 01: NLP 기초와 임베딩

텍스트 처리, 임베딩 생성, 의미 검색의 기초를 학습합니다.

## 📚 학습 구조

이 실습은 **3개 챕터**로 구성되어 있으며, 순서대로 진행하는 것을 권장합니다.

```
Chapter 1 (준비) → Chapter 2 (핵심) → Chapter 3 (심화)
   20분              30분              40분
```

### 📖 Chapter 1: 텍스트 이해의 기초

**파일:** `chapter1_text_basics.py`

GPT가 텍스트를 어떻게 이해하는지, 전처리가 왜 필요한지 배웁니다.

**실습 내용:**
- 실습 1: tiktoken으로 토큰 이해하기
  - GPT의 토큰화 방식
  - 한글 vs 영어 토큰 비교
  - API 비용 계산 방법
  
- 실습 2: NLTK 전처리 파이프라인
  - 토큰화, 불용어 제거, 표제어 추출
  - BM25 vs 임베딩 검색의 전처리 필요성 ⚠️

**실행:**
```bash
python chapter1_text_basics.py
```

**필요 라이브러리:**
- tiktoken
- nltk

---

### 🎯 Chapter 2: 임베딩의 핵심 사이클 ⭐ (메인)

**파일:** `chapter2_embedding_core.py`

**이것이 RAG의 핵심입니다!** 텍스트를 벡터로 변환하고 검색하는 전체 사이클을 실습합니다.

**실습 내용:**
- 실습 3: OpenAI 임베딩 생성
  - 텍스트 → 벡터 변환
  - 배치 처리로 효율화
  - L2 정규화 이해
  
- 실습 4: 코사인 유사도 계산
  - 벡터 간 유사성 측정
  - 1:N, N:M 유사도 계산
  - 정규화된 벡터의 내적 = 코사인 유사도
  
- 실습 5: 간단한 검색 엔진 구현
  - 문서 인덱싱
  - 의미 기반 검색
  - 키워드 vs 의미 검색 비교

**실행:**
```bash
python chapter2_embedding_core.py
```

**필요사항:**
- `OPENAI_API_KEY` 환경변수 설정 (프로젝트 루트 `.env` 파일)

---

### 🔬 Chapter 3: 깊이 이해하기

**파일:** `chapter3_deep_dive.py`

임베딩을 시각화하고, 다양한 모델을 비교하며, 실무 선택 기준을 배웁니다.

**실습 내용:**
- 실습 6: 임베딩 시각화 (t-SNE)
  - 1536차원 → 2D 시각화
  - 의미적 클러스터 확인
  - PNG 파일 저장
  
- 실습 7: Sentence Transformers
  - 무료 오픈소스 대안
  - OpenAI vs 오픈소스 비교
  - 비용 절감 전략
  
- 실습 8: 임베딩 모델 비교
  - small vs large 성능/비용 분석
  - 모델 선택 의사결정 트리
  - MTEB 벤치마크 해석
  
- 실습 9: 한글-영어 임베딩 비교 🆕
  - 다국어 의미 정렬(Cross-lingual Alignment)
  - 한글 RAG 구축 가이드
  - 모델별 다국어 품질 비교

**실행:**
```bash
python chapter3_deep_dive.py
```

**추가 라이브러리 (선택):**
- scikit-learn (t-SNE 시각화)
- matplotlib (차트 저장)
- sentence-transformers (오픈소스 모델)

---

## 🚀 빠른 시작

### 1. 환경 설정

```bash
# 필수 라이브러리 설치
pip install -r ../requirements.txt

# .env 파일에 API 키 추가 (프로젝트 루트)
OPENAI_API_KEY=sk-...
```

### 2. 순서대로 학습

```bash
# Chapter 1: 텍스트 기초 (20분)
python chapter1_text_basics.py

# Chapter 2: 임베딩 핵심 (30분) ⭐
python chapter2_embedding_core.py

# Chapter 3: 심화 학습 (40분)
python chapter3_deep_dive.py
```

### 3. 선택적 학습

각 챕터는 독립적으로 실행 가능하지만, Chapter 2는 필수입니다!

---

## 📊 학습 흐름도

```
┌─────────────────────────────────────────────────────────┐
│  Chapter 1: 준비                                         │
│  ───────────────────────────────────────────────────── │
│  • 토큰이란? (GPT의 처리 단위)                          │
│  • 전처리란? (언제 필요/불필요?)                        │
└─────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│  Chapter 2: 핵심 ⭐ (이것이 RAG!)                        │
│  ───────────────────────────────────────────────────── │
│  • 텍스트 → 임베딩 (벡터로 변환)                        │
│  • 코사인 유사도 (벡터 간 거리)                         │
│  • 의미 검색 (유사 문서 찾기)                           │
└─────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│  Chapter 3: 심화                                         │
│  ───────────────────────────────────────────────────── │
│  • 시각화 (벡터 공간 이해)                              │
│  • 모델 비교 (최적 선택)                                │
│  • 한글 처리 (실무 팁)                                  │
└─────────────────────────────────────────────────────────┘
```

---

## 💡 핵심 개념 요약

### 토큰 (Token)
- GPT가 텍스트를 처리하는 단위
- 한글은 영어보다 2~3배 더 많은 토큰 소모
- API 비용 = 토큰 수 × 단가

### 전처리 (Preprocessing)
- **BM25/키워드 검색**: 전처리 필수 ✓
- **임베딩 기반 검색**: 전처리 불필요 (오히려 해로움!) ✗

### 임베딩 (Embedding)
- 텍스트를 고차원 벡터로 변환
- OpenAI text-embedding-3-small: 1536차원
- 의미적 유사성을 벡터 공간의 거리로 표현

### 코사인 유사도 (Cosine Similarity)
- 벡터 간 방향의 유사성 측정 (-1 ~ 1)
- 해석: 0.8+ (매우 유사), 0.6~0.8 (관련), 0.4~0.6 (약간 관련)
- OpenAI 임베딩은 정규화되어 있어 내적만으로 계산 가능

### 의미 검색 (Semantic Search)
1. 문서 인덱싱: 문서들을 임베딩으로 변환 → 저장
2. 쿼리 임베딩: 검색어를 임베딩으로 변환
3. 유사도 계산: 모든 문서와 코사인 유사도 계산
4. 순위 정렬: 가장 유사한 상위 k개 반환

---

## ⚠️ 주의사항

### 1. API 키 필요
Chapter 2, 3의 일부 실습은 `OPENAI_API_KEY`가 필요합니다.

### 2. 전처리 혼동 주의!
- ❌ 임베딩 검색에 전처리 적용 (의미 손실!)
- ✓ 원문 그대로 임베딩 생성

### 3. 한글 RAG 구축 시
- OpenAI 모델이 가장 안전한 선택
- 영어 중심 모델(all-MiniLM 등)은 한글 성능 저하
- 반드시 한글 테스트로 검증!

---

## 🎓 학습 목표 체크리스트

### Chapter 1 완료 후
- [ ] 토큰의 개념과 비용 계산 방법 이해
- [ ] 전처리의 필요성과 한계 파악
- [ ] BM25 vs 임베딩 검색의 전처리 차이 구분

### Chapter 2 완료 후 ⭐
- [ ] 임베딩 생성 방법과 배치 처리 이해
- [ ] 코사인 유사도 계산 방법 숙지
- [ ] 의미 기반 검색 엔진 구현 가능
- [ ] RAG의 Retrieval 부분 이해

### Chapter 3 완료 후
- [ ] t-SNE로 임베딩 시각화 가능
- [ ] 오픈소스 대안 (Sentence Transformers) 이해
- [ ] 모델 선택 기준 (품질/비용/속도) 파악
- [ ] 한글 RAG 구축 시 주의사항 숙지

---

## 📁 파일 구조

```
lab01/
├── README.md                      # 이 파일
├── chapter1_text_basics.py        # 챕터 1: 텍스트 기초
├── chapter2_embedding_core.py     # 챕터 2: 임베딩 핵심 ⭐
├── chapter3_deep_dive.py          # 챕터 3: 심화 학습
└── embedding_tsne_demo.png        # 시각화 결과 (자동 생성)
```

---

## 🔗 다음 단계

### Lab 02: Vector DB (ChromaDB)
- 대용량 문서 인덱싱
- ANN(Approximate Nearest Neighbor) 알고리즘
- 시간 복잡도 O(n) → O(log n)

### Lab 03: RAG 시스템
- Retrieval + Generation
- LangChain을 활용한 RAG 구현
- Hybrid 검색 (BM25 + 임베딩)

### Lab 04: Agent 시스템
- Function Calling
- Multi-Agent 협업
- 실전 응용

---

## 📚 참고 자료

- [OpenAI Embeddings Guide](https://platform.openai.com/docs/guides/embeddings)
- [Sentence Transformers](https://www.sbert.net/)
- [MTEB Leaderboard](https://huggingface.co/spaces/mteb/leaderboard)
- [t-SNE 논문](https://www.jmlr.org/papers/volume9/vandermaaten08a/vandermaaten08a.pdf)

---

**Happy Learning! 🚀**

궁금한 점이 있으면 각 챕터 코드의 주석을 참고하세요.

# Lab 03 - RAG 기초 실습 (rag_basic 시리즈)

RAG (Retrieval-Augmented Generation) 시스템의 기초부터 실무까지 단계별로 학습합니다.

## 📚 학습 구조

`rag_basic.py` 원본 파일이 2684줄로 너무 길어서 학습 플로우에 따라 4개의 챕터로 분리했습니다.

```
Lab 03 - RAG 기초 실습
│
├── chapter1_rag_foundations.py      [기초 파이프라인]
│   ├── 실습 1: 청킹(Chunking) 이해하기
│   ├── 실습 2: 기본 RAG 파이프라인
│   └── 실습 3: RAG 있음 vs 없음 비교
│
├── chapter2_rag_optimization.py     [최적화]
│   ├── 실습 4: 컨텍스트 관리
│   └── 실습 5: 고급 RAG - 컨텍스트 압축
│
├── chapter3_advanced_search.py      [고급 검색 기법]
│   ├── 실습 6: Query Rewriting
│   └── 실습 7: HyDE (가상 문서 임베딩)
│
└── chapter4_production_ready.py     [프로덕션 준비]
    ├── 실습 8: RAG 평가 (Faithfulness, Relevancy)
    ├── 실습 9: Citation 출처 표기
    └── 실습 10: Streaming 응답
```

---

## 🎯 Chapter 1: RAG 기초 파이프라인

**파일:** `chapter1_rag_foundations.py`

### 학습 목표
- RAG가 무엇인지, 왜 필요한지 이해
- 청킹의 중요성과 전략 파악
- RAG 사용 시 답변 품질 향상 확인

### 실습 내용

#### 실습 1: 청킹(Chunking) 이해하기
- 텍스트 분할 전략
- 청크 크기와 오버랩의 영향
- 다양한 청킹 기법 소개

```bash
python chapter1_rag_foundations.py
```

**핵심 개념:**
- 청크 크기 (Chunk Size)
- 오버랩 (Overlap)
- 문맥 유지 전략

#### 실습 2: 기본 RAG 파이프라인
- 문서 → 임베딩 → 검색 → 답변 생성
- Vector DB 활용
- Recall@K 평가

**파이프라인:**
1. 문서 로드 → 텍스트 추출
2. 청킹 → 검색 단위로 분할
3. 임베딩 → 텍스트를 벡터로 변환
4. 인덱싱 → Vector DB에 저장
5. 검색 → 쿼리와 유사한 문서 찾기

#### 실습 3: RAG 있음 vs 없음 비교
- 답변 품질 차이 분석
- 환각(Hallucination) 감소 확인
- RAG의 필요성 이해

---

## 🔧 Chapter 2: RAG 최적화

**파일:** `chapter2_rag_optimization.py`

### 학습 목표
- 토큰 제한 문제와 해결 방법 이해
- 컨텍스트 압축 기법 학습
- RAG 최적화 전략 파악

### 실습 내용

#### 실습 4: 컨텍스트 관리
- 토큰 계산 및 제한 대응
- Truncation (자르기)
- Summarization (요약)
- Multi-doc Compression (압축 결합)

**방법 선택 가이드:**
- 실시간 응답 → Truncation
- 정보 손실 최소화 → Summarization
- 여러 문서 통합 → Compression

#### 실습 5: 고급 RAG - 컨텍스트 압축 적용
- 많은 검색 결과를 효율적으로 활용
- 토큰 절약 + 관련 정보 유지
- 실무 최적화 전략

```bash
python chapter2_rag_optimization.py
```

---

## 🚀 Chapter 3: 고급 검색 기법

**파일:** `chapter3_advanced_search.py`

### 학습 목표
- 쿼리 최적화 전략 학습
- Multi-Query 기법 이해
- HyDE 원리와 효과 파악

### 실습 내용

#### 실습 6: Query Rewriting
- 모호한 쿼리 → 명확한 쿼리
- Query Reformulation (재구성)
- Multi-Query (다중 쿼리)
- Query Expansion (확장)

**효과:**
- 검색 품질 10~30% 향상
- 검색 커버리지 증가

#### 실습 7: HyDE (Hypothetical Document Embedding)
- 가상 문서 생성 기법
- 질문 스타일 → 답변 스타일 변환
- 짧은 쿼리에서 검색 품질 향상

**원리:**
```
사용자 질문 
  → LLM으로 가상 답변 생성 
  → 가상 답변의 임베딩으로 검색 
  → 실제 관련 문서 반환
```

```bash
python chapter3_advanced_search.py
```

---

## 📊 Chapter 4: 프로덕션 준비

**파일:** `chapter4_production_ready.py`

### 학습 목표
- RAG 시스템 품질 측정 방법 이해
- 출처 표기를 통한 신뢰성 확보
- Streaming으로 UX 개선
- 실서비스 배포 준비

### 실습 내용

#### 실습 8: RAG 평가
- **Faithfulness:** 답변이 컨텍스트에 충실한가? (환각 감지)
- **Answer Relevancy:** 답변이 질문에 적절한가?
- **Context Precision:** 검색된 문서가 관련 있는가?
- RAGAS 프레임워크 소개

**평가 기준:**
- Faithfulness > 0.9: 프로덕션 가능
- Answer Relevancy > 0.8: 양호
- Context Precision > 0.7: 검색 품질 OK

#### 실습 9: Citation 출처 표기
- 답변에 출처 명시 ([1], [2] 형식)
- 신뢰성 확보
- 법적/감사 요구사항 대응
- 환각 방지

**패턴:**
- 인라인 Citation: `[1]`, `[2]`
- 문장별 Citation
- 하이퍼링크 Citation (웹 UI)

#### 실습 10: Streaming 응답
- 토큰 단위로 실시간 출력
- 체감 응답 시간 대폭 감소
- UX 개선
- RAG + Streaming 조합

**효과:**
- 첫 토큰까지 0.1초 (vs 전체 완료 3초)
- 사용자가 답변을 미리 읽기 시작 가능

```bash
python chapter4_production_ready.py
```

---

## 🏃‍♂️ 실행 방법

### 환경 설정
```bash
# 프로젝트 루트에서 실행
cd ai-basic-labs

# 환경변수 설정 (.env 파일)
# OPENAI_API_KEY=your_api_key_here
```

### 전체 실습 순서대로 실행
```bash
# Chapter 1: RAG 기초
cd lab03
python chapter1_rag_foundations.py

# Chapter 2: RAG 최적화
python chapter2_rag_optimization.py

# Chapter 3: 고급 검색 기법
python chapter3_advanced_search.py

# Chapter 4: 프로덕션 준비
python chapter4_production_ready.py
```

### 원본 파일 실행 (전체 실습)
```bash
# 모든 실습을 한 번에 실행하고 싶다면 (2684줄)
python rag_basic.py
```

---

## 📦 필요한 패키지

```bash
pip install -r ../requirements.txt
```

**주요 패키지:**
- `openai` - OpenAI API
- `chromadb` - Vector Database
- `tiktoken` - 토큰 계산
- `pdfplumber` - PDF 파일 로드
- `beautifulsoup4` - HTML 파싱
- `python-dotenv` - 환경변수 관리

---

## 🎓 학습 팁

### 1. 순서대로 학습하기
각 챕터는 이전 챕터의 개념을 기반으로 하므로 순서대로 진행하세요.

### 2. 실습 코드 직접 수정하기
- 청크 크기 변경해보기
- 다른 임베딩 모델 시도
- 검색 결과 개수 조정

### 3. 샘플 데이터 준비
```bash
# lab03/ 폴더에 sample.pdf 파일을 넣으면 자동으로 로드됩니다
# 없으면 내장 샘플 텍스트를 사용합니다
```

### 4. 출력 결과 분석
각 실습은 상세한 출력과 설명을 제공하므로 천천히 읽어보세요.

---

## 🔍 주요 개념 요약

### RAG 파이프라인
```
사용자 질문
  ↓
Query Embedding (쿼리 벡터화)
  ↓
Vector Search (유사한 문서 검색)
  ↓
Context + Question (컨텍스트 + 질문)
  ↓
LLM (답변 생성)
  ↓
답변 + 출처
```

### 검색 품질 향상 기법
1. **청킹 최적화** - 적절한 크기와 오버랩
2. **Query Rewriting** - 쿼리 명확화
3. **HyDE** - 가상 문서 임베딩
4. **Re-ranking** - 검색 결과 재정렬 (advanced 시리즈에서)
5. **Hybrid Search** - Sparse + Dense 검색 (advanced 시리즈에서)

### 평가 메트릭
- **Recall@K** - 상위 K개 중 정답 비율
- **Faithfulness** - 컨텍스트 충실도
- **Relevancy** - 답변 적절성
- **Precision** - 검색 정확도

---

## 🔗 다음 단계

이 시리즈를 완료했다면:

1. **advanced_retrieval_langchain.py 시리즈**
   - Hybrid 검색 (Sparse + Dense)
   - Re-ranking (Cross-Encoder)
   - Multi-hop RAG
   - Graph RAG

2. **실전 프로젝트 적용**
   - 사내 문서 검색 시스템
   - 고객 지원 챗봇
   - 도메인 특화 QA 시스템

---

## ❓ FAQ

**Q: 원본 파일은 왜 남겨두나요?**
A: 모든 실습을 한 번에 실행하고 싶을 때 사용할 수 있습니다. 또한 코드 참조용으로 유용합니다.

**Q: 어떤 순서로 학습해야 하나요?**
A: Chapter 1 → 2 → 3 → 4 순서를 권장합니다. 각 챕터는 이전 챕터의 개념을 기반으로 합니다.

**Q: PDF 파일이 없으면 어떻게 하나요?**
A: 자동으로 내장 샘플 텍스트를 사용합니다. PDF가 있으면 더 풍부한 실습이 가능합니다.

**Q: API 키가 필요한가요?**
A: 네, OpenAI API 키가 필요합니다. `.env` 파일에 `OPENAI_API_KEY`를 설정하세요.

---

## 📝 라이선스

이 실습 자료는 교육 목적으로 제공됩니다.

---

**즐거운 RAG 학습 되세요! 🚀**


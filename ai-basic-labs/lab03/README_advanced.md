# Lab03 - Advanced RAG 실습 가이드

이 디렉토리는 고급 RAG (Retrieval-Augmented Generation) 기법을 학습하기 위한 실습 코드입니다.

## 📂 파일 구조

### Advanced 시리즈 (분리된 파일)
```
advanced_chapter1_hybrid_search.py    # 하이브리드 검색
advanced_chapter2_reranking.py        # Re-ranking
advanced_chapter3_advanced_patterns.py # 고급 패턴
```

### 원본 파일 (참고용)
```
advanced_retrieval_langchain.py       # 전체 통합 버전 (원본)
```

---

## 📚 학습 순서

### **Chapter 1: 하이브리드 검색** (`advanced_chapter1_hybrid_search.py`)

**실습 내용:**
1. Sparse vs Dense vs Hybrid 검색 비교
   - Sparse (BM25): 키워드 기반
   - Dense (Vector): 의미 기반
   - Hybrid: 두 방식의 결합

**학습 목표:**
- BM25와 Dense 검색의 차이 이해
- 하이브리드 검색으로 두 장점 결합
- Alpha 값으로 가중치 조정

**실행 방법:**
```bash
python ai-basic-labs/lab03/advanced_chapter1_hybrid_search.py
```

**핵심 개념:**
- **BM25**: 키워드 매칭 기반 검색 (정확한 용어 매칭)
- **Dense Search**: 임베딩 기반 의미 검색 (의미적 유사도)
- **Hybrid**: `score = α × sparse + (1-α) × dense`
- **Trade-off**: 정확성(Sparse) vs 유연성(Dense)

---

### **Chapter 2: Re-ranking** (`advanced_chapter2_reranking.py`)

**실습 내용:**
2. Re-ranking 적용 (Cross-Encoder)
   - 초기 검색 후 Cross-Encoder로 재정렬
   - Precision 향상 효과 측정
   - Before/After 비교

**학습 목표:**
- Re-ranking의 필요성과 효과
- Bi-Encoder vs Cross-Encoder 차이
- 실무 파이프라인 설계

**실행 방법:**
```bash
python ai-basic-labs/lab03/advanced_chapter2_reranking.py
```

**핵심 개념:**
- **Bi-Encoder**: Query와 Doc을 각각 인코딩 (빠름, 덜 정확)
- **Cross-Encoder**: Query-Doc 쌍을 함께 인코딩 (느림, 매우 정확)
- **실무 파이프라인**: Bi-Encoder로 20~50개 추출 → Cross-Encoder로 Top 5 재정렬
- **효과**: Precision@5가 0.4 → 0.8로 2배 향상

---

### **Chapter 3: 고급 패턴** (`advanced_chapter3_advanced_patterns.py`)

**실습 내용:**
3. Multi-hop 검색 (다단계 추론)
   - 복잡한 질문을 하위 질문으로 분해
   - 단계적 검색으로 답 찾기
4. Chunk Size 최적화
   - 다양한 청크 크기 실험
   - 최적 크기 결정
5. Context Window 관리
   - 토큰 제한 고려
   - 결과 조정 및 우선순위 관리

**학습 목표:**
- Multi-hop 검색으로 복잡한 질문 처리
- 최적 Chunk Size 결정 방법
- Context Window 관리 전략

**실행 방법:**
```bash
python ai-basic-labs/lab03/advanced_chapter3_advanced_patterns.py
```

**핵심 개념:**
- **Multi-hop**: 질문 분해 → 첫 검색 → 결과 활용 → 두 번째 검색
- **Chunk Size**:
  - 작은 청크 (256~512): 정밀 검색, 더 많은 청크 필요
  - 중간 청크 (512~1024): 균형 잡힌 선택 (일반 권장)
  - 큰 청크 (1024+): 넓은 컨텍스트, 노이즈 증가 가능
- **Context Window**: 토큰 제한 내에서 최대한 많은 관련 결과 포함

---

## 🎯 전체 학습 흐름

```
1. Hybrid Search (Chapter 1)
   ↓
   Sparse + Dense 결합으로 검색 품질 개선
   
2. Re-ranking (Chapter 2)
   ↓
   Cross-Encoder로 Precision 향상
   
3. Advanced Patterns (Chapter 3)
   ↓
   Multi-hop, Chunk Size, Context Window로 실무 최적화
```

---

## 🔑 핵심 포인트

### 1. 검색 방식 선택
- **키워드 중요**: Sparse 가중치 ↑ (α = 0.7)
- **의미 중요**: Dense 가중치 ↑ (α = 0.3)
- **균형**: Hybrid (α = 0.5)

### 2. Re-ranking 적용
- 초기 검색: 빠르지만 덜 정확 (Bi-Encoder)
- Re-ranking: 느리지만 매우 정확 (Cross-Encoder)
- 전략: 20~50개 → Top 5

### 3. Chunk Size
- 도메인별 실험 필수
- 일반 권장: 512~1024자
- Trade-off: 정밀도 vs 컨텍스트

### 4. Context Window
- 모델별 토큰 제한 고려
- 우선순위: 상위 결과부터 포함
- 부분 포함: 마지막 결과 자르기 가능

---

## 📝 참고사항

1. **원본 파일 보존**: `advanced_retrieval_langchain.py`는 전체 통합 버전으로, 참고용으로 보관되었습니다.

2. **순차 학습 권장**: Chapter 1 → 2 → 3 순서로 학습하면 개념이 자연스럽게 연결됩니다.

3. **실습 환경**: 
   - Python 3.8+
   - `.env` 파일에 OpenAI API 키 설정 필요
   - 필요한 패키지는 `requirements.txt` 참고

4. **에러 발생 시**: 
   - API 키 확인
   - SSL 인증서 문제는 코드 내에서 자동 우회 처리됨
   - 네트워크 연결 확인

---

## 🚀 다음 단계

Advanced 시리즈를 완료했다면, 이제 실무에서:

1. **자신의 데이터로 실험**
   - 도메인별 최적 Chunk Size 찾기
   - 테스트셋으로 Precision/Recall 측정

2. **파이프라인 최적화**
   - Hybrid → Re-ranking → Context Window 관리
   - 비용 vs 성능 Trade-off 고려

3. **프로덕션 배포**
   - 캐싱 전략
   - 모니터링 및 로깅
   - A/B 테스트로 지속 개선

---

## 📖 추가 학습 자료

- `README_rag_basic.md`: RAG 기초 (Chapter 1-4)
- `rag_basic.py`: 기본 RAG 파이프라인 (원본)
- `advanced_retrieval_langchain.py`: 고급 RAG 전체 (원본)

---

**Happy Learning! 🎓**

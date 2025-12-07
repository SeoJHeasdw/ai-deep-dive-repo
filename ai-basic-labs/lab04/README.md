# Lab04: AI Agent 시스템

단일 에이전트부터 멀티 에이전트 오케스트레이션, 그리고 프로덕션 레벨 패턴까지 단계적으로 학습합니다.

## 📚 학습 구조

### **전체 흐름: 기초 → 실용 → 고급 → 프로덕션**

```
Chapter 1 (기초)
    ↓
Chapter 2 (RAG) ← Chapter 1의 Classifier 재사용
    ↓
Chapter 3 (멀티) ← Chapter 1, 2 재사용
    ↓
Chapter 4 (프로덕션) ← 모든 패턴 통합
```

---

## 📂 파일 구조

```
lab04/
├── shared_agent_utils.py          # 공통 유틸리티 (모든 챕터에서 사용)
├── chapter1_agent_basics.py       # Chapter 1: 단일 에이전트 기초
├── chapter2_rag_agents.py         # Chapter 2: RAG 에이전트 통합
├── chapter3_multi_agent_systems.py # Chapter 3: 멀티 에이전트 시스템
├── chapter4_production_patterns.py # Chapter 4: 프로덕션 패턴
└── chroma_db/                     # Vector DB 저장소 (자동 생성)
```

---

## 🎯 챕터별 목표

### **Chapter 1: 단일 에이전트 기초** (~800줄)
**`chapter1_agent_basics.py`**

✅ **학습 목표:**
- LLM으로 구조화된 JSON 출력 받기
- 의도/카테고리 분류 에이전트 구현
- LLM Confidence의 한계 이해
- 앙상블 기법으로 신뢰도 개선

🔑 **핵심 개념:**
- JSON 프롬프트 (Structured Output)
- `IntentClassifierAgent`
- LLM Confidence vs 실제 정확도
- 앙상블 분류 (`use_ensemble=True`)

📝 **실행:**
```bash
cd ai-basic-labs/lab04
python chapter1_agent_basics.py
```

**주요 내용:**
- ✅ JSON 출력 스키마 설계
- ✅ 의도 분류 에이전트 구현
- ✅ LLM Confidence 과신 문제 분석
- ✅ 다중 샘플 앙상블로 일관성 기반 confidence 계산

---

### **Chapter 2: RAG 에이전트 통합** (~1200줄)
**`chapter2_rag_agents.py`**

✅ **학습 목표:**
- 질문 → 분류 → 검색 → 답변 파이프라인 구현
- Top-2 듀얼 검색으로 분류 오류 보완
- unknown 카테고리 안전한 처리
- 실무 안전장치 (환각 방지, confidence 후처리)

🔑 **핵심 개념:**
- `RetrievalAgent` (검색)
- `SummarizationAgent` (요약)
- `FinalAnswerAgent` (답변 생성)
- `SimpleRAGAgent` (통합 파이프라인)
- `UnknownStrategy` (REJECT/GENERIC_LLM/FULL_SEARCH)

📝 **실행:**
```bash
python chapter2_rag_agents.py
```

**주요 내용:**
- ✅ Vector DB 문서 인덱싱
- ✅ 카테고리별 필터링 검색
- ✅ Top-2 듀얼 검색 (`use_dual_search=True`)
- ✅ unknown 처리 전략 (REJECT 권장)
- ✅ 환각 방지 시스템 프롬프트
- ✅ 후처리 confidence 계산

---

### **Chapter 3: 멀티 에이전트 시스템** (~1500줄)
**`chapter3_multi_agent_systems.py`**

✅ **학습 목표:**
- Tool/Function Calling 구현
- 멀티 에이전트 오케스트레이션 (Planner -> Worker)
- 대화 기록 유지 (Memory)
- API 비용 분석 및 최적화 전략

🔑 **핵심 개념:**
- `ToolCallingAgent` (도구 자동 호출)
- `OrchestratorAgent` (Planner)
- `ConversationMemory` (대화 기록)
- 멀티 에이전트 비용 분석

📝 **실행:**
```bash
python chapter3_multi_agent_systems.py
```

**주요 내용:**
- ✅ Tool/Function Calling 실습
- ✅ OpenAI tools 스키마 정의
- ✅ 멀티 에이전트 오케스트레이션
- ✅ Planner -> Worker 구조
- ✅ 대화 기록 관리 (Window Memory)
- ✅ API 비용 분석 (LLM 4회 + Embedding 1회)
- ✅ 전체 파이프라인 통합 실습

---

### **Chapter 4: 프로덕션 패턴** (~1200줄)
**`chapter4_production_patterns.py`**

✅ **학습 목표:**
- ReAct 패턴으로 추론 과정 명시적 구현
- Guardrails로 입출력 안전성 검증
- 에러 핸들링 전략 (재시도, 폴백)
- 트레이싱과 모니터링
- API 비용 최적화 기법

🔑 **핵심 개념:**
- `ReActAgent` (Thought -> Action -> Observation)
- `InputGuardrail` / `OutputGuardrail`
- Retry + Fallback 패턴
- LangSmith / Phoenix (모니터링 도구)
- 모델 티어링, 캐싱, 배치 처리

📝 **실행:**
```bash
python chapter4_production_patterns.py
```

**주요 내용:**
- ✅ ReAct 패턴 구현
- ✅ Guardrails (PII 탐지, Prompt Injection 방어)
- ✅ 에러 핸들링 전략 (재시도, 폴백, 그레이스풀 디그레이드)
- ✅ JSON 파싱 오류 대응
- ✅ 트레이싱과 모니터링 (LangSmith, Phoenix)
- ✅ 비용 최적화 (모델 티어링, 캐싱, 배치)

---

## 🔧 공통 유틸리티

**`shared_agent_utils.py`**

모든 챕터에서 공통으로 사용하는 유틸리티:

```python
# Enum
- IntentCategory, IntentType

# 데이터 클래스
- ClassificationResult
- SearchResult
- AgentResponse

# 해석 함수
- interpret_similarity_score()
- interpret_confidence()
- visualize_similarity_bar()
- visualize_confidence_bar()

# 상수
- CONFIDENCE_THRESHOLDS
```

---

## 🚀 빠른 시작

### 1. 환경 설정

```bash
# 프로젝트 루트에 .env 파일 생성
OPENAI_API_KEY=your_api_key_here
```

### 2. 순차 실행 (권장)

```bash
cd ai-basic-labs/lab04

# Chapter 1: 기초
python chapter1_agent_basics.py

# Chapter 2: RAG
python chapter2_rag_agents.py

# Chapter 3: 멀티 에이전트
python chapter3_multi_agent_systems.py

# Chapter 4: 프로덕션
python chapter4_production_patterns.py
```

### 3. 특정 챕터만 실행

각 챕터는 독립적으로 실행 가능합니다:

```bash
python chapter2_rag_agents.py  # RAG 에이전트만 실습
```

---

## 📊 학습 로드맵

```
┌─────────────────────────────────────────────────────────────┐
│  Week 1: 기초 (Chapter 1-2)                                 │
│  ─────────────────────────────────────────────────────────  │
│  Day 1-2: Chapter 1 - 단일 에이전트, JSON 프롬프트         │
│  Day 3-4: Chapter 2 - RAG 파이프라인, 듀얼 검색            │
│  Day 5  : 복습 및 실습 문제                                 │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│  Week 2: 고급 (Chapter 3-4)                                 │
│  ─────────────────────────────────────────────────────────  │
│  Day 1-2: Chapter 3 - Tool Calling, 멀티 에이전트          │
│  Day 3-4: Chapter 4 - ReAct, Guardrails, 비용 최적화       │
│  Day 5  : 프로젝트 적용 및 최적화                          │
└─────────────────────────────────────────────────────────────┘
```

---

## 💡 주요 학습 포인트

### ⚠️ 비판적 사고 (Critical Thinking)

1. **LLM Confidence의 함정**
   - LLM이 반환하는 확신도 ≠ 실제 정확도
   - 과신(Overconfidence) 문제 심각
   - 해결: 앙상블 분류 + 검색 점수 결합

2. **분류 → 검색 파이프라인 의존성**
   - 분류 오류 → 검색 오류 → 답변 오류
   - 해결: Top-2 듀얼 검색

3. **unknown 처리의 위험성**
   - FULL_SEARCH는 환각/개인정보 오답 위험
   - 권장: REJECT (즉시 거절) 전략

4. **멀티 에이전트 비용**
   - 단순 RAG 대비 2~3배 API 호출
   - 복잡한 질문에만 사용 권장

---

## 🔗 이전 Lab 연계

- **Lab01**: 토큰, 임베딩, 유사도 계산의 기초
- **Lab02**: Vector DB (ChromaDB) 저장 및 검색
- **Lab03**: RAG 파이프라인, 점수 해석, 컨텍스트 관리
- **Lab04**: 에이전트 기반 자동화 (현재 실습)

---

## 📈 실무 적용 예시

### 1. 챗봇 시스템
```python
from chapter1_agent_basics import IntentClassifierAgent
from chapter2_rag_agents import SimpleRAGAgent

# 의도 분류 후 적절한 핸들러로 라우팅
classifier = IntentClassifierAgent()
rag_agent = SimpleRAGAgent()

result = classifier.classify(question)
if result.category == "customer_service":
    answer = rag_agent.answer(question)
```

### 2. 지원 시스템
```python
from chapter3_multi_agent_systems import OrchestratorAgent

# 복잡한 질문은 멀티 에이전트로 처리
orchestrator = OrchestratorAgent()
orchestrator.setup()

answer = orchestrator.process_question(complex_question)
```

### 3. 프로덕션 배포
```python
from chapter4_production_patterns import InputGuardrail, OutputGuardrail

# 안전성 검증
input_guard = InputGuardrail()
output_guard = OutputGuardrail()

if input_guard.validate(user_input)["is_safe"]:
    answer = agent.answer(user_input)
    if output_guard.validate(answer)["is_safe"]:
        return answer
```

---

## 🎓 학습 순서 권장

1. **Chapter 1** ← 시작 (JSON 프롬프트, 분류 기초)
2. **Chapter 2** ← RAG 파이프라인 구축
3. **Lab03 복습** ← 검색 점수 해석 재확인
4. **Chapter 3** ← 멀티 에이전트 오케스트레이션
5. **Chapter 4** ← 프로덕션 레벨 패턴
6. **실제 프로젝트** ← 학습 내용 적용

---

## 🐛 트러블슈팅

### 1. SSL 인증서 오류
```python
# 이미 코드에 포함됨 (httpx.Client(verify=False))
```

### 2. ChromaDB 초기화 오류
```bash
# 기존 DB 삭제
rm -rf ai-basic-labs/lab04/chroma_db
```

### 3. API Key 오류
```bash
# .env 파일 확인
cat ai-basic-labs/.env

# 환경 변수 확인
echo $OPENAI_API_KEY
```

---

## 📚 참고 자료

- [OpenAI Function Calling](https://platform.openai.com/docs/guides/function-calling)
- [LangChain Agents](https://python.langchain.com/docs/modules/agents/)
- [ReAct Paper](https://arxiv.org/abs/2210.03629)
- [LangSmith](https://www.langchain.com/langsmith)
- [Arize Phoenix](https://github.com/Arize-ai/phoenix)

---

## ⚙️ 고급 설정

### 1. 커스텀 모델 사용
```python
# chapter1_agent_basics.py
classifier = IntentClassifierAgent(model="gpt-4o")
```

### 2. 비용 추적
```python
# chapter4_production_patterns.py
tracker = CostTracker()
# 모든 API 호출 후 tracker.track() 호출
```

### 3. 메모리 타입 변경
```python
# chapter3_multi_agent_systems.py
agent = ConversationalAgent(memory_type="buffer")  # 전체 저장
```

---

## 🎯 다음 단계

Lab04를 완료한 후:

1. **실제 프로젝트에 적용**
   - 사내 챗봇, 고객 지원 시스템
   - 문서 검색 시스템
   - 업무 자동화 도구

2. **성능 평가**
   - 100개 이상 라벨링 데이터로 정확도 측정
   - A/B 테스트

3. **프로덕션 배포**
   - Guardrails 적용
   - 비용 모니터링
   - 에러 핸들링 강화

---

## 📞 문의

문제가 발생하면:
1. README를 다시 확인
2. 각 챕터의 주석 참고
3. `shared_agent_utils.py`의 함수 사용법 확인

---

**Happy Learning! 🚀**

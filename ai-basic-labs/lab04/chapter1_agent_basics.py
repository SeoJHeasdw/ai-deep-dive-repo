"""
Chapter 1: 단일 에이전트 기초
- JSON 프롬프트 에이전트로 의도/카테고리 분류
- 구조화된 출력 (Structured Output)
- LLM Confidence vs 실제 정확도
- 앙상블 분류로 신뢰도 향상

학습 목표:
1. LLM으로 구조화된 JSON 출력 받기
2. 의도 분류 에이전트 구현
3. LLM Confidence의 한계 이해
4. 앙상블 기법으로 신뢰도 개선
"""

import os
import sys
import json
from pathlib import Path
from typing import List, Tuple

# OpenAI
from openai import OpenAI

# Pydantic for structured output
from pydantic import BaseModel, Field

# 환경 변수
from dotenv import load_dotenv

# 프로젝트 루트의 .env 파일 로드
project_root = Path(__file__).parent.parent
load_dotenv(dotenv_path=project_root / '.env')

# 공통 유틸리티 import
sys.path.insert(0, str(project_root))
from utils import print_section_header, print_key_points, get_openai_client

# Lab04 공통 유틸리티
from shared_agent_utils import (
    ClassificationResult,
    interpret_confidence,
    visualize_confidence_bar
)

# 공통 데이터 임포트
from shared_data import CATEGORIES


# ============================================================================
# Pydantic 모델 (구조화된 출력용)
# ============================================================================

class IntentClassification(BaseModel):
    """의도 분류 결과 스키마"""
    category: str = Field(description="질문의 카테고리: customer_service, development, planning, unknown 중 하나")
    intent: str = Field(description="질문의 의도: inquiry, troubleshooting, process, policy 중 하나")
    confidence: float = Field(description="분류 확신도: 0.0 ~ 1.0")
    reasoning: str = Field(description="분류 이유 설명")
    keywords: List[str] = Field(description="질문에서 추출한 핵심 키워드 리스트")


# ============================================================================
# 의도 분류 에이전트
# ============================================================================

class IntentClassifierAgent:
    """
    의도 분류 에이전트
    사용자 질문을 분석하여 카테고리와 의도를 JSON 형식으로 추론
    """
    
    def __init__(self, model: str = "gpt-4o-mini"):
        import httpx
        # SSL 인증서 검증 우회 설정 (회사 방화벽 등으로 인한 인증서 문제 해결)
        http_client = httpx.Client(verify=False)
        self.client = OpenAI(http_client=http_client)
        self.model = model
        self.name = "IntentClassifier"
        
        # 분류 프롬프트
        self.system_prompt = """당신은 고객 질문을 분류하는 전문가입니다.
주어진 질문을 분석하여 적절한 카테고리와 의도를 분류해주세요.

## 카테고리 정의
- customer_service: 환불, 배송, 교환, 결제, 회원 등급 등 고객 서비스 관련
- development: API, 코드, 개발, 배포, 에러, 버그 등 개발 관련
- planning: 기획, 일정, 요구사항, 스프린트, KPI 등 기획 관련
- unknown: 위 카테고리에 해당하지 않는 경우

## 의도 유형
- inquiry: 정보나 방법에 대한 문의
- troubleshooting: 문제 해결 요청
- process: 절차나 프로세스 문의
- policy: 정책이나 규정 문의

## 응답 형식
반드시 다음 JSON 형식으로만 응답하세요:
{
    "category": "카테고리명",
    "intent": "의도유형",
    "confidence": 0.0~1.0 사이의 확신도,
    "reasoning": "분류 이유",
    "keywords": ["핵심", "키워드", "리스트"]
}"""
    
    def classify(self, question: str, use_ensemble: bool = False) -> ClassificationResult:
        """
        질문을 분류하고 JSON 형식으로 결과 반환
        
        Args:
            question: 사용자 질문
            use_ensemble: 다중 샘플 앙상블 사용 여부 (비용 증가, 정확도 향상)
        
        Returns:
            ClassificationResult 객체
            
        Note:
            [!] LLM이 반환하는 confidence는 과신(Overconfidence) 경향이 있습니다.
            use_ensemble=True로 설정하면 3회 분류 후 일관성 기반으로 confidence를 재계산합니다.
        """
        if use_ensemble:
            return self._classify_with_ensemble(question)
        
        return self._classify_single(question)
    
    def _classify_single(self, question: str) -> ClassificationResult:
        """단일 분류 (기본)"""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": f"질문: {question}"}
            ],
            temperature=0.1,
            response_format={"type": "json_object"}
        )
        
        # JSON 파싱
        try:
            result = json.loads(response.choices[0].message.content)
            # [!] LLM confidence는 참고용으로만 사용
            llm_confidence = result.get("confidence", 0.0)
            
            return ClassificationResult(
                category=result.get("category", "unknown"),
                intent=result.get("intent", "inquiry"),
                confidence=llm_confidence,  # LLM 원본 (후처리 권장)
                reasoning=result.get("reasoning", ""),
                keywords=result.get("keywords", [])
            )
        except json.JSONDecodeError:
            return ClassificationResult(
                category="unknown",
                intent="inquiry",
                confidence=0.0,
                reasoning="JSON 파싱 실패",
                keywords=[]
            )
    
    def _classify_with_ensemble(self, question: str, n_samples: int = 3) -> ClassificationResult:
        """
        다중 샘플 앙상블 분류 (권장)
        
        [!] 실무 권장 방식: LLM confidence 대신 일관성 기반 confidence 사용
        
        방법:
        1. 동일 질문을 n_samples회 분류 (temperature > 0으로 다양성 확보)
        2. 가장 많이 나온 category를 최종 선택
        3. 일관성 비율을 confidence로 사용
           - 3회 중 3회 동일: confidence = 1.0
           - 3회 중 2회 동일: confidence = 0.67
           - 3회 모두 다름: confidence = 0.33
        """
        results = []
        
        for _ in range(n_samples):
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": f"질문: {question}"}
                ],
                temperature=0.3,  # 약간의 다양성
                response_format={"type": "json_object"}
            )
            
            try:
                result = json.loads(response.choices[0].message.content)
                results.append(result)
            except json.JSONDecodeError:
                results.append({"category": "unknown", "intent": "inquiry"})
        
        # 가장 많이 나온 category 선택
        from collections import Counter
        categories = [r.get("category", "unknown") for r in results]
        category_counts = Counter(categories)
        top_category, top_count = category_counts.most_common(1)[0]
        
        # 일관성 기반 confidence 계산 (LLM confidence 대체)
        consistency_confidence = top_count / n_samples
        
        # 최종 결과 (top_category에 해당하는 첫 번째 결과 사용)
        final_result = next((r for r in results if r.get("category") == top_category), results[0])
        
        return ClassificationResult(
            category=top_category,
            intent=final_result.get("intent", "inquiry"),
            confidence=consistency_confidence,  # 일관성 기반 (LLM confidence 아님!)
            reasoning=f"앙상블 {n_samples}회 중 {top_count}회 일치. " + final_result.get("reasoning", ""),
            keywords=final_result.get("keywords", [])
        )
    
    def classify_batch(self, questions: List[str]) -> List[ClassificationResult]:
        """여러 질문을 일괄 분류"""
        return [self.classify(q) for q in questions]
    
    def get_top2_categories(self, question: str) -> List[Tuple[str, float]]:
        """
        Top-2 카테고리 반환 (듀얼 검색용)
        
        [!] 실무 권장: 분류 → 검색 파이프라인 의존성 완화
        애매한 질문에서 분류 오류 시에도 정답 문서를 검색할 확률 증가
        
        Returns:
            [(category1, score1), (category2, score2)]
        """
        # 3회 분류로 Top-2 추출
        results = []
        for _ in range(3):
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": f"질문: {question}"}
                ],
                temperature=0.4,
                response_format={"type": "json_object"}
            )
            try:
                result = json.loads(response.choices[0].message.content)
                results.append(result.get("category", "unknown"))
            except:
                results.append("unknown")
        
        from collections import Counter
        counts = Counter(results)
        top_2 = counts.most_common(2)
        
        # (category, score) 형태로 반환
        return [(cat, count/3) for cat, count in top_2]


# ============================================================================
# 데모 함수
# ============================================================================

def demo_single_agent():
    """실습 1: 단일 JSON 프롬프트 에이전트"""
    print("\n" + "="*80)
    print("[Chapter 1] 단일 JSON 프롬프트 에이전트")
    print("="*80)
    print("목표: 사용자 질문을 JSON 형식으로 의도/카테고리 추론")
    print("핵심: 구조화된 출력으로 다음 단계 처리 용이")
    
    if not os.getenv("OPENAI_API_KEY"):
        print("\n[!] OPENAI_API_KEY 환경변수를 설정해주세요!")
        return
    
    # JSON 스키마 명시 (먼저 보여주기)
    print_section_header("JSON 출력 스키마", "[SCHEMA]")
    print("""
  ┌─────────────────────────────────────────────────────────┐
  │  LLM이 반환하는 JSON 구조:                               │
  │  ─────────────────────────────────────────────────────  │
  │                                                         │
  │  {                                                      │
  │    "category": "customer_service | development |        │
  │                 planning | unknown",                    │
  │    "intent": "inquiry | troubleshooting |               │
  │               process | policy",                        │
  │    "confidence": 0.0 ~ 1.0,                            │
  │    "reasoning": "분류 이유 설명 문자열",                │
  │    "keywords": ["핵심", "키워드", "배열"]               │
  │  }                                                      │
  │                                                         │
  │  [카테고리 정의]                                         │
  │  * customer_service: 환불, 배송, 회원 등급 등           │
  │  * development: API, 코드, 배포, 에러 등                │
  │  * planning: 기획, 일정, 스프린트 등                    │
  │  * unknown: 위 카테고리에 해당하지 않음                 │
  │                                                         │
  │  [의도 유형]                                             │
  │  * inquiry: 정보 문의                                   │
  │  * troubleshooting: 문제 해결                           │
  │  * process: 절차 문의                                   │
  │  * policy: 정책/규정 문의                               │
  └─────────────────────────────────────────────────────────┘
    """)
    
    # 분류기 초기화
    classifier = IntentClassifierAgent()
    
    # 테스트 질문들
    test_questions = [
        "구매한 지 10일 됐는데 환불 가능한가요?",
        "API 인증은 어떤 방식을 사용하나요?",
        "새 프로젝트 기획서에 뭘 포함해야 하나요?",
        "오늘 날씨 어때요?"  # unknown 테스트
    ]
    
    print_section_header("질문 분류 테스트", "[>>>]")
    
    confidence_values = []  # 확신도 수집 (분석용)
    
    for question in test_questions:
        print(f"\n{'─'*60}")
        print(f"[*] 질문: {question}")
        print(f"{'─'*60}")
        
        result = classifier.classify(question)
        confidence_values.append(result.confidence)
        
        # 카테고리 이름
        category_name = CATEGORIES.get(result.category, {}).get("name", "알 수 없음")
        
        print(f"\n[JSON] 분류 결과:")
        print(f"  {{")
        print(f'    "category": "{result.category}" ({category_name}),')
        print(f'    "intent": "{result.intent}",')
        print(f'    "confidence": {result.confidence:.2f},')
        print(f'    "reasoning": "{result.reasoning}",')
        print(f'    "keywords": {result.keywords}')
        print(f"  }}")
        
        # 확신도 시각화
        bar = visualize_confidence_bar(result.confidence, 20)
        conf_interp = interpret_confidence(result.confidence)
        print(f"\n  확신도: [{bar}] {result.confidence:.0%} {conf_interp}")
    
    # 확신도 분석 및 경고
    print_section_header("확신도(Confidence) 분석", "[!]")
    
    avg_confidence = sum(confidence_values) / len(confidence_values)
    min_confidence = min(confidence_values)
    max_confidence = max(confidence_values)
    
    print(f"""
  ┌─────────────────────────────────────────────────────────┐
  │  [!] LLM 확신도(Confidence)의 한계                       │
  │  ─────────────────────────────────────────────────────  │
  │                                                         │
  │  테스트 결과 통계:                                       │
  │    * 평균 확신도: {avg_confidence:.0%}                                    │
  │    * 최소: {min_confidence:.0%} / 최대: {max_confidence:.0%}                            │
  │                                                         │
  │  [!] 관찰된 문제: 전부 {min_confidence:.0%}~{max_confidence:.0%} 범위의 높은 확신도      │
  │                                                         │
  │  [vs] 이상적 vs 실제 확신도 분포 비교                   │
  │  ─────────────────────────────────────────────────────  │
  │                                                         │
  │  이상적인 경우 (calibrated model):                       │
  │    * 명확한 질문: 90~95% (예: 환불 가능한가요?)         │
  │    * 모호한 질문: 50~70% (예: 이거 어떻게 해요?)        │
  │    * 도메인 외: 20~40% (예: 오늘 날씨 어때요?)          │
  │                                                         │
  │  실제 LLM 출력 (이번 테스트):                            │
  │    * 명확한 질문: {max_confidence:.0%}                                   │
  │    * 도메인 외 질문도: {min_confidence:.0%} (← 이게 문제!)               │
  │                                                         │
  │  왜 이것이 문제인가?                                     │
  │  ─────────────────────────────────────────────────────  │
  │  1. LLM은 '과신(Overconfidence)' 경향이 있음            │
  │     - 잘 모르는 질문에도 높은 확신도를 반환             │
  │     - "오늘 날씨" 같은 도메인 외 질문도 {min_confidence:.0%}!             │
  │                                                         │
  │  2. 확신도 ≠ 실제 정확도 (calibration 문제)             │
  │     - LLM이 90%라고 해도 실제 정확률은 알 수 없음       │
  │     - 별도의 평가 데이터셋으로 검증 필요                │
  │                                                         │
  │  [!!!] 실무 권장 해결책 (매우 중요):                     │
  │  ─────────────────────────────────────────────────────  │
  │  1. LLM confidence를 직접 쓰지 마세요!                  │
  │     대신 후처리 기반 계산:                              │
  │                                                         │
  │     [기본] 2지표 결합:                                  │
  │     final_confidence = (                                │
  │         classification_consistency * 0.4 +              │
  │         top_search_score * 0.6                          │
  │     )                                                   │
  │                                                         │
  │     [권장] 3지표 결합 (고신뢰 서비스용):               │
  │     final_confidence = (                                │
  │         classification_consistency * 0.3 +              │
  │         top_search_score * 0.4 +                        │
  │         answer_self_consistency * 0.3                   │
  │     )                                                   │
  │                                                         │
  │     answer_self_consistency =                           │
  │     → 동일 질문 2~3회 답변 생성 후 의미 일치도 비교    │
  │     → 답변이 일관되면 높은 확신, 불일관하면 불확실    │
  │                                                         │
  │  2. 다중 샘플 앙상블 (비용 증가, 정확도 향상):          │
  │     - 동일 질문 3회 분류                                │
  │     - 서로 다른 category 나오면 confidence 자동 하락    │
  │     - 3회 중 3회 동일: 1.0                              │
  │     - 3회 중 2회 동일: 0.67                             │
  │     - 3회 모두 다름: 0.33                               │
  │                                                         │
  │  [CODE] 앙상블 사용법:                                   │
  │  ─────────────────────────────────────────────────────  │
  │  result = classifier.classify(question, use_ensemble=True)│
  │  # confidence = 일관성 기반 (LLM confidence 아님!)      │
  └─────────────────────────────────────────────────────────┘
    """)
    
    # 앙상블 데모 (선택적)
    print_section_header("앙상블 분류 데모 (실무 권장)", "[DEMO]")
    print("목표: LLM confidence 대신 일관성 기반 confidence 계산")
    
    ensemble_question = "오늘 날씨 어때요?"  # 애매한 질문
    print(f"\n테스트 질문: {ensemble_question}")
    print("(3회 분류 후 일관성 기반 confidence 계산)")
    
    ensemble_result = classifier.classify(ensemble_question, use_ensemble=True)
    
    print(f"\n[앙상블 결과]")
    print(f"  카테고리: {ensemble_result.category}")
    print(f"  일관성 기반 confidence: {ensemble_result.confidence:.0%}")
    print(f"  설명: {ensemble_result.reasoning}")
    
    if ensemble_result.confidence < 1.0:
        print(f"\n  [!] 일관성 {ensemble_result.confidence:.0%} = 분류가 불안정함")
        print(f"      → 이 질문은 unknown 처리 또는 사람 검토 권장")
    
    # 핵심 포인트
    print_key_points([
        "- JSON 출력: 구조화된 형식으로 후처리 용이",
        "- response_format: {'type': 'json_object'}로 JSON 강제",
        "- [!!!] LLM confidence는 참고용으로만! 과신 문제 심각",
        "- [권장] 앙상블 분류: classify(q, use_ensemble=True)",
        "- [권장] 후처리 confidence: 검색점수 + 일관성 결합",
        "- 키워드: 후속 검색 쿼리 확장에 활용 가능"
    ], "JSON 프롬프트 에이전트 핵심")


# ============================================================================
# 메인 실행
# ============================================================================

def main():
    """Chapter 1 메인 실행"""
    print("\n" + "="*80)
    print("[LAB04 - Chapter 1] 단일 에이전트 기초")
    print("="*80)
    
    demo_single_agent()
    
    print("\n" + "="*80)
    print("[OK] Chapter 1 완료!")
    print("="*80)
    print("\n다음 단계: Chapter 2 (RAG 에이전트)로 이동하세요.")


if __name__ == "__main__":
    main()


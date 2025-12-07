"""
Lab04 공통 유틸리티 및 데이터 클래스
모든 챕터에서 공통으로 사용하는 함수와 상수 정의
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import List, Dict, Any


# ============================================================================
# 상수 및 해석 기준
# ============================================================================

# 확신도 해석 기준
# [!] 주의: LLM의 확신도는 실제 정확도와 다를 수 있음 (과신 문제)
CONFIDENCE_THRESHOLDS = {
    'high': 0.85,     # 85% 이상 = 높은 확신
    'medium': 0.65,   # 65~85% = 중간 확신
    # 65% 미만 = 낮은 확신 (재검토 필요)
}


# ============================================================================
# Enum 정의
# ============================================================================

class IntentCategory(str, Enum):
    """질문 카테고리"""
    CUSTOMER_SERVICE = "customer_service"
    DEVELOPMENT = "development"
    PLANNING = "planning"
    UNKNOWN = "unknown"


class IntentType(str, Enum):
    """질문 의도 유형"""
    INQUIRY = "inquiry"           # 정보 문의
    TROUBLESHOOTING = "troubleshooting"  # 문제 해결
    PROCESS = "process"           # 절차 문의
    POLICY = "policy"             # 정책 문의


# ============================================================================
# 데이터 클래스
# ============================================================================

@dataclass
class ClassificationResult:
    """분류 결과 데이터 클래스"""
    category: str
    intent: str
    confidence: float
    reasoning: str
    keywords: List[str] = field(default_factory=list)


@dataclass
class SearchResult:
    """검색 결과 데이터 클래스"""
    content: str
    score: float
    metadata: Dict[str, Any]
    rank: int


@dataclass
class AgentResponse:
    """에이전트 응답 데이터 클래스"""
    agent_name: str
    output: Any
    elapsed_time: float
    metadata: Dict[str, Any] = field(default_factory=dict)


# ============================================================================
# 해석 및 시각화 함수
# ============================================================================

def interpret_similarity_score(score: float) -> str:
    """
    유사도 점수 해석 (1/(1+distance) 변환 후)
    
    Args:
        score: 0~1 범위의 유사도 점수
    
    Returns:
        해석 문자열
    """
    if score >= 0.50:
        return "[v] 높음"
    elif score >= 0.35:
        return "[~] 중간"
    else:
        return "[x] 낮음"


def interpret_confidence(confidence: float) -> str:
    """
    확신도 해석
    
    [!] 주의: LLM이 반환하는 확신도는 실제 정확도와 다를 수 있습니다.
    - LLM은 대부분 80~95% 범위의 높은 확신도를 반환하는 경향이 있음
    - 이것은 '과신(Overconfidence)' 문제로 알려져 있음
    - 실무에서는 확신도를 참고용으로만 사용하고, 임계값 기반 필터링 권장
    """
    if confidence >= CONFIDENCE_THRESHOLDS['high']:
        return "[v] 높은 확신"
    elif confidence >= CONFIDENCE_THRESHOLDS['medium']:
        return "[~] 중간 확신"
    else:
        return "[!] 낮은 확신 (재검토 필요)"


def visualize_similarity_bar(score: float, width: int = 30) -> str:
    """유사도를 시각적 막대로 표시"""
    filled = int(score * width)
    empty = width - filled
    return "█" * filled + "░" * empty


def visualize_confidence_bar(confidence: float, width: int = 20) -> str:
    """확신도를 시각적 막대로 표시"""
    filled = int(confidence * width)
    empty = width - filled
    return "=" * filled + "-" * empty


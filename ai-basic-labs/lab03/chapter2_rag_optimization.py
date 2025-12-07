"""
RAG (Retrieval-Augmented Generation) 최적화
Chapter 2: RAG 최적화 - 컨텍스트 관리, 고급 RAG

실습 항목:
4. 컨텍스트 관리 - 토큰 제한 대응
5. 고급 RAG - 컨텍스트 압축 적용

학습 목표:
- 토큰 제한 문제와 해결 방법 이해
- 컨텍스트 압축 기법 학습
- RAG 최적화 전략 파악
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import chromadb
from openai import OpenAI
from dotenv import load_dotenv
import tiktoken

# 프로젝트 루트의 .env 파일 로드
project_root = Path(__file__).parent.parent
load_dotenv(dotenv_path=project_root / '.env')

# 공통 유틸리티 import를 위한 경로 추가
sys.path.insert(0, str(project_root))
from utils import (
    print_section_header,
    print_subsection,
    print_key_points,
    get_openai_client,
    interpret_l2_distance,
    l2_distance_to_similarity
)

# 공통 데이터 임포트
from shared_data import SAMPLE_TEXT, MIN_TEXT_LENGTH, get_sample_or_document_text

# Chapter 1 클래스 임포트
from lab03.chapter1_rag_foundations import (
    SearchResult,
    DocumentLoader,
    TextChunker,
    RAGSystem,
    format_chunk,
    format_chunk_id,
    print_search_result,
    find_sample_document
)


# ============================================================================
# 컨텍스트 관리
# ============================================================================

class ContextManager:
    """컨텍스트 토큰 관리 및 압축"""
    
    def __init__(self, model: str = "gpt-4o-mini"):
        self.model = model
        self.encoding = tiktoken.encoding_for_model(model)
        # 공통 헬퍼 사용 (SSL 인증서 검증 우회 포함)
        self.client = get_openai_client()
    
    def count_tokens(self, text: str) -> int:
        """텍스트의 토큰 수 계산"""
        return len(self.encoding.encode(text))
    
    def truncate_context(self, context: str, max_tokens: int) -> str:
        """컨텍스트를 최대 토큰 수로 자르기"""
        tokens = self.encoding.encode(context)
        
        if len(tokens) <= max_tokens:
            return context
        
        truncated_tokens = tokens[:max_tokens]
        return self.encoding.decode(truncated_tokens)
    
    def summarize_context(self, context: str, max_length: int = 200) -> str:
        """컨텍스트를 요약"""
        prompt = f"""다음 텍스트를 {max_length}자 이내로 핵심 내용만 요약해주세요:

{context}

요약:"""
        
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=max_length
        )
        
        return response.choices[0].message.content
    
    def compress_contexts(self, contexts: List[str], max_tokens: int) -> str:
        """여러 컨텍스트를 요약하고 결합"""
        summaries = []
        
        for i, context in enumerate(contexts):
            tokens = self.count_tokens(context)
            
            if tokens > max_tokens // len(contexts):
                # 너무 길면 요약
                summary = self.summarize_context(context, max_length=150)
                summaries.append(f"[문서 {i+1}] {summary}")
            else:
                summaries.append(f"[문서 {i+1}] {context}")
        
        return "\n\n".join(summaries)


# ============================================================================
# 데모 함수들
# ============================================================================

def demo_context_management():
    """실습 4: 컨텍스트 관리"""
    print("\n" + "="*80)
    print("[4] 실습 4: 컨텍스트 관리")
    print("="*80)
    print("목표: 토큰 제한 내에서 효과적으로 컨텍스트 관리")
    print("핵심: 토큰 계산, 자르기, 요약 기법")
    
    if not os.getenv("OPENAI_API_KEY"):
        print("\n[!] OPENAI_API_KEY 환경변수를 설정해주세요!")
        return
    
    context_manager = ContextManager()
    
    # 토큰 계산 방법 설명
    print_section_header("토큰(Token)이란?", "[INFO]")
    print("  LLM은 텍스트를 '토큰' 단위로 처리합니다.")
    print()
    print("  [CODE] 토큰 계산 코드:")
    print("  ┌─────────────────────────────────────────────────────")
    print("  │ import tiktoken")
    print("  │ encoder = tiktoken.encoding_for_model('gpt-4o-mini')")
    print("  │ tokens = encoder.encode(text)")
    print("  │ print(f'토큰 수: {len(tokens)}')")
    print("  └─────────────────────────────────────────────────────")
    print()
    print("  [TIP] 토큰 변환 기준 (GPT-4o-mini 기준):")
    print("  * 영어: 1 단어 ≈ 1.3 토큰 (단어 단위가 효율적)")
    print("  * 한글: 1 글자 ≈ 1.5~2.5 토큰 (글자당 토큰이 많음)")
    print()
    print("  [예시] 실제 측정:")
    print("  * '안녕하세요' (5글자) → 약 8~10 토큰")
    print("  * 'Hello' (5글자) → 약 1 토큰")
    print("  * → 한글이 영어보다 약 8~10배 비효율적!")
    print()
    print("  [참고] Lab 1과의 연계:")
    print("  * Lab 1에서는 '토큰당 글자 수'를 측정 (역수 관계)")
    print("  * 예: 한글 1.18 글자/토큰 = 약 0.85 토큰/글자")
    
    # 긴 텍스트 예제
    long_text = SAMPLE_TEXT[:3000]  # 처음 3000자
    
    print(f"\n원본 텍스트 길이: {len(long_text)} 문자")
    original_tokens = context_manager.count_tokens(long_text)
    print(f"원본 토큰 수: {original_tokens}")
    
    # 방법 1: 토큰 수로 자르기
    print_section_header("방법 1: 토큰 수로 자르기 (Truncation)")
    
    truncated = context_manager.truncate_context(long_text, max_tokens=200)
    truncated_tokens = context_manager.count_tokens(truncated)
    print(f"자른 후 토큰 수: {truncated_tokens} ({(1-truncated_tokens/original_tokens)*100:.1f}% 감소)")
    print(f"자른 텍스트 미리보기:")
    print(f"  {truncated[:300]}...")
    
    # 방법 2: 요약하기 (단일 문서)
    print_section_header("방법 2: 단일 문서 요약 (Abstractive Summarization)")
    print("[설명] LLM을 사용하여 한 문서의 핵심 내용만 추출")
    print()
    
    summarized = context_manager.summarize_context(long_text[:1500], max_length=100)
    summarized_tokens = context_manager.count_tokens(summarized)
    print(f"요약 토큰 수: {summarized_tokens}")
    print(f"요약 내용: {summarized}")
    
    # 방법 3: 여러 컨텍스트 압축
    print_section_header("방법 3: 복수 문서 요약 후 결합 (Multi-doc Summarization)")
    print("[설명] 여러 문서를 각각 요약한 후 하나로 결합")
    print("[차이점] 방법 2는 단일 문서, 방법 3은 여러 문서를 병렬 처리")
    print()
    
    contexts = [
        "인공지능(AI)은 인간의 학습, 추론 능력을 컴퓨터로 구현하는 기술입니다.",
        "머신러닝은 데이터에서 패턴을 학습하여 예측하는 AI의 하위 분야입니다.",
        "딥러닝은 신경망을 여러 층으로 쌓아 복잡한 패턴을 학습합니다."
    ]
    
    compressed = context_manager.compress_contexts(contexts, max_tokens=200)
    print(f"압축 후 토큰 수: {context_manager.count_tokens(compressed)}")
    print(f"압축된 내용:\n{compressed}")
    
    # 방법 선택 가이드
    print("\n" + "="*60)
    print("[*] 방법 선택 가이드:")
    print("="*60)
    print()
    print("  [CASE 1] 언제 자르기(Truncation)를 사용할까?")
    print("  ─────────────────────────────────────")
    print("  * 실시간 응답이 중요할 때 (지연 최소화)")
    print("  * 비용 최소화가 우선일 때")
    print("  * 예: 챗봇, 실시간 검색")
    print("  * [!] 단점: 뒤쪽 정보 손실")
    print()
    print("  [CASE 2] 언제 요약(Summarization)을 사용할까?")
    print("  ─────────────────────────────────────")
    print("  * 정보 손실을 최소화해야 할 때")
    print("  * 문맥 이해가 중요할 때")
    print("  * 예: 내부 보고서, 일반 QA")
    print("  * [!] 단점: 추가 API 호출 비용")
    print()
    print("  ⚠️ [주의] 요약은 항상 LLM의 '재해석'입니다!")
    print("     사실 왜곡 가능성을 0으로 만들 수는 없습니다.")
    print("     법률/의료/금융 등 정확성 필수 도메인은 원문 인용 권장.")
    print()
    print("  [CASE 3] 언제 압축 결합(Compression)을 사용할까?")
    print("  ─────────────────────────────────────")
    print("  * 여러 문서를 통합해야 할 때")
    print("  * 토큰과 품질의 균형이 필요할 때")
    print("  * 예: 연구 논문 분석, 종합 보고서")
    
    # 요약
    print("\n" + "="*60)
    print("[TIP] 컨텍스트 관리 핵심:")
    print("="*60)
    print("  - LLM마다 토큰 제한 있음 (GPT-4: 8K~128K, GPT-4o: 128K)")
    print("  - 프롬프트 + 컨텍스트 + 응답 <= 최대 토큰")
    print("  - 자르기: 빠르지만 정보 손실")
    print("  - 요약: 정보 보존, API 비용 추가")
    print("  - 실무: 관련성 높은 청크만 선별 후 포함")
    
    print("""
  ────────────────────────────────────────────────────────────
  [!] 법적/감사 실무: 원문 인용 vs 요약
  ────────────────────────────────────────────────────────────
  
  ┌─────────────────────────────────────────────────────────┐
  │  원문 인용 (Verbatim Citation)                          │
  │  ─────────────────────────────────────────────────────  │
  │  * 법적 증빙 가능 (원본 그대로)                         │
  │  * 감사 추적 가능 (어디서 왔는지 명확)                  │
  │  * 금융, 공공, 의료 RAG에서 필수                        │
  │  * 단점: 토큰 많이 소모                                 │
  └─────────────────────────────────────────────────────────┘
  
  ┌─────────────────────────────────────────────────────────┐
  │  요약 (Summarization)                                   │
  │  ─────────────────────────────────────────────────────  │
  │  * 증빙 불가 (LLM이 재해석한 결과물)                    │
  │  * 정보가 변형/누락될 수 있음                           │
  │  * 일반 QA, 내부 업무용에 적합                          │
  │  * 장점: 토큰 절약                                      │
  └─────────────────────────────────────────────────────────┘
  
  [TIP] 도메인별 선택:
  * 금융/공공/의료 → 원문 인용 필수 (규제 준수)
  * 고객 서비스/내부 QA → 요약 OK (효율 우선)
  * 하이브리드: 요약 답변 + 원문 출처 링크 제공
  ────────────────────────────────────────────────────────────
    """)


def demo_advanced_rag():
    """실습 5: 고급 RAG - 컨텍스트 압축 적용"""
    print("\n" + "="*80)
    print("[5] 실습 5: 고급 RAG - 컨텍스트 압축 적용")
    print("="*80)
    print("목표: 많은 검색 결과를 압축하여 효율적으로 활용")
    print("핵심: 토큰 절약 + 관련 정보 유지")
    
    if not os.getenv("OPENAI_API_KEY"):
        print("\n[!] OPENAI_API_KEY 환경변수를 설정해주세요!")
        return
    
    rag = RAGSystem("demo_rag_advanced")
    context_manager = ContextManager()
    
    # 문서 추가
    rag.add_text(SAMPLE_TEXT, source_name="AI_가이드", check_duplicate=True)
    
    question = "인공지능의 역사와 주요 분류를 설명해주세요."
    print(f"\n[*] 질문: {question}")
    
    # 많은 문서 검색
    search_results = rag.search(question, n_results=5)
    
    print(f"\n검색된 문서 수: {len(search_results)}개")
    
    # 원본 컨텍스트
    original_context = "\n\n".join([r.content for r in search_results])
    original_tokens = context_manager.count_tokens(original_context)
    print(f"원본 컨텍스트 토큰 수: {original_tokens}")
    
    # 컨텍스트 압축
    print_section_header("컨텍스트 압축 적용")
    
    compressed_context = context_manager.compress_contexts(
        [r.content for r in search_results], 
        max_tokens=500
    )
    compressed_tokens = context_manager.count_tokens(compressed_context)
    print(f"압축 후 토큰 수: {compressed_tokens}")
    print(f"토큰 절약: {original_tokens - compressed_tokens} ({(1 - compressed_tokens/original_tokens)*100:.1f}%)")
    
    # 압축된 컨텍스트로 답변 생성
    print_section_header("압축된 컨텍스트로 답변 생성")
    
    prompt = f"""다음 문서들을 참고하여 질문에 답변해주세요.

참고 문서:
{compressed_context}

질문: {question}

답변:"""
    
    response = rag.client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "당신은 주어진 문서를 기반으로 정확하게 답변하는 AI 어시스턴트입니다."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3
    )
    
    print(response.choices[0].message.content)
    
    # 요약
    print("\n" + "="*60)
    print("[TIP] 고급 RAG 핵심:")
    print("="*60)
    print("  - 많은 검색 결과 -> 압축으로 토큰 절약")
    print("  - 압축 방법: 요약, 핵심 문장 추출, 관련도 필터링")
    print("  - 실무: Re-ranking + 컨텍스트 압축 조합 사용")
    print("  - 주의: 과도한 압축은 정보 손실 유발")


# ============================================================================
# 메인 실행
# ============================================================================

def main():
    """Chapter 2 실습 실행"""
    print("\n" + "="*80)
    print("[LAB 03 - Chapter 2] RAG 최적화")
    print("="*80)
    
    print("\n[LIST] 실습 항목:")
    print("  4. 컨텍스트 관리 - 토큰 제한 대응")
    print("  5. 고급 RAG - 컨텍스트 압축 적용")
    
    try:
        # 4. 컨텍스트 관리
        demo_context_management()
        
        # 5. 고급 RAG
        demo_advanced_rag()
        
        print("\n" + "="*80)
        print("[OK] Chapter 2 완료!")
        print("="*80)
        print("\n[NEXT] 다음 단계:")
        print("   - chapter3_advanced_search.py : 고급 검색 기법 (Query Rewriting, HyDE)")
        
    except Exception as e:
        print(f"\n[X] 오류 발생: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()


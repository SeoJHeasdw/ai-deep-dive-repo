"""
Chapter 3: 멀티 에이전트 시스템
- Tool/Function Calling
- Orchestrator (Planner -> Worker)
- Conversation Memory (대화 기록 관리)
- 전체 파이프라인 통합 실습

학습 목표:
1. LLM이 도구를 자동으로 호출하는 Tool Calling
2. 멀티 에이전트 오케스트레이션 (Planner -> Worker)
3. 대화 기록 유지 (Memory)
4. API 비용 분석 및 최적화 전략
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import List, Dict, Any

# OpenAI
from openai import OpenAI

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
    SearchResult,
    interpret_confidence,
    interpret_similarity_score,
    visualize_confidence_bar,
    visualize_similarity_bar
)

# Chapter 1, 2에서 import
from chapter1_agent_basics import IntentClassifierAgent
from chapter2_rag_agents import (
    RetrievalAgent,
    SummarizationAgent,
    FinalAnswerAgent
)

# 공통 데이터
from shared_data import CATEGORIES, SAMPLE_QUESTIONS


# ============================================================================
# Tool/Function Calling Agent
# ============================================================================

class ToolCallingAgent:
    """
    OpenAI Function Calling을 활용한 Tool Agent
    
    Tool Calling이란?
    - LLM이 직접 외부 함수를 호출할 수 있게 하는 기능
    - 계산, 검색, API 호출 등을 LLM이 자동으로 수행
    """
    
    def __init__(self, model: str = "gpt-4o-mini"):
        self.client = get_openai_client()
        self.model = model
        
        # 사용 가능한 도구 정의
        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "calculate",
                    "description": "수학 계산을 수행합니다. 사칙연산, 제곱, 제곱근 등을 계산할 수 있습니다.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "expression": {
                                "type": "string",
                                "description": "계산할 수학 표현식 (예: '2 + 3 * 4', 'sqrt(16)', '2 ** 10')"
                            }
                        },
                        "required": ["expression"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_current_time",
                    "description": "현재 시간을 반환합니다.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "timezone": {
                                "type": "string",
                                "description": "시간대 (예: 'Asia/Seoul', 'UTC'). 기본값은 로컬 시간."
                            }
                        },
                        "required": []
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "search_knowledge",
                    "description": "내부 지식 베이스에서 정보를 검색합니다.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "검색할 키워드 또는 질문"
                            },
                            "category": {
                                "type": "string",
                                "enum": ["customer_service", "development", "planning", "all"],
                                "description": "검색할 카테고리"
                            }
                        },
                        "required": ["query"]
                    }
                }
            }
        ]
        
        # 도구 실행 함수 매핑
        self.tool_functions = {
            "calculate": self._tool_calculate,
            "get_current_time": self._tool_get_time,
            "search_knowledge": self._tool_search
        }
    
    def _tool_calculate(self, expression: str) -> str:
        """계산 도구 구현"""
        import math
        try:
            allowed_names = {
                'sqrt': math.sqrt,
                'sin': math.sin,
                'cos': math.cos,
                'pi': math.pi,
                'e': math.e,
                'abs': abs,
                'round': round,
                'pow': pow
            }
            result = eval(expression, {"__builtins__": {}}, allowed_names)
            return f"계산 결과: {expression} = {result}"
        except Exception as e:
            return f"계산 오류: {str(e)}"
    
    def _tool_get_time(self, timezone: str = None) -> str:
        """현재 시간 도구 구현"""
        from datetime import datetime, timedelta
        try:
            if timezone == "Asia/Seoul":
                now = datetime.utcnow() + timedelta(hours=9)
                return f"현재 시간 ({timezone}): {now.strftime('%Y-%m-%d %H:%M:%S')}"
            elif timezone == "UTC":
                now = datetime.utcnow()
                return f"현재 시간 (UTC): {now.strftime('%Y-%m-%d %H:%M:%S')}"
            
            now = datetime.now()
            return f"현재 시간 (로컬): {now.strftime('%Y-%m-%d %H:%M:%S')}"
        except Exception as e:
            return f"시간 조회 오류: {str(e)}"
    
    def _tool_search(self, query: str, category: str = "all") -> str:
        """지식 검색 도구 구현 (간단한 키워드 매칭)"""
        from shared_data import CUSTOMER_SERVICE_DOCS, DEVELOPMENT_DOCS, PLANNING_DOCS
        
        category_docs = {
            "customer_service": {"title": "고객센터 가이드", "content": CUSTOMER_SERVICE_DOCS},
            "development": {"title": "개발팀 가이드", "content": DEVELOPMENT_DOCS},
            "planning": {"title": "기획팀 가이드", "content": PLANNING_DOCS}
        }
        
        results = []
        search_categories = category_docs.keys() if category == "all" else [category]
        
        for cat in search_categories:
            if cat not in category_docs:
                continue
            doc = category_docs[cat]
            content = doc["content"]
            
            if query.lower() in content.lower():
                idx = content.lower().find(query.lower())
                start = max(0, idx - 50)
                end = min(len(content), idx + len(query) + 100)
                snippet = content[start:end].replace('\n', ' ').strip()
                if start > 0:
                    snippet = "..." + snippet
                if end < len(content):
                    snippet = snippet + "..."
                
                results.append({
                    "title": doc["title"],
                    "category": cat,
                    "snippet": snippet
                })
        
        if not results:
            return f"'{query}'에 대한 검색 결과가 없습니다."
        
        output = f"'{query}' 검색 결과 ({len(results)}건):\n"
        for i, r in enumerate(results[:3], 1):
            output += f"\n[{i}] {r['title']} ({r['category']})\n    {r['snippet']}"
        
        return output
    
    def run(self, user_message: str, max_iterations: int = 5) -> Dict[str, Any]:
        """
        Tool Calling 에이전트 실행
        
        Args:
            user_message: 사용자 메시지
            max_iterations: 최대 도구 호출 횟수
        
        Returns:
            실행 결과
        """
        messages = [
            {
                "role": "system",
                "content": """당신은 도구를 활용할 수 있는 AI 어시스턴트입니다.
사용자의 요청을 처리하기 위해 필요한 도구를 호출하세요."""
            },
            {"role": "user", "content": user_message}
        ]
        
        tool_calls_history = []
        iteration = 0
        
        while iteration < max_iterations:
            iteration += 1
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=self.tools,
                tool_choice="auto"
            )
            
            message = response.choices[0].message
            
            if not message.tool_calls:
                return {
                    "answer": message.content,
                    "tool_calls": tool_calls_history,
                    "iterations": iteration
                }
            
            messages.append(message)
            
            for tool_call in message.tool_calls:
                function_name = tool_call.function.name
                function_args = json.loads(tool_call.function.arguments)
                
                if function_name in self.tool_functions:
                    tool_result = self.tool_functions[function_name](**function_args)
                else:
                    tool_result = f"알 수 없는 도구: {function_name}"
                
                tool_calls_history.append({
                    "tool": function_name,
                    "args": function_args,
                    "result": tool_result
                })
                
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": tool_result
                })
        
        return {
            "answer": "최대 반복 횟수에 도달했습니다.",
            "tool_calls": tool_calls_history,
            "iterations": iteration
        }


# ============================================================================
# Conversation Memory
# ============================================================================

class ConversationMemory:
    """
    대화 기록 관리 클래스
    
    Memory 유형:
    - Buffer: 모든 대화 저장 (토큰 제한 주의)
    - Window: 최근 N개 대화만 유지
    """
    
    def __init__(self, max_messages: int = 20, memory_type: str = "window"):
        """
        Args:
            max_messages: 최대 저장 메시지 수 (window 타입용)
            memory_type: "buffer", "window" 중 하나
        """
        self.messages: List[Dict[str, str]] = []
        self.max_messages = max_messages
        self.memory_type = memory_type
    
    def add_message(self, role: str, content: str):
        """메시지 추가"""
        self.messages.append({"role": role, "content": content})
        
        # Window 타입: 최대 개수 초과 시 오래된 것 제거
        if self.memory_type == "window" and len(self.messages) > self.max_messages:
            non_system = [m for m in self.messages if m["role"] != "system"]
            system = [m for m in self.messages if m["role"] == "system"]
            non_system = non_system[-(self.max_messages - len(system)):]
            self.messages = system + non_system
    
    def get_messages(self) -> List[Dict[str, str]]:
        """현재 저장된 메시지 반환"""
        return self.messages.copy()
    
    def clear(self):
        """대화 기록 초기화"""
        self.messages = []


class ConversationalAgent:
    """대화 기록을 유지하는 에이전트"""
    
    def __init__(self, model: str = "gpt-4o-mini", memory_type: str = "window"):
        self.client = get_openai_client()
        self.model = model
        self.memory = ConversationMemory(max_messages=10, memory_type=memory_type)
        
        self.system_prompt = """당신은 친절하고 유용한 AI 어시스턴트입니다.
이전 대화 맥락을 고려하여 자연스럽게 대화를 이어가세요.
사용자가 이전에 언급한 내용을 기억하고 참조하세요."""
        
        self.memory.add_message("system", self.system_prompt)
    
    def chat(self, user_message: str) -> str:
        """사용자 메시지에 응답 (대화 기록 유지)"""
        self.memory.add_message("user", user_message)
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=self.memory.get_messages(),
            temperature=0.7,
            max_tokens=500
        )
        
        assistant_message = response.choices[0].message.content
        self.memory.add_message("assistant", assistant_message)
        
        return assistant_message
    
    def get_history(self) -> List[Dict[str, str]]:
        """대화 기록 반환"""
        return self.memory.get_messages()
    
    def reset(self):
        """대화 초기화"""
        self.memory.clear()
        self.memory.add_message("system", self.system_prompt)


# ============================================================================
# Orchestrator Agent
# ============================================================================

class OrchestratorAgent:
    """
    오케스트레이터 에이전트 (Planner)
    멀티 에이전트 실행을 조율
    """
    
    def __init__(self, model: str = "gpt-4o-mini"):
        import httpx
        http_client = httpx.Client(verify=False)
        self.client = OpenAI(http_client=http_client)
        self.model = model
        self.name = "Orchestrator"
        
        # 에이전트 초기화
        self.classifier = IntentClassifierAgent(model)
        self.retriever = RetrievalAgent()
        self.summarizer = SummarizationAgent(model)
        self.final_answer = FinalAnswerAgent(model)
        
        # 실행 로그
        self.execution_log = []
        
        self.plan_prompt = """당신은 멀티 에이전트 시스템의 플래너입니다.
사용자의 질문을 분석하여 어떤 에이전트들을 어떤 순서로 실행할지 계획을 세워주세요.

## 사용 가능한 에이전트
- IntentClassifier: 질문의 카테고리와 의도를 분류
- Retrieval: 관련 문서를 검색
- Summarization: 검색 결과를 요약
- FinalAnswer: 최종 답변 생성

## 응답 형식
반드시 다음 JSON 형식으로 응답하세요:
{
    "question": "원본 질문",
    "steps": [
        {"step_number": 1, "agent": "IntentClassifier", "action": "질문 분류", "input_from": null},
        {"step_number": 2, "agent": "Retrieval", "action": "문서 검색", "input_from": 1},
        {"step_number": 3, "agent": "Summarization", "action": "결과 요약", "input_from": 2},
        {"step_number": 4, "agent": "FinalAnswer", "action": "답변 생성", "input_from": 3}
    ],
    "final_agent": "FinalAnswer"
}"""
    
    def setup(self, ingest_documents: bool = True):
        """시스템 초기화"""
        print("\n[SETUP] 오케스트레이터 초기화 중...")
        
        if ingest_documents:
            self.retriever.ingest_documents()
        
        print("[OK] 오케스트레이터 준비 완료")
    
    def create_plan(self, question: str) -> Dict[str, Any]:
        """실행 계획 생성"""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.plan_prompt},
                {"role": "user", "content": f"질문: {question}"}
            ],
            temperature=0.1,
            response_format={"type": "json_object"}
        )
        
        try:
            plan = json.loads(response.choices[0].message.content)
        except:
            # 기본 계획
            plan = {
                "question": question,
                "steps": [
                    {"step_number": 1, "agent": "IntentClassifier", "action": "질문 분류", "input_from": None},
                    {"step_number": 2, "agent": "Retrieval", "action": "문서 검색", "input_from": 1},
                    {"step_number": 3, "agent": "Summarization", "action": "결과 요약", "input_from": 2},
                    {"step_number": 4, "agent": "FinalAnswer", "action": "답변 생성", "input_from": 3}
                ],
                "final_agent": "FinalAnswer"
            }
        
        # FinalAnswer 단계가 없으면 추가
        agent_names = [step.get("agent") for step in plan.get("steps", [])]
        if "FinalAnswer" not in agent_names:
            last_step = len(plan.get("steps", []))
            plan["steps"].append({
                "step_number": last_step + 1,
                "agent": "FinalAnswer",
                "action": "최종 답변 생성",
                "input_from": last_step
            })
            plan["final_agent"] = "FinalAnswer"
        
        return plan
    
    def execute_plan(self, question: str, verbose: bool = True) -> Dict[str, Any]:
        """
        계획을 실행하고 최종 결과 반환
        
        Args:
            question: 사용자 질문
            verbose: 상세 출력 여부
        
        Returns:
            실행 결과 딕셔너리
        """
        start_time = time.time()
        self.execution_log = []
        
        # 1. 계획 생성
        if verbose:
            print(f"\n{'='*60}")
            print(f"[PLAN] 실행 계획 생성 중...")
        
        plan = self.create_plan(question)
        
        if verbose:
            print(f"[OK] 계획 생성 완료: {len(plan['steps'])}단계")
            for step in plan['steps']:
                print(f"   {step['step_number']}. {step['agent']}: {step['action']}")
        
        # 2. 단계별 실행
        results = {}
        classification = None
        search_results = []
        summary = {}
        final_answer = ""
        
        for step in plan['steps']:
            step_start = time.time()
            agent_name = step['agent']
            
            if verbose:
                print(f"\n{'─'*40}")
                print(f"[>] Step {step['step_number']}: {agent_name}")
            
            if agent_name == "IntentClassifier":
                classification = self.classifier.classify(question)
                results['classification'] = classification
                
                if verbose:
                    print(f"   카테고리: {classification.category}")
                    print(f"   의도: {classification.intent}")
                    print(f"   확신도: {classification.confidence:.2f}")
            
            elif agent_name == "Retrieval":
                category = classification.category if classification else None
                search_results = self.retriever.search(question, k=5, category_filter=category)
                results['search_results'] = search_results
                
                if verbose:
                    print(f"   검색 결과: {len(search_results)}개")
                    if search_results:
                        print(f"   상위 결과 점수: {search_results[0].score:.4f}")
            
            elif agent_name == "Summarization":
                summary = self.summarizer.summarize(question, search_results)
                results['summary'] = summary
                
                if verbose:
                    print(f"   핵심 포인트: {len(summary.get('key_points', []))}개")
            
            elif agent_name == "FinalAnswer":
                final_answer = self.final_answer.generate_answer(
                    question, classification, summary, search_results
                )
                results['final_answer'] = final_answer
            
            step_time = time.time() - step_start
            self.execution_log.append({
                "step": step['step_number'],
                "agent": agent_name,
                "elapsed_time": step_time
            })
            
            if verbose:
                print(f"   소요 시간: {step_time:.2f}초")
        
        total_time = time.time() - start_time
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"[OK] 전체 실행 완료 (총 {total_time:.2f}초)")
        
        return {
            "question": question,
            "plan": plan,
            "classification": classification,
            "search_results": search_results,
            "summary": summary,
            "final_answer": final_answer,
            "total_time": total_time,
            "execution_log": self.execution_log
        }
    
    def process_question(self, question: str, verbose: bool = True) -> str:
        """질문 처리 및 답변 반환 (간단한 인터페이스)"""
        result = self.execute_plan(question, verbose=verbose)
        return result['final_answer']


# ============================================================================
# 데모 함수
# ============================================================================

def demo_tool_calling():
    """Tool/Function Calling 데모"""
    print("\n" + "="*80)
    print("[3-1] Tool/Function Calling")
    print("="*80)
    print("목표: LLM이 외부 도구를 자동으로 호출하는 방법 이해")
    
    if not os.getenv("OPENAI_API_KEY"):
        print("\n[!] OPENAI_API_KEY 환경변수를 설정해주세요!")
        return
    
    print_section_header("Tool Calling 개념", "[INFO]")
    print("""
  기존 LLM의 한계:
  * 실시간 정보 접근 불가 (현재 시간, 주가 등)
  * 정확한 계산 어려움
  * 외부 시스템 연동 불가
  
  Tool Calling의 해결:
  1. 개발자가 도구(함수)를 정의
  2. LLM이 필요한 도구와 인자를 자동 결정
  3. 도구 실행 후 결과를 LLM에 전달
  4. LLM이 최종 답변 생성
    """)
    
    agent = ToolCallingAgent()
    
    test_cases = [
        "2의 10제곱은 얼마인가요?",
        "현재 시간이 몇 시인가요?",
        "환불 절차에 대해 알려주세요."
    ]
    
    print_section_header("Tool Calling 실행 테스트", "[>>>]")
    
    for i, query in enumerate(test_cases, 1):
        print(f"\n{'─'*60}")
        print(f"[테스트 {i}] 질문: {query}")
        
        result = agent.run(query)
        
        if result["tool_calls"]:
            print(f"\n  [TOOL] 호출된 도구:")
            for j, tc in enumerate(result["tool_calls"], 1):
                print(f"    {j}. {tc['tool']}({tc['args']})")
                print(f"       결과: {tc['result'][:100]}...")
        else:
            print(f"\n  [INFO] 도구 호출 없음")
        
        print(f"\n  [답변] {result['answer']}")
        print(f"  [반복] {result['iterations']}회")
    
    print_key_points([
        "- Tool 정의: name, description, parameters (JSON Schema)",
        "- tool_choice: 'auto' (자동), 'none' (사용 안함)",
        "- 반복 루프: 도구 결과 -> LLM -> (추가 도구 or 최종 답변)",
        "- [!] 도구 결과 검증 필수: None/빈 결과 → 환각 차단"
    ], "Tool Calling 핵심")


def demo_conversation_memory():
    """대화 기록 관리 데모"""
    print("\n" + "="*80)
    print("[3-2] 대화 기록 관리 (Memory)")
    print("="*80)
    print("목표: 멀티턴 대화에서 맥락을 유지하는 방법")
    
    if not os.getenv("OPENAI_API_KEY"):
        print("\n[!] OPENAI_API_KEY 환경변수를 설정해주세요!")
        return
    
    print_section_header("Memory가 필요한 이유", "[INFO]")
    print("""
  [!] LLM은 기본적으로 "기억"이 없습니다!
  
  해결: 모든 대화를 메시지 리스트로 저장하고 매 API 호출 시 전달
  
  Memory 유형:
  * Buffer: 전체 저장 (토큰 제한 주의)
  * Window: 최근 N개만 유지 (구현 간단)
    """)
    
    agent = ConversationalAgent()
    
    conversations = [
        "안녕하세요, 제 이름은 민수입니다.",
        "저는 개발팀에서 일하고 있어요.",
        "제 이름이 뭐라고 했죠?",
        "어떤 팀에서 일한다고 했나요?"
    ]
    
    print_section_header("멀티턴 대화 테스트", "[>>>]")
    
    for i, user_msg in enumerate(conversations, 1):
        print(f"\n{'─'*60}")
        print(f"[턴 {i}] 사용자: {user_msg}")
        
        response = agent.chat(user_msg)
        print(f"[AI]: {response}")
    
    print_key_points([
        "- 대화 기록 = messages 리스트로 관리",
        "- 매 호출 시 전체 기록을 API에 전달",
        "- Window 방식: 최근 N개만 유지 (간단하고 효과적)",
        "- [!!!] 개인정보(이름, 전화번호)는 저장 금지!"
    ], "Memory 관리 핵심")


def demo_multi_agent():
    """멀티 에이전트 오케스트레이션 데모"""
    print("\n" + "="*80)
    print("[3-3] 멀티 에이전트 오케스트레이션")
    print("="*80)
    print("목표: Planner -> Worker 구조로 복잡한 질문 처리")
    
    if not os.getenv("OPENAI_API_KEY"):
        print("\n[!] OPENAI_API_KEY 환경변수를 설정해주세요!")
        return
    
    print_section_header("멀티 에이전트 아키텍처", "[ARCH]")
    print("""
  +-----------------------------------------------------------+
  |                Orchestrator (Planner)                      |
  |  - 실행 계획 수립                                          |
  |  - 에이전트 간 데이터 전달                                 |
  +-----------------------------------------------------------+
                        |
       +----------------+----------------+
       |                |                |
       v                v                v
  IntentClassifier  Retrieval  Summarization  FinalAnswer
    """)
    
    # API 비용 분석
    print_section_header("API 비용 분석", "[COST]")
    print("""
  질문 1개 처리 시 API 호출:
  1. Planner (계획 수립)     : LLM 1회
  2. IntentClassifier       : LLM 1회
  3. Retrieval (임베딩 생성) : Embedding 1회
  4. Summarization          : LLM 1회
  5. FinalAnswer            : LLM 1회
  ───────────────────────────────────────
  합계: LLM 4회 + Embedding 1회
  
  [!] 비용: 단순 RAG 대비 2~3배 API 호출
      복잡한 질문에만 사용 권장
    """)
    
    orchestrator = OrchestratorAgent()
    orchestrator.setup()
    
    test_questions = [
        "VIP 등급이 되려면 얼마나 구매해야 하나요?",
        "개발 환경 설정에 대해 알려주세요."
    ]
    
    print_section_header("멀티 에이전트 질의응답", "[>>>]")
    
    for question in test_questions:
        print(f"\n{'='*70}")
        print(f"[*] 질문: {question}")
        
        result = orchestrator.execute_plan(question, verbose=True)
        
        print(f"\n{'='*70}")
        print(f"[FINAL] 최종 답변:")
        print(f"{'─'*60}")
        for line in result['final_answer'].split('\n')[:5]:
            print(f"  {line}")
        print(f"{'─'*60}")
        
        print(f"\n[LOG] 총 소요 시간: {result['total_time']:.2f}초")
        print(f"      API 호출 횟수: ~{len(result['execution_log'])+1}회")
    
    print_key_points([
        "- Planner: 질문 분석 후 실행 계획 수립",
        "- Worker Agents: 각자 전문 영역 담당",
        "- 장점: 모듈화, 재사용성, 디버깅 용이",
        "- [!] 비용: 단순 RAG 대비 2~3배 (복잡한 질문에만 사용)"
    ], "멀티 에이전트 핵심")


def demo_full_pipeline():
    """전체 파이프라인 통합 실습"""
    print("\n" + "="*80)
    print("[3-4] 전체 파이프라인 통합 실습")
    print("="*80)
    print("시나리오: 다양한 부서의 질문을 자동으로 분류하고 답변 제공")
    
    if not os.getenv("OPENAI_API_KEY"):
        print("\n[!] OPENAI_API_KEY 환경변수를 설정해주세요!")
        return
    
    orchestrator = OrchestratorAgent()
    orchestrator.setup()
    
    # 샘플 질문으로 테스트
    test_questions = [
        SAMPLE_QUESTIONS[0],  # 고객센터
        SAMPLE_QUESTIONS[3],  # 개발
        SAMPLE_QUESTIONS[6],  # 기획
    ]
    
    print_section_header("자동 분류 + RAG 응답 테스트", "[TEST]")
    
    results = []
    
    for sample in test_questions:
        question = sample["question"]
        expected = sample["expected_category"]
        
        print(f"\n{'='*70}")
        print(f"[*] 질문: {question}")
        print(f"   예상 카테고리: {expected}")
        
        result = orchestrator.execute_plan(question, verbose=False)
        
        actual = result['classification'].category
        confidence = result['classification'].confidence
        match = "[OK]" if actual == expected else "[X]"
        
        print(f"\n[CLASSIFY] 분류 결과: {actual} {match}")
        print(f"   확신도: {confidence:.0%}")
        
        if result.get('search_results'):
            top_result = result['search_results'][0]
            print(f"   상위 검색 점수: {top_result.score:.4f}")
        
        print(f"\n[ANSWER] 답변:")
        print(f"{'─'*60}")
        for line in result['final_answer'].split('\n')[:5]:
            print(f"  {line}")
        print(f"{'─'*60}")
        
        results.append({
            "question": question,
            "expected": expected,
            "actual": actual,
            "match": actual == expected
        })
    
    # 결과 요약
    correct = sum(1 for r in results if r['match'])
    total = len(results)
    
    print_section_header("테스트 결과 요약", "[RESULT]")
    print(f"\n분류 정확도: {correct}/{total} ({correct/total*100:.0f}%)")
    print(f"\n[!] 주의: {total}개 샘플로는 성능 평가 부족")
    print(f"    실무에서는 수백~수천 개의 라벨링 데이터로 평가 필요")
    
    print_key_points([
        "- 자동 분류: 질문을 적절한 부서로 라우팅",
        "- RAG 응답: 해당 부서 문서에서 검색 후 답변",
        "- [!] 분류 정확도가 전체 품질을 결정",
        f"- [!] {total}개 샘플 테스트는 성능 평가로 부족",
        "- 실무 적용: 챗봇, 헬프데스크, 내부 지원 시스템"
    ], "전체 파이프라인 핵심")


# ============================================================================
# 메인 실행
# ============================================================================

def main():
    """Chapter 3 메인 실행"""
    print("\n" + "="*80)
    print("[LAB04 - Chapter 3] 멀티 에이전트 시스템")
    print("="*80)
    
    demo_tool_calling()
    demo_conversation_memory()
    demo_multi_agent()
    demo_full_pipeline()
    
    print("\n" + "="*80)
    print("[OK] Chapter 3 완료!")
    print("="*80)
    print("\n다음 단계: Chapter 4 (프로덕션 패턴)으로 이동하세요.")


if __name__ == "__main__":
    main()


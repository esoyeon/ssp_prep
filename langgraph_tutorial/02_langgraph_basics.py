"""
LangGraph 기본 개념 가이드
========================

LangGraph의 핵심 개념들을 차근차근 알아봅시다.
랭체인과의 차이점과 LangGraph만의 특별한 기능들을 이해해보겠습니다.
"""

from typing_extensions import TypedDict, Literal
from typing import List, Dict, Any, Optional
import time

# LangGraph가 무엇인가?
print("=== LangGraph란? ===")
print("""
🔍 LangGraph는 LangChain의 확장으로, 복잡한 AI 워크플로우를 
   그래프 형태로 구성할 수 있게 해주는 라이브러리입니다.

📊 기존 LangChain vs LangGraph:
   - LangChain: 순차적인 체인 (A → B → C)
   - LangGraph: 그래프 구조 (조건부 분기, 루프, 병렬 처리 가능)

🎯 언제 사용하나요?
   - 복잡한 의사결정이 필요할 때
   - 조건에 따라 다른 경로로 처리해야 할 때
   - 사용자 피드백이나 중간 결과에 따라 흐름을 바꿔야 할 때
   - 멀티 에이전트 시스템을 구축할 때
""")

# 1. 기본 구성 요소들
print("\n=== 1. LangGraph의 핵심 구성 요소 ===")

# State: 그래프 전체에서 공유되는 데이터
class BasicState(TypedDict):
    input_text: str
    processed_text: str
    step_count: int
    is_complete: bool

print("✅ State (상태)")
print("  - 그래프 전체에서 공유되는 데이터 저장소")
print("  - 각 노드가 읽고 수정할 수 있음")
print("  - TypedDict로 정의하여 타입 안정성 확보")

# Node: 실제 작업을 수행하는 함수
def example_node(state: BasicState) -> BasicState:
    """
    노드는 state를 받아서 작업을 수행하고 
    수정된 state를 반환하는 함수입니다.
    """
    return {
        **state,
        "processed_text": f"처리됨: {state['input_text']}",
        "step_count": state["step_count"] + 1
    }

print("\n✅ Node (노드)")
print("  - 실제 작업을 수행하는 함수")
print("  - state를 입력받고 수정된 state를 반환")
print("  - LLM 호출, 도구 사용, 데이터 처리 등 담당")

# Edge: 노드들 사이의 연결
print("\n✅ Edge (엣지)")
print("  - 노드들 사이의 연결선")
print("  - 일반 엣지: 항상 특정 노드로 이동")
print("  - 조건부 엣지: 조건에 따라 다른 노드로 이동")

# 2. 간단한 워크플로우 시뮬레이션
print("\n=== 2. 간단한 워크플로우 예시 ===")

class WorkflowState(TypedDict):
    user_input: str
    analysis_result: str
    final_response: str
    current_step: Literal["start", "analyze", "respond", "end"]

def start_node(state: WorkflowState) -> WorkflowState:
    """시작 노드: 사용자 입력을 받아 분석 단계로 넘어감"""
    print(f"🟢 START: 사용자 입력 받음 - '{state['user_input']}'")
    return {
        **state,
        "current_step": "analyze"
    }

def analyze_node(state: WorkflowState) -> WorkflowState:
    """분석 노드: 입력을 분석하여 결과 생성"""
    user_input = state["user_input"]
    
    # 간단한 분석 로직
    if "안녕" in user_input:
        analysis = "인사 표현 감지"
    elif "?" in user_input or "뭐" in user_input:
        analysis = "질문 의도 감지"
    elif "도움" in user_input:
        analysis = "도움 요청 감지"
    else:
        analysis = "일반 대화 감지"
    
    print(f"🔍 ANALYZE: {analysis}")
    
    return {
        **state,
        "analysis_result": analysis,
        "current_step": "respond"
    }

def respond_node(state: WorkflowState) -> WorkflowState:
    """응답 노드: 분석 결과를 바탕으로 응답 생성"""
    analysis = state["analysis_result"]
    
    # 분석 결과에 따른 응답 생성
    if "인사" in analysis:
        response = "안녕하세요! 반갑습니다 😊"
    elif "질문" in analysis:
        response = "좋은 질문이네요! 자세히 설명해드릴게요."
    elif "도움" in analysis:
        response = "물론 도와드리겠습니다! 어떤 도움이 필요하신가요?"
    else:
        response = "네, 말씀하신 내용을 잘 이해했습니다."
    
    print(f"💬 RESPOND: {response}")
    
    return {
        **state,
        "final_response": response,
        "current_step": "end"
    }

# 워크플로우 실행 시뮬레이션
print("\n📋 워크플로우 실행 예시:")

def simulate_workflow(user_input: str):
    """워크플로우 시뮬레이션"""
    print(f"\n--- 입력: '{user_input}' ---")
    
    # 초기 상태
    state = WorkflowState(
        user_input=user_input,
        analysis_result="",
        final_response="",
        current_step="start"
    )
    
    # 단계별 실행
    state = start_node(state)
    state = analyze_node(state)
    state = respond_node(state)
    
    print(f"✅ 최종 응답: {state['final_response']}")
    return state

# 여러 입력으로 테스트
test_inputs = [
    "안녕하세요!",
    "날씨가 어때요?",
    "도움이 필요해요",
    "오늘 기분이 좋네요"
]

for test_input in test_inputs:
    simulate_workflow(test_input)

# 3. 조건부 라우팅의 개념
print("\n=== 3. 조건부 라우팅 (Conditional Routing) ===")

def routing_example(state: WorkflowState) -> Literal["urgent", "normal", "simple"]:
    """
    조건부 라우팅 함수
    현재 상태를 보고 다음에 어떤 노드로 갈지 결정
    """
    user_input = state["user_input"]
    
    if "긴급" in user_input or "urgent" in user_input.lower():
        return "urgent"
    elif len(user_input) > 50:
        return "normal"
    else:
        return "simple"

print("🔀 조건부 라우팅의 활용:")
print("  - 입력 내용에 따라 다른 처리 경로 선택")
print("  - 복잡도에 따라 다른 모델 사용")
print("  - 사용자 권한에 따라 다른 기능 제공")

# 라우팅 테스트
routing_tests = [
    "긴급한 문제가 생겼어요!",
    "이것은 매우 복잡하고 긴 질문입니다. 여러 가지 조건들을 고려해서 답변해주세요.",
    "안녕하세요"
]

print("\n📍 라우팅 테스트:")
for test in routing_tests:
    test_state = WorkflowState(
        user_input=test,
        analysis_result="",
        final_response="",
        current_step="start"
    )
    route = routing_example(test_state)
    print(f"입력: '{test[:30]}...' → 라우팅: {route}")

# 4. LangGraph의 장점들
print("\n=== 4. LangGraph의 주요 장점들 ===")

advantages = {
    "🔄 순환 처리": "조건에 따라 이전 단계로 돌아가거나 반복 처리 가능",
    "🤖 멀티 에이전트": "여러 AI 에이전트가 협력하여 작업 수행",
    "👤 인간 개입": "중간에 사람의 승인이나 입력을 받을 수 있음",
    "🔀 동적 라우팅": "실행 중에 다음 경로를 동적으로 결정",
    "💾 상태 관리": "복잡한 상태를 체계적으로 관리",
    "🔍 관찰성": "각 단계의 실행 과정을 명확히 추적 가능"
}

for title, description in advantages.items():
    print(f"{title}: {description}")

# 5. 실제 사용 시나리오
print("\n=== 5. 실제 사용 시나리오 예시 ===")

scenarios = [
    {
        "제목": "📝 문서 작성 AI",
        "설명": "초안 작성 → 검토 → 수정 → 승인 → 최종화",
        "특징": "각 단계에서 품질 검사 및 피드백 반영"
    },
    {
        "제목": "🛒 쇼핑 도우미",
        "설명": "요구사항 분석 → 상품 검색 → 비교 → 추천 → 구매 지원",
        "특징": "사용자 선호도에 따라 다른 검색 전략 사용"
    },
    {
        "제목": "🔧 문제 해결 봇",
        "설명": "문제 진단 → 해결책 제시 → 실행 → 결과 확인 → 추가 조치",
        "특징": "해결 실패 시 다른 방법으로 재시도"
    },
    {
        "제목": "📚 학습 튜터",
        "설명": "레벨 테스트 → 맞춤 커리큘럼 → 학습 → 평가 → 피드백",
        "특징": "학습자 수준에 따라 난이도 자동 조절"
    }
]

for scenario in scenarios:
    print(f"\n{scenario['제목']}")
    print(f"  흐름: {scenario['설명']}")
    print(f"  특징: {scenario['특징']}")

# 6. 다음 단계 안내
print("\n=== 다음에 배울 내용들 ===")

next_topics = [
    "🏗️ 실제 그래프 구성하기 (StateGraph 사용)",
    "🔧 노드와 엣지 추가하는 방법",
    "🎯 조건부 엣지 구현하기",
    "💾 메모리와 체크포인트 활용",
    "🤖 실제 LLM과 연동하기",
    "📊 그래프 시각화 및 디버깅"
]

for i, topic in enumerate(next_topics, 1):
    print(f"{i}. {topic}")

print("\n" + "="*60)
print("🎓 LangGraph 기본 개념 학습 완료!")
print("이제 다음 파일에서 실제 그래프를 만들어보겠습니다.")
print("="*60)

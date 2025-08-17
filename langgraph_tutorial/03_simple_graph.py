"""
LangGraph 첫 번째 실습: 간단한 그래프 만들기
=============================================

실제 LangGraph를 사용해서 간단한 그래프를 만들어봅시다.
단계별로 차근차근 따라하면서 LangGraph의 작동 원리를 이해해보겠습니다.
"""

# 필요한 라이브러리 설치가 필요합니다:
# pip install langgraph

from typing_extensions import TypedDict, Literal
from typing import List, Dict, Any
from langgraph.graph import StateGraph, END, START
import time

print("=== LangGraph 첫 번째 실습 ===")
print("간단한 대화형 그래프를 만들어보겠습니다!")

# 1. State 정의
# =============

class ConversationState(TypedDict):
    """대화 상태를 정의하는 TypedDict"""
    user_input: str                    # 사용자 입력
    messages: List[Dict[str, str]]     # 대화 기록
    current_mood: str                  # 현재 감정 상태
    response: str                      # AI 응답
    step_count: int                    # 실행된 단계 수

print("✅ 1단계: State 정의 완료")
print("   - 대화에 필요한 모든 정보를 State에 저장")

# 2. 노드 함수들 정의
# ===================

def input_processor(state: ConversationState) -> ConversationState:
    """
    사용자 입력을 처리하는 노드
    입력을 분석하고 대화 기록에 추가
    """
    print(f"🔍 입력 처리 중: '{state['user_input']}'")
    
    # 사용자 메시지를 대화 기록에 추가
    new_messages = state["messages"] + [
        {"role": "user", "content": state["user_input"]}
    ]
    
    # 간단한 감정 분석
    user_input = state["user_input"].lower()
    if any(word in user_input for word in ["기쁘", "좋", "행복", "즐거"]):
        mood = "positive"
    elif any(word in user_input for word in ["슬프", "우울", "힘들", "속상"]):
        mood = "negative"
    elif any(word in user_input for word in ["화", "짜증", "화나", "분노"]):
        mood = "angry"
    else:
        mood = "neutral"
    
    return {
        **state,
        "messages": new_messages,
        "current_mood": mood,
        "step_count": state["step_count"] + 1
    }

def response_generator(state: ConversationState) -> ConversationState:
    """
    AI 응답을 생성하는 노드
    감정 상태에 따라 적절한 응답 생성
    """
    print(f"💭 응답 생성 중... (감정: {state['current_mood']})")
    
    mood = state["current_mood"]
    user_input = state["user_input"]
    
    # 감정에 따른 응답 생성
    if mood == "positive":
        response = f"좋은 기분이시네요! 😊 '{user_input}'라고 말씀하시니 저도 기분이 좋아집니다!"
    elif mood == "negative":
        response = f"조금 힘드신 것 같네요. 😔 괜찮으시다면 더 자세히 말씀해주세요."
    elif mood == "angry":
        response = f"화가 나신 것 같네요. 😤 어떤 일 때문인지 말씀해주시면 도움을 드릴게요."
    else:
        response = f"말씀해주신 '{user_input}'에 대해 더 알고 싶어요. 자세히 설명해주실 수 있나요?"
    
    # 응답을 대화 기록에 추가
    new_messages = state["messages"] + [
        {"role": "assistant", "content": response}
    ]
    
    return {
        **state,
        "messages": new_messages,
        "response": response,
        "step_count": state["step_count"] + 1
    }

def conversation_logger(state: ConversationState) -> ConversationState:
    """
    대화 로그를 출력하는 노드
    최종 결과를 보기 좋게 정리
    """
    print("📝 대화 요약 생성 중...")
    
    # 대화 요약 출력
    print("\n" + "="*50)
    print("📋 대화 요약")
    print("="*50)
    
    for i, message in enumerate(state["messages"], 1):
        role = "👤 사용자" if message["role"] == "user" else "🤖 AI"
        print(f"{i}. {role}: {message['content']}")
    
    print(f"\n감정 상태: {state['current_mood']}")
    print(f"실행된 단계: {state['step_count']}")
    print("="*50)
    
    return {
        **state,
        "step_count": state["step_count"] + 1
    }

print("✅ 2단계: 노드 함수들 정의 완료")
print("   - input_processor: 입력 처리 및 감정 분석")
print("   - response_generator: 감정에 따른 응답 생성")
print("   - conversation_logger: 대화 요약 출력")

# 3. 조건부 라우팅 함수 정의
# ==========================

def should_continue(state: ConversationState) -> Literal["continue", "end"]:
    """
    대화를 계속할지 종료할지 결정하는 함수
    """
    user_input = state["user_input"].lower()
    
    # 종료 키워드 체크
    end_keywords = ["끝", "종료", "그만", "bye", "exit", "quit"]
    
    if any(keyword in user_input for keyword in end_keywords):
        return "end"
    else:
        return "continue"

print("✅ 3단계: 조건부 라우팅 함수 정의 완료")
print("   - 사용자가 '끝', '종료' 등을 말하면 대화 종료")

# 4. 그래프 구성
# ==============

def create_conversation_graph():
    """대화 그래프를 생성하는 함수"""
    
    # StateGraph 인스턴스 생성
    workflow = StateGraph(ConversationState)
    
    # 노드들 추가
    workflow.add_node("input", input_processor)
    workflow.add_node("generate", response_generator)
    workflow.add_node("log", conversation_logger)
    
    # 엣지 연결
    # START → input → generate → log
    workflow.add_edge(START, "input")
    workflow.add_edge("input", "generate")
    workflow.add_edge("generate", "log")
    
    # 조건부 엣지: log 이후에 계속할지 종료할지 결정
    workflow.add_conditional_edges(
        "log",
        should_continue,
        {
            "continue": "input",  # 계속하면 input으로 돌아감
            "end": END           # 종료하면 END로
        }
    )
    
    # 그래프 컴파일
    app = workflow.compile()
    
    return app

print("✅ 4단계: 그래프 구성 완료")
print("   - 노드들을 연결하여 대화 플로우 완성")

# 5. 그래프 실행 함수
# ===================

def run_conversation_demo():
    """대화 데모를 실행하는 함수"""
    
    print("\n🚀 그래프 실행 시작!")
    
    # 그래프 생성
    app = create_conversation_graph()
    
    # 테스트 대화들
    test_conversations = [
        "안녕하세요! 오늘 기분이 정말 좋아요!",
        "요즘 일이 너무 힘들어서 우울해요...",
        "화가 나서 참을 수가 없어요!",
        "날씨가 어떤가요?",
        "끝"  # 종료 키워드
    ]
    
    # 초기 상태 설정
    initial_state = ConversationState(
        user_input="",
        messages=[],
        current_mood="neutral",
        response="",
        step_count=0
    )
    
    current_state = initial_state
    
    # 각 대화 실행
    for i, user_input in enumerate(test_conversations, 1):
        print(f"\n{'='*20} 대화 {i} {'='*20}")
        print(f"사용자 입력: '{user_input}'")
        
        # 사용자 입력을 상태에 설정
        current_state["user_input"] = user_input
        
        # 그래프 실행 (한 번의 대화 턴)
        try:
            # invoke를 사용하여 그래프 실행
            result = app.invoke(current_state)
            current_state = result
            
            # 종료 조건 체크
            if should_continue(current_state) == "end":
                print("\n👋 대화가 종료되었습니다!")
                break
                
        except Exception as e:
            print(f"❌ 오류 발생: {e}")
            break
    
    return current_state

# 6. 실행 및 결과 확인
# ===================

if __name__ == "__main__":
    try:
        print("🎯 LangGraph 실습을 시작합니다!")
        
        # 데모 실행
        final_state = run_conversation_demo()
        
        print("\n📊 최종 통계:")
        print(f"   - 총 메시지 수: {len(final_state['messages'])}")
        print(f"   - 실행된 단계 수: {final_state['step_count']}")
        print(f"   - 마지막 감정 상태: {final_state['current_mood']}")
        
    except ImportError:
        print("❌ langgraph가 설치되지 않았습니다!")
        print("다음 명령어로 설치해주세요:")
        print("pip install langgraph")
        
        # 설치 없이도 이해할 수 있는 시뮬레이션
        print("\n🔄 시뮬레이션 모드로 실행합니다...")
        
        def simulate_graph():
            """LangGraph 없이 동작을 시뮬레이션"""
            test_inputs = [
                "안녕하세요! 오늘 기분이 좋아요!",
                "요즘 힘들어서 우울해요",
                "끝"
            ]
            
            state = ConversationState(
                user_input="",
                messages=[],
                current_mood="neutral",
                response="",
                step_count=0
            )
            
            for user_input in test_inputs:
                print(f"\n--- 입력: '{user_input}' ---")
                state["user_input"] = user_input
                
                # 노드들 순차 실행
                state = input_processor(state)
                state = response_generator(state)
                state = conversation_logger(state)
                
                if should_continue(state) == "end":
                    break
        
        simulate_graph()

print("\n" + "="*60)
print("🎉 간단한 그래프 실습 완료!")
print("주요 학습 내용:")
print("  1. StateGraph 생성 및 노드 추가")
print("  2. 엣지 연결 (일반 엣지 + 조건부 엣지)")
print("  3. 그래프 컴파일 및 실행")
print("  4. 상태 관리 및 데이터 흐름")
print("\n다음 파일에서는 더 실용적인 에이전트를 만들어보겠습니다!")
print("="*60)

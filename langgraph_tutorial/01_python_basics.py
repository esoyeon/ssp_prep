
"""
LangGraph 사용을 위한 Python 기본 문법 가이드
===========================================

LangGraph를 사용하기 전에 알아야 할 Python 문법들을 설명합니다.
랭체인은 사용해봤지만 LangGraph는 처음인 분들을 위한 기초 가이드입니다.
"""

# 1. TypedDict - 타입이 명시된 딕셔너리
# ====================================

from typing_extensions import TypedDict, Literal, Union, Optional, List
from typing import Any, Dict

print("=== 1. TypedDict 기본 사용법 ===")

# 기본적인 TypedDict 정의
class UserInfo(TypedDict):
    name: str
    age: int
    email: str

# 사용 예시
user = UserInfo(name="홍길동", age=30, email="hong@example.com")
print(f"사용자 정보: {user}")
print(f"이름: {user['name']}, 나이: {user['age']}")

# LangGraph에서 자주 사용되는 State 정의 예시
class GraphState(TypedDict):
    messages: List[str]         # 메시지 리스트
    current_step: str          # 현재 단계
    result: Optional[str]      # 결과 (있을 수도, 없을 수도)
    
print("\n=== GraphState 예시 ===")
state = GraphState(
    messages=["안녕하세요", "도움이 필요합니다"],
    current_step="processing",
    result=None
)
print(f"현재 상태: {state}")


# 2. Literal - 특정 값만 허용하는 타입
# ==================================

print("\n=== 2. Literal 타입 ===")

# 특정 문자열만 허용
NodeType = Literal["start", "process", "end"]

def handle_node(node_type: NodeType) -> str:
    if node_type == "start":
        return "시작 노드입니다"
    elif node_type == "process":
        return "처리 노드입니다"
    elif node_type == "end":
        return "종료 노드입니다"
    
# 사용 예시
print(handle_node("start"))
print(handle_node("process"))
# print(handle_node("invalid"))  # 이것은 타입 에러를 발생시킵니다

# LangGraph에서 자주 사용되는 패턴
class NodeConfig(TypedDict):
    type: Literal["llm", "tool", "human"]
    name: str
    
config = NodeConfig(type="llm", name="gpt-4")
print(f"노드 설정: {config}")


# 3. Union - 여러 타입 중 하나
# ===========================

print("\n=== 3. Union 타입 ===")

# 여러 타입을 허용
MessageContent = Union[str, Dict[str, Any]]

def process_message(content: MessageContent) -> str:
    if isinstance(content, str):
        return f"텍스트 메시지: {content}"
    elif isinstance(content, dict):
        return f"구조화된 메시지: {content.get('text', '내용 없음')}"

# 사용 예시
print(process_message("안녕하세요"))
print(process_message({"text": "안녕하세요", "type": "greeting"}))


# 4. Optional - None이 될 수 있는 타입
# ==================================

print("\n=== 4. Optional 타입 ===")

def get_user_age(user_id: str) -> Optional[int]:
    # 실제로는 데이터베이스에서 조회하겠지만, 예시용으로
    if user_id == "123":
        return 25
    return None  # 사용자를 찾을 수 없음

age = get_user_age("123")
if age is not None:
    print(f"사용자 나이: {age}")
else:
    print("사용자를 찾을 수 없습니다")


# 5. LangGraph에서 자주 사용되는 패턴들
# ===================================

print("\n=== 5. LangGraph 패턴 예시 ===")

# 그래프 상태 정의
class ChatState(TypedDict):
    messages: List[Dict[str, str]]     # 대화 메시지들
    current_tool: Optional[str]        # 현재 사용 중인 도구
    step_count: int                    # 실행된 단계 수
    is_complete: bool                  # 완료 여부

# 노드 함수의 일반적인 형태
def my_node_function(state: ChatState) -> ChatState:
    """
    LangGraph의 노드 함수는 일반적으로:
    1. state를 입력받고
    2. state를 수정한 후
    3. 수정된 state를 반환합니다
    """
    # 상태 복사 (원본 수정 방지)
    new_state = state.copy()
    
    # 메시지 추가
    new_state["messages"].append({
        "role": "assistant",
        "content": "안녕하세요! 도움이 필요하시나요?"
    })
    
    # 단계 수 증가
    new_state["step_count"] += 1
    
    return new_state

# 초기 상태
initial_state = ChatState(
    messages=[{"role": "user", "content": "안녕하세요"}],
    current_tool=None,
    step_count=0,
    is_complete=False
)

print(f"초기 상태: {initial_state}")

# 노드 함수 실행
updated_state = my_node_function(initial_state)
print(f"업데이트된 상태: {updated_state}")


# 6. 조건부 라우팅을 위한 함수
# ============================

print("\n=== 6. 조건부 라우팅 예시 ===")

def decide_next_step(state: ChatState) -> Literal["continue", "tool_use", "end"]:
    """
    현재 상태를 보고 다음에 어떤 노드로 갈지 결정하는 함수
    LangGraph에서 conditional_edges에 사용됩니다
    """
    last_message = state["messages"][-1]["content"] if state["messages"] else ""
    
    if "도구" in last_message or "tool" in last_message.lower():
        return "tool_use"
    elif state["step_count"] >= 5:
        return "end"
    else:
        return "continue"

# 테스트
test_state = ChatState(
    messages=[{"role": "user", "content": "날씨 도구를 사용해주세요"}],
    current_tool=None,
    step_count=2,
    is_complete=False
)

next_step = decide_next_step(test_state)
print(f"다음 단계: {next_step}")


# 7. 실제 LangGraph에서 사용할 수 있는 완전한 예시
# ===============================================

print("\n=== 7. 완전한 예시 ===")

class CompleteState(TypedDict):
    input: str                          # 사용자 입력
    messages: List[Dict[str, str]]      # 대화 내역
    current_step: Literal["input", "process", "output"]  # 현재 단계
    result: Optional[str]               # 최종 결과
    error: Optional[str]                # 에러 메시지
    metadata: Dict[str, Any]            # 추가 정보

def input_node(state: CompleteState) -> CompleteState:
    """입력을 처리하는 노드"""
    return {
        **state,
        "current_step": "process",
        "messages": state["messages"] + [
            {"role": "user", "content": state["input"]}
        ]
    }

def process_node(state: CompleteState) -> CompleteState:
    """실제 처리를 하는 노드"""
    user_input = state["input"]
    
    # 간단한 처리 로직
    if "안녕" in user_input:
        response = "안녕하세요! 무엇을 도와드릴까요?"
    else:
        response = f"'{user_input}'에 대해 처리했습니다."
    
    return {
        **state,
        "current_step": "output",
        "result": response,
        "messages": state["messages"] + [
            {"role": "assistant", "content": response}
        ]
    }

# 사용 예시
complete_state = CompleteState(
    input="안녕하세요",
    messages=[],
    current_step="input",
    result=None,
    error=None,
    metadata={"timestamp": "2024-01-01"}
)

print(f"1. 초기 상태: {complete_state}")

# 단계별 실행
step1 = input_node(complete_state)
print(f"2. 입력 처리 후: {step1}")

step2 = process_node(step1)
print(f"3. 최종 결과: {step2}")

print("\n" + "="*50)
print("🎉 Python 기본 문법 학습 완료!")
print("다음 파일에서 실제 LangGraph 사용법을 배워보세요.")
print("="*50)

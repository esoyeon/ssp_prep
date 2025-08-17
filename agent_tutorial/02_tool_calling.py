"""
Tool Calling - 두 번째 단계
============================

이 파일에서는 LLM이 Tool을 어떻게 호출하는지 배워봅니다.

Tool Calling이란?
------------------
Tool Calling은 LLM이 주어진 문제를 해결하기 위해 
적절한 Tool을 선택하고 실행하는 과정입니다.

과정:
1. 사용자가 질문을 합니다
2. LLM이 질문을 분석해서 어떤 Tool이 필요한지 판단
3. LLM이 Tool을 호출하고 결과를 받음
4. LLM이 결과를 바탕으로 최종 답변 생성

이렇게 하면 LLM이 계산, 검색, 데이터 처리 등을 할 수 있게 됩니다!
"""

# 필요한 라이브러리 import
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from typing import Annotated
import json

# 환경변수 로드 (OpenAI API 키가 필요합니다)
from dotenv import load_dotenv
load_dotenv()

# ============================================================================
# 1단계: Tool 준비하기 (01_basic_tool.py에서 배운 내용)
# ============================================================================

@tool
def calculator(
    operation: Annotated[str, "연산 종류: add, subtract, multiply, divide"], 
    a: Annotated[float, "첫 번째 숫자"], 
    b: Annotated[float, "두 번째 숫자"]
) -> float:
    """수학 계산을 수행하는 계산기입니다."""
    
    if operation == "add":
        result = a + b
        print(f"계산: {a} + {b} = {result}")
    elif operation == "subtract":
        result = a - b
        print(f"계산: {a} - {b} = {result}")
    elif operation == "multiply":
        result = a * b
        print(f"계산: {a} × {b} = {result}")
    elif operation == "divide":
        if b == 0:
            return "0으로 나눌 수 없습니다!"
        result = a / b
        print(f"계산: {a} ÷ {b} = {result}")
    else:
        return f"지원하지 않는 연산입니다: {operation}"
    
    return result

@tool
def get_user_info(name: Annotated[str, "사용자 이름"]) -> dict:
    """사용자 정보를 조회하는 가짜 데이터베이스입니다."""
    
    # 가짜 사용자 데이터베이스
    users_db = {
        "김철수": {"나이": 25, "직업": "개발자", "취미": "독서"},
        "이영희": {"나이": 30, "직업": "디자이너", "취미": "그림그리기"},
        "박민수": {"나이": 28, "직업": "데이터분석가", "취미": "영화감상"},
    }
    
    if name in users_db:
        result = users_db[name]
        print(f"{name}님의 정보를 찾았습니다: {result}")
        return result
    else:
        return {"오류": f"{name}님의 정보를 찾을 수 없습니다."}

@tool  
def get_current_time() -> str:
    """현재 시간을 반환합니다."""
    from datetime import datetime
    now = datetime.now()
    time_str = now.strftime("%Y년 %m월 %d일 %H시 %M분")
    print(f"현재 시간: {time_str}")
    return time_str

# ============================================================================
# 2단계: LLM에 Tool 연결하기
# ============================================================================

def demo_tool_calling():
    """LLM이 Tool을 호출하는 과정을 보여주는 데모"""
    
    print("=== Tool Calling 데모 시작 ===\n")
    
    # LLM 모델 설정
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,  # 일관된 결과를 위해 0으로 설정
    )
    
    # Tool 목록
    tools = [calculator, get_user_info, get_current_time]
    
    # LLM에 Tool을 사용할 수 있도록 연결
    llm_with_tools = llm.bind_tools(tools)
    
    # ========================================================================
    # 예시 1: 계산 요청
    # ========================================================================
    print("🔢 예시 1: 계산 요청")
    user_question = "15에 23을 곱한 다음, 결과에서 47을 빼주세요"
    print(f"사용자 질문: {user_question}")
    
    # LLM에게 질문 전달
    messages = [
        SystemMessage(content="당신은 도움이 되는 AI 어시스턴트입니다. 주어진 도구들을 사용해서 사용자의 질문에 답해주세요."),
        HumanMessage(content=user_question)
    ]
    
    # LLM이 응답 생성 (Tool 호출 포함)
    response = llm_with_tools.invoke(messages)
    
    print(f"LLM 응답 타입: {type(response)}")
    print(f"응답 내용: {response.content}")
    
    # Tool 호출이 있는지 확인
    if response.tool_calls:
        print(f"\n🔧 LLM이 호출한 Tool 개수: {len(response.tool_calls)}")
        
        for i, tool_call in enumerate(response.tool_calls, 1):
            print(f"\nTool 호출 {i}:")
            print(f"  - Tool 이름: {tool_call['name']}")
            print(f"  - 파라미터: {tool_call['args']}")
            
            # 실제로 Tool 실행
            for tool in tools:
                if tool.name == tool_call['name']:
                    result = tool.invoke(tool_call['args'])
                    print(f"  - 실행 결과: {result}")
                    break
    
    print("\n" + "="*60 + "\n")
    
    # ========================================================================
    # 예시 2: 정보 조회 요청
    # ========================================================================
    print("👤 예시 2: 사용자 정보 조회")
    user_question = "김철수의 정보를 알려주세요"
    print(f"사용자 질문: {user_question}")
    
    messages = [
        SystemMessage(content="당신은 도움이 되는 AI 어시스턴트입니다. 주어진 도구들을 사용해서 사용자의 질문에 답해주세요."),
        HumanMessage(content=user_question)
    ]
    
    response = llm_with_tools.invoke(messages)
    print(f"응답 내용: {response.content}")
    
    if response.tool_calls:
        print(f"\n🔧 LLM이 호출한 Tool:")
        for tool_call in response.tool_calls:
            print(f"  - Tool: {tool_call['name']}")
            print(f"  - 파라미터: {tool_call['args']}")
            
            # Tool 실행
            for tool in tools:
                if tool.name == tool_call['name']:
                    result = tool.invoke(tool_call['args'])
                    print(f"  - 결과: {result}")
                    break
    
    print("\n" + "="*60 + "\n")
    
    # ========================================================================
    # 예시 3: 시간 조회 요청
    # ========================================================================
    print("🕐 예시 3: 현재 시간 조회")
    user_question = "지금 몇 시예요?"
    print(f"사용자 질문: {user_question}")
    
    messages = [
        SystemMessage(content="당신은 도움이 되는 AI 어시스턴트입니다. 주어진 도구들을 사용해서 사용자의 질문에 답해주세요."),
        HumanMessage(content=user_question)
    ]
    
    response = llm_with_tools.invoke(messages)
    print(f"응답 내용: {response.content}")
    
    if response.tool_calls:
        print(f"\n🔧 LLM이 호출한 Tool:")
        for tool_call in response.tool_calls:
            print(f"  - Tool: {tool_call['name']}")
            print(f"  - 파라미터: {tool_call['args']}")
            
            # Tool 실행
            for tool in tools:
                if tool.name == tool_call['name']:
                    result = tool.invoke(tool_call['args'])
                    print(f"  - 결과: {result}")
                    break

# ============================================================================
# 3단계: Tool Calling 과정 상세 설명
# ============================================================================

def explain_tool_calling_process():
    """Tool Calling이 어떻게 작동하는지 단계별로 설명"""
    
    print("\n🎯 Tool Calling 작동 원리:")
    print("="*50)
    
    print("""
1. 🧠 LLM이 사용자 질문을 분석
   - "15에 23을 곱한 다음, 47을 빼주세요"
   - → 이건 수학 계산이니까 calculator tool이 필요하겠네!

2. 🔧 LLM이 Tool 호출 계획 수립
   - 첫 번째: calculator(operation="multiply", a=15, b=23)
   - 두 번째: calculator(operation="subtract", a=결과, b=47)

3. ⚡ Tool 실행
   - 15 × 23 = 345
   - 345 - 47 = 298

4. 💬 최종 답변 생성
   - "계산 결과는 298입니다."
   
이 모든 과정이 자동으로 일어납니다!
    """)
    
    print("\n🎓 핵심 개념:")
    print("- LLM이 스스로 어떤 Tool을 사용할지 결정")
    print("- Tool의 파라미터도 LLM이 자동으로 채움")
    print("- 여러 Tool을 순서대로 사용 가능")
    print("- Tool 결과를 바탕으로 자연스러운 답변 생성")

if __name__ == "__main__":
    try:
        demo_tool_calling()
        explain_tool_calling_process()
        
        print("\n" + "="*50)
        print("다음 단계: 03_agent.py에서 Agent가 무엇인지 배워봅시다!")
        print("Agent는 Tool Calling을 더 똑똑하게 만들어줍니다.")
        
    except Exception as e:
        print(f"⚠️  오류 발생: {e}")
        print("OpenAI API 키가 설정되어 있는지 확인해주세요!")
        print("환경변수 OPENAI_API_KEY를 설정하거나 .env 파일을 생성해주세요.")

"""
실용적인 LangGraph 에이전트 만들기
=================================

실제로 사용할 수 있는 멀티 도구 에이전트를 만들어봅시다.
이 에이전트는 계산, 날씨 조회, 검색 등 여러 도구를 사용할 수 있습니다.
"""

from typing_extensions import TypedDict, Literal
from typing import List, Dict, Any, Optional, Union
import json
import random
import time
from datetime import datetime

# LangGraph 관련 imports
try:
    from langgraph.graph import StateGraph, END, START
    from langgraph.prebuilt import ToolNode
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    print("⚠️ LangGraph가 설치되지 않았습니다. 시뮬레이션 모드로 실행됩니다.")

print("=== 실용적인 멀티 도구 에이전트 ===")
print("다양한 도구를 사용할 수 있는 스마트 에이전트를 만들어봅시다!")

# 1. State 정의
# =============

class AgentState(TypedDict):
    """에이전트 상태를 정의하는 TypedDict"""
    messages: List[Dict[str, Any]]      # 대화 메시지들
    user_input: str                     # 현재 사용자 입력
    current_tool: Optional[str]         # 현재 사용 중인 도구
    tool_results: List[Dict[str, Any]]  # 도구 실행 결과들
    final_response: str                 # 최종 응답
    step_count: int                     # 실행 단계 수
    is_complete: bool                   # 작업 완료 여부

print("✅ 1단계: 복합 상태 정의 완료")

# 2. 도구 함수들 정의
# ===================

def calculator_tool(expression: str) -> Dict[str, Any]:
    """계산기 도구"""
    try:
        # 안전한 계산을 위해 eval 대신 간단한 연산만 지원
        expression = expression.replace(" ", "")
        
        # 기본 연산자들만 허용
        allowed_chars = set("0123456789+-*/.() ")
        if not all(c in allowed_chars for c in expression):
            raise ValueError("지원하지 않는 문자가 포함되어 있습니다.")
        
        result = eval(expression)
        
        return {
            "tool_name": "calculator",
            "input": expression,
            "result": result,
            "success": True,
            "message": f"{expression} = {result}"
        }
    except Exception as e:
        return {
            "tool_name": "calculator",
            "input": expression,
            "result": None,
            "success": False,
            "message": f"계산 오류: {str(e)}"
        }

def weather_tool(city: str) -> Dict[str, Any]:
    """날씨 조회 도구 (시뮬레이션)"""
    # 실제로는 날씨 API를 호출하겠지만, 여기서는 시뮬레이션
    weather_data = {
        "서울": {"temp": 23, "condition": "맑음", "humidity": 60},
        "부산": {"temp": 26, "condition": "흐림", "humidity": 70},
        "대구": {"temp": 28, "condition": "비", "humidity": 80},
        "인천": {"temp": 22, "condition": "맑음", "humidity": 55},
    }
    
    city = city.strip()
    if city in weather_data:
        data = weather_data[city]
        message = f"{city} 날씨: {data['condition']}, 온도: {data['temp']}°C, 습도: {data['humidity']}%"
        
        return {
            "tool_name": "weather",
            "input": city,
            "result": data,
            "success": True,
            "message": message
        }
    else:
        return {
            "tool_name": "weather",
            "input": city,
            "result": None,
            "success": False,
            "message": f"'{city}' 지역의 날씨 정보를 찾을 수 없습니다."
        }

def search_tool(query: str) -> Dict[str, Any]:
    """검색 도구 (시뮬레이션)"""
    # 실제로는 검색 API를 사용하겠지만, 여기서는 시뮬레이션
    search_results = {
        "파이썬": "Python은 1991년 귀도 반 로썸이 개발한 프로그래밍 언어입니다.",
        "AI": "인공지능(AI)은 인간의 지능을 모방하는 컴퓨터 시스템을 의미합니다.",
        "LangChain": "LangChain은 대규모 언어 모델을 활용한 애플리케이션 개발 프레임워크입니다.",
        "서울": "서울은 대한민국의 수도이며 약 970만 명의 인구가 거주합니다.",
    }
    
    # 키워드 매칭으로 간단한 검색 시뮬레이션
    for keyword, info in search_results.items():
        if keyword in query:
            return {
                "tool_name": "search",
                "input": query,
                "result": {"snippet": info, "keyword": keyword},
                "success": True,
                "message": f"'{keyword}' 검색 결과: {info}"
            }
    
    return {
        "tool_name": "search",
        "input": query,
        "result": None,
        "success": False,
        "message": f"'{query}'에 대한 검색 결과를 찾을 수 없습니다."
    }

def time_tool() -> Dict[str, Any]:
    """현재 시간 조회 도구"""
    current_time = datetime.now()
    
    return {
        "tool_name": "time",
        "input": "current_time",
        "result": {
            "datetime": current_time.isoformat(),
            "formatted": current_time.strftime("%Y년 %m월 %d일 %H시 %M분")
        },
        "success": True,
        "message": f"현재 시간: {current_time.strftime('%Y년 %m월 %d일 %H시 %M분')}"
    }

print("✅ 2단계: 다양한 도구 함수들 정의 완료")
print("   - calculator_tool: 수식 계산")
print("   - weather_tool: 날씨 조회")
print("   - search_tool: 정보 검색")
print("   - time_tool: 현재 시간")

# 3. 인텐트 분석 및 도구 선택
# ===========================

def analyze_intent(user_input: str) -> Dict[str, Any]:
    """사용자 입력을 분석해서 적절한 도구와 파라미터를 결정"""
    user_input = user_input.lower().strip()
    
    # 계산 인텐트
    calc_keywords = ["계산", "더하", "빼", "곱하", "나누", "+", "-", "*", "/", "="]
    if any(keyword in user_input for keyword in calc_keywords):
        # 수식 추출 시도
        import re
        # 숫자와 연산자가 포함된 부분 찾기
        math_pattern = r'[0-9+\-*/.() ]+'
        matches = re.findall(math_pattern, user_input)
        if matches:
            expression = max(matches, key=len).strip()
            return {
                "tool": "calculator",
                "params": {"expression": expression},
                "confidence": 0.9
            }
    
    # 날씨 인텐트
    weather_keywords = ["날씨", "기온", "온도", "비", "맑", "흐림"]
    if any(keyword in user_input for keyword in weather_keywords):
        # 도시명 추출
        cities = ["서울", "부산", "대구", "인천", "광주", "대전", "울산"]
        for city in cities:
            if city in user_input:
                return {
                    "tool": "weather",
                    "params": {"city": city},
                    "confidence": 0.9
                }
        # 도시가 명시되지 않으면 서울로 기본값
        return {
            "tool": "weather",
            "params": {"city": "서울"},
            "confidence": 0.7
        }
    
    # 시간 인텐트
    time_keywords = ["시간", "몇시", "현재", "지금"]
    if any(keyword in user_input for keyword in time_keywords):
        return {
            "tool": "time",
            "params": {},
            "confidence": 0.9
        }
    
    # 검색 인텐트 (기본값)
    return {
        "tool": "search",
        "params": {"query": user_input},
        "confidence": 0.6
    }

print("✅ 3단계: 지능적 인텐트 분석 시스템 완료")

# 4. 에이전트 노드 함수들
# =======================

def input_analyzer(state: AgentState) -> AgentState:
    """사용자 입력을 분석하는 노드"""
    print(f"🔍 입력 분석 중: '{state['user_input']}'")
    
    # 사용자 메시지를 대화 기록에 추가
    new_messages = state["messages"] + [
        {
            "role": "user",
            "content": state["user_input"],
            "timestamp": datetime.now().isoformat()
        }
    ]
    
    return {
        **state,
        "messages": new_messages,
        "step_count": state["step_count"] + 1
    }

def tool_selector(state: AgentState) -> AgentState:
    """적절한 도구를 선택하는 노드"""
    print("🔧 도구 선택 중...")
    
    # 인텐트 분석
    intent = analyze_intent(state["user_input"])
    print(f"   선택된 도구: {intent['tool']} (신뢰도: {intent['confidence']})")
    
    return {
        **state,
        "current_tool": intent["tool"],
        "step_count": state["step_count"] + 1,
        # 도구 파라미터를 메시지에 저장
        "messages": state["messages"] + [
            {
                "role": "system",
                "content": f"도구 선택: {intent['tool']}",
                "tool_params": intent["params"]
            }
        ]
    }

def tool_executor(state: AgentState) -> AgentState:
    """선택된 도구를 실행하는 노드"""
    current_tool = state["current_tool"]
    print(f"⚙️ 도구 실행 중: {current_tool}")
    
    # 마지막 시스템 메시지에서 도구 파라미터 가져오기
    tool_params = {}
    for message in reversed(state["messages"]):
        if message.get("role") == "system" and "tool_params" in message:
            tool_params = message["tool_params"]
            break
    
    # 도구 실행
    if current_tool == "calculator":
        result = calculator_tool(tool_params.get("expression", ""))
    elif current_tool == "weather":
        result = weather_tool(tool_params.get("city", "서울"))
    elif current_tool == "search":
        result = search_tool(tool_params.get("query", state["user_input"]))
    elif current_tool == "time":
        result = time_tool()
    else:
        result = {
            "tool_name": "unknown",
            "input": state["user_input"],
            "result": None,
            "success": False,
            "message": "알 수 없는 도구입니다."
        }
    
    print(f"   결과: {result['message']}")
    
    # 도구 결과를 상태에 추가
    new_tool_results = state["tool_results"] + [result]
    
    return {
        **state,
        "tool_results": new_tool_results,
        "step_count": state["step_count"] + 1
    }

def response_generator(state: AgentState) -> AgentState:
    """최종 응답을 생성하는 노드"""
    print("💬 응답 생성 중...")
    
    # 마지막 도구 결과 가져오기
    if state["tool_results"]:
        last_result = state["tool_results"][-1]
        
        if last_result["success"]:
            response = f"✅ {last_result['message']}"
            
            # 도구별 추가 정보 제공
            if last_result["tool_name"] == "calculator":
                response += "\n💡 다른 계산이 필요하시면 언제든 말씀해주세요!"
            elif last_result["tool_name"] == "weather":
                response += "\n🌤️ 다른 지역의 날씨도 궁금하시면 말씀해주세요!"
            elif last_result["tool_name"] == "search":
                response += "\n🔍 더 자세한 정보가 필요하시면 구체적으로 질문해주세요!"
            elif last_result["tool_name"] == "time":
                response += "\n⏰ 시간 관련해서 다른 궁금한 것이 있으시면 말씀해주세요!"
        else:
            response = f"❌ {last_result['message']}"
            response += "\n다시 시도해보시거나 다른 방식으로 질문해주세요."
    else:
        response = "죄송합니다. 요청을 처리할 수 없었습니다."
    
    # 응답을 메시지에 추가
    new_messages = state["messages"] + [
        {
            "role": "assistant",
            "content": response,
            "timestamp": datetime.now().isoformat()
        }
    ]
    
    return {
        **state,
        "messages": new_messages,
        "final_response": response,
        "is_complete": True,
        "step_count": state["step_count"] + 1
    }

print("✅ 4단계: 에이전트 노드 함수들 정의 완료")

# 5. 조건부 라우팅
# ================

def should_use_tool(state: AgentState) -> Literal["use_tool", "direct_response"]:
    """도구를 사용할지 직접 응답할지 결정"""
    user_input = state["user_input"].lower()
    
    # 간단한 인사나 감사 표현은 직접 응답
    simple_responses = ["안녕", "고마워", "감사", "반가워", "잘가", "bye"]
    
    if any(word in user_input for word in simple_responses):
        return "direct_response"
    else:
        return "use_tool"

def simple_response_generator(state: AgentState) -> AgentState:
    """간단한 응답을 생성하는 노드"""
    print("💭 간단 응답 생성 중...")
    
    user_input = state["user_input"].lower()
    
    if "안녕" in user_input:
        response = "안녕하세요! 😊 무엇을 도와드릴까요?"
    elif any(word in user_input for word in ["고마워", "감사"]):
        response = "천만에요! 😊 언제든 도움이 필요하시면 말씀해주세요!"
    elif any(word in user_input for word in ["잘가", "bye"]):
        response = "안녕히 가세요! 👋 좋은 하루 되세요!"
    else:
        response = "네, 말씀해주세요! 계산, 날씨, 검색, 시간 등 다양한 도움을 드릴 수 있어요."
    
    new_messages = state["messages"] + [
        {
            "role": "assistant",
            "content": response,
            "timestamp": datetime.now().isoformat()
        }
    ]
    
    return {
        **state,
        "messages": new_messages,
        "final_response": response,
        "is_complete": True,
        "step_count": state["step_count"] + 1
    }

print("✅ 5단계: 조건부 라우팅 로직 완료")

# 6. 그래프 구성
# ==============

def create_agent_graph():
    """멀티 도구 에이전트 그래프 생성"""
    
    if not LANGGRAPH_AVAILABLE:
        return None
    
    # StateGraph 인스턴스 생성
    workflow = StateGraph(AgentState)
    
    # 노드들 추가
    workflow.add_node("analyze_input", input_analyzer)
    workflow.add_node("select_tool", tool_selector)
    workflow.add_node("execute_tool", tool_executor)
    workflow.add_node("generate_response", response_generator)
    workflow.add_node("simple_response", simple_response_generator)
    
    # 엣지 연결
    workflow.add_edge(START, "analyze_input")
    
    # 조건부 엣지: 도구 사용 여부 결정
    workflow.add_conditional_edges(
        "analyze_input",
        should_use_tool,
        {
            "use_tool": "select_tool",
            "direct_response": "simple_response"
        }
    )
    
    # 도구 사용 플로우
    workflow.add_edge("select_tool", "execute_tool")
    workflow.add_edge("execute_tool", "generate_response")
    
    # 종료점들
    workflow.add_edge("generate_response", END)
    workflow.add_edge("simple_response", END)
    
    # 그래프 컴파일
    app = workflow.compile()
    
    return app

print("✅ 6단계: 복합 그래프 구성 완료")

# 7. 실행 및 테스트
# =================

def run_agent_demo():
    """에이전트 데모 실행"""
    
    print("\n🚀 멀티 도구 에이전트 시작!")
    
    # 테스트 케이스들
    test_cases = [
        "안녕하세요!",
        "15 + 27을 계산해주세요",
        "서울 날씨 어때요?",
        "현재 시간이 몇 시인가요?",
        "파이썬에 대해 알려주세요",
        "100 * 50 / 2는?",
        "부산 날씨도 알려주세요",
        "감사합니다!"
    ]
    
    if LANGGRAPH_AVAILABLE:
        # LangGraph로 실행
        app = create_agent_graph()
        
        for i, user_input in enumerate(test_cases, 1):
            print(f"\n{'='*15} 테스트 {i} {'='*15}")
            print(f"👤 사용자: {user_input}")
            
            # 초기 상태 설정
            initial_state = AgentState(
                messages=[],
                user_input=user_input,
                current_tool=None,
                tool_results=[],
                final_response="",
                step_count=0,
                is_complete=False
            )
            
            try:
                # 그래프 실행
                result = app.invoke(initial_state)
                print(f"🤖 에이전트: {result['final_response']}")
                print(f"📊 실행 단계: {result['step_count']}")
                
            except Exception as e:
                print(f"❌ 오류 발생: {e}")
    
    else:
        # 시뮬레이션 모드
        print("🔄 시뮬레이션 모드로 실행합니다...")
        
        for i, user_input in enumerate(test_cases, 1):
            print(f"\n{'='*15} 테스트 {i} {'='*15}")
            print(f"👤 사용자: {user_input}")
            
            # 상태 초기화
            state = AgentState(
                messages=[],
                user_input=user_input,
                current_tool=None,
                tool_results=[],
                final_response="",
                step_count=0,
                is_complete=False
            )
            
            # 노드들 순차 실행
            state = input_analyzer(state)
            
            if should_use_tool(state) == "use_tool":
                state = tool_selector(state)
                state = tool_executor(state)
                state = response_generator(state)
            else:
                state = simple_response_generator(state)
            
            print(f"🤖 에이전트: {state['final_response']}")
            print(f"📊 실행 단계: {state['step_count']}")

# 8. 실행
# =======

if __name__ == "__main__":
    run_agent_demo()

print("\n" + "="*60)
print("🎉 실용적인 멀티 도구 에이전트 완성!")
print("\n주요 특징:")
print("  🧠 지능적 인텐트 분석")
print("  🔧 다양한 도구 통합 (계산, 날씨, 검색, 시간)")
print("  🔀 조건부 라우팅 (간단한 질문은 직접 응답)")
print("  📝 완전한 대화 기록 관리")
print("  ⚡ 확장 가능한 구조")
print("\n🚀 이제 여러분만의 도구를 추가해서 에이전트를 확장해보세요!")
print("="*60)

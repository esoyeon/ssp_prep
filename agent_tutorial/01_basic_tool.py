"""
LangChain Tool 기초 - 첫 번째 단계
=====================================

이 파일에서는 LangChain의 Tool이 무엇인지, 어떻게 만드는지 배워봅니다.

Tool이란?
---------
Tool은 LLM(AI 모델)이 특정 작업을 수행할 수 있도록 해주는 함수입니다.
예를 들어:
- 계산기 역할을 하는 Tool
- 웹 검색을 하는 Tool  
- 파일을 읽는 Tool
- 데이터베이스를 조회하는 Tool

LLM은 텍스트만 생성할 수 있지만, Tool을 통해 실제 작업을 수행할 수 있게 됩니다!
"""

from langchain.tools import tool
from typing import Annotated

# ============================================================================
# 1단계: 가장 기본적인 Tool 만들기
# ============================================================================

@tool
def simple_calculator(
    a: Annotated[int, "첫 번째 숫자"], 
    b: Annotated[int, "두 번째 숫자"]
) -> int:
    """두 숫자를 더하는 간단한 계산기입니다."""
    result = a + b
    print(f"계산 결과: {a} + {b} = {result}")
    return result

# Tool을 직접 호출해보기
if __name__ == "__main__":
    print("=== 1단계: 기본 Tool 사용법 ===")
    
    # Tool의 정보 확인
    print(f"Tool 이름: {simple_calculator.name}")
    print(f"Tool 설명: {simple_calculator.description}")
    print(f"Tool 파라미터: {simple_calculator.args}")
    print()
    
    # Tool 직접 실행
    result = simple_calculator.invoke({"a": 5, "b": 3})
    print(f"직접 실행 결과: {result}")


# ============================================================================
# 2단계: 더 복잡한 Tool 만들기
# ============================================================================

@tool
def text_analyzer(text: Annotated[str, "분석할 텍스트"]) -> dict:
    """텍스트를 분석해서 다양한 정보를 반환합니다."""
    
    # 간단한 텍스트 분석
    word_count = len(text.split())
    char_count = len(text)
    char_count_no_space = len(text.replace(" ", ""))
    
    # 가장 긴 단어 찾기
    words = text.split()
    longest_word = max(words, key=len) if words else ""
    
    result = {
        "단어_개수": word_count,
        "문자_개수": char_count,
        "공백_제외_문자_개수": char_count_no_space,
        "가장_긴_단어": longest_word,
        "가장_긴_단어_길이": len(longest_word)
    }
    
    return result

@tool
def weather_info(city: Annotated[str, "도시 이름"]) -> str:
    """가짜 날씨 정보를 제공합니다 (실제 API 연결 없음)."""
    
    # 실제로는 날씨 API를 호출하겠지만, 여기서는 가짜 데이터 사용
    import random
    
    temperatures = [15, 18, 22, 25, 28, 30, 12, 8]
    weather_conditions = ["맑음", "흐림", "비", "눈", "안개"]
    
    temp = random.choice(temperatures)
    condition = random.choice(weather_conditions)
    
    return f"{city}의 현재 날씨: {condition}, 기온: {temp}°C"

# ============================================================================
# 3단계: 여러 Tool을 함께 사용해보기
# ============================================================================

if __name__ == "__main__":
    print("\n=== 2단계: 복잡한 Tool 사용법 ===")
    
    # 텍스트 분석 Tool 사용
    sample_text = "안녕하세요! LangChain Tool을 배우고 있습니다. 정말 재미있어요!"
    analysis_result = text_analyzer.invoke({"text": sample_text})
    print(f"텍스트 분석 결과: {analysis_result}")
    
    # 날씨 정보 Tool 사용
    weather_result = weather_info.invoke({"city": "서울"})
    print(f"날씨 정보: {weather_result}")
    
    print("\n=== 3단계: Tool 목록 관리 ===")
    
    # 여러 Tool을 하나의 리스트로 관리
    my_tools = [simple_calculator, text_analyzer, weather_info]
    
    print("사용 가능한 Tool 목록:")
    for i, tool in enumerate(my_tools, 1):
        print(f"{i}. {tool.name}: {tool.description}")
    
    print("\n" + "="*50)
    print("핵심 포인트:")
    print("1. @tool 데코레이터로 일반 함수를 Tool로 변환")
    print("2. Annotated로 파라미터 설명 추가")
    print("3. docstring으로 Tool 기능 설명")
    print("4. Tool.invoke()로 직접 실행 가능")
    print("5. 여러 Tool을 리스트로 관리")
    print("\n다음 파일에서는 LLM이 이러한 Tool을 어떻게 사용하는지 배워봅시다!")

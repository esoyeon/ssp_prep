"""
기본 SQL Agent 튜토리얼

이 파일은 LangGraph를 사용하여 간단한 SQL Agent를 구현하는 기본적인 예제입니다.
Agent와 LangGraph의 기본기를 알고 있는 학생들을 위한 입문 수준의 SQL Agent입니다.

주요 학습 목표:
1. SQL 데이터베이스와 상호작용하는 기본적인 Agent 구조 이해
2. LangGraph를 사용한 간단한 워크플로우 구현
3. SQL 도구 사용법 및 오류 처리 기초
"""

import os
import requests
from typing import Annotated, Literal, TypedDict

from dotenv import load_dotenv

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import AnyMessage, add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

# 환경 변수 로드
load_dotenv()

# 1. 데이터베이스 설정
def setup_database():
    """Chinook 샘플 데이터베이스를 다운로드하고 설정합니다."""
    
    # Chinook 데이터베이스 다운로드
    url = "https://storage.googleapis.com/benchmarks-artifacts/chinook/Chinook.db"
    
    if not os.path.exists("Chinook.db"):
        print("Chinook 데이터베이스를 다운로드 중...")
        response = requests.get(url)
        
        if response.status_code == 200:
            with open("Chinook.db", "wb") as file:
                file.write(response.content)
            print("데이터베이스 다운로드 완료!")
        else:
            raise Exception(f"데이터베이스 다운로드 실패: {response.status_code}")
    
    # SQLDatabase 인스턴스 생성
    db = SQLDatabase.from_uri("sqlite:///Chinook.db")
    print(f"사용 가능한 테이블: {db.get_usable_table_names()}")
    
    return db

# 2. Agent 상태 정의
class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

# 3. SQL 쿼리 실행 도구 정의
@tool
def execute_sql_query(query: str) -> str:
    """
    SQL 쿼리를 실행하고 결과를 반환합니다.
    오류가 발생하면 오류 메시지를 반환합니다.
    """
    try:
        result = db.run(query)
        return str(result)
    except Exception as e:
        return f"쿼리 실행 오류: {str(e)}"

# 4. 기본 SQL Agent 구현
def create_basic_sql_agent():
    """기본적인 SQL Agent를 생성합니다."""
    
    # LLM 초기화
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    # 도구 목록
    tools = [execute_sql_query]
    
    # 도구가 바인딩된 LLM
    llm_with_tools = llm.bind_tools(tools)
    
    # Agent 노드 정의
    def agent_node(state: State):
        """Agent가 사용자 질문을 분석하고 적절한 도구를 호출합니다."""
        
        # 시스템 메시지 추가
        system_message = """당신은 SQL 데이터베이스 전문가입니다.
        
사용자의 질문을 분석하여 적절한 SQL 쿼리를 작성하고 실행하세요.
        
데이터베이스 정보:
- Chinook 음악 스토어 데이터베이스
- 주요 테이블: Artist, Album, Track, Customer, Invoice, Employee 등

다음 단계를 따르세요:
1. 사용자 질문 분석
2. 적절한 SQL 쿼리 작성
3. execute_sql_query 도구를 사용하여 쿼리 실행
4. 결과를 사용자가 이해하기 쉽게 설명

주의사항:
- SELECT 쿼리만 사용하세요 (INSERT, UPDATE, DELETE 금지)
- 쿼리는 정확한 SQLite 문법을 사용하세요
"""
        
        messages = [AIMessage(content=system_message)] + state["messages"]
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}
    
    # 도구 노드
    tool_node = ToolNode(tools)
    
    # 라우팅 함수
    def should_continue(state: State) -> Literal["tools", "end"]:
        """다음 단계를 결정합니다."""
        last_message = state["messages"][-1]
        
        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            return "tools"
        else:
            return "end"
    
    # 그래프 생성
    workflow = StateGraph(State)
    
    # 노드 추가
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", tool_node)
    
    # 엣지 추가
    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": "tools",
            "end": END
        }
    )
    workflow.add_edge("tools", "agent")
    
    # 그래프 컴파일
    app = workflow.compile(checkpointer=MemorySaver())
    
    return app

# 5. Agent 실행 함수
def run_basic_agent(app, question: str):
    """기본 SQL Agent를 실행합니다."""
    
    config = {"configurable": {"thread_id": "basic_sql_agent"}}
    
    inputs = {
        "messages": [HumanMessage(content=question)]
    }
    
    print(f"\n📋 질문: {question}")
    print("=" * 50)
    
    for output in app.stream(inputs, config):
        for key, value in output.items():
            print(f"\n🔄 {key}:")
            if "messages" in value:
                last_message = value["messages"][-1]
                if hasattr(last_message, 'content') and last_message.content:
                    print(f"내용: {last_message.content}")
                if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
                    for tool_call in last_message.tool_calls:
                        print(f"도구 호출: {tool_call['name']}")
                        print(f"쿼리: {tool_call['args']['query']}")

# 6. 메인 실행부
if __name__ == "__main__":
    print("🎵 기본 SQL Agent 시작!")
    print("=" * 50)
    
    # 데이터베이스 설정
    db = setup_database()
    
    # Agent 생성
    app = create_basic_sql_agent()
    
    # 테스트 질문들
    test_questions = [
        "데이터베이스에 어떤 테이블들이 있나요?",
        "가장 인기 있는 아티스트 10명을 보여주세요",
        "2009년에 총 얼마의 매출이 있었나요?",
        "각 국가별 고객 수를 알려주세요"
    ]
    
    # 각 질문에 대해 Agent 실행
    for question in test_questions:
        try:
            run_basic_agent(app, question)
            print("\n" + "="*70 + "\n")
        except Exception as e:
            print(f"❌ 오류 발생: {str(e)}")
            continue
    
    print("✅ 기본 SQL Agent 테스트 완료!")

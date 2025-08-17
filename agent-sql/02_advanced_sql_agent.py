"""
실무 수준 SQL Agent 튜토리얼

이 파일은 LangGraph를 사용하여 실무에서 사용할 수 있는 고급 SQL Agent를 구현합니다.
기본 SQL Agent를 이해한 학생들을 위한 실무 수준의 구현입니다.

주요 특징:
1. 단계별 워크플로우 (테이블 조회 → 스키마 분석 → 쿼리 생성 → 검증 → 실행)
2. 오류 처리 및 재시도 메커니즘
3. SQL 쿼리 검증 시스템
4. 구조화된 응답 생성
5. 성능 최적화 및 로깅
"""

import os
import requests
from typing import Annotated, Literal, TypedDict, Any
from datetime import datetime

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableWithFallbacks
from langchain_openai import ChatOpenAI
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import AnyMessage, add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

# 환경 변수 로드
load_dotenv()

# 1. 고급 상태 정의
class AdvancedState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    current_tables: list[str]  # 현재 작업 중인 테이블들
    query_attempts: int  # 쿼리 시도 횟수
    error_history: list[str]  # 오류 이력
    execution_log: list[dict]  # 실행 로그

# 2. 구조화된 응답 모델
class FinalAnswer(BaseModel):
    """최종 답변을 위한 구조화된 모델"""
    answer: str = Field(description="사용자 질문에 대한 명확하고 완전한 답변")
    sql_query: str = Field(description="실행된 SQL 쿼리")
    execution_time: float = Field(description="쿼리 실행 시간 (초)")
    result_count: int = Field(description="반환된 결과 행 수")

# 3. 데이터베이스 설정 (향상된 버전)
class AdvancedDatabaseManager:
    def __init__(self):
        self.db = None
        self.setup_database()
    
    def setup_database(self):
        """고급 데이터베이스 설정 및 초기화"""
        
        # Chinook 데이터베이스 다운로드
        url = "https://storage.googleapis.com/benchmarks-artifacts/chinook/Chinook.db"
        
        if not os.path.exists("Chinook.db"):
            print("🔄 Chinook 데이터베이스 다운로드 중...")
            response = requests.get(url)
            
            if response.status_code == 200:
                with open("Chinook.db", "wb") as file:
                    file.write(response.content)
                print("✅ 데이터베이스 다운로드 완료!")
            else:
                raise Exception(f"❌ 데이터베이스 다운로드 실패: {response.status_code}")
        
        # SQLDatabase 인스턴스 생성
        self.db = SQLDatabase.from_uri("sqlite:///Chinook.db")
        print(f"📊 사용 가능한 테이블: {self.db.get_usable_table_names()}")
    
    def get_table_info(self, tables: list[str]) -> str:
        """특정 테이블들의 상세 정보를 반환"""
        try:
            return self.db.get_table_info(tables)
        except Exception as e:
            return f"테이블 정보 조회 오류: {str(e)}"
    
    def execute_query_safe(self, query: str) -> tuple[str, bool, float]:
        """안전한 쿼리 실행 (실행시간 측정 포함)"""
        start_time = datetime.now()
        try:
            result = self.db.run(query)
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()
            return str(result), True, execution_time
        except Exception as e:
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()
            return f"쿼리 실행 오류: {str(e)}", False, execution_time

# 전역 데이터베이스 매니저 인스턴스
db_manager = AdvancedDatabaseManager()

# 4. 고급 도구들 정의

@tool
def list_all_tables() -> str:
    """데이터베이스의 모든 테이블 목록을 반환합니다."""
    try:
        tables = db_manager.db.get_usable_table_names()
        return f"사용 가능한 테이블: {', '.join(tables)}"
    except Exception as e:
        return f"테이블 목록 조회 오류: {str(e)}"

@tool  
def get_table_schema(table_names: str) -> str:
    """
    특정 테이블들의 스키마 정보를 반환합니다.
    table_names: 쉼표로 구분된 테이블 이름들 (예: "Customer, Invoice")
    """
    try:
        tables = [t.strip() for t in table_names.split(",")]
        return db_manager.get_table_info(tables)
    except Exception as e:
        return f"스키마 조회 오류: {str(e)}"

@tool
def execute_sql_query(query: str) -> str:
    """
    SQL 쿼리를 안전하게 실행하고 결과를 반환합니다.
    SELECT 쿼리만 허용되며, DML 작업은 금지됩니다.
    """
    # DML 쿼리 방지
    dangerous_keywords = ["INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "CREATE"]
    query_upper = query.upper().strip()
    
    for keyword in dangerous_keywords:
        if keyword in query_upper:
            return f"❌ 보안상 {keyword} 쿼리는 허용되지 않습니다."
    
    result, success, exec_time = db_manager.execute_query_safe(query)
    
    if success:
        return f"✅ 쿼리 실행 성공 (실행시간: {exec_time:.3f}초)\n결과:\n{result}"
    else:
        return f"❌ {result}"

# 5. 오류 처리 유틸리티
def handle_tool_error(state) -> dict:
    """도구 실행 오류를 처리합니다."""
    error = state.get("error")
    tool_calls = state["messages"][-1].tool_calls
    
    error_message = f"❌ 도구 실행 오류: {repr(error)}\n\n다시 시도해주세요."
    
    return {
        "messages": [
            ToolMessage(
                content=error_message,
                tool_call_id=tc["id"],
            )
            for tc in tool_calls
        ]
    }

def create_tool_node_with_fallback(tools: list) -> RunnableWithFallbacks[Any, dict]:
    """오류 처리가 포함된 ToolNode를 생성합니다."""
    return ToolNode(tools).with_fallbacks(
        [RunnableLambda(handle_tool_error)], exception_key="error"
    )

# 6. SQL 쿼리 검증 시스템
class SQLQueryValidator:
    def __init__(self, model_name: str = "gpt-4o"):
        self.llm = ChatOpenAI(model=model_name, temperature=0)
        self.validation_prompt = ChatPromptTemplate.from_messages([
            ("system", """당신은 SQL 전문가입니다. 주어진 SQLite 쿼리의 일반적인 오류를 검토하고 수정하세요.

검토할 항목:
- NULL 값과 NOT IN 사용 오류
- UNION vs UNION ALL 적절성
- BETWEEN 범위 설정 오류  
- 데이터 타입 불일치
- 식별자 인용 오류
- 함수 인수 개수 오류
- 조인 컬럼 정확성
- 문법 오류

오류가 있다면 수정된 쿼리를 반환하고, 없다면 원본 쿼리를 그대로 반환하세요.
쿼리만 반환하고 추가 설명은 하지 마세요."""),
            ("user", "검토할 쿼리: {query}")
        ])
    
    def validate_query(self, query: str) -> str:
        """SQL 쿼리를 검증하고 수정된 버전을 반환합니다."""
        try:
            response = self.validation_prompt.invoke({"query": query}) | self.llm
            return response.invoke({}).content.strip()
        except Exception as e:
            print(f"⚠️ 쿼리 검증 중 오류: {str(e)}")
            return query

# 7. 고급 SQL Agent 구현
def create_advanced_sql_agent():
    """실무 수준의 고급 SQL Agent를 생성합니다."""
    
    # LLM 및 도구 초기화
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    query_validator = SQLQueryValidator()
    
    # 도구 목록
    tools = [list_all_tables, get_table_schema, execute_sql_query]
    
    # 워크플로우 노드들 정의
    
    def start_analysis(state: AdvancedState):
        """분석을 시작하고 테이블 목록을 강제로 조회합니다."""
        return {
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[{
                        "name": "list_all_tables",
                        "args": {},
                        "id": "initial_table_list",
                    }]
                )
            ],
            "current_tables": [],
            "query_attempts": 0,
            "error_history": [],
            "execution_log": []
        }
    
    def select_relevant_tables(state: AdvancedState):
        """사용자 질문과 관련된 테이블을 선택합니다."""
        
        table_selection_prompt = ChatPromptTemplate.from_messages([
            ("system", """당신은 데이터베이스 전문가입니다. 
            
사용자의 질문을 분석하여 관련된 테이블들을 선택하세요.

Chinook 데이터베이스 테이블 정보:
- Artist: 아티스트 정보
- Album: 앨범 정보  
- Track: 음악 트랙 정보
- Customer: 고객 정보
- Invoice: 주문/청구서 정보
- InvoiceLine: 주문 상세 정보
- Employee: 직원 정보
- Genre: 장르 정보
- MediaType: 미디어 타입 정보
- Playlist, PlaylistTrack: 플레이리스트 정보

사용자 질문에 답하기 위해 필요한 테이블들을 쉼표로 구분하여 나열하세요.
예: Customer, Invoice"""),
            ("placeholder", "{messages}")
        ])
        
        model_with_schema_tool = llm.bind_tools([get_table_schema])
        schema_chain = table_selection_prompt | model_with_schema_tool
        
        response = schema_chain.invoke(state)
        return {"messages": [response]}
    
    def generate_and_validate_query(state: AdvancedState):
        """SQL 쿼리를 생성하고 검증합니다."""
        
        query_generation_prompt = ChatPromptTemplate.from_messages([
            ("system", """당신은 SQL 전문가입니다.

사용자의 질문과 제공된 테이블 스키마를 바탕으로 정확한 SQLite 쿼리를 작성하세요.

지침:
1. SELECT 쿼리만 작성하세요
2. 정확한 테이블명과 컬럼명을 사용하세요
3. 적절한 JOIN을 사용하여 관련 테이블을 연결하세요
4. WHERE, GROUP BY, ORDER BY 등을 적절히 활용하세요
5. 결과가 너무 많으면 LIMIT을 사용하세요

현재까지의 대화를 분석하여:
- 오류가 발생했다면 오류 메시지를 그대로 반환하세요
- 쿼리 결과가 있다면 "Answer: " 형식으로 답변하세요  
- 쿼리가 필요하다면 SQL 쿼리만 반환하세요"""),
            ("placeholder", "{messages}")
        ])
        
        model_with_submit = llm.bind_tools([FinalAnswer])
        query_chain = query_generation_prompt | model_with_submit
        
        response = query_chain.invoke(state)
        
        # 쿼리 검증 (쿼리가 생성된 경우에만)
        if not response.tool_calls and response.content.strip():
            validated_query = query_validator.validate_query(response.content)
            response.content = validated_query
        
        return {"messages": [response]}
    
    def should_continue(state: AdvancedState) -> Literal[END, "execute_query", "generate_query"]:
        """다음 단계를 결정합니다."""
        last_message = state["messages"][-1]
        
        if hasattr(last_message, 'content'):
            content = last_message.content
            if content.startswith("Answer:"):
                return END
            elif content.startswith("❌") or content.startswith("Error:"):
                # 최대 재시도 횟수 확인
                if state.get("query_attempts", 0) >= 3:
                    return END
                return "generate_query"
            elif content.strip() and not hasattr(last_message, 'tool_calls'):
                return "execute_query"
        
        return "generate_query"
    
    def execute_query_node(state: AdvancedState):
        """검증된 쿼리를 실행합니다."""
        last_message = state["messages"][-1]
        query = last_message.content.strip()
        
        # 쿼리 실행
        response = AIMessage(
            content="",
            tool_calls=[{
                "name": "execute_sql_query", 
                "args": {"query": query},
                "id": f"query_exec_{state.get('query_attempts', 0)}"
            }]
        )
        
        # 실행 로그 업데이트
        new_log = {
            "attempt": state.get("query_attempts", 0) + 1,
            "query": query,
            "timestamp": datetime.now().isoformat()
        }
        
        execution_log = state.get("execution_log", [])
        execution_log.append(new_log)
        
        return {
            "messages": [response],
            "query_attempts": state.get("query_attempts", 0) + 1,
            "execution_log": execution_log
        }
    
    # 그래프 구성
    workflow = StateGraph(AdvancedState)
    
    # 노드 추가
    workflow.add_node("start_analysis", start_analysis)
    workflow.add_node("list_tables", create_tool_node_with_fallback([list_all_tables]))
    workflow.add_node("select_tables", select_relevant_tables)
    workflow.add_node("get_schema", create_tool_node_with_fallback([get_table_schema]))
    workflow.add_node("generate_query", generate_and_validate_query)
    workflow.add_node("execute_query", execute_query_node)
    workflow.add_node("run_query", create_tool_node_with_fallback([execute_sql_query]))
    
    # 엣지 정의
    workflow.add_edge(START, "start_analysis")
    workflow.add_edge("start_analysis", "list_tables")
    workflow.add_edge("list_tables", "select_tables")
    workflow.add_edge("select_tables", "get_schema")
    workflow.add_edge("get_schema", "generate_query")
    
    workflow.add_conditional_edges(
        "generate_query",
        should_continue,
        {
            "execute_query": "execute_query",
            "generate_query": "generate_query",
            END: END
        }
    )
    
    workflow.add_edge("execute_query", "run_query")
    workflow.add_edge("run_query", "generate_query")
    
    # 컴파일
    app = workflow.compile(checkpointer=MemorySaver())
    
    return app

# 8. 실행 및 모니터링 함수
def run_advanced_agent(app, question: str, verbose: bool = True):
    """고급 SQL Agent를 실행하고 상세한 로그를 출력합니다."""
    
    config = {"configurable": {"thread_id": f"advanced_sql_agent_{datetime.now().timestamp()}"}}
    
    inputs = {
        "messages": [HumanMessage(content=question)],
        "current_tables": [],
        "query_attempts": 0,
        "error_history": [],
        "execution_log": []
    }
    
    print(f"\n🎯 질문: {question}")
    print("=" * 80)
    
    step_count = 0
    
    try:
        for output in app.stream(inputs, config):
            step_count += 1
            for key, value in output.items():
                if verbose:
                    print(f"\n📍 단계 {step_count}: {key}")
                    print("-" * 40)
                
                if "messages" in value and value["messages"]:
                    last_message = value["messages"][-1]
                    
                    if hasattr(last_message, 'content') and last_message.content:
                        if verbose:
                            print(f"내용: {last_message.content[:200]}...")
                    
                    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
                        for tool_call in last_message.tool_calls:
                            if verbose:
                                print(f"🔧 도구: {tool_call['name']}")
                                if 'query' in tool_call.get('args', {}):
                                    print(f"📝 쿼리: {tool_call['args']['query']}")
        
        # 최종 상태 정보 출력
        final_state = app.get_state(config).values
        if verbose and final_state.get("execution_log"):
            print(f"\n📊 실행 통계:")
            print(f"- 총 쿼리 시도: {len(final_state['execution_log'])}")
            print(f"- 최종 시도 횟수: {final_state.get('query_attempts', 0)}")
            
    except Exception as e:
        print(f"❌ Agent 실행 중 오류: {str(e)}")

# 9. 메인 실행부
if __name__ == "__main__":
    print("🚀 고급 SQL Agent 시작!")
    print("=" * 80)
    
    # Agent 생성
    app = create_advanced_sql_agent()
    
    # 실무 수준 테스트 질문들
    advanced_test_questions = [
        "2009년에 가장 많은 매출을 올린 국가는 어디이고, 얼마를 벌었나요?",
        "가장 인기 있는 음악 장르 5개와 각각의 판매량을 알려주세요",
        "직원별 2009년 매출 실적을 내림차순으로 정렬해서 보여주세요",
        "평균 주문 금액이 가장 높은 고객 10명의 정보를 알려주세요",
        "Led Zeppelin의 앨범 수와 트랙 수를 알려주세요"
    ]
    
    # 각 질문에 대해 Agent 실행
    for i, question in enumerate(advanced_test_questions, 1):
        print(f"\n🔍 테스트 {i}/{len(advanced_test_questions)}")
        try:
            run_advanced_agent(app, question, verbose=True)
            print("\n" + "="*100 + "\n")
        except Exception as e:
            print(f"❌ 테스트 {i} 실행 중 오류: {str(e)}")
            continue
    
    print("✅ 고급 SQL Agent 테스트 완료!")
    
    # 대화형 모드
    print("\n💬 대화형 모드 시작 (종료하려면 'quit' 입력)")
    print("-" * 50)
    
    while True:
        try:
            user_input = input("\n❓ 질문을 입력하세요: ").strip()
            
            if user_input.lower() in ['quit', 'exit', '종료']:
                print("👋 SQL Agent를 종료합니다.")
                break
            
            if user_input:
                run_advanced_agent(app, user_input, verbose=False)
                
        except KeyboardInterrupt:
            print("\n👋 사용자에 의해 종료되었습니다.")
            break
        except Exception as e:
            print(f"❌ 오류: {str(e)}")
            continue

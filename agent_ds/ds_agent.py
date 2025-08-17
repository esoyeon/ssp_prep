from dotenv import load_dotenv
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from langchain.tools import tool
from langchain_experimental.tools.python.tool import PythonAstREPLTool
from langchain_openai import ChatOpenAI
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate
from typing import Annotated


# API 키 정보 로드
load_dotenv()


# 1) CSV를 읽어 Pandas DataFrame 생성
df = pd.read_csv("./california_housing_train.csv")


# 2) 커스텀 Python 실행 툴 생성
@tool
def python_data_analysis_tool(
    code: Annotated[str, "Python code using pandas, matplotlib, seaborn for data analysis and visualization"]
) -> str:
    """
    Execute Python code for data analysis and visualization.
    The DataFrame is available as 'df' variable.
    Use pandas for data manipulation, matplotlib/seaborn for visualization.
    """
    try:
        # PythonAstREPLTool에 DataFrame과 필요한 라이브러리들을 locals로 전달
        python_tool = PythonAstREPLTool(
            locals={
                "df": df, 
                "pd": pd, 
                "plt": plt, 
                "sns": sns
            }
        )
        result = python_tool.invoke(code)
        return result
    except Exception as e:
        return f"실행 중 오류가 발생했습니다: {repr(e)}"


# 3) LLM 설정
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()],
)

# 4) 프롬프트 템플릿 생성
prompt = ChatPromptTemplate.from_messages([
    ("system", """
당신은 전문적인 데이터 분석가이며 Pandas 전문가입니다.
사용자의 요청에 따라 pandas DataFrame(변수명: df)을 사용하여 데이터 분석과 시각화를 수행해야 합니다.

## 데이터프레임 정보:
{dataframe_info}

## 중요한 규칙:
1. 반드시 python_data_analysis_tool을 사용해서 코드를 실행하세요.
2. DataFrame 변수 'df'를 생성하거나 덮어쓰지 마세요. 이미 로드되어 있습니다.
3. 시각화 코드의 경우 반드시 마지막에 plt.show()를 포함하세요.
4. seaborn을 선호하지만 matplotlib도 사용 가능합니다.

## 시각화 스타일 가이드:
- muted 색상 팔레트 사용
- 흰색 배경
- 그리드 제거

## 사용 가능한 라이브러리:
- pandas (pd)
- matplotlib.pyplot (plt) 
- seaborn (sns)
"""),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}")
])

# 5) 에이전트 생성
tools = [python_data_analysis_tool]
agent = create_tool_calling_agent(llm, tools, prompt)

# 6) AgentExecutor 생성
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    max_iterations=10,
    handle_parsing_errors=True,
)


if __name__ == "__main__":
    # 데이터프레임 정보를 프롬프트에 추가
    dataframe_info = f"""
컬럼 정보: {list(df.columns)}
데이터 타입: {df.dtypes.to_dict()}
데이터 형태: {df.shape}
데이터 샘플 (처음 5행):
{df.head().to_string()}
"""

    # 테스트 실행
    print("=== 커스텀 데이터 분석 에이전트 테스트 ===")
    result = agent_executor.invoke({
        "input": "corr()를 구해서 히트맵 시각화",
        "dataframe_info": dataframe_info
    })
    print(f"\n최종 결과: {result['output']}")
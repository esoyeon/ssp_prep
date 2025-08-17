from dotenv import load_dotenv
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate
from typing import Annotated
import sys
import io
import contextlib
import traceback
import warnings
warnings.filterwarnings('ignore')

# API 키 정보 로드
load_dotenv()

# CSV 데이터 로드
df = pd.read_csv("./california_housing_train.csv")

class SafePythonExecutor:
    """
    LangChain의 PythonAstREPLTool과 유사한 기능을 제공하는 안전한 Python 실행기
    """
    def __init__(self, locals_dict=None):
        self.locals_dict = locals_dict or {}
        self.globals_dict = {
            '__builtins__': {
                # 기본적인 내장 함수들만 허용
                'len': len,
                'range': range,
                'enumerate': enumerate,
                'zip': zip,
                'list': list,
                'dict': dict,
                'tuple': tuple,
                'set': set,
                'str': str,
                'int': int,
                'float': float,
                'bool': bool,
                'sum': sum,
                'min': min,
                'max': max,
                'abs': abs,
                'round': round,
                'sorted': sorted,
                'print': print,
                'type': type,
                'isinstance': isinstance,
            }
        }
        
    def execute(self, code: str) -> str:
        """
        Python 코드를 안전하게 실행하고 결과를 반환
        """
        # 코드 출력을 캡처하기 위한 StringIO 객체
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        
        try:
            # 표준 출력/에러를 캡처
            sys.stdout = stdout_capture
            sys.stderr = stderr_capture
            
            # 위험한 키워드들 체크
            dangerous_keywords = [
                'import os', 'import sys', 'import subprocess', 
                'exec', 'eval', '__import__', 'open', 'file',
                'input', 'raw_input', 'globals', 'locals',
                'getattr', 'setattr', 'delattr', 'hasattr'
            ]
            
            for keyword in dangerous_keywords:
                if keyword in code:
                    return f"보안상 금지된 키워드가 포함되어 있습니다: {keyword}"
            
            # locals_dict에 필요한 변수들 추가
            execution_locals = self.locals_dict.copy()
            
            # 코드 실행
            try:
                # 단일 표현식인지 확인 (결과값을 반환해야 하는 경우)
                try:
                    compile(code, '<string>', 'eval')
                    # 표현식이면 eval로 실행하고 결과 반환
                    result = eval(code, self.globals_dict, execution_locals)
                    if result is not None:
                        return str(result)
                except SyntaxError:
                    # 표현식이 아니면 exec로 실행
                    exec(code, self.globals_dict, execution_locals)
                
                # 출력 결과 가져오기
                output = stdout_capture.getvalue()
                if output:
                    return output.strip()
                else:
                    return "코드가 성공적으로 실행되었습니다."
                    
            except Exception as e:
                error_output = stderr_capture.getvalue()
                if error_output:
                    return f"실행 오류:\n{error_output}"
                else:
                    return f"실행 오류: {str(e)}\n{traceback.format_exc()}"
                
        finally:
            # 표준 출력/에러 복원
            sys.stdout = old_stdout
            sys.stderr = old_stderr

# Python 실행기 인스턴스 생성
python_executor = SafePythonExecutor(locals_dict={
    'df': df,
    'pd': pd,
    'plt': plt,
    'sns': sns,
    'np': np,
    'numpy': np
})

@tool
def python_repl_tool(
    code: Annotated[str, "Python code to execute. DataFrame is available as 'df'. You can use pandas, matplotlib, seaborn, numpy."]
) -> str:
    """
    Execute Python code for data analysis and visualization.
    Available variables:
    - df: The loaded DataFrame
    - pd: pandas library
    - plt: matplotlib.pyplot
    - sns: seaborn
    - np: numpy
    
    For visualization, make sure to call plt.show() at the end.
    """
    return python_executor.execute(code)

# LLM 설정
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()],
)

# 프롬프트 템플릿 (LangChain pandas agent와 유사하게)
prompt = ChatPromptTemplate.from_messages([
    ("system", """
You are working with a pandas dataframe in Python. The name of the dataframe is `df`.
You should use the tool below to answer the question posed of you:

python_repl_tool: A Python shell. Use this to execute python commands. Input should be a valid python command. 
When using this tool, sometimes output is abbreviated - make sure it does not look abbreviated before using it in your answer.

This is the result of `print(df.head())`:
{df_head}

[IMPORTANT] You are a professional data analyst and expert in Pandas. 
You must use Pandas DataFrame(`df`) to answer user's request. 

[IMPORTANT] DO NOT create or overwrite the `df` variable in your code. 

If you generate visualization code, please use `plt.show()` at the end. 
I prefer seaborn, but matplotlib is OK.

<Visualization Preference>
- muted cmap, white background, no grid.
"""),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}")
])

# 툴 리스트
tools = [python_repl_tool]

# 에이전트 생성
agent = create_tool_calling_agent(llm, tools, prompt)

# AgentExecutor 생성 (LangChain pandas agent와 동일한 설정)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    max_iterations=20,
    max_execution_time=60,
    handle_parsing_errors=True,
    return_intermediate_steps=True
)

if __name__ == "__main__":
    print("=== LangChain 스타일 데이터 분석 에이전트 ===")
    print("* LangChain의 create_pandas_dataframe_agent와 동일한 방식으로 동작")
    print("* 유연한 Python 코드 실행 지원")
    print("* 복잡한 데이터 분석 및 시각화 가능")
    print()
    
    # 데이터프레임 정보 준비
    df_head_info = df.head().to_string()
    
    # 테스트 실행
    test_queries = [
        "각 컬럼의 기본 통계를 확인하고 median_house_value와 다른 변수들 간의 상관관계 분석",
        "median_income이 8 이상인 데이터만 필터링해서 median_house_value의 분포를 히스토그램으로 그려줘",
        "corr()를 구해서 히트맵 시각화"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{'='*60}")
        print(f"테스트 {i}: {query}")
        print('='*60)
        
        try:
            result = agent_executor.invoke({
                "input": query,
                "df_head": df_head_info
            })
            print(f"\n[최종 결과] {result['output']}")
        except Exception as e:
            print(f"\n[오류] {str(e)}")
        
        if i < len(test_queries):
            input("\n다음 테스트로 계속하려면 Enter를 누르세요...")

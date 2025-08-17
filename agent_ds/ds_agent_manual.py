from dotenv import load_dotenv
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import io
import base64
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate
from typing import Annotated, Optional, List, Dict, Any
import warnings
warnings.filterwarnings('ignore')

# API 키 정보 로드
load_dotenv()

# CSV 데이터 로드
df = pd.read_csv("./california_housing_train.csv")

# === 기본 데이터 탐색 툴들 ===

@tool
def get_dataframe_info() -> str:
    """Get basic information about the dataframe including shape, columns, and data types."""
    info = f"""
데이터프레임 기본 정보:
- 형태 (행, 열): {df.shape}
- 컬럼명: {list(df.columns)}
- 데이터 타입:
{df.dtypes.to_string()}
- 메모리 사용량: {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB
- Null 값 개수:
{df.isnull().sum().to_string()}
"""
    return info

@tool
def get_dataframe_head(n: Annotated[int, "Number of rows to display"] = 5) -> str:
    """Display the first n rows of the dataframe."""
    return f"데이터프레임 상위 {n}행:\n{df.head(n).to_string()}"

@tool
def get_dataframe_tail(n: Annotated[int, "Number of rows to display"] = 5) -> str:
    """Display the last n rows of the dataframe."""
    return f"데이터프레임 하위 {n}행:\n{df.tail(n).to_string()}"

@tool
def get_dataframe_describe() -> str:
    """Get descriptive statistics of numerical columns."""
    return f"수치형 컬럼 기술통계:\n{df.describe().to_string()}"

@tool
def get_column_unique_values(column: Annotated[str, "Column name to get unique values"]) -> str:
    """Get unique values and their counts for a specific column."""
    if column not in df.columns:
        return f"오류: '{column}' 컬럼이 존재하지 않습니다. 사용 가능한 컬럼: {list(df.columns)}"
    
    if df[column].dtype in ['object', 'category']:
        result = f"'{column}' 컬럼의 고유값 및 개수:\n{df[column].value_counts().to_string()}"
    else:
        unique_count = df[column].nunique()
        result = f"'{column}' 컬럼의 고유값 개수: {unique_count}\n"
        if unique_count <= 20:
            result += f"고유값들:\n{sorted(df[column].unique())}"
        else:
            result += f"일부 고유값들 (처음 20개):\n{sorted(df[column].unique())[:20]}"
    
    return result

# === 통계 분석 툴들 ===

@tool
def calculate_correlation() -> str:
    """Calculate correlation matrix for numerical columns."""
    numeric_df = df.select_dtypes(include=[np.number])
    corr_matrix = numeric_df.corr()
    return f"상관관계 행렬:\n{corr_matrix.to_string()}"

@tool
def group_by_analysis(
    group_column: Annotated[str, "Column to group by"],
    target_column: Annotated[str, "Column to analyze"],
    operation: Annotated[str, "Operation: mean, sum, count, min, max, std"] = "mean"
) -> str:
    """Perform group by analysis on the dataframe."""
    if group_column not in df.columns:
        return f"오류: '{group_column}' 컬럼이 존재하지 않습니다."
    if target_column not in df.columns:
        return f"오류: '{target_column}' 컬럼이 존재하지 않습니다."
    
    try:
        grouped = df.groupby(group_column)[target_column]
        
        if operation == "mean":
            result = grouped.mean()
        elif operation == "sum":
            result = grouped.sum()
        elif operation == "count":
            result = grouped.count()
        elif operation == "min":
            result = grouped.min()
        elif operation == "max":
            result = grouped.max()
        elif operation == "std":
            result = grouped.std()
        else:
            return f"지원하지 않는 연산입니다: {operation}. 사용 가능: mean, sum, count, min, max, std"
        
        return f"'{group_column}' 기준 '{target_column}' {operation} 결과:\n{result.to_string()}"
    except Exception as e:
        return f"그룹 분석 중 오류 발생: {str(e)}"

@tool
def filter_dataframe(
    column: Annotated[str, "Column to filter"],
    operator: Annotated[str, "Operator: >, <, >=, <=, ==, !="],
    value: Annotated[float, "Value to compare"]
) -> str:
    """Filter dataframe based on condition and show basic stats."""
    if column not in df.columns:
        return f"오류: '{column}' 컬럼이 존재하지 않습니다."
    
    try:
        if operator == ">":
            filtered_df = df[df[column] > value]
        elif operator == "<":
            filtered_df = df[df[column] < value]
        elif operator == ">=":
            filtered_df = df[df[column] >= value]
        elif operator == "<=":
            filtered_df = df[df[column] <= value]
        elif operator == "==":
            filtered_df = df[df[column] == value]
        elif operator == "!=":
            filtered_df = df[df[column] != value]
        else:
            return f"지원하지 않는 연산자: {operator}"
        
        result = f"필터 조건: {column} {operator} {value}\n"
        result += f"필터링된 데이터 수: {len(filtered_df)}행 (전체의 {len(filtered_df)/len(df)*100:.1f}%)\n"
        result += f"필터링된 데이터 샘플:\n{filtered_df.head().to_string()}"
        
        return result
    except Exception as e:
        return f"필터링 중 오류 발생: {str(e)}"

# === 시각화 툴들 ===

@tool
def create_correlation_heatmap() -> str:
    """Create a correlation heatmap for numerical columns."""
    try:
        numeric_df = df.select_dtypes(include=[np.number])
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            numeric_df.corr(), 
            annot=True, 
            cmap='muted', 
            center=0,
            square=True,
            fmt='.2f'
        )
        plt.title('상관관계 히트맵')
        plt.tight_layout()
        
        # 그리드 제거 및 배경 설정
        plt.gca().set_facecolor('white')
        plt.grid(False)
        
        plt.show()
        return "상관관계 히트맵이 생성되었습니다."
    except Exception as e:
        return f"히트맵 생성 중 오류 발생: {str(e)}"

@tool
def create_histogram(
    column: Annotated[str, "Column name for histogram"],
    bins: Annotated[int, "Number of bins"] = 30
) -> str:
    """Create histogram for a specific column."""
    if column not in df.columns:
        return f"오류: '{column}' 컬럼이 존재하지 않습니다."
    
    try:
        plt.figure(figsize=(10, 6))
        plt.hist(df[column].dropna(), bins=bins, alpha=0.7, color='skyblue', edgecolor='black')
        plt.title(f'{column} 히스토그램')
        plt.xlabel(column)
        plt.ylabel('빈도')
        plt.grid(False)
        plt.gca().set_facecolor('white')
        plt.tight_layout()
        plt.show()
        return f"'{column}' 컬럼의 히스토그램이 생성되었습니다."
    except Exception as e:
        return f"히스토그램 생성 중 오류 발생: {str(e)}"

@tool
def create_scatter_plot(
    x_column: Annotated[str, "X-axis column name"],
    y_column: Annotated[str, "Y-axis column name"]
) -> str:
    """Create scatter plot between two columns."""
    if x_column not in df.columns:
        return f"오류: '{x_column}' 컬럼이 존재하지 않습니다."
    if y_column not in df.columns:
        return f"오류: '{y_column}' 컬럼이 존재하지 않습니다."
    
    try:
        plt.figure(figsize=(10, 6))
        plt.scatter(df[x_column], df[y_column], alpha=0.6, color='coral')
        plt.title(f'{x_column} vs {y_column} 산점도')
        plt.xlabel(x_column)
        plt.ylabel(y_column)
        plt.grid(False)
        plt.gca().set_facecolor('white')
        plt.tight_layout()
        plt.show()
        return f"'{x_column}'과 '{y_column}' 간의 산점도가 생성되었습니다."
    except Exception as e:
        return f"산점도 생성 중 오류 발생: {str(e)}"

@tool
def create_box_plot(column: Annotated[str, "Column name for box plot"]) -> str:
    """Create box plot for a specific column."""
    if column not in df.columns:
        return f"오류: '{column}' 컬럼이 존재하지 않습니다."
    
    try:
        plt.figure(figsize=(8, 6))
        sns.boxplot(data=df, y=column, palette='muted')
        plt.title(f'{column} 박스플롯')
        plt.grid(False)
        plt.gca().set_facecolor('white')
        plt.tight_layout()
        plt.show()
        return f"'{column}' 컬럼의 박스플롯이 생성되었습니다."
    except Exception as e:
        return f"박스플롯 생성 중 오류 발생: {str(e)}"

@tool
def create_pairplot(columns: Annotated[str, "Comma-separated column names"] = None) -> str:
    """Create pairplot for specified columns or all numerical columns."""
    try:
        if columns:
            column_list = [col.strip() for col in columns.split(',')]
            # 존재하지 않는 컬럼 체크
            invalid_cols = [col for col in column_list if col not in df.columns]
            if invalid_cols:
                return f"오류: 다음 컬럼들이 존재하지 않습니다: {invalid_cols}"
            plot_df = df[column_list]
        else:
            plot_df = df.select_dtypes(include=[np.number])
        
        if len(plot_df.columns) > 10:
            return "경고: 컬럼이 너무 많습니다 (10개 초과). 특정 컬럼들을 지정해주세요."
        
        sns.pairplot(plot_df, diag_kind='hist', palette='muted')
        plt.suptitle('Pairplot', y=1.02)
        plt.show()
        return f"Pairplot이 생성되었습니다. (사용된 컬럼: {list(plot_df.columns)})"
    except Exception as e:
        return f"Pairplot 생성 중 오류 발생: {str(e)}"

# === 에이전트 설정 ===

# 모든 툴 리스트
tools = [
    get_dataframe_info,
    get_dataframe_head,
    get_dataframe_tail,
    get_dataframe_describe,
    get_column_unique_values,
    calculate_correlation,
    group_by_analysis,
    filter_dataframe,
    create_correlation_heatmap,
    create_histogram,
    create_scatter_plot,
    create_box_plot,
    create_pairplot
]

# LLM 설정
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()],
)

# 프롬프트 템플릿
prompt = ChatPromptTemplate.from_messages([
    ("system", """
당신은 전문적인 데이터 분석가입니다. California Housing 데이터셋을 분석하기 위한 다양한 툴들이 제공됩니다.

## 사용 가능한 툴들:

### 기본 탐색:
- get_dataframe_info: 데이터프레임 기본 정보
- get_dataframe_head/tail: 데이터 미리보기
- get_dataframe_describe: 기술통계
- get_column_unique_values: 컬럼의 고유값 확인

### 통계 분석:
- calculate_correlation: 상관관계 계산
- group_by_analysis: 그룹별 분석
- filter_dataframe: 조건부 필터링

### 시각화:
- create_correlation_heatmap: 상관관계 히트맵
- create_histogram: 히스토그램
- create_scatter_plot: 산점도
- create_box_plot: 박스플롯
- create_pairplot: 페어플롯

## 분석 가이드라인:
1. 먼저 데이터의 기본 정보를 파악하세요
2. 사용자 요청에 맞는 적절한 툴을 선택하세요
3. 시각화 시에는 결과를 명확히 설명하세요
4. 복잡한 분석은 단계별로 진행하세요

현재 데이터셋: California Housing (주택 가격 예측 데이터)
"""),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}")
])

# 에이전트 생성
agent = create_tool_calling_agent(llm, tools, prompt)

# AgentExecutor 생성
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    max_iterations=15,
    handle_parsing_errors=True,
)

if __name__ == "__main__":
    print("=== 수동 구현 데이터 분석 에이전트 ===")
    print("사용 가능한 기능:")
    print("1. 기본 데이터 탐색")
    print("2. 통계 분석")  
    print("3. 다양한 시각화")
    print()
    
    # 테스트 실행
    test_query = "corr()를 구해서 히트맵 시각화"
    print(f"테스트 쿼리: {test_query}")
    print("-" * 50)
    
    result = agent_executor.invoke({"input": test_query})
    print(f"\n최종 결과: {result['output']}")

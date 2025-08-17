"""
실습 예제 - 네 번째 단계
========================

이 파일에서는 지금까지 배운 내용을 종합해서 
실제로 유용한 데이터 분석 Agent를 만들어봅니다.

종합 실습: 학생 성적 분석 Agent
-------------------------------
이 Agent는 다음 기능들을 제공합니다:
1. 📊 학생별 성적 조회
2. 📈 과목별 통계 분석  
3. 🏆 순위 계산
4. 📋 성적표 생성
5. 📉 시각화 (그래프)

실제 데이터 분석 업무에서 자주 사용되는 패턴들을 배워봅시다!
"""

# 필요한 라이브러리 import
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate
from typing import Annotated, List, Dict
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
from datetime import datetime

# 환경변수 로드
from dotenv import load_dotenv
load_dotenv()

# ============================================================================
# 1단계: 샘플 데이터 생성
# ============================================================================

def create_sample_data():
    """학생 성적 샘플 데이터를 생성합니다."""
    
    students = ["김철수", "이영희", "박민수", "최지은", "정다영", "윤석민", "한소영", "오준호"]
    subjects = ["수학", "영어", "과학", "국어", "사회"]
    
    data = []
    for student in students:
        for subject in subjects:
            score = random.randint(60, 100)  # 60~100점 사이
            data.append({
                "학생명": student,
                "과목": subject,
                "점수": score,
                "학기": "2024-1학기"
            })
    
    return pd.DataFrame(data)

# 전역 데이터프레임 생성
df_scores = create_sample_data()

# ============================================================================
# 2단계: 데이터 분석 Tool들 정의
# ============================================================================

@tool
def get_student_scores(student_name: Annotated[str, "학생 이름"]) -> dict:
    """특정 학생의 모든 과목 성적을 조회합니다."""
    
    student_data = df_scores[df_scores["학생명"] == student_name]
    
    if student_data.empty:
        return {"오류": f"'{student_name}' 학생을 찾을 수 없습니다."}
    
    scores = {}
    total = 0
    for _, row in student_data.iterrows():
        scores[row["과목"]] = row["점수"]
        total += row["점수"]
    
    average = total / len(scores)
    
    result = {
        "학생명": student_name,
        "과목별_점수": scores,
        "총점": total,
        "평균": round(average, 1),
        "과목수": len(scores)
    }
    
    print(f"{student_name} 학생 성적 조회 완료")
    return result

@tool 
def get_subject_statistics(subject: Annotated[str, "과목명"]) -> dict:
    """특정 과목의 통계를 분석합니다."""
    
    subject_data = df_scores[df_scores["과목"] == subject]
    
    if subject_data.empty:
        return {"오류": f"'{subject}' 과목을 찾을 수 없습니다."}
    
    scores = subject_data["점수"].tolist()
    
    result = {
        "과목": subject,
        "평균": round(subject_data["점수"].mean(), 1),
        "최고점": subject_data["점수"].max(),
        "최저점": subject_data["점수"].min(),
        "표준편차": round(subject_data["점수"].std(), 1),
        "학생수": len(scores),
        "최고점_학생": subject_data[subject_data["점수"] == subject_data["점수"].max()]["학생명"].iloc[0]
    }
    
    print(f"{subject} 과목 통계 분석 완료")
    return result

@tool
def get_class_ranking() -> List[dict]:
    """전체 학생의 평균 성적 순위를 계산합니다."""
    
    # 학생별 평균 계산
    student_averages = df_scores.groupby("학생명")["점수"].agg(['mean', 'sum', 'count']).reset_index()
    student_averages.columns = ["학생명", "평균", "총점", "과목수"]
    student_averages["평균"] = student_averages["평균"].round(1)
    
    # 평균 기준으로 정렬
    student_averages = student_averages.sort_values("평균", ascending=False).reset_index(drop=True)
    student_averages["순위"] = range(1, len(student_averages) + 1)
    
    ranking = []
    for _, row in student_averages.iterrows():
        ranking.append({
            "순위": row["순위"],
            "학생명": row["학생명"],
            "평균": row["평균"],
            "총점": int(row["총점"])
        })
    
    print("전체 학급 순위 계산 완료")
    return ranking

@tool
def create_score_visualization(chart_type: Annotated[str, "차트 종류: bar, scatter, heatmap"]) -> str:
    """성적 데이터를 시각화합니다."""
    
    plt.figure(figsize=(12, 8))
    plt.style.use('default')
    
    if chart_type == "bar":
        # 과목별 평균 점수 막대 그래프
        subject_avg = df_scores.groupby("과목")["점수"].mean()
        
        plt.subplot(2, 1, 1)
        bars = plt.bar(subject_avg.index, subject_avg.values, color='skyblue', alpha=0.7)
        plt.title("과목별 평균 점수", fontsize=14, fontweight='bold')
        plt.ylabel("평균 점수")
        plt.ylim(0, 100)
        
        # 막대 위에 점수 표시
        for bar, value in zip(bars, subject_avg.values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                    f'{value:.1f}', ha='center', va='bottom')
        
        # 학생별 평균 점수 막대 그래프
        student_avg = df_scores.groupby("학생명")["점수"].mean().sort_values(ascending=False)
        
        plt.subplot(2, 1, 2)
        bars = plt.bar(range(len(student_avg)), student_avg.values, color='lightcoral', alpha=0.7)
        plt.title("학생별 평균 점수", fontsize=14, fontweight='bold')
        plt.ylabel("평균 점수")
        plt.xlabel("학생")
        plt.xticks(range(len(student_avg)), student_avg.index, rotation=45)
        plt.ylim(0, 100)
        
        # 막대 위에 점수 표시
        for bar, value in zip(bars, student_avg.values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                    f'{value:.1f}', ha='center', va='bottom')
    
    elif chart_type == "scatter":
        # 수학 vs 영어 산점도
        math_scores = df_scores[df_scores["과목"] == "수학"]["점수"].values
        english_scores = df_scores[df_scores["과목"] == "영어"]["점수"].values
        
        plt.scatter(math_scores, english_scores, alpha=0.7, s=100, color='purple')
        plt.title("수학 vs 영어 점수 상관관계", fontsize=14, fontweight='bold')
        plt.xlabel("수학 점수")
        plt.ylabel("영어 점수")
        plt.grid(True, alpha=0.3)
        
        # 대각선 그리기 (완벽한 상관관계 표시)
        plt.plot([60, 100], [60, 100], 'r--', alpha=0.5, label='완벽한 상관관계')
        plt.legend()
    
    elif chart_type == "heatmap":
        # 학생-과목 성적 히트맵
        pivot_data = df_scores.pivot(index="학생명", columns="과목", values="점수")
        
        sns.heatmap(pivot_data, annot=True, cmap='RdYlBu_r', center=80, 
                   square=True, linewidths=0.5, fmt='d')
        plt.title("학생별 과목별 성적 히트맵", fontsize=14, fontweight='bold')
        plt.ylabel("학생")
        plt.xlabel("과목")
    
    else:
        return f"지원하지 않는 차트 타입입니다: {chart_type}"
    
    plt.tight_layout()
    plt.show()
    
    return f"{chart_type} 차트가 생성되었습니다."

@tool
def generate_report(student_name: Annotated[str, "학생 이름"]) -> str:
    """특정 학생의 종합 성적표를 생성합니다."""
    
    # 학생 기본 정보
    student_info = get_student_scores(student_name)
    if "오류" in student_info:
        return student_info["오류"]
    
    # 전체 순위에서 학생 위치 찾기
    ranking = get_class_ranking()
    student_rank = None
    for rank_info in ranking:
        if rank_info["학생명"] == student_name:
            student_rank = rank_info["순위"]
            break
    
    # 각 과목별 상대적 위치 계산
    subject_analysis = {}
    for subject in ["수학", "영어", "과학", "국어", "사회"]:
        subject_stats = get_subject_statistics(subject)
        student_score = student_info["과목별_점수"][subject]
        
        # 상대적 위치 계산 (평균 대비)
        if student_score >= subject_stats["평균"]:
            performance = "우수"
        elif student_score >= subject_stats["평균"] - 5:
            performance = "보통"
        else:
            performance = "개선필요"
        
        subject_analysis[subject] = {
            "점수": student_score,
            "과목_평균": subject_stats["평균"],
            "평가": performance
        }
    
    # 보고서 생성
    report = f"""
{'='*50}
📋 {student_name} 학생 종합 성적표
{'='*50}

📊 기본 정보:
- 총점: {student_info['총점']}점
- 평균: {student_info['평균']}점
- 전체 순위: {student_rank}위 (전체 {len(ranking)}명 중)

📈 과목별 상세 분석:
"""
    
    for subject, analysis in subject_analysis.items():
        gap = analysis["점수"] - analysis["과목_평균"]
        report += f"- {subject}: {analysis['점수']}점 (평균 대비 {gap:+.1f}점) - {analysis['평가']}\n"
    
    # 종합 평가
    if student_info["평균"] >= 90:
        overall = "매우 우수"
    elif student_info["평균"] >= 80:
        overall = "우수"
    elif student_info["평균"] >= 70:
        overall = "보통"
    else:
        overall = "개선 필요"
    
    report += f"\n🎯 종합 평가: {overall}\n"
    report += f"📅 생성일: {datetime.now().strftime('%Y년 %m월 %d일')}\n"
    report += "="*50
    
    print(f"{student_name} 학생 종합 성적표 생성 완료")
    return report

# ============================================================================
# 3단계: 성적 분석 Agent 생성
# ============================================================================

def create_grade_analysis_agent():
    """성적 분석 Agent를 생성합니다."""
    
    # LLM 설정
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
    )
    
    # Tool 목록
    tools = [
        get_student_scores,
        get_subject_statistics, 
        get_class_ranking,
        create_score_visualization,
        generate_report
    ]
    
    # Agent 프롬프트
    prompt = ChatPromptTemplate.from_messages([
        ("system", """
당신은 전문적인 교육 데이터 분석가입니다.
학생들의 성적 데이터를 분석하고 인사이트를 제공하는 것이 목표입니다.

사용 가능한 도구들:
- get_student_scores: 학생별 성적 조회
- get_subject_statistics: 과목별 통계 분석
- get_class_ranking: 전체 순위 계산
- create_score_visualization: 차트 생성 (bar, scatter, heatmap)
- generate_report: 종합 성적표 생성

분석 가이드라인:
1. 사용자의 질문을 정확히 파악하세요
2. 필요한 데이터를 단계별로 수집하세요
3. 수치만 제공하지 말고 의미 있는 해석을 포함하세요
4. 시각화가 도움이 될 때는 적극 활용하세요
5. 교육적 관점에서 조언을 제공하세요

항상 친절하고 전문적인 톤으로 답변해주세요.
        """),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}")
    ])
    
    # Agent 생성
    agent = create_tool_calling_agent(llm, tools, prompt)
    
    # AgentExecutor 생성
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        max_iterations=10,
        handle_parsing_errors=True,
    )
    
    return agent_executor

# ============================================================================
# 4단계: Agent 데모 실행
# ============================================================================

def demo_grade_analysis_agent():
    """성적 분석 Agent 데모를 실행합니다."""
    
    print("🎓 학생 성적 분석 Agent")
    print("="*50)
    print("샘플 데이터로 8명의 학생, 5개 과목 성적이 준비되어 있습니다.")
    print("학생: 김철수, 이영희, 박민수, 최지은, 정다영, 윤석민, 한소영, 오준호")
    print("과목: 수학, 영어, 과학, 국어, 사회")
    print()
    
    # Agent 생성
    grade_agent = create_grade_analysis_agent()
    
    # 데모 시나리오들
    scenarios = [
        {
            "제목": "개별 학생 성적 조회",
            "질문": "김철수 학생의 성적을 자세히 알려주세요. 다른 학생들과 비교해서 어떤 수준인지도 분석해주세요."
        },
        {
            "제목": "과목별 분석",
            "질문": "수학 과목의 전체적인 성적 분포를 분석하고, 막대 그래프로 시각화해주세요."
        },
        {
            "제목": "종합 순위 및 성적표",
            "질문": "전체 학급 순위를 보여주고, 1등 학생의 종합 성적표를 만들어주세요."
        },
        {
            "제목": "상관관계 분석",
            "질문": "수학과 영어 점수 간의 상관관계를 산점도로 분석해주세요. 어떤 패턴이 보이나요?"
        },
        {
            "제목": "히트맵 시각화",
            "질문": "모든 학생의 모든 과목 성적을 한눈에 볼 수 있는 히트맵을 만들어주세요."
        }
    ]
    
    print("다음 시나리오들을 차례로 실행합니다:")
    for i, scenario in enumerate(scenarios, 1):
        print(f"{i}. {scenario['제목']}")
    print()
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"\n{'='*70}")
        print(f"📊 시나리오 {i}: {scenario['제목']}")
        print(f"❓ 분석 요청: {scenario['질문']}")
        print("-" * 70)
        
        try:
            # Agent 실행
            result = grade_agent.invoke({"input": scenario["질문"]})
            
            print(f"\n🤖 Agent 분석 결과:")
            print(result['output'])
            
        except Exception as e:
            print(f"❌ 오류 발생: {e}")
        
        if i < len(scenarios):
            print(f"\n⏭️  다음 시나리오를 보려면 Enter를 누르세요...")
            input()

# ============================================================================
# 5단계: 학습 정리
# ============================================================================

def learning_summary():
    """전체 학습 내용을 정리합니다."""
    
    print("\n🎯 학습 완료! 전체 내용 정리")
    print("="*50)
    
    print("""
📚 배운 내용 요약:

1️⃣ Tool (도구)
   - @tool 데코레이터로 함수를 Tool로 변환
   - LLM이 사용할 수 있는 기능 단위
   - 예: 계산기, 검색기, 데이터 분석기

2️⃣ Tool Calling (도구 호출)
   - LLM이 필요에 따라 Tool을 선택하고 실행
   - 단발성 작업에 적합
   - bind_tools()로 LLM에 Tool 연결

3️⃣ Agent (에이전트)
   - 복잡한 문제를 단계별로 해결
   - 여러 Tool을 연속적으로 사용
   - ReAct 패턴: Reasoning → Action → Observation
   - AgentExecutor로 실행

4️⃣ 실제 응용
   - 데이터 분석 Agent 구현
   - 여러 Tool을 조합한 복합 기능
   - 시각화와 보고서 생성
   - 실용적인 비즈니스 로직

🚀 다음 단계:
- 더 복잡한 Agent 설계
- 외부 API 연동 Tool 개발
- 메모리 기능 추가
- 멀티 Agent 시스템 구축

축하합니다! Tool, Tool Calling, Agent의 핵심을 모두 이해했습니다! 🎉
    """)

# ============================================================================
# 실행 부분
# ============================================================================

if __name__ == "__main__":
    try:
        # 샘플 데이터 확인
        print("📊 샘플 데이터 미리보기:")
        print(df_scores.head(10))
        print(f"\n전체 데이터: {len(df_scores)}행")
        print("-" * 50)
        
        # 데모 실행
        demo_grade_analysis_agent()
        
        # 학습 정리
        learning_summary()
        
    except Exception as e:
        print(f"⚠️  오류 발생: {e}")
        print("OpenAI API 키가 설정되어 있는지 확인해주세요!")
        print("\n그래도 샘플 데이터는 확인할 수 있습니다:")
        print(df_scores.head())

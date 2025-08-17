# LangChain Tool, Tool Calling, Agent 튜토리얼

이 튜토리얼은 LangChain의 핵심 개념인 Tool, Tool Calling, Agent를 단계별로 학습할 수 있도록 구성되었습니다.

## 📚 학습 순서

### 1단계: [01_basic_tool.py](01_basic_tool.py)
**Tool의 기본 개념과 생성 방법**
- Tool이 무엇인지 이해
- `@tool` 데코레이터 사용법
- 함수를 Tool로 변환하는 방법
- Tool의 파라미터와 설명 추가

```python
@tool
def simple_calculator(a: int, b: int) -> int:
    """두 숫자를 더하는 간단한 계산기입니다."""
    return a + b
```

### 2단계: [02_tool_calling.py](02_tool_calling.py)
**LLM이 Tool을 호출하는 방법**
- Tool Calling의 개념과 작동 원리
- LLM에 Tool 연결하기 (`bind_tools`)
- LLM이 자동으로 Tool을 선택하고 실행하는 과정
- Tool 호출 결과 확인하기

```python
llm_with_tools = llm.bind_tools([calculator, weather_info])
response = llm_with_tools.invoke("15에 23을 곱해주세요")
```

### 3단계: [03_agent.py](03_agent.py)
**Agent의 개념과 복잡한 문제 해결**
- Agent vs Tool Calling의 차이점
- ReAct 패턴 (Reasoning → Action → Observation)
- 여러 Tool을 순차적으로 사용하는 방법
- AgentExecutor를 통한 Agent 실행

```python
agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
```

### 4단계: [04_practical_example.py](04_practical_example.py)
**실제 업무에 적용할 수 있는 종합 실습**
- 학생 성적 분석 Agent 구현
- 데이터 조회, 통계 분석, 시각화, 보고서 생성
- 복잡한 비즈니스 로직을 Agent로 구현
- 실무에서 활용 가능한 패턴 학습

## 🎯 학습 목표

이 튜토리얼을 완료하면 다음을 할 수 있게 됩니다:

1. **Tool 개발**: 비즈니스 로직을 Tool로 변환
2. **Tool Calling**: LLM이 필요에 따라 Tool을 사용하도록 설정
3. **Agent 구현**: 복잡한 문제를 단계별로 해결하는 Agent 개발
4. **실무 적용**: 데이터 분석, 업무 자동화 등에 Agent 활용

## 🚀 실행 방법

### 필수 패키지 설치
```bash
pip install langchain langchain-openai python-dotenv pandas matplotlib seaborn
```

### OpenAI API 키 설정
1. `.env` 파일 생성
2. `OPENAI_API_KEY=your_api_key_here` 추가

### 파일 실행
각 파일을 순서대로 실행:
```bash
python 01_basic_tool.py
python 02_tool_calling.py  # API 키 필요
python 03_agent.py         # API 키 필요
python 04_practical_example.py  # API 키 필요
```

## 📖 핵심 개념 요약

| 개념 | 설명 | 사용 시기 |
|------|------|-----------|
| **Tool** | LLM이 사용할 수 있는 기능 단위 | 특정 작업을 수행하는 함수가 필요할 때 |
| **Tool Calling** | LLM이 상황에 맞는 Tool을 선택해서 실행 | 단순한 Tool 사용이 필요할 때 |
| **Agent** | 복잡한 문제를 단계별로 해결하는 시스템 | 여러 Tool을 조합한 복잡한 업무가 필요할 때 |

## 💡 실무 활용 예시

- **데이터 분석**: SQL 쿼리, 통계 계산, 시각화 Tool들을 조합
- **고객 서비스**: 정보 조회, 문제 해결, 티켓 생성 Tool들을 순차 실행  
- **콘텐츠 생성**: 리서치, 글쓰기, 편집, 발행 Tool들을 연결
- **업무 자동화**: 이메일, 스케줄링, 보고서 생성 등의 Tool들을 조합

## 🔧 문제 해결

### API 키 오류
- `.env` 파일에 올바른 OpenAI API 키가 설정되어 있는지 확인
- 계정에 충분한 크레딧이 있는지 확인

### 패키지 오류
- 모든 필수 패키지가 설치되어 있는지 확인
- Python 3.8 이상 버전 사용 권장

### 실행 오류
- 각 파일의 주석을 참고해서 단계별로 이해
- 코드를 수정해서 자신만의 Tool과 Agent 만들어보기

---

이 튜토리얼을 통해 LangChain의 핵심 기능을 마스터하고, 실무에 바로 적용할 수 있는 Agent를 개발해보세요! 🎉

# LangGraph 기초 튜토리얼 🚀

랭체인은 사용해봤지만 LangGraph는 처음인 학생들을 위한 체계적인 학습 가이드입니다.

## 📚 튜토리얼 구성

### 1️⃣ Python 기본 문법 (`01_python_basics.py`)
LangGraph 사용에 필요한 Python 문법들을 학습합니다.

**주요 내용:**
- `TypedDict`: 타입이 명시된 딕셔너리
- `Literal`: 특정 값만 허용하는 타입
- `Union`: 여러 타입 중 하나를 허용
- `Optional`: None이 될 수 있는 타입
- LangGraph에서 자주 사용되는 패턴들

```python
# 실행 방법
python 01_python_basics.py
```

### 2️⃣ LangGraph 기본 개념 (`02_langgraph_basics.py`)
LangGraph의 핵심 개념과 랭체인과의 차이점을 이해합니다.

**주요 내용:**
- LangGraph란 무엇인가?
- State, Node, Edge의 개념
- 조건부 라우팅의 이해
- 실제 사용 시나리오들
- 워크플로우 시뮬레이션

```python
# 실행 방법
python 02_langgraph_basics.py
```

### 3️⃣ 간단한 그래프 만들기 (`03_simple_graph.py`)
실제 LangGraph를 사용해서 첫 번째 그래프를 구현합니다.

**주요 내용:**
- StateGraph 생성 및 설정
- 노드 함수 작성
- 엣지 연결 (일반 엣지 + 조건부 엣지)
- 그래프 컴파일 및 실행
- 대화형 시스템 구현

```python
# 실행 방법 (LangGraph 설치 필요)
pip install langgraph
python 03_simple_graph.py
```

### 4️⃣ 실용적인 멀티 도구 에이전트 (`04_practical_agent.py`)
실제로 사용할 수 있는 복합 에이전트를 구현합니다.

**주요 내용:**
- 다양한 도구 통합 (계산기, 날씨, 검색, 시간)
- 지능적 인텐트 분석
- 조건부 라우팅
- 완전한 대화 기록 관리
- 확장 가능한 구조 설계

```python
# 실행 방법
python 04_practical_agent.py
```

## 🛠️ 설치 및 준비사항

### 필수 설치
```bash
pip install langgraph
pip install typing-extensions
```

### 선택적 설치 (실제 LLM 연동 시)
```bash
pip install langchain
pip install openai
pip install python-dotenv
```

## 📖 학습 순서

1. **01_python_basics.py** → Python 기본 문법 익히기
2. **02_langgraph_basics.py** → 개념 이해하기
3. **03_simple_graph.py** → 실제 구현 경험하기
4. **04_practical_agent.py** → 실용적인 응용 만들기

## 🎯 학습 목표

이 튜토리얼을 완료하면 다음과 같은 능력을 갖게 됩니다:

- ✅ LangGraph의 핵심 개념 이해
- ✅ 기본적인 그래프 구조 설계 및 구현
- ✅ 조건부 라우팅을 활용한 복잡한 워크플로우 구성
- ✅ 멀티 도구 에이전트 개발
- ✅ 상태 관리 및 데이터 흐름 제어
- ✅ 실제 프로젝트에 LangGraph 적용

## 🔧 주요 구성 요소

### State (상태)
```python
class MyState(TypedDict):
    messages: List[Dict[str, str]]
    current_step: str
    result: Optional[str]
```

### Node (노드)
```python
def my_node(state: MyState) -> MyState:
    # 작업 수행
    return updated_state
```

### Edge (엣지)
```python
# 일반 엣지
workflow.add_edge("node1", "node2")

# 조건부 엣지
workflow.add_conditional_edges(
    "node1",
    decision_function,
    {"option1": "node2", "option2": "node3"}
)
```

## 🚀 다음 단계

튜토리얼을 완료한 후 도전할 수 있는 프로젝트들:

1. **📝 문서 작성 AI**: 초안 → 검토 → 수정 → 승인 플로우
2. **🛒 쇼핑 도우미**: 요구사항 분석 → 상품 검색 → 비교 → 추천
3. **🔧 문제 해결 봇**: 진단 → 해결책 제시 → 실행 → 검증
4. **📚 개인화 학습 시스템**: 레벨 테스트 → 맞춤 커리큘럼 → 진도 관리

## 🤝 추가 리소스

- [LangGraph 공식 문서](https://langchain-ai.github.io/langgraph/)
- [LangChain 튜토리얼](https://python.langchain.com/docs/get_started/introduction)
- [TypedDict 가이드](https://docs.python.org/3/library/typing.html#typing.TypedDict)

## 💡 팁

1. **작게 시작하기**: 복잡한 그래프보다는 간단한 구조부터 시작
2. **상태 설계**: State 구조를 명확히 정의하는 것이 중요
3. **디버깅**: `verbose=True` 옵션으로 실행 과정 관찰
4. **테스트**: 다양한 입력으로 충분히 테스트
5. **확장성**: 나중에 노드나 도구를 추가하기 쉽게 설계

---

📧 문의사항이나 개선 제안이 있으시면 언제든 말씀해주세요!

**Happy Learning! 🎉**

"""
Agent - 세 번째 단계
====================

이 파일에서는 Agent가 무엇인지, Tool Calling과 어떻게 다른지 배워봅니다.

Agent란?
--------
Agent는 복잡한 문제를 해결하기 위해 여러 Tool을 연속적으로 사용하는 
더 똑똑한 시스템입니다.

Tool Calling vs Agent:
- Tool Calling: LLM이 한 번에 Tool을 호출하고 끝
- Agent: 문제를 단계별로 나누어 여러 Tool을 순차적으로 사용

Agent의 특징:
1. 🧠 생각하기 (Reasoning): 문제를 분석하고 계획 수립
2. 🔧 행동하기 (Action): 적절한 Tool 선택하고 실행
3. 👀 관찰하기 (Observation): Tool 결과 확인
4. 🔄 반복하기: 목표 달성까지 2-3 과정 반복
"""

# 필요한 라이브러리 import
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate
from typing import Annotated
import random

# 환경변수 로드
from dotenv import load_dotenv
load_dotenv()

# ============================================================================
# 1단계: Agent용 Tool 준비하기
# ============================================================================

@tool
def search_product(query: Annotated[str, "검색할 상품명"]) -> list:
    """온라인 쇼핑몰에서 상품을 검색합니다."""
    
    # 가짜 상품 데이터베이스
    products = [
        {"이름": "무선 이어폰", "가격": 89000, "평점": 4.5, "재고": 15},
        {"이름": "블루투스 이어폰", "가격": 65000, "평점": 4.2, "재고": 8},
        {"이름": "노이즈캔슬링 헤드폰", "가격": 159000, "평점": 4.8, "재고": 3},
        {"이름": "게이밍 헤드셋", "가격": 95000, "평점": 4.3, "재고": 12},
        {"이름": "스마트폰", "가격": 899000, "평점": 4.6, "재고": 5},
        {"이름": "태블릿", "가격": 549000, "평점": 4.4, "재고": 7},
        {"이름": "노트북", "가격": 1299000, "평점": 4.7, "재고": 2},
    ]
    
    # 검색어와 관련된 상품 찾기
    results = []
    for product in products:
        if query.lower() in product["이름"].lower():
            results.append(product)
    
    print(f"'{query}' 검색 결과: {len(results)}개 상품 발견")
    return results

@tool
def check_inventory(product_name: Annotated[str, "상품명"]) -> dict:
    """특정 상품의 재고를 확인합니다."""
    
    # 가짜 재고 정보
    inventory = {
        "무선 이어폰": {"재고": 15, "입고예정": "2024-01-15"},
        "블루투스 이어폰": {"재고": 8, "입고예정": "2024-01-10"},
        "노이즈캔슬링 헤드폰": {"재고": 3, "입고예정": "2024-01-20"},
        "게이밍 헤드셋": {"재고": 12, "입고예정": "2024-01-12"},
        "스마트폰": {"재고": 5, "입고예정": "2024-01-18"},
    }
    
    result = inventory.get(product_name, {"재고": 0, "입고예정": "미정"})
    print(f"'{product_name}' 재고 확인: {result}")
    return result

@tool
def calculate_shipping_cost(product_price: Annotated[float, "상품 가격"], quantity: Annotated[int, "수량"]) -> dict:
    """배송비를 계산합니다."""
    
    total_price = product_price * quantity
    
    if total_price >= 50000:
        shipping_cost = 0
        shipping_type = "무료배송"
    else:
        shipping_cost = 3000
        shipping_type = "일반배송"
    
    result = {
        "상품금액": total_price,
        "배송비": shipping_cost,
        "총금액": total_price + shipping_cost,
        "배송타입": shipping_type
    }
    
    print(f"배송비 계산 결과: {result}")
    return result

@tool
def add_to_cart(product_name: Annotated[str, "상품명"], quantity: Annotated[int, "수량"]) -> str:
    """상품을 장바구니에 추가합니다."""
    
    cart_id = random.randint(10000, 99999)
    message = f"'{product_name}' {quantity}개가 장바구니(ID: {cart_id})에 추가되었습니다."
    print(message)
    return message

# ============================================================================
# 2단계: Agent 생성하기
# ============================================================================

def create_shopping_agent():
    """쇼핑 도우미 Agent를 생성합니다."""
    
    # LLM 설정
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
    )
    
    # Tool 목록
    tools = [search_product, check_inventory, calculate_shipping_cost, add_to_cart]
    
    # Agent가 사용할 프롬프트 템플릿
    prompt = ChatPromptTemplate.from_messages([
        ("system", """
당신은 친절한 온라인 쇼핑 도우미 Agent입니다.
고객의 쇼핑을 도와주는 것이 목표입니다.

다음 도구들을 사용할 수 있습니다:
- search_product: 상품 검색
- check_inventory: 재고 확인  
- calculate_shipping_cost: 배송비 계산
- add_to_cart: 장바구니 추가

고객의 요청을 단계별로 처리해주세요:
1. 먼저 상품을 검색합니다
2. 재고를 확인합니다
3. 필요하면 배송비를 계산합니다
4. 고객이 원하면 장바구니에 추가합니다

항상 친절하고 도움이 되는 답변을 해주세요.
        """),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}")
    ])
    
    # Agent 생성
    agent = create_tool_calling_agent(llm, tools, prompt)
    
    # AgentExecutor 생성 (Agent를 실행하는 엔진)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,  # Agent의 사고 과정을 보여줌
        max_iterations=10,  # 최대 반복 횟수
        handle_parsing_errors=True,  # 오류 처리
    )
    
    return agent_executor

# ============================================================================
# 3단계: Agent 사용해보기
# ============================================================================

def demo_agent():
    """Agent 데모를 실행합니다."""
    
    print("🛒 쇼핑 도우미 Agent 데모")
    print("="*50)
    
    # Agent 생성
    shopping_agent = create_shopping_agent()
    
    # 테스트 시나리오들
    scenarios = [
        {
            "제목": "시나리오 1: 간단한 상품 검색",
            "질문": "이어폰을 찾고 있어요. 어떤 제품들이 있나요?"
        },
        {
            "제목": "시나리오 2: 구체적인 구매 상담",
            "질문": "무선 이어폰 2개를 사고 싶은데, 재고가 있는지 확인하고 총 금액이 얼마인지 알려주세요."
        },
        {
            "제목": "시나리오 3: 복합적인 쇼핑 요청",
            "질문": "게이밍 헤드셋을 3개 주문하려고 합니다. 재고 확인하고, 배송비 포함 총액 계산해서 장바구니에 넣어주세요."
        }
    ]
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"\n{'='*60}")
        print(f"📋 {scenario['제목']}")
        print(f"❓ 고객 질문: {scenario['질문']}")
        print("-" * 60)
        
        try:
            # Agent 실행
            result = shopping_agent.invoke({"input": scenario["질문"]})
            
            print(f"\n🤖 Agent 최종 답변:")
            print(f"{result['output']}")
            
        except Exception as e:
            print(f"❌ 오류 발생: {e}")
        
        if i < len(scenarios):
            print(f"\n다음 시나리오를 보려면 Enter를 누르세요...")
            input()

# ============================================================================
# 4단계: Agent vs Tool Calling 비교
# ============================================================================

def explain_agent_vs_tool_calling():
    """Agent와 Tool Calling의 차이점을 설명합니다."""
    
    print("\n🎯 Agent vs Tool Calling 비교")
    print("="*50)
    
    print("""
📞 Tool Calling:
- LLM이 한 번에 필요한 Tool을 호출
- 단순한 작업에 적합
- 예: "2 + 3을 계산해줘" → calculator 호출

🤖 Agent:
- 복잡한 문제를 단계별로 해결
- 여러 Tool을 순차적으로 사용
- 중간 결과를 바탕으로 다음 행동 결정
- 예: "이어폰 구매" → 검색 → 재고확인 → 가격계산 → 장바구니추가

🔄 Agent의 사고 과정 (ReAct 패턴):
1. Reasoning (추론): "고객이 이어폰을 원하니까 먼저 검색해야겠다"
2. Action (행동): search_product("이어폰") 실행
3. Observation (관찰): 검색 결과 확인
4. Reasoning (추론): "재고도 확인해야겠다"
5. Action (행동): check_inventory("무선 이어폰") 실행
... 반복 ...

이런 식으로 Agent는 스스로 생각하고 계획하며 문제를 해결합니다!
    """)

# ============================================================================
# 실행 부분
# ============================================================================

if __name__ == "__main__":
    try:
        demo_agent()
        explain_agent_vs_tool_calling()
        
        print("\n" + "="*50)
        print("🎓 학습 완료!")
        print("Tool → Tool Calling → Agent 순서로 배웠습니다.")
        print("다음 파일에서는 실제 데이터 분석 Agent를 만들어봅시다!")
        
    except Exception as e:
        print(f"⚠️  오류 발생: {e}")
        print("OpenAI API 키가 설정되어 있는지 확인해주세요!")

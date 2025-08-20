"""
네이버 뉴스 기반 한국어 RAG 튜토리얼
====================================

이 튜토리얼은 네이버 뉴스 기사를 활용하여 한국어 기반의 RAG(Retrieval-Augmented Generation) 
시스템을 구축하는 방법을 보여줍니다.

RAG 파이프라인:
1. 문서 로드(Document Loading): 네이버 뉴스 기사 웹 스크래핑
2. 텍스트 분할(Text Splitting): 문서를 청크로 분할
3. 임베딩(Embedding): 텍스트를 벡터로 변환
4. 벡터 저장소 생성(Vector Store): FAISS를 사용한 벡터 인덱싱
5. 검색기 생성(Retriever): 유사성 검색 엔진
6. 프롬프트 생성(Prompt): 한국어 질답을 위한 프롬프트
7. 언어모델(LLM): OpenAI GPT 모델
8. 체인 생성(Chain): 전체 RAG 파이프라인 연결

필요한 라이브러리:
- langchain
- langchain-community 
- langchain-openai
- langchain-text-splitters
- beautifulsoup4
- faiss-cpu
- python-dotenv
"""

import os
from dotenv import load_dotenv
import bs4
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings


def setup_environment():
    """환경 설정 및 API 키 로드"""
    print("🔧 환경 설정 중...")
    load_dotenv()
    
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY가 설정되지 않았습니다.")
        print("💡 .env 파일에 OPENAI_API_KEY=your_api_key 를 추가하세요.")
        return False
    
    print("✅ 환경 설정 완료!")
    return True


def load_naver_news(url):
    """
    네이버 뉴스 기사를 로드합니다.
    
    Args:
        url (str): 네이버 뉴스 기사 URL
        
    Returns:
        list: 로드된 문서 리스트
    """
    print(f"📰 네이버 뉴스 로딩 중: {url}")
    
    # 네이버 뉴스 기사의 본문과 제목을 추출하기 위한 CSS 선택자
    loader = WebBaseLoader(
        web_paths=(url,),
        bs_kwargs=dict(
            parse_only=bs4.SoupStrainer(
                "div",
                attrs={"class": ["newsct_article _article_body", "media_end_head_title"]},
            )
        ),
    )
    
    docs = loader.load()
    print(f"✅ 문서 로드 완료! 총 {len(docs)}개 문서")
    
    if docs:
        print(f"📄 문서 미리보기: {docs[0].page_content[:200]}...")
    
    return docs


def split_documents(docs, chunk_size=1000, chunk_overlap=100):
    """
    문서를 작은 청크로 분할합니다.
    
    Args:
        docs (list): 분할할 문서 리스트
        chunk_size (int): 각 청크의 최대 크기
        chunk_overlap (int): 청크 간 겹치는 문자 수
        
    Returns:
        list: 분할된 문서 청크 리스트
    """
    print(f"✂️ 문서 분할 중... (청크 크기: {chunk_size}, 겹침: {chunk_overlap})")
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap
    )
    
    splits = text_splitter.split_documents(docs)
    print(f"✅ 문서 분할 완료! 총 {len(splits)}개 청크 생성")
    
    return splits


def create_vectorstore(splits):
    """
    벡터 저장소를 생성합니다.
    
    Args:
        splits (list): 분할된 문서 청크 리스트
        
    Returns:
        FAISS: FAISS 벡터 저장소
    """
    print("🔍 벡터 저장소 생성 중... (임베딩 처리 중)")
    
    # OpenAI 임베딩을 사용하여 벡터 저장소 생성
    vectorstore = FAISS.from_documents(
        documents=splits, 
        embedding=OpenAIEmbeddings()
    )
    
    print("✅ 벡터 저장소 생성 완료!")
    return vectorstore


def create_rag_chain(vectorstore):
    """
    RAG 체인을 생성합니다.
    
    Args:
        vectorstore: 벡터 저장소
        
    Returns:
        Chain: RAG 체인
    """
    print("🔗 RAG 체인 생성 중...")
    
    # 검색기 생성
    retriever = vectorstore.as_retriever()
    
    # 한국어 질답을 위한 프롬프트 템플릿
    prompt = PromptTemplate.from_template(
        """당신은 질문-답변(Question-Answering)을 수행하는 친절한 AI 어시스턴트입니다. 
당신의 임무는 주어진 문맥(context)에서 주어진 질문(question)에 답하는 것입니다.

검색된 다음 문맥(context)을 사용하여 질문(question)에 답하세요. 
만약, 주어진 문맥(context)에서 답을 찾을 수 없다면, 
`주어진 정보에서 질문에 대한 정보를 찾을 수 없습니다`라고 답하세요.

한글로 답변해 주세요. 단, 기술적인 용어나 이름은 번역하지 않고 그대로 사용해 주세요.

#Question: 
{question} 

#Context: 
{context} 

#Answer:"""
    )
    
    # LLM 모델 설정
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
    
    # RAG 체인 생성
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    print("✅ RAG 체인 생성 완료!")
    return rag_chain


def ask_question(rag_chain, question):
    """
    RAG 시스템에 질문을 합니다.
    
    Args:
        rag_chain: RAG 체인
        question (str): 질문
        
    Returns:
        str: 답변
    """
    print(f"\n❓ 질문: {question}")
    print("🤔 답변 생성 중...")
    
    try:
        response = rag_chain.invoke(question)
        print(f"💬 답변: {response}")
        return response
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        return None


def stream_response(rag_chain, question):
    """
    스트리밍 방식으로 답변을 받습니다.
    
    Args:
        rag_chain: RAG 체인
        question (str): 질문
    """
    print(f"\n❓ 질문: {question}")
    print("💬 답변: ", end="", flush=True)
    
    try:
        for chunk in rag_chain.stream(question):
            print(chunk, end="", flush=True)
        print()  # 줄바꿈
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")


def main():
    """메인 함수"""
    print("🚀 네이버 뉴스 기반 한국어 RAG 튜토리얼 시작!")
    print("=" * 60)
    
    # 1. 환경 설정
    if not setup_environment():
        return
    
    # 2. 네이버 뉴스 URL (예시)
    # 실제 사용시에는 최신 뉴스 URL로 변경하세요
    news_url = "https://n.news.naver.com/article/437/0000378416"
    
    try:
        # 3. 문서 로드
        docs = load_naver_news(news_url)
        if not docs:
            print("❌ 문서를 로드할 수 없습니다.")
            return
        
        # 4. 문서 분할
        splits = split_documents(docs)
        
        # 5. 벡터 저장소 생성
        vectorstore = create_vectorstore(splits)
        
        # 6. RAG 체인 생성
        rag_chain = create_rag_chain(vectorstore)
        
        print("\n" + "=" * 60)
        print("🎉 RAG 시스템 구축 완료! 이제 질문을 해보세요.")
        print("=" * 60)
        
        # 7. 예제 질문들
        example_questions = [
            "이 기사의 주요 내용을 요약해주세요.",
            "기사에서 언급된 주요 인물이나 기관은 누구인가요?",
            "기사에서 다루는 주요 이슈는 무엇인가요?",
            "이 기사가 다루는 분야나 주제는 무엇인가요?"
        ]
        
        # 예제 질문 실행
        for question in example_questions:
            ask_question(rag_chain, question)
            print("-" * 40)
        
        # 8. 대화형 질문 모드
        print("\n🗣️ 대화형 모드를 시작합니다. 'quit' 또는 '종료'를 입력하면 종료됩니다.")
        print("-" * 60)
        
        while True:
            user_question = input("\n질문을 입력하세요: ").strip()
            
            if user_question.lower() in ['quit', '종료', 'exit', 'q']:
                print("👋 튜토리얼을 종료합니다. 감사합니다!")
                break
            
            if not user_question:
                print("💡 질문을 입력해주세요.")
                continue
            
            # 스트리밍 방식으로 답변 출력
            stream_response(rag_chain, user_question)
        
    except Exception as e:
        print(f"❌ 오류가 발생했습니다: {e}")
        print("💡 인터넷 연결, API 키, 또는 뉴스 URL을 확인해주세요.")


def demo_with_custom_url():
    """
    사용자 정의 URL로 데모를 실행하는 함수
    """
    print("🔧 사용자 정의 URL 데모")
    
    if not setup_environment():
        return
    
    print("📝 네이버 뉴스 URL을 입력하세요:")
    print("예시: https://n.news.naver.com/article/437/0000378416")
    
    url = input("URL: ").strip()
    
    if not url.startswith("https://n.news.naver.com/"):
        print("❌ 올바른 네이버 뉴스 URL을 입력해주세요.")
        return
    
    try:
        docs = load_naver_news(url)
        splits = split_documents(docs)
        vectorstore = create_vectorstore(splits)
        rag_chain = create_rag_chain(vectorstore)
        
        print("\n✅ 설정 완료! 질문을 입력하세요.")
        
        while True:
            question = input("\n질문 (종료: quit): ").strip()
            if question.lower() in ['quit', '종료']:
                break
            
            if question:
                stream_response(rag_chain, question)
    
    except Exception as e:
        print(f"❌ 오류: {e}")


if __name__ == "__main__":
    print("🌟 네이버 뉴스 RAG 튜토리얼")
    print("1. 기본 데모 실행")
    print("2. 사용자 정의 URL로 실행")
    
    choice = input("선택하세요 (1 또는 2): ").strip()
    
    if choice == "1":
        main()
    elif choice == "2":
        demo_with_custom_url()
    else:
        print("올바른 선택지를 입력해주세요.")

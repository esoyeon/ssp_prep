# 네이버 뉴스 기반 한국어 RAG 튜토리얼

LangChain을 활용하여 네이버 뉴스 기사를 기반으로 한국어 질답 시스템을 구축하는 튜토리얼입니다.

## 📋 목차

- [개요](#개요)
- [설치 방법](#설치-방법)
- [사용 방법](#사용-방법)
- [RAG 파이프라인](#rag-파이프라인)
- [예제](#예제)
- [주요 기능](#주요-기능)
- [문제 해결](#문제-해결)

## 🎯 개요

이 튜토리얼은 다음과 같은 기능을 제공합니다:

- **네이버 뉴스 스크래핑**: 실시간 뉴스 기사 내용 추출
- **한국어 텍스트 처리**: 한국어에 최적화된 텍스트 분할 및 임베딩
- **벡터 검색**: FAISS를 활용한 고속 유사성 검색
- **한국어 질답**: GPT 모델을 활용한 자연스러운 한국어 답변 생성
- **대화형 인터페이스**: 실시간 질의응답 시스템

## 🚀 설치 방법

### 1. 저장소 클론 또는 파일 다운로드

```bash
# 필요한 파일들을 다운로드 받거나 클론합니다
git clone <repository-url>
cd langchain_basic
```

### 2. 가상환경 생성 (권장)

```bash
# Python 가상환경 생성
python -m venv venv

# 가상환경 활성화
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

### 3. 의존성 설치

```bash
pip install -r requirements.txt
```

### 4. 환경 변수 설정

`.env` 파일을 생성하고 OpenAI API 키를 설정합니다:

```bash
# .env 파일 생성
touch .env
```

`.env` 파일 내용:
```
OPENAI_API_KEY=your_openai_api_key_here
```

> 🔑 OpenAI API 키는 [OpenAI 플랫폼](https://platform.openai.com/api-keys)에서 발급받을 수 있습니다.

## 📖 사용 방법

### 기본 실행

```bash
python naver_news_rag_tutorial.py
```

실행 후 다음 옵션을 선택할 수 있습니다:
- `1`: 기본 데모 실행 (예시 뉴스 기사 사용)
- `2`: 사용자 정의 URL로 실행

### 사용자 정의 URL 사용

```python
from naver_news_rag_tutorial import demo_with_custom_url

# 원하는 네이버 뉴스 URL로 실행
demo_with_custom_url()
```

## 🔄 RAG 파이프라인

이 튜토리얼은 다음 8단계 RAG 파이프라인을 구현합니다:

### 1. 문서 로드 (Document Loading)
```python
loader = WebBaseLoader(
    web_paths=(url,),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            "div",
            attrs={"class": ["newsct_article _article_body", "media_end_head_title"]},
        )
    ),
)
```

### 2. 텍스트 분할 (Text Splitting)
```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, 
    chunk_overlap=100
)
```

### 3. 임베딩 (Embedding)
```python
embeddings = OpenAIEmbeddings()
```

### 4. 벡터 저장소 (Vector Store)
```python
vectorstore = FAISS.from_documents(
    documents=splits, 
    embedding=embeddings
)
```

### 5. 검색기 (Retriever)
```python
retriever = vectorstore.as_retriever()
```

### 6. 프롬프트 (Prompt)
```python
prompt = PromptTemplate.from_template("""
당신은 질문-답변을 수행하는 친절한 AI 어시스턴트입니다.
주어진 문맥에서 질문에 답하세요.
...
""")
```

### 7. 언어모델 (LLM)
```python
llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
```

### 8. 체인 (Chain)
```python
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
```

## 💡 예제

### 기본 질문 예시

```python
# 예제 질문들
questions = [
    "이 기사의 주요 내용을 요약해주세요.",
    "기사에서 언급된 주요 인물이나 기관은 누구인가요?",
    "기사에서 다루는 주요 이슈는 무엇인가요?",
    "이 기사가 다루는 분야나 주제는 무엇인가요?"
]

# 질문 실행
for question in questions:
    response = rag_chain.invoke(question)
    print(f"Q: {question}")
    print(f"A: {response}")
```

### 스트리밍 응답

```python
# 실시간 스트리밍 응답
for chunk in rag_chain.stream("질문을 입력하세요"):
    print(chunk, end="", flush=True)
```

## ⭐ 주요 기능

### 1. 네이버 뉴스 자동 파싱
- 뉴스 제목과 본문 자동 추출
- 광고 및 불필요한 요소 제거
- 다양한 뉴스 카테고리 지원

### 2. 한국어 최적화
- 한국어 전용 프롬프트 템플릿
- 자연스러운 한국어 답변 생성
- 기술 용어 및 고유명사 보존

### 3. 대화형 인터페이스
- 실시간 질의응답
- 스트리밍 응답 지원
- 사용자 친화적 인터페이스

### 4. 확장 가능성
- 다양한 뉴스 소스 추가 가능
- 커스텀 프롬프트 템플릿 지원
- 다른 LLM 모델 연동 가능

## 🛠️ 문제 해결

### 일반적인 문제들

#### 1. OpenAI API 키 오류
```
❌ OPENAI_API_KEY가 설정되지 않았습니다.
```
**해결방법**: `.env` 파일에 올바른 API 키를 설정하세요.

#### 2. 네이버 뉴스 로딩 실패
```
❌ 문서를 로드할 수 없습니다.
```
**해결방법**: 
- 인터넷 연결 확인
- 유효한 네이버 뉴스 URL 확인
- 뉴스 기사가 삭제되지 않았는지 확인

#### 3. 의존성 설치 오류
```
❌ 라이브러리 설치 실패
```
**해결방법**:
```bash
# 개별 설치
pip install langchain langchain-community langchain-openai
pip install beautifulsoup4 faiss-cpu python-dotenv

# 또는 업그레이드
pip install --upgrade -r requirements.txt
```

#### 4. FAISS 설치 문제
```
❌ faiss-cpu 설치 실패
```
**해결방법**:
```bash
# Apple Silicon Mac의 경우
conda install faiss-cpu -c conda-forge

# 또는 GPU 버전 (CUDA 필요)
pip install faiss-gpu
```

### 성능 최적화

#### 1. 청크 크기 조정
```python
# 더 작은 청크 (정확도 높음, 속도 느림)
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500, 
    chunk_overlap=50
)

# 더 큰 청크 (속도 빠름, 정확도 낮음)
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500, 
    chunk_overlap=150
)
```

#### 2. 검색 결과 수 조정
```python
# 더 많은 문서 검색 (정확도 높음, 비용 증가)
retriever = vectorstore.as_retriever(search_kwargs={"k": 6})

# 적은 문서 검색 (비용 절약, 정확도 낮음)  
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
```

## 📚 추가 학습 자료

- [LangChain 공식 문서](https://python.langchain.com/)
- [OpenAI API 문서](https://platform.openai.com/docs)
- [FAISS 라이브러리](https://github.com/facebookresearch/faiss)
- [Beautiful Soup 문서](https://www.crummy.com/software/BeautifulSoup/bs4/doc/)

## 🤝 기여하기

이 튜토리얼에 기여하고 싶으시다면:

1. 이슈 리포트
2. 기능 개선 제안
3. 버그 수정
4. 문서 개선

## 📄 라이선스

이 프로젝트는 MIT 라이선스를 따릅니다.

---

**Happy Coding! 🎉**

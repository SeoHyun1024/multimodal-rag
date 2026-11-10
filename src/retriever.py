"""②③ 다중 벡터 검색기 + 멀티모달 최종 체인.

요약문은 Chroma(벡터 DB)에, 원본(텍스트 청크·표·이미지 Base64)은 InMemoryStore에
같은 doc_id로 저장합니다. 질문이 오면 요약으로 검색하고, doc_id로 원본을 꺼내
멀티모달 LLM에 넘겨 답을 생성합니다.

입력: data/processed_assignment/extracted.pkl, summaries.pkl
실행: python src/retriever.py   (먼저 src/extract.py → src/summarize.py 실행)
"""

import os
import pickle
import re
import time
import uuid
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_classic.retrievers.multi_vector import MultiVectorRetriever
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.stores import InMemoryStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed_assignment"
EMBEDDING_MODEL = "text-embedding-3-small"
ID_KEY = "doc_id"
BASE64_PATTERN = re.compile(r"^[A-Za-z0-9+/]{100,}={0,2}$")

INSTRUCTION = (
    "자료를 근거로 질문에 한국어로 답하세요\n"
    "그래프로 나타난 수치는 이미지를 분석하여 대략적인 값을 요약에 포함하세요\n"
    "자료에 없는 내용은 모른다고 답하세요.\n\n"
)

# 표·그래프에만 있는 수치를 묻는 질문 5개
QUERIES = [
    "2025년 9월 30일 기준 현금 및 현금성 자산과 총자산은 각각 얼마인가요?",           # 재무상태표
    "2025년 4분기 매출 가이던스와 조정 EBITDA 가이던스는 각각 얼마인가요?",          # 가이던스 표
    "2025년 3분기 자유현금흐름과 자유현금흐름 마진은 전년 동기 대비 어떻게 변했나요?",   # 현금흐름 표
    "2025년 3분기 DAU, MAU, 유료 구독자 수와 각각의 전년 대비 성장률은 얼마인가요?",   # 사용자 지표 표·그래프
    "2025년 3분기 GAAP 연구개발비와 Non-GAAP 연구개발비는 각각 얼마인가요?",          # R&D 조정표
]


def load_api_key():
    load_dotenv(PROJECT_ROOT / ".env")
    if not os.getenv("OPENAI_API_KEY", "").strip():
        raise ValueError("OPENAI_API_KEY 가 없습니다. .env 파일을 확인해 주세요.")


def load_processed():
    with open(PROCESSED_DIR / "extracted.pkl", "rb") as f:
        extracted = pickle.load(f)
    with open(PROCESSED_DIR / "summaries.pkl", "rb") as f:
        summaries = pickle.load(f)
    return extracted, summaries


def build_retriever(collection_name="multimodal_rag"):
    """요약용 벡터 DB(색인 카드 서랍)와 원본 저장소(서가)를 doc_id로 연결한 검색기."""
    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=OpenAIEmbeddings(model=EMBEDDING_MODEL),
    )
    vectorstore.reset_collection()  # 같은 프로세스에서 다시 만들 때 중복 저장 방지
    return MultiVectorRetriever(vectorstore=vectorstore, docstore=InMemoryStore(), id_key=ID_KEY)


def add_documents(retriever, doc_summaries, doc_contents):
    """요약본은 벡터 DB에, 원본은 docstore에 '같은 id'로 저장합니다."""
    doc_ids = [str(uuid.uuid4()) for _ in doc_contents]
    summary_docs = [
        Document(page_content=s, metadata={ID_KEY: doc_ids[i]})
        for i, s in enumerate(doc_summaries)
    ]
    retriever.vectorstore.add_documents(summary_docs)
    retriever.docstore.mset(list(zip(doc_ids, doc_contents)))


def split_image_text_types(docs):
    """검색 결과를 Base64 이미지와 텍스트로 나눕니다."""
    images, texts = [], []
    for doc in docs:
        content = doc.page_content if isinstance(doc, Document) else doc
        (images if BASE64_PATTERN.match(content) else texts).append(content)
    return {"images": images, "texts": texts}


def img_prompt_func(data_dict):
    """검색된 원본을 멀티모달 LLM이 읽을 메시지로 조립합니다. 이미지를 먼저 넣습니다."""
    messages = [
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image}"}}
        for image in data_dict["context"]["images"]
    ]
    formatted_texts = "\n".join(data_dict["context"]["texts"])
    messages.append(
        {
            "type": "text",
            "text": INSTRUCTION + f"질문: {data_dict['question']}\n\n참고 텍스트 :\n {formatted_texts}",
        }
    )
    return [HumanMessage(content=messages)]


def build_multimodal_chain(retriever, model):
    return (
        {
            "context": retriever | RunnableLambda(split_image_text_types),
            "question": RunnablePassthrough(),
        }
        | RunnableLambda(img_prompt_func)
        | model
        | StrOutputParser()
    )


def build_text_only_chain(texts, model):
    """비교용: 본문 텍스트만 검색하는 일반 RAG. 모델·지시문은 멀티모달 체인과 같습니다."""
    vectorstore = Chroma(
        collection_name="text_only_rag",
        embedding_function=OpenAIEmbeddings(model=EMBEDDING_MODEL),
    )
    vectorstore.reset_collection()
    vectorstore.add_texts(texts)
    prompt = ChatPromptTemplate.from_template(INSTRUCTION + "질문: {question}\n\n참고 텍스트 :\n {context}")
    return (
        {
            "context": vectorstore.as_retriever(search_kwargs={"k": 4})
            | RunnableLambda(lambda docs: "\n".join(d.page_content for d in docs)),
            "question": RunnablePassthrough(),
        }
        | prompt
        | model
        | StrOutputParser()
    )


def main():
    load_api_key()
    extracted, summaries = load_processed()
    texts_2k_token, tables = extracted["texts_2k_token"], extracted["tables"]
    print(
        f"복원 완료: 텍스트 {len(texts_2k_token)} / 표 {len(tables)} / "
        f"이미지 {len(summaries['img_base64_list'])}"
    )

    retriever = build_retriever()
    pairs = [
        (summaries["text_summaries"], texts_2k_token),
        (summaries["table_summaries"], tables),
        (summaries["image_summaries"], summaries["img_base64_list"]),
    ]
    for doc_summaries, doc_contents in pairs:
        if doc_summaries:
            add_documents(retriever, doc_summaries, doc_contents)
    print("색인 완료:", len(retriever.vectorstore.get()["ids"]), "건")

    model = ChatOpenAI(temperature=0, model="gpt-4o-mini", max_tokens=1024)
    chain = build_multimodal_chain(retriever, model)

    print("=" * 60)
    print("질의 5건")
    print("=" * 60)
    for i, q in enumerate(QUERIES, 1):
        start = time.time()
        kinds = split_image_text_types(retriever.invoke(q))
        answer = chain.invoke(q)
        print(f"[Q{i}] {q}")
        print(f"  검색 원본: 텍스트/표 {len(kinds['texts'])}건, 이미지 {len(kinds['images'])}건 | {time.time() - start:.1f}초")
        print("-" * 60)
        print(answer)
        print("=" * 60)

    text_only_chain = build_text_only_chain(texts_2k_token, model)
    query = QUERIES[0]
    print()
    print("=" * 60)
    print("Q:", query)
    print("=" * 60)
    print("[BEFORE] 텍스트 전용 RAG")
    print("-" * 60)
    print(text_only_chain.invoke(query))
    print()
    print("[AFTER] 멀티모달 RAG")
    print("-" * 60)
    print(chain.invoke(query))
    print("=" * 60)


if __name__ == "__main__":
    main()

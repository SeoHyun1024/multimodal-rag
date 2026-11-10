"""② 요약 생성: 텍스트·표·이미지마다 검색용 요약을 만듭니다.

입력: data/processed_assignment/extracted.pkl, data/figures_assignment/*.jpg
산출물: data/processed_assignment/summaries.pkl
        (text_summaries, table_summaries, image_summaries, img_base64_list)

실행: python src/summarize.py   (먼저 src/extract.py 실행)
"""

import base64
import os
import pickle
import time
from pathlib import Path

import tiktoken
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from openai import AuthenticationError, OpenAI

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIGURES_DIR = PROJECT_ROOT / "data" / "figures_assignment"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed_assignment"
MODEL_NAME = "gpt-4o-mini"

# 질문을 한국어로 하므로, 요약도 한국어로 만들어야 임베딩 공간에서 질문과 가깝게 잡힙니다.
TEXT_PROMPT_KOR = """당신은 표와 텍스트를 요약하여 검색에 활용할 수 있도록 돕는 도우미입니다.
이 요약본들은 임베딩되어 원본 텍스트나 표 요소를 검색하는 데 사용될 것입니다.
주어진 표나 텍스트의 내용을 검색에 최적화된 간결한 요약으로 작성해주세요.
요약할 표 또는 텍스트: {element}"""

# 이미지 설명은 영어 지시를 더 잘 따르므로, 지시는 영어로 하고 출력만 한국어로 받습니다.
IMAGE_PROMPT = """You are an assistant tasked with summarizing images for retrieval.
These summaries will be embedded and used to retrieve the raw image.
Provide a concise summary of the image that is well optimized for retrieval.
The summary should be written in Korean (Hangul)."""


def load_api_key() -> str:
    """PROJECT_ROOT/.env 에서 키를 읽고 유효한지 확인합니다. 키 값은 출력하지 않습니다."""
    load_dotenv(PROJECT_ROOT / ".env")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY 가 없습니다. .env 파일을 확인해 주세요.")
    try:
        OpenAI(api_key=api_key).models.list()
    except AuthenticationError as e:
        raise ValueError("OPENAI_API_KEY 가 유효하지 않습니다. .env 파일을 확인해 주세요.") from e
    print("API 키 로드·검증 완료")
    return api_key


def generate_text_summaries(texts, tables, summarize_texts=False):
    """텍스트와 표를 검색용 요약으로 만듭니다."""
    prompt = ChatPromptTemplate.from_template(TEXT_PROMPT_KOR)
    model = ChatOpenAI(temperature=0, model=MODEL_NAME)
    summarize_chain = {"element": lambda x: x} | prompt | model | StrOutputParser()

    text_summaries, table_summaries = [], []
    if texts and summarize_texts:
        text_summaries = summarize_chain.batch(texts, {"max_concurrency": 5})
    elif texts:
        text_summaries = texts
    if tables:
        table_summaries = summarize_chain.batch(tables, {"max_concurrency": 5})
    return text_summaries, table_summaries


def encode_image(image_path) -> str:
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def image_summarize(img_base64: str, prompt: str) -> str:
    """멀티모달 LLM으로 이미지 한 장을 요약합니다."""
    chat = ChatOpenAI(temperature=0, model=MODEL_NAME, max_tokens=1024)
    msg = chat.invoke(
        [
            HumanMessage(
                content=[
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}},
                ]
            )
        ]
    )
    return msg.content


def generate_img_summaries(path: Path):
    """폴더의 이미지마다 Base64 문자열과 요약을 만듭니다."""
    img_base64_list, image_summaries = [], []
    for img_file in sorted(os.listdir(path)):
        if img_file.lower().endswith((".jpg", ".jpeg", ".png")):
            base64_image = encode_image(Path(path) / img_file)
            img_base64_list.append(base64_image)
            image_summaries.append(image_summarize(base64_image, IMAGE_PROMPT))
    return img_base64_list, image_summaries


def main():
    load_api_key()

    with open(PROCESSED_DIR / "extracted.pkl", "rb") as f:
        data = pickle.load(f)
    tables, texts_2k_token = data["tables"], data["texts_2k_token"]
    print(f"복원: 텍스트 청크 {len(texts_2k_token)}개, 표 {len(tables)}개")

    start = time.time()
    text_summaries, table_summaries = generate_text_summaries(texts_2k_token, tables, summarize_texts=True)
    print(f"텍스트·표 요약 완료 — {time.time() - start:.1f}초, 텍스트 {len(text_summaries)}건 / 표 {len(table_summaries)}건")

    print("-" * 60)
    print("[텍스트 요약 예시]\n" + text_summaries[0])
    if table_summaries:
        print("-" * 60)
        print("[표 원본 앞 300자]\n" + tables[0][:300])
        print("[표 요약]\n" + table_summaries[0])
    print("-" * 60)

    start = time.time()
    img_base64_list, image_summaries = generate_img_summaries(FIGURES_DIR)
    print(f"이미지 요약 완료 — {time.time() - start:.1f}초, {len(image_summaries)}장")
    for name, summary in zip(sorted(os.listdir(FIGURES_DIR)), image_summaries):
        print(f"  {name}: {summary}")

    enc = tiktoken.encoding_for_model(MODEL_NAME)
    input_tokens = sum(len(enc.encode(t)) for t in texts_2k_token)
    output_tokens = sum(len(enc.encode(s)) for s in text_summaries)
    print(f"텍스트 요약 토큰(추정): 입력 {input_tokens:,} / 출력 {output_tokens:,} / 합계 {input_tokens + output_tokens:,}")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    with open(PROCESSED_DIR / "summaries.pkl", "wb") as f:
        pickle.dump(
            {
                "text_summaries": text_summaries,
                "table_summaries": table_summaries,
                "image_summaries": image_summaries,
                "img_base64_list": img_base64_list,
            },
            f,
        )
    print("저장 완료:", PROCESSED_DIR / "summaries.pkl")


if __name__ == "__main__":
    main()

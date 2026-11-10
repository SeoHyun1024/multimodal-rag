"""① PDF 파싱: 듀오링고 Q3 2025 주주서한에서 텍스트·표·이미지를 추출합니다.

산출물
- data/figures_assignment/*.jpg          : 추출된 이미지
- data/processed_assignment/extracted.pkl : texts, tables, texts_2k_token

실행: python src/extract.py
"""

import pickle
import time
from collections import Counter
from pathlib import Path

from langchain_text_splitters import CharacterTextSplitter
from unstructured.partition.pdf import partition_pdf

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PDF_PATH = PROJECT_ROOT / "data" / "duolingo_q3_2025.pdf"
FIGURES_DIR = PROJECT_ROOT / "data" / "figures_assignment"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed_assignment"


def extract_pdf_elements(pdf_path: Path, figures_dir: Path):
    """PDF에서 이미지·표·텍스트 블록을 추출합니다. 이미지는 figures_dir에 저장됩니다."""
    return partition_pdf(
        filename=str(pdf_path),
        strategy="hi_res",                 # 레이아웃 분석 기반 정밀 추출 (표·이미지 인식에 필요)
        extract_images_in_pdf=True,        # 이미지 추출
        infer_table_structure=True,        # 표 구조 추론
        chunking_strategy="by_title",      # 제목 기준으로 텍스트 분할
        max_characters=4000,               # 블록 최대 4000자
        new_after_n_chars=3800,            # 3800자 이후 새 블록 시작
        combine_text_under_n_chars=2000,   # 2000자 이하 조각은 결합
        extract_image_block_output_dir=str(figures_dir),
    )


def categorize_elements(raw_pdf_elements):
    """추출된 요소를 텍스트와 표로 분류합니다."""
    texts, tables = [], []
    for element in raw_pdf_elements:
        element_type = str(type(element))
        if "unstructured.documents.elements.Table" in element_type:
            tables.append(str(element))
        elif "unstructured.documents.elements.CompositeElement" in element_type:
            texts.append(str(element))
    return texts, tables


def split_texts(texts, chunk_size=2000, chunk_overlap=200):
    """텍스트 블록을 합친 뒤 토큰 기준 약 2천 토큰 단위로 다시 나눕니다."""
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.split_text(" ".join(texts))


def list_images(figures_dir: Path):
    return sorted(
        f for f in figures_dir.iterdir()
        if f.suffix.lower() in (".jpg", ".jpeg", ".png")
    )


def main():
    print("PDF :", PDF_PATH, "(존재)" if PDF_PATH.exists() else "(없음)")

    start = time.time()
    raw_pdf_elements = extract_pdf_elements(PDF_PATH, FIGURES_DIR)
    print(f"추출 완료 - {time.time() - start:.1f}초 소요, 전체 요소 {len(raw_pdf_elements)}개")

    type_counts = Counter(type(el).__name__ for el in raw_pdf_elements)
    for element_type, count in type_counts.most_common():
        print(f"  {element_type:22s}: {count}개")

    texts, tables = categorize_elements(raw_pdf_elements)
    print("텍스트 블록 수:", len(texts))
    print("표 블록 수    :", len(tables))

    image_files = list_images(FIGURES_DIR)
    print(f"추출된 이미지: {len(image_files)}개")
    for f in image_files:
        print(f"  {f.name}  ({f.stat().st_size // 1024} KB)")

    texts_2k_token = split_texts(texts)
    print(f"텍스트 재분할: {len(texts)}개 → {len(texts_2k_token)}개")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_DIR / "extracted.pkl", "wb") as f:
        pickle.dump({"texts": texts, "tables": tables, "texts_2k_token": texts_2k_token}, f)
    print("저장 완료:", OUTPUT_DIR / "extracted.pkl")


if __name__ == "__main__":
    main()

"""
loader.py

문서 로딩 및 청킹
강의 코드 참조:
- utils.py의 chunk_document (lines 37-56)
- utils.py의 build_text_splitter (lines 23-29)
"""

import os
from dataclasses import dataclass
from typing import List, Dict, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter


@dataclass
class Chunk:
    """
    문서 청크 데이터 클래스

    참고: utils.py의 Chunk (lines 31-34)
    """
    id: str
    text: str
    metadata: Dict[str, Any]


def build_text_splitter(chunk_size: int = 700, chunk_overlap: int = 120) -> RecursiveCharacterTextSplitter:
    """
    텍스트 분할기 생성

    참고: utils.py의 build_text_splitter (lines 23-29)
    """
    return RecursiveCharacterTextSplitter(
        separators=["\n\n", ". ", "! ", "? ", "\n", " "],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )


def chunk_document(doc_text: str, source: str, splitter=None) -> List[Chunk]:
    """
    문서를 청크 단위로 나누기

    영화 PDF의 경우 "N번째 영화"로 구분하여 영화 단위로 chunking
    일반 문서는 기존 방식대로 처리

    참고: utils.py의 chunk_document (lines 37-56)

    Args:
        doc_text: 문서 텍스트
        source: 문서 출처 (파일 경로)
        splitter: 텍스트 분할기 (None이면 기본 생성)

    Returns:
        Chunk 리스트
    """
    chunks = []

    # 영화 PDF인 경우: "N번째 영화"로 분할 (영화 단위 chunking)
    if "번째 영화" in doc_text:
        # "N번째 영화"로 split
        import re
        # 패턴: "1번째 영화", "2번째 영화" 등
        movie_parts = re.split(r'(\d+번째 영화)', doc_text)

        current_movie = ""
        movie_number = None

        for i, part in enumerate(movie_parts):
            if re.match(r'\d+번째 영화', part):
                # "N번째 영화" 발견
                if current_movie.strip() and movie_number is not None:
                    # 이전 영화 저장
                    chunks.append(
                        Chunk(
                            id=f"{os.path.basename(source)}::movie_{movie_number}",
                            text=current_movie.strip(),
                            metadata={
                                "source": source,
                                "chunk_id": movie_number,
                                "type": "movie"
                            }
                        )
                    )

                # 새 영화 시작
                movie_number = int(re.search(r'\d+', part).group())
                current_movie = ""
            else:
                # 영화 내용 추가
                current_movie += part

        # 마지막 영화 저장
        if current_movie.strip() and movie_number is not None:
            chunks.append(
                Chunk(
                    id=f"{os.path.basename(source)}::movie_{movie_number}",
                    text=current_movie.strip(),
                    metadata={
                        "source": source,
                        "chunk_id": movie_number,
                        "type": "movie"
                    }
                )
            )

    else:
        # 일반 문서: 기존 방식 (RecursiveCharacterTextSplitter 사용)
        if splitter is None:
            splitter = build_text_splitter()

        pieces = splitter.split_text(doc_text)

        for i, piece in enumerate(pieces):
            meta = {
                "source": source,
                "chunk_id": i,
            }
            chunks.append(
                Chunk(
                    id=f"{os.path.basename(source)}::chunk_{i}",
                    text=piece,
                    metadata=meta,
                )
            )

    return chunks


def load_text_file(file_path: str) -> str:
    """
    텍스트 파일 로드

    Args:
        file_path: 파일 경로

    Returns:
        파일 내용
    """
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def load_pdf_file(file_path: str) -> str:
    """
    PDF 파일 로드 (pypdf 사용)

    Args:
        file_path: PDF 파일 경로

    Returns:
        추출된 텍스트
    """
    try:
        from pypdf import PdfReader

        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text
    except ImportError:
        raise ImportError("pypdf가 설치되지 않았습니다. pip install pypdf를 실행하세요.")


def load_documents_from_directory(directory: str, file_extension: str = ".txt") -> List[Chunk]:
    """
    디렉토리에서 모든 문서 로드 및 청킹

    Args:
        directory: 문서가 있는 디렉토리
        file_extension: 파일 확장자 (.txt, .pdf 등)

    Returns:
        모든 문서의 Chunk 리스트
    """
    all_chunks = []
    splitter = build_text_splitter()

    if not os.path.exists(directory):
        print(f"Warning: Directory not found: {directory}")
        return all_chunks

    for filename in os.listdir(directory):
        if not filename.endswith(file_extension):
            continue

        file_path = os.path.join(directory, filename)

        try:
            # 파일 로드
            if file_extension == ".pdf":
                doc_text = load_pdf_file(file_path)
            else:
                doc_text = load_text_file(file_path)

            # 청킹
            chunks = chunk_document(doc_text, file_path, splitter)
            all_chunks.extend(chunks)

            print(f"Loaded {len(chunks)} chunks from {filename}")

        except Exception as e:
            print(f"Error loading {filename}: {e}")

    return all_chunks


# 테스트용
if __name__ == "__main__":
    # 예시: data/pdfs/ 디렉토리에서 PDF 로드
    chunks = load_documents_from_directory("data/pdfs", ".pdf")
    print(f"\nTotal chunks loaded: {len(chunks)}")

    if chunks:
        print(f"\nFirst chunk example:")
        print(f"ID: {chunks[0].id}")
        print(f"Text preview: {chunks[0].text[:200]}...")
        print(f"Metadata: {chunks[0].metadata}")

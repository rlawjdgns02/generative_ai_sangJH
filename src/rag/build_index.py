"""
build_index.py

data/ 폴더의 PDF 파일들을 ChromaDB에 색인
"""

import os
import sys
from dotenv import load_dotenv

import pathlib, sys
ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.tools.search_tools import initialize_rag_database

load_dotenv()


def main():
    """data/ 폴더의 영화 PDF 파일들을 색인"""

    print("=" * 60)
    print("영화 PDF 파일 색인 시작")
    print("=" * 60)

    # .env 확인
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ 오류: OPENAI_API_KEY가 설정되지 않았습니다.")
        print("   .env 파일을 확인하세요.")
        return

    # data/ 폴더 확인
    if not os.path.exists("data"):
        print("❌ 오류: data/ 폴더가 없습니다.")
        return

    # PDF 파일 확인
    pdf_files = [f for f in os.listdir("data") if f.endswith('.pdf')]

    if not pdf_files:
        print("❌ 오류: data/ 폴더에 PDF 파일이 없습니다.")
        return

    print(f"\n📄 발견된 PDF 파일: {len(pdf_files)}개")
    for pdf in sorted(pdf_files):
        print(f"   - {pdf}")

    # 색인 시작
    print("\n🔄 색인 시작...\n")

    result = initialize_rag_database(
        document_directory="data",
        file_extension=".pdf"
    )

    # 결과 출력
    print("\n" + "=" * 60)
    if result.get("skipped"):
        print("⏭️  색인 스킵")
        print(f"📊 {result['message']}")
        print(f"💡 {result.get('hint', '')}")
    elif result["ok"]:
        print("✅ 색인 완료!")
        print(f"📊 총 {result['count']}개 청크가 저장되었습니다.")
    else:
        print("❌ 색인 실패")
        print(f"   오류: {result.get('error', 'Unknown error')}")
    print("=" * 60)


if __name__ == "__main__":
    main()

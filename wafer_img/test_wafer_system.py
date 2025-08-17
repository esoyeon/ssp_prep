#!/usr/bin/env python3
"""
웨이퍼맵 유사도 검색 시스템 테스트 스크립트
더미 데이터와 실제 데이터를 모두 테스트할 수 있습니다.
"""

import os
import sys
from wafer_similarity_search import WaferSimilaritySearch
from wafer_dummy_generator import WaferDummyGenerator


def test_dummy_data():
    """더미 데이터 테스트"""
    print("=== 더미 데이터 테스트 ===")

    # 더미 데이터가 없으면 생성
    if not os.path.exists("wafer_dummy_data.pkl"):
        print("더미 데이터를 생성합니다...")
        generator = WaferDummyGenerator(wafer_size=64)
        generator.save_dummy_dataset(num_samples=100, filename="wafer_dummy_data.pkl")
        print("더미 데이터 생성 완료!")

    # 더미 데이터로 테스트
    searcher = WaferSimilaritySearch("wafer_dummy_data.pkl")

    if searcher.load_data():
        print("더미 데이터 로드 성공!")

        # 특징 추출 (자동으로 간단한 방법 선택)
        searcher.extract_features(method="auto")

        # 유사도 인덱스 구축
        searcher.build_similarity_index(method="cosine")

        # 검색 품질 평가
        searcher.evaluate_search_quality(num_queries=5)

        # 예시 검색
        query_idx = 0
        similar_images = searcher.search_similar_images(query_idx, top_k=3)

        print(f"\n이미지 {query_idx}와 유사한 이미지들:")
        for i, (idx, score) in enumerate(similar_images):
            failure_type = searcher.df.iloc[idx]["failureType"]
            print(
                f"  {i+1}. 이미지 {idx}: 유사도 {score:.3f}, 실패유형: {failure_type}"
            )

        # 특징 저장
        searcher.save_features("wafer_dummy_features.pkl")
        print("\n더미 데이터 테스트 완료!")
        return True
    else:
        print("더미 데이터 테스트 실패!")
        return False


def test_real_data():
    """실제 MIR-WM811K 데이터 테스트"""
    print("\n=== 실제 데이터 테스트 ===")

    data_path = "MIR-WM811K/Python/WM811K.pkl"

    if not os.path.exists(data_path):
        print(f"실제 데이터를 찾을 수 없습니다: {data_path}")
        print("MIR-WM811K 데이터셋을 다운로드해주세요.")
        return False

    # 실제 데이터로 테스트
    searcher = WaferSimilaritySearch(data_path)

    if searcher.load_data():
        print("실제 데이터 로드 성공!")

        # 특징 추출 (자동으로 앙상블 방법 선택)
        searcher.extract_features(method="auto")

        # 유사도 인덱스 구축
        searcher.build_similarity_index(method="cosine")

        # 예시 검색
        query_idx = 0
        similar_images = searcher.search_similar_images(query_idx, top_k=3)

        print(f"\n이미지 {query_idx}와 유사한 이미지들:")
        for i, (idx, score) in enumerate(similar_images):
            failure_type = searcher.df.iloc[idx]["failureType"]
            print(
                f"  {i+1}. 이미지 {idx}: 유사도 {score:.3f}, 실패유형: {failure_type}"
            )

        # 특징 저장
        searcher.save_features("wafer_real_features.pkl")
        print("\n실제 데이터 테스트 완료!")
        return True
    else:
        print("실제 데이터 테스트 실패!")
        return False


def main():
    """메인 테스트 함수"""
    print("웨이퍼맵 유사도 검색 시스템 테스트\n")

    # 더미 데이터 테스트
    dummy_success = test_dummy_data()

    # 실제 데이터 테스트
    real_success = test_real_data()

    # 결과 요약
    print("\n=== 테스트 결과 요약 ===")
    print(f"더미 데이터 테스트: {'성공' if dummy_success else '실패'}")
    print(f"실제 데이터 테스트: {'성공' if real_success else '실패'}")

    if dummy_success:
        print("\n더미 데이터로 빠른 테스트가 가능합니다!")
        print("python wafer_similarity_search.py")

    if real_success:
        print("\n실제 데이터로 정확한 분석이 가능합니다!")
        print("python wafer_similarity_search.py")

    if not dummy_success and not real_success:
        print("\n사용 가능한 데이터가 없습니다.")
        print("더미 데이터를 생성하려면: python wafer_dummy_generator.py")


if __name__ == "__main__":
    main()

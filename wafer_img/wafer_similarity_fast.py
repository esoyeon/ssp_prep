import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Tuple
import time


class WaferSimilarityFast:
    """
    빠른 웨이퍼맵 유사도 검색 시스템 (더미 데이터용)
    """

    def __init__(self, data_path: str = "wafer_dummy_data.pkl"):
        """
        초기화

        Args:
            data_path: 웨이퍼맵 데이터 파일 경로
        """
        self.data_path = data_path
        self.df = None
        self.feature_vectors = None
        self.similarity_matrix = None

    def load_data(self):
        """데이터 로드"""
        try:
            self.df = pd.read_pickle(self.data_path)
            print(f"데이터 로드 완료: {len(self.df)} 개의 웨이퍼맵")
            print(f"결함 유형: {self.df['failureType'].unique()}")
            return True
        except Exception as e:
            print(f"데이터 로드 실패: {e}")
            print("더미 데이터를 먼저 생성해주세요.")
            return False

    def extract_simple_features(self, wafer_map: np.ndarray) -> np.ndarray:
        """간단한 특징 추출 (빠른 처리용)"""
        features = []

        # 1. 기본 통계 특징
        features.extend(
            [
                wafer_map.mean(),
                wafer_map.std(),
                wafer_map.min(),
                wafer_map.max(),
                np.percentile(wafer_map, 25),
                np.percentile(wafer_map, 50),
                np.percentile(wafer_map, 75),
            ]
        )

        # 2. 결함 밀도 특징
        defect_threshold = np.percentile(wafer_map, 75)
        defect_mask = wafer_map > defect_threshold
        defect_density = defect_mask.sum() / defect_mask.size
        features.append(defect_density)

        # 3. 중심과 엣지 특징
        center_y, center_x = np.array(wafer_map.shape) // 2
        center_region = wafer_map[
            center_y - 5 : center_y + 5, center_x - 5 : center_x + 5
        ]
        edge_region = wafer_map[0:5, :].flatten()

        features.extend(
            [
                center_region.mean(),
                center_region.std(),
                edge_region.mean(),
                edge_region.std(),
            ]
        )

        # 4. 간단한 텍스처 특징
        # 수평/수직 방향의 변화량
        horizontal_diff = np.diff(wafer_map, axis=1).mean()
        vertical_diff = np.diff(wafer_map, axis=0).mean()
        features.extend([horizontal_diff, vertical_diff])

        return np.array(features)

    def extract_features(self):
        """모든 웨이퍼맵에서 특징 추출"""
        if self.df is None:
            print("먼저 데이터를 로드해주세요.")
            return

        print("간단한 특징을 추출하는 중...")
        start_time = time.time()

        features = []
        for idx, row in self.df.iterrows():
            wafer_map = row["waferMap"]
            feature_vector = self.extract_simple_features(wafer_map)
            features.append(feature_vector)

        self.feature_vectors = np.array(features)
        end_time = time.time()

        print(f"특징 추출 완료: {self.feature_vectors.shape}")
        print(f"소요 시간: {end_time - start_time:.2f}초")

    def build_similarity_index(self):
        """유사도 인덱스 구축"""
        if self.feature_vectors is None:
            print("먼저 특징을 추출해주세요.")
            return

        print("유사도 인덱스를 구축하는 중...")
        start_time = time.time()

        # 코사인 유사도 계산
        self.similarity_matrix = cosine_similarity(self.feature_vectors)

        end_time = time.time()
        print(f"유사도 인덱스 구축 완료")
        print(f"소요 시간: {end_time - start_time:.2f}초")

    def search_similar_images(
        self, query_idx: int, top_k: int = 10
    ) -> List[Tuple[int, float]]:
        """유사한 이미지 검색"""
        if self.similarity_matrix is None:
            print("먼저 유사도 인덱스를 구축해주세요.")
            return []

        if query_idx >= len(self.similarity_matrix):
            print(f"인덱스 {query_idx}가 범위를 벗어났습니다.")
            return []

        # 자기 자신을 제외한 유사도 점수
        similarities = self.similarity_matrix[query_idx]

        # 상위 k개 유사 이미지 찾기
        top_indices = np.argsort(similarities)[::-1][1 : top_k + 1]
        top_scores = similarities[top_indices]

        results = list(zip(top_indices, top_scores))
        return results

    def search_by_image(
        self, query_image: np.ndarray, top_k: int = 10
    ) -> List[Tuple[int, float]]:
        """새로운 이미지와 유사한 이미지 검색"""
        if self.feature_vectors is None:
            print("먼저 특징을 추출해주세요.")
            return []

        # 쿼리 이미지의 특징 추출
        query_features = self.extract_simple_features(query_image)

        # 모든 이미지와의 유사도 계산
        similarities = []
        for features in self.feature_vectors:
            sim = cosine_similarity([query_features], [features])[0][0]
            similarities.append(sim)

        # 상위 k개 유사 이미지 찾기
        similarities = np.array(similarities)
        top_indices = np.argsort(similarities)[::-1][:top_k]
        top_scores = similarities[top_indices]

        results = list(zip(top_indices, top_scores))
        return results

    def visualize_similar_images(self, query_idx: int, top_k: int = 5):
        """유사 이미지 시각화"""
        if self.df is None:
            print("먼저 데이터를 로드해주세요.")
            return

        # 유사 이미지 검색
        similar_images = self.search_similar_images(query_idx, top_k)

        if not similar_images:
            return

        # 시각화
        fig, axes = plt.subplots(1, top_k + 1, figsize=(15, 3))

        # 쿼리 이미지
        query_wafer = self.df.iloc[query_idx]["waferMap"]
        axes[0].imshow(query_wafer, cmap="viridis")
        axes[0].set_title(f'Query\nType: {self.df.iloc[query_idx]["failureType"]}')
        axes[0].axis("off")

        # 유사 이미지들
        for i, (idx, score) in enumerate(similar_images):
            wafer = self.df.iloc[idx]["waferMap"]
            axes[i + 1].imshow(wafer, cmap="viridis")
            axes[i + 1].set_title(
                f'Similar {i+1}\nScore: {score:.3f}\nType: {self.df.iloc[idx]["failureType"]}'
            )
            axes[i + 1].axis("off")

        plt.tight_layout()
        plt.show()

    def evaluate_search_quality(self, num_queries: int = 10):
        """검색 품질 평가"""
        if self.similarity_matrix is None:
            print("먼저 유사도 인덱스를 구축해주세요.")
            return

        print(f"검색 품질을 평가합니다 ({num_queries}개 쿼리)...")

        same_type_accuracy = []
        top_5_accuracy = []

        for _ in range(num_queries):
            query_idx = np.random.randint(0, len(self.df))
            query_type = self.df.iloc[query_idx]["failureType"]

            # 유사 이미지 검색
            similar_images = self.search_similar_images(query_idx, top_k=10)

            if similar_images:
                # 상위 5개 중 같은 타입의 비율
                top_5_types = [
                    self.df.iloc[idx]["failureType"] for idx, _ in similar_images[:5]
                ]
                top_5_accuracy.append(
                    sum(1 for t in top_5_types if t == query_type) / 5
                )

                # 전체 10개 중 같은 타입의 비율
                all_types = [
                    self.df.iloc[idx]["failureType"] for idx, _ in similar_images
                ]
                same_type_accuracy.append(
                    sum(1 for t in all_types if t == query_type) / 10
                )

        print(f"상위 5개 중 같은 타입 비율: {np.mean(top_5_accuracy):.3f}")
        print(f"전체 10개 중 같은 타입 비율: {np.mean(same_type_accuracy):.3f}")

    def get_failure_type_distribution(self):
        """실패 유형 분포 확인"""
        if self.df is None:
            print("먼저 데이터를 로드해주세요.")
            return

        failure_counts = self.df["failureType"].value_counts()

        plt.figure(figsize=(10, 6))
        failure_counts.plot(kind="bar")
        plt.title("Wafer Failure Type Distribution (Dummy Data)")
        plt.xlabel("Failure Type")
        plt.ylabel("Count")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

        return failure_counts


def main():
    """메인 실행 함수"""
    print("=== 빠른 웨이퍼맵 유사도 검색 시스템 ===\n")

    # 시스템 초기화
    searcher = WaferSimilarityFast()

    # 데이터 로드
    if not searcher.load_data():
        print("더미 데이터를 먼저 생성해주세요:")
        print("python wafer_dummy_generator.py")
        return

    # 실패 유형 분포 확인
    print("\n실패 유형 분포:")
    failure_dist = searcher.get_failure_type_distribution()
    print(failure_dist)

    # 특징 추출
    print("\n특징 추출 중...")
    searcher.extract_features()

    # 유사도 인덱스 구축
    print("\n유사도 인덱스 구축 중...")
    searcher.build_similarity_index()

    # 검색 품질 평가
    print("\n검색 품질 평가:")
    searcher.evaluate_search_quality(num_queries=20)

    # 예시 검색
    print("\n=== 유사도 검색 예시 ===")

    # 첫 번째 이미지로 유사도 검색
    query_idx = 0
    similar_images = searcher.search_similar_images(query_idx, top_k=5)

    print(f"\n이미지 {query_idx}와 유사한 이미지들:")
    for i, (idx, score) in enumerate(similar_images):
        failure_type = searcher.df.iloc[idx]["failureType"]
        print(f"  {i+1}. 이미지 {idx}: 유사도 {score:.3f}, 실패유형: {failure_type}")

    # 시각화
    print(f"\n이미지 {query_idx}와 유사한 이미지들을 시각화합니다...")
    searcher.visualize_similar_images(query_idx, top_k=5)

    print("\n=== 시스템 사용법 ===")
    print(
        "1. searcher.search_similar_images(idx, top_k) - 특정 이미지와 유사한 이미지 검색"
    )
    print(
        "2. searcher.search_by_image(new_image, top_k) - 새로운 이미지와 유사한 이미지 검색"
    )
    print("3. searcher.visualize_similar_images(idx, top_k) - 유사 이미지 시각화")
    print("4. searcher.evaluate_search_quality() - 검색 품질 평가")


if __name__ == "__main__":
    main()

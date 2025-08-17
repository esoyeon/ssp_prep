import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction import image
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import cv2
from PIL import Image
import pickle
import os
from typing import List, Tuple, Dict
import warnings

warnings.filterwarnings("ignore")


class WaferSimilaritySearch:
    """
    MIR-WM811K 데이터셋을 이용한 웨이퍼맵 이미지 유사도 검색 시스템
    """

    def __init__(self, data_path: str = "MIR-WM811K/Python/WM811K.pkl"):
        """
        초기화

        Args:
            data_path: WM811K.pkl 파일 경로 또는 더미 데이터 파일 경로
        """
        self.data_path = data_path
        self.df = None
        self.feature_vectors = None
        self.feature_names = None
        self.pca = None
        self.tsne = None
        self.is_dummy_data = False

    def load_data(self):
        """데이터 로드"""
        print("데이터를 로드하는 중...")
        try:
            self.df = pd.read_pickle(self.data_path)

            # 더미 데이터인지 확인
            if "waferMap" in self.df.columns and "failureType" in self.df.columns:
                if len(self.df) <= 1000:  # 더미 데이터는 보통 1000개 이하
                    self.is_dummy_data = True
                    print(f"더미 데이터 로드 완료: {len(self.df)} 개의 웨이퍼맵")
                else:
                    print(
                        f"실제 MIR-WM811K 데이터 로드 완료: {len(self.df)} 개의 웨이퍼맵"
                    )

                print(f"컬럼: {list(self.df.columns)}")
                return True
            else:
                print("웨이퍼맵 데이터 형식이 올바르지 않습니다.")
                return False

        except Exception as e:
            print(f"데이터 로드 실패: {e}")
            return False

    def extract_features(self, method: str = "auto"):
        """
        웨이퍼맵에서 특징 벡터 추출

        Args:
            method: 특징 추출 방법 ('auto', 'histogram', 'hog', 'pca', 'combined', 'ensemble', 'deep', 'simple')
        """
        if self.df is None:
            print("먼저 데이터를 로드해주세요.")
            return

        # 더미 데이터일 때는 자동으로 간단한 방법 사용
        if method == "auto":
            if self.is_dummy_data:
                method = "simple"
                print("더미 데이터 감지: 간단한 특징 추출 방법을 사용합니다.")
            else:
                method = "ensemble"
                print("실제 데이터 감지: 앙상블 특징 추출 방법을 사용합니다.")

        print(f"{method} 방법으로 특징을 추출하는 중...")

        features = []
        feature_names = []

        for idx, row in self.df.iterrows():
            wafer_map = row["waferMap"]

            if method == "simple":
                # 더미 데이터용 간단한 특징 추출
                simple_features = self._extract_simple_features(wafer_map)
                features.append(simple_features)
                if idx == 0:
                    feature_names = [f"simple_{i}" for i in range(len(simple_features))]

            elif method == "histogram":
                # 히스토그램 기반 특징
                hist_features = self._extract_histogram_features(wafer_map)
                features.append(hist_features)
                if idx == 0:
                    feature_names = [f"hist_bin_{i}" for i in range(len(hist_features))]

            elif method == "hog":
                # HOG (Histogram of Oriented Gradients) 특징
                hog_features = self._extract_hog_features(wafer_map)
                features.append(hog_features)
                if idx == 0:
                    feature_names = [f"hog_{i}" for i in range(len(hog_features))]

            elif method == "pca":
                # PCA 기반 특징
                pca_features = self._extract_pca_features(wafer_map)
                features.append(pca_features)
                if idx == 0:
                    feature_names = [f"pca_{i}" for i in range(len(pca_features))]

            elif method == "combined":
                # 여러 방법을 조합한 특징
                combined_features = self._extract_combined_features(wafer_map)
                features.append(combined_features)
                if idx == 0:
                    feature_names = [
                        f"combined_{i}" for i in range(len(combined_features))
                    ]

            elif method == "ensemble":
                # 앙상블 특징 추출 (권장)
                ensemble_features = self._extract_ensemble_features(wafer_map)
                features.append(ensemble_features)
                if idx == 0:
                    feature_names = [
                        f"ensemble_{i}" for i in range(len(ensemble_features))
                    ]

            elif method == "deep":
                # 딥러닝 기반 특징
                deep_features = self._extract_deep_features(wafer_map)
                features.append(deep_features)
                if idx == 0:
                    feature_names = [f"deep_{i}" for i in range(len(deep_features))]

        self.feature_vectors = np.array(features)
        self.feature_names = feature_names
        print(f"특징 추출 완료: {self.feature_vectors.shape}")

        # 특징 품질 평가
        self._evaluate_feature_quality()

    def _extract_histogram_features(self, wafer_map: np.ndarray) -> np.ndarray:
        """히스토그램 기반 특징 추출"""
        # 웨이퍼맵을 정규화 (0-255)
        normalized_map = (wafer_map * 255).astype(np.uint8)

        # 히스토그램 계산 (10개 구간)
        hist = cv2.calcHist([normalized_map], [0], None, [10], [0, 256])
        hist = hist.flatten() / hist.sum()  # 정규화

        return hist

    def _extract_hog_features(self, wafer_map: np.ndarray) -> np.ndarray:
        """HOG 특징 추출"""
        # 웨이퍼맵을 정규화 (0-255)
        normalized_map = (wafer_map * 255).astype(np.uint8)

        # HOG 특징 추출
        cell_size = (8, 8)
        block_size = (16, 16)
        block_stride = (8, 8)
        num_bins = 9

        hog = cv2.HOGDescriptor(
            block_size, cell_size, block_stride, cell_size, num_bins
        )
        features = hog.compute(normalized_map)

        return features.flatten()

    def _extract_pca_features(
        self, wafer_map: np.ndarray, n_components: int = 50
    ) -> np.ndarray:
        """PCA 기반 특징 추출"""
        # 웨이퍼맵을 1차원으로 평탄화
        flattened = wafer_map.flatten()

        # PCA 적용
        if len(flattened) > n_components:
            # 간단한 차원 축소를 위해 랜덤 샘플링
            indices = np.random.choice(len(flattened), n_components, replace=False)
            features = flattened[indices]
        else:
            features = np.pad(flattened, (0, n_components - len(flattened)), "constant")

        return features

    def _extract_combined_features(self, wafer_map: np.ndarray) -> np.ndarray:
        """여러 특징을 조합"""
        hist_features = self._extract_histogram_features(wafer_map)
        pca_features = self._extract_pca_features(wafer_map, 20)

        # 통계적 특징 추가
        statistical_features = [
            wafer_map.mean(),
            wafer_map.std(),
            wafer_map.min(),
            wafer_map.max(),
            np.percentile(wafer_map, 25),
            np.percentile(wafer_map, 75),
        ]

        combined = np.concatenate([hist_features, pca_features, statistical_features])
        return combined

    def _extract_ensemble_features(self, wafer_map: np.ndarray) -> np.ndarray:
        """앙상블 특징 추출 - 여러 방법을 조합"""
        features = []

        # 1. 기본 특징들
        hist_features = self._extract_histogram_features(wafer_map)
        features.extend(hist_features)

        # 2. 웨이퍼맵 특화 특징
        wafer_features = self._extract_wafer_specific_features(wafer_map)
        features.extend(wafer_features)

        # 3. 고급 이미지 처리 특징
        advanced_features = self._extract_advanced_image_features(wafer_map)
        features.extend(advanced_features)

        # 4. 통계적 특징
        statistical_features = self._extract_statistical_features(wafer_map)
        features.extend(statistical_features)

        return np.array(features)

    def _extract_wafer_specific_features(self, wafer_map: np.ndarray) -> np.ndarray:
        """웨이퍼맵에 특화된 특징 추출"""
        features = []

        # 원형 마스크 생성
        center = np.array(wafer_map.shape) / 2
        y, x = np.ogrid[: wafer_map.shape[0], : wafer_map.shape[1]]
        mask = (x - center[1]) ** 2 + (y - center[0]) ** 2 <= (
            min(wafer_map.shape) / 2
        ) ** 2

        # 마스크 적용된 웨이퍼맵
        masked_wafer = wafer_map * mask

        # 1. 방사형 특징 (중심에서 바깥쪽으로)
        radial_features = self._extract_radial_features(masked_wafer, center)
        features.extend(radial_features)

        # 2. 각도별 특징 (0도, 45도, 90도 등)
        angular_features = self._extract_angular_features(masked_wafer, center)
        features.extend(angular_features)

        # 3. 결함 패턴 특징
        defect_pattern_features = self._extract_defect_pattern_features(masked_wafer)
        features.extend(defect_pattern_features)

        return np.array(features)

    def _extract_radial_features(
        self, wafer_map: np.ndarray, center: np.ndarray
    ) -> List[float]:
        """방사형 특징 추출"""
        features = []

        # 중심에서 바깥쪽으로 5개 구간으로 나누기
        max_radius = min(wafer_map.shape) / 2
        radius_steps = np.linspace(0, max_radius, 6)

        for i in range(len(radius_steps) - 1):
            r1, r2 = radius_steps[i], radius_steps[i + 1]

            # 해당 구간의 마스크 생성
            y, x = np.ogrid[: wafer_map.shape[0], : wafer_map.shape[1]]
            mask = ((x - center[1]) ** 2 + (y - center[0]) ** 2 >= r1**2) & (
                (x - center[1]) ** 2 + (y - center[0]) ** 2 < r2**2
            )

            # 해당 구간의 특징 계산
            region = wafer_map * mask
            if mask.sum() > 0:
                features.extend(
                    [region.mean(), region.std(), region.sum() / mask.sum()]  # 밀도
                )
            else:
                features.extend([0.0, 0.0, 0.0])

        return features

    def _extract_angular_features(
        self, wafer_map: np.ndarray, center: np.ndarray
    ) -> List[float]:
        """각도별 특징 추출"""
        features = []

        # 8개 방향으로 나누기 (0도, 45도, 90도, 135도, 180도, 225도, 270도, 315도)
        angles = np.linspace(0, 2 * np.pi, 9)[:-1]

        for angle in angles:
            # 해당 방향의 특징 계산
            direction_features = self._extract_direction_features(
                wafer_map, center, angle
            )
            features.extend(direction_features)

        return features

    def _extract_direction_features(
        self, wafer_map: np.ndarray, center: np.ndarray, angle: float
    ) -> List[float]:
        """특정 방향의 특징 추출"""
        # 방향 벡터
        dx = np.cos(angle)
        dy = np.sin(angle)

        # 중심에서 해당 방향으로 선을 따라 특징 계산
        max_dist = min(wafer_map.shape) / 2
        distances = np.linspace(0, max_dist, 10)

        line_values = []
        for dist in distances:
            x = int(center[1] + dist * dx)
            y = int(center[0] + dist * dy)

            if 0 <= x < wafer_map.shape[1] and 0 <= y < wafer_map.shape[0]:
                line_values.append(wafer_map[y, x])

        if line_values:
            return [np.mean(line_values), np.std(line_values)]
        else:
            return [0.0, 0.0]

    def _extract_defect_pattern_features(self, wafer_map: np.ndarray) -> List[float]:
        """결함 패턴 특징 추출"""
        features = []

        # 1. 결함 클러스터 분석
        from scipy import ndimage
        from scipy.spatial.distance import pdist, squareform

        # 결함 픽셀 찾기 (임계값 이상)
        threshold = np.percentile(wafer_map, 75)
        defect_mask = wafer_map > threshold

        if defect_mask.sum() > 0:
            # 결함 픽셀의 위치
            defect_coords = np.column_stack(np.where(defect_mask))

            # 결함 간 거리
            if len(defect_coords) > 1:
                distances = pdist(defect_coords)
                features.extend(
                    [
                        np.mean(distances),
                        np.std(distances),
                        np.min(distances),
                        np.max(distances),
                    ]
                )
            else:
                features.extend([0.0, 0.0, 0.0, 0.0])

            # 결함 영역의 형태
            labeled, num_features = ndimage.label(defect_mask)
            if num_features > 0:
                sizes = [np.sum(labeled == i) for i in range(1, num_features + 1)]
                features.extend(
                    [
                        len(sizes),  # 결함 영역 개수
                        np.mean(sizes),  # 평균 결함 영역 크기
                        np.std(sizes),  # 결함 영역 크기 표준편차
                    ]
                )
            else:
                features.extend([0.0, 0.0, 0.0])
        else:
            features.extend([0.0] * 7)

        return features

    def _extract_advanced_image_features(self, wafer_map: np.ndarray) -> List[float]:
        """고급 이미지 처리 기반 특징 추출"""
        features = []

        # 1. 텍스처 특징 (GLCM)
        glcm_features = self._extract_glcm_features(wafer_map)
        features.extend(glcm_features)

        # 2. 경계 특징
        edge_features = self._extract_edge_features(wafer_map)
        features.extend(edge_features)

        # 3. 주파수 도메인 특징
        frequency_features = self._extract_frequency_features(wafer_map)
        features.extend(frequency_features)

        return features

    def _extract_glcm_features(self, wafer_map: np.ndarray) -> List[float]:
        """GLCM 텍스처 특징 추출"""
        try:
            from skimage.feature import graycomatrix, graycoprops

            # 정규화
            normalized = (wafer_map * 255).astype(np.uint8)

            # GLCM 계산
            glcm = graycomatrix(
                normalized,
                [1],
                [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
                levels=256,
                symmetric=True,
                normed=True,
            )

            # 텍스처 특징 계산
            contrast = graycoprops(glcm, "contrast").flatten()
            dissimilarity = graycoprops(glcm, "dissimilarity").flatten()
            homogeneity = graycoprops(glcm, "homogeneity").flatten()
            energy = graycoprops(glcm, "energy").flatten()
            correlation = graycoprops(glcm, "correlation").flatten()

            return (
                list(contrast)
                + list(dissimilarity)
                + list(homogeneity)
                + list(energy)
                + list(correlation)
            )

        except ImportError:
            return [0.0] * 20  # 기본값

    def _extract_edge_features(self, wafer_map: np.ndarray) -> List[float]:
        """경계 특징 추출"""
        try:
            from skimage.feature import canny
            from skimage.filters import sobel

            # Canny 엣지 검출
            edges_canny = canny(wafer_map, sigma=1.0)

            # Sobel 엣지 검출
            edges_sobel = sobel(wafer_map)

            features = [
                edges_canny.sum(),  # 엣지 픽셀 개수
                edges_canny.mean(),  # 엣지 밀도
                edges_sobel.mean(),  # Sobel 응답 평균
                edges_sobel.std(),  # Sobel 응답 표준편차
            ]

            return features

        except ImportError:
            return [0.0] * 4

    def _extract_frequency_features(self, wafer_map: np.ndarray) -> List[float]:
        """주파수 도메인 특징 추출"""
        try:
            from scipy.fft import fft2

            # 2D FFT
            fft_result = fft2(wafer_map)
            magnitude = np.abs(fft_result)

            # 주파수 대역별 에너지
            center_y, center_x = np.array(magnitude.shape) // 2

            # 저주파 (중앙 영역)
            low_freq = magnitude[
                center_y - 5 : center_y + 5, center_x - 5 : center_x + 5
            ]

            # 중주파 (중간 영역)
            mid_freq = magnitude[
                center_y - 15 : center_y + 15, center_x - 15 : center_x + 15
            ]
            mid_freq = mid_freq - low_freq

            # 고주파 (바깥 영역)
            high_freq = magnitude - mid_freq - low_freq

            features = [
                low_freq.mean(),  # 저주파 에너지
                mid_freq.mean(),  # 중주파 에너지
                high_freq.mean(),  # 고주파 에너지
                magnitude.std(),  # 전체 주파수 분포 표준편차
            ]

            return features

        except ImportError:
            return [0.0] * 4

    def _extract_statistical_features(self, wafer_map: np.ndarray) -> List[float]:
        """통계적 특징 추출"""
        features = [
            wafer_map.mean(),
            wafer_map.std(),
            wafer_map.min(),
            wafer_map.max(),
            np.percentile(wafer_map, 25),
            np.percentile(wafer_map, 50),  # 중앙값
            np.percentile(wafer_map, 75),
            wafer_map.var(),  # 분산
            wafer_map.sum(),  # 총합
            np.median(wafer_map),  # 중앙값 (다른 방법)
        ]

        return features

    def _extract_deep_features(self, wafer_map: np.ndarray) -> np.ndarray:
        """사전 훈련된 CNN 모델을 이용한 특징 추출"""
        try:
            import torch
            import torchvision.models as models
            import torchvision.transforms as transforms

            # 이미지 전처리
            transform = transforms.Compose(
                [
                    transforms.ToPILImage(),
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )

            # 웨이퍼맵을 3채널로 변환 (그레이스케일을 3번 반복)
            wafer_3ch = np.stack([wafer_map, wafer_map, wafer_map], axis=2)
            wafer_tensor = transform(wafer_3ch).unsqueeze(0)

            # ResNet 특징 추출
            model = models.resnet18(pretrained=True)
            model.eval()
            with torch.no_grad():
                features = model(wafer_tensor)

            return features.squeeze().numpy()

        except ImportError:
            print("PyTorch가 설치되지 않았습니다. 기본 특징을 사용합니다.")
            return self._extract_histogram_features(wafer_map)

    def _extract_simple_features(self, wafer_map: np.ndarray) -> np.ndarray:
        """더미 데이터용 간단한 특징 추출"""
        features = []

        # 1. 기본 통계 특징
        features.extend(
            [
                wafer_map.mean(),
                wafer_map.std(),
                wafer_map.min(),
                wafer_map.max(),
                np.percentile(wafer_map, 25),
                np.percentile(wafer_map, 50),  # 중앙값
                np.percentile(wafer_map, 75),
                wafer_map.var(),  # 분산
                wafer_map.sum(),  # 총합
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

        # 5. 웨이퍼 특화 특징 (간단한 버전)
        # 원형 마스크 생성
        center = np.array(wafer_map.shape) / 2
        y, x = np.ogrid[: wafer_map.shape[0], : wafer_map.shape[1]]
        mask = (x - center[1]) ** 2 + (y - center[0]) ** 2 <= (
            min(wafer_map.shape) / 2
        ) ** 2

        # 마스크 적용된 웨이퍼맵
        masked_wafer = wafer_map * mask

        # 방사형 특징 (간단한 버전)
        max_radius = min(wafer_map.shape) / 2
        inner_region = wafer_map[
            int(center[0] - max_radius / 3) : int(center[0] + max_radius / 3),
            int(center[1] - max_radius / 3) : int(center[1] + max_radius / 3),
        ]
        outer_region = wafer_map * (mask & (wafer_map > 0))

        features.extend(
            [
                inner_region.mean(),
                inner_region.std(),
                outer_region.mean(),
                outer_region.std(),
            ]
        )

        return np.array(features)

    def _evaluate_feature_quality(self):
        """추출된 특징의 품질 평가"""
        if self.feature_vectors is None:
            return

        print("\n=== 특징 품질 평가 ===")

        # 1. 특징 분산 분석
        feature_vars = np.var(self.feature_vectors, axis=0)
        low_variance_features = np.sum(feature_vars < 1e-6)

        print(f"총 특징 수: {self.feature_vectors.shape[1]}")
        print(f"낮은 분산 특징 수: {low_variance_features}")
        print(f"특징 분산 범위: {feature_vars.min():.6f} ~ {feature_vars.max():.6f}")

        # 2. 특징 간 상관관계 분석
        if self.feature_vectors.shape[1] > 1:
            corr_matrix = np.corrcoef(self.feature_vectors.T)
            high_corr_pairs = np.sum(np.abs(corr_matrix) > 0.95) - len(corr_matrix)
            print(f"높은 상관관계 특징 쌍: {high_corr_pairs}")

        # 3. 특징 선택 제안
        if low_variance_features > 0:
            print(
                f"제안: {low_variance_features}개의 낮은 분산 특징을 제거하는 것을 고려하세요."
            )

        print("특징 품질 평가 완료\n")

    def build_similarity_index(self, method: str = "cosine"):
        """
        유사도 검색을 위한 인덱스 구축

        Args:
            method: 유사도 측정 방법 ('cosine', 'euclidean')
        """
        if self.feature_vectors is None:
            print("먼저 특징을 추출해주세요.")
            return

        print(f"{method} 유사도 인덱스를 구축하는 중...")

        if method == "cosine":
            self.similarity_matrix = cosine_similarity(self.feature_vectors)
        elif method == "euclidean":
            self.similarity_matrix = euclidean_distances(self.feature_vectors)
            # 유클리드 거리를 유사도로 변환 (0~1 범위)
            max_dist = self.similarity_matrix.max()
            self.similarity_matrix = 1 - (self.similarity_matrix / max_dist)

        print("유사도 인덱스 구축 완료")

    def search_similar_images(
        self, query_idx: int, top_k: int = 10
    ) -> List[Tuple[int, float]]:
        """
        특정 이미지와 유사한 이미지 검색

        Args:
            query_idx: 검색할 이미지의 인덱스
            top_k: 반환할 유사 이미지 개수

        Returns:
            유사 이미지 인덱스와 유사도 점수 리스트
        """
        if not hasattr(self, "similarity_matrix"):
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
        """
        새로운 이미지와 유사한 이미지 검색

        Args:
            query_image: 검색할 새로운 이미지
            top_k: 반환할 유사 이미지 개수

        Returns:
            유사 이미지 인덱스와 유사도 점수 리스트
        """
        if self.feature_vectors is None:
            print("먼저 특징을 추출해주세요.")
            return []

        # 쿼리 이미지의 특징 추출
        if hasattr(self, "_extract_combined_features"):
            query_features = self._extract_combined_features(query_image)
        else:
            query_features = self._extract_histogram_features(query_image)

        # 모든 이미지와의 유사도 계산
        similarities = []
        for features in self.feature_vectors:
            if len(features) == len(query_features):
                sim = cosine_similarity([query_features], [features])[0][0]
                similarities.append(sim)
            else:
                similarities.append(0)

        # 상위 k개 유사 이미지 찾기
        similarities = np.array(similarities)
        top_indices = np.argsort(similarities)[::-1][:top_k]
        top_scores = similarities[top_indices]

        results = list(zip(top_indices, top_scores))
        return results

    def visualize_similar_images(self, query_idx: int, top_k: int = 5):
        """
        유사 이미지 시각화

        Args:
            query_idx: 쿼리 이미지 인덱스
            top_k: 표시할 유사 이미지 개수
        """
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
        axes[0].set_title(f'Query (Type: {self.df.iloc[query_idx]["failureType"]})')
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

    def get_failure_type_distribution(self):
        """실패 유형 분포 확인"""
        if self.df is None:
            print("먼저 데이터를 로드해주세요.")
            return

        failure_counts = self.df["failureType"].value_counts()

        plt.figure(figsize=(12, 6))
        failure_counts.plot(kind="bar")
        plt.title("Wafer Failure Type Distribution")
        plt.xlabel("Failure Type")
        plt.ylabel("Count")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

        return failure_counts

    def save_features(self, filepath: str):
        """추출된 특징을 파일로 저장"""
        if self.feature_vectors is None:
            print("먼저 특징을 추출해주세요.")
            return

        data = {
            "feature_vectors": self.feature_vectors,
            "feature_names": self.feature_names,
            "metadata": {"shape": self.feature_vectors.shape, "method": "combined"},
        }

        with open(filepath, "wb") as f:
            pickle.dump(data, f)
        print(f"특징을 {filepath}에 저장했습니다.")

    def load_features(self, filepath: str):
        """저장된 특징을 파일에서 로드"""
        try:
            with open(filepath, "rb") as f:
                data = pickle.load(f)

            self.feature_vectors = data["feature_vectors"]
            self.feature_names = data["feature_names"]
            print(f"특징을 {filepath}에서 로드했습니다: {self.feature_vectors.shape}")
            return True
        except Exception as e:
            print(f"특징 로드 실패: {e}")
            return False

    def evaluate_search_quality(self, num_queries: int = 10):
        """검색 품질 평가"""
        if not hasattr(self, "similarity_matrix") or self.similarity_matrix is None:
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

        if top_5_accuracy and same_type_accuracy:
            print(f"상위 5개 중 같은 타입 비율: {np.mean(top_5_accuracy):.3f}")
            print(f"전체 10개 중 같은 타입 비율: {np.mean(same_type_accuracy):.3f}")
        else:
            print("검색 품질 평가를 위한 충분한 데이터가 없습니다.")


def main():
    """메인 실행 함수"""
    print("=== MIR-WM811K 웨이퍼맵 유사도 검색 시스템 ===\n")

    # 시스템 초기화 (더미 데이터 우선 시도)
    data_paths = ["wafer_dummy_data.pkl", "MIR-WM811K/Python/WM811K.pkl"]
    searcher = None

    for data_path in data_paths:
        print(f"데이터 경로 시도: {data_path}")
        searcher = WaferSimilaritySearch(data_path=data_path)
        if searcher.load_data():
            print(f"데이터 로드 성공: {data_path}")
            break
        else:
            print(f"데이터 로드 실패: {data_path}")

    if searcher is None or searcher.df is None:
        print("사용 가능한 데이터를 찾을 수 없습니다.")
        print("더미 데이터를 생성하려면: python wafer_dummy_generator.py")
        return

    # 실패 유형 분포 확인
    print("\n실패 유형 분포:")
    failure_dist = searcher.get_failure_type_distribution()
    print(failure_dist)

    # 특징 추출 (자동으로 적절한 방법 선택)
    print("\n특징 추출 중...")
    searcher.extract_features(method="auto")

    # 유사도 인덱스 구축
    print("\n유사도 인덱스를 구축하는 중...")
    searcher.build_similarity_index(method="cosine")

    # 특징 저장 (선택사항)
    if searcher.is_dummy_data:
        searcher.save_features("wafer_dummy_features.pkl")
    else:
        searcher.save_features("wafer_features.pkl")

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

    # 더미 데이터일 때 추가 테스트
    if searcher.is_dummy_data:
        print("\n=== 더미 데이터 추가 테스트 ===")

        # 검색 품질 평가
        print("검색 품질 평가:")
        searcher.evaluate_search_quality(num_queries=10)

        # 다른 결함 유형으로도 테스트
        unique_types = searcher.df["failureType"].unique()
        print(f"\n사용 가능한 결함 유형: {unique_types}")

        # 각 결함 유형별로 하나씩 테스트
        for failure_type in unique_types[:3]:  # 처음 3개 유형만 테스트
            type_indices = searcher.df[searcher.df["failureType"] == failure_type].index
            if len(type_indices) > 0:
                test_idx = type_indices[0]
                print(f"\n=== {failure_type} 유형 테스트 (이미지 {test_idx}) ===")

                similar_images = searcher.search_similar_images(test_idx, top_k=3)
                for i, (idx, score) in enumerate(similar_images):
                    similar_failure_type = searcher.df.iloc[idx]["failureType"]
                    print(
                        f"  유사 이미지 {i+1}: {idx} (유사도: {score:.3f}, 유형: {similar_failure_type})"
                    )

    print("\n=== 시스템 사용법 ===")
    print(
        "1. searcher.search_similar_images(idx, top_k) - 특정 이미지와 유사한 이미지 검색"
    )
    print(
        "2. searcher.search_by_image(new_image, top_k) - 새로운 이미지와 유사한 이미지 검색"
    )
    print("3. searcher.visualize_similar_images(idx, top_k) - 유사 이미지 시각화")
    print("4. searcher.save_features(filepath) - 특징 저장")
    print("5. searcher.load_features(filepath) - 특징 로드")

    if searcher.is_dummy_data:
        print("\n더미 데이터 모드:")
        print("- 간단한 특징 추출으로 빠른 처리")
        print("- 검색 품질 평가 가능")
        print("- 다양한 결함 유형 테스트 가능")
    else:
        print("\n실제 데이터 모드:")
        print("- 고급 특징 추출으로 정확한 검색")
        print("- 대용량 데이터 처리")
        print("- 실제 웨이퍼 결함 패턴 분석")


if __name__ == "__main__":
    main()

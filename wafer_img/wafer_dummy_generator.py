import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import List, Tuple, Dict
import random


class WaferDummyGenerator:
    """
    웨이퍼맵 더미 데이터 생성기
    실제 MIR-WM811K 데이터셋과 유사한 패턴의 결함 이미지들을 생성
    """

    def __init__(self, wafer_size: int = 64):
        """
        초기화

        Args:
            wafer_size: 웨이퍼맵 크기 (기본값: 64x64)
        """
        self.wafer_size = wafer_size
        self.center = np.array([wafer_size // 2, wafer_size // 2])
        self.radius = wafer_size // 2 - 2

        # 결함 유형 정의
        self.defect_types = [
            "Center",
            "Donut",
            "Edge-Loc",
            "Edge-Ring",
            "Loc",
            "Near-full",
            "Random",
            "Scratch",
        ]

    def create_wafer_mask(self) -> np.ndarray:
        """원형 웨이퍼 마스크 생성"""
        y, x = np.ogrid[: self.wafer_size, : self.wafer_size]
        mask = (x - self.center[1]) ** 2 + (y - self.center[0]) ** 2 <= self.radius**2
        return mask.astype(float)

    def generate_center_defect(self) -> np.ndarray:
        """중앙 결함 패턴 생성 - 더 현실적으로"""
        wafer = np.zeros((self.wafer_size, self.wafer_size))
        mask = self.create_wafer_mask()

        # 중앙 영역에 결함 생성 (여러 개의 작은 결함 클러스터)
        center_radius = self.radius // 3

        # 주요 결함 영역
        y, x = np.ogrid[: self.wafer_size, : self.wafer_size]
        center_mask = (x - self.center[1]) ** 2 + (
            y - self.center[0]
        ) ** 2 <= center_radius**2

        # 결함 강도를 불규칙하게 설정
        defect_intensity = np.random.uniform(0.6, 0.9)
        wafer[center_mask] = defect_intensity

        # 추가적인 작은 결함들 (중앙 주변에)
        num_small_defects = np.random.randint(3, 8)
        for _ in range(num_small_defects):
            angle = np.random.uniform(0, 2 * np.pi)
            distance = np.random.uniform(center_radius * 0.5, center_radius * 1.2)

            defect_x = int(self.center[1] + distance * np.cos(angle))
            defect_y = int(self.center[0] + distance * np.sin(angle))

            defect_size = np.random.randint(2, 6)
            defect_intensity = np.random.uniform(0.5, 0.8)

            # 작은 결함 영역 생성
            for dy in range(-defect_size, defect_size + 1):
                for dx in range(-defect_size, defect_size + 1):
                    y, x = defect_y + dy, defect_x + dx
                    if 0 <= y < self.wafer_size and 0 <= x < self.wafer_size:
                        if dx**2 + dy**2 <= defect_size**2:
                            wafer[y, x] = defect_intensity

        # 텍스처와 노이즈 추가
        wafer = self._add_realistic_texture(wafer, mask)

        return wafer * mask

    def generate_donut_defect(self) -> np.ndarray:
        """도넛형 결함 패턴 생성 - 더 현실적으로"""
        wafer = np.zeros((self.wafer_size, self.wafer_size))
        mask = self.create_wafer_mask()

        # 도넛 영역 정의 (불규칙한 경계)
        inner_radius = self.radius // 3
        outer_radius = 2 * self.radius // 3

        # 도넛 경계를 불규칙하게 만들기
        y, x = np.ogrid[: self.wafer_size, : self.wafer_size]
        distances = np.sqrt((x - self.center[1]) ** 2 + (y - self.center[0]) ** 2)

        # 불규칙한 도넛 마스크 생성
        donut_mask = np.zeros_like(wafer, dtype=bool)
        for i in range(self.wafer_size):
            for j in range(self.wafer_size):
                if mask[i, j]:
                    dist = np.sqrt(
                        (i - self.center[0]) ** 2 + (j - self.center[1]) ** 2
                    )
                    # 불규칙한 경계 추가
                    noise = np.random.normal(0, 2)
                    if inner_radius + noise <= dist <= outer_radius + noise:
                        donut_mask[i, j] = True

        # 결함 강도를 불규칙하게 설정
        defect_intensity = np.random.uniform(0.5, 0.8)
        wafer[donut_mask] = defect_intensity

        # 도넛 내부에 추가 결함들
        num_inner_defects = np.random.randint(2, 5)
        for _ in range(num_inner_defects):
            angle = np.random.uniform(0, 2 * np.pi)
            distance = np.random.uniform(inner_radius * 0.3, inner_radius * 0.8)

            defect_x = int(self.center[1] + distance * np.cos(angle))
            defect_y = int(self.center[0] + distance * np.sin(angle))

            defect_size = np.random.randint(1, 4)
            defect_intensity = np.random.uniform(0.4, 0.7)

            for dy in range(-defect_size, defect_size + 1):
                for dx in range(-defect_size, defect_size + 1):
                    y, x = defect_y + dy, defect_x + dx
                    if 0 <= y < self.wafer_size and 0 <= x < self.wafer_size:
                        if dx**2 + dy**2 <= defect_size**2:
                            wafer[y, x] = defect_intensity

        # 텍스처와 노이즈 추가
        wafer = self._add_realistic_texture(wafer, mask)

        return wafer * mask

    def generate_edge_loc_defect(self) -> np.ndarray:
        """엣지 위치 결함 패턴 생성 - 더 현실적으로"""
        wafer = np.zeros((self.wafer_size, self.wafer_size))
        mask = self.create_wafer_mask()

        # 엣지 근처에 여러 개의 결함 영역 생성
        num_edge_defects = np.random.randint(2, 6)

        for _ in range(num_edge_defects):
            # 결함 위치 (엣지 근처)
            edge_distance = self.radius - np.random.randint(3, 12)
            angle = np.random.uniform(0, 2 * np.pi)

            defect_x = int(self.center[1] + edge_distance * np.cos(angle))
            defect_y = int(self.center[0] + edge_distance * np.sin(angle))

            # 결함 크기와 강도를 다양하게 설정
            defect_size = np.random.randint(2, 8)
            defect_intensity = np.random.uniform(0.6, 0.9)

            # 불규칙한 결함 모양 생성
            defect_mask = np.zeros(
                (defect_size * 2 + 1, defect_size * 2 + 1), dtype=bool
            )

            # 기본 원형 결함
            for dy in range(-defect_size, defect_size + 1):
                for dx in range(-defect_size, defect_size + 1):
                    if dx**2 + dy**2 <= defect_size**2:
                        defect_mask[dy + defect_size, dx + defect_size] = True

            # 결함 모양을 불규칙하게 만들기
            defect_mask = self._add_shape_noise(defect_mask)

            # 웨이퍼에 결함 적용
            for dy in range(-defect_size, defect_size + 1):
                for dx in range(-defect_size, defect_size + 1):
                    y, x = defect_y + dy, defect_x + dx
                    if 0 <= y < self.wafer_size and 0 <= x < self.wafer_size:
                        if defect_mask[dy + defect_size, dx + defect_size]:
                            wafer[y, x] = defect_intensity

        # 텍스처와 노이즈 추가
        wafer = self._add_realistic_texture(wafer, mask)

        return wafer * mask

    def generate_edge_ring_defect(self) -> np.ndarray:
        """엣지 링 결함 패턴 생성 - 더 현실적으로"""
        wafer = np.zeros((self.wafer_size, self.wafer_size))
        mask = self.create_wafer_mask()

        # 엣지 근처 링 형태 결함 (불규칙한 두께)
        ring_radius = self.radius - np.random.randint(2, 8)
        ring_thickness = np.random.randint(3, 8)

        y, x = np.ogrid[: self.wafer_size, : self.wafer_size]
        distances = np.sqrt((x - self.center[1]) ** 2 + (y - self.center[0]) ** 2)

        # 불규칙한 링 마스크 생성
        ring_mask = np.zeros_like(wafer, dtype=bool)
        for i in range(self.wafer_size):
            for j in range(self.wafer_size):
                if mask[i, j]:
                    dist = np.sqrt(
                        (i - self.center[0]) ** 2 + (j - self.center[1]) ** 2
                    )
                    # 링 두께를 불규칙하게 만들기
                    thickness_variation = np.random.normal(0, 1.5)
                    if (
                        ring_radius - ring_thickness / 2 + thickness_variation
                        <= dist
                        <= ring_radius + ring_thickness / 2 + thickness_variation
                    ):
                        ring_mask[i, j] = True

        defect_intensity = np.random.uniform(0.5, 0.8)
        wafer[ring_mask] = defect_intensity

        # 링 내부에 추가 결함들
        num_inner_defects = np.random.randint(3, 8)
        for _ in range(num_inner_defects):
            angle = np.random.uniform(0, 2 * np.pi)
            distance = np.random.uniform(0, ring_radius - ring_thickness)

            defect_x = int(self.center[1] + distance * np.cos(angle))
            defect_y = int(self.center[0] + distance * np.sin(angle))

            defect_size = np.random.randint(1, 4)
            defect_intensity = np.random.uniform(0.4, 0.7)

            for dy in range(-defect_size, defect_size + 1):
                for dx in range(-defect_size, defect_size + 1):
                    y, x = defect_y + dy, defect_x + dx
                    if 0 <= y < self.wafer_size and 0 <= x < self.wafer_size:
                        if dx**2 + dy**2 <= defect_size**2:
                            wafer[y, x] = defect_intensity

        # 텍스처와 노이즈 추가
        wafer = self._add_realistic_texture(wafer, mask)

        return wafer * mask

    def generate_loc_defect(self) -> np.ndarray:
        """지역 결함 패턴 생성 - 더 현실적으로"""
        wafer = np.zeros((self.wafer_size, self.wafer_size))
        mask = self.create_wafer_mask()

        # 웨이퍼 내부에 랜덤한 위치에 결함 생성
        num_defects = np.random.randint(3, 8)

        for _ in range(num_defects):
            # 결함 위치 (중심에서 랜덤)
            distance = np.random.uniform(0, self.radius * 0.8)
            angle = np.random.uniform(0, 2 * np.pi)

            defect_x = int(self.center[1] + distance * np.cos(angle))
            defect_y = int(self.center[0] + distance * np.sin(angle))

            # 결함 크기와 강도를 다양하게 설정
            defect_size = np.random.randint(2, 7)
            defect_intensity = np.random.uniform(0.6, 0.9)

            # 불규칙한 결함 모양 생성
            defect_mask = np.zeros(
                (defect_size * 2 + 1, defect_size * 2 + 1), dtype=bool
            )

            # 기본 원형 결함
            for dy in range(-defect_size, defect_size + 1):
                for dx in range(-defect_size, defect_size + 1):
                    if dx**2 + dy**2 <= defect_size**2:
                        defect_mask[dy + defect_size, dx + defect_size] = True

            # 결함 모양을 불규칙하게 만들기
            defect_mask = self._add_shape_noise(defect_mask)

            # 웨이퍼에 결함 적용
            for dy in range(-defect_size, defect_size + 1):
                for dx in range(-defect_size, defect_size + 1):
                    y, x = defect_y + dy, defect_x + dx
                    if 0 <= y < self.wafer_size and 0 <= x < self.wafer_size:
                        if defect_mask[dy + defect_size, dx + defect_size]:
                            wafer[y, x] = defect_intensity

        # 텍스처와 노이즈 추가
        wafer = self._add_realistic_texture(wafer, mask)

        return wafer * mask

    def generate_near_full_defect(self) -> np.ndarray:
        """거의 전체 결함 패턴 생성 - 더 현실적으로"""
        wafer = np.zeros((self.wafer_size, self.wafer_size))
        mask = self.create_wafer_mask()

        # 웨이퍼의 대부분을 결함으로 덮기 (불규칙한 패턴)
        defect_coverage = np.random.uniform(0.75, 0.95)

        # 체계적인 결함 패턴 생성 (완전 랜덤이 아닌)
        y, x = np.ogrid[: self.wafer_size, : self.wafer_size]
        distances = np.sqrt((x - self.center[1]) ** 2 + (y - self.center[0]) ** 2)

        # 중심에서 바깥쪽으로 결함 밀도가 증가하는 패턴
        for i in range(self.wafer_size):
            for j in range(self.wafer_size):
                if mask[i, j]:
                    dist = np.sqrt(
                        (i - self.center[0]) ** 2 + (j - self.center[1]) ** 2
                    )
                    # 거리에 따른 결함 확률
                    distance_factor = dist / self.radius
                    defect_prob = defect_coverage * (0.5 + 0.5 * distance_factor)

                    # 추가적인 불규칙성
                    noise = np.random.normal(0, 0.1)
                    defect_prob = np.clip(defect_prob + noise, 0, 1)

                    if np.random.random() < defect_prob:
                        defect_intensity = np.random.uniform(0.5, 0.9)
                        wafer[i, j] = defect_intensity

        # 텍스처와 노이즈 추가
        wafer = self._add_realistic_texture(wafer, mask)

        return wafer * mask

    def generate_random_defect(self) -> np.ndarray:
        """랜덤 결함 패턴 생성 - 더 현실적으로"""
        wafer = np.zeros((self.wafer_size, self.wafer_size))
        mask = self.create_wafer_mask()

        # 랜덤하게 결함 픽셀 생성 (완전 랜덤이 아닌 클러스터링)
        defect_density = np.random.uniform(0.15, 0.45)

        # 클러스터링을 위한 시드 포인트 생성
        num_seeds = np.random.randint(5, 15)
        seed_points = []

        for _ in range(num_seeds):
            angle = np.random.uniform(0, 2 * np.pi)
            distance = np.random.uniform(0, self.radius * 0.9)

            seed_x = int(self.center[1] + distance * np.cos(angle))
            seed_y = int(self.center[0] + distance * np.sin(angle))

            if 0 <= seed_y < self.wafer_size and 0 <= seed_x < self.wafer_size:
                seed_points.append((seed_y, seed_x))

        # 각 시드 주변에 결함 생성
        for seed_y, seed_x in seed_points:
            cluster_size = np.random.randint(3, 12)
            cluster_intensity = np.random.uniform(0.5, 0.9)

            for dy in range(-cluster_size, cluster_size + 1):
                for dx in range(-cluster_size, cluster_size + 1):
                    y, x = seed_y + dy, seed_x + dx
                    if 0 <= y < self.wafer_size and 0 <= x < self.wafer_size:
                        if mask[y, x]:
                            # 클러스터 중심에서 멀어질수록 결함 확률 감소
                            cluster_distance = np.sqrt(dx**2 + dy**2)
                            if cluster_distance <= cluster_size:
                                defect_prob = defect_density * (
                                    1 - cluster_distance / cluster_size
                                )
                                if np.random.random() < defect_prob:
                                    wafer[y, x] = cluster_intensity

        # 텍스처와 노이즈 추가
        wafer = self._add_realistic_texture(wafer, mask)

        return wafer * mask

    def generate_scratch_defect(self) -> np.ndarray:
        """스크래치 결함 패턴 생성 - 더 현실적으로"""
        wafer = np.zeros((self.wafer_size, self.wafer_size))
        mask = self.create_wafer_mask()

        # 여러 개의 스크래치 생성
        num_scratches = np.random.randint(1, 4)

        for _ in range(num_scratches):
            # 스크래치 시작점과 끝점
            start_angle = np.random.uniform(0, 2 * np.pi)
            end_angle = start_angle + np.random.uniform(0.2, 1.5)

            start_radius = np.random.uniform(0, self.radius * 0.8)
            end_radius = np.random.uniform(start_radius, self.radius)

            # 스크래치 경로 생성 (곡선 형태)
            num_points = np.random.randint(30, 80)
            angles = np.linspace(start_angle, end_angle, num_points)
            radii = np.linspace(start_radius, end_radius, num_points)

            # 스크래치 경로에 곡선성 추가
            curve_factor = np.random.uniform(-0.1, 0.1)
            for i in range(num_points):
                angles[i] += curve_factor * np.sin(i / num_points * np.pi)

            defect_intensity = np.random.uniform(0.6, 0.9)
            scratch_width = np.random.randint(1, 5)

            # 스크래치 경로를 따라 결함 생성
            for i in range(num_points):
                angle = angles[i]
                radius = radii[i]

                x = int(self.center[1] + radius * np.cos(angle))
                y = int(self.center[0] + radius * np.sin(angle))

                # 스크래치 폭 적용 (불규칙한 폭)
                current_width = scratch_width + np.random.randint(-1, 2)
                current_width = max(1, current_width)

                for dy in range(-current_width, current_width + 1):
                    for dx in range(-current_width, current_width + 1):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < self.wafer_size and 0 <= nx < self.wafer_size:
                            if dx**2 + dy**2 <= current_width**2:
                                # 스크래치 경계에서 강도 감소
                                edge_factor = 1 - (dx**2 + dy**2) / (current_width**2)
                                current_intensity = defect_intensity * edge_factor
                                wafer[ny, nx] = max(wafer[ny, nx], current_intensity)

        # 텍스처와 노이즈 추가
        wafer = self._add_realistic_texture(wafer, mask)

        return wafer * mask

    def _add_realistic_texture(self, wafer: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """현실적인 텍스처와 노이즈 추가"""
        # 1. 기본 노이즈 (가우시안)
        noise = np.random.normal(0, 0.05, wafer.shape)
        wafer = wafer + noise

        # 2. 텍스처 노이즈 (Perlin 노이즈와 유사한 효과)
        texture_noise = np.zeros_like(wafer)
        for i in range(0, self.wafer_size, 4):
            for j in range(0, self.wafer_size, 4):
                texture_value = np.random.normal(0, 0.03)
                texture_noise[i : i + 4, j : j + 4] = texture_value

        wafer = wafer + texture_noise

        # 3. 결함 경계 부드럽게 만들기
        from scipy import ndimage

        wafer = ndimage.gaussian_filter(wafer, sigma=0.5)

        # 4. 추가적인 미세한 결함들 (스팟 노이즈)
        spot_noise = np.random.random(wafer.shape) < 0.02
        spot_intensity = np.random.uniform(0.1, 0.3, wafer.shape)
        wafer[spot_noise] = np.maximum(wafer[spot_noise], spot_intensity[spot_noise])

        # 5. 값 범위 조정
        wafer = np.clip(wafer, 0, 1)

        return wafer

    def _add_shape_noise(self, defect_mask: np.ndarray) -> np.ndarray:
        """결함 모양을 불규칙하게 만들기"""
        # 결함 마스크에 노이즈 추가
        noisy_mask = defect_mask.copy()

        # 경계에 불규칙성 추가
        for i in range(1, defect_mask.shape[0] - 1):
            for j in range(1, defect_mask.shape[1] - 1):
                if defect_mask[i, j]:
                    # 주변 픽셀 상태에 따라 경계를 불규칙하게 만들기
                    neighbors = defect_mask[i - 1 : i + 2, j - 1 : j + 2]
                    if neighbors.sum() < 9:  # 경계 픽셀
                        if np.random.random() < 0.3:  # 30% 확률로 제거
                            noisy_mask[i, j] = False
                else:
                    # 주변에 결함이 있으면 추가할 확률
                    neighbors = defect_mask[i - 1 : i + 2, j - 1 : j + 2]
                    if neighbors.sum() > 0 and np.random.random() < 0.2:
                        noisy_mask[i, j] = True

        return noisy_mask

    def generate_defect_by_type(self, defect_type: str) -> np.ndarray:
        """결함 유형에 따른 웨이퍼맵 생성"""
        if defect_type == "Center":
            return self.generate_center_defect()
        elif defect_type == "Donut":
            return self.generate_donut_defect()
        elif defect_type == "Edge-Loc":
            return self.generate_edge_loc_defect()
        elif defect_type == "Edge-Ring":
            return self.generate_edge_ring_defect()
        elif defect_type == "Loc":
            return self.generate_loc_defect()
        elif defect_type == "Near-full":
            return self.generate_near_full_defect()
        elif defect_type == "Random":
            return self.generate_random_defect()
        elif defect_type == "Scratch":
            return self.generate_scratch_defect()
        else:
            return self.generate_random_defect()

    def generate_dummy_dataset(
        self, num_samples: int = 100, defect_distribution: Dict[str, float] = None
    ) -> pd.DataFrame:
        """
        더미 데이터셋 생성

        Args:
            num_samples: 생성할 샘플 수
            defect_distribution: 결함 유형별 분포 (기본값: 균등 분포)

        Returns:
            웨이퍼맵 데이터가 포함된 DataFrame
        """
        if defect_distribution is None:
            # 균등 분포
            defect_distribution = {
                defect_type: 1.0 / len(self.defect_types)
                for defect_type in self.defect_types
            }

        # 결함 유형별 샘플 수 계산
        defect_counts = {}
        for defect_type, ratio in defect_distribution.items():
            defect_counts[defect_type] = int(num_samples * ratio)

        # 부족한 샘플은 랜덤으로 보충
        total_count = sum(defect_counts.values())
        if total_count < num_samples:
            remaining = num_samples - total_count
            for _ in range(remaining):
                defect_type = np.random.choice(self.defect_types)
                defect_counts[defect_type] += 1

        # 데이터 생성
        data = []
        for defect_type, count in defect_counts.items():
            for _ in range(count):
                wafer_map = self.generate_defect_by_type(defect_type)

                # 노이즈 추가 (현실감 향상)
                noise = np.random.normal(0, 0.05, wafer_map.shape)
                wafer_map = np.clip(wafer_map + noise, 0, 1)

                data.append(
                    {
                        "waferMap": wafer_map,
                        "failureType": defect_type,
                        "trainTestLabel": (
                            "Training" if np.random.random() < 0.8 else "Test"
                        ),
                    }
                )

        # 데이터 순서 섞기
        random.shuffle(data)

        return pd.DataFrame(data)

    def visualize_defect_types(self, num_samples_per_type: int = 3):
        """각 결함 유형별 샘플 시각화"""
        fig, axes = plt.subplots(
            len(self.defect_types),
            num_samples_per_type,
            figsize=(15, 3 * len(self.defect_types)),
        )

        if len(self.defect_types) == 1:
            axes = axes.reshape(1, -1)

        for i, defect_type in enumerate(self.defect_types):
            for j in range(num_samples_per_type):
                wafer_map = self.generate_defect_by_type(defect_type)

                if len(self.defect_types) == 1:
                    ax = axes[j]
                else:
                    ax = axes[i, j]

                ax.imshow(wafer_map, cmap="viridis", vmin=0, vmax=1)
                ax.set_title(f"{defect_type} - Sample {j+1}")
                ax.axis("off")

        plt.tight_layout()
        plt.show()

    def save_dummy_dataset(
        self, num_samples: int = 100, filename: str = "wafer_dummy_data.pkl"
    ):
        """더미 데이터셋을 파일로 저장"""
        df = self.generate_dummy_dataset(num_samples)
        df.to_pickle(filename)
        print(f"{num_samples}개의 웨이퍼맵 샘플을 {filename}에 저장했습니다.")
        return df


def main():
    """메인 실행 함수"""
    print("=== 웨이퍼맵 더미 데이터 생성기 ===\n")

    # 생성기 초기화
    generator = WaferDummyGenerator(wafer_size=64)

    # 1. 각 결함 유형별 샘플 시각화
    print("각 결함 유형별 샘플을 생성하고 시각화합니다...")
    generator.visualize_defect_types(num_samples_per_type=3)

    # 2. 더미 데이터셋 생성 (100개 샘플)
    print("\n100개의 웨이퍼맵 샘플을 생성합니다...")
    df = generator.generate_dummy_dataset(num_samples=100)

    print(f"생성된 데이터셋 크기: {len(df)}")
    print(f"결함 유형별 분포:")
    print(df["failureType"].value_counts())

    # 3. 파일로 저장
    print("\n데이터를 파일로 저장합니다...")
    generator.save_dummy_dataset(num_samples=100, filename="wafer_dummy_data.pkl")

    # 4. 샘플 데이터 확인
    print("\n=== 샘플 데이터 정보 ===")
    print(f"컬럼: {list(df.columns)}")
    print(f"첫 번째 웨이퍼맵 크기: {df.iloc[0]['waferMap'].shape}")
    print(f"데이터 타입: {df.iloc[0]['waferMap'].dtype}")

    print("\n=== 사용법 ===")
    print("1. generator.generate_defect_by_type('Center') - 특정 결함 유형 생성")
    print("2. generator.generate_dummy_dataset(100) - 100개 샘플 생성")
    print("3. generator.visualize_defect_types() - 결함 유형별 시각화")
    print("4. generator.save_dummy_dataset(100) - 파일로 저장")


if __name__ == "__main__":
    main()

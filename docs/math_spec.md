🧠 **Adaptive Hybrid-PEFT Mamba 수학적 정식화 문서 (Formal Mathematical Specification)**

버전: 1.2
작성일: 2025. 6. 9
작성 목적: 각 핵심 구성 요소(Pillar 1–3)의 동작 메커니즘을 **정량적 수식 및 정의**를 통해 정형화함으로써 이론적 타당성과 구현 가능성 확보

---

## ✅ Pillar 1: **하드웨어 친화적 백본 최적화 (Variable-Aware Scan)**

### 1.1 상태 공간 모델의 연속-이산 변환 및 선택성

Mamba의 기본은 연속 시간 선형 시스템을 기반으로 하며, 다음과 같이 표현됩니다:

$$
\begin{aligned}
h'(t) &= A_c h(t) + B_c x(t) \\
\end{aligned}
$$

이를 시간 간격 $\Delta$에서 Zero-Order Hold 기반으로 이산화하면:

$$
\begin{aligned}
\bar{A} &= \exp(\Delta A_c), \\
\bar{B} &= (\exp(\Delta A_c) - I)A_c^{-1} B_c, \\
h_t &= \bar{A} h_{t-1} + \bar{B} x_t
\end{aligned}
$$

> 🔍 선택성(Selectivity) 정식화:

Mamba의 핵심은 상태 공간 파라미터를 현재 입력 $x_t$에 따라 동적으로 생성함으로써, 토큰별로 정보 선택성을 구현하는 점에 있습니다:

$$
\begin{aligned}
\Delta_t &= \text{proj}_{\Delta}(\text{Linear}_{\Delta}(x_t)) \\
B_t &= \text{Linear}_{B}(x_t) \\
C_t &= \text{Linear}_{C}(x_t)
\end{aligned}
$$

이러한 구조는 각 토큰의 중요도 기반 정보 필터링과 기억 유지에 기여하며, Scan 경로 최적화의 이론적 정당성을 강화합니다.

### 1.2 Scan Path 최적화

* 상태 변수 간의 상관관계 행렬:

$$
\Sigma_{i,j} = \mathbb{E}[(h_i - \mu_i)(h_j - \mu_j)]
$$

* 경로 비용 정의 (상관계수 기반):

$$
\text{Cost}(i,j) = 1 - |\rho_{i,j}|
$$

* 최적 스캔 경로 (이론적 정의):

$$
\pi^* = \arg\min_{\pi \in S_d} \sum_{t=1}^{d-1} \text{Cost}(\pi(t), \pi(t+1))
$$

> ⚠️ **실제 구현에서는 근사 알고리즘 사용**: Nearest Neighbor Heuristic, Simulated Annealing 등

* 경로 적용:

$$
h_t' = h_t[\pi^*]
$$

* $\Sigma$는 학습된 모델을 통해 입력 데이터를 사전 통과시켜 수집한 평균 상태 벡터 기반으로 **사전 계산**됨

---

## ✅ Pillar 2: **지능형 동적 희소화 (Learned Masking)**

### 2.1 Mamba 구조에 특화된 희소화 수식

Mamba의 선택성 구현은 특정 프로젝션 레이어 가중치에 대한 조절로 구현되며, 마스킹은 해당 가중치에 적용됩니다. 예시:

$$
B_t = \text{Linear}(M_B \odot W_B)(x_t)
$$

이는 마스크 $M_B$가 입력 투영 가중치에 직접 적용됨을 의미하며, 연산량 절감뿐 아니라 Pillar 3에서 중요도 기반 PEFT 적용의 근거가 됩니다.

또는 글로벌 컨볼루션 커널 $K$에 희소 마스크 적용:

$$
K_{\text{sparse}} = M \odot K
$$

### 2.2 Gumbel-Sigmoid 기반 이진 마스크 샘플링

$$
M_{i,j} = \text{Sigmoid}\left( \frac{L_{i,j} + G_{i,j}}{\tau} \right), \quad G_{i,j} \sim \text{Gumbel}(0,1)
$$

* 이는 Binary Concrete 분포로도 불리며, gradient가 전파 가능한 이진 확률화 방식입니다.
* 테스트 시: $M_{i,j} = \mathbb{1}(L_{i,j} > 0)$

---

## ✅ Pillar 3: **적응형 경량 튜닝 (Hybrid PEFT)**

### 3.1 중요도 기반 파라미터 선택 및 시너지 구조

* 중요도 계산:

$$
\text{Importance}_{i,j} = |L_{i,j}| \quad \text{or} \quad \sum_{t=1}^T M_{i,j}^{(t)} / T
$$

* 적용 방식:

> 중요도가 높은 파라미터 → LoRA (e.g., in\_proj, out\_proj, 확장 레이어)
> 중요도가 낮은 파라미터 → IA³ (e.g., LayerNorm 주변, non-critical linear)

* Pillar 2와의 시너지:

Pillar 2의 희소 마스크 $M$는 단지 연산량을 줄이는 데 그치지 않고, **튜닝 대상 파라미터를 필터링하는 역할도 수행**합니다. 즉:

* $M_{i,j} = 0$ → 튜닝 대상 제외
* $M_{i,j} = 1$ → 중요도 기반 LoRA/IA³ 적용

이 구조는 **튜닝 자원의 낭비를 원천적으로 차단**하고 효율성을 극대화합니다.

### 3.2 LoRA 적용

$$
W x \rightarrow W x + \Delta W x, \quad \Delta W = A B, \quad A \in \mathbb{R}^{d \times r}, B \in \mathbb{R}^{r \times d}
$$

### 3.3 IA³ 적용

$$
z' = \alpha \cdot z, \quad \alpha \in \mathbb{R}^d
$$

---

## 🔁 전략 간 상호작용 공식화

### 연산량 총합

$$
\text{Total Cost} = \underbrace{\text{Cost}_{\text{precompute}}(\pi^*)}_{\text{One-time}} + N \times \left( \underbrace{\text{FLOPs}_{\text{SSM}}(M)}_{\text{Per-token}} + \underbrace{\text{FLOPs}_{\text{PEFT}}}_{\text{Per-token}} \right)
$$

### 효율성 지표

$$
\mathcal{E} = \frac{\text{Accuracy}}{\text{FLOPs}} \times \frac{1}{\text{Params}}
$$

---

## 🔬 전략 요약표 (개선 반영)

| 전략               | 주요 수식                               | 기대 효과            | 핵심 고려사항                |
| ---------------- | ----------------------------------- | ---------------- | ---------------------- |
| Variable Scan    | $\pi^* = \arg\min \sum \text{Cost}$ | Latency 감소       | NP-hard, 근사 알고리즘 사용 필요 |
| Learned Mask     | $M = \text{Sigmoid}((L + G)/\tau)$  | FLOPs 절감 + 정보 보존 | 적용 위치 (B, K 등) 구체화 필요  |
| LoRA (High Imp.) | $\Delta W = AB$                     | 표현력 집중 강화        | Rank r 및 적용 대상 명확화 필요  |
| IA³ (Low Imp.)   | $z' = \alpha \cdot z$               | 초경량 파라미터 미세 조정   | 마스킹 연계 적용 권장           |

---

필요시 다음 문서를 추가로 작성할 수 있습니다:

* 🔬 **이론적 가설(Hypothesis): 세 전략 조합의 비선형 시너지 근거**
* 🔧 **구현 세부 파라미터 설정 표 (e.g., $r$, $\tau$, 중요도 스레숄드)**
* 📊 **Ablation Study 설계 및 결과 요약**
* 📈 **각 전략의 FLOPs / 파라미터 / 정확도 비교 시각화**

📘 **Adaptive Hybrid-PEFT Mamba 연구 산출물 계획서**

작성일: 2025. 6. 9
목표: 제안 아키텍처의 이론적 정당성, 구현 명세, 실험 검증 및 성능 우수성을 종합적으로 입증하기 위한 연구 산출물 체계 구축

---

## 1. 🔬 이론적 가설 정립 (Research Hypothesis)

### 🎯 핵심 가설:

**"Adaptive Hybrid-PEFT Mamba는 개별 최적화 기법을 결합하는 것 이상의 시너지를 창출하여, Accuracy–FLOPs–Params 트레이드오프 공간에서 비선형적인 효율성 개선을 달성한다."**

### 🔄 전략 간 상호작용에 기반한 세부 가설:

| 전략 조합                 | 기대 시너지                                                    |
| --------------------- | --------------------------------------------------------- |
| Scan + Masking        | 지역적으로 정렬된 경로 상에서 비-핵심 연산 제거 → 정보 손실 최소화 + FLOPs 감소        |
| Masking + Hybrid PEFT | 희소화된 중요 영역에만 집중 튜닝 → 파라미터 효율 극대화                          |
| Scan + Hybrid PEFT    | 빠른 경로 + selective 튜닝으로 latency와 학습 효율 동시 최적화              |
| All Combined          | 모델 구조 자체가 self-regularizing system으로 작동 → 고정된 구조보다 적응성 우수 |

---

## 2. ⚙️ 구현 및 하이퍼파라미터 설정 (Implementation & Parameters)

| 항목               | 기호       | 설명                 | 실험 설정 범위            |
| ---------------- | -------- | ------------------ | ------------------- |
| LoRA rank        | $r$      | Low-rank matrix 차원 | {4, 8, 16}          |
| Mask temperature | $\tau$   | Gumbel-Sigmoid 온도  | {0.3, 0.5, 0.8}     |
| 중요도 임계값          | $\theta$ | LoRA vs IA³ 선택 기준  | 상위 10%, 20%, 30%    |
| Scan path 계산 범위  | $d$      | 상태 차원 수            | 64–2048 (단위별 비교 실험) |
| 마스킹 비율           | –        | 전체 커널 중 유지 비율      | {0.3, 0.5, 0.7}     |

---

## 3. 🧪 Ablation Study 설계 및 실행 계획

### 📐 실험 그룹

| 실험 구성       | 전략 적용               | 설명                    |
| ----------- | ------------------- | --------------------- |
| Base Model  | –                   | Full Mamba (Baseline) |
| +Pillar 1   | Variable-Aware Scan | 연산 경로 최적화만 적용         |
| +Pillar 2   | + Learned Masking   | 동적 희소화 추가 적용          |
| +Pillar 3   | + Hybrid PEFT       | 경량 튜닝 결합 (LoRA + IA³) |
| All Pillars | Full Architecture   | 제안 아키텍처 전체 통합         |

### 🧾 평가 지표

* Perplexity (WikiText-2, PG-19)
* Summarization ROUGE (CNN/DM)
* QA Accuracy (HotpotQA)
* Code Pass\@1 (HumanEval)
* FLOPs, Params, 학습 시간, peak memory

---

## 4. 📊 결과 시각화 계획 (Visualization)

| 시각화 항목                              | 설명                              |
| ----------------------------------- | ------------------------------- |
| FLOPs vs. Accuracy Plot             | 각 전략 조합의 연산량 대비 정확도 변화 시각화      |
| Params vs. Accuracy Plot            | 학습 파라미터 수 대비 정확도 비교             |
| Efficiency Score Surface            | 3D: Accuracy / (FLOPs × Params) |
| Ablation Layer Contribution Heatmap | 각 레이어별 LoRA/IA³의 성능 기여도 시각화     |

---

이 계획서는 Adaptive Hybrid-PEFT Mamba 프로젝트의 핵심 연구 산출물 로드맵으로, 추후 논문 구조/실험 결과 보고서/시각화 기반 데모 설계에 활용됩니다.

👉 다음 작성 우선순위:

1. 논문 초안용 Introduction & Related Work
2. PyTorch 기반 모듈별 구현 설계도
3. 실험 자동화 스크립트 및 W\&B 통합

진행할 항목을 선택해주세요.

# 🧠 딥러닝 NLP 스터디

> Sequence Model → Language Model → Transformer → BERT → CLIP

| 주차 | 과제 | 핵심 기술 | 상태 |
|:---:|---|---|:---:|
| 1주차 | NSMC LSTM 감성 분석 | Keras · KoNLPy · Bidirectional LSTM | ✅ |
| 2주차 | Seq2Seq 번역 비교 실험 | PyTorch · GRU · Teacher Forcing | 🔜 |
| 3주차 | Transformer Attention 구현 | Scaled Dot-Product · Multi-Head | 🔜 |
| 4주차 | BERT Fine-Tuning | Hugging Face · klue/bert-base | 🔜 |
| 5주차 | CLIP Zero-shot 분류 | CLIP · ViT · Contrastive Learning | 🔜 |

---

## Week 1 · NSMC LSTM 감성 분석

**목표** : Bidirectional LSTM으로 네이버 영화 리뷰 감성 분류, Test Accuracy **85% 이상** 달성

| 항목 | 내용 |
|---|---|
| 데이터셋 | NSMC — 학습 150,000건 / 테스트 50,000건 |
| 주요 기술 | TensorFlow/Keras · KoNLPy (Okt) · NSMC |

### 🔧 전처리 파이프라인

| 단계 | 내용 |
|:---:|---|
| STEP 1 | 결측치 · 빈 문자열 제거 (`dropna` + 조건 필터링) |
| STEP 2 | 특수문자 제거 — 한글 · 공백만 유지 (`re.sub`) |
| STEP 3 | Okt 형태소 분석 — **Noun / Verb / Adjective** 만 추출 |
| STEP 4 | 토큰 1개 이하 문장 제거 |

조사·어미 등 기능어는 감성 정보를 거의 담지 않으므로 품사 필터링만으로 성능 향상 효과가 있었다.

### 🏗️ 모델 구조

```
Embedding(30000, 128, input_length=100)
    ↓
Bidirectional(LSTM(64, return_sequences=True))  +  Dropout(0.3)
    ↓
Bidirectional(LSTM(32))  +  Dropout(0.3)
    ↓
Dense(64, relu)  →  Dense(1, sigmoid)
```

- Bidirectional LSTM 2층으로 앞뒤 문맥을 모두 활용
- `return_sequences=True` 로 시퀀스 전체를 2층에 전달
- EarlyStopping · ReduceLROnPlateau · ModelCheckpoint 콜백 적용

### 📊 실험 결과

| 지표 | 값 |
|---|---|
| Test Loss | - |
| Test Accuracy | - |

**하이퍼파라미터 비교 실험**

| 구분 | MAX_LEN | LSTM | Dropout | Accuracy |
|:---:|:---:|---|:---:|:---:|
| 실험 A (메인) | 100 | 64 / 32 | 0.3 | - |
| 실험 B | 50 | 128 / 64 | 0.4 | - |

### 💬 고찰

단순 형태소 추출보다 **품사 필터링(Noun·Verb·Adjective)** 을 적용했을 때 어휘 사전이 정제되고 모델이 감성 관련 단어에 집중할 수 있었다. Bidirectional 구조는 단방향 대비 문맥 이해력이 높아 정확도 향상에 기여했다.

---

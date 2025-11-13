## EfficientNetB3 기반 패션 스타일 이미지 분류 회고록

#### 1. 프로젝트 개요

본 프로젝트는 6가지 패션 스타일 ```casual, chic, classic, girlish, retro, street```을 분류하기 위한 이미지 분류 모델을 구축한 실험 코드입니다.
EfficientNetB3를 기반으로 Transfer Learning 및 Fine-tuning을 적용했으며, 모델 경량화를 위해 TFLite를 진행했습니다.


#### 2. 모델 구조 및 학습 전략

#### 1️⃣ 데이터 및 데이터 증강

무신사 SMAP의 데이터 크롤링과 전처리 과정을 거친 후 총 6만장의 데이터로 학습을 진행했습니다.

이 과정에서 **ImageDataGenerator**를 활용해 데이터 증강 작업을 거쳐 훈련 데이터에 적용시켰습니다.
검증 데이터는 ```rescale=1./255``` 정규화(rescale) 만 적용시키기고 증강은 하지 않았고, 데이터 중 ImageDataGenerator의 classes 인자에 클래스명을 고정시켜, 학습/검증 단계에서 클래스 순서 불일치 문제를 방지했습니다.

#### 2️⃣ EfficientNetB3 기반 분류 모델 설계

EfficientNetB3 의 Feature Extractor를 기반으로 커스텀 분류기를 추가

Fully Connected Layer 구성:

```
GlobalAveragePooling2D → BatchNormalization →
Dense(512, ReLU) → Dropout(0.4) →
Dense(256, ReLU) → Dropout(0.3) →
Dense(128, ReLU) →
Dense(6, Softmax)

Optimizer: Adam (lr=1e-4)

Loss: categorical_crossentropy

Metrics: accuracy, top-2 accuracy
```



#### Training PipeLine

#### Phase 1. Transfer Learning(전이학습)
- ImageNet으로 사전 훈련된 가중치 ```weights='imagenet'``` 로드
  
- EfficientNet 백본 전체 동결
  
- 6개 클래스 분류를 위한 새로운 MLP 설계
  
- 높은 학습률(0.001)로 새로운 분류기 파라미터만 빠르게 학습
  
- 가장 높은 검증 정확도를 보인 시점의 가중치 저장


#### Phase 2. Fine-Tuning
- Phase 1의 최적 모델 로드
  
- EfficientNet 백본의 후반부 레이어 동결을 해제하여 학습 재개

- 초저 학습률을 적용하여 파인튜닝 진행

- Dropout 및 EarlyStopping(patience=7) 적용
  

#### Result
    
- ```chic, calssic, girlish ``` 특정 카테고리의 낮은 정확도
- Overfitting(과적합) 문제 발생

<p align="left">
  <img src="https://github.com/user-attachments/assets/836c1f5d-211e-4508-9424-53ad1e27f190" width="280" alt="Per-Class Accuracy Result"/>
</p>


### How to Next...
- YOLOs FashionPedia와 OpenAI의 CLIP모델 활용 예정...

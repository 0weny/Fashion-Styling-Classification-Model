"""Colab과 GoogleDrive 연결 필요"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import confusion_matrix, classification_report

print("=" * 70)
print("GPU Detection and Configuration (Colab Optimized)")
print("=" * 70)

# GPU 감지 및 설정
print("\n1. TensorFlow version:", tf.__version__)

print("\n2. Checking GPU availability...")
physical_devices = tf.config.list_physical_devices('GPU')
print(f"   Number of GPUs Available: {len(physical_devices)}")

if len(physical_devices) > 0:
    for i, device in enumerate(physical_devices):
        print(f"   GPU {i}: {device}")

    try:
        # GPU 메모리 증가 허용 (OOM 방지)
        for gpu in physical_devices:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("\n   ✓ GPU memory growth enabled (OOM prevention)")
    except RuntimeError as e:
        print(f"\n   ! Warning: {e}")

    # --- 불필요한 GPU 연산 테스트 부분 제거 ---
    # (GPU가 실제로 작동하는지는 model.fit() 단계에서 확인됩니다)

    # Mixed Precision 활성화 (T4/V100 GPU에서 2배 향상)
    try:
        from tensorflow.keras import mixed_precision
        policy = mixed_precision.Policy('mixed_float16')
        mixed_precision.set_global_policy(policy)
        print("\n3. Mixed Precision: ENABLED (2x faster training)")
    except Exception as e:
        print(f"\n3. Mixed Precision: DISABLED ({e})")
else:
    print("\n   ! No GPU found. Training will use CPU (very slow)")
    print("   Tip: Colab에서 [런타임] > [런타임 유형 변경] > [T4 GPU]를 선택하세요.")

print("=" * 70)

# Configuration
CONFIG = {
    # 'data_dir'를 삭제하고 아래 두 경로로 대체
    'train_dir': '/content/data/fashion_style_classifier/train',
    'val_dir': '/content/data/fashion_style_classifier/val',
    'img_size': (300, 300),
    'batch_size': 16,  # Reduce if GPU memory is insufficient
    'epochs': 40,
    'fine_tune_epochs': 25,
    'learning_rate': 0.001,
    'fine_tune_lr': 0.00005,
    'num_classes': 6,
    # 'girllish' -> 'girlish' (l 1개)로 수정
    'class_names': ['casual', 'chic', 'girlish', 'retro', 'street', 'classic'],
    'model_save_path': 'models/fashion_model_custom.h5',
    'tflite_save_path': 'models/fashion_model_custom.tflite'
}

# CLASS_NAMES를 CONFIG 밖에서 다시 정의해줍니다.
CLASS_NAMES = ['casual', 'chic', 'girlish', 'retro', 'street', 'classic']


# [기존 check_data_directory 함수를 삭제하고 이걸로 붙여넣으세요]

def check_data_directory():
    """Check data directories and count images"""
    train_dir = CONFIG['train_dir']
    val_dir = CONFIG['val_dir']

    print("\n" + "=" * 70)
    print("Data Directory Check")
    print("=" * 70)

    all_found = True
    # 1. Train/Val 기본 경로 확인
    if not os.path.exists(train_dir):
        print(f"ERROR: Train directory not found: {train_dir}")
        all_found = False
    else:
        print(f"✓ Found Train Dir: {train_dir}")

    if not os.path.exists(val_dir):
        print(f"ERROR: Validation directory not found: {val_dir}")
        all_found = False
    else:
        print(f"✓ Found Validation Dir: {val_dir}")

    if not all_found:
        print("\n! 압축 해제 경로가 올바른지 확인하세요.")
        print("! 이전 단계의 압축 해제 셀을 다시 실행해 주세요.")
        return False

    # 2. Train/Val 하위의 클래스 폴더 확인
    print("\nChecking class subfolders...")
    missing_classes = []
    for class_name in CLASS_NAMES:
        # train 폴더 내 클래스 확인
        if not os.path.exists(os.path.join(train_dir, class_name)):
            missing_classes.append(f"'{class_name}' in 'train' folder")

        # val 폴더 내 클래스 확인
        if not os.path.exists(os.path.join(val_dir, class_name)):
            missing_classes.append(f"'{class_name}' in 'val' folder")

    if missing_classes:
        print(f"\nERROR: Missing class folders:")
        for missing in missing_classes:
            print(f"  - {missing}")
        print("\n! CLASS_NAMES 리스트의 오타('girlish')를 확인하세요.")
        print("! 1단계의 'cihc' -> 'chic' 이름 변경 셀을 실행했는지 확인하세요.")
        return False

    print("✓ All class folders found in train and val dirs.")
    print("=" * 70)
    return True


def create_directories():
    """Create output directories"""
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    print("Output directories created: models/, results/")


def prepare_data_generators():
    """Prepare data generators with augmentation"""

    print("\n" + "=" * 70)
    print("Data Loading and Augmentation")
    print("=" * 70)
    print(f"Train Dir: {CONFIG['train_dir']}")
    print(f"Val Dir: {CONFIG['val_dir']}")

   # --- 학습 데이터용 Generator (Augmentation 적용) ---
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=30,
        width_shift_range=0.25,
        height_shift_range=0.25,
        shear_range=0.2,
        zoom_range=0.25,
        horizontal_flip=True,
        brightness_range=[0.7, 1.3],
        channel_shift_range=0.2,
        fill_mode='nearest'
        # validation_split=0.2 -> 삭제 (별도 val 폴더 사용)
    )

    train_generator = train_datagen.flow_from_directory(
        CONFIG['train_dir'],  # train 폴더 경로
        target_size=CONFIG['img_size'],
        batch_size=CONFIG['batch_size'],
        class_mode='categorical',
        # subset='training' -> 삭제
        shuffle=True,
        seed=42,
        classes=CLASS_NAMES # 클래스 순서 고정
    )


    # --- 검증 데이터용 Generator (Augmentation 미적용) ---
    # 검증 데이터는 절대 변형(augmentation)하면 안 됩니다.
    validation_datagen = ImageDataGenerator(
        rescale=1./255
    )

    validation_generator = validation_datagen.flow_from_directory(
        CONFIG['val_dir'],   # val 폴더 경로
        target_size=CONFIG['img_size'],
        batch_size=CONFIG['batch_size'],
        class_mode='categorical',
        # subset='validation' -> 삭제
        shuffle=False, # 평가지표 계산을 위해 순서 섞지 않음
        seed=42,
        classes=CLASS_NAMES # 클래스 순서 고정
    )

    print(f"\nTraining samples: {train_generator.samples}")
    print(f"Validation samples: {validation_generator.samples}")
    print(f"Batch size: {CONFIG['batch_size']}")
    print(f"Image size: {CONFIG['img_size']}")

    print("\nClass mapping (Train):")
    for class_name, idx in train_generator.class_indices.items():
        print(f"  {idx}. {class_name}")

    print("\nClass mapping (Validation):")
    for class_name, idx in validation_generator.class_indices.items():
        print(f"  {idx}. {class_name}")

    print("=" * 70)

    return train_generator, validation_generator


def build_efficientnet_model(num_classes=6):
    """Build EfficientNetB3 model"""

    print("\n" + "=" * 70)
    print("Model Architecture")
    print("=" * 70)
    print("Base: EfficientNetB3 (ImageNet pretrained)")

    base_model = EfficientNetB3(
        input_shape=(*CONFIG['img_size'], 3),
        include_top=False,
        weights='imagenet'
    )

    base_model.trainable = False

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.BatchNormalization(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.4),
        layers.BatchNormalization(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(num_classes, activation='softmax', dtype='float32')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=CONFIG['learning_rate']),
        loss='categorical_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.TopKCategoricalAccuracy(k=2, name='top_2_accuracy')
        ]
    )

    print(f"Total parameters: {model.count_params():,}")
    print("=" * 70)

    return model


def get_callbacks():
    """Training callbacks"""

    callbacks = [
        EarlyStopping(
            monitor='val_accuracy',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),

        ModelCheckpoint(
            CONFIG['model_save_path'],
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),

        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=4,
            min_lr=1e-7,
            verbose=1
        )
    ]

    return callbacks


def train_model(model, train_gen, val_gen):
    """Phase 1: Transfer Learning"""

    print("\n" + "=" * 70)
    print("Phase 1: Transfer Learning")
    print("=" * 70)
    print(f"Epochs: {CONFIG['epochs']}")
    print(f"Learning rate: {CONFIG['learning_rate']}")

    # Class weights for imbalanced data
    from sklearn.utils.class_weight import compute_class_weight

    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(train_gen.classes),
        y=train_gen.classes
    )
    class_weight_dict = dict(enumerate(class_weights))

    print("Class weights (for imbalanced data):")
    for idx, weight in class_weight_dict.items():
        print(f"  {CLASS_NAMES[idx]:12s}: {weight:.2f}")

    print("\nStarting training...\n")

    callbacks = get_callbacks()

    history = model.fit(
        train_gen,
        epochs=CONFIG['epochs'],
        validation_data=val_gen,
        callbacks=callbacks,
        class_weight=class_weight_dict,
        verbose=1
    )

    return history


def fine_tune_model(model, train_gen, val_gen):
    """Phase 2: Fine-tuning"""

    print("\n" + "=" * 70)
    print("Phase 2: Fine-tuning")
    print("=" * 70)

    base_model = model.layers[0]
    base_model.trainable = True

    # Keep first 200 layers frozen
    for layer in base_model.layers[:200]:
        layer.trainable = False

    trainable = sum([1 for layer in base_model.layers if layer.trainable])
    print(f"Trainable layers: {trainable}/{len(base_model.layers)}")
    print(f"Learning rate: {CONFIG['fine_tune_lr']} (very low)")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=CONFIG['fine_tune_lr']),
        loss='categorical_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.TopKCategoricalAccuracy(k=2, name='top_2_accuracy')
        ]
    )

    print("\nStarting fine-tuning...\n")

    callbacks = get_callbacks()

    history = model.fit(
        train_gen,
        epochs=CONFIG['fine_tune_epochs'],
        validation_data=val_gen,
        callbacks=callbacks,
        verbose=1
    )

    return history


def plot_training_history(history, filename='training_history.png'):
    """Plot training history"""

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Accuracy
    axes[0].plot(history.history['accuracy'], label='Train', linewidth=2)
    axes[0].plot(history.history['val_accuracy'], label='Validation', linewidth=2)
    axes[0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Loss
    axes[1].plot(history.history['loss'], label='Train', linewidth=2)
    axes[1].plot(history.history['val_loss'], label='Validation', linewidth=2)
    axes[1].set_title('Model Loss', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'results/{filename}', dpi=300, bbox_inches='tight')
    print(f"Saved: results/{filename}")
    plt.show()


def evaluate_model(model, val_gen):
    """Evaluate model and generate confusion matrix"""

    print("\n" + "=" * 70)
    print("Model Evaluation")
    print("=" * 70)

    # Predictions
    val_gen.reset()
    predictions = model.predict(val_gen, verbose=1)
    y_pred = np.argmax(predictions, axis=1)
    y_true = val_gen.classes

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Plot
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        cbar_kws={'label': 'Count'}
    )
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('results/confusion_matrix.png', dpi=300, bbox_inches='tight')
    print("Saved: results/confusion_matrix.png")
    plt.show()

    # Classification report
    print("\nClassification Report:")
    report = classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4)
    print(report)

    # Save report
    with open('results/classification_report.txt', 'w', encoding='utf-8') as f:
        f.write("Fashion Style Classification Model - Evaluation\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Classes: {', '.join(CLASS_NAMES)}\n\n")
        f.write(report)

    # Per-class accuracy
    class_accuracy = cm.diagonal() / cm.sum(axis=1)
    print("\nPer-Class Accuracy:")
    for name, acc in zip(CLASS_NAMES, class_accuracy):
        print(f"  {name:12s}: {acc:.2%}")

    # Overall accuracy
    overall_accuracy = np.sum(cm.diagonal()) / np.sum(cm)
    print(f"\nOverall Accuracy: {overall_accuracy:.2%}")
    print("=" * 70)

    return overall_accuracy


def convert_to_tflite(model):
    """Convert to TensorFlow Lite"""

    print("\n" + "=" * 70)
    print("TFLite Conversion")
    print("=" * 70)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]

    tflite_model = converter.convert()

    with open(CONFIG['tflite_save_path'], 'wb') as f:
        f.write(tflite_model)

    h5_size = os.path.getsize(CONFIG['model_save_path']) / (1024 * 1024)
    tflite_size = os.path.getsize(CONFIG['tflite_save_path']) / (1024 * 1024)

    print(f"H5 model: {h5_size:.2f} MB")
    print(f"TFLite model: {tflite_size:.2f} MB")
    print(f"Compression: {(1 - tflite_size/h5_size)*100:.1f}%")
    print("=" * 70)


def save_model_info(accuracy):
    """Save model information"""

    info = f"""Fashion Style Classification Model - Results
{'=' * 70}

Model: EfficientNetB3
Final Accuracy: {accuracy:.2%}

Classes (6):
{chr(10).join([f'  - {name}' for name in CLASS_NAMES])}

Hyperparameters:
  - Image size: {CONFIG['img_size']}
  - Batch size: {CONFIG['batch_size']}
  - Initial epochs: {CONFIG['epochs']}
  - Fine-tuning epochs: {CONFIG['fine_tune_epochs']}
  - Initial learning rate: {CONFIG['learning_rate']}
  - Fine-tuning learning rate: {CONFIG['fine_tune_lr']}

Saved files:
  - H5 model: {CONFIG['model_save_path']}
  - TFLite model: {CONFIG['tflite_save_path']}
  - Confusion matrix: results/confusion_matrix.png
  - Classification report: results/classification_report.txt
"""

    with open('results/model_info.txt', 'w', encoding='utf-8') as f:
        f.write(info)

    print("Saved: results/model_info.txt")


def main():
    """Main training pipeline"""

    print("\n" + "=" * 70)
    print("Fashion Style Classifier - Training Pipeline")
    print("Classes: casual, chic, girllish, retro, street, classic")
    print("=" * 70)

    # 1. Check data
    if not check_data_directory():
        print("\nERROR: Data directory check failed")
        return

    # 2. Create directories
    create_directories()

    # 3. Prepare data
    train_gen, val_gen = prepare_data_generators()

    # 4. Build model
    model = build_efficientnet_model(num_classes=CONFIG['num_classes'])

    # 5. Phase 1: Transfer learning
    history = train_model(model, train_gen, val_gen)
    plot_training_history(history, 'phase1_transfer_learning.png')

    # 6. Phase 2: Fine-tuning
    history_fine = fine_tune_model(model, train_gen, val_gen)
    plot_training_history(history_fine, 'phase2_fine_tuning.png')

    # 7. Load best model and evaluate
    print("\nLoading best model...")
    model = tf.keras.models.load_model(CONFIG['model_save_path'])
    accuracy = evaluate_model(model, val_gen)

    # 8. Convert to TFLite
    convert_to_tflite(model)

    # 9. Save info
    save_model_info(accuracy)

    print("\n" + "=" * 70)
    print("Training Complete!")
    print("=" * 70)
    print(f"Final Accuracy: {accuracy:.2%}")
    print(f"H5 model: {CONFIG['model_save_path']}")
    print(f"TFLite model: {CONFIG['tflite_save_path']}")
    print(f"Results: results/")
    print("=" * 70)


if __name__ == '__main__':
    main()

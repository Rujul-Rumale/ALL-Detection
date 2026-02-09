
import tensorflow as tf
from tensorflow.keras import layers, models, applications

def build_mobilenetv2(input_shape=(224, 224, 3), alpha=1.0, dropout_rate=0.2):
    """
    Builds a MobileNetV2 model for binary classification (ALL vs HEM).
    
    Args:
        input_shape: Tuple of input dimensions (H, W, C).
        alpha: Width multiplier for MobileNetV2 (1.0 is default).
        dropout_rate: Dropout rate for the classification head.
        
    Returns:
        A compiled Keras model.
    """
    
    # 1. Base Model (Pre-trained on ImageNet, but we'll retrain or fine-tune)
    # Note: 'weights=None' because we're training on a very specific domain (cells)
    # and we want to learn domain-specific features from scratch or fine-tune carefully.
    # For this robust pipeline, we'll start with ImageNet weights to speed up convergence.
    base_model = applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet',
        alpha=alpha
    )
    
    # 2. Freeze base model initially (optional, but good practice)
    # We will unfreeze it later or if we have enough data. 
    # For C-NMC (~10k images), fine-tuning is safe.
    base_model.trainable = True 
    
    # 3. Classification Head
    inputs = layers.Input(shape=input_shape)
    
    # Ensure correct preprocessing is part of the model graph if not done externally
    # But our pipeline does preprocessing. 
    # We'll assume inputs are already [0,1] or standardized.
    
    x = base_model(inputs)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout_rate)(x)
    
    # Binary output: 1 sigmoid unit
    outputs = layers.Dense(1, activation='sigmoid', name='prediction')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs, name="LeukemiaNet_V2")
    
    # 4. Compile
    # Using Recall as primary metric as per spec
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.AUC(name='auc')
        ]
    )
    
    return model

if __name__ == "__main__":
    model = build_mobilenetv2()
    model.summary()

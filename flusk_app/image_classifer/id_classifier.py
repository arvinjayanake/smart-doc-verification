import os
from typing import List, Dict, Optional, Union

import numpy as np
import tensorflow as tf

def _model_has_rescaling_layer(model: tf.keras.Model) -> bool:
    return any(isinstance(l, tf.keras.layers.Rescaling) for l in model.layers)

def _ensure_single_tensor(x):
    # If x is a list/tuple of tensors, keep only the first
    if isinstance(x, (list, tuple)):
        return x[0]
    return x

def _ensure_dense_ready(x: tf.Tensor) -> tf.Tensor:
    # If x is 4D (e.g., [B, H, W, C]), reduce spatial dims for Dense
    if len(x.shape) == 4:
        return tf.keras.layers.GlobalAveragePooling2D(name="auto_gap_for_dense")(x)
    return x

def _clone_layer_with_weights(layer: tf.keras.layers.Layer, new_input: tf.Tensor) -> tf.Tensor:
    """Clone a layer (same config + weights) and call it on new_input."""
    LayerClass = layer.__class__
    new_layer = LayerClass.from_config(layer.get_config())
    # Build to create weights with the correct shape, then set weights.
    # Some layers build on first call; if so, skip explicit build.
    try:
        new_layer.build(new_input.shape)
    except Exception:
        pass
    _ = new_layer(new_input)  # ensure weights are created
    new_layer.set_weights(layer.get_weights())
    return new_layer(new_input)

def patch_model_single_input_dense(model: tf.keras.Model,
                                   dense_name: Optional[str] = None) -> tf.keras.Model:
    """
    Rebuilds the tail so that a Dense receives EXACTLY one tensor.
    - If dense_name is provided, patch that specific layer.
    - Otherwise, tries to locate the last Dense in the model.

    Returns a new Keras Model with the same inputs and corrected outputs.
    """
    # Pick target Dense layer
    target_dense = None
    if dense_name:
        target_dense = model.get_layer(dense_name)
        if not isinstance(target_dense, tf.keras.layers.Dense):
            raise ValueError(f"Layer '{dense_name}' is not a Dense layer.")
    else:
        # Find last Dense in topological order
        for l in reversed(model.layers):
            if isinstance(l, tf.keras.layers.Dense):
                target_dense = l
                break
        if target_dense is None:
            raise ValueError("No Dense layer found to patch.")

    prev = target_dense.input  # could be a KerasTensor OR list/tuple
    prev = _ensure_single_tensor(prev)
    prev = _ensure_dense_ready(prev)

    # Recreate the Dense on the fixed input
    logits = _clone_layer_with_weights(target_dense, prev)

    # If the original Dense was not the final output (i.e., was followed by more layers),
    # rebuild that tail too by cloning the remaining layers in order.
    # Find the path from target_dense to original outputs (assume linear tail).
    # If the Dense is already the last layer, we just return logits.

    # Quick heuristic: if model.outputs[0] depends on target_dense.output, clone subsequent layers.
    # Otherwise, we just expose logits.
    def _depends_on(tensor, target_layer_output):
        try:
            # KerasTensor keeps history; string compare in graph path is brittle, but works in practice.
            return target_layer_output in tensor._keras_history.layer.output if hasattr(tensor, "_keras_history") else False
        except Exception:
            return False

    # If target_dense is not the penultimate/last, try to reproduce any post-processing layers:
    rebuilt_output = logits
    chain_started = False
    for layer in model.layers:
        if not chain_started:
            if layer is target_dense:
                chain_started = True
            continue
        # For the rest of the layers after target_dense, only keep layers that
        # take the previous layer’s output as sole input (linear chain).
        if isinstance(layer.input, (list, tuple)):
            # If next layer expects multiple inputs, stop cloning (cannot infer safely).
            break
        if hasattr(layer.input, "name") and hasattr(rebuilt_output, "name"):
            if layer.input.name.split(":")[0] != rebuilt_output.name.split(":")[0]:
                # The layer is not connected linearly to our rebuilt_output
                break
        # Clone this layer onto rebuilt_output
        rebuilt_output = _clone_layer_with_weights(layer, rebuilt_output)

    patched = tf.keras.Model(inputs=model.inputs, outputs=rebuilt_output, name=model.name + "_patched")
    return patched


class IDClassifier:
    """
    Loads a Keras (.keras or SavedModel) image-classification model,
    auto-patches a miswired Dense tail if necessary, and predicts classes.
    """

    def __init__(
        self,
        model_or_path: Union[str, tf.keras.Model],
        class_names: List[str],
        dense_to_patch: Optional[str] = None,
        auto_patch: bool = True
    ):
        """
        Args:
            model_or_path: Path to a .keras file / SavedModel dir OR an already-loaded tf.keras.Model.
            class_names: Class names in the exact training order.
            dense_to_patch: Name of the Dense layer to patch (if known). If None, picks the last Dense.
            auto_patch: If True, try to fix "Dense expects 1 input, got 2" graphs automatically.
        """
        if isinstance(model_or_path, tf.keras.Model):
            model = model_or_path
        else:
            if not os.path.exists(model_or_path):
                raise FileNotFoundError(f"Model file/folder not found at: {model_or_path}")
            model = tf.keras.models.load_model(model_or_path)

        # Try a dry run to detect the multi-input Dense error quickly.
        self.class_names = class_names
        self.model = model
        self._has_rescaling = _model_has_rescaling_layer(self.model)

        # Infer input shape
        input_shape = self.model.input_shape
        if isinstance(input_shape, list):
            input_shape = input_shape[0]
        if len(input_shape) != 4:
            raise ValueError(f"Unexpected input shape {input_shape}. Expected (None, H, W, C).")

        self.image_size = (int(input_shape[1]), int(input_shape[2]))
        self.channels = int(input_shape[3])

        # Attempt to detect the error by making a dummy forward pass (small zero tensor).
        if auto_patch:
            try:
                dummy = tf.zeros((1, self.image_size[0], self.image_size[1], self.channels), dtype=tf.float32)
                _ = self.model(dummy, training=False)
            except Exception as e:
                msg = str(e)
                if "expects 1 input" in msg and "but it received 2 input tensors" in msg:
                    # Patch and replace
                    self.model = patch_model_single_input_dense(self.model, dense_name=dense_to_patch)
                else:
                    # Other errors should be raised for visibility
                    raise

    def _preprocess_image(self, img_path: str) -> tf.Tensor:
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image file not found at: {img_path}")

        img = tf.keras.utils.load_img(
            img_path,
            target_size=self.image_size,
            color_mode="rgb" if self.channels >= 3 else "grayscale",
        )
        x = tf.keras.utils.img_to_array(img)  # [H, W, C]

        # Avoid double-normalization if the model already contains a Rescaling layer
        if not self._has_rescaling:
            x = x / 255.0

        # Ensure channels match
        if self.channels == 1 and x.shape[-1] != 1:
            x = tf.reduce_mean(x, axis=-1, keepdims=True)

        x = tf.expand_dims(x, axis=0)  # [1, H, W, C]
        return tf.convert_to_tensor(x)

    def predict(self, img_path: str, confidence_threshold: float = 0.86) -> Dict:
        x = self._preprocess_image(img_path)
        preds = self.model.predict(x, verbose=0)

        # Ensure 1D vector
        probs = preds[0]
        s = float(np.sum(probs))
        if not (0.98 <= s <= 1.02):
            probs = tf.nn.softmax(probs).numpy()

        idx = int(np.argmax(probs))
        conf = float(np.max(probs))

        if conf < confidence_threshold:
            return {
                "class_name": "Unknown",
                "confidence": float(f"{conf * 100:.2f}"),
                "is_known": False,
            }

        name = self.class_names[idx] if idx < len(self.class_names) else f"class_{idx}"
        return {
            "class_name": name,
            "confidence": float(f"{conf * 100:.2f}"),
            "is_known": True,
        }

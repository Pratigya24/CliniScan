
import torch
import numpy as np
import cv2
from torchvision.models.feature_extraction import create_feature_extractor


def generate_gradcam(model, img_tensor, layer_name="features.8"):
    """
    Generates a Grad-CAM style heatmap for a classification model.
    layer_name is the last feature block of EfficientNet-B0.
    """
    model.eval()
    feature_extractor = create_feature_extractor(
        model,
        {layer_name: "feat"}
    )

    with torch.no_grad():
        out = feature_extractor(img_tensor.unsqueeze(0))
        preds = model(img_tensor.unsqueeze(0))

    feat_map = out["feat"].squeeze().detach().mean(dim=0).numpy()
    heatmap = cv2.resize(feat_map, (224, 224))
    heatmap = np.maximum(heatmap, 0)
    heatmap = heatmap / (np.max(heatmap) + 1e-8)

    return heatmap, preds


def overlay_heatmap(original_img_np, heatmap, alpha=0.4):
    """
    Overlays a heatmap on top of the original image.
    """
    heatmap_colored = cv2.applyColorMap(
        np.uint8(255 * heatmap),
        cv2.COLORMAP_JET
    )
    overlay = cv2.addWeighted(
        original_img_np,
        1 - alpha,
        heatmap_colored,
        alpha,
        0
    )
    return overlay

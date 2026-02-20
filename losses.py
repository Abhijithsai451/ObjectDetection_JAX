import jax
import optax
import jax.numpy as jnp

def compute_loss(model_type, preds, targets):
    if model_type == "DETR":
        return hungarian_loss(preds, targets)
    else:
        return focal_box_loss(preds, targets)
    
def hungarian_loss(preds, targets):
    loss = jnp.mean(jnp.square(preds["logits"] - targets["boxes"]))


def focal_loss(logits, targets, alpha=0.25, gamma=2.0):
    """
    Pure JAX implementation of Sigmoid Focal Loss.
    Args:
        logits: [batch, num_anchors, num_classes] raw model outputs.
        targets: [batch, num_anchors, num_classes] one-hot labels.
    """
    # Use sigmoid_cross_entropy as the base
    # p = sigmoid(logits)
    p = jax.nn.sigmoid(logits)

    # Standard Binary Cross Entropy
    bce_loss = -targets * jnp.log(p + 1e-8) - (1 - targets) * jnp.log(1 - p + 1e-8)

    # Modulating factor: (1-p)^gamma for positive, p^gamma for negative
    p_t = jnp.where(targets == 1, p, 1 - p)
    modulating_factor = jnp.power(1.0 - p_t, gamma)

    # Alpha balancing (optional but recommended)
    alpha_weight = jnp.where(targets == 1, alpha, 1 - alpha)

    return jnp.mean(alpha_weight * modulating_factor * bce_loss)


def iou_loss(pred_boxes, target_boxes):
    """
    Computes IoU loss between predicted and target boxes.
    Boxes are in [y1, x1, y2, x2] format.
    """
    # 1. Calculate intersection coordinates
    y1 = jnp.maximum(pred_boxes[..., 0], target_boxes[..., 0])
    x1 = jnp.maximum(pred_boxes[..., 1], target_boxes[..., 1])
    y2 = jnp.minimum(pred_boxes[..., 2], target_boxes[..., 2])
    x2 = jnp.minimum(pred_boxes[..., 3], target_boxes[..., 3])

    # 2. Intersection and Union areas
    inter_area = jnp.maximum(0, y2 - y1) * jnp.maximum(0, x2 - x1)

    area_pred = (pred_boxes[..., 2] - pred_boxes[..., 0]) * \
                (pred_boxes[..., 3] - pred_boxes[..., 1])
    area_target = (target_boxes[..., 2] - target_boxes[..., 0]) * \
                  (target_boxes[..., 3] - target_boxes[..., 1])

    union_area = area_pred + area_target - inter_area
    iou = inter_area / (union_area + 1e-8)

    # 3. Return 1 - IoU as the loss
    return jnp.mean(1.0 - iou)

def focal_box_loss(preds, batch_targets):
    # 1. Classification Loss (on all anchors)
    cls_loss = focal_loss(preds['logits'], batch_targets['cls'])

    # 2. Regression Loss (only on foreground/positive anchors)
    # Masking ensures we don't try to regress 'background' boxes
    pos_mask = batch_targets['cls_labels'] > 0
    reg_loss = iou_loss(preds['boxes'] * pos_mask, batch_targets['boxes'] * pos_mask)

    # EfficientDet weight balancing (e.g., 50.0 for regression)
    return cls_loss + 50.0 * reg_loss
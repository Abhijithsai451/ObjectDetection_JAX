import jax
import jax.numpy as jnp
from flax import nnx
class ModelFactory:
    @staticmethod
    def create(model_type, num_classes, rngs):
        if model_type == "DETR":
            return DETR(num_classes, num_queries = 100, rngs = rngs)
        elif model_type == "EfficientDet":
            return EfficientDet(num_classes, rngs = rngs)
        raise ValueError(f"Unknown Model type {model_type}")


class DETR(nnx.Module):
    def __init__(self, num_classes, num_queries, rngs ):
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.queries = nnx.Param(jax.random.normal(rngs.params(), (num_queries, 256)))
        self.backbone = nnx.Linear(3, 256, rngs= rngs)
        self.transformer = nnx.Linear(256,256, rngs= rngs)
        self.class_head = nnx.Linear(256, num_classes +1 , rngs= rngs)
        self.bbox_head = nnx.Linear(256, 4, rngs= rngs)

    def __call__(self, x):
        features = self.backbone(x)
        out = self.transformer(features + self.queries)
        return {"logits":self.class_head(out), "boxes":self.bbox_head(out)}


class EfficientDet(nnx.Module):
    def __init__(self, num_classes, rngs):
        self.conv = nnx.Conv(3, 64, (3,3), rngs = rngs)
        self.class_head = nnx.Linear(64, num_classes , rngs= rngs)
        self.bbox_head = nnx.Linear(64, 4 , rngs= rngs)

    def __call__(self, x):
        x = jnp.mean(self.conv(x), axis = (1,2) )
        return {"logits": self.class_head(x), "boxes": self.bbox_head(x)}
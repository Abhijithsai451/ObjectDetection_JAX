import jax
import optax

from losses import compute_loss


def train_one_step(model, optimizer, batch, model_type):
    def loss_fn(model):
        preds = model(batch['image'])
        return compute_loss(model_type, preds, batch['label'])

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    loss, grads = grad_fn(model)

    optimizer.update(grads)
    return loss

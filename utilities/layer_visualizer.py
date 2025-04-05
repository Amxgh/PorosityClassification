import tensorflow.keras as keras
import tensorflow as tf

# ------------------------------------------------------------
# 1. Load the trained model (skip if you already have `model`)
# ------------------------------------------------------------
model = keras.models.load_model("models/best_model_simplified")   # or whatever you saved it as
model.summary()                                                     # confirm layer names

# ------------------------------------------------------------
# 2. Pick the layer you want to inspect
#    – first Conv2D is usually "conv2d", second "conv2d_1", etc.
# ------------------------------------------------------------
layer_name  = "conv2d"              # change to "conv2d_1", "conv2d_2", … to probe deeper
target_layer = model.get_layer(name=layer_name)
truncated_model = keras.Model(inputs=model.inputs,
                              outputs=target_layer.output)

# ------------------------------------------------------------
# 3. Hyper‑parameters for visualisation
# ------------------------------------------------------------
IMG_H, IMG_W, _ = model.input_shape[1:]        # (128,128,1)
N_FILTERS       = target_layer.filters         # 32 for the first conv layer
STEPS           = 30                           # gradient‑ascent iterations
STEP_SIZE       = 10.0                         # learning‑rate for ascent
MARGIN          = 5                            # px between thumbnails

# ------------------------------------------------------------
# 4. Helper functions
# ------------------------------------------------------------
@tf.function
def ascent_step(img, f_idx):
    with tf.GradientTape() as tape:
        tape.watch(img)
        activation = truncated_model(img)
        # ignore border artefacts
        act_crop  = activation[:, 2:-2, 2:-2, f_idx]
        loss      = tf.reduce_mean(act_crop)

    grads = tape.gradient(loss, img)
    grads = tf.math.l2_normalize(grads)
    img.assign_add(STEP_SIZE * grads)
    return loss

def init_image():
    """Start from small random noise centred at 0."""
    return tf.Variable(tf.random.uniform((1, IMG_H, IMG_W, 1),
                                         minval=-.25, maxval=.25))

def deprocess(img):
    """Convert a single‑channel float32 tensor to an 8‑bit RGB array for display."""
    img -= img.mean()
    img /= (img.std() + 1e-5)
    img *= .15
    img += .5
    img = tf.clip_by_value(img, 0, 1)
    img = tf.image.grayscale_to_rgb(img)       # -> (H,W,3)
    return tf.cast(img * 255, tf.uint8).numpy()

def visualise_filter(f_idx):
    img = init_image()
    for _ in range(STEPS):
        ascent_step(img, f_idx)
    return deprocess(img[0])

# ------------------------------------------------------------
# 5. Generate thumbnails for the first N_FILTERS filters
# ------------------------------------------------------------
thumbnails = [visualise_filter(i) for i in range(N_FILTERS)]

# ------------------------------------------------------------
# 6. Stitch into a grid and save/show
# ------------------------------------------------------------
import math, numpy as np, keras
from IPython.display import display, Image as IPImage

n = int(math.ceil(math.sqrt(N_FILTERS)))       # e.g. 6×6 if 32 filters
thumb_h, thumb_w = thumbnails[0].shape[:2]
canvas_h = n * thumb_h + (n-1) * MARGIN
canvas_w = n * thumb_w + (n-1) * MARGIN
canvas   = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

for idx, thumb in enumerate(thumbnails):
    i, j = divmod(idx, n)
    y0 = i * (thumb_h + MARGIN)
    x0 = j * (thumb_w + MARGIN)
    canvas[y0:y0+thumb_h, x0:x0+thumb_w, :] = thumb

keras.preprocessing.image.save_img("filters_%s.png" % layer_name, canvas)
display(IPImage("filters_%s.png" % layer_name))

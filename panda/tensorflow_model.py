"""Current Keras equivalent of Xie29 EfficientNet-B3, GeM and provider-specific loss."""
import tensorflow as tf


@tf.keras.utils.register_keras_serializable(package='panda')
class GeM(tf.keras.layers.Layer):
    def build(self, input_shape):
        self.p = self.add_weight(name='p', shape=(), initializer=tf.keras.initializers.Constant(3.0))

    def call(self, inputs):
        x = tf.cast(inputs, tf.float32)
        p = tf.clip_by_value(self.p, 0.1, 10.0)
        return tf.pow(tf.reduce_mean(tf.pow(tf.abs(x)+1e-6, p), axis=(1, 2))+1e-6, 1/p)


@tf.keras.utils.register_keras_serializable(package='panda')
def provider_loss(y_true, y_pred):
    grade, radboud = y_true[:, 0], y_true[:, 1]
    difference = tf.cast(tf.reshape(y_pred, [-1]), tf.float32) - tf.cast(grade, tf.float32)
    absolute = tf.abs(difference)
    huber = tf.where(absolute < 1, 0.5*tf.square(difference), absolute-0.5)
    return tf.where(radboud > 0, huber, tf.square(difference))


def build_model(image_size, pretrained=True, tiny=False):
    inputs = tf.keras.Input((image_size, image_size, 3))
    if tiny:
        x = tf.keras.layers.Rescaling(1/255)(inputs)
        x = tf.keras.layers.Conv2D(8, 3, strides=4, activation='relu')(x)
    else:
        # Current Keras EfficientNet includes input rescaling and accepts [0,255].
        base = tf.keras.applications.EfficientNetB3(include_top=False,
                                                   weights='imagenet' if pretrained else None,
                                                   input_shape=(image_size, image_size, 3))
        x = base(inputs)
    x = GeM()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(128, activation='relu', kernel_initializer='he_normal')(x)
    outputs = tf.keras.layers.Dense(1, dtype='float32')(x)
    return tf.keras.Model(inputs, outputs)


def parse_example(serialized, image_size, labeled=True):
    value = tf.io.parse_single_example(serialized, {
        'image': tf.io.FixedLenFeature([], tf.string), 'id': tf.io.FixedLenFeature([], tf.string),
        'isup_grade': tf.io.FixedLenFeature([], tf.int64), 'data_provider': tf.io.FixedLenFeature([], tf.int64)})
    image = tf.cast(tf.io.decode_jpeg(value['image'], channels=3), tf.float32)
    image = tf.ensure_shape(image, (image_size, image_size, 3))
    label = tf.cast(tf.stack([value['isup_grade'], value['data_provider']]), tf.float32)
    return image, label


def dataset(paths, image_size, batch_size, training=False, seed=329, tpu=False):
    # Fixed read/map/prefetch buffers; no cache() of the full dataset.
    ds = tf.data.TFRecordDataset(paths, num_parallel_reads=1)
    if training:
        ds = ds.shuffle(32, seed=seed, reshuffle_each_iteration=True)
    ds = ds.map(lambda value: parse_example(value, image_size), num_parallel_calls=1)
    if training:
        def augment(image, label):
            image = tf.image.random_flip_left_right(image)
            image = tf.image.random_flip_up_down(image)
            image = tf.image.rot90(image, tf.random.uniform([], 0, 4, dtype=tf.int32))
            image = tf.image.random_brightness(image, 12.0)
            return tf.clip_by_value(image, 0, 255), label
        ds = ds.map(augment, num_parallel_calls=1)
    # Keep all validation rows, including the last partial batch.
    ds = ds.batch(batch_size, drop_remainder=training and tpu)
    options = tf.data.Options()
    options.threading.private_threadpool_size = 2
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
    return ds.with_options(options).prefetch(1)

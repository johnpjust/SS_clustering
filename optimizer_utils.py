import tensorflow as tf


def get_train_steps(num_examples):
  """Determine the number of training steps."""
  return FLAGS.train_steps or (
      num_examples * FLAGS.train_epochs // FLAGS.train_batch_size + 1)


def learning_rate_schedule(base_learning_rate, num_examples):
  """Build learning rate schedule."""
  global_step = tf.train.get_or_create_global_step()
  warmup_steps = int(round(
      FLAGS.warmup_epochs * num_examples // FLAGS.train_batch_size))
  if FLAGS.learning_rate_scaling == 'linear':
    scaled_lr = base_learning_rate * FLAGS.train_batch_size / 256.
  elif FLAGS.learning_rate_scaling == 'sqrt':
    scaled_lr = base_learning_rate * math.sqrt(FLAGS.train_batch_size)
  else:
    raise ValueError('Unknown learning rate scaling {}'.format(
        FLAGS.learning_rate_scaling))
  learning_rate = (tf.to_float(global_step) / int(warmup_steps) * scaled_lr
                   if warmup_steps else scaled_lr)

  # Cosine decay learning rate schedule
  total_steps = get_train_steps(num_examples)
  learning_rate = tf.where(
      global_step < warmup_steps, learning_rate,
      tf.train.cosine_decay(
          scaled_lr,
          global_step - warmup_steps,
          total_steps - warmup_steps))

  return learning_rate


def get_optimizer(learning_rate):
  """Returns an optimizer."""
  if FLAGS.optimizer == 'momentum':
    optimizer = tf.train.MomentumOptimizer(
        learning_rate, FLAGS.momentum, use_nesterov=True)
  elif FLAGS.optimizer == 'adam':
    optimizer = tf.train.AdamOptimizer(
        learning_rate)
  elif FLAGS.optimizer == 'lars':
    optimizer = LARSOptimizer(
        learning_rate,
        momentum=FLAGS.momentum,
        weight_decay=FLAGS.weight_decay,
        exclude_from_weight_decay=['batch_normalization', 'bias',
                                   'head_supervised'])
  else:
    raise ValueError('Unknown optimizer {}'.format(FLAGS.optimizer))

  if FLAGS.use_tpu:
    optimizer = tf.tpu.CrossShardOptimizer(optimizer)
  return optimizer

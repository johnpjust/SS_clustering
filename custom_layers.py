import tensorflow as tf
from tensorflow import keras
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.framework import ops
from tensorflow.python.keras import backend as K
from tensorflow.python.ops import nn
from tensorflow.python.ops import state_ops

from tensorflow.python.keras.layers import normalization
import tensorflow as tf

class batch_norm_agg(normalization.BatchNormalizationBase):

    bn_state = 0
    bn_update_cntr = 0
    agg_mean = 0
    agg_var = 0

    def _assign_moving_average(self, variable, value, momentum, inputs_size):
        with K.name_scope('AssignMovingAvg') as scope:
            with ops.colocate_with(variable):
                # decay = ops.convert_to_tensor(1.0 - momentum, name='decay')
                # if decay.dtype != variable.dtype.base_dtype:
                #     decay = math_ops.cast(decay, variable.dtype.base_dtype)
                if 'moving_mean' in variable.name:
                    if self.bn_state == 1:
                        self.agg_mean += value
                    elif self.bn_state == 2:
                        if self.bn_update_cntr > 0:
                            self.agg_mean = self.agg_mean/self.bn_update_cntr
                            update_delta = math_ops.cast(self.agg_mean, variable.dtype)
                            if inputs_size is not None:
                                update_delta = array_ops.where(inputs_size > 0, update_delta, K.zeros_like(update_delta))
                            return state_ops.assign(variable, update_delta, name=scope)
                elif 'moving_variance' in variable.name:
                    if self.bn_state == 1:
                        self.agg_var += value
                        self.bn_update_cntr += 1
                    elif self.bn_state == 2:
                        if self.bn_update_cntr > 0:
                            self.agg_var = self.agg_var/self.bn_update_cntr
                            update_delta = math_ops.cast(self.agg_var, variable.dtype)
                            if inputs_size is not None:
                                update_delta = array_ops.where(inputs_size > 0, update_delta, K.zeros_like(update_delta))
                            return state_ops.assign(variable, update_delta, name=scope)

                return state_ops.assign(variable, variable, name=scope)


    def call(self, inputs, training=None):
        if training is None:
            training_value = None
            training = self._get_training_value(training)
        else:
            training = self._get_training_value(training)
            training_value = tf_utils.constant_value(training)

        if training_value == False:
            self.bn_state = 0
            self.bn_update_cntr = 0
            self.agg_mean = K.zeros_like(self.moving_mean)
            self.agg_var = K.zeros_like(self.moving_variance)
        elif self.bn_state > 0 and training_value == None:  ## inference using aggregated parameters
            self.bn_state = 2
            if self._support_zero_size_input():
                inputs_size = array_ops.size(inputs)
            else:
                inputs_size = None
            self._assign_moving_average(self.moving_mean, 0, 0, inputs_size)
            self._assign_moving_average(self.moving_variance, 0, 0, inputs_size)
            self.bn_state = 0
            self.bn_update_cntr = 0
            self.agg_mean = K.zeros_like(self.moving_mean)
            self.agg_var = K.zeros_like(self.moving_variance)
        elif self.bn_state == 0 and training_value == True:  ## act as normal training mode
            self.bn_state = 1
            self.bn_update_cntr = 0
            self.agg_mean = K.zeros_like(self.moving_mean)
            self.agg_var = K.zeros_like(self.moving_variance)

        if self.virtual_batch_size is not None:
            # Virtual batches (aka ghost batches) can be simulated by reshaping the
            # Tensor and reusing the existing batch norm implementation
            original_shape = [-1] + inputs.shape.as_list()[1:]
            expanded_shape = [self.virtual_batch_size, -1] + original_shape[1:]

            # Will cause errors if virtual_batch_size does not divide the batch size
            inputs = array_ops.reshape(inputs, expanded_shape)

            def undo_virtual_batching(outputs):
                outputs = array_ops.reshape(outputs, original_shape)
                return outputs

        if self.fused:
            outputs = self._fused_batch_norm(inputs, training=training)
            if self.virtual_batch_size is not None:
                # Currently never reaches here since fused_batch_norm does not support
                # virtual batching
                outputs = undo_virtual_batching(outputs)
            return outputs

        # Compute the axes along which to reduce the mean / variance
        input_shape = inputs.shape
        ndims = len(input_shape)
        reduction_axes = [i for i in range(ndims) if i not in self.axis]
        if self.virtual_batch_size is not None:
            del reduction_axes[1]  # Do not reduce along virtual batch dim

        # Broadcasting only necessary for single-axis batch norm where the axis is
        # not the last dimension
        broadcast_shape = [1] * ndims
        broadcast_shape[self.axis[0]] = input_shape.dims[self.axis[0]].value

        def _broadcast(v):
            if (v is not None and len(v.shape) != ndims and
                    reduction_axes != list(range(ndims - 1))):
                return array_ops.reshape(v, broadcast_shape)
            return v

        scale, offset = _broadcast(self.gamma), _broadcast(self.beta)

        def _compose_transforms(scale, offset, then_scale, then_offset):
            if then_scale is not None:
                scale *= then_scale
                offset *= then_scale
            if then_offset is not None:
                offset += then_offset
            return (scale, offset)

        # Determine a boolean value for `training`: could be True, False, or None.
        training_value = tf_utils.constant_value(training)
        if training_value == False:  # pylint: disable=singleton-comparison,g-explicit-bool-comparison
            mean, variance = self.moving_mean, self.moving_variance
        else:
            if self.adjustment:
                adj_scale, adj_bias = self.adjustment(array_ops.shape(inputs))
                # Adjust only during training.
                adj_scale = tf_utils.smart_cond(training,
                                                lambda: adj_scale,
                                                lambda: array_ops.ones_like(adj_scale))
                adj_bias = tf_utils.smart_cond(training,
                                               lambda: adj_bias,
                                               lambda: array_ops.zeros_like(adj_bias))
                scale, offset = _compose_transforms(adj_scale, adj_bias, scale, offset)

            # Some of the computations here are not necessary when training==False
            # but not a constant. However, this makes the code simpler.
            keep_dims = self.virtual_batch_size is not None or len(self.axis) > 1
            mean, variance = self._moments(
                math_ops.cast(inputs, self._param_dtype),
                reduction_axes,
                keep_dims=keep_dims)

            moving_mean = self.moving_mean
            moving_variance = self.moving_variance

            mean = tf_utils.smart_cond(training,
                                       lambda: mean,
                                       lambda: ops.convert_to_tensor(moving_mean))
            variance = tf_utils.smart_cond(
                training,
                lambda: variance,
                lambda: ops.convert_to_tensor(moving_variance))

            if self.virtual_batch_size is not None:
                # This isn't strictly correct since in ghost batch norm, you are
                # supposed to sequentially update the moving_mean and moving_variance
                # with each sub-batch. However, since the moving statistics are only
                # used during evaluation, it is more efficient to just update in one
                # step and should not make a significant difference in the result.
                new_mean = math_ops.reduce_mean(mean, axis=1, keepdims=True)
                new_variance = math_ops.reduce_mean(variance, axis=1, keepdims=True)
            else:
                new_mean, new_variance = mean, variance

            if self._support_zero_size_input():
                inputs_size = array_ops.size(inputs)
            else:
                inputs_size = None
            if self.renorm:
                r, d, new_mean, new_variance = self._renorm_correction_and_moments(
                    new_mean, new_variance, training, inputs_size)
                # When training, the normalized values (say, x) will be transformed as
                # x * gamma + beta without renorm, and (x * r + d) * gamma + beta
                # = x * (r * gamma) + (d * gamma + beta) with renorm.
                r = _broadcast(array_ops.stop_gradient(r, name='renorm_r'))
                d = _broadcast(array_ops.stop_gradient(d, name='renorm_d'))
                scale, offset = _compose_transforms(r, d, scale, offset)

            def _do_update(var, value):
                """Compute the updates for mean and variance."""
                return self._assign_moving_average(var, value, self.momentum,
                                                   inputs_size)

            def mean_update():
                true_branch = lambda: _do_update(self.moving_mean, new_mean)
                false_branch = lambda: self.moving_mean
                return tf_utils.smart_cond(training, true_branch, false_branch)

            def variance_update():
                """Update the moving variance."""

                def true_branch_renorm():
                    # We apply epsilon as part of the moving_stddev to mirror the training
                    # code path.
                    moving_stddev = _do_update(self.moving_stddev,
                                               math_ops.sqrt(new_variance + self.epsilon))
                    return self._assign_new_value(
                        self.moving_variance,
                        # Apply relu in case floating point rounding causes it to go
                        # negative.
                        K.relu(moving_stddev * moving_stddev - self.epsilon))

                if self.renorm:
                    true_branch = true_branch_renorm
                else:
                    true_branch = lambda: _do_update(self.moving_variance, new_variance)

                false_branch = lambda: self.moving_variance
                return tf_utils.smart_cond(training, true_branch, false_branch)

            self.add_update(mean_update)
            self.add_update(variance_update)

        mean = math_ops.cast(mean, inputs.dtype)
        variance = math_ops.cast(variance, inputs.dtype)
        if offset is not None:
            offset = math_ops.cast(offset, inputs.dtype)
        if scale is not None:
            scale = math_ops.cast(scale, inputs.dtype)
        # TODO(reedwm): Maybe do math in float32 if given float16 inputs, if doing
        # math in float16 hurts validation accuracy of popular models like resnet.
        outputs = nn.batch_normalization(inputs,
                                         _broadcast(mean),
                                         _broadcast(variance),
                                         offset,
                                         scale,
                                         self.epsilon)
        # If some components of the shape got lost due to adjustments, fix that.
        outputs.set_shape(input_shape)

        if self.virtual_batch_size is not None:
            outputs = undo_virtual_batching(outputs)
        return outputs
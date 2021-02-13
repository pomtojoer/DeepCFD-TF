import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer

class MaxPoolingWithArgmax2D(Layer):
    def __init__(
            self,
            pool_size=(2, 2),
            strides=(2, 2),
            padding='valid',
            **kwargs):
        super().__init__()
        self.padding = padding
        self.pool_size = pool_size
        self.strides = strides

    def call(self, inputs, **kwargs):
        padding = self.padding
        pool_size = self.pool_size
        strides = self.strides
        ksize = [1, pool_size[0], pool_size[1], 1]
        padding = padding.upper()
        strides = [1, strides[0], strides[1], 1]
        output, argmax = tf.nn.max_pool_with_argmax(
            inputs,
            ksize=ksize,
            strides=strides,
            padding=padding)

        argmax = K.cast(argmax, K.floatx())
        return [output, argmax]

    def compute_output_shape(self, input_shape):
        ratio = (1, 2, 2, 1)
        output_shape = [
            dim // ratio[idx]
            if dim is not None else None
            for idx, dim in enumerate(input_shape)]
        output_shape = tuple(output_shape)
        return [output_shape, output_shape]

    def compute_mask(self, inputs, mask=None):
        return 2 * [None]

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'padding': self.padding,
            'pool_size': self.pool_size,
            'strides': self.strides,
        })
        return config    

# class MaxUnpooling2D(Layer):
#     def __init__(self, pool_size=(2, 2), out_shape=None, **kwargs):
#         super().__init__()
#         self.pool_size = pool_size
#         self.out_shape = out_shape

#     def call(self, inputs):
#         updates, mask = inputs[0], inputs[1]

#         mask = K.cast(mask, 'int32')
#         input_shape = tf.shape(updates, out_type='int32')
        
#         out_shape = self.out_shape
        
#         if out_shape is None:
#             out_shape = (
#                 input_shape[0],
#                 input_shape[1] * self.pool_size[0],
#                 input_shape[2] * self.pool_size[1],
#                 input_shape[3])
#         else:
#             out_shape = (
#                 -1,
#                 out_shape[1],
#                 out_shape[2],
#                 out_shape[3]
#             )
        
#         ret = tf.scatter_nd(K.expand_dims(K.flatten(mask)),
#                               K.flatten(updates),
#                               [K.prod(out_shape)])
#         # if out_shape is None:
#         #     input_shape = updates.shape
#         #     out_shape = [
#         #         -1,
#         #         input_shape[1] * self.pool_size[0],
#         #         input_shape[2] * self.pool_size[1],
#         #         input_shape[3]
#         #     ]
#         # else:
#         #     out_shape = [
#         #         -1,
#         #         out_shape[1],
#         #         out_shape[2],
#         #         out_shape[3]
#         #     ]
#         out_shape = list(out_shape)
#         return K.reshape(ret, out_shape)

#     def compute_output_shape(self, input_shape):
#         mask_shape = input_shape[1]
#         return (
#                 mask_shape[0],
#                 mask_shape[1]*self.pool_size[0],
#                 mask_shape[2]*self.pool_size[1],
#                 mask_shape[3]
#                 )





class MaxUnpooling2D(Layer):
    def __init__(self, size=(2, 2), out_shape=None, **kwargs):
        super().__init__()
        self.size = size
        self.out_shape = out_shape

    def call(self, inputs):
        updates, mask = inputs[0], inputs[1]
        mask = K.cast(mask, 'int32')
        input_shape = tf.shape(updates, out_type='int32')

        if self.out_shape is None:
            out_shape = (
                input_shape[0],
                input_shape[1] * self.size[0],
                input_shape[2] * self.size[1],
                input_shape[3])
        else:
            out_shape = (
                input_shape[0],
                self.out_shape[1],
                self.out_shape[2],
                self.out_shape[3],
            )

        ret = tf.scatter_nd(K.expand_dims(K.flatten(mask)),
                              K.flatten(updates),
                              [K.prod(out_shape)])

        input_shape = updates.shape
        if self.out_shape is None:
            out_shape = [-1,
                        input_shape[1] * self.size[0],
                        input_shape[2] * self.size[1],
                        input_shape[3]]
        else:
            out_shape = [
                -1,
                self.out_shape[1],
                self.out_shape[2],
                self.out_shape[3],
            ]

        return K.reshape(ret, out_shape)

    def compute_output_shape(self, input_shape):
        mask_shape = input_shape[1]
        if self.out_shape is None:
            return (
                    mask_shape[0],
                    mask_shape[1]*self.size[0],
                    mask_shape[2]*self.size[1],
                    mask_shape[3]
                    )
        else:
            return (
                mask_shape[0],
                self.out_shape[1],
                self.out_shape[2],
                self.out_shape[3],
            )

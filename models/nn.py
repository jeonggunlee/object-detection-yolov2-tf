import time
from abc import abstractmethod, ABCMeta
import tensorflow as tf
import numpy as np
from models.layers import conv_layer, max_pool, fc_layer, batchNormalization


class DetectNet(metaclass=ABCMeta):
    """Base class for Convolutional Neural Networks for detection."""

    def __init__(self, input_shape, num_classes, **kwargs):

        self.X = tf.placeholder(tf.float32, [None] + input_shape)
        self.is_train = tf.placeholder(tf.bool)
        self.num_classes = num_classes
        self.d = self._build_model(**kwargs)
        self.pred = self.d['pred']
        self.loss = self._build_loss(**kwargs)

    @abstractmethod
    def _build_model(self, **kwargs):
        """
        Build model.
        This should be implemented.
        """
        pass

    @abstractmethod
    def _build_loss(self, **kwargs):
        """
        build loss function for the model training.
        This should be implemented.
        """
        pass

    def predict(self, sess, dataset, verbose=False, **kwargs):
        """
        Make predictions for the given dataset.
        :param ses: tf.Session.
        :param dataset: DataSet.
        :param verbose: bool, whether to print details during prediction.
        :param kwargs: dict, extra arguments for prediction.
                -batch_size: int, batch size for each iteraction.
        :return _y_pred: np.ndarray, shape: shape of self.pred
        """

        batch_size = kwargs.pop('batch_size', 64)

        num_classes = self.num_classes
        pred_size = dataset.num_examples
        num_steps = pred_size // batch_size

        if verbose:
            print('Running prediction loop...')

        # Start prediction loop
        _y_pred = []
        start_time = time.time()
        for i in range(num_steps + 1):
            if i == num_steps:
                _batch_size = pred_size - num_steps * batch_size
            else:
                _batch_size = batch_size
            X, _ = dataset.next_batch(
                _batch_size, shuffle=False, is_train=False)

            # Compute predictions
            # (N, Cell, Cell, 5 + num_classes)
            y_pred = sess.run(self.pred, feed_dict={
                              self.X: X, self.is_train: False})

            _y_pred.append(y_pred)

        if verbose:
            print('Total prediction time(sec): {}'.format(
                time.time() - start_time))

        _y_pred = np.concatenate(_y_pred, axis=0)

        return _y_pred


class YOLO(DetectNet):
    """YOLO class"""

    def __init__(self, input_shape, num_classes, grid_size=13, num_anchors=5, **kwargs):

        self.y = tf.placeholder(tf.float32, [None] +
                                [grid_size, grid_size, num_anchors, 5 + num_classes])
        self.grid_size = gird_size
        self.num_anchors = num_anchors
        cxcy = np.transpose([np.tile(np.arange(self.grid_size), self.grid_size),
			np.repeat(np.arange(self.grid_size), self.grid_size)])
        self.cxcy = np.reshape(cxcy, (1, self.grid_size, self.grid_size, 1, 2))
        super(YOLO, self).__init__(input_shape, num_classes, **kwargs)

    def _build_model(self, **kwargs):
        """
        Build model.
        :param kwargs: dict, extra arguments for building YOLO.
                -image_mean: np.ndarray, mean image for each input channel, shape: (C,).
        :return d: dict, containing outputs on each layer.
        """

        d = dict()
        x_mean = kwargs.pop('image_mean', 0.0)

        # input
        X_input = self.X - x_mean
        is_train = tf.placeholder(tf.bool)

        #conv1 - batch_norm1 - leaky_relu1 - pool1
        with tf.variable_scope('conv1'):
            d['conv1'] = conv_layer(X_input, 3, 1, 32,
                                    padding='SAME', use_bias=False, weights_stddev=0.01)
            print('conv1.shape', d['conv1'].get_shape().as_list())
        d['batch_norm1'] = batchNormalization(d['conv1'], is_train)
        d['leaky_relu1'] = tf.nn.leaky_relu(d['batch_norm1'], alpha=0.1)
        d['pool1'] = max_pool(d['leaky_relu1'], 2, 2, padding='SAME')
        # (416, 416, 3) --> (208, 208, 32)
        print('pool1.shape', d['pool1'].get_shape().as_list())

        #conv2 - batch_norm2 - leaky_relu2 - pool2
        with tf.variable_scope('conv2'):
            d['conv2'] = conv_layer(d['pool1'], 3, 1, 64,
                                    padding='SAME', use_bias=False, weights_stddev=0.01)
            print('conv2.shape', d['conv2'].get_shape().as_list())

        d['batch_norm2'] = batchNormalization(d['conv2'], is_train)
        d['leaky_relu2'] = tf.nn.leaky_relu(d['batch_norm2'], alpha=0.1)
        d['pool2'] = max_pool(d['leaky_relu2'], 2, 2, padding='SAME')
        # (208, 208, 32) --> (104, 104, 64)
        print('pool2.shape', d['pool2'].get_shape().as_list())

        #conv3 - batch_norm3 - leaky_relu3
        with tf.variable_scope('conv3'):
            d['conv3'] = conv_layer(d['pool2'], 3, 1, 128,
                                    padding='SAME', use_bias=False, weights_stddev=0.01)
            print('conv3.shape', d['conv3'].get_shape().as_list())
                d['batch_norm3'] = batchNormalization(d['conv3'], is_train)
        d['leaky_relu3'] = tf.nn.leaky_relu(d['batch_norm3'], alpha=0.1)
        # (104, 104, 64) --> (104, 104, 128)

        #conv4 - batch_norm4 - leaky_relu4
        with tf.variable_scope('conv4'):
            d['conv4'] = conv_layer(d['leaky_relu3'], 1, 1, 64,
                                    padding='SAME', use_bias=False, weights_stddev=0.01)
            print('conv4.shape', d['conv4'].get_shape().as_list())
                d['batch_norm4'] = batchNormalization(d['conv4'], is_train)
        d['leaky_relu4'] = tf.nn.leaky_relu(d['conv4'], alpha=0.1)
        # (104, 104, 128) --> (104, 104, 64)

        #conv5 - batch_norm5 - leaky_relu5 - pool5
        with tf.variable_scope('conv5'):
            d['conv5'] = conv_layer(d['leaky_relu4'], 3, 1, 128,
                                    padding='SAME', use_bias=False, weights_stddev=0.01)
            print('conv5.shape', d['conv5'].get_shape().as_list())
                d['batch_norm5'] = batchNormalization(d['conv5'], is_train)
        d['leaky_relu5'] = tf.nn.leaky_relu(d['batch_norm5'], alpha=0.1)
        d['pool5'] = max_pool(d['leaky_relu5'], 2, 2, padding='SAME')
        # (104, 104, 64) --> (52, 52, 128)
        print('pool5.shape', d['pool5'].get_shape().as_list())

        #conv6 - batch_norm6 - leaky_relu6
        with tf.variable_scope('conv6'):
            d['conv6'] = conv_layer(d['pool5'], 3, 1, 256,
                                    padding='SAME', use_bias=False, weights_stddev=0.01)
            print('conv6.shape', d['conv6'].get_shape().as_list())
                d['batch_norm6'] = batchNormalization(d['conv6'], is_train)
        d['leaky_relu6'] = tf.nn.leaky_relu(d['batch_norm6'], alpha=0.1)
        # (52, 52, 128) --> (52, 52, 256)

        #conv7 - batch_norm7 - leaky_relu7
        with tf.variable_scope('conv7'):
            d['conv7'] = conv_layer(d['leaky_relu6'], 1, 1, 128,
                                    padding='SAME', weights_stddev=0.01, biases_value=0.0)
            print('conv7.shape', d['conv7'].get_shape().as_list())
                d['batch_norm7'] = batchNormalization(d['conv7'], is_train)
        d['leaky_relu7'] = tf.nn.leaky_relu(d['batch_norm7'], alpha=0.1)
        # (52, 52, 256) --> (52, 52, 128)

        #conv8 - batch_norm8 - leaky_relu8 - pool8
        with tf.variable_scope('conv8'):
            d['conv8'] = conv_layer(d['leaky_relu7'], 3, 1, 256,
                                    padding='SAME', use_bias=False, weights_stddev=0.01)
            print('conv8.shape', d['conv8'].get_shape().as_list())
                d['batch_norm8'] = batchNormalization(d['conv8'], is_train)
        d['leaky_relu8'] = tf.nn.leaky_relu(d['batch_norm8'], alpha=0.1)
        d['pool8'] = max_pool(d['leaky_relu8'], 2, 2, padding='SAME')
        # (52, 52, 128) --> (26, 26, 256)
        print('pool8.shape', d['pool8'].get_shape().as_list())

        #conv9 - batch_norm9 - leaky_relu9
        with tf.variable_scope('conv9'):
            d['conv9'] = conv_layer(d['pool8'], 3, 1, 512,
                                    padding='SAME', use_bias=False, weights_stddev=0.01)
            print('conv9.shape', d['conv9'].get_shape().as_list())
                d['batch_norm9'] = batchNormalization(d['conv9'], is_train)
        d['leaky_relu9'] = tf.nn.leaky_relu(d['batch_norm9'], alpha=0.1)
        # (26, 26, 256) --> (26, 26, 512)

        #conv10 - batch_norm10 - leaky_relu10
        with tf.variable_scope('conv10'):
            d['conv10'] = conv_layer(d['leaky_relu9'], 1, 1, 256,
                                     padding='SAME', use_bias=False, weights_stddev=0.01)
            print('conv10.shape', d['conv10'].get_shape().as_list())
                d['batch_norm10'] = batchNormalization(d['conv10'], is_train)
        d['leaky_relu10'] = tf.nn.leaky_relu(d['batch_norm10'], alpha=0.1)
        # (26, 26, 512) --> (26, 26, 256)

        #conv11 - batch_norm11 - leaky_relu11
        with tf.variable_scope('conv11'):
            d['conv11'] = conv_layer(d['leaky_relu10'], 3, 1, 512,
                                     padding='SAME', use_bias=False, weights_stddev=0.01)
            print('conv11.shape', d['conv11'].get_shape().as_list())
                d['batch_norm11'] = batchNormalization(d['conv11'], is_train)
        d['leaky_relu11'] = tf.nn.leaky_relu(d['batch_norm11'], alpha=0.1)
        # (26, 26, 256) --> (26, 26, 512)

        #conv12 - batch_norm12 - leaky_relu12
        with tf.variable_scope('conv12'):
            d['conv12'] = conv_layer(d['leaky_relu11'], 1, 1, 256,
                                     padding='SAME', use_bias=False, weights_stddev=0.01)
            print('conv12.shape', d['conv12'].get_shape().as_list())
                d['batch_norm12'] = batchNormalization(d['conv12'], is_train)
        d['leaky_relu12'] = tf.nn.leaky_relu(d['batch_norm12'], alpha=0.1)
        # (26, 26, 512) --> (26, 26, 256)

        #conv13 - batch_norm13 - leaky_relu13 - pool13
        with tf.variable_scope('conv13'):
            d['conv13'] = conv_layer(d['leaky_relu12'], 3, 1, 512,
                                     padding='SAME', use_bias=False, weights_stddev=0.01)
            print('conv13.shape', d['conv13'].get_shape().as_list())
                d['batch_norm13'] = batchNormalization(d['conv13'], is_train)
        d['leaky_relu13'] = tf.nn.leaky_relu(d['batch_norm13'], alpha=0.1)
        d['pool13'] = max_pool(d['leaky_relu13'], 2, 2, padding='SAME')
        print('pool13.shape', d['pool13'].get_shape().as_list())
        # (26, 26, 256) --> (13, 13, 512)

        #conv14 - batch_norm14 - leaky_relu14
        with tf.variable_scope('conv14'):
            d['conv14'] = conv_layer(d['pool13'], 3, 1, 1024,
                                     padding='SAME', use_bias=False, weights_stddev=0.01)
            print('conv14.shape', d['conv14'].get_shape().as_list())
                d['batch_norm14'] = batchNormalization(d['conv14'], is_train)
        d['leaky_relu14'] = tf.nn.leaky_relu(d['batch_norm14'], alpha=0.1)
        # (13, 13, 512) --> (13, 13, 1024)

        #conv15 - batch_norm15 - leaky_relu15
        with tf.variable_scope('conv15'):
            d['conv15'] = conv_layer(d['leaky_relu14'], 1, 1, 512,
                                     padding='SAME', use_bias=False, weights_stddev=0.01)
            print('conv15.shape', d['conv15'].get_shape().as_list())
                d['batch_norm15'] = batchNormalization(d['conv15'], is_train)
        d['leaky_relu15'] = tf.nn.leaky_relu(d['batch_norm15'], alpha=0.1)
        # (13, 13, 1024) --> (13, 13, 512)

        #conv16 - batch_norm16 - leaky_relu16
        with tf.variable_scope('conv16'):
            d['conv16'] = conv_layer(d['leaky_relu15'], 3, 1, 1024,
                                     padding='SAME', use_bias=False, weights_stddev=0.01)
            print('conv16.shape', d['conv16'].get_shape().as_list())
                d['batch_norm16'] = batchNormalization(d['conv16'], is_train)
        d['leaky_relu16'] = tf.nn.leaky_relu(d['batch_norm16'], alpha=0.1)
        # (13, 13, 512) --> (13, 13, 1024)

        #conv17 - batch_norm16 - leaky_relu17
        with tf.variable_scope('conv17'):
            d['conv17'] = conv_layer(d['leaky_relu16'], 1, 1, 512,
                                     padding='SAME', use_bias=False, weights_stddev=0.01)
            print('conv17.shape', d['conv17'].get_shape().as_list())
                d['batch_norm17'] = batchNormalization(d['conv17'], is_train)
        d['leaky_relu17'] = tf.nn.leaky_relu(d['batch_norm17'], alpha=0.1)
        # (13, 13, 1024) --> (13, 13, 512)

        #conv18 - batch_norm18 - leaky_relu18
        with tf.variable_scope('conv18'):
            d['conv18'] = conv_layer(d['leaky_relu17'], 3, 1, 1024,
                                     padding='SAME', use_bias=False, weights_stddev=0.01)
            print('conv18.shape', d['conv18'].get_shape().as_list())
                d['batch_norm18'] = batchNormalization(d['conv18'], is_train)
        d['leaky_relu18'] = tf.nn.leaky_relu(d['batch_norm18'], alpha=0.1)
        # (13, 13, 512) --> (13, 13, 1024)

        #conv19 - batch_norm19 - leaky_relu19
        with tf.variable_scope('conv19'):
            d['conv19'] = conv_layer(d['leaky_relu18'], 3, 1, 1024,
                                     padding='SAME', use_bias=False, weights_stddev=0.01)
            print('conv19.shape', d['conv19'].get_shape().as_list())
                d['batch_norm19'] = batchNormalization(d['conv19'], is_train)
        d['leaky_relu19'] = tf.nn.leaky_relu(d['batch_norm19'], alpha=0.1)
        # (13, 13, 1024) --> (13, 13, 1024)

        #conv20 - batch_norm20 - leaky_relu20
        with tf.variable_scope('conv20'):
            d['conv20'] = conv_layer(d['leaky_relu19'], 3, 1, 1024,
                                     padding='SAME', use_bias=False, weights_stddev=0.01)
            print('conv20.shape', d['conv20'].get_shape().as_list())
                d['batch_norm20'] = batchNormalization(d['conv20'], is_train)
        d['leaky_relu20'] = tf.nn.leaky_relu(d['batch_norm20'], alpha=0.1)
        # (13, 13, 1024) --> (13, 13, 1024)

        # concatenate layer21 and layer 13 using space to depth
        d['skip_connection'] = conv_layer(d['leaky_relu13'], 1, 1, 64,
                                          padding='SAME', use_bias=False, weights_stddev=0.01)
        d['skip_batch'] = batchNormalization(d['skip_connection'], is_train)
        d['skip_leaky_relu'] = tf.nn.leaky_relu(d['skip_batch'], alpha=0.1)
        d['skip_space_to_depth_x2'] = tf.space_to_depth(
            d['skip_leaky_relu'], block_size=2)
        d['concat21'] = tf.concat(
            [d['skip_space_to_depth_x2'], d['leaky_relu21']], axis=-1)
        # (13, 13, 1024) --> (13, 13, 256+1024)

        #conv22 - leaky_relu22
        with tf.variable_scope('conv22'):
            d['conv22'] = conv_layer(d['leaky_relu21'], 3, 1, 1024,
                                     padding='SAME', use_bias=False, weights_stddev=0.01)
            print('conv22.shape', d['conv22'].get_shape().as_list())
                d['batch_norm22'] = batchNormalization(d['conv22'], is_train)
        d['leaky_relu22'] = tf.nn.leaky_relu(d['batch_norm22'], alpha=0.1)
        # (13, 13, 1280) --> (13, 13, 1024)

        output_channel = self.num_anchors * (5 + self.num_classes)
        d['pred'] = conv_layer(d['leaky_relu22'], 1, 1, output_channel,
                               padding='SAME', use_bias=True, weights_stddev=0.01, biases_value=0.1)
        print('pred.shape', d['pred'].get_shape().as_list())
        # (13, 13, 1024) --> (13, 13, num_anchors * (5 + num_classes))

        return d

    def _build_loss(self, **kwargs):
        """
        Build loss function for the model training.
        :param kwargs: dict, extra arguments
                - scale: list, [xy, wh, resp_confidence, no_resp_confidence, class_probs]
        :return tf.Tensor.
        """

        scale = kwargs.pop('scale', [5, 5, 5, 0.5, 1.0])

        # Prepare parameters


        return total_loss

    def interpret_output(self, sess, images, **kwargs):
        """
        Interpret outputs to decode bounding box from y_pred.
        :param sess: tf.Session
        :param kwargs: dict, extra arguments for prediction.
                -batch_size: int, batch size for each iteraction.
        :param images: np.ndarray, shape (N, H, W, C)
        :return bbox_pred: np.ndarray, shape (N, Cell*Cell*num_boxes, 4 + num_classes)
        """
        batch_size = kwargs.pop('batch_size', 128)

        is_batch = len(images.shape) == 4
        if not is_batch:
            images = np.expand_dims(images, 0)
        pred_size = images.shape[0]
        num_steps = pred_size // batch_size

        bboxes = []
        for i in range(num_steps + 1):
            if i == num_steps:
                image = images[i * batch_size:]
            else:
                image = images[i * batch_size:(i + 1) * batch_size]
            bbox = sess.run(self.bboxes, feed_dict={self.X: image})
            bbox = np.reshape(bbox, (bbox.shape[0], -1, bbox.shape[-1]))
            bboxes.append(bbox)

        bboxes = np.concatenate(bboxes, axis=0)
        if is_batch:
            return bboxes
        else:
            return bboxes[0]

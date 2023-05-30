import os
import tensorflow as tf
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1.train import AdamOptimizer
from tensorflow.estimator import Estimator, RunConfig
from data.images import ImageShower
from .model import Network
from .losses import dice_loss, piece_wise_cross, regulator_l1


class Model:
    def __init__(self, learning_rate, shape, version, batch_size=32, path=''):
        """

        Parameters
        ----------
        learning_rate: float
            Initial learning rate of the algorithm
        shape: iterable
            Shape of the images to work with
        version: iterable or int
            Name of the model, it would be convert into a string
        batch_size: int
            Batch size os the training
        """
        session_config = ConfigProto(allow_soft_placement=True)
        config = RunConfig(save_summary_steps=5, save_checkpoints_steps=500, session_config=session_config)
        self.network = Network()
        self.batch = batch_size
        self.main_path = os.path.join(path, 'logs', 'version' + str(version))  # str(shape)
        self.model = Estimator(model_fn=self.estimator_function, model_dir=self.main_path, config=config)
        self.optimizer = AdamOptimizer(learning_rate=learning_rate)
        self.images = ImageShower(shape)
        self._inner_loss = None
        self.mse_and_l1 = []

    def training_step(self, features, labels):
        _y_pred = tf.cast(self.network.main(features), tf.float32)  # call the prediction
        y_pred = tf.nn.sigmoid(_y_pred)  # apply sigmoid to get probabilities
        # loss = dice_loss(labels, y_pred)  + piece_wise_cross(labels, y_pred) # + regulator_l1(y_pred, 1.5e-6)/2
        loss = dice_loss(labels, y_pred)
        return loss, tf.cast(y_pred, tf.float32)

    def classes_show(self, y_pred, labels, features):
        y1 = tf.where(labels >= .5, 1., 0.)
        s1 = tf.compat.v1.summary.image("class1", tf.concat([y_pred, y1, y1], -1), max_outputs=8)
        sum_input = tf.compat.v1.summary.image("Entrance", features, max_outputs=8)
        return s1, sum_input

    def estimator_summary(self, mode, loss, y_pred, labels, features):
        if mode == tf.estimator.ModeKeys.TRAIN:
            sum_loss = tf.compat.v1.summary.scalar('loss', loss)
            s1, sum_input = self.classes_show(y_pred, labels, features)
            tf.compat.v1.summary.merge([s1, sum_input, sum_loss])
            tf.compat.v1.summary.merge([sum_loss])

            return
        sum_loss = tf.compat.v1.summary.scalar('eval_loss', loss)
        return [sum_loss]

    def estimator_function(self, features, labels, mode):
        if mode == tf.estimator.ModeKeys.PREDICT:
            y_pred = self.network.main(features['x'])
            return tf.estimator.EstimatorSpec(mode=mode, predictions=y_pred)
        else:
            training = mode == tf.estimator.ModeKeys.TRAIN
            loss, y_pred = self.training_step(features, labels)
            self.estimator_summary(mode, loss, y_pred, labels, features)
            if training:
                train_op = self.optimizer.minimize(loss, tf.compat.v1.train.get_global_step(),
                                                   colocate_gradients_with_ops=True)  # optimization step
                return tf.estimator.EstimatorSpec(
                    mode=mode,
                    loss=loss,
                    train_op=train_op,
                    predictions=y_pred
                )
            else:
                evaluation_hook = tf.estimator.SummarySaverHook(
                    save_steps=1,
                    output_dir=self.main_path,
                    summary_op=[*self.estimator_summary(mode, loss, y_pred, labels, features)]
                )

                return tf.estimator.EstimatorSpec(
                    mode=mode,
                    loss=loss,
                    predictions=y_pred,
                    evaluation_hooks=[evaluation_hook]
                )

    def train(self, train_spec, eval_spec):
        tf.estimator.train_and_evaluate(self.model, train_spec, eval_spec)

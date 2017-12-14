'''
You are going to train the CNN model here.

'''
from load_data import LoadTrainBatch, LoadValBatch
from model import inference
import tensorflow as tf
import numpy as np
batch_size = 5000
input_w = 66
input_h = 200
input_channel = 3
max_iterator_step = int(1e+5)
learning_rate = 1e-7


def train():
    image_tensor = tf.placeholder(
        dtype=tf.float32,
        shape=[
            batch_size,
            input_w,
            input_h,
            input_channel
        ]
    )
    label_tensor = tf.placeholder(
        dtype=tf.float32,
        shape=[
            batch_size
        ]
    )
    global_step = tf.Variable(0, trainable=False)
    inference_result_tensor = inference(image_tensor)
    loss_tensor = tf.reduce_mean(tf.square(inference_result_tensor - label_tensor))
    tf.summary.scalar('training loss', loss_tensor)
    optimizer = tf.train.GradientDescentOptimizer(
        learning_rate=learning_rate
    )
    train_op = optimizer.minimize(
        loss=loss_tensor,
        global_step=global_step
    )
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        summary_writer = tf.summary.FileWriter('./log', sess.graph)
        summary_merge_tensor = tf.summary.merge_all()
        for _ in range(max_iterator_step):
            training_image_batch, training_label_batch = LoadTrainBatch(batch_size)
            _, train_loss_value, train_predict_label, global_step_value, summary_value = sess.run(
                [train_op, loss_tensor, inference_result_tensor, global_step, summary_merge_tensor],
                feed_dict={
                     image_tensor: training_image_batch,
                     label_tensor: np.squeeze(training_label_batch)
                }
            )
            summary_writer.add_summary(summary_value, global_step_value)
            if global_step_value % 10 == 0:
                validation_image_batch, validation_label_batch = LoadValBatch(batch_size)
                validation_loss_value, validation_predict_value = sess.run(
                    [loss_tensor, inference_result_tensor],
                    feed_dict={
                        image_tensor: validation_image_batch,
                        label_tensor: np.squeeze(validation_label_batch)
                    }
                )
                print 'global step: %d, training loss is %.3f, validation loss is %.3f' \
                      % (global_step_value, train_loss_value, validation_loss_value)
                print validation_label_batch[:10]
                print validation_predict_value[:10]
        summary_writer.close()
if __name__ == '__main__':
    train()
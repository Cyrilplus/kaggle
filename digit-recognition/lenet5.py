#!/usr/bin/env python
# -*- coding: utf-8 -*-

import paddle.v2 as paddle
import numpy as np
import data as dataset
from datetime import datetime

import os


PARAMETERS_TAR = 'lenet5-parameters.tar'

def letnet_5(img):
    conv_pool_layer_1 = paddle.networks.simple_img_conv_pool(
        input=img,
        filter_size=3,
        num_filters=20,
        num_channel=1,
        pool_size=2,
        pool_stride=2,
        act=paddle.activation.Relu()
    )

    conv_pool_layer_2 = paddle.networks.simple_img_conv_pool(
        input=conv_pool_layer_1,
        filter_size=3,
        num_filters=50,
        num_channel=20,
        pool_size=2,
        pool_stride=2,
        act=paddle.activation.Relu()
    )

    predict = paddle.layer.fc(
        input=conv_pool_layer_2,
        size=10,
        act=paddle.activation.Softmax()
    )
    return predict

def main():
    paddle.init(use_gpu=True, trainer_count=1)

    img = paddle.layer.data(name='img', type=paddle.data_type.dense_vector(784))
    label = paddle.layer.data(name='label', type=paddle.data_type.integer_value(10))

    predict = letnet_5(img)

    cost = paddle.layer.classification_cost(input=predict, label=label)
    if os.path.exists(PARAMETERS_TAR):
        with open(PARAMETERS_TAR, 'r') as f:
            parameters = paddle.parameters.Parameters.from_tar(f)
    else:
        parameters = paddle.parameters.create(cost)
    optimizer = paddle.optimizer.Momentum(
        learning_rate=0.1 / 128,
        momentum=0.9,
        regularization=paddle.optimizer.L2Regularization(rate=0.0001) 
    )

    trainer = paddle.trainer.SGD(
        cost=cost,
        parameters=parameters,
        update_equation=optimizer
    )

    def event_handler(evt):
        if isinstance(evt, paddle.event.EndPass):
            print 'Pass id: %d, %s' % (evt.pass_id, evt.metrics)
    

    trainer.train(
        reader=paddle.batch(paddle.reader.shuffle(dataset.train(), buf_size=8192), batch_size=64),
        event_handler=event_handler,
        num_passes=30
    )

    with open(PARAMETERS_TAR, 'w') as f:
        trainer.save_parameter_to_tar(f)

    test_reader = dataset.test()()
    test_list = []
    for img in test_reader:
        test_list.append(img)

    labels = paddle.infer(
        output_layer=predict,
        parameters=parameters,
        input=test_list
    )

    labels = np.argmax(labels, axis=1)
    dataset.save_result2csv('./result.csv', labels)
    print 'training end'

if __name__ == '__main__':
    begin_time = datetime.now()
    main()
    end_time = datetime.now()
    delta_seconds = (end_time - begin_time).seconds
    print 'time consuming: %ds' % delta_seconds
    
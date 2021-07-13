import numpy as np
import mxnet as mx
from mxnet import nd, autograd, gluon
mx.random.seed(1)

class AGE_PREDICTION:

    def __init__(self):
        self.smoothing_constant = .1

    def create_model(self):
        num_fc = 10
        self.net = gluon.nn.Sequential()
        with self.net.name_scope():
            self.net.add(gluon.nn.Conv2D(channels=10, kernel_size=(3,2), activation='relu'))
            self.net.add(gluon.nn.MaxPool2D(pool_size=2, strides=2))
            self.net.add(gluon.nn.Conv2D(channels=10, kernel_size=(3,2), activation='relu'))
            self.net.add(gluon.nn.MaxPool2D(pool_size=2, strides=2))
            self.net.add(gluon.nn.Flatten())
            self.net.add(gluon.nn.Dense(num_fc, activation="relu"))
            self.net.add(gluon.nn.Dense(1))
        self.ctx = mx.cpu()
        self.net.collect_params().initialize(mx.init.Xavier(magnitude=2.24), ctx=self.ctx)

    def train(self,epochs,train_data,test_data,label):
        l1_loss = gluon.loss.L1Loss()
        trainer = gluon.Trainer(self.net.collect_params(), 'sgd', {'learning_rate': .001})
        for e in range(epochs):
            for i, data in enumerate([train_data]):
                data = data.as_in_context(self.ctx)
                # print(data)
                label = label.as_in_context(self.ctx)
                with autograd.record():
                    output = self.net(data)
                    # print(output)
                    loss = l1_loss(output, label)
                loss.backward()
                trainer.step(data.shape[0])

                ##########################
                #  Keep a moving average of the losses
                ##########################
                curr_loss = nd.mean(loss).asscalar()
                moving_loss = (curr_loss if ((i == 0) and (e == 0))
                            else (1 - self.smoothing_constant) * moving_loss + self.smoothing_constant * curr_loss)

            test_accuracy = self.evaluate_accuracy(test_data,label)
            train_accuracy = self.evaluate_accuracy(train_data,label)
            print("Epoch: %s | Loss: %.2f | Train_acc: %.2f | Test_acc: %.2f" % (e, moving_loss, train_accuracy, test_accuracy))


    def evaluate_accuracy(self,data_iterator,label):
        # print("yes")
        acc = mx.metric.RMSE()
        for i, data in enumerate([data_iterator]):
            data = data.as_in_context(self.ctx)
            label = label.as_in_context(self.ctx)
            output = self.net(data)
            # print(output)
            # predictions = nd.argmax(output, axis=1)
            
            acc.update(preds=output, labels=label)
        return acc.get()[1]
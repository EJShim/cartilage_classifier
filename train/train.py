import tensorflow as tf
import numpy as np
import sys, os
import matplotlib.pyplot as plt
import xlsxwriter

#Add Root Dir
file_path = os.path.dirname(os.path.realpath(__file__))
root_path = os.path.abspath(os.path.join(file_path, os.pardir))
sys.path.insert(0, root_path)
# sys.setrecursionlimit(2000)

#Import Network
import VRN

#Training Data Path
TRAIN_DATA_PATH = os.path.join(root_path, "data", "train.npz")
TEST_DATA_PATH =  os.path.join(root_path, "data", "test.npz")


def save_data(worksheet, loss, acc, acc_bin):
    worksheet.write(0, 2, "Accuracy(Binary)")

    for i, l in enumerate(loss):
        worksheet.write(i+1, 0, l)

    for i, a in enumerate(acc):
        worksheet.write(i+1, 1, a)
        worksheet.write(i+1, 2, acc_bin[i])

    return



def subsample_array(arr):
    ratio = int(len(arr)/100)
    if ratio == 0:
        return arr

    return arr[0::ratio]


if __name__ == '__main__':

    #Load Features and Targets
    data_load = np.load(TRAIN_DATA_PATH)
    features = data_load['features']
    features = np.reshape(features, (features.shape[0], features.shape[1], features.shape[2], features.shape[3], 1))
    targets = data_load['targets']

    #Load Test Features and Targets
    test_data_load = np.load(TEST_DATA_PATH)
    test_features = test_data_load['features']
    test_features = np.reshape(test_features, (test_features.shape[0], test_features.shape[1], test_features.shape[2], test_features.shape[3], 1))
    test_targets = test_data_load['targets']


    #Get Configuration
    cfg = VRN.cfg
    max_epochs = cfg['max_epochs']
    batch_size = cfg['batch_size']

    #Initialize Figure
    figure = plt.figure(1)
    loss_ax = figure.add_subplot(211)
    loss_data = []

    acc_ax = figure.add_subplot(212)
    acc_data = []
    acc_bin = []

    #Get TF Functions
    x, y, keep_prob, last_conv = VRN.get_deep_model()    
    y_true = tf.placeholder(tf.int32)
    sess = tf.InteractiveSession()

    #Predict Functions
    y_deter = tf.layers.flatten(y)
    pred_classes = tf.argmax(y_deter, axis=1)
    pred_probs = tf.nn.softmax(y_deter)


    #Training Functions
    learning_rate = tf.placeholder(tf.float32)
    loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y_deter, labels=y_true))
    regularizer = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name])#
    loss = loss_op + 0.001*regularizer
    trainer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9, use_nesterov=True)

    #batch normalizer?
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        optimize = trainer.minimize(loss)
    
    lr = 0.002    
    sess.run(tf.global_variables_initializer())
    


    #Run Training
    for epoch in range(max_epochs):
        #Initialize WorkSheet
        workbook = xlsxwriter.Workbook(os.path.join(root_path,'train','record.xlsx'))
        worksheet = workbook.add_worksheet()
        worksheet.write(0, 0, "Loss")
        worksheet.write(0, 1, "Accuracy(3-classes)")
        worksheet.write(0, 2, "Accuracy(Binary)")


        #Shuffle Features  and Targets
        p = np.random.permutation(len(targets))
        features = features[p]
        targets = targets[p]

        #Run Test, Get Accuracy 
        score = []
        n_true = 0
        for idx, t_target in enumerate(test_targets):
            pred, soft= sess.run([pred_classes, pred_probs], feed_dict={x:[test_features[idx]], keep_prob:1.0})
            
            if pred[0] == t_target:
                n_true += 1

            score.append(soft[0][1])
            
        #Plot Accuracy Test
        acc_data.append((n_true / len(test_targets)) * 100.0)
        acc_bin.append((n_true / len(test_targets)) * 100.0)
        acc_ax.cla()
        acc_ax.set_ylabel("accuracy")
        acc_ax.set_xlabel("epoch")
        acc_ax.plot(acc_data, 'b-')
        acc_ax.plot(acc_bin, 'b--')
        plt.pause(1e-45)


        #Save everytime Runing Test
        save_data(worksheet, loss_data, acc_data, acc_bin)
        workbook.close()


        #Run Training
        if epoch == 12: lr=0.0002
        idx = 0
        while idx < len(features):
            input_feed = features[idx:idx+batch_size]
            label_feed = targets[idx:idx+batch_size]
            idx = idx+batch_size

            # print(input_feed.shape)
            # print(label_feed.shape)
            # print(label_feed)
            _,loss_out = sess.run([optimize, loss_op], feed_dict={x:input_feed, y_true:label_feed, learning_rate:lr, keep_prob:0.0})
            loss_data.append(loss_out)

            if epoch < 2:     
                print(loss_out)
                loss_ax.cla()
                loss_ax.set_ylabel("loss")
                loss_ax.set_xlabel("iterations")
                loss_ax.plot(loss_data, 'r-')
                plt.pause(1e-10)
        save_path = os.path.join(file_path, 'saved', 'epoch'+str(epoch)+'model')
        builder = tf.saved_model.builder.SavedModelBuilder(save_path)
        builder.add_meta_graph_and_variables(sess, ['foo-tag'])
        builder.save()

    plt.show()
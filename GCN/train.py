from __future__ import division
from __future__ import print_function
import pdb
import time
import tensorflow as tf
import random
from utils import *
from models import GCN,MLP

# Set random seed
seed = 400
random.seed(400)
np.random.seed(seed)
tf.set_random_seed(seed)

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('dataset', 'cora', 'Dataset string.')  # 'cora', 'citeseer', 'pubmed'
flags.DEFINE_string('model', 'gcn', 'Model string.')  # 'gcn', 'gcn_cheby', 'dense'
flags.DEFINE_float('learning_rate', 0.00005, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 2000, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 32, 'Number of units in hidden layer 1.')
flags.DEFINE_float('dropout', 0.3, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 5e-4, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_integer('early_stopping', 10, 'Tolerance for early stopping (# of epochs).')
flags.DEFINE_integer('max_degree', 2, 'Maximum Chebyshev polynomial degree.')
flags.DEFINE_integer('feature_dim', 32, 'Number of the member special dim')
flags.DEFINE_integer('hidden_dim', 32, 'Number of the RNN dim')
flags.DEFINE_integer('time_step', 25, 'Number of the member special dim')






# Load data
member_idx,weighted_adj=load_graph_data(FLAGS.dataset)
member_vote, vote_list, description_list=load_vote_data(FLAGS.dataset)
#sample_description,y,y_mask=vote_vector('1',member_vote,member_idx)
feature_dim=FLAGS.feature_dim+FLAGS.hidden_dim
#adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data(FLAGS.dataset)
votenumber=np.array([0,0,0],dtype='float64')
acculist=[]
for item in vote_list:
    sample_description,y,y_mask=vote_vector(str(item),member_vote,member_idx)
    sums=np.sum(y,axis=0)
    votenumber+=sums
    summax=sums.max()
    accu=sums[0]/sums.sum()
    acculist.append(accu)
np.mean(acculist)
# Some preprocessing
#features = preprocess_features(features)
if FLAGS.model == 'gcn':
    #weighted_adj=threshhold(weighted_adj,FLAGS.adj_threshhold)
    support = [preprocess_adj(weighted_adj)]

    num_supports = 1
    model_func = GCN
elif FLAGS.model == 'gcn_cheby':
    support = chebyshev_polynomials(weighted_adj, FLAGS.max_degree)
    num_supports = 1 + FLAGS.max_degree
    model_func = GCN
elif FLAGS.model == 'dense':
    #support = [preprocess_adj(adj)]  # Not used
    num_supports = 1
    model_func = MLP
else:
    raise ValueError('Invalid argument for model: ' + str(FLAGS.model))

# Define placeholders
placeholders = {

    'support': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
    "description": tf.placeholder(tf.int32,shape=(1,FLAGS.time_step)),
    "word_number":tf.placeholder_with_default(len(description_list),shape=()),
    'features': tf.sparse_placeholder(tf.float32, shape=tf.constant((y.shape[0],feature_dim), dtype=tf.int64)),
    'labels': tf.placeholder(tf.float32, shape=(y.shape[0], y.shape[1])),
    'labels_mask': tf.placeholder(tf.int32),
    'dropout': tf.placeholder_with_default(0., shape=()),
    'num_features_nonzero': tf.placeholder(tf.int32)  # helper variable for sparse dropout
}
# Create model
model = model_func(placeholders,feature_dim ,len(description_list), logging=True)

# Initialize session
sess = tf.Session()


# Define model evaluation function
def evaluate(description, support, labels, mask, placeholders):
    t_test = time.time()
    feed_dict_val = construct_feed_dict(description, support, labels, mask, placeholders)
    outs_val = sess.run([model.loss, model.accuracy], feed_dict=feed_dict_val)
    return outs_val[0], outs_val[1], (time.time() - t_test)


# Init variables




# Train model
np.random.shuffle(vote_list)

def cross_validation(dataset,k):
    number=len(dataset)
    step=number//k
    train_list=[]
    for i in range(k):
        if step*(i+2)>number:
            test_vote=dataset[step*(i):]
            train_vote=dataset[0:step*(i)]
        elif i==0:
            test_vote=dataset[step*(i):step*(i+1)]
            train_vote=dataset[step*(i+1):]
        else:
            test_vote=dataset[step*(i):step*(i+1)]
            train_vote=np.append(dataset[0:step*(i)],dataset[step*(i+1):])
        print(step*i)
        train_list.append((train_vote,test_vote))
    return train_list

k_folds=5
train_list=cross_validation(vote_list,k_folds)
accuracy_test=[]
for item in train_list:

    train_vote=item[0]
    test_vote=item[1]

    sess.run(tf.global_variables_initializer())
    cost_val = []
    for epoch in range(FLAGS.epochs):

        t = time.time()
        # Construct feed dictionary
        item=random.choice(train_vote)
        #while item  in list(test_vote):
        #    item = random.randint(1,len(vote_list))
        print(item)
        description, y, y_mask = vote_vector(str(item), member_vote, member_idx)
        y_train_mask,y_val_mask=valtest(y,y_mask,vote_list)
        description=padding(description,FLAGS.time_step)
        description.shape=(FLAGS.time_step,1)
        description=np.transpose(description)
        #print(description)
        feed_dict = construct_feed_dict(description, support, y, y_mask, placeholders)
        feed_dict.update({placeholders['dropout']: FLAGS.dropout})

        # Training step
        outs = sess.run([model.opt_op, model.loss, model.accuracy], feed_dict=feed_dict)

        # Validation
        cost, acc, duration = evaluate(description, support, y, y_val_mask, placeholders)
        cost_val.append(cost)

        # Print results

        print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(outs[1]),
              "train_acc=", "{:.5f}".format(outs[2]), "val_loss=", "{:.5f}".format(cost),
              "val_acc=", "{:.5f}".format(acc), "time=", "{:.5f}".format(time.time() - t))

        #if epoch > FLAGS.early_stopping and cost_val[-1] > np.mean(cost_val[-(FLAGS.early_stopping+1):-1]):
            #print("Early stopping...")
            #break

    print("Optimization Finished!")
    test_cost_all=0
    test_acc_all=0
    test_duration_all=0
    test_acclist=[]
    for item in test_vote:
        description, y, y_mask = vote_vector(str(item), member_vote, member_idx)
        description=padding(description,FLAGS.time_step)
        description.shape=(FLAGS.time_step,1)
        description=np.transpose(description)
        test_cost, test_acc, test_duration = evaluate(description, support, y, y_mask, placeholders)
        test_acclist.append(test_acc)
        test_cost_all+=test_cost
        test_acc_all+=test_acc
        test_duration_all+=test_duration
    test_cost_all=test_cost_all/len(test_vote)
    test_acc_all=test_acc_all/len(test_vote)
    test_duration_all=test_duration_all/len(test_vote)
    print("Test set results:", "cost=", "{:.5f}".format(test_cost_all),
         " #accuracy=", "{:.5f}".format(test_acc_all), "time=", "{:.5f}".format(test_duration_all))
    accuracy_test.append(test_acc_all)
test_final=np.array(accuracy_test)
score=test_final.sum()/len(test_final)
print("cross_validation result:","#accuracy=","{:,f}".format(score))
#import matplotlib.pyplot as plt
#test_acclist.sort()
#plt.plot(np.array(test_acclist))
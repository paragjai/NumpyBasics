# linearClassification using Perceptron Model

import numpy as np
from matplotlib import pyplot as plt

no_of_data_points = 100 + 100 # 100 in cluster 1 and another 100 in cluster 2

# Generating 2 random clusters of 2D data

cluster1 = 0.2*np.random.randn(100,2) + [1,1] # (100,2) means 100 rows and 2 columns each, [1,1] gets added to each row because of numpy broadcasting
cluster2 = 0.2*np.random.randn(100,2) + [3,3]

# Preparing the data

bias = np.ones(100).reshape(100,1)

cluster1_with_bias = np.concatenate((bias, cluster1), axis=1)
print("cluster1_with_bias", cluster1_with_bias[0:5])
label1 = np.ones((100,1))
print("label1[0:5]", label1[0:5])

cluster2_with_bias = np.concatenate((bias, cluster2), axis=1)
print("cluster2_with_bias", cluster2_with_bias[0:5])
label2 = np.ones((100,1))*-1
print("label2[0:5]", label2[0:5])


X = np.vstack((cluster1_with_bias, cluster2_with_bias))
print("X", X[0:5])
print("X", X[100:105])

# X = np.hstack((np.ones(2*100).reshape(2*100,1), np.vstack((cluster1,cluster2)))) # (200,3), ones are appended to accomodate the bias vector
print("X.shape", X.shape) # (200,3) # x_i = [1 x1 x2], so w should be [-c -m 1]
# print("X[0:5]", X[0:5])

# Ground-truth labels, Y
Y = np.vstack((label1,label2))
print("Y[0:5]", Y[0:5])
print("Y[100:105]", Y[100:105])
# Y = np.vstack( (-1*np.ones(100).reshape(100,1), np.ones(100).reshape(100,1) ) ) # (200,1) with labels for cluster1 as -1 and for cluster2 as +1


w = np.array([0.0, 0.0, 0.0]) # We could have initialized it to [0, 0, 1] as well
w_old = np.array([0.0, 0.0, 0.0]) # We could have initialized it to [0, 0, 1] as well

max_iterations = 500 # Variable, try changing it and observe the output.
learning_rate = 5E-2 # 0.05, a hyper-parameter
delta = 1E-7  # a hyper-parameter

# y = mx + c
# y - mx - c = 0
# [1 -m -c].dot(y x 1) = 0, here y and x both are features. y is not a label here. (y x 1) is similar to (x2 x1 1). Since it is CLASSIFICATION we don't talk about
# that y is dependent and x is independent. Instead we say that, x1 and x2 are the features cummulatively used to describe the data labelled as y_i.
# Here, instead of y, see it as x2. y_hat_i is the predicted label for np.sign(w.dot(x_i)) and y_i is the ground-truth label for x_i
# w.dot(x) = 0

#  y = mx + c is the decision boundary
# y - mx - c = 0 is the decision boundary, m = -1*w[1], c = -1*w[0] => m = (-1*w[1])/w[2], c = (-1*w[0])/w[2]

# For a particular x_i, if w.dot(x_i) > 0, then y_hat = +1
#                     , if w.dot(x_i) = 0, then x_i is a point which falls on the decision boundary itelf,
#                       and decision boundary is a NO MAN'S LAND. Therefore, x_i neither belongs to class +1 or class -1
#                       y_hat = 0
#                     , if w.dot(x_i) = 0, then y_hat = -1

# This is referred to as the sign function. In numpy, we use it as np.sign(w.dot(x_i)


# Perceptron Model says that:
    # w and w_old initialized to zero. Remember w is a vector
    # For any training example x_i,
        # if   np.sign(w.dot(x_i)) is not equal to y_i (the ground-truth label) i.e., y_hat for that particular x_i ! = y_i
        # then update the weights with w = w + learning_rate * y_i * x_i (Here, we are adding to the weight as opposed to what we did in Linear Regression. Also in
        # Linear Regression, weights were updated only once per iteration. Here, they are update once for every training example if the condition is met.)

    # Do the above for EACH or ALL of the training example x_i


    # Then,
        # if np.sum(w_old-w)/ no_of_data_points  < delta :
        # then we have learnt our weights and we break out of the loop

    # Else,
        # if this_iteration == max_iterations and we are here it means we have not learnt the desired weights
        # Print : We have not learnt the desired weights. Try for few less or few more number of iterations.
        # else:
        # w_old = w
        # And now do the entire process for one more iteration.


for _ in range(0, max_iterations):

    for i in range(0,no_of_data_points):
        x_i = X[i,:]
        y_i = Y[i]
        y_hat_i = np.sign(w.dot(x_i))


        if y_i != y_hat_i:
            w = w + learning_rate * y_i * x_i

    numerator = float(np.abs(np.sum(w_old-w)))
    denominator = no_of_data_points

    if numerator/denominator < delta: # similar to tf.reduce_mean(w_old - w)
        print("We learnt the desired Weights in " + str(_) + " iterations.")
        print("Also called as 'CONVERGED' :P")
        break

    w_old = w

    elif _ == max_iterations-1:
        print("We have not learnt the desired weights. Try for few less or few more number of iterations.")




print("Weights learned:", w)
# print("Label 1:", np.sign(w.dot(cluster1)))

# Slope, m
m = -w[1]/float(w[2])

# Intercept, c
c = -w[0]/float(w[2])

data_points_for_best_fit_line = np.linspace(start=np.min(X[:,1]), stop=np.max(X[:,1 ]), num=10)
best_fit_line = -1*(float(w[1])/w[2])*data_points_for_best_fit_line + -1*w[0]/float(w[2])
print("W:", w)

def accuracy(ground_truth_labels, cluster, weight):
    true = 0
    for i, x_i in enumerate(cluster):
        # print(, ground_truth_labels[i])
        y_pred_cls = np.sign(weight.dot(x_i))
        if y_pred_cls == ground_truth_labels[i]:
                true=true+1

    print("accuracy", float(true/len(ground_truth_labels))*100)

# Checking accuracy on training data itself. BAD PRACTICE!!!
accuracy(label1, cluster1_with_bias, w)
accuracy(label2, cluster2_with_bias, w)

# Plotting the clusters
plt.plot(cluster1[:,0], cluster1[:,1], 'ro', cluster2[:,0], cluster2[:,1], 'bo', data_points_for_best_fit_line, best_fit_line, 'k-')
# plt.plot(best_fit_line, 'g-')
plt.show()

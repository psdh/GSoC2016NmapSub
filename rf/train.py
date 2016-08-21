#!/usr/bin/env python

import ctypes
import getopt
import sys
import tempfile

import numpy as np

sys.path.append("liblinear-1.8/python")
import liblinear
import liblinearutil as ll

# import the opencv library!
import cv2

import common
import impute
import parse
import vectorize

global num_features
num_features = 695

global s1groups
s1groups = ["BSD", "Linux", "Windows", "Macintosh", "Others"]

def make_liblinear(groups, size, num_features):

    # Type of the target variable is pivotal in making opencv consider this a classification problem
    # make sure that you use varType in trainData to ensure this :)
    # np.uint32 doesn't work with cv2 (it simply doesn't allow using those types)
    y = np.zeros((size), np.uint64)
    x = np.zeros((size, num_features), np.float32)

    x_cout = 0
    y_cout = 0
    numerical_label = 0
    for group in groups:
        for features in group.features:
            y[y_cout] = numerical_label
            y_cout += 1
            x[x_cout] = np.asarray(list(features))
            x_cout += 1
        numerical_label += 1

    return y, x

def scale(features):
    m, n = features.shape
    s_min = np.zeros(n)
    s_max = np.zeros(n)
    for i in range(n):
        assigned = [x for x in features[:,i] if x >= 0]
        if assigned:
            mn = min(assigned)
            mx = max(assigned)
        else:
            mn = 0.0
            mx = 0.0
        s_min[i] = mn
        s_max[i] = mx
        if mn == mx:
            denom = 1.0
        else:
            denom = mx - mn
        for j in range(m):
            if features[j, i] >= 0:
                features[j, i] = (features[j, i] - mn) / denom
    return features, s_min, s_max

def prepare_features(groups, do_scale):
    """Impute and scale features, and assign them back to groups."""
    feature_list = []
    for group in groups:
        for features in group.features:
            feature_list.append(features)

    feature_matrix = np.vstack(feature_list)
    feature_matrix = impute.impute(feature_matrix)
    if do_scale:
        feature_matrix, s_min, s_max = scale(feature_matrix)
        scale_params = zip(s_min, s_max)
    else:
        scale_params = None

    f_i = iter(feature_matrix)
    for group in groups:
        group.features = []
        for i in range(len(group.rs_list)):
            group.features.append(f_i.next())

    return scale_params

def train_liblinear(train_x, train_y, test_x, test_y, group_names, cost):
    # -s 0: L2-regularized logistic regression (primal)
    # -q:   quiet mode
    # -c cost: set the parameter C (default 1)
    param_str = "-s 0 -q -c " + str(cost)
    param = ll.parameter(param_str)
    prob = ll.problem(train_y, train_x)

    print "Training LibLinear model now"
    model = ll.train(prob, param)
    print "Model trained"

    confusion_matrix = np.zeros(shape=(len(group_names), len(group_names)), dtype="int64")

    p_labels, p_acc, _ = ll.predict(test_y, test_x, model)

    acc = 0
    for i in range(len(test_y)):
        confusion_matrix[p_labels[i]][test_y[i]] += 1;
        if p_labels[i] != test_y[i]:
            print group_names[test_y[i]], " | ", group_names[p_labels[i]]
        else:
            acc += 1

    print "Number of correct predictions is:", acc

    # uncomment to generate precision and recall too
    # getPrecisionRecall(confusion_matrix, group_names)

def getZippedGroupCount(y):
    z = np.bincount(y)
    k = np.nonzero(z)[0]

    return zip(k, z[k])


def getGroupCount(y):
    return np.bincount(y)

def getPrecisionRecall(confusion_matrix, group_names):
    col_tot = np.sum(confusion_matrix, axis=0)
    row_tot = np.sum(confusion_matrix, axis=1)

    av_prec = 0.0
    av_reca = 0.0

    for i in range(confusion_matrix.shape[0]):
        recall = float(confusion_matrix[i][i])/float(col_tot[i]) if col_tot[i] != 0 else -1
        precision = float(confusion_matrix[i][i])/float(row_tot[i]) if row_tot[i] != 0 else -1

        print "Precision: ", precision
        print "Recall: ", recall

        # calculate average precision here, understand the number of entries that should come as output


def getS1Group(y, i, group_names):
    if ("bsd" in group_names[y[i]].lower()):
            return 0
    elif ("linux" in group_names[y[i]].lower()):
        return 1
    elif ("windows" in group_names[y[i]].lower()):
        return 2
    elif ("apple" in group_names[y[i]].lower() or "mac" in group_names[y[i]].lower()):
        return 3
    else:
        return 4

def getTrainTestS2(s2buckets, x, y):
    train_buckets = []
    y_buckets = []

    stageonecounts = []

    for i in range(len(s2buckets)):
        stageonecounts.append(len(s2buckets[i]))
        train = np.zeros(shape=(stageonecounts[i], 695), dtype=np.float32)
        pred_f = np.zeros(shape=(stageonecounts[i]), dtype=np.uint64)

        for k in range(len(s2buckets[i])):
            train[k] = x[s2buckets[i][k]]
            pred_f[k] = y[s2buckets[i][k]]

        train_buckets.append(train)
        y_buckets.append(pred_f)

    return (train_buckets, y_buckets)

def trainStage1(train_x, k):
    # training a random forest model here
    model = cv2.ml.RTrees_create()

    # Setting parameters
    var_types = np.array([cv2.ml.VAR_NUMERICAL] * 695 + [cv2.ml.VAR_CATEGORICAL], np.uint32)
    model.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER, 400, 0.01))

    model.setCalculateVarImportance(True)
    model.setActiveVarCount(num_features)
    model.setMaxDepth(70)
    # 1% of size of training dataset: recommended
    model.setMinSampleCount(3)
 
    # print "Training Stage One model now"
    model.train(train_x, cv2.ml.ROW_SAMPLE, k)

    return model


def predict_ms(s1results, x, s2models, y, group_names):
    num_cases = len(s1results)

    acc = 0
    test_x = np.zeros((1, 695), np.float32)
    for i in range(num_cases):
        res = int(s1results[i])
        model = s2models[res]


        test_x[0] = x[i]
        # print test_x
        finres = model.predict(test_x, flags=cv2.ml.DTREES_PREDICT_AUTO)[1][0][0]

        if (finres == y[i]):
            acc += 1
        else:
            print "Stage 2 wrong prediction", group_names[int(finres)], "||", group_names[y[i]]

    print "Stage 2 accuracy", acc

    import sys
    sys.exit(0)


def train(train_groups, test_groups, cost, trainsize_db, test_size_db):
    num_features = 695
    y, x = make_liblinear(train_groups, train_size_db, num_features)

    # test_y, test_x = make_liblinear(test_groups, test_size_db, num_features)

    # training a random forest model here
    model = cv2.ml.RTrees_create()

    print "length of x is: ", str(len(x[0]))
    print "length of y is: ", len(y)

    test_nums = []

    num_groups = len(y)
    counts = getGroupCount(y.astype('int64'))

    from names_group import group_names
    from names_group_old import group_names_test

    # remove for testing on stuff other than nmap.groups file
    group_names_test = group_names
    # group_names = group_names_test

    s1 = np.empty_like(y);
    s2buckets = [[], [], [], [], []]
    import random


    num_testing_db = 40

    random_sel = False

    if random_sel == True:
        for i in range(num_testing_db):
            r = random.randrange(trainsize_db)

            while r in test_nums:
                r = random.randrange(trainsize_db)

            test_nums.append(r)
    else:
        # taking 80% of fps into training (group wise)
        x_counter = 0
        group_no = 0
        group_c = 0

        import math
        for i in range(0, len(y)):

            if y[i] == group_no:
                group_c += 1
            else:
                if group_c == 1:
                    pass
                elif group_c < 5:
                    test_nums.append(i - 1)
                elif group_c >= 5:
                    c = float(group_c) * 0.2
                    twenty = int(math.ceil(c))
                    for k in range(0, twenty):
                        test_nums.append(i - 1 - k)

                print group_c
                group_c = 1
                group_no = y[i]


    num_testing_db = len(test_nums)
    l = len(y)
    l -= len(test_nums)

    print "length: test_nums", len(test_nums)
    train_y = np.zeros((l), np.uint64)
    train_x = np.zeros((l, 695), np.float32)
    train_k = np.empty_like(train_y)

    test_y = np.zeros((num_testing_db), np.uint64)
    test_x = np.zeros((num_testing_db, 695), np.float32)
    test_k = np.empty_like(test_y)

    j = k = 0

    for i in range(len(y)):
        if i in test_nums:
            test_x[j] = x[i]
            test_y[j] = y[i]
            s1[i] = getS1Group(y, i, group_names)
            # won't append these in s2buckets (don't have to train them)
            # s2buckets[s1[i]].append(i)
            test_k[j] = s1[i]

            j += 1
        else:
            train_x[k] = x[i]
            train_y[k] = y[i]
            s1[i] = getS1Group(y, i, group_names)
            s2buckets[s1[i]].append(i)
            train_k[k] = s1[i]
            k += 1

    print "Here are the test prints:"
    for i in range(len(test_nums)):
        print test_y[i],":", group_names_test[test_y[i]]

    print "\nLibLinear:"
    train_liblinear(train_x.tolist(), train_y.tolist(), test_x.tolist(), test_y.tolist(), group_names, cost=100)

    print "\n Now Random Forest:"

    var_types = np.array([cv2.ml.VAR_NUMERICAL] * 695 + [cv2.ml.VAR_CATEGORICAL], np.uint32)
    # Setting parameters
    model.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER, 400, 0.01))

    model.setCalculateVarImportance(True)
    model.setActiveVarCount(num_features)
    model.setMaxDepth(70)
    # 1% of size of training dataset: recommended
    model.setMinSampleCount(3)
 
    print "Training Model now!"
    model.train(train_x, cv2.ml.ROW_SAMPLE, train_y)

    print "Training done!"
    print "Now saving model"

    model.save("rfmodel")

    accr = 0

    # model.predict returns a tuple (retval, list_predicted_values)
    # res is list_predicted_values
    res1 = model.predict(test_x, flags=cv2.ml.DTREES_PREDICT_AUTO)[1]
    # res2 produces a seg fault
    res2 = model.predict(test_x,flags=cv2.ml.DTREES_PREDICT_MAX_VOTE)[1]
    res3 = model.predict(test_x, flags=cv2.ml.DTREES_PREDICT_SUM)[1]
    res4 = model.predict(test_x,flags=cv2.ml.DTREES_PREDICT_MASK)[1]

    confusion_matrix = np.zeros(shape=(num_groups, num_groups), dtype="int64")

    for i in range(len(test_x)):
        comp1 = int(test_y[i])
        comp2 = int(res1[i])

        confusion_matrix[comp2][comp1] += 1

        comp1 = group_names_test[comp1]
        comp2 = group_names[comp2]

        if comp1 == comp2:
            accr += 1
        else:
            print comp1, " | ", comp2

    print "Number of correct predictions is:", accr

    # uncomment to generate precision and recall too
    # getPrecisionRecall(confusion_matrix, group_names)

    print "\n\n\n Multi-Stage Random Forest model now!"


    # get train and test buckets for stage 2 classification
    train_buckets, y_buckets = getTrainTestS2(s2buckets, x, y)

    model = trainStage1(train_x, train_k)

    # Training models for stage 2 now

    s2models = []

    for i in range(len(s2buckets)):
        # print "Stage 2: Training ", str(i) + "th model"
        mod1 = cv2.ml.RTrees_create()

        mod1.setMinSampleCount(1)
        mod1.setMaxDepth(20)
        mod1.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER, 50, 0.01))
        mod1.setActiveVarCount(num_features)

        mod1.train(train_buckets[i], cv2.ml.ROW_SAMPLE, y_buckets[i])

        res = mod1.predict(train_buckets[i], flags=cv2.ml.DTREES_PREDICT_AUTO)[1]
        acc = 0
        for kk in range(len(res)):
            if res[kk] == y_buckets[i][kk]:
                acc += 1

        # print "Stage 2 model", i,"Accuracy = ", acc, "length", len(res)
        # print "\n\n\n"

        s2models.append(mod1)


    s1results = model.predict(test_x, flags=cv2.ml.DTREES_PREDICT_AUTO)[1]
    # print s1results

    acc = 0

    for i in range(len(s1results)):
        if s1results[i] == test_k[i]:
            acc += 1
        else:
            print "Stage 1: Wrong prediction", "Group name: ", group_names[y[i]], "Expected: ", s1groups[s1[i]], "Predicted: ", s1groups[int(s1results[i][0])]
    print "Stage 1 Accuracy is", acc
    print
    predict_ms(s1results, test_x, s2models, test_y, group_names)


    import sys
    sys.exit(0)

    # train a similar model (using the sklearn library)
    from sklearn.ensemble import RandomForestClassifier

    clf = RandomForestClassifier(n_estimators=400, n_jobs=4, min_samples_split=3)
    clf.fit(x, y)

    accr = 0
    for i in range(test_size_db):
        pred_prob = clf.predict_proba(np.array(test_x[i]).reshape(1, -1))

        pred = pred_prob[0].argsort()[-3:][::-1]
        # print pred

        print "expected:", test_y[i],"predicted:- 1)", pred[0], ":", round(100 * pred_prob[0][pred[0]], 3) ,
        print "2)", pred[1], ":", round(100 * pred_prob[0][pred[1]], 3),
        print "3)", pred[2], ":", round(100 * pred_prob[0][pred[2]], 3)

        if group_names[int(pred[0])] == group_names_test[int(test_y[i])]:
            accr += 1

    print "(Sklearn) New Accuracy: ", accr

    return model


def find_same(names):
    for i in range(96):
        print i,
        for j in range(i+1, 96):
            if names[i] == names[j]:
                print j,
        print


def main(set_filename, train_group_filename, test_group_filename, cost, do_scale, train_size_db, test_size_db):
    feature_names = parse.parse_feature_set_file(set_filename)

    train_groups = parse.parse_groups_file(train_group_filename)
    test_groups = parse.parse_groups_file(test_group_filename)

    for group in train_groups:
        group.features = []

        for rs in group.rs_list:
            features = vectorize.vectorize(feature_names, rs)
            group.features.append(features)

    for group in test_groups:
        group.features = []

        for rs in group.rs_list:
            features = vectorize.vectorize(feature_names, rs)
            group.features.append(features)

    scale_params = prepare_features(train_groups, do_scale)

    scale_params_test = prepare_features(test_groups, do_scale)

    print >> sys.stderr, "Training with cost = ", cost, " not sure if this will be of help here"

    model = train(train_groups, test_groups, cost, train_size_db, test_size_db)


if __name__ == "__main__":
    train_size_db = 301
    train_group = "db_att1_one_per_class.groups"

    test_group = "db_att1_one_per_class.groups"
    test_size_db = 301
    main("nmap.set", train_group, test_group, 100, True, train_size_db, test_size_db)

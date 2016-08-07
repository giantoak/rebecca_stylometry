from sklearn import linear_model
import numpy
from data import Data
from features import Features
from values import Values
from io import StringIO
from sklearn.metrics import classification_report
from sklearn import svm
import json
from string import digits
#import pybrain
#from pybrain.datasets import SupervisedDataSet
#from pybrain.datasets            import ClassificationDataSet
#from pybrain.utilities           import percentError
#from pybrain.tools.shortcuts     import buildNetwork
#from pybrain.supervised.trainers import BackpropTrainer
#from pybrain.structure.modules   import SoftmaxLayer

class Model(object):
    
    def __init__(self, C, path):
        self.C = C
        self.path = path
        self.lr = linear_model.LogisticRegression(C=self.C,class_weight={0:1,1:1})
        self.all_names = []
        name_file = open('/Users/cusgadmin/BackPageStylometry/data-drop-1/extractions/BackupFiles/names_uniq.tsv', 'r')
        for line in name_file: self.all_names.append(line.split("\t")[0].lower().rstrip())
        
    def train(self, X, y):
        self.lr.fit(X, y)
        return self.lr
    
    def json_blob(self, top, _id):
        data = {
                'cdr_id' : _id,
                'similar_cdr_ids' : list(top[:,2]),
                'similarity_scores' : list(map(float,top[:,1]))
                }
        return json.dumps(data)
    
    def get_top_similar(self, X, N, _ids, _id):
        pred = self.lr.predict(X)
        p = self.lr.predict_proba(X)
        probs = p[:,1] #get the probabilities that the class=same]
        all = numpy.column_stack((map(int,pred), map(float,probs), _ids))
        print all
        sorted_top = all[all[:,1].argsort()[::-1]][:N]
        return self.json_blob(sorted_top, _id)
    
    def extract_comparison_features(self, cdr_id, auth, pwd):
        d = Data(cdr_id, auth, pwd)
        t_title, t_text, t_extract_time = d.get_target_ad()
        t_title = t_title.replace('- backpage.com','')
        t_title = t_title.replace(t_title.split()[-1],'')
        #t_title = t_title.replace('escorts','')
        
        comparison_set = d.get_comparison_set(t_extract_time)
        
        X = numpy.empty([len(comparison_set), 28]) #the number of comparison instances by the number of features
        _ids = []
        print t_title
        print t_text
        print
        
        #for each item in comparison set, extract the features with the target ad
        target_val = Values(t_text, self.path, self.all_names)
        target_title_val = Values(t_title, self.path, self.all_names)
        i = 0
        for k,v in comparison_set.items():
            v_title = v[0]
            v_text = v[1]

            v_title = v_title.replace('- backpage.com','')
            v_title = v_title.replace(v_title.split()[-1],'') 
            #v_title = v_title.replace('escorts','')
        
            print v_title
            print v_text 
            comparison_val = Values(v_text, self.path, self.all_names)
            f_text = Features(target_val, comparison_val)
            f_text_feat = f_text.extract_feats(False)
            
            comparison_title_val = Values(v_title, self.path, self.all_names)
            f_title = Features(target_title_val, comparison_title_val)
            f_title_feat = f_title.extract_feats(True)
            
            X[i] = f_text_feat + f_title_feat
            print len(X[i])
            print X[i]
            print
            _ids.append(k)
            i = i + 1
        
        return X, _ids


if __name__ == '__main__':
    
    path_to_jars = '/Users/cusgadmin/BackPageStylometry/data-drop-1/extractions/groundTruth/FindMatchings/src/'
    
    #initialize the model
    lm = Model(1.0, path_to_jars)

    #get the training data. for now, using one of my original datasets
    X = numpy.loadtxt('/Users/cusgadmin/BackPageStylometry/data-drop-1/extractions/ARFF/go_1small.txt')
    X = numpy.delete(X,5, axis=1)
    X_names = numpy.loadtxt('/Users/cusgadmin/BackPageStylometry/data-drop-1/extractions/ARFF/go_1nameFeatsmall.txt')
    X = numpy.column_stack((X, X_names))
    
    X_titles = numpy.loadtxt('/Users/cusgadmin/BackPageStylometry/data-drop-1/extractions/ARFF/go1TITLESsmall.txt')

    X = numpy.concatenate((X, X_titles), axis = 1)
    y = numpy.empty([7500])
    for i in range(0, 7500):
        if (i <= 4999) : y[i] = 0
        else : y[i] = 1
    
    #train the model
    lm.train(X,y)
    
    #get the comparison data
    _id = 'F6B69F2CD3E2701373745DB9F86459AD438DE74A25CA1F4B9DACFCF078BBEEEC'
    auth = 'cdr-memex'
    pw = '5OaYUNBhjO68O7Pn'
    X_comparison, _ids_comparison = lm.extract_comparison_features(_id, auth, pw)
    print len(X_comparison)
    print _ids_comparison
    
    top_json = lm.get_top_similar(X_comparison, 2, _ids_comparison, _id)
    print top_json
    
    #REBECCA'S ORIGINAL MODEL
    '''
    X1 = numpy.loadtxt('/Users/cusgadmin/BackPageStylometry/data-drop-1/extractions/ARFF/go_1small.txt')
    X1 = numpy.delete(X1,5, axis=1)
    X1_names = numpy.loadtxt('/Users/cusgadmin/BackPageStylometry/data-drop-1/extractions/ARFF/go_1nameFeatsmall.txt')
    X1 = numpy.column_stack((X1, X1_names))
    
    X1_titles = numpy.loadtxt('/Users/cusgadmin/BackPageStylometry/data-drop-1/extractions/ARFF/go1TITLESsmall.txt')
    #X1_titles = numpy.delete(X1_titles, 0, axis=1)
    #X1_titles = numpy.delete(X1_titles, 0, axis=1)
    #X1_titles = numpy.delete(X1_titles, 3, axis=1)
    #X1_titlesnames = numpy.loadtxt('/Users/cusgadmin/BackPageStylometry/data-drop-1/extractions/ARFF/go_1_titlenameFeatsmall.txt')
    #X1_titles = numpy.column_stack((X1_titles, X1_titlesnames))
    
    X1 = numpy.concatenate((X1, X1_titles), axis = 1)
    y = numpy.empty([7500])
    #y = numpy.empty([10000])
    for i in range(0, 7500):
    #for i in range(0, 10000):
        if (i <= 4999) : y[i] = 0
        else : y[i] = 1
        
    X2 = numpy.loadtxt('/Users/cusgadmin/BackPageStylometry/data-drop-1/extractions/ARFF/go_2small.txt')
    X2 = numpy.delete(X2,5, axis=1)
    X2_names = numpy.loadtxt('/Users/cusgadmin/BackPageStylometry/data-drop-1/extractions/ARFF/go_2nameFeatsmall.txt')
    X2 = numpy.column_stack((X2, X2_names))
    
    X2_titles = numpy.loadtxt('/Users/cusgadmin/BackPageStylometry/data-drop-1/extractions/ARFF/go2TITLESsmall.txt')
    #X2_titles = numpy.delete(X2_titles, 0, axis=1)
    #X2_titles = numpy.delete(X2_titles, 0, axis=1)
    #X2_titles = numpy.delete(X2_titles, 3, axis=1)
    #X2_titlesnames = numpy.loadtxt('/Users/cusgadmin/BackPageStylometry/data-drop-1/extractions/ARFF/go_2_titlenameFeatsmall.txt')
    #X2_titles = numpy.column_stack((X2_titles, X2_titlesnames))

    X2 = numpy.concatenate((X2, X2_titles), axis = 1)
    
    X3 = numpy.loadtxt('/Users/cusgadmin/BackPageStylometry/data-drop-1/extractions/ARFF/go_3small.txt')
    X3 = numpy.delete(X3,5, axis=1)
    X3_names = numpy.loadtxt('/Users/cusgadmin/BackPageStylometry/data-drop-1/extractions/ARFF/go_3nameFeatsmall.txt')
    X3 = numpy.column_stack((X3, X3_names))
    
    X3_titles = numpy.loadtxt('/Users/cusgadmin/BackPageStylometry/data-drop-1/extractions/ARFF/go3TITLESsmall.txt')
    #X3_titles = numpy.delete(X3_titles, 0, axis=1)
    #X3_titles = numpy.delete(X3_titles, 0, axis=1)
    #X3_titles = numpy.delete(X3_titles, 3, axis=1)
    #X3_titlesnames = numpy.loadtxt('/Users/cusgadmin/BackPageStylometry/data-drop-1/extractions/ARFF/go_3_titlenameFeatsmall.txt')
    #X3_titles = numpy.column_stack((X3_titles, X3_titlesnames))

    X3 = numpy.concatenate((X3, X3_titles), axis = 1)
    
    lr = linear_model.LogisticRegression(penalty = 'l2', dual = False, tol=0.0001, C=10, 
                                         fit_intercept = True, class_weight= {0:1,1:1})
    #lr = ensemble.AdaBoostClassifier(n_estimators=100)
    #print 'training model...'
    lr.fit(X1, y)
    #print 'done'
    
    #print 'testing model...'
    y_pred = lr.predict(X2)
    #print 'done'

    print "Train 1, Test 2"
    FP = 0
    TP = 0
    FN = 0
    TN = 0
    for i in range(0, len(y_pred)):
        if y_pred[i] == y[i]: 
            if y[i] == 1: TP = TP + 1
            if y[i] == 0: TN = TN + 1
        elif y_pred[i] == 1 and y[i] == 0: FP = FP + 1
        elif y_pred[i] == 0 and y[i] == 1: FN = FN + 1
    acc1 = (TP+TN)/float(len(y_pred))
    print acc1
    #print classification_report(y, y_pred)
    fpr1 = FP/float((FP + TN))
    print fpr1
    tpr1 = TP/float((TP + FN))
    print tpr1
    
    print
    print "Train1, Test 3"
    FP = 0
    TP = 0
    FN = 0
    TN = 0
    y_pred = lr.predict(X3)
    for i in range(0, len(y_pred)):
        if y_pred[i] == y[i]: 
            if y[i] == 1: TP = TP + 1
            if y[i] == 0: TN = TN + 1
        elif y_pred[i] == 1 and y[i] == 0: FP = FP + 1
        elif y_pred[i] == 0 and y[i] == 1: FN = FN + 1
    acc2 = (TP+TN)/float(len(y_pred))
    print acc2
    #print classification_report(y, y_pred)
    fpr2 = FP/float((FP + TN))
    print fpr2
    tpr2 = TP/float((TP + FN))
    print tpr2
    
    print
    print "Train2, Test 1"
    FP = 0
    TP = 0
    FN = 0
    TN = 0
    lr.fit(X2, y)
    y_pred = lr.predict(X1)
    for i in range(0, len(y_pred)):
        if y_pred[i] == y[i]: 
            if y[i] == 1: TP = TP + 1
            if y[i] == 0: TN = TN + 1
        elif y_pred[i] == 1 and y[i] == 0: FP = FP + 1
        elif y_pred[i] == 0 and y[i] == 1: FN = FN + 1
    acc3 = (TP+TN)/float(len(y_pred))
    print acc3
    #print classification_report(y, y_pred)
    fpr3 = FP/float((FP + TN))
    print fpr3
    tpr3 = TP/float((TP + FN))
    print tpr3
    
    print
    print "Train2, Test 3"
    FP = 0
    TP = 0
    FN = 0
    TN = 0
    y_pred = lr.predict(X3)
    for i in range(0, len(y_pred)):
        if y_pred[i] == y[i]: 
            if y[i] == 1: TP = TP + 1
            if y[i] == 0: TN = TN + 1
        elif y_pred[i] == 1 and y[i] == 0: FP = FP + 1
        elif y_pred[i] == 0 and y[i] == 1: FN = FN + 1
    acc4 = (TP+TN)/float(len(y_pred))
    print acc4
    #print classification_report(y, y_pred)
    fpr4 = FP/float((FP + TN))
    print fpr4
    tpr4 = TP/float((TP + FN))
    print tpr4
    
    print
    print "Train3, Test 1"
    FP = 0
    TP = 0
    FN = 0
    TN = 0
    lr.fit(X3, y)
    y_pred = lr.predict(X1)
    for i in range(0, len(y_pred)):
        if y_pred[i] == y[i]: 
            if y[i] == 1: TP = TP + 1
            if y[i] == 0: TN = TN + 1
        elif y_pred[i] == 1 and y[i] == 0: FP = FP + 1
        elif y_pred[i] == 0 and y[i] == 1: FN = FN + 1
    acc5 = (TP+TN)/float(len(y_pred))
    print acc5
    #print classification_report(y, y_pred)
    fpr5 = FP/float((FP + TN))
    print fpr5
    tpr5 = TP/float((TP + FN))
    print tpr5
    
    print
    print "Train3, Test 2"
    FP = 0
    TP = 0
    FN = 0
    TN = 0
    y_pred = lr.predict(X2)
    for i in range(0, len(y_pred)):
        if y_pred[i] == y[i]: 
            if y[i] == 1: TP = TP + 1
            if y[i] == 0: TN = TN + 1
        elif y_pred[i] == 1 and y[i] == 0: FP = FP + 1
        elif y_pred[i] == 0 and y[i] == 1: FN = FN + 1
    acc6 = (TP+TN)/float(len(y_pred))
    print acc6
    #print classification_report(y, y_pred)
    fpr6 = FP/float((FP + TN))
    print fpr6
    tpr6 = TP/float((TP + FN))
    print tpr6
    
    print
    print (acc1+acc2+acc3+acc4+acc5+acc6)/6
    print (fpr1+fpr2+fpr3+fpr4+fpr5+fpr6)/6
    print (tpr1+tpr2+tpr3+tpr4+tpr5+tpr6)/6
    
    '''
    '''
    #NEURAL NETWORK
    trndata = SupervisedDataSet(15, 1)
    for i in range(0, 10000):
        if (i <= 4999) : trndata.addSample(X_train[i], (0,))
        else : trndata.addSample(X_train[i], (1,))
    
    tstdata = SupervisedDataSet(15, 1)
    for i in range(0, 10000):
        if (i <= 4999) : tstdata.addSample(X_test[i], (0,))
        else : tstdata.addSample(X_test[i], (1,))
    
    print "Number of training patterns: ", len(trndata)
    print "Input and output dimensions: ", trndata.indim, trndata.outdim
    print "First sample (input, target, class):"
    print trndata['input'][0], trndata['target'][0]#, trndata['class'][0]
    
    fnn = buildNetwork( trndata.indim, 5, trndata.outdim, outclass=SoftmaxLayer )
    trainer = BackpropTrainer( fnn, dataset=trndata, momentum=0.1, verbose=True, weightdecay=0.01)
    for i in range(20): 
        trainer.trainEpochs( 5 )
        trnresult = percentError( trainer.testOnClassData(),trndata['target'] )
        tstresult = percentError( trainer.testOnClassData(dataset=tstdata ), tstdata['target'] )

        print "epoch: %4d" % trainer.totalepochs,"  train error: %5.2f%%" % trnresult, "  test error: %5.2f%%" % tstresult
    
    '''
    
    
    

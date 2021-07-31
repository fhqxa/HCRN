#-------------------------------------
#Project: Hierarchical Few-shot Learning Based on Coarse and Fine-grained Relation Network
# Date: 2020.12.21
# Author: Zhiping Wu
# All Rights Reserved
#-------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
import numpy as np
import task_generator as tg
import sys
import os
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
import math
import argparse
import scipy as sp
import scipy.stats
from sklearn.cluster import SpectralClustering
from sklearn import metrics
from scipy.optimize import linear_sum_assignment as linear_assignment
import warnings
warnings.filterwarnings('ignore')
parser = argparse.ArgumentParser(description="One Shot Visual Recognition")
parser.add_argument("-f","--feature_dim",type = int, default = 64)
parser.add_argument("-p","--relation_coarse_dim",type = int, default = 4)
parser.add_argument("-r","--relation_dim",type = int, default = 8)
parser.add_argument("-w","--class_num",type = int, default = 5)
parser.add_argument("-s","--sample_num_per_class",type = int, default = 1)
parser.add_argument("-b","--batch_num_per_class",type = int, default = 19)
parser.add_argument("-e","--episode",type = int, default= 1000)#500000
parser.add_argument("-t","--test_episode", type = int, default = 2000)
parser.add_argument("-l","--learning_rate", type = float, default = 0.001)
parser.add_argument("-g","--gpu",type=int, default= 0)
parser.add_argument("-u","--hidden_unit",type=int,default=10)
args = parser.parse_args()

FEATURE_DIM = args.feature_dim
RELATION_COARSE_DIM = args.relation_coarse_dim
RELATION_DIM = args.relation_dim
CLASS_NUM = args.class_num
SAMPLE_NUM_PER_CLASS = args.sample_num_per_class
BATCH_NUM_PER_CLASS = args.batch_num_per_class
EPISODE = args.episode
TEST_EPISODE = args.test_episode
LEARNING_RATE = args.learning_rate
GPU = args.gpu
HIDDEN_UNIT = args.hidden_unit
my_lambda = 0.65

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0*np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * sp.stats.t._ppf((1+confidence)/2., n-1)
    return m,h

class CNNEncoder(nn.Module):
    """docstring for ClassName"""
    def __init__(self):
        super(CNNEncoder, self).__init__()
        self.layer1 = nn.Sequential(
                        nn.Conv2d(3,64,kernel_size=3,padding=0),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
                        nn.Conv2d(64,64,kernel_size=3,padding=0),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.MaxPool2d(2))
        self.layer3 = nn.Sequential(
                        nn.Conv2d(64,64,kernel_size=3,padding=1),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU())
        self.layer4 = nn.Sequential(
                        nn.Conv2d(64,64,kernel_size=3,padding=1),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU())

    def forward(self,x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        #out = out.view(out.size(0),-1)
        return out # 64

class RelationNetwork(nn.Module):
    """docstring for RelationNetwork"""
    def __init__(self,input_size,hidden_size):
        super(RelationNetwork, self).__init__()
        self.layer1 = nn.Sequential(
                        nn.Conv2d(64*2,64,kernel_size=3,padding=0),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
                        nn.Conv2d(64,64,kernel_size=3,padding=0),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.MaxPool2d(2))
        self.fc1 = nn.Linear(input_size*3*3,hidden_size)
        self.fc2 = nn.Linear(hidden_size,1)

    def forward(self,x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0),-1)
        out = F.relu(self.fc1(out))
        out = F.sigmoid(self.fc2(out))
        # out = F.softmax(self.fc2(out))
        return out

class GetcoareNetwork(nn.Module):
    def __init__(self):
        super(GetcoareNetwork, self).__init__()
        self.layer1 = nn.Sequential(
                        nn.Conv2d(3,64,kernel_size=3,padding=0),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
                        nn.Conv2d(64,64,kernel_size=3,padding=0),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.MaxPool2d(2))

    def forward(self,x):
        out = self.layer1(x)
        out = self.layer2(out)
        return out #64

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm') != -1:
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        n = m.weight.size(1)
        m.weight.data.normal_(0, 0.01)
        m.bias.data = torch.ones(m.bias.data.size())

def cal_clsvec_init(data, fine_labels, num_class):  #90*64*19*19
    class_vec = np.zeros([num_class, data.shape[1],data.shape[2],data.shape[3]]) # 18*64*19*19
    for i in range(num_class):
        idx = [j for j, x in enumerate(fine_labels) if x == i]
        sigma_cls = np.zeros([data.shape[0], data.shape[1], data.shape[2], data.shape[3]]) 
        for m in range(len(idx)):
            s = data[idx[m], :, :, :]
            avg_s = sum(s) / len(s)
            sigma_cls += avg_s
        vec = sum(sigma_cls) / len(idx)
        class_vec[i] = vec

    return class_vec

def get_coare_real_labels(labels,groups):
    #                       90    18
    coarse_labels = []
    for i in range(len(labels)):
        labels_value = labels[i]
        coarse_value = groups[labels_value]
        coarse_labels.append(coarse_value)
    
    return coarse_labels

def get_new_labels(labels,depend_labels):
    
    new_labels = []
    for c in labels:
        label = []
        for j in range(len(depend_labels)):
            if c == depend_labels[j]:
                label.append(1)
            else:
                label.append(0)
        new_labels.append(label)
    return new_labels

def normalize_2darray(range_min, range_max, matrix):
    x = matrix.shape[0]
    y = matrix.shape[1]
    norm_mat = np.zeros(([x,y]))
    cur_max = max(map(max,matrix))
    cur_min = min(map(min,matrix))
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if i == j:
                continue
            norm_mat[i][j] = (
                ((range_max - range_min)*(matrix[i][j]) / float(cur_max - cur_min)) + range_min)
    
    return norm_mat

def gen_superclass(data, fine_labels, num_class, num_clusters):
    class_vec = cal_clsvec_init(data, fine_labels, num_class)
    aff_mat = np.zeros([num_class, num_class])
    for a in range(0, num_class - 1):
        for b in range(a + 1, num_class):
            distance = np.linalg.norm(class_vec[a] - class_vec[b])
            aff_mat[a, b] = distance
            aff_mat[b, a] = aff_mat[a, b]
    beta = 0.1
    # aff_mat = normalize_2darray(0.0, 1.0, aff_mat)
    aff_mat = np.exp(-beta * aff_mat / aff_mat.std())
    for i in range(num_class):
        aff_mat[i, i] = 0.0001
    sc = SpectralClustering(num_clusters,n_init=10, affinity='precomputed',n_neighbors=10, assign_labels='kmeans') 
    groups = sc.fit_predict(aff_mat)

    return groups

def NMI(y_true,y_pred):
    return metrics.normalized_mutual_info_score(y_true,y_pred)

def ARI(y_true,y_pred):
    return metrics.adjusted_mutual_info_score(y_true,y_pred)

def main():
    # Step 1: init data foldersd
    print("init data folders")
    # init character folders for dataset construction
    metatrain_folders,metatest_folders = tg.omniglot_character_folders()
    # Step 2: init neural networks
    print("init neural networks")

    feature_encoder = CNNEncoder()
    relation_coarse_network = RelationNetwork(FEATURE_DIM,RELATION_COARSE_DIM)
    relation_network = RelationNetwork(FEATURE_DIM,RELATION_DIM)
    Coare_encoder = GetcoareNetwork()

    feature_encoder.apply(weights_init)
    relation_coarse_network.apply(weights_init)
    relation_network.apply(weights_init)
    Coare_encoder.apply(weights_init)

    feature_encoder.cuda(GPU)
    relation_coarse_network.cuda(GPU)
    relation_network.cuda(GPU)
    Coare_encoder.cuda(GPU)
    mse = nn.MSELoss().cuda(GPU)

    feature_encoder_optim = torch.optim.Adam(feature_encoder.parameters(),lr=LEARNING_RATE)
    feature_encoder_scheduler = StepLR(feature_encoder_optim,step_size=10000,gamma=0.5)

    relation_network_optim = torch.optim.Adam(relation_network.parameters(),lr=LEARNING_RATE)
    relation_network_scheduler = StepLR(relation_network_optim,step_size=10000,gamma=0.5)

    relation_coarse_network_optim = torch.optim.Adam(relation_coarse_network.parameters(),lr=LEARNING_RATE)
    relation_coarse_network_scheduler = StepLR(relation_coarse_network_optim,step_size=10000,gamma=0.5)

    Coare_encoder_optim = torch.optim.Adam(Coare_encoder.parameters(),lr=LEARNING_RATE)
    Coare_encoder_scheduler = StepLR(Coare_encoder_optim,step_size=10000,gamma=0.5)

    if os.path.exists(str("./models/HC_RN_Omniglot_5to5_s1b19_L2_feature_encoder_" + str(CLASS_NUM) +"way_" + str(SAMPLE_NUM_PER_CLASS) +"shot.pkl")):
        feature_encoder.load_state_dict(torch.load(str("./models/HC_RN_Omniglot_5to5_s1b19_L2_feature_encoder_" + str(CLASS_NUM) +"way_" + str(SAMPLE_NUM_PER_CLASS) +"shot.pkl")))
        print("load feature encoder success")
    if os.path.exists(str("./models/HC_RN_Omniglot_5to5_s1b19_L2_relation_network_"+ str(CLASS_NUM) +"way_" + str(SAMPLE_NUM_PER_CLASS) +"shot.pkl")):
        relation_network.load_state_dict(torch.load(str("./models/HC_RN_Omniglot_5to5_s1b19_L2_relation_network_"+ str(CLASS_NUM) +"way_" + str(SAMPLE_NUM_PER_CLASS) +"shot.pkl")))
        print("load relation network success")
    if os.path.exists(str("./models/HC_RN_Omniglot_5to5_s1b19_L2_relation_coarse_network_"+ str(CLASS_NUM) +"way_" + str(SAMPLE_NUM_PER_CLASS) +"shot.pkl")):
        relation_coarse_network.load_state_dict(torch.load(str("./models/HC_RN_Omniglot_5to5_s1b19_L2_relation_coarse_network_"+ str(CLASS_NUM) +"way_" + str(SAMPLE_NUM_PER_CLASS) +"shot.pkl")))
        print("load relation_coarse_network success")
    if os.path.exists(str("./models/HC_RN_Omniglot_5to5_s1b19_L2_Coare_encoder_"+ str(CLASS_NUM) +"way_" + str(SAMPLE_NUM_PER_CLASS) +"shot.pkl")):
        Coare_encoder.load_state_dict(torch.load(str("./models/HC_RN_Omniglot_5to5_s1b19_L2_Coare_encoder_"+ str(CLASS_NUM) +"way_" + str(SAMPLE_NUM_PER_CLASS) +"shot.pkl")))
        print("load Coare_encoder success")

    # Step 3: build graph
    print("Training...")
    last_accuracy = 0.0
    for episode in range(EPISODE):
        feature_encoder_scheduler.step(episode)
        relation_coarse_network_scheduler.step(episode)
        relation_network_scheduler.step(episode)
        Coare_encoder_scheduler.step(episode)
        task = tg.MiniImagenetTask(metatrain_folders,CLASS_NUM,SAMPLE_NUM_PER_CLASS,BATCH_NUM_PER_CLASS)#BATCH_NUM_PER_CLASS=10
        sample_dataloader = tg.get_mini_imagenet_data_loader(task,num_per_class=SAMPLE_NUM_PER_CLASS,split="train",shuffle=False)#25
        batch_dataloader = tg.get_mini_imagenet_data_loader(task,num_per_class=BATCH_NUM_PER_CLASS,split="test",shuffle=True)#50
        samples,sample_labels = sample_dataloader.__iter__().next() #25*3*84*84
        batches,batch_labels = batch_dataloader.__iter__().next()#every class(18) have 10 sample
        #coarse extract feature
        sample_coarse_features = Coare_encoder(Variable(samples).cuda(GPU)) # 25*64*19*19
        batch_coarse_features = Coare_encoder(Variable(batches).cuda(GPU))
        sample_coarse_features = sample_coarse_features.cpu().detach().numpy()
        cluster_num = 5
        coarse_groups = gen_superclass(sample_coarse_features, sample_labels, CLASS_NUM, cluster_num) #18
        coarse_data_samplearry = cal_clsvec_init(sample_coarse_features,sample_labels,CLASS_NUM)#18*64*19*19
        coarse_clustr_arry = cal_clsvec_init(coarse_data_samplearry,coarse_groups,cluster_num)#4*64*19*19
        coarse_data_samplearry = torch.Tensor(coarse_data_samplearry).type(torch.FloatTensor).cuda(GPU)
        coarse_clustr_arry = torch.Tensor(coarse_clustr_arry).type(torch.FloatTensor).cuda(GPU)
        batch_features_exts = batch_coarse_features.unsqueeze(0).repeat(coarse_clustr_arry.size(0),1,1,1,1)
        batch_features_exts = batch_features_exts.transpose(0,1)
        #coarse extract feature to torch.cat
        coarse_clustr_arry_exts = coarse_clustr_arry.unsqueeze(0).repeat(BATCH_NUM_PER_CLASS*CLASS_NUM,1,1,1,1)
        relation_pairs = torch.cat((coarse_clustr_arry_exts,batch_features_exts),2).view(-1,FEATURE_DIM*2,19,19)#180 128 19 19
        #relation_coarse_network
        relations = relation_coarse_network(relation_pairs).view(-1,cluster_num)# 180/CLASS_NUM  CLASS_NUM
        batch_coarse_labels = get_coare_real_labels(batch_labels,coarse_groups)
        coarse_batch_label = Variable(torch.zeros(BATCH_NUM_PER_CLASS*CLASS_NUM,cluster_num).scatter_(1, torch.Tensor(batch_coarse_labels).long().view(-1,1), 1).cuda(GPU))
        coarse_loss = mse(relations,coarse_batch_label)
        #predict labels
        _,predict_coarse_labels = torch.max(relations.data,1)#180  18
        rewards = [1 if predict_coarse_labels[j]==batch_coarse_labels[j] else 0 for j in range(BATCH_NUM_PER_CLASS*CLASS_NUM)]
        coarse_accuracy = np.sum(rewards)/1.0/BATCH_NUM_PER_CLASS/CLASS_NUM
        sample_fine_features = feature_encoder(Variable(samples).cuda(GPU)) # 25*64*19*19
        batch_fine_features = feature_encoder(Variable(batches).cuda(GPU))
        fine_data_samplearry = cal_clsvec_init(sample_fine_features.cpu().detach().numpy(),sample_labels,CLASS_NUM)#18*64*19*19
        fine_clustr_arry = cal_clsvec_init(fine_data_samplearry,coarse_groups,cluster_num)#4*64*19*19
        fine_data_samplearry = torch.Tensor(fine_data_samplearry).cuda(GPU)
        sample_parents = {}
        for i in range(fine_data_samplearry.size(0)):
            coarse_label= coarse_groups[i]
            if coarse_label not in sample_parents:
                sample_parents[coarse_label] = [i]
            else:
                sample_parents[coarse_label].append(i)
        predict_batch_fine_labels = []
        for i in range(batch_labels.size(0)):
            batch_label = batch_labels[i].view(-1,1)  
            batch_fine_feature = batch_fine_features[i,:] # 64*19*19
            predict_batch_coarse_label = predict_coarse_labels[i].item()
            bro_batch_labels = sample_parents[predict_batch_coarse_label]
            bro_num = len(bro_batch_labels)
            sample_fine_features_ext = fine_data_samplearry[bro_batch_labels,:]  # ?*64*19*19
            batch_fine_feature_ext = batch_fine_feature.unsqueeze(0).repeat(bro_num,1,1,1)# ?*64*19*19
            fine_pairs = torch.cat((sample_fine_features_ext,batch_fine_feature_ext),1).view(-1,FEATURE_DIM*2,19,19)
            fine_relations = relation_network(fine_pairs).view(-1,bro_num)
            new_predict_batch_fine_label = Variable(torch.Tensor(get_new_labels(batch_label,bro_batch_labels)).view(-1,bro_num)).cuda(GPU)
            if i == 0:
                fine_loss = mse(fine_relations,new_predict_batch_fine_label)
            else:
                fine_loss += mse(fine_relations,new_predict_batch_fine_label)
            #predict labels
            _,predict_fine_label = torch.max(fine_relations.data,1)#180  18
            predict_batch_fine_labels.append(bro_batch_labels[predict_fine_label[0]])
        rewards = [1 if predict_batch_fine_labels[j]==batch_labels[j] else 0 for j in range(batch_labels.size(0))]
        NMI_SCORE = NMI(batch_labels,predict_batch_fine_labels)
        ARI_SCORE = ARI(batch_labels,predict_batch_fine_labels)
        train_fine_accuracy = np.sum(rewards)/1.0/CLASS_NUM/BATCH_NUM_PER_CLASS
        fine_loss = fine_loss/BATCH_NUM_PER_CLASS/CLASS_NUM
        loss = my_lambda*coarse_loss+(1-my_lambda)*fine_loss
        if (episode+1)%100 ==0:
            print("coarse_loss: ",coarse_loss," fine_loss:",fine_loss," loss:",loss," train_fine_accuracy:",train_fine_accuracy,"  NMI_SCORE:",NMI_SCORE," ARI_SCORE:",ARI_SCORE)

        feature_encoder.zero_grad()
        relation_coarse_network.zero_grad()
        relation_network.zero_grad()
        Coare_encoder.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm(feature_encoder.parameters(),0.5)
        torch.nn.utils.clip_grad_norm(relation_coarse_network.parameters(),0.5)
        torch.nn.utils.clip_grad_norm(relation_network.parameters(),0.5)
        torch.nn.utils.clip_grad_norm(Coare_encoder.parameters(),0.5)
        feature_encoder_optim.step()
        relation_coarse_network_optim.step()
        relation_network_optim.step()
        Coare_encoder_optim.step()
        
        if (episode+1)%100 == 0:
            print("Testing...")
            for test_episode in range(TEST_EPISODE):
                task = tg.MiniImagenetTask(metatest_folders,CLASS_NUM,SAMPLE_NUM_PER_CLASS,BATCH_NUM_PER_CLASS)
                sample_dataloader = tg.get_mini_imagenet_data_loader(task,num_per_class=SAMPLE_NUM_PER_CLASS,split="train",shuffle=False)
                test_dataloader = tg.get_mini_imagenet_data_loader(task,num_per_class=BATCH_NUM_PER_CLASS,split="test",shuffle=True)
                sample_images,sample_labels = sample_dataloader.__iter__().next()
                test_images,test_labels = test_dataloader.__iter__().next()
                # #coarse extract to data
                sample_coarse_images_features = Coare_encoder(Variable(torch.tensor(sample_images)).cuda(GPU))
                test_coarse_images_features = Coare_encoder(Variable(torch.tensor(test_images)).cuda(GPU))
                sample_coarse_images_features = sample_coarse_images_features.cpu().detach().numpy()
                cluster_num = 5
                coarse_groups = gen_superclass(sample_coarse_images_features, sample_labels, CLASS_NUM, cluster_num) #18
                sample_coarse_data_samplearry = cal_clsvec_init(sample_coarse_images_features,sample_labels,CLASS_NUM)#18*64*19*19
                sample_coarse_clustr_arry = cal_clsvec_init(sample_coarse_data_samplearry,coarse_groups,cluster_num)#4*64*19*19
                sample_coarse_data_samplearry = torch.Tensor(sample_coarse_data_samplearry).type(torch.FloatTensor).cuda(GPU)
                sample_coarse_clustr_arry = torch.Tensor(sample_coarse_clustr_arry).type(torch.FloatTensor).cuda(GPU)
                test_features_exts = test_coarse_images_features.unsqueeze(0).repeat(sample_coarse_clustr_arry.size(0),1,1,1,1)
                test_features_exts = test_features_exts.transpose(0,1)
                #coarse extract feature to torch.cat
                sample_coarse_clustr_arry_exts = sample_coarse_clustr_arry.unsqueeze(0).repeat(BATCH_NUM_PER_CLASS*CLASS_NUM,1,1,1,1)
                relation_pairs = torch.cat((sample_coarse_clustr_arry_exts,test_features_exts),2).view(-1,FEATURE_DIM*2,19,19)#180 128 19 19
                #relation_coarse_network
                relations = relation_coarse_network(relation_pairs).view(-1,cluster_num)# 180/CLASS_NUM  CLASS_NUM
                test_coarse_labels = get_coare_real_labels(test_labels,coarse_groups)
                #predict labels
                _,predict_coarse_labels = torch.max(relations.data,1)#180  18
                rewards = [1 if predict_coarse_labels[j]==test_coarse_labels[j] else 0 for j in range(BATCH_NUM_PER_CLASS*CLASS_NUM)]
                coarse_accuracy = np.sum(rewards)/1.0/BATCH_NUM_PER_CLASS/CLASS_NUM
                sample_fine_images_features = feature_encoder(Variable(torch.tensor(sample_images)).cuda(GPU))# 25*64*19*19
                test_fine_images_features = feature_encoder(Variable(torch.tensor(test_images)).cuda(GPU))
                fine_data_samplearry = cal_clsvec_init(sample_fine_images_features.cpu().detach().numpy(),sample_labels,CLASS_NUM)#18*64*19*19
                fine_clustr_arry = cal_clsvec_init(fine_data_samplearry,coarse_groups,cluster_num)#4*64*19*19
                fine_data_samplearry = torch.Tensor(fine_data_samplearry).cuda(GPU)
                test_sample_parents = {}
                for i in range(fine_data_samplearry.size(0)):
                    coarse_label= coarse_groups[i]
                    if coarse_label not in test_sample_parents:
                        test_sample_parents[coarse_label] = [i]
                    else:
                        test_sample_parents[coarse_label].append(i)
                predict_test_fine_labels = []
                for i in range(test_labels.size(0)):
                    #test_fine_images_feature 
                    test_fine_images_feature = test_fine_images_features[i,:] # 64*19*19
                    predict_test_coarse_label = predict_coarse_labels[i].item()
                    bro_test_labels = test_sample_parents[predict_test_coarse_label]
                    bro_num = len(bro_test_labels)
                    sample_fine_features_ext = fine_data_samplearry[bro_test_labels,:]  # ?*64*19*19
                    test_fine_images_feature_ext = test_fine_images_feature.unsqueeze(0).repeat(bro_num,1,1,1)# ?*64*19*19
                    fine_pairs = torch.cat((sample_fine_features_ext,test_fine_images_feature_ext),1).view(-1,FEATURE_DIM*2,19,19)
                    fine_relations = relation_network(fine_pairs).view(-1,bro_num)
                    _,predict_fine_label = torch.max(fine_relations.data,1)#180  18
                    predict_test_fine_labels.append(bro_test_labels[predict_fine_label[0]])
                rewards = [1 if predict_test_fine_labels[j]==test_labels[j] else 0 for j in range(test_labels.size(0))]
                NMI_SCORE = NMI(test_labels,predict_test_fine_labels)
                ARI_SCORE = ARI(test_labels,predict_test_fine_labels)
                test_fine_accuracy = np.sum(rewards)/1.0/CLASS_NUM/BATCH_NUM_PER_CLASS
                print("accuracy", test_fine_accuracy, "NMI_SCORE", NMI_SCORE, "ARI_SCORE", ARI_SCORE)
                if test_fine_accuracy > last_accuracy:
                    # save networks
                    torch.save(feature_encoder.state_dict(),str("./models/HC_RN_Omniglot_5to5_s1b19_L2_feature_encoder_" + str(CLASS_NUM) +"way_" + str(SAMPLE_NUM_PER_CLASS) +"shot.pkl"))
                    torch.save(relation_network.state_dict(),str("./models/HC_RN_Omniglot_5to5_s1b19_L2_relation_network_"+ str(CLASS_NUM) +"way_" + str(SAMPLE_NUM_PER_CLASS) +"shot.pkl"))
                    torch.save(relation_coarse_network.state_dict(),str("./models/HC_RN_Omniglot_5to5_s1b19_L2_relation_coarse_network_"+ str(CLASS_NUM) +"way_" + str(SAMPLE_NUM_PER_CLASS) +"shot.pkl"))
                    torch.save(Coare_encoder.state_dict(),str("./models/HC_RN_Omniglot_5to5_s1b19_L2_Coare_encoder_"+ str(CLASS_NUM) +"way_" + str(SAMPLE_NUM_PER_CLASS) +"shot.pkl"))
                    last_accuracy = test_fine_accuracy

if __name__ == '__main__':
    main()

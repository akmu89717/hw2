import numpy as np
from PIL import Image,ImageFilter
from tqdm import tqdm
from cyvlfeat.sift.dsift import dsift
from cyvlfeat.kmeans import kmeans
from scipy.spatial.distance import cdist

CAT = ['Kitchen', 'Store', 'Bedroom', 'LivingRoom', 'Office',
       'Industrial', 'Suburb', 'InsideCity', 'TallBuilding', 'Street',
       'Highway', 'OpenCountry', 'Coast', 'Mountain', 'Forest']

CAT2ID = {v: k for k, v in enumerate(CAT)}

########################################
###### FEATURE UTILS              ######
###### use TINY_IMAGE as features ######
########################################

###### Step 1-a
def get_tiny_images(img_paths):
    
    

    tiny_img_feats = np.zeros((len(img_paths),256))
    for i in tqdm(range(len(img_paths))):
        image_input = Image.open(img_paths[i])
        Gauss_image = image_input.filter(ImageFilter.GaussianBlur(radius = 2))
        Resize_image=Gauss_image.resize((16, 16))
        mean_ima=np.mean(Resize_image)
        varian_ima = np.var(Resize_image)
        normalized_ima = (Resize_image-mean_ima)/varian_ima        
        normalized_ima=normalized_ima.reshape(1,256)        
        tiny_img_feats[i,:]=normalized_ima
 
    #################################################################
    #                        END OF YOUR CODE                       #
    #################################################################

    return tiny_img_feats

#########################################
###### FEATURE UTILS               ######
###### use BAG_OF_SIFT as features ######
#########################################

###### Step 1-b-1
def build_vocabulary(img_paths, vocab_size=400):
    image_feats = []
    count=0
    for path in tqdm(img_paths):
        if count%3==0:
            img =np.array(Image.open(path),dtype='float32')
            frames, descriptors = dsift(img, step=[3, 3], fast=True)       
            if descriptors is not None:
                for des in descriptors:
                    image_feats.append(des)
        count+=1            
    vocab = kmeans(np.array(image_feats).astype('float32'),vocab_size) 
    return vocab

###### Step 1-b-2
def get_bags_of_sifts(img_paths, vocab):
    img_feats = np.zeros([len(img_paths),len(vocab)])
    c=0
    for path in tqdm(img_paths):
        img =np.array(Image.open(path),dtype='float32')
        frames, descriptors = dsift(img, step=[3, 3], fast=True)    
        dist = cdist(vocab, descriptors)
        idx = np.argmin(dist, axis=0)
        hist, bin_edges = np.histogram(idx, bins=len(vocab))
        hist_norm = hist/sum(hist)
        img_feats[c,:]=hist_norm
        if c%100 == 0 :
            # print('hist_norm size',len(hist_norm))
            print(c)
        c+=1
    return img_feats

################################################
###### CLASSIFIER UTILS                   ######
###### use NEAREST_NEIGHBOR as classifier ######
################################################

###### Step 2
def nearest_neighbor_classify(train_img_feats, train_labels, test_img_feats):
    test_predicts = []
    KNN_num=4
    for i in tqdm(range(len(test_img_feats))):
        d = []
        KNN=[]
        for j in range(len(train_img_feats)):
            dist = cdist(train_img_feats[j].reshape(1,-1), test_img_feats[i].reshape(1,-1), metric='minkowski',p=0.3)
            d.append([dist, j])   
        d.sort()
        for K in range(KNN_num):
            KNN.append(train_labels[d[K][1]])
        maxlabel = max(KNN,key=KNN.count)
        test_predicts.append(maxlabel)
    return test_predicts

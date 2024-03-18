import sys

sys.path.append('/beegfs/desy/user/akorol/projects/point-cloud-diffusion')



import numpy as np

import matplotlib.pyplot as plt

import importlib

import pickle

from scipy.stats import wasserstein_distance



import utils.metrics as metrics

from sklearn.metrics import roc_auc_score



import torch

import torch.nn as nn

import random
# seed everything for reproducibility

def seed_everything(seed=42):

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed(seed)

    random.seed(seed)

    torch.backends.cudnn.deterministic = True

    torch.backends.cudnn.benchmark = False



seed_everything()
pickle_path = '/beegfs/desy/user/buhmae/6_PointCloudDiffusion/output/metrics/'



dict_real = pickle.load(open(pickle_path + 'merge_dict_10-90GeV_500000_g4.pickle', 'rb'))

dict_ddpm = pickle.load(open(pickle_path + 'merge_dict_10-90GeV_500000_ddpm.pickle', 'rb'))

dict_edm = pickle.load(open(pickle_path + 'merge_dict_10-90GeV_500000_edm.pickle', 'rb'))

dict_cm = pickle.load(open(pickle_path + 'merge_dict_10-90GeV_500000_cm.pickle', 'rb'))



print(dict_cm.keys())

# combine observables in a single array



obs_real = metrics.get_event_observables_from_dict(dict_real)

obs_ddpm = metrics.get_event_observables_from_dict(dict_ddpm)

obs_edm = metrics.get_event_observables_from_dict(dict_edm)

obs_cm = metrics.get_event_observables_from_dict(dict_cm)



# shuffle the data

obs_real = obs_real[np.random.permutation(len(obs_real))]

obs_ddpm = obs_ddpm[np.random.permutation(len(obs_ddpm))]

obs_edm = obs_edm[np.random.permutation(len(obs_edm))]

obs_cm = obs_cm[np.random.permutation(len(obs_cm))]



print(obs_real.shape)



mean_real, std_real = np.mean(obs_real, axis=0).reshape(1,-1), np.std(obs_real, axis=0).reshape(1,-1)



print(mean_real.shape)
# standardise the data

def standardize(ary, mean, std):

    return (ary - mean) / std



obs_std_real = standardize(obs_real, mean=mean_real, std=std_real)

obs_std_ddpm = standardize(obs_ddpm, mean=mean_real, std=std_real)

obs_std_edm = standardize(obs_edm, mean=mean_real, std=std_real)

obs_std_cm = standardize(obs_cm, mean=mean_real, std=std_real)
# array without hits

obs_std_real_woutHits = np.concatenate([obs_std_real[:,0:5], obs_std_real[:,6:]], axis=1)

obs_std_ddpm_woutHits = np.concatenate([obs_std_ddpm[:,0:5], obs_std_ddpm[:,6:]], axis=1)

obs_std_edm_woutHits = np.concatenate([obs_std_edm[:,0:5], obs_std_edm[:,6:]], axis=1)

obs_std_cm_woutHits = np.concatenate([obs_std_cm[:,0:5], obs_std_cm[:,6:]], axis=1)



print(obs_std_real_woutHits.shape)
obs_std_real_woutHits_split = np.array(np.array_split(obs_std_real_woutHits, 10))

obs_std_ddpm_woutHits_split = np.array(np.array_split(obs_std_ddpm_woutHits, 10))

obs_std_edm_woutHits_split = np.array(np.array_split(obs_std_edm_woutHits, 10))

obs_std_cm_woutHits_split = np.array(np.array_split(obs_std_cm_woutHits, 10))
# # Low lvl clasifier
device = torch.device('cuda')





class Discriminator(nn.Module):

    def __init__(self, num_features):

        super(Discriminator, self).__init__()



        

        self.Classifier = nn.Sequential(

            nn.Linear(num_features, 32),

            nn.LeakyReLU(0.2),

            

            nn.Linear(32, 16),

            nn.LeakyReLU(0.2),

            

            nn.Linear(16, 1)

        )



    def forward(self, x):

        x = self.Classifier(x)

        return x

def get_dataloader(features_real, features_fake, batch_size=256, shuffle=True, num_workers=4):

    

    labels_real = np.ones(len(features_real))

    labels_fake = np.zeros(len(features_fake))

    

    dataset = list(

        zip(

            np.concatenate((features_real, features_fake)), 

            np.concatenate((labels_real, labels_fake))

        )

    )

    

    dataloader = torch.utils.data.DataLoader(

        dataset,

        batch_size=batch_size, 

        shuffle=shuffle, 

        num_workers=num_workers

    )

    

    return dataloader
def train(mode):



    seeds = [1, 2, 3, 5, 8, 13, 21, 34, 55, 89]



    results = {'test_acccs': [], 'auc_score': [], 'best_loss': []}

    for i in range(len(obs_std_real_woutHits_split)):

        print('Fold: ', i)



        seed_everything(seeds[i])



        clasifier = Discriminator(num_features=25)

        clasifier.to(device)

        

        index_test = i

        index_validation = i+1 if i+1 < len(obs_std_real_woutHits_split) else 0

        index_train = [j for j in range(len(obs_std_real_woutHits_split)) if j not in [index_test, index_validation]]



        dataset_real_train = np.concatenate(obs_std_real_woutHits_split[index_train])

        dataset_real_test = obs_std_real_woutHits_split[index_test]

        dataset_real_validation = obs_std_real_woutHits_split[index_validation]



        if mode == 'real_VS_ddpm':

            dataset_fake_train = np.concatenate(obs_std_ddpm_woutHits_split[index_train])

            dataset_fake_test = obs_std_ddpm_woutHits_split[index_test]

            dataset_fake_validation = obs_std_ddpm_woutHits_split[index_validation]



        elif mode == 'real_VS_edm':

            dataset_fake_train = np.concatenate(obs_std_edm_woutHits_split[index_train])

            dataset_fake_test = obs_std_edm_woutHits_split[index_test]

            dataset_fake_validation = obs_std_edm_woutHits_split[index_validation]



        elif mode == 'real_VS_cm':

            dataset_fake_train = np.concatenate(obs_std_cm_woutHits_split[index_train])

            dataset_fake_test = obs_std_cm_woutHits_split[index_test]

            dataset_fake_validation = obs_std_cm_woutHits_split[index_validation]



        train_dataloader = get_dataloader(

            dataset_real_train,

            dataset_fake_train

        )

        test_dataloader = get_dataloader(

            dataset_real_test,

            dataset_fake_test

        )

        validation_dataloader = get_dataloader(

            dataset_real_validation,

            dataset_fake_validation

        )



        criterion = nn.BCEWithLogitsLoss()

        optimizer = torch.optim.Adam(clasifier.parameters(), lr=0.001)



        num_epochs = 10



        test_acccs = []

        best_loss = 1e10

        for epoch in range(num_epochs):

            clasifier.train()

            for i, (features, labels) in enumerate(train_dataloader):

                features = features.to(device).float()

                labels = labels.to(device).float()

                

                outputs = clasifier(features).view(-1)

                loss = criterion(outputs, labels)

                

                optimizer.zero_grad()

                loss.backward()

                optimizer.step()

                    

                if (i+1) % 1000 == 0:

                    print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(

                        epoch+1, num_epochs, i+1, len(train_dataloader), loss.item()

                    ))

                    clasifier.eval()

                    with torch.no_grad():

                        correct = 0

                        total = 0

                        val_loss = 0

                        for features, labels in validation_dataloader:

                            features = features.to(device).float()

                            labels = labels.to(device).float()

                            

                            outputs = clasifier(features).view(-1)



                            val_loss += criterion(outputs, labels).item()



                            predicted = torch.round(torch.sigmoid(outputs))

                            

                            total += labels.size(0)

                            correct += (predicted == labels).sum().item()



                        val_loss /= len(validation_dataloader)



                        if val_loss < best_loss:

                            best_loss = val_loss

                            torch.save(clasifier.state_dict(), f'clasifier_best_val_loss_{mode}.pth')



                        print('Test Accuracy of the model on the {} test samples: {} %'.format(total, 100 * correct / total))

                        test_acccs.append(100 * correct / total)

                    clasifier.train()

        print('Finished Training')

        print('Best val loss: {}'.format(best_loss))



        print('predicting...')

        clasifier.load_state_dict(torch.load(f'clasifier_best_val_loss_{mode}.pth'))



        # get AUROC

        clasifier.eval()

        with torch.no_grad():

            predistions = []

            labels = []

            for features, label in test_dataloader:

                features = features.to(device).float()



                outputs = clasifier(features).view(-1)

                predicted = torch.sigmoid(outputs)



                predistions.append(predicted.cpu().numpy())

                labels.append(label.numpy())



        predistions = np.concatenate(predistions)

        labels = np.concatenate(labels)



        auc_score = roc_auc_score(labels, predistions)



        results['test_acccs'].append(test_acccs)

        results['auc_score'].append(auc_score)

        results['best_loss'].append(best_loss)



    return results


results_10_folds = []

for mode in ['real_VS_ddpm', 'real_VS_edm', 'real_VS_cm']:

    results = train(mode)

    results_10_folds.append(results)
for i, mode in enumerate(['real_VS_ddpm', 'real_VS_edm', 'real_VS_cm']):

    auc_scres = results_10_folds[i]['auc_score']

    print(f"{mode} test: auc={np.mean(auc_scres)}, std: {np.std(auc_scres)}")









import json
import os
import numpy as np
import pickle
from absl import flags
from absl import app
from sklearn.metrics._classification import accuracy_score
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import torchvision
from torchvision import transforms


FLAGS = flags.FLAGS

def unpickle(file):
    with open(file, 'rb') as fo:
        myDict = pickle.load(fo, encoding='latin1')
    return myDict

def load_fashion_targets(root='./data'):
    """
    Load fashion labels directly using torchvision datasets.

    Parameters:
        root (str): The root directory where the dataset is or will be downloaded.

    Returns:
        tuple: (train_labels, test_labels) as lists.
    """

    # Minimal transform required to load dataset
    transform = transforms.Compose([transforms.ToTensor()])

    # Load training dataset and extract labels
    train_set = torchvision.datasets.FashionMNIST(root='./data', train=True,
                                download=True)

    train_labels = [label for _, label in train_set]

    # Load test dataset and extract labels
    test_set = torchvision.datasets.FashionMNIST(root='./data', train=False,download=True)
    test_labels = [label for _, label in test_set]

    return train_labels, test_labels

def get_nonbin_target(bin, y, num_classes):
    np.random.seed(0)
    nonbin = np.zeros(len(y), dtype=int)    
    correct = 0
    orc = 1
    for i in range(len(y)):

        # If the binary target == 1, keep the ground-truth label
        if bin[i] == 1:
            nonbin[i] = y[i]
        # Otherwise randomly pick a label in [0, num_classes-1]
        else:
            nonbin[i] = int(np.random.uniform(0, num_classes))
    return nonbin.tolist()

def noise_transistion_matrix(y_pred, y_true, num_classes,file_location):
    """
    Calculate the noise transition matrix for a given set of predictions and true labels.
    The transition matrix T is defined such that T[i][j] = P(y_pred = j | y_true = i).
    """
    # Initialize the transition matrix
    T = np.zeros((num_classes, num_classes))

    # Count occurrences of each pair (y_true, y_pred)
    for i in range(len(y_true)):
        T[y_true[i]][y_pred[i]] += 1

    # Normalize the rows to get probabilities
    for i in range(num_classes):
        if np.sum(T[i]) > 0:
            T[i] /= np.sum(T[i])

    #save transisiton matrix figure 
    class_names = [str(i) for i in range(num_classes)]

    plt.figure(figsize=(30, 20))
    sns.heatmap(
        T,
        annot=True,
        fmt='.3f',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Noise Transition Matrix')
    plt.tight_layout()
    plt.savefig(file_location)
    plt.close()

    return 
     




def main(argv):    
    train_seeds = [x + 1 for x in [0,1,2,3,4,5,6,7,8,9]] 
    test_seeds = [x + 1 for x in [0,1,2,3,4,10,11,12,13,14]]


    NUM_CLASSES = 10
    labels = [20,40,60,100,200,500,2500]  # Example array, as in your original code
    EX_STRENGTHS = [8]
    seeds = [i+1 for i in range(15)]
    # seeds.append(69)  # optional
    
    for l in labels:
        for strength in EX_STRENGTHS:
            train_array = []
            test_array = []
            for s in seeds:
                try:
                    in_file = f'expert_{s}_fashion_expert{strength}.{0}@{l}_predictions'
                    print(f'Preprocess artificial_expert_labels from file {in_file}')
                    p_strength = strength
                    file_location = f'fashion/L_{l}_p{p_strength}/'+in_file+'.json'
                    with open(file_location, 'r') as f:
                        bmt_pred = json.load(f)
                except FileNotFoundError:
                    print(f'File {in_file} not found')
                    continue    
                predictions = bmt_pred
                print('Transform to multiclass')
                
                # Load CIFAR-10 ground truth (train and test) from your local dir
                train_gt_targets, test_gt_targets = load_fashion_targets(os.getcwd())
                # predictions['train'] = get_nonbin_target(predictions['train'], train_gt_targets, NUM_CLASSES)
                # predictions['test']  = get_nonbin_target(predictions['test'], test_gt_targets, NUM_CLASSES)
                
                file_location = f'fashion/L_{l}_p{p_strength}/transistion_matrix_{s}_expert{strength}.{s}@{l}_train.png'
                noise_transistion_matrix(predictions['train'], train_gt_targets, NUM_CLASSES,file_location)

                #get transition matrix 
                file_location = f'fashion/L_{l}_p{p_strength}/transistion_matrix_{s}_expert{strength}.{s}@{l}_test.png'
                noise_transistion_matrix(predictions['test'], test_gt_targets, NUM_CLASSES,file_location)

                # Check how well they match the CIFAR-10 ground truth
                print('Check train:', accuracy_score(train_gt_targets, predictions['train']))
                print('Check test:', accuracy_score(test_gt_targets, predictions['test']))
                
                out_file = f'expert_{s}_fashion_expert{strength}.{s}@{l}_true_predictions'
                file_location = f'fashion/L_{l}_p{p_strength}/'+out_file+'.json'
                
                with open(file_location, 'w') as f:
                    json.dump(predictions, f)                        
                print(f'Saved to {out_file}')
            
            for train_experts in train_seeds:
                file_name = f'fashion/L_{l}_p{p_strength}/expert_{train_experts}_fashion_expert{strength}.{0}@{l}_predictions.json'
                with open(file_name, 'r') as f:
                    data = json.load(f)
                    train_array.append(data['train'])

            for test_experts in test_seeds:
                file_name = f'fashion/L_{l}_p{p_strength}/expert_{test_experts}_fashion_expert{strength}.{0}@{l}_predictions.json'
                with open(file_name, 'r') as f:
                    data = json.load(f)
                    test_array.append(data['test'])
            train_array = np.array(train_array)
            test_array = np.array(test_array)
            
            np.save(f'fashion/L_{l}_p{p_strength}/train_array.npy', train_array)
            np.save(f'fashion/L_{l}_p{p_strength}/test_array.npy', test_array)

    # for s in seeds:
    #     for l in labels:
    #         try:
    #             in_file = f'expert_{s}_cifar10_expert{EX_STRENGTH}.{s}@{l}_predictions'
    #             print(f'Preprocess artificial_expert_labels from file {in_file}')
    #             with open('artificial_expert_labels/'+in_file+'.json', 'r') as f:
    #                 bmt_pred = json.load(f)
    #         except FileNotFoundError:
    #             print(f'File {in_file} not found')
    #             continue

    #         predictions = bmt_pred
    #         print('Transform to multiclass')

    #         # Load CIFAR-10 ground truth (train and test) from your local dir
    #         train_gt_targets, test_gt_targets = load_cifar10_targets(os.getcwd())

    #         # Convert the binary predictions to multiclass
    #         predictions['train'] = get_nonbin_target(predictions['train'], train_gt_targets, NUM_CLASSES)
    #         predictions['test']  = get_nonbin_target(predictions['test'], test_gt_targets, NUM_CLASSES)

    #         # Check how well they match the CIFAR-10 ground truth
    #         print('Check train:', accuracy_score(train_gt_targets, predictions['train']))
    #         print('Check test:', accuracy_score(test_gt_targets, predictions['test']))

    #         out_file = f'expert_{s}_cifar10_expert{EX_STRENGTH}.{s}@{l}_true_predictions'
    #         with open('artificial_expert_labels/'+out_file+'.json', 'w') as f:
    #             json.dump(predictions, f)
    #         print(f'Saved to {out_file}')
    
#Make npy when done 

if __name__ == '__main__':
    flags.DEFINE_string('approach', 'EmbeddingCM_bin', 'Approach for predicting the expert labels')
    flags.DEFINE_integer('ex_strength', 40, 'Expert Strength')
    app.run(main)

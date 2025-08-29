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


FLAGS = flags.FLAGS

def unpickle(file):
    with open(file, 'rb') as fo:
        myDict = pickle.load(fo, encoding='latin1')
    return myDict

def load_cifar10_targets(wkdir):
    """
    Load CIFAR-10 labels from all 5 training batches and 1 test batch.
    Returns (train_labels, test_labels) as lists.
    """
    train_labels = []

    # CIFAR-10 has data_batch_1..5 for training
    for i in range(1, 6):
        batch_dict = unpickle(os.path.join(wkdir, 'data', 'cifar-10-batches-py', f'data_batch_{i}'))
        # "labels" key contains the class labels for CIFAR-10
        train_labels.extend(batch_dict['labels'])

    # test_batch
    test_dict = unpickle(os.path.join(wkdir, 'data', 'cifar-10-batches-py', 'test_batch'))
    test_labels = test_dict['labels']
    return train_labels, test_labels

# def get_nonbin_target(bin, y, num_classes):
#     np.random.seed(0)
#     nonbin = np.zeros(len(y), dtype=int)    
#     correct = 0
#     orc = 1
#     for i in range(len(y)):

#         correct = correct + 1 if (y[i]==orc and bin[i]==1) or (y[i] != orc and bin[i] ==0) else correct
#         # If the binary target == 1, keep the ground-truth label
#         if bin[i] == 1:
#             nonbin[i] = y[i]
#         # Otherwise randomly pick a label in [0, num_classes-1]
#         else:
#             nonbin[i] = int(np.random.uniform(0, num_classes))
#     print(f'{correct} / {str(len(y))} = {correct/len(y)}')
#     return nonbin.tolist()

def noise_transistion_matrix(y_pred, y_true, num_classes,file_location):
    """
    Calculate the noise transition matrix for a given set of predictions and true labels.
    The transition matrix T is defined such that T[i][j] = P(y_pred = j | y_true = i).
    """
    # Initialize the transition matrix

    T = np.zeros((num_classes, num_classes))
    # print("y pred", y_pred) 



    # Count occurrences of each pair (y_true, y_pred)
    for i in range(len(y_true)):
        T[y_true[i]][y_pred[i]] += 1

    # # Normalize the rows to get probabilities
    for i in range(num_classes):
        if np.sum(T[i]) > 0:
            T[i] /= np.sum(T[i])

    #save transisiton matrix figure 
    class_names = [str(i) for i in range(num_classes)]

    plt.figure(figsize=(10, 8))
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
   


    # Since we are now on CIFAR-10, set this to 10
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
                    in_file = f'expert_{s}_cifar10_expert{strength}.{0}@{l}_predictions'
                    print(f'Preprocess artificial_expert_labels from file {in_file}')
                    p_strength = strength
                    file_location = f'cifar10/L_{l}_p{p_strength}/'+in_file+'.json'
                    with open(file_location, 'r') as f:
                        bmt_pred = json.load(f)
                        print(f'Loaded {in_file}')
                except FileNotFoundError:
                    print(f'File {file_location} not found')
                    continue    
                predictions = bmt_pred
                print('Transform to multiclass')
                
                # Load CIFAR-10 ground truth (train and test) from your local dir
                train_gt_targets, test_gt_targets = load_cifar10_targets(os.getcwd())
                # predictions['train'] = get_nonbin_target(predictions['train'], train_gt_targets, NUM_CLASSES)
                # predictions['test']  = get_nonbin_target(predictions['test'], test_gt_targets, NUM_CLASSES)

                #get transition matrix 
                file_location = f'cifar10/L_{l}_p{p_strength}/transistion_matrix_{s}_expert{strength}.{s}@{l}_train.png'
                noise_transistion_matrix(predictions['train'], train_gt_targets, NUM_CLASSES,file_location)
                file_location = f'cifar10/L_{l}_p{p_strength}/transistion_matrix_{s}_expert{strength}.{s}@{l}_test.png'
                noise_transistion_matrix(predictions['test'], test_gt_targets, NUM_CLASSES,file_location)
            

                # Check how well they match the CIFAR-10 ground truth
                print('Check train:', accuracy_score(train_gt_targets, predictions['train']))
                print('Check test:', accuracy_score(test_gt_targets, predictions['test']))
                
                out_file = f'expert_{s}_cifar10_expert{strength}.{s}@{l}_true_predictions'
                file_location = f'cifar10/L_{l}_p{p_strength}/'+out_file+'.json'
                
                with open(file_location, 'w') as f:
                    json.dump(predictions, f)                        
                print(f'Saved to {out_file}')
            
            for train_experts in train_seeds:
                file_name = f'cifar10/L_{l}_p{p_strength}/expert_{train_experts}_cifar10_expert{strength}.{0}@{l}_predictions.json'
                with open(file_name, 'r') as f:
                    data = json.load(f)
                    train_array.append(data['train'])

            for test_experts in test_seeds:
                file_name = f'cifar10/L_{l}_p{p_strength}/expert_{test_experts}_cifar10_expert{strength}.{0}@{l}_predictions.json'
                with open(file_name, 'r') as f:
                    data = json.load(f)
                    test_array.append(data['test'])
            train_array = np.array(train_array)
            test_array = np.array(test_array)
            
            np.save(f'cifar10/L_{l}_p{p_strength}/train_array.npy', train_array)
            np.save(f'cifar10/L_{l}_p{p_strength}/test_array.npy', test_array)

    
#Make npy when done 

if __name__ == '__main__':
    flags.DEFINE_string('approach', 'EmbeddingCM_bin', 'Approach for predicting the expert labels')
    flags.DEFINE_integer('ex_strength', 40, 'Expert Strength')
    app.run(main)

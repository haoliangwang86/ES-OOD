import os
import csv
import sys
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import norm
from itertools import product
from torch.multiprocessing import Pool, cpu_count
from sklearn.preprocessing import StandardScaler
import backbone_models.densenet as densenet
import backbone_models.resnet as resnet
import backbone_models.vgg as vgg
from global_settings import *
from utility import get_inter_outputs, get_statistics, load_ood_detector, load_standard_scaler, load_standard_scaler_score, sort_csv_results, calc_cumulative_params


parser = argparse.ArgumentParser()
parser.add_argument('--model', default='vgg16', type=str,
                    help='model name')
parser.add_argument('--ind', default='cifar10', type=str,
                    help='InD dataset name')
parser.add_argument('--thr', default=0.99, type=float,
                    help='Early stopping threshold')
parser.add_argument('--k', default=2, type=int,
                    help='Early stopping k value')
parser.add_argument('--folder', default='results', type=str,
                    help='Results save folder')


def detect_oods_for_all_ood_datasets():
    args = parser.parse_args()

    results = []
    for ood_name in OOD_LIST:
        result = detect_oods(args.model, args.ind, ood_name, args.thr, args.k, args.folder)
        results.append(result)

    with open(f'{args.folder}/{args.model}-{args.ind}-summary.txt', 'w') as f:
        # Redirect standard output to the file
        sys.stdout = f

        print(f"EF-OOD ({args.thr}) results for {args.model} backbone, {args.ind} InD:")
        print(f"{'OOD': <15}{'FPR-at-95%-TPR' : ^15}{'AUROC' : ^15}{'AUPR-out' : ^15}{'AUPR-in' : >15}{'Efficiency Impr' : >15}")
        for i in range(len(OOD_LIST)):
            print(f"{OOD_LIST[i]: <15}{100*results[i][0]:^15.2f}{100*results[i][1]:^15.2f}"
                  f"{100*results[i][2]:^15.2f}{100*results[i][3]:>15.2f}{100*results[i][4]:>15.2f}")

        # Reset stdout to its default value to enable printing to the console again
        sys.stdout = sys.__stdout__


def detect_oods(model_name, ind_name, ood_name, thr, k, folder_name):
    print(f"Detecting OODs for {model_name}, {ind_name} vs. {ood_name}:")
    print(f"Layer idx\tAUROC")

    if not os.path.exists(os.path.join(f"{folder_name}", f"{model_name}")):
        os.makedirs(os.path.join(f"{folder_name}", f"{model_name}"))
    result_filename = f"{folder_name}/{model_name}/{model_name}-{ind_name}-vs-{ood_name}.csv"
    fields = ['Layer', 'FPR-at-95%-TPR', 'AUROC', 'AUPR-out', 'AUPR-in',
              'InD-correct', 'OOD-correct', 'Stop-Num', 'Pos', 'TP', 'Neg', 'TN', 'SavedOPs', 'EF-Impr']

    with open(result_filename, 'w') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(fields)

    scores, ind_length, ood_length, final_efficiency_improvement = compute_ood_scores_of_all_layers(model_name, ind_name, ood_name, folder_name, thr, k)

    y_test = [-1] * ind_length + [1] * ood_length  # OOD as positive, ID as negative

    fpr_at_95_tpr, auroc, aupr_out, aupr_in = get_statistics(y_test, scores)

    headers, results = sort_csv_results(result_filename)

    la_ood = ["EF-OOD", f'{100*fpr_at_95_tpr:.2f}', f'{100*auroc:.2f}', f'{100*aupr_out:.2f}', f'{100*aupr_in:.2f}']
    ef_imp = ["Final Efficiency Improvement", f'{100*final_efficiency_improvement:.2f}']
    with open(result_filename, 'w') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(headers)
        csvwriter.writerows(results)
        csvwriter.writerow(la_ood)
        csvwriter.writerow(ef_imp)

    print(f"EF-OOD:\t\t{100*auroc:.2f}")
    print(f"Final Efficiency Improvement:\t\t{100*final_efficiency_improvement:.2f}")
    print()
    return fpr_at_95_tpr, auroc, aupr_out, aupr_in, final_efficiency_improvement


def compute_ood_scores_of_all_layers(model_name, ind_name, ood_name, folder_name, thr, k):
    layers = 0
    if model_name == 'vgg16':
        layers = VGG16_LAYERS
        selected_layers = vgg_selected_layers
    elif model_name == 'resnet34':
        layers = RESNET34_LAYERS
        selected_layers = resnet_selected_layers
    elif model_name == 'densenet100':
        layers = DENSENET100_LAYERS
        selected_layers = densenetnet_selected_layers

    if model_name == "vgg16":
        if ind_name == "cifar100":
            model = vgg.vgg16_cifar100().cuda()
        elif ind_name == "cifar10":
            model = vgg.vgg16().cuda()
    elif model_name == "resnet34":
        if ind_name == "cifar100":
            model = resnet.ResNet34_cifar100().cuda()
        else:
            model = resnet.ResNet34().cuda()
    elif model_name == "densenet100":
        if ind_name == "cifar100":
            model = densenet.DenseNet100_cifar100().cuda()
        elif ind_name == "cifar10":
            model = densenet.DenseNet100().cuda()

    n_params = calc_cumulative_params(model)
    total_params = n_params[-1]
    n_params = np.array(n_params)[selected_layers]

    final_scores = None
    previous_stop_count = None
    total_save_ops = 0
    X = [l+1 for l in layers]
    Y1 = []
    Y2 = []
    all_scores = []
    for i, layer_idx in enumerate(layers):
        _, updated_stop_count, ind_length, ood_length, saved_ops, greater_than_thr_ood_count, newly_stop_ood_count, scores = compute_ood_scores(model_name, ind_name, ood_name, layer_idx, folder_name, thr, previous_stop_count, k, total_params, n_params)
        Y1.append(greater_than_thr_ood_count)
        Y2.append(newly_stop_ood_count)
        all_scores.append(scores)
        if layer_idx == 0:
            final_scores = scores
            previous_stop_count = updated_stop_count
        elif i == (len(layers) - 1):  # last layer
            stopped = previous_stop_count >= k
            all_scores = np.array(all_scores)
            max_scores = all_scores.max(axis=0)
            final_scores[~stopped] = max_scores[~stopped]  # update OOD scores
        else:
            stopped = previous_stop_count >= k
            final_stop_mask = updated_stop_count >= k
            newly_stop_mask = ~stopped & final_stop_mask
            # final_scores[newly_stop_mask] = scores[newly_stop_mask]

            final_scores[newly_stop_mask] = np.full(len(scores), 1000)[newly_stop_mask]

            previous_stop_count = updated_stop_count
        total_save_ops += saved_ops

    total_ops = total_params * ood_length
    final_efficiency_improvement = total_save_ops / total_ops

    # Set the width of the bars
    width = 0.6

    # Create the bar plot
    fig, ax = plt.subplots(figsize=(5, 3))

    # Plot 'Suspected OOD' bars
    ax.bar(X, Y1, width, label='Suspected OOD', color='skyblue')

    # Plot 'Stopped OOD' bars on top of 'Suspected OOD' bars
    ax.bar(X, Y2, width, label='Stopped OOD', color='darkblue')

    # Set the x-axis tick labels
    ax.set_xticks(X)
    ax.set_xticklabels(X)

    # Set the labels and title
    ax.set_xlabel('Layers')
    ax.set_ylabel('Number of Samples')
    ax.set_title(f'OOD: {ood_name}, k = {k}')

    # Add a legend
    ax.legend()

    # Display the plot
    plt.tight_layout()
    os.makedirs(f'{folder_name}/count_plots', exist_ok=True)
    plt.savefig(f'{folder_name}/count_plots/{model_name}-{ind_name}-vs-{ood_name}-ood-count.pdf')
    plt.close()

    return final_scores, ind_length, ood_length, final_efficiency_improvement


def compute_ood_scores(model_name, ind_name, ood_name, layer_idx, folder_name, thr, stop_count, k, total_params, n_params):
    ood_detector = load_ood_detector(model_name, ind_name, layer_idx)
    ss = load_standard_scaler(model_name, ind_name, layer_idx)
    ss_score = load_standard_scaler_score(model_name, ind_name, layer_idx)

    ind_testing_features = get_inter_outputs(model_name, ind_name, ind_name, layer_idx)
    ind_length = len(ind_testing_features)

    if ood_name == "combined":
        ood_features = None
        for i in range(len(OOD_LIST)):
            if i == 0:
                ood_features = get_inter_outputs(model_name, ind_name, OOD_LIST[i], layer_idx)
            else:
                ood_features = \
                    np.concatenate((ood_features, get_inter_outputs(model_name, ind_name, OOD_LIST[i], layer_idx)))
        ood_length = len(ood_features)
    else:
        ood_features = get_inter_outputs(model_name, ind_name, ood_name, layer_idx)
        ood_length = len(ood_features)
        ind_length = ood_length = min(ind_length, ood_length)   # balanced testing set

    ind_testing_features = ind_testing_features[:ind_length]
    ood_features = ood_features[:ood_length]

    ind_testing_features = ss.transform(ind_testing_features)
    ood_features = ss.transform(ood_features)

    data = np.vstack((ind_testing_features, ood_features))

    preds = ood_detector.predict(data)

    id_correct = np.count_nonzero(preds[:ind_length] == 1)  # ID is positive for OCSVM
    ood_correct = np.count_nonzero(preds[ind_length:] == -1)    # OOD is negative for OCSVM

    # calculate the average and std of OOD samples' scores
    scores = -ood_detector.decision_function(data)  # negated scores, positive for OOD, negative for InD

    # fit standard Gaussian and calculate OOD prob.
    scores = scores.reshape(-1, 1)
    standardized_scores = ss_score.transform(scores)
    gaussian_dist = norm(loc=0, scale=1)
    ood_probabilities = gaussian_dist.cdf(standardized_scores)
    ood_probabilities = ood_probabilities.flatten()
    early_stop_mask = ood_probabilities > thr

    # mark already stopped samples
    if stop_count is not None:
        stopped = stop_count >= k
    else:
        stop_count = np.zeros(len(ood_probabilities))
        stopped = None

    # update stop count
    updated_stop_count = stop_count.copy()
    updated_stop_count[early_stop_mask] += 1

    # find newly stopped samples at the current layer
    if stopped is not None:
        final_stop_mask = updated_stop_count >= k
        newly_stop_mask = ~stopped & final_stop_mask

        greater_than_thr_mask = ~stopped & early_stop_mask
    elif k == 1:
        newly_stop_mask = early_stop_mask

        greater_than_thr_mask = early_stop_mask
    else:  # first layer and k greater than 1
        newly_stop_mask = None

        greater_than_thr_mask = early_stop_mask

    # calculate statistics for current layer
    y_test = [-1] * ind_length + [1] * ood_length
    y_test = np.array(y_test)

    if newly_stop_mask is not None:
        stop_num = np.count_nonzero(newly_stop_mask)
        pos = np.count_nonzero(newly_stop_mask)
        neg = len(ood_probabilities) - pos
        tp = np.count_nonzero((newly_stop_mask == 1) & (y_test == 1))
        tn = np.count_nonzero((newly_stop_mask == 0) & (y_test == -1))
    else:
        stop_num = 0
        pos = 0
        neg = ind_length + ood_length
        tp = 0
        tn = ind_length

    # calculate efficiency improvement
    if layer_idx == (len(n_params) - 1):  # last layer
        saved_ops = 0
        efficiency_improvement = 0
    else:
        saved_ops = (total_params - n_params[layer_idx]) * tp
        total_ops = total_params * ood_length
        efficiency_improvement = saved_ops/total_ops

    fpr_at_95_tpr, auroc, aupr_out, aupr_in = get_statistics(y_test, ood_probabilities)
    print(f'layer {layer_idx}:\t{100*auroc:.2f}')

    greater_than_thr_ood_count = np.count_nonzero((greater_than_thr_mask == 1) & (y_test == 1))
    newly_stop_ood_count = tp

    # Save results to file
    row = [layer_idx + 1, f'{100*fpr_at_95_tpr:.2f}', f'{100*auroc:.2f}', f'{100*aupr_out:.2f}', f'{100*aupr_in:.2f}',
           id_correct, ood_correct, stop_num, pos, tp, neg, tn, saved_ops, efficiency_improvement]

    result_filename = f"{folder_name}/{model_name}/{model_name}-{ind_name}-vs-{ood_name}.csv"
    with open(result_filename, 'a') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(row)



    # Plot the InD and OOD scores in subfigure 1
    scores = scores.flatten()
    standardized_scores = standardized_scores.flatten()
    ood_probabilities = ood_probabilities.flatten()

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    ax1.hist(scores[:ind_length], bins=100, alpha=0.5, label='InD')
    ax1.hist(scores[ind_length:], bins=100, alpha=0.5, label='OOD')
    ax1.set_xlabel('Raw Scores')
    ax1.set_ylabel('Frequency')
    ax1.set_title('InD and OOD Scores')
    ax1.legend()

    # Plot the standardized InD and OOD scores in subfigure 2
    ax2.hist(standardized_scores[:ind_length], bins=100, alpha=0.5, label='InD')
    ax2.hist(standardized_scores[ind_length:], bins=100, alpha=0.5, label='OOD')
    ax2.set_xlabel('Standardized Scores')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Standardized InD and OOD Scores')
    ax2.legend()

    # Plot the InD and OOD "ood_probabilities" in subfigure 3
    ax3.hist(ood_probabilities[:ind_length], bins=100, alpha=0.5, label='InD')
    ax3.hist(ood_probabilities[ind_length:], bins=100, alpha=0.5, label='OOD')
    ax3.set_xlabel('OOD Probabilities')
    ax3.set_ylabel('Frequency')
    ax3.set_title('InD and OOD OOD Probabilities')
    ax3.legend()

    # Adjust the spacing between subplots
    plt.tight_layout()

    # Save the image
    os.makedirs(f'{folder_name}/plots/{model_name}-{ind_name}-vs-{ood_name}-score-dist', exist_ok=True)
    plt.savefig(f'{folder_name}/plots/{model_name}-{ind_name}-vs-{ood_name}-score-dist/layer_{layer_idx}_ind_ood_scores.pdf')
    plt.close()

    return ood_probabilities, updated_stop_count, ind_length, ood_length, saved_ops, greater_than_thr_ood_count, newly_stop_ood_count, scores


if __name__ == '__main__':
    detect_oods_for_all_ood_datasets()

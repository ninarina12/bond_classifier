import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import balanced_accuracy_score, confusion_matrix

from joblib import dump, load
from tqdm import tqdm
from utils.data import bonds, bond_to_float
from utils.plot import palette, fontsize, textsize, cmap_mono

bar_format = '{l_bar}{bar:10}{r_bar}{bar:-10b}'
tqdm.pandas(bar_format=bar_format)


def prepare_data(data, n_components, test_size, seed=12, pca=None, scaler=None, columns=[]):
    # parse data and labels
    X = np.stack(data['elf'].sum())
    
    # additional columns
    for col in columns:
        X = np.hstack([X, np.expand_dims(data[col].sum(), axis=1)])
    
    # pca transform
    if pca:
        X = pca.transform(X)
    else:
        pca = PCA(n_components=n_components, svd_solver='full', random_state=seed)
        X = pca.fit_transform(X)
    
    # standardize data
    if scaler:
        X = scaler.transform(X)
    else:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

    # train/test split
    if test_size:
        y = data[['bond_length', 'label']].explode('bond_length')['label'].apply(bond_to_float).values
        idx_train, idx_test = train_test_split(np.arange(X.shape[0]), test_size=test_size, random_state=seed, stratify=y)
        X_train, y_train = X[idx_train], y[idx_train]
        X_test, y_test = X[idx_test], y[idx_test]
        X_data = (X_train, X_test)
        y_data = (y_train, y_test)
    else:
        X_data = X
        y_data = None
    
    return X_data, y_data, pca, scaler


def train_models(X_data, y_data, pca, scaler, n_models=100, seed=12, save_path=None):
    # compute sample weights
    X_train, X_test = X_data
    y_train, y_test = y_data
    w_train = compute_sample_weight('balanced', y_train)
    w_test = compute_sample_weight('balanced', y_test)
    dev_size = len(y_test)/(len(y_train) + len(y_test))
    
    # N-fold stratified shuffle split
    sss = StratifiedShuffleSplit(n_splits=n_models, test_size=dev_size, random_state=seed)
    clfs = [MLPClassifier(alpha=0.01, hidden_layer_sizes=(10,), activation='relu', max_iter=1000,
                          batch_size=32, random_state=seed) for i in range(n_models)]
    acc = {'train': np.zeros((n_models,)), 'dev': np.zeros((n_models,)), 'test': np.zeros((n_models,))}
    acc_wt = {'train': np.zeros((n_models,)), 'dev': np.zeros((n_models,)), 'test': np.zeros((n_models,))}
    
    # train models and save scores
    for i, (idx_train, idx_dev) in tqdm(enumerate(sss.split(X_train, y_train)), total=n_models, bar_format=bar_format):
        clfs[i].fit(X_train[idx_train], y_train[idx_train])    
        acc['train'][i] = clfs[i].score(X_train[idx_train], y_train[idx_train])
        acc['dev'][i] = clfs[i].score(X_train[idx_dev], y_train[idx_dev])
        acc['test'][i] = clfs[i].score(X_test, y_test)
        acc_wt['train'][i] = balanced_accuracy_score(y_train[idx_train], clfs[i].predict(X_train[idx_train]),
                                                     sample_weight=w_train[idx_train])
        acc_wt['dev'][i] = balanced_accuracy_score(y_train[idx_dev], clfs[i].predict(X_train[idx_dev]),
                                                   sample_weight=w_train[idx_dev])
        acc_wt['test'][i] = balanced_accuracy_score(y_test, clfs[i].predict(X_test), sample_weight=w_test)
    
    CLFs = {'clfs': clfs, 'acc': acc, 'acc_wt': acc_wt, 'pca': pca, 'scaler': scaler}
    
    if save_path:
        dump(CLFs, save_path + '.joblib')
       
    return CLFs


def plot_scores(y_pred_mean, y_pred_std, y_true, save_path=None):
    fig, ax = plt.subplots(1,4, figsize=(7,5), sharex=True, sharey=True)
    for j in range(4):
        y_pred_mean_j = y_pred_mean[y_true==j]
        y_pred_std_j = y_pred_std[y_true==j]
        for i in range(4):
            ax[j].scatter(y_pred_std_j[:,i], y_pred_mean_j[:,i], s=40, ec='black', fc=palette[i])
        ax[j].set_title(list(bonds.keys())[j].capitalize(), color=palette[j], fontsize=fontsize)

    ax[0].set_ylabel('Mean predicted score')
    ax[1].set_xlabel('Std. predicted score', x=1)

    fig.suptitle('True class', fontsize=fontsize)
    fig.tight_layout()
    
    if save_path:
        fig.savefig(save_path + '.png', bbox_inches='tight', dpi=200)
        
        
def predict(data, CLFs, columns=[], save_path=None):
    # get model properties
    n_components = CLFs['scaler'].n_features_in_
    n_models = len(CLFs['clfs'])
    
    # predict on data
    X_eval = prepare_data(data, n_components, 0, pca=CLFs['pca'], scaler=CLFs['scaler'], columns=columns)[0]
    y_pred_mean = np.stack([CLFs['clfs'][i].predict_proba(X_eval) for i in range(n_models)]).mean(axis=0)
    y_pred_std = np.stack([CLFs['clfs'][i].predict_proba(X_eval) for i in range(n_models)]).std(axis=0)
    y_class = y_pred_mean.argmax(axis=1)
    y_class_mean = y_pred_mean[np.arange(len(y_class)), y_class]
    y_class_std = y_pred_std[np.arange(len(y_class)), y_class]
    
    # store predictions in dataframe
    data['ID'] = data.index.values
    data['ID'] = data[['ID', 'elf']].apply(lambda x: np.tile(x.ID, len(x.elf)), axis=1)
    IDs = np.hstack(data['ID'].values)
    data = data.drop(columns=['ID'])
    
    results = pd.DataFrame({'ID': IDs, 'y_class': y_class, 'y_class_mean': y_class_mean, 'y_class_std': y_class_std})
    dg = results.groupby('ID')[['y_class', 'y_class_mean', 'y_class_std']].apply(
        lambda x: [list(x.y_class), list(x.y_class_mean), list(x.y_class_std)]).apply(pd.Series).reset_index(drop=True)
    dg.columns = results[['y_class', 'y_class_mean', 'y_class_std']].columns
    data['y_class'] = dg['y_class'].copy()
    data['y_class_mean'] = dg['y_class_mean'].copy()
    data['y_class_std'] = dg['y_class_std'].copy()
    
    # save dataframe
    if save_path:
        data.to_csv(save_path + '.csv', index=False)
    
    return data


def plot_confusion_matrix(data, normalize=True, save_path=None):
    y_true = data[['bond_length', 'label']].explode('bond_length')['label'].apply(bond_to_float).values
    y_pred = data['y_class'].sum()
    
    if normalize:
        cm = confusion_matrix(y_true, y_pred, normalize='true')
    else:
        cm = confusion_matrix(y_true, y_pred)
        
    fig, ax = plt.subplots(figsize=(4.5,4.5))
    ax.imshow(np.sqrt(cm), aspect='auto', cmap=cmap_mono)
    
    labels = [k.capitalize() for k in bonds.keys()]
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if i == j: fc='white'
            else: fc = cmap_mono(100)
            if normalize:
                ax.text(j,i, '{:.2}'.format(cm[i,j]), ha='center', va='center', color=fc)
            else:
                ax.text(j,i, cm[i,j], ha='center', va='center', color=fc)
  
    ax.set_xlabel('Predicted class')
    ax.set_ylabel('True class')
    if save_path:
        fig.savefig(save_path + '.png', bbox_inches='tight', dpi=200)
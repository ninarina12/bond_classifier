import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import time, os, warnings
import cmcrameri.cm as cm

from tqdm import tqdm
from joblib import dump, load

from ase import Atoms
from ase.io import read as read_ase
from ase.data import covalent_radii
from ase.visualize.plot import plot_atoms
from ase.symbols import Symbols

from scipy.interpolate import interp1d
from scipy.stats import gaussian_kde

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.neural_network import MLPClassifier
from sklearn.semi_supervised import LabelSpreading, SelfTrainingClassifier
from sklearn.metrics import precision_recall_fscore_support, classification_report, ConfusionMatrixDisplay
from sklearn.metrics.pairwise import pairwise_distances_argmin_min
from sklearn.base import clone

bar_format = '{l_bar}{bar:10}{r_bar}{bar:-10b}'
fontsize = 14

plt.rcParams['font.family'] = 'Trebuchet MS'
plt.rcParams['font.size'] = fontsize
plt.rcParams['axes.linewidth'] = 1
plt.rcParams['mathtext.default'] = 'regular'

class ELFNN:
    def __init__(self):
        # definitions
        self.bonds = {'covalent': 0, 'metallic': 1, 'ionic': 2, 'vdw': 3}
        self.bkg = '#E4E4E4'
        self.dmap = mpl.colors.ListedColormap(['#1F608B', '#A7AFB2', '#E6A355', '#C76048'])
        self.dmap_l = mpl.colors.ListedColormap(['#9DB5C7', '#D7DADC', '#EED5B4', '#E6B6AA'])
        self.cmap = cm.lapaz
        self.norm = plt.Normalize(vmin=0, vmax=3)
        self.seed = 12
        self.trained = False
        
    
    ''' Data loading methods '''
    def load_model(self, model_path):
        saved = load(model_path + '.joblib')
        for k, v in saved.items():
            setattr(self, k, v)
        self.trained = True
    
    
    def load_data(self, dirname='data/', sort=True, structure=False, additional=False, drop_duplicates=False):
        # load data
        self.structure = structure
        self.additional = additional
        
        if self.trained:
            data = self.load_batch_data(dirname, structure=structure, additional=additional)
            
            # sort profiles
            if self.sort:
                data = self.sort_by_cdf(data)
            return data
                
        else:
            #self.data = self.load_all_data('data/unlabeled/', additional=additional)
            self.data = self.load_batch_data('data/unlabeled/', structure=structure, additional=additional)
            self.bm = self.load_batch_data('data/labeled/', structure=structure, label=True, additional=additional)
            
            # sort profiles
            self.sort = sort
            if self.sort:
                self.data = self.sort_by_cdf(self.data)
                self.bm = self.sort_by_cdf(self.bm)

            # drop duplicates
            self.drop = drop_duplicates
            if drop_duplicates:
                self.data = self.drop_duplicates(self.data)
                self.bm = self.drop_duplicates(self.bm)
    
    
    def load_all_data(self, dirname, additional=False):
        # load data aggregated into single .csv
        if dirname[-1] != '/':
            dirname += '/'
        
        # read ELF profiles
        self.columns_elf = ['e_diff', 'edge_src', 'edge_dst', 'l']
        data = pd.read_csv(dirname + 'allElfProfs.csv', header=None, usecols=range(45),
                           names=list(range(41)) + self.columns_elf)
        data = data.assign(elf=data[list(range(41))].apply(np.array, axis=1)).drop(list(range(41)), axis=1)        
        data['id'] = np.cumsum((np.diff(data.edge_src, prepend=1) < 0).astype(int)) - 1
        self.columns_elf += ['elf']
        
        if additional:
            # read additional features
            self.columns_add = ['r_src', 'r_dst', 'e_src', 'e_dst', 'g_src', 'g_dst']
            features = pd.read_csv(dirname + 'allAdditional.csv', header=None, usecols=range(6), names=self.columns_add)
            
            features['r_diff'] = features[['r_src', 'r_dst']].apply(
                lambda x: -1 if (x.r_src < 0) or (x.r_dst < 0) else np.abs(x.r_src - x.r_dst), axis=1)
            features['g_diff'] = features[['g_src', 'g_dst']].apply(lambda x: np.abs(x.g_src - x.g_dst), axis=1)
            
            data = pd.concat([features, data], axis=1)
            self.columns_add += ['r_diff', 'g_diff']
        else:
            self.columns_add = []
            
        return data
    
    
    def load_batch_data(self, dirname, structure=False, label=False, additional=False):
        # load data batched into folders for each material
        if dirname[-1] != '/':
            dirname += '/'
            
        materials = next(os.walk(dirname))[1]
        data = pd.DataFrame({'formula': materials})
        
        # read ELF profile
        data['data'] = data['formula'].apply(lambda x: self.parse_elf(x, dirname)) 
        self.columns_elf = list(data.iloc[0]['data'].keys())
        
        if structure:
            # read structure
            tqdm.pandas(desc='Parse structures', bar_format=bar_format)
            data['structure'] = data['formula'].progress_apply(lambda x: read_ase(dirname + x + '/POSCAR'))
            
        if label:
            # read label
            tqdm.pandas(desc='Parse labels', bar_format=bar_format)
            data['label'] = data['formula'].progress_apply(lambda x: self.parse_label(x, dirname))
        
        if additional:
            # read additional features
            tqdm.pandas(desc='Parse additional', bar_format=bar_format)
            data['features'] = data['formula'].progress_apply(lambda x: self.parse_additional(x, dirname))
            self.columns_add = list(data.iloc[0]['features'].keys())
            data = pd.concat([data.drop(['features', 'data'], axis=1), data['features'].apply(pd.Series),
                              data['data'].apply(pd.Series)], axis=1)
            
            data['r_diff'] = data[['r_src', 'r_dst']].apply(
                lambda x: [-1 if (s < 0) or (d < 0) else np.abs(s - d) for (s,d) in zip(x.r_src, x.r_dst)], axis=1)
            data['g_diff'] = data[['g_src', 'g_dst']].apply(
                lambda x: [np.abs(s - d) for (s,d) in zip(x.g_src, x.g_dst)], axis=1)
            self.columns_add += ['r_diff', 'g_diff']
            
        else:
            self.columns_add = []
            data = pd.concat([data.drop(['data'], axis=1), data['data'].apply(pd.Series)], axis=1)
        
        data['id'] = range(len(data))
        data = data.explode(column=self.columns_elf + self.columns_add).reset_index(drop=True)
        return data
    

    def parse_label(self, x, dirname):
        # read label data
        with open(dirname + x + '/label.txt', 'r') as f:
            label = f.read().splitlines()[0]
        return label
    
    
    def parse_elf(self, x, dirname):
        # read elf profile data
        keys = ['elf', 'e_diff', 'edge_src', 'edge_dst', 'l']
        data = dict(zip(keys, [[] for k in range(len(keys))]))
        with open(dirname + x + '/elfProfs.csv', 'r') as f:
            for line in f.readlines():
                entry = line.split(',')[:-1]
                data['elf'].append(np.array([float(k) for k in entry[:41]]))
                data['e_diff'].append(float(entry[41]))
                data['edge_src'].append(int(entry[42]))
                data['edge_dst'].append(int(entry[43]))
                data['l'].append(float(entry[44]))
        return data
    
   
    def parse_additional(self, x, dirname):
        # read additional features
        keys = ['r_src', 'r_dst', 'e_src', 'e_dst', 'g_src', 'g_dst']
        data = dict(zip(keys, [[] for k in range(len(keys))]))
        with open(dirname + x + '/additional.csv', 'r') as f:
            for line in f.readlines():
                entry = line.split(',')[:-1]
                data['r_src'].append(float(entry[0]))
                data['r_dst'].append(float(entry[1]))
                data['e_src'].append(float(entry[2]))
                data['e_dst'].append(float(entry[3]))
                data['g_src'].append(int(entry[4]))
                data['g_dst'].append(int(entry[5]))
        return data
    
    
    ''' Data processing methods '''
    def sort_by_cdf(self, data):
        # compute pdf and cdf and sort to maximize area under cdf
        data['elf_pdf'] = data[['elf','l']].apply(lambda x: x.elf/(x.elf.sum()*x.l/len(x.elf)), axis=1)
        data['elf_cdf'] = data[['elf_pdf','l']].apply(lambda x: x.l/len(x.elf_pdf)*np.cumsum(x.elf_pdf), axis=1)
        data['elf_cdf_r'] = data[['elf_pdf', 'l']].apply(
            lambda x: x.l/len(x.elf_pdf)*np.cumsum(x.elf_pdf[::-1]), axis=1)
        data['v'] = data[['elf_cdf', 'elf_cdf_r']].apply(
            lambda x: 2*(x.elf_cdf.sum() >= x.elf_cdf_r.sum()) - 1 , axis=1) 
        data['elf_orig'] = data['elf'].copy()
        data['elf'] = data[['elf_orig', 'v']].apply(lambda x: x.elf_orig[::x.v], axis=1)
        return data
    

    def drop_duplicates(self, data):
        # drop duplicate elf profiles within 6 decimals
        '''
        data['_'] = range(len(data))
        dg = pd.concat([data[['_', 'id', 'r_diff', 'g_diff', 'e_diff', 'l']],
                        data['elf'].apply(lambda x: np.trunc(x*1e6)).apply(pd.Series)], axis=1)
        index_list = dg.groupby(
            dg.columns.tolist()[1:], sort=False)['_'].apply(list).reset_index(name='idx')['idx'].tolist()
        
        tqdm.pandas(desc='Drop duplicates', bar_format=bar_format)
        data['index_orig'] = data['_'].progress_apply(lambda x: self.isin(x, index_list))
        
        data = data[~data['index_orig'].apply(pd.Series).duplicated()]
        data = data.drop(['_'], axis=1).reset_index(drop=True)
        '''
        dg = pd.concat([data[['id', 'r_diff', 'g_diff', 'e_diff', 'l']],
                        data['elf'].apply(lambda x: np.trunc(x*1e6)).apply(pd.Series)], axis=1)
        data = data[~dg.duplicated()].reset_index(drop=True)
        return data
    
    
    def regroup(self, data, by=['id', 'formula'], columns=['y_class', 'y_class_mean', 'y_class_std']):
        # by = columns shared among all bonds in a single material
        # columns = columns to be grouped for a single material
        d = data.copy()
        if 'index_orig' in data.columns:
            # expand out dropped duplicates
            print('Note: _src and _dst columns not restored due to dropped duplicates ...')
            d = d.explode('index_orig').sort_values('index_orig').reset_index(drop=True)
            columns += ['r_diff', 'g_diff', 'elf', 'e_diff', 'l']
        else:
            columns += self.columns_add + self.columns_elf
            
        if 'structure' in by:
            d['structure'] = d['structure'].apply(lambda x: str({k: v.tolist() for (k,v) in x.todict().items()}))
            
        d = d.groupby(by, as_index=False, sort=False)[columns].agg(list)
        d = d.drop('id', axis=1)
        
        if 'structure' in by:
            d['structure'] = d['structure'].apply(lambda x: Atoms.fromdict(eval(x)))
            
        return d

    
    def get_distances(self, column='elf'):
        # calculate minimal cosine, Euclidean, and Earth mover's distances between unlabeled and labeled data
        z_data = np.stack(self.data[column].values)
        z_bm = np.stack(self.bm[column].values)
        cdf_bm = np.stack(self.bm.apply(lambda x: 0.5*((1 + x.v)*x.elf_cdf + (1 - x.v)*x.elf_cdf_r), axis=1).values)

        _, self.data['d_cos'] = pairwise_distances_argmin_min(z_data, z_bm, metric='cosine')
        _, self.data['d_euc'] = pairwise_distances_argmin_min(z_data, z_bm, metric='euclidean')

        tqdm.pandas(desc='Compute EMD', bar_format=bar_format)
        self.data['d_emd'] = self.data.progress_apply(lambda x: np.abs(
            0.5*((1 + x.v)*x.elf_cdf + (1 - x.v)*x.elf_cdf_r)[None,:] - cdf_bm).sum(axis=-1).min(axis=-1), axis=1)
    
    
    def isin(self, x, x_list):
        for k in x_list:
            if x in k:
                break
        return k


    def bond_to_float(self, x):
        try: len(x)
        except:
            return self.bonds[x]
        else:
            return np.array([self.bonds[k] for k in x])
    
    
    ''' Data transform methods '''
    def pca_fit(self, ev=0.9995):
        pca = PCA()
        pca.fit(np.stack(self.data['elf'].values))
        n_components = np.argmin(np.abs(pca.explained_variance_ratio_.cumsum() - ev)) + 1
        
        self.pca = PCA(n_components=n_components)
        self.pca.fit(np.stack(self.data['elf'].values))
        
        self.data['z'] = self.data['elf'].apply(lambda x: self.pca.transform([x])[0])
        self.bm['z'] = self.bm['elf'].apply(lambda x: self.pca.transform([x])[0])
            
        if self.sort:
            self.pca_orig = PCA(n_components=n_components)
            self.pca_orig.fit(np.stack(self.data['elf_orig'].values))
            self.data['z_orig'] = self.data['elf_orig'].apply(lambda x: self.pca_orig.transform([x])[0])
            self.bm['z_orig'] = self.bm['elf_orig'].apply(lambda x: self.pca_orig.transform([x])[0])
        
        
    def pca_transform(self, data):
        if not hasattr(self, 'pca'):
            print('Fitting PCA using default parameters ...')
            self.pca_fit()
        
        data['z'] = data['elf'].apply(lambda x: self.pca.transform([x])[0])
        
        if hasattr(self, 'pca_orig'):
            data['z_orig'] = data['elf_orig'].apply(lambda x: self.pca_orig.transform([x])[0])
        return data
        
    
    def prepare_inputs(self, inputs=['elf'], data=None):
        self.inputs = inputs
        try: len(data)
        except:
            self.bm['input'] = self.bm[inputs].apply(lambda x:
                    np.concatenate([[x[k]] if isinstance(x[k], (float, int)) else x[k] for k in inputs]), axis=1)

            self.data['input'] = self.data[inputs].apply(lambda x:
                    np.concatenate([[x[k]] if isinstance(x[k], (float, int)) else x[k] for k in inputs]), axis=1)
        else:
            data['input'] = data[inputs].apply(lambda x:
                    np.concatenate([[x[k]] if isinstance(x[k], (float, int)) else x[k] for k in inputs]), axis=1)
            return data
        
        
    def scaler_fit(self):
        self.scaler = StandardScaler()
        self.scaler.fit(np.stack(self.data['input'].values))
        self.data['input_scaled'] = self.data['input'].apply(lambda x: self.scaler.transform([x])[0])
        self.bm['input_scaled'] = self.bm['input'].apply(lambda x: self.scaler.transform([x])[0])
        
    
    def scaler_transform(self, data):
        if not hasattr(self, 'scaler'):
            print('Fitting standard scaler ...')
            self.scaler_fit()
        
        data['input_scaled'] = data['input'].apply(lambda x: self.scaler.transform([x])[0])
        return data
        
            
    ''' Train/test split methods '''        
    def stratified_split(self, test_size, y='label'):
        self.test_size = test_size        
        self.idx_train, self.idx_test = train_test_split(range(len(self.bm)),
                                                         test_size=test_size,
                                                         random_state=self.seed,
                                                         stratify=self.bond_to_float(self.bm['label'].tolist()))
        
        t = '_scaled' if hasattr(self, 'scaler') else ''
            
        # labeled training data
        self.X_bm = np.stack(self.bm.iloc[self.idx_train]['input' + t].values)
        if y == 'label':
            self.y_bm = self.bond_to_float(self.bm.iloc[self.idx_train]['label'].tolist())
        else:
            self.y_bm = np.array(self.bm.iloc[self.idx_train][y].tolist())
            
        # unlabeled data
        self.X_data = np.stack(self.data['input' + t].values)
        self.y_data = [-1 for _ in range(len(self.data))]
        self.X_train = np.concatenate((self.X_bm, self.X_data))
        self.y_train = np.concatenate((self.y_bm, self.y_data))
        
        # labeled testing data
        self.X_test = np.stack(self.bm.iloc[self.idx_test]['input' + t].values)
        if y == 'label':
            self.y_test = self.bond_to_float(self.bm.iloc[self.idx_test]['label'].tolist())
        else:
            self.y_test = np.array(self.bm.iloc[self.idx_test][y].tolist())
        
        _, counts = np.unique(self.y_bm, return_counts=True)
        self.class_weight = counts.sum()/counts
        self.class_weight /= self.class_weight.sum()
        print('Class weights:', self.class_weight)
        
    
    def kfold_split(self, n_splits, test_size):
        self.n_splits = n_splits
        self.kfold = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=self.seed)
        
    
    ''' Machine learning methods '''
    def init_clf(self, alpha=0.01, hidden_layer_sizes=(10,), activation='relu', max_iter=1000, batch_size=32):
        self.clf = MLPClassifier(alpha=alpha, hidden_layer_sizes=hidden_layer_sizes, activation=activation,
                                 max_iter=max_iter, batch_size=batch_size, random_state=self.seed)
        
    
    def train_clf(self, model_path='clf'):
        acc_bm = np.zeros((self.n_splits,))
        acc_train = np.zeros_like(acc_bm)
        acc_test = np.zeros_like(acc_bm)
        clf = [clone(self.clf) for i in range(self.n_splits)]

        for i, (idx_train, idx_dev) in tqdm(enumerate(self.kfold.split(self.X_bm, self.y_bm)),
                                            total=self.n_splits, bar_format=bar_format):
            
            X_train = self.X_bm[idx_train]
            y_train = self.y_bm[idx_train]
                
            clf[i].fit(X_train, y_train)
            acc_bm[i] = clf[i].score(self.X_bm, self.y_bm, sample_weight=self.class_weight[self.y_bm])
            acc_train[i] = clf[i].score(X_train, y_train, sample_weight=self.class_weight[y_train])
            acc_test[i] = clf[i].score(self.X_test, self.y_test, sample_weight=self.class_weight[self.y_test])
        
        clf_stats = {'acc_bm': acc_bm,
                     'acc_train': acc_train,
                     'acc_test': acc_test

        }

        saved = {'sort': self.sort,
                 'inputs': self.inputs,
                 'pca': self.pca,
                 'scaler': self.scaler,
                 'class_weight': self.class_weight,
                 'clf_stats': clf_stats,
                 'clf': clf     
        }
        
        if self.sort:
            saved['pca_orig'] = self.pca_orig
            
        self.clf_stats = clf_stats
        self.clf = clf
        self.trained = True
        dump(saved, model_path + '.joblib')
    
        
    def self_train_clf(self, threshold=[0.4], max_iter=50, model_path='stc'):
        if isinstance(threshold, float):
            threshold = [threshold]
        
        # train and save models
        n_iter = np.zeros((len(threshold), self.n_splits))
        f_labeled = np.zeros_like(n_iter)
        acc_bm = np.zeros_like(n_iter)
        acc_train = np.zeros_like(n_iter)
        acc_test = np.zeros_like(n_iter)
        clf = [[SelfTrainingClassifier(clone(self.clf), threshold=threshold[j], max_iter=max_iter)
                for i in range(self.n_splits)] for j in range(len(threshold))]

        for i, (idx_train, idx_dev) in tqdm(enumerate(self.kfold.split(self.X_bm, self.y_bm)),
                                            total=self.n_splits, bar_format=bar_format):
            
            X_train = np.concatenate((self.X_bm[idx_train], self.X_data))
            y_train = np.concatenate((self.y_bm[idx_train], self.y_data))
                
            for j in range(len(threshold)):
                clf[j][i].fit(X_train, y_train)

                f_labeled[j,i] = len(clf[j][i].labeled_iter_[clf[j][i].labeled_iter_ > 0])/len(self.X_data)
                n_iter[j,i] = clf[j][i].n_iter_
                acc_bm[j,i] = clf[j][i].base_estimator_.score(self.X_bm, self.y_bm,
                                                              sample_weight=self.class_weight[self.y_bm])
                acc_train[j,i] = clf[j][i].base_estimator_.score(self.X_bm[idx_train], self.y_bm[idx_train],
                                                              sample_weight=self.class_weight[self.y_bm[idx_train]])
                acc_test[j,i] = clf[j][i].base_estimator_.score(self.X_test, self.y_test,
                                                                sample_weight=self.class_weight[self.y_test])
        
        clf_stats = {'threshold': threshold,
                     'n_iter': n_iter,
                     'f_labeled': f_labeled,
                     'acc_bm': acc_bm,
                     'acc_train': acc_bm,
                     'acc_test': acc_test

        }

        saved = {'sort': self.sort,
                 'inputs': self.inputs,
                 'pca': self.pca,
                 'scaler': self.scaler,
                 'class_weight': self.class_weight,
                 'clf_stats': clf_stats,
                 'clf': clf     
        }
        
        if self.sort:
            saved['pca_orig'] = self.pca_orig
            
        self.clf_stats = clf_stats
        self.clf = clf
        self.trained = True
        dump(saved, model_path + '.joblib')
        
    
    def predict(self, data, threshold=0):
        if isinstance(self.clf[0], list):
            if isinstance(threshold, int):
                print('Threshold:', self.clf_stats['threshold'][threshold])
                clf = self.clf[threshold]
            else:
                i = np.argmin(np.abs(threshold - np.array(self.clf_stats['threshold'])))
                print('Threshold:', self.clf_stats['threshold'][i])
                clf = self.clf[i]
        else:
            clf = self.clf
        
        t = '_scaled' if hasattr(self, 'scaler') else ''
        X = np.stack(data['input' + t].values)
        y_pred = np.stack([clf[i].predict_proba(X) for i in range(len(clf))])
        
        data['y_pred_mean'] = y_pred.mean(axis=0).tolist()
        data['y_pred_std'] = y_pred.std(axis=0).tolist()
        data['y_class'] = data['y_pred_mean'].apply(np.argmax)
        data['y_class_mean'] = data[['y_pred_mean', 'y_class']].apply(lambda x: x.y_pred_mean[x.y_class], axis=1)
        data['y_class_std'] = data[['y_pred_std', 'y_class']].apply(lambda x: x.y_pred_std[x.y_class], axis=1)
        return data


    ''' Plotting methods '''
    def plot_structure(self, struct):
        # plot crystal structure
        Z = struct.get_atomic_numbers()
        norm = plt.Normalize(vmin=Z.min(), vmax=Z.max())
        
        fig, ax = plt.subplots(figsize=(3,3))
        plot_atoms(struct, ax, colors=self.dmap(norm(Z)), radii=0.3*covalent_radii[Z], rotation=('0x,0y,0z'))
        ax.set_xlabel('x $(\AA)$')
        ax.set_ylabel('y $(\AA)$')

        Z = np.unique(Z)
        S = Symbols(Z)
        R = 0.3*covalent_radii[Z]
        x = ax.get_xlim()[1] + 1
        y = ax.get_ylim()[1]
        s = 0.8 if y > 8 else 0.4
        k = 0
        for i, r in enumerate(R):
            k += r
            ax.add_patch(plt.Circle((x, y - k), r, ec='black', fc=self.dmap(norm(Z[i])), clip_on=False))
            ax.text(x + R.max() + 0.3, y - k, S[i], ha='left', va='center')
            k += r + s
        return fig
    
    
    def plot_pca_distribution(self, orig=False):
        t = '_orig' if (orig and self.sort) else ''
        z_data = np.stack(self.data['z' + t].values)
        z_bm = np.stack(self.bm['z' + t].values)
        c = min(5, z_data.shape[-1])
        bins = 50

        fig, ax = plt.subplots(c,c+1, figsize=(2.5*c,2.4*c), sharex='col',
                               gridspec_kw={'width_ratios': [1]*c + [0.07]})
        h = np.histogram2d(z_data[:,0], z_data[:,1], bins=bins)[0]
        norm = mpl.colors.LogNorm(vmin=1, vmax=10**np.ceil(np.log10(h.max())))
        lmax = -np.Inf
        lmin = np.Inf
        for i in range(c):
            for j in range(i+1,c):
                ax[j,i].hist2d(z_data[:,i], z_data[:,j], bins=bins, cmap=self.cmap, norm=norm)
                ax[i,j].hist2d(z_bm[:,i], z_bm[:,j], bins=bins, cmap=self.cmap, norm=norm)
                ax[j,i].set_facecolor(self.bkg)
                ax[i,j].set_facecolor(self.bkg)
                lmax = max(lmax, np.max([ax[j,i].get_xlim()[1], ax[j,i].get_ylim()[1],
                                         ax[i,j].get_xlim()[1], ax[i,j].get_ylim()[1]]))
                lmin = min(lmin, np.min([ax[j,i].get_xlim()[0], ax[j,i].get_ylim()[0],
                                         ax[i,j].get_xlim()[0], ax[i,j].get_ylim()[0]]))

        for i in range(c):
            h, b = np.histogram(z_data[:,i], bins=bins, density=True)
            ax[i,i].stairs(h, b, color=self.cmap(50), fill=True, alpha=0.2)
            ax[i,i].stairs(h, b, color=self.cmap(50))

            h, b = np.histogram(z_bm[:,i], bins=bins, density=True)
            ax[i,i].stairs(h, b, color=self.cmap(130), fill=True, alpha=0.2)
            ax[i,i].stairs(h, b, color=self.cmap(130))

            if i == 0:
                ax[i,i].text(0.9, 0.85, 'Unlabeled', color=self.cmap(50), ha='right', va='center',
                             fontsize=fontsize-2, transform=ax[i,i].transAxes)
                ax[i,i].text(0.9, 0.72, 'Labeled', color=self.cmap(130), ha='right', va='center',
                             fontsize=fontsize-2, transform=ax[i,i].transAxes)

            if i < c - 1:
                ax[i,-1].remove()

            for j in range(c):
                ax[j,i].set_xlim(lmin,lmax)
                ax[j,i].set_xticks([])
                ax[j,i].set_yticks([])

            ax[i,0].set_ylabel('$PC_{' + str(i+1) + '}$')
            ax[-1,i].set_xlabel('$PC_{' + str(i+1) + '}$')

        sm = mpl.cm.ScalarMappable(cmap=self.cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, cax=ax[-1,-1])
        cbar.ax.set_ylabel('Counts', labelpad=10)
        fig.subplots_adjust(hspace=0.1, wspace=0.1)
        return fig
    
    
    def plot_pca_labels(self, orig=False):
        t = '_orig' if (orig and self.sort) else ''
        z_data = np.stack(self.data['z' + t].values)
        z_bm = np.stack(self.bm['z' + t].values)
        y = self.bond_to_float(self.bm['label'].tolist())

        c = min(5, z_data.shape[-1])
        bins = 30

        fig, ax = plt.subplots(c,c+1, figsize=(2.5*c,2.5*c), sharex='col',
                               gridspec_kw={'width_ratios': [1]*c + [0.07]})

        norm = plt.Normalize(vmin=0, vmax=3)
        lmax = -np.Inf
        lmin = np.Inf
        for i in range(c):
            for j in range(i+1,c):
                ax[j,i].scatter(z_data[:,i], z_data[:,j], color='#E4E4E4', s=16, alpha=0.5, ec='none',
                                cmap=self.dmap, norm=norm)
                for k in range(4):
                    ax[j,i].scatter(z_bm[y==k,i], z_bm[y==k,j], s=16, color=self.dmap_l(norm(k)),
                                    ec=self.dmap(norm(k)))

                ax[i,j].remove()
                lmax = max(lmax, np.max([ax[j,i].get_xlim()[1], ax[j,i].get_ylim()[1]]))
                lmin = min(lmin, np.min([ax[j,i].get_xlim()[0], ax[j,i].get_ylim()[0]]))

        for i in range(c):
            h, b = np.histogram(z_data[:,i], bins=bins, density=True)
            ax[i,i].stairs(h, b, color='#E4E4E4', fill=True)
            ax_ = ax[i,i].twinx()
            ax_.axis('off')
        
            for k in range(4):
                h, b = np.histogram(z_bm[y==k,i], bins=bins, density=True)
                ax_.stairs(h, b, color=self.dmap(norm(k)), fill=True, alpha=0.5)
                ax_.stairs(h, b, color=self.dmap(norm(k)))

            if i == 0:
                ax[i,i].text(0.9, 0.85, 'Unlabeled', color='#C0C0C0', ha='right', va='center',
                             fontsize=fontsize-2, transform=ax[i,i].transAxes)

            if i < c - 1:
                ax[i,-1].remove()

            for j in range(c):
                ax[j,i].set_xlim(lmin,lmax)
                ax[j,i].set_xticks([])
                ax[j,i].set_yticks([])

            ax[i,0].set_ylabel('$PC_{' + str(i+1) + '}$')
            ax[-1,i].set_xlabel('$PC_{' + str(i+1) + '}$')

        sm = mpl.cm.ScalarMappable(cmap=self.dmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, cax=ax[-1,-1], ticks=[(i-0.5)*3/4 for i in range(1,5)])
        cbar.ax.set_ylabel('Class', labelpad=10)
        cbar.ax.set_yticklabels([k.capitalize() for k in self.bonds.keys()], fontsize=fontsize-2)
        fig.subplots_adjust(hspace=0.1, wspace=0.1)
        return fig


    def plot_pca_distances(self, orig=False):
        t = '_orig' if (orig and self.sort) else ''
        z = np.stack(self.data['z' + t].values)
        fig, ax = plt.subplots(1,4, figsize=(9.5,3), gridspec_kw={'width_ratios': [1,1,1,0.05]})
        norm = plt.Normalize(vmin=0, vmax=1)
        
        for i, (c, title) in enumerate(zip(['d_cos', 'd_euc', 'd_emd'],
                                           ['Cosine', 'Euclidean', "Earth mover's"])):
            d = np.stack(self.data[c].values)
            idx = np.argsort(d)
            d_std = (d - d.min())/(d.max() - d.min())
            ax[i].scatter(z[idx,0], z[idx,1], c=d_std[idx], s=20, alpha=0.7, ec='none', cmap=self.cmap, norm=norm)
            ax[i].set_facecolor(self.bkg)
            ax[i].set_xticks([])
            ax[i].set_yticks([])
            ax[i].set_title(title, fontsize=fontsize)

        sm = mpl.cm.ScalarMappable(cmap=self.cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, cax=ax[-1])
        cbar.ax.set_ylabel('Standardized distance', labelpad=16)
        fig.subplots_adjust(wspace=0.1)
        return fig
    
    
    def plot_pca_features(self, col, orig=False):
        col_names = {'l': 'Bond length ($\AA$)', 'e_diff': 'Electronegativity diff.'}
        t = '_orig' if (orig and self.sort) else ''

        z_data = np.stack(self.data['z' + t].values)
        z_bm = np.stack(self.bm['z' + t].values)
        y_data = np.stack(self.data[col].values)
        y_bm = np.stack(self.bm[col].values)
        i_data = np.argsort(y_data)
        i_bm = np.argsort(y_bm)

        c = min(5, z_data.shape[-1])
        bins = 50

        fig, ax = plt.subplots(c,c+1, figsize=(2.5*c,2.5*c), sharex='col',
                               gridspec_kw={'width_ratios': [1]*c + [0.07]})
        vmin, vmax = 0, np.concatenate([y_data, y_bm]).max()
        norm = plt.Normalize(vmin=vmin, vmax=vmax)
        lmax = -np.Inf
        lmin = np.Inf
        for i in range(c):
            for j in range(i+1,c):
                ax[j,i].scatter(z_data[i_data,i], z_data[i_data,j], c=y_data[i_data], s=6, cmap=self.cmap, norm=norm)
                ax[i,j].scatter(z_bm[i_bm,i], z_bm[i_bm,j], c=y_bm[i_bm], s=6, cmap=self.cmap, norm=norm)
                ax[j,i].set_facecolor(self.bkg)
                ax[i,j].set_facecolor(self.bkg)
                lmax = max(lmax, np.max([ax[j,i].get_xlim()[1], ax[j,i].get_ylim()[1],
                                         ax[i,j].get_xlim()[1], ax[i,j].get_ylim()[1]]))
                lmin = min(lmin, np.min([ax[j,i].get_xlim()[0], ax[j,i].get_ylim()[0],
                                         ax[i,j].get_xlim()[0], ax[i,j].get_ylim()[0]]))

        gmap_50 = mpl.colors.LinearSegmentedColormap.from_list('gmap', [(1,1,1,0), self.cmap(50)])
        gmap_130 = mpl.colors.LinearSegmentedColormap.from_list('gmap', [(1,1,1,0), self.cmap(130)])
        
        xx, yy = np.mgrid[lmin:lmax:eval(str(bins)+'j'), vmin:vmax:eval(str(bins)+'j')]
        for i in range(c):
            ax[i,i].contourf(xx, yy, self.kde(z_data[:,i], y_data, xx, yy), cmap=gmap_50, levels=10)
            ax[i,i].contourf(xx, yy, self.kde(z_bm[:,i], y_bm, xx, yy), cmap=gmap_130, levels=10)
            ax[i,i].set_ylim([vmin, vmax])

            if i == 0:
                ax[i,i].text(0.9, 0.85, 'Unlabeled', color=self.cmap(50), ha='right', va='center',
                             fontsize=fontsize-2, transform=ax[i,i].transAxes)
                ax[i,i].text(0.9, 0.72, 'Labeled', color=self.cmap(130), ha='right', va='center',
                             fontsize=fontsize-2, transform=ax[i,i].transAxes)

            if i < c - 1:
                ax[i,-1].remove()

            for j in range(c):
                ax[j,i].set_xlim(lmin,lmax)
                ax[j,i].set_xticks([])
                ax[j,i].set_yticks([])

            ax[i,0].set_ylabel('$PC_{' + str(i+1) + '}$')
            ax[-1,i].set_xlabel('$PC_{' + str(i+1) + '}$')

        sm = mpl.cm.ScalarMappable(cmap=self.cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, cax=ax[-1,-1])
        cbar.ax.set_ylabel(col_names[col], labelpad=10)
        fig.subplots_adjust(hspace=0.1, wspace=0.1)
        return fig
    
    
    def kde(self, x, y, xx, yy):
        pos = np.vstack([xx.ravel(), yy.ravel()])
        val = np.vstack([x, y])
        ker = gaussian_kde(val)
        return np.reshape(ker(pos).T, xx.shape)
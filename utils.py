import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import time, os
import cmcrameri.cm as cm
import networkx as nx

from tqdm import tqdm
from joblib import dump, load

from scipy.interpolate import griddata
from scipy.spatial import Delaunay

from ase import Atom, Atoms
from ase.symbols import Symbols
from ase.io import read as read_ase
from ase.data import covalent_radii
from ase.visualize.plot import plot_atoms
from ase.neighborlist import neighbor_list

from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import pairwise_distances_argmin_min
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import clone

fontsize = 12
plt.rcParams['font.family'] = 'Lato'
plt.rcParams['font.size'] = fontsize
plt.rcParams['axes.linewidth'] = 1
plt.rcParams['mathtext.default'] = 'regular'


class ELF:
    def __init__(self, n_classes=4):
        # definitions
        self.labels = ['covalent', 'metallic', 'ionic', 'vdw']
        self.bkg = '#E4E4E4'
        self.palette = ['#1F608B', '#A7AFB2', '#E6A355', '#C76048']
        self.palette_l = ['#9DB5C7', '#D7DADC', '#EED5B4', '#E6B6AA']
        
        self.n_classes = n_classes
        if self.n_classes == 2:
            self.bonds = dict(zip(self.labels, [0,0,1,1]))
            self.palette = [self.palette[0], self.palette[2]]
            self.palette_l = [self.palette_l[0], self.palette_l[2]]
        elif self.n_classes == 3:
            self.bonds = dict(zip(self.labels, [0,1,2,2]))
            self.palette = self.palette[:3]
            self.palette_l = self.palette_l[:3]
        else:
            self.bonds = dict(zip(self.labels, range(len(self.labels))))
        
        self.dmap = mpl.colors.ListedColormap(self.palette)
        self.dmap_l = mpl.colors.ListedColormap(self.palette_l)
        self.dmap_c = mpl.colors.LinearSegmentedColormap.from_list('dmap_c', self.palette)
        self.cmap = cm.oslo
        self.norm = plt.Normalize(vmin=0, vmax=max(self.bonds.values()))
        
        self.bar_format = '{l_bar}{bar:10}{r_bar}{bar:-10b}'
        
        
    def bond_to_float(self, x):
        try: len(x)
        except: return self.bonds[x]
        else: return np.array([self.bonds[k] for k in x])
        
        

class ELFData(ELF):
    def __init__(self):
        super().__init__()        
        
    def load_data(self, dirname='data/', structure=False, label=False, additional=False):
        # load data
        self.data = self.load_batch_data(dirname, structure, label, additional)
    
    
    def load_batch_data(self, dirname, structure=False, label=False, additional=False):
        # load data batched into folders for each material
        if dirname[-1] != '/':
            dirname += '/'
        self.dirname = dirname
        
        materials = next(os.walk(dirname))[1]
        data = pd.DataFrame({'formula': materials})
        
        # read ELF profile
        tqdm.pandas(desc='Parse profiles', bar_format=self.bar_format)
        data['data'] = data['formula'].progress_apply(lambda x: self.parse_elf(x, dirname)) 
        self.columns_bonds = list(data.iloc[0]['data'].keys())
        
        if structure:
            # read structure
            tqdm.pandas(desc='Parse structures', bar_format=self.bar_format)
            data['structure'] = data['formula'].progress_apply(lambda x: read_ase(dirname + x + '/POSCAR'))
            
        if label:
            # read label
            tqdm.pandas(desc='Parse labels', bar_format=self.bar_format)
            data['label'] = data['formula'].progress_apply(lambda x: self.parse_label(x, dirname))
        
        if additional:
            # read additional features
            tqdm.pandas(desc='Parse additional', bar_format=self.bar_format)
            data['features'] = data['formula'].progress_apply(lambda x: self.parse_additional(x, dirname))
            self.columns_bonds += list(data.iloc[0]['features'].keys())
            data = pd.concat([data.drop(['features', 'data'], axis=1), data['features'].apply(pd.Series),
                              data['data'].apply(pd.Series)], axis=1)
            
            data['r_diff'] = data[['r_src', 'r_dst']].apply(
                lambda x: [-1 if (s < 0) or (d < 0) else np.abs(s - d) for (s,d) in zip(x.r_src, x.r_dst)], axis=1)
            data['g_diff'] = data[['g_src', 'g_dst']].apply(
                lambda x: [np.abs(s - d) for (s,d) in zip(x.g_src, x.g_dst)], axis=1)
            self.columns_bonds += ['r_diff', 'g_diff']
            
        else:
            data = pd.concat([data.drop(['data'], axis=1), data['data'].apply(pd.Series)], axis=1)
        
        data = data.sort_values(by='formula').reset_index(drop=True)
        return data
    
    
    def load_processed(self, dirname='data/', structure=False):
        if dirname[-1] != '/':
            dirname += '/'
        self.dirname = dirname
        
        filename = '/processed/'.join(dirname.split('/')[:-1]) + '.csv'
        
        self.data = pd.read_csv(filename)
        tqdm.pandas(desc='Parse profiles', bar_format=self.bar_format)
        self.data['elf'] = self.data['elf'].progress_apply(eval).apply(np.array)
        
        if structure:
        # read structure
            tqdm.pandas(desc='Parse structures', bar_format=self.bar_format)
            self.data['structure'] = self.data['formula'].progress_apply(lambda x: read_ase(dirname + str(x) + '/POSCAR'))
    
    
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
    
    
    def get_species(self):
        tqdm.pandas(desc='Get species src', bar_format=self.bar_format)
        self.data['specie_src'] = self.data[['structure', 'edge_src']].progress_apply(
            lambda x: [x.structure.get_chemical_symbols()[k] for k in x.edge_src], axis=1)
        tqdm.pandas(desc='Get species dst', bar_format=self.bar_format)
        self.data['specie_dst'] = self.data[['structure', 'edge_dst']].progress_apply(
            lambda x: [x.structure.get_chemical_symbols()[k] for k in x.edge_dst], axis=1)
        self.columns_bonds += ['specie_src', 'specie_dst']
                
    
    def get_projection(self, u, v):
        l = np.linalg.norm(u)
        return np.dot(v,u)/l**2
    
    
    def get_pdf_cdf(self):
        # compute pdf and cdf
        self.data['A'] = self.data['elf'].apply(lambda x: x.sum())
        self.data['pdf'] = self.data['elf'].apply(lambda x: x/x.sum())
        self.data['cdf'] = self.data['pdf'].apply(lambda x: x.cumsum())
        
    
    def get_distances(self, ref, data='data', column='elf', metric='euclidean'):
        x_data = np.stack(getattr(self, data)[column].values)
        x_ref = np.stack(getattr(ref, data)[column].values)
        _, getattr(self, data)['d_' + column + '_' + metric[:3]] = pairwise_distances_argmin_min(x_data, x_ref, metric=metric)
    
    
    def get_graph(self, nodes, edges):
        G = nx.Graph()
        G.add_nodes_from(k for k in nodes.items())
        G.add_edges_from(edges)
        return G


    def get_mdhs(self, column, n_classes, threshold=0.5, formulas=None):       
        columns = [column + '_pred_proba', column + '_pred', 'edge_src', 'edge_dst', 'specie_src', 'specie_dst']
        try: len(formulas)
        except:
            mdh = self.data.groupby('formula', as_index=False)[columns].agg(list)
        else:
            mdh = self.data[self.data['formula'].apply(
                lambda x: x in formulas)].groupby('formula', as_index=False)[columns].agg(list)

        # threshold
        mdh = mdh[mdh[column + '_pred_proba'].apply(np.array).apply(
            lambda x: np.all(x >= threshold))].reset_index(drop=True)

        # read structures
        mdh['structure'] = mdh['formula'].apply(lambda x: read_ase(self.dirname + str(x) + '/POSCAR'))

        # parse nodes and edges
        mdh['nodes'] = mdh['structure'].apply(
            lambda x: {k: {'symbol': x[k].symbol, 'number': x[k].number, 'radius': covalent_radii[x[k].number],
                           'pos': x.get_positions()[k,:-1]} for k in range(len(x))})
        mdh['edges'] = mdh[['edge_src', 'edge_dst', column + '_pred_proba', column + '_pred']].apply(
            lambda x: [(i,j, {'class': k, 'score': s}) for (i,j,k,s) in
                       zip(x.edge_src, x.edge_dst, x[column + '_pred'], x[column + '_pred_proba']) if k < (n_classes-1)], axis=1)

        # get number of connected components
        mdh['n_components'] = mdh[['nodes', 'edges']].apply(
            lambda x: nx.number_connected_components(self.get_graph(x.nodes, x.edges)), axis=1)

        # filter out mdhs
        mdh = mdh[mdh['n_components'] > 1].reset_index(drop=True)
        print('Number of candidate MDHs:', len(mdh))
        return mdh


    def tag_mixed(self):
        mixed = [[] for k in range(len(self.data))]

        for idx, sample in enumerate(tqdm(self.data.itertuples(), total=len(self.data), bar_format=self.bar_format)):
            # extract species and sort by bond lengths in increasing order
            specie_src = np.array(sample.specie_src)
            specie_dst = np.array(sample.specie_dst)
            l = np.array(sample.l)

            srt = np.argsort(l)
            specie_src = specie_src[srt]
            specie_dst = specie_dst[srt]
            l = l[srt]

            # set cutoff distance relative to bond length
            cutoffs = {(i,j):np.ceil(100*np.sqrt(2)*k)/100. for (i,j,k) in zip(specie_src, specie_dst, l)}

            # if 2 different cutoffs exist for the same atom pair, pick the maximum
            for k in cutoffs.keys():
                try: cutoffs[(k[1],k[0])]
                except: pass
                else: cutoffs[(k[0],k[1])] = max(cutoffs[(k[0],k[1])], cutoffs[(k[1],k[0])])

            edge_src, edge_dst, edge_vec = neighbor_list("ijD", sample.structure, cutoffs)

            mixed[idx] = [False for k in range(len(specie_src))]
            for i, (src, dst) in enumerate(zip(np.array(sample.edge_src), np.array(sample.edge_dst))):
                for j, v in enumerate(edge_vec[(edge_src==src) & (edge_dst != dst)]):

                    u = sample.structure.get_distance(src, dst, mic=True, vector=True)
                    lam = self.get_projection(u, v)

                    if (lam > 0) & (lam < 1):
                        l = np.linalg.norm(u)
                        d = np.linalg.norm(v - lam*u)
                        if (d < lam*l) & (d < (1 - lam)*l):
                            # if an intermediate atom exists, label the bond
                            mixed[idx][i] = True
                            break
        self.data['mixed'] = mixed
        self.columns_bonds += ['mixed']
                    
    
    def expand_data(self):
        self.data['id'] = range(len(self.data))

        columns = self.data.columns.tolist()
        columns.remove('structure')

        self.data = self.data[columns].explode(column=self.columns_bonds).reset_index(drop=True)
        self.data['elf'] = self.data['elf'].apply(lambda x: x.tolist())
        
    
    def sort_by_cdf(self):
        # sort to maximize area under cdf
        self.data['cdf_r'] = self.data['pdf'].apply(lambda x: x[::-1].cumsum())
        self.data['v'] = self.data[['cdf', 'cdf_r']].apply(
            lambda x: 2*(x.cdf.sum() >= x.cdf_r.sum()) - 1 , axis=1) 
        self.data['elf_srt'] = self.data[['elf', 'v']].apply(lambda x: x.elf[::x.v], axis=1)
        self.data['pdf_srt'] = self.data[['pdf', 'v']].apply(lambda x: x.pdf[::x.v], axis=1)
        self.data['cdf_srt'] = self.data['pdf_srt'].apply(lambda x: x.cumsum())
        self.data = self.data.drop(labels=['cdf_r', 'v'], axis=1)
    

    def drop_duplicates(self, column='elf', precision=1e-9):
        # drop duplicate elf profiles within precision of selected column
        tqdm.pandas(desc='Drop duplicates', bar_format=self.bar_format)
        dg = pd.concat([self.data[['id', 'r_diff', 'g_diff', 'e_diff', 'l']],
                        self.data[column].progress_apply(lambda x: np.trunc(x/precision)).apply(pd.Series)], axis=1)
        self.data_nodup = self.data[~dg.duplicated()].reset_index(drop=True)
    
    
    def plot_structure(self, struct, rotation=('0x,0y,0z')):
        # plot crystal structure
        numbers = struct.get_atomic_numbers()
        symbols = Symbols(numbers)
        usymbols = np.unique(symbols)
        sym2num = dict(zip(usymbols, range(len(usymbols))))
        norm = plt.Normalize(vmin=0, vmax=len(usymbols)-1)

        fig, ax = plt.subplots(figsize=(3,3))
        colors = [self.dmap_c(norm(sym2num[k])) for k in symbols]
        plot_atoms(struct, ax, colors=colors, radii=0.3*covalent_radii[numbers], rotation=rotation)
        ax.set_xlabel('x $(\AA)$')
        ax.set_ylabel('y $(\AA)$')

        numbers = np.unique(numbers)
        symbols = Symbols(numbers)
        radii = 0.3*covalent_radii[numbers]
        x = ax.get_xlim()[1] + 1
        y = ax.get_ylim()[1]
        s = 0.8 if y > 8 else 0.4
        k = 0
        for i, r in enumerate(radii):
            k += r
            ax.add_patch(plt.Circle((x, y - k), r, ec='black', fc=self.dmap_c(norm(sym2num[symbols[i]])), clip_on=False))
            ax.text(x + radii.max() + 0.3, y - k, symbols[i], ha='left', va='center')
            k += r + s
        return fig
    
    
    
class ELFModel(ELF):
    def __init__(self, n_classes=4):
        super().__init__(n_classes)
        self.seed = 12
    
    
    def _adjacent_values(self, vals, q1, q3):
        upper_adjacent_value = q3 + 1.5*(q3 - q1)
        upper_adjacent_value = np.clip(upper_adjacent_value, q3, vals[-1])

        lower_adjacent_value = q1 - 1.5*(q3 - q1)
        lower_adjacent_value = np.clip(lower_adjacent_value, vals[0], q1)
        return lower_adjacent_value, upper_adjacent_value
    
    
    def load_model(self, model_path):
        saved = load(model_path + '.joblib')
        for k, v in saved.items():
            setattr(self, k, v)
        
    
    def save_model(self, model_path):
        dump(vars(self), model_path + '.joblib')
        
        
    def get_columns(self, data, columns=['elf'], idx=None):
        try: len(idx)
        except: return np.hstack([np.stack(data[i].values).reshape(len(data),-1) for i in columns])
        else: return np.hstack([np.stack(data.iloc[idx][i].values).reshape(len(idx),-1) for i in columns])

    
    def get_inputs(self, data):
        data['inputs'] = data[self.inputs].apply(lambda x:
            np.concatenate([[x[k]] if isinstance(x[k], (float, int)) else x[k] for k in self.inputs]), axis=1)
        return data
    
            
    def set_inputs(self, inputs):
        self.inputs = inputs
        
    
    def split(self, data, test_size=0.1, stratify=True, seed=None):
        idx_train, idx_test = train_test_split(range(len(data)), test_size=test_size,
            random_state=seed if seed else self.seed,
            stratify=self.bond_to_float(data['label'].tolist()) if stratify else None)
        return idx_train, idx_test

    
    def split_kfold(self, n_folds, test_size, seed=None):
        return StratifiedShuffleSplit(n_splits=n_folds, test_size=test_size,
                                      random_state=seed if seed else self.seed)
    

    def pca_fit(self, data, n_components, column='elf'):
        if not isinstance(column, list):
            column = [column]
        self.pca = []
        self.z_fit = []
        for _column in column:
            self.pca.append(PCA(n_components=n_components))
            self.pca[-1].column = _column
            data['z_' + _column] = list(self.pca[-1].fit_transform(self.get_columns(data, columns=[_column])))
            self.z_fit.append(np.stack(data['z_' + _column].values))
        return data
        
    
    def pca_transform(self, data):
        self.z = []
        for i in range(len(self.pca)):
            data['z_' + self.pca[i].column] = list(self.pca[i].transform(
                self.get_columns(data, columns=[self.pca[i].column])))
            self.z.append(np.stack(data['z_' + self.pca[i].column].values))
        return data
    
    
    def scaler_fit(self, data):
        self.scaler = StandardScaler()
        self.scaler.fit(np.stack(self.get_inputs(data)['inputs'].values))
        
        
    def scaler_transform(self, data):
        x = self.get_columns(data, ['inputs'])
        data['inputs_scaled'] = self.scaler.transform(x).tolist()
        return data
        
    
    def prepare_inputs(self, data):
        data = self.pca_transform(data)
        data = self.get_inputs(data)
        data = self.scaler_transform(data)
        return data
    
        
    def clf_init(self, n_estimators, max_depth, max_samples, seed=None):
        self.clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, max_features='sqrt',
                                          max_samples=max_samples, bootstrap=True, oob_score=True,
                                          class_weight='balanced', random_state=seed if seed else self.seed)
        
    
    def clf_predict(self, data):
        columns = [k.column for k in self.pca]
        if len(columns[0].split('_')) > 1:
            tag = '_' + columns[0].split('_')[-1]
        else:
            tag = ''
        column = '_'.join([''.join(k.split('_')[:-1]) for k in columns]) + tag
        x = self.get_columns(data, ['inputs_scaled'])
        
        preds = self.clf.predict(x)
        data[column + '_pred'] = preds.tolist()
        
        _preds = np.zeros((self.clf.n_estimators, len(x)), dtype=int)
        for j in range(self.clf.n_estimators):
            _preds[j] = self.clf.estimators_[j].predict(x)
        _probas = np.count_nonzero(_preds == preds, axis=0)/self.clf.n_estimators
        data[column + '_pred_proba'] = _probas.tolist()
        
        return data


    def plot_evr(self, target=None, index=0):
        evr = self.pca[index].explained_variance_ratio_.cumsum()

        fig, ax = plt.subplots(figsize=(3.5,3))
        ax.plot(range(1, self.pca[index].n_components + 1), evr, color=self.palette[0])
        
        if target:
            n_evr = 1 + np.argmin(np.abs(evr - target))
            ax.axvline(n_evr, color='black', ls='dashed')
            print('Number of components:', n_evr)
            
        ax.set_xlabel('Number of components')
        ax.set_ylabel('Explained variance fraction')
        return fig
        
    
    def plot_projection(self, axes=[0,1], x=None, y=None, cmap=None, order=None, index=0, ec='black', bkg=True):
        try: len(x)
        except: x = self.z[index]

        try: len(order)
        except: order = np.arange(len(x))

        e1, e2 = axes
        try: len(y)
        except: colors = [self.palette[0] for k in range(len(x))]
        else:
            norm = plt.Normalize(vmin=y.min(), vmax=y.max())
            cmap = cmap if cmap else self.cmap
            colors = cmap(norm(y[order]))

        fig, ax = plt.subplots(figsize=(5,5))
        if bkg and hasattr(self, 'z_fit'):
            ax.scatter(self.z_fit[index][:,e1], self.z_fit[index][:,e2], color=self.bkg)
        ax.scatter(x[order,e1], x[order,e2], ec=ec, color=colors)
        ax.axis('off')
        return fig
    
    
    def plot_projection_slices(self, x, y, axes=[0,1], cmap=None, order=False, index=0):
        e1, e2 = axes
        norm = plt.Normalize(vmin=min([k.min() for k in y]), vmax=max([k.max() for k in y]))
        cmap = cmap if cmap else self.cmap

        n = int(np.ceil(np.sqrt(len(x))))
        fig, ax = plt.subplots(n,n, figsize=(3*n,3*n))
        ax = ax.ravel()
        for i in range(len(x)):
            if hasattr(self, 'z_fit'):
                ax[i].scatter(self.z_fit[index][:,e1], self.z_fit[index][:,e2], color=self.bkg, s=24)
            if order: idx = np.argsort(y[i])
            else: idx = np.arange(len(y[i]))

            colors = cmap(norm(y[i][idx]))
            ax[i].scatter(x[i][idx,e1], x[i][idx,e2], ec='black', color=colors, s=24)
            ax[i].axis('off')

        for j in range(i+1,n*n):
            ax[j].remove()

        return fig


    def plot_profiles(self, n=15, axes=[0,1], y=None, cmap=None, transform=None, index=0):
        e1, e2 = axes
        
        points = self.z[index][:,np.array([e1,e2])]
        hull = Delaunay(points)

        v1 = np.linspace(hull.min_bound[0], hull.max_bound[0], n)
        v2 = np.linspace(hull.min_bound[1], hull.max_bound[1], n)[::-1]
        p1, p2 = np.meshgrid(v1, v2)
        
        zz = self.z[index].mean(axis=0, keepdims=True)*np.ones((n**2, self.z[index].shape[-1]))
        zz[:,e1] = p1.ravel()
        zz[:,e2] = p2.ravel()

        if transform == None:
            x = self.pca[index].inverse_transform(zz)
        else:
            _x = self.pca[index].inverse_transform(zz)
            x = np.zeros_like(_x)
            l = range(len(_x[0,:]))
            for i in range(len(x)):
                x[i,:] = transform(l, _x[i,:])
        x_min, x_max = x.min(), x.max()
        
        try: len(y)
        except:
            colors = [self.palette[0] for k in range(n*n)]
        else:
            v = griddata(points, y, zz[:,np.array([e1,e2])], method='nearest')
            norm = plt.Normalize(vmin=y.min(), vmax=y.max())
            cmap = cmap if cmap else self.cmap
            colors = cmap(norm(v))

        fig, ax = plt.subplots(n, n, figsize=(5,5))
        fig.subplots_adjust(wspace=0.2, hspace=-0.2)
        ax = ax.ravel()
        for i in range(len(ax)):
            ax[i].axis('off')
            if hull.find_simplex(zz[i,np.array([e1,e2])]) >= 0:
                ax[i].plot(x[i,:], color=colors[i], lw=1.5)
                ax[i].set_xticklabels([])
                ax[i].set_yticklabels([])
                ax[i].set_xticks([])
                ax[i].set_yticks([])
                ax[i].set_ylim([x_min - 0.005, x_max + 0.005])
            else:
                ax[i].remove()
        return fig
    
    
    def plot_scores(self, scores, columns, features, n_estimators, max_depth, y_min=None, y_shift=0.07):
        fig, ax = plt.subplots(len(columns), len(features), figsize=(2.5*len(features), 2.5*len(columns)),
                           sharex=True, sharey=True)
        fig.subplots_adjust(wspace=0.05, hspace=0.05)
        ax = ax.ravel()
        all_columns = set(['elf_srt', 'pdf_srt', 'pdf-l_srt', 'cdf_srt', 'cdf-A_srt'])

        y_max = scores.max()
        k = 0
        for i in range(len(columns)):
            if len(columns[i]) == 5:
                ax[k].text(0.1, 0.85, 'All', ha='left', va='top', transform=ax[k].transAxes)
            elif len(columns[i]) == 4:
                label = list(all_columns.difference(set(columns[i])))[0]
                ax[k].text(0.1, 0.85, 'All - ' + label.split('_')[0].upper(), ha='left', va='top', transform=ax[k].transAxes)
            else:
                ax[k].text(0.1, 0.8, ' + '.join([k.split('_')[0].upper() for k in columns[i]]),
                           ha='left', va='top', transform=ax[k].transAxes)
            for j in range(len(features)):
                ax[k].set_prop_cycle('color', list(self.cmap(np.linspace(0,0.75,len(max_depth)))))
                ax[k].plot(n_estimators, scores[i,j], label=max_depth)
                ax[k].axhline(1., color='black', ls='--')
                ax[k].axhline(y_max, color=self.palette[1], ls=':')
                ax[k].locator_params('x', nbins=4)
                ax[k].tick_params(direction='in')
                if y_min:
                    ax[k].set_ylim(bottom=y_min)
                if i == 0:
                    tag = 'Baseline'
                    if len(features[j]):
                        tag += ' + '
                    ax[k].set_title(tag + ' + '.join([k.upper() for k in features[j]]), fontsize=plt.rcParams['font.size'])
                k += 1

        fig.supxlabel('Estimators', fontsize=plt.rcParams['font.size'], y=y_shift)
        fig.supylabel('Score', fontsize=plt.rcParams['font.size'], x=0.05)
        ax[-1].legend(frameon=False, ncol=2, title='Max. depth', loc='lower center')
        return fig
        
        
    def plot_violins(self, scores, columns):
        scores_sorted = [sorted(scores[i]) for i in range(len(columns))]
        fig, ax = plt.subplots(figsize=(len(columns),2))
        violin = ax.violinplot(scores_sorted, showextrema=False);
        for v in violin['bodies']:
            v.set_facecolor(self.palette[0])
            v.set_edgecolor(self.palette[0])
            v.set_alpha(0.5)

        q1, medians, q3 = np.percentile(scores_sorted, [25, 50, 75], axis=1)
        whiskers = np.array([self._adjacent_values(s, i, j) for s, i, j in zip(scores_sorted, q1, q3)])
        whiskers_min, whiskers_max = whiskers[:,0], whiskers[:,1]

        ax.scatter(range(1,len(columns)+1), medians, marker='o', fc='white', ec='black', s=50, zorder=3)
        ax.vlines(range(1,len(columns)+1), q1, q3, color='k', linestyle='-', lw=5)
        ax.vlines(range(1,len(columns)+1), whiskers_min, whiskers_max, color='k', linestyle='-', lw=1)
        
        all_columns = set(['elf_srt', 'pdf_srt', 'pdf-l_srt', 'cdf_srt', 'cdf-A_srt'])
        ax.set_xticks(range(1,len(columns)+1))
        if len(columns[0]) == 5:
            ax.set_xticklabels(['All'])
        elif len(columns[0]) == 4:
            xticklabels = ['All - ' + list(all_columns.difference(set(i)))[0].split('_')[0].upper() for i in columns]
            ax.set_xticklabels(xticklabels)
        else:
            ax.set_xticklabels([' + '.join([k.split('_')[0].upper() for k in i]) for i in columns])
        ax.set_ylim(top=1.)
        ax.set_xlabel('Descriptor(s)')
        ax.set_ylabel('Score')
        return fig


    def plot_importances(self, importances, columns, features, vmin=None, vmax=None, cmap=cm.oslo, tag=''):
        if len(tag):
            label = tag.capitalize() + ' importance'
        else:
            label = 'Importance'
        all_columns = set(['elf_srt', 'pdf_srt', 'pdf-l_srt', 'cdf_srt', 'cdf-A_srt'])
        
        n_components = importances.shape[-1] - len(features)
        components = ['$z_{' + str(k + 1) + '}$' for k in range(n_components)]
        fig, ax = plt.subplots(figsize=(importances.shape[-1]/2.,2))
        sm = ax.imshow(importances, vmin=vmin, vmax=vmax, cmap=cmap)
        ax.set_xticks(range(importances.shape[-1]))
        ax.set_xticklabels(components + ['+'.join(i.capitalize()) for i in features])
        ax.set_yticks(range(len(columns)))
        if len(columns[0]) == 5:
            ax.set_yticklabels(['All'])
        elif len(columns[0]) == 4:
            yticklabels = ['All - ' + list(all_columns.difference(set(i)))[0].split('_')[0].upper() for i in columns]
            ax.set_yticklabels(yticklabels)    
        else:
            ax.set_yticklabels([' + '.join([k.split('_')[0].upper() for k in i]) for i in columns])
        plt.colorbar(sm, aspect=16, label=label)
        return fig
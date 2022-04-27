import numpy as np
import pandas as pd
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx

from ase.io import read as read_ase
from ase.data import covalent_radii
from ase.visualize.plot import plot_atoms
from scipy.interpolate import griddata
from scipy.spatial import Delaunay
from utils.plot import palette, cmap, cmap_mono, cmap_disc, norm, fontsize, textsize


bonds = {'covalent': 0, 'metallic': 1, 'ionic': 2, 'vdw': 3}


def read_label(x, dirname):
    # read label data
    with open(dirname + x + '/label.txt', 'r') as f:
        label = f.read().splitlines()[0]
    return label


def read_elf(x, dirname):
    # read elf profile data
    keys = ['elf', 'en_diff', 'edge_src', 'edge_dst', 'bond_length']
    data = dict(zip(keys, [[] for k in range(len(keys))]))
    with open(dirname + x + '/elfProfs.csv', 'r') as f:
        for line in f.readlines():
            entry = line.split(',')[:-1]
            data['elf'].append([float(k) for k in entry[:41]])
            data['en_diff'].append(float(entry[41]))
            data['edge_src'].append(int(entry[42]))
            data['edge_dst'].append(int(entry[43]))
            data['bond_length'].append(float(entry[44]))
    return data


def load_data(dirname, structure=True, labeled=False, sort=False):
    # load data from directory
    materials = next(os.walk(dirname))[1]
    data = pd.DataFrame({'formula': materials})
    if structure:
        # read structure
        data['structure'] = data['formula'].apply(lambda x: read_ase(dirname + x + '/POSCAR'))
    if labeled:
        # read label
        data['label'] = data['formula'].apply(lambda x: read_label(x, dirname))
    data['data'] = data['formula'].apply(lambda x: read_elf(x, dirname))
    data = pd.concat([data.drop(['data'], axis=1), data['data'].apply(pd.Series)], axis=1)
    
    if sort:
        # sort ELF profiles so larger peaks come first
        data = sort_elf(data)
    return data


def bond_to_float(x):
    return bonds[x]


def second_moment(x, l):
    r = np.linspace(-l/2., l/2., len(x))
    return (x*r**2).sum()


def calculate_moments(data):
    data['I'] = data[['elf', 'bond_length']].apply(
        lambda x: [second_moment(k,l) for (k,l) in zip(x.elf, x.bond_length)], axis=1)
    data['A'] = data[['elf', 'bond_length']].apply(
        lambda x: [np.trapz(k, np.linspace(0.,l,len(k))) for (k,l) in zip(x.elf, x.bond_length)], axis=1)
    return data


def sort_elf(data):
    # compute pdf and cdf
    data['elf_pdf'] = data[['elf', 'bond_length']].apply(
        lambda x: [np.array(k)/(np.sum(k)*l/len(k)) for (k,l) in zip(x.elf, x.bond_length)], axis=1)

    data['elf_cdf'] = data[['elf_pdf', 'bond_length']].apply(
        lambda x: [l/len(k)*np.cumsum(k) for (k,l) in zip(x.elf_pdf, x.bond_length)], axis=1)

    data['elf_cdf_r'] = data[['elf_pdf', 'bond_length']].apply(
        lambda x: [l/len(k)*np.cumsum(k[::-1]) for (k,l) in zip(x.elf_pdf, x.bond_length)], axis=1)

    data['v'] = data[['elf_cdf', 'elf_cdf_r']].apply(
        lambda x: [2*(c1.sum() >= c2.sum()) - 1 for (c1,c2) in zip(x.elf_cdf, x.elf_cdf_r)], axis=1)
    
    data['elf_orig'] = data['elf'].copy()
    data['elf'] = data[['elf_orig', 'v']].apply(lambda x: [k[::l] for (k,l) in zip(x.elf_orig, x.v)], axis=1)
    return data


def plot_cevr(evr, save_path=None):
    # plot cumulative explained variance as a funciton of the number of PCs
    fig, ax = plt.subplots(figsize=(4,3.5))
    evr = np.cumsum(evr)
    nc = np.argmin(np.abs(evr - 0.99)) + 1
    ax.plot(range(1,len(evr)+1), evr, color='black');
    ax.axvline(nc, ls='dashed', color='dimgray')
    ax.set_xlabel('Number of PCs')
    ax.set_ylabel('Explained variance')
    print('EVR of 0.99 at', nc, 'components')
    if save_path:
        fig.savefig(save_path + '.png', bbox_inches='tight', dpi=200)
        
        
def plot_pca(z, data, bonds, axes=[0,1,2], save_path=None):
    # plot projections of three PCs colored by descriptors
    e1, e2, e3 = axes
    labeled = 'label' in data.columns
    if labeled:
        y1 = data[['bond_length', 'label']].explode('bond_length')['label'].apply(bond_to_float).tolist()
        y1_label = 'Bond type'
    else:
        y1 = data['A'].sum()
        y1_label = 'Integ. ELF area'
        
    y2 = data['bond_length'].sum()
    y3 = data['I'].sum()
    y4 = data['en_diff'].sum()
    L = len(bonds)
    
    fig, ax = plt.subplots(4,4, figsize=(15,12), gridspec_kw={'height_ratios': [0.07,1,1,1]})
    for i, y in enumerate([y1, y2, y3, y4]):
        if (i == 0) & labeled:
            g = ax[1,i].scatter(z[:,e1], z[:,e2], c=y, s=40, lw=1.5, cmap=cmap_disc)
            plt.colorbar(g, cax=ax[0,i], orientation='horizontal',
                         ticks=[(i-0.5)*(L-1)/L for i in range(1, len(palette) + 1)])
            ax[1,i].clear()

            ax[1,i].scatter(z[:,e1], z[:,e2], c=y, s=40, alpha=0.6, lw=1.5, cmap=cmap_disc)
            ax[2,i].scatter(z[:,e2], z[:,e3], c=y, s=40, alpha=0.6, lw=1.5, cmap=cmap_disc)
            ax[3,i].scatter(z[:,e1], z[:,e3], c=y, s=40, alpha=0.6, lw=1.5, cmap=cmap_disc)

        else:
            g = ax[1,i].scatter(z[:,e1], z[:,e2], c=y, s=40, lw=1.5, cmap=cmap)
            plt.colorbar(g, cax=ax[0,i], orientation='horizontal')
            ax[1,i].clear()

            ax[1,i].scatter(z[:,e1], z[:,e2], c=y, s=40, alpha=0.6, lw=1.5, cmap=cmap)
            ax[2,i].scatter(z[:,e2], z[:,e3], c=y, s=40, alpha=0.6, lw=1.5, cmap=cmap)
            ax[3,i].scatter(z[:,e1], z[:,e3], c=y, s=40, alpha=0.6, lw=1.5, cmap=cmap)
            ax[0,i].locator_params(axis='x', nbins=5)

        ax[1,i].set_xlabel('$PC_' + str(e1+1) + '$')
        ax[2,i].set_xlabel('$PC_' + str(e2+1) + '$')
        ax[3,i].set_xlabel('$PC_' + str(e1+1) + '$')

    ax[1,0].set_ylabel('$PC_' + str(e2+1) + '$')
    ax[2,0].set_ylabel('$PC_' + str(e3+1) + '$')
    ax[3,0].set_ylabel('$PC_' + str(e3+1) + '$')

    for k in ax[1:,:].ravel():
        k.set_xticks([])
        k.set_yticks([])
    
    if labeled:
        ax[0,0].set_xticklabels([k.capitalize() for k in bonds.keys()], fontsize=fontsize-2)
    ax[0,0].set_title(y1_label, fontsize=fontsize)
    ax[0,1].set_title(r'$Bond\ length\ (\AA)$', fontsize=fontsize)
    ax[0,2].set_title(r'$ELF\ second\ moment\ (\AA^2)$', fontsize=fontsize)
    ax[0,3].set_title(r'$Electroneg.\ difference$', fontsize=fontsize)

    fig.subplots_adjust(hspace=0.2, wspace=0.1)
    
    if save_path:
        fig.savefig(save_path + '.png', bbox_inches='tight', dpi=200)
        

def plot_profiles(z, data, pca, colorby='I', axes=[0,1], save_path=None):
    # plot reconstructed profiles along chosen principal components
    e1, e2 = axes
    c = 11
    points = z[:,np.array([e1,e2])]
    hull = Delaunay(points)
    
    try: data[colorby]
    except:
        print('Descriptor not in data.')
    else:
        if colorby == 'label':
            y = data[['bond_length', 'label']].explode('bond_length')['label'].apply(bond_to_float).tolist()
            normy = norm
        else:
            y = data[colorby].sum()
            normy = plt.Normalize(vmin=min(y), vmax=max(y))
            
        v1 = np.linspace(hull.min_bound[0], hull.max_bound[0], c)
        v2 = np.linspace(hull.min_bound[1], hull.max_bound[1], c)[::-1]
        p1, p2 = np.meshgrid(v1, v2)
        zz = np.zeros((c**2, pca.n_components_))
        zz[:,e1] = p1.ravel()
        zz[:,e2] = p2.ravel()
        x = pca.inverse_transform(zz)
        x_min, x_max = x.min(), x.max()
        color = griddata(points, y, zz[:,np.array([e1,e2])], method='nearest')

        fig, ax = plt.subplots(c,c, figsize=(10,10))
        ax = ax.ravel()
        for i in range(len(ax)):
            ax[i].axis('off')
            if hull.find_simplex(zz[i,np.array([e1,e2])])>=0:
                ax[i].plot(x[i,:], color=cmap(normy(color[i])), lw=1.5)
                ax[i].set_xticklabels([]); ax[i].set_yticklabels([])
                ax[i].set_xticks([]); ax[i].set_yticks([])
                ax[i].set_ylim([x_min, x_max])
            else:
                if i == (c*c - c):
                    ax[i].arrow(0, 0, x_max, 0, width=0.1, color='black')
                    ax[i].arrow(0, 0, 0, x_max, width=0.1, color='black')
                    ax[i].text(1.5*x_max, 0, '$PC_' + str(e1+1) + '$', ha='left', va='center', color='black', fontsize=textsize)
                    ax[i].text(0, 1.5*x_max, '$PC_' + str(e2+1) + '$', ha='center', va='bottom', color='black', fontsize=textsize)
                else:
                    ax[i].remove()

        if save_path:
            fig.savefig(save_path + '.png', bbox_inches='tight', dpi=200)
            

def plot_crystal_graph(data, index=0, save_path=None):
    # extract example and compute graph properties
    entry = data.iloc[index]
    nodes = np.arange(len(entry.structure))
    node_size = 400*covalent_radii[entry.structure.get_atomic_numbers()]
    node_attr = dict(zip(nodes, entry.structure.get_chemical_symbols()))
    edge_attr = dict(zip(list(zip(entry.edge_src, entry.edge_dst)), entry.y_class))
    pos = dict(zip(nodes, [k[:-1] for k in entry.structure.get_positions()]))

    # construct graph
    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(list(zip(entry.edge_src, entry.edge_dst)))

    # plot graph
    fig, ax = plt.subplots(2,1, figsize=(4.5,5), gridspec_kw={'height_ratios': [0.07,1]})
    nx.draw_networkx(G, labels=node_attr, pos=pos, font_family='Myriad Pro', font_color='white',
                     node_size=node_size, node_color='slategray', edge_color='none', ax=ax[1])
    nx.draw_networkx_edges(G, pos=pos, edge_color=cmap_disc(norm(entry.y_class)), alpha=0.9, width=3, ax=ax[1])
    pad = np.array([-0.5, 0.5])
    ax[1].set_aspect('equal')
    ax[1].axis('off')
    ax[1].set_xlim(np.array(ax[1].get_xlim()) + pad)
    ax[1].set_ylim(np.array(ax[1].get_ylim()) + pad);

    # plot colorbar
    L = len(bonds)
    sm = mpl.cm.ScalarMappable(cmap=cmap_disc, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, cax=ax[0], orientation='horizontal', ticks=[(i-0.5)*(L-1)/L for i in range(1, len(palette) + 1)])
    ax[0].set_xticklabels([k.capitalize() for k in bonds.keys()], fontsize=fontsize-2)
    
    if save_path:
        fig.savefig(save_path + '.png', bbox_inches='tight', dpi=200)
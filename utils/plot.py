import matplotlib as mpl
import matplotlib.pyplot as plt

# plot settings
plt.rcParams['font.family'] = 'Myriad Pro'
plt.rcParams['axes.linewidth'] = 1
plt.rcParams['mathtext.default'] = 'regular'
plt.rcParams['xtick.bottom'] = True
plt.rcParams['ytick.left'] = True
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'

fontsize = 18
textsize = 16
plt.rcParams['font.size'] = fontsize
plt.rcParams['axes.labelsize'] = fontsize
plt.rcParams['xtick.labelsize'] = fontsize
plt.rcParams['ytick.labelsize'] = fontsize
plt.rcParams['legend.fontsize'] = textsize
plt.rcParams['legend.title_fontsize'] = textsize

# colors
palette = ['#1F608B', '#D7DADA', '#E6A355', '#C76048']
palette_grad = ['#164061', '#3D5C92', '#A45B85', '#C95B38', '#BD7D39']
cmap = mpl.colors.LinearSegmentedColormap.from_list('BuGrYR', colors=palette, N=100)
cmap_mono = mpl.colors.LinearSegmentedColormap.from_list('WGrBu', colors=['white', '#D7DADA', '#1F608B'], N=100)
cmap_disc = mpl.colors.ListedColormap(palette)
cmap_grad = mpl.colors.LinearSegmentedColormap.from_list('BuPuY', colors=palette_grad, N=100)
norm = plt.Normalize(vmin=0, vmax=3)
import graphviz
import numpy as np
from itertools import product

dot = graphviz.Digraph('NAM_2', comment='NAM_2')

bug_ls = []
for microbe in np.arange(3):
    dot.node( 'bc' + str(microbe), 'Microbe\nCluster ' + str(microbe))
    bug_ls.append('bc' + str(microbe))

met_ls = []
# for metabolite in np.arange(6):
metabolite = 2
dot.node('mc' + str(metabolite), 'Metabolite\nCluster ' + str(metabolite))
met_node = 'mc' + str(metabolite)

for bug_node in bug_ls:
    h1_ls = []
    for hidden_layer_1 in np.arange(5):
        dot.node('bc' + str(bug_node) + '-h1' + str(hidden_layer_1), '')
        dot.edge(bug_node, 'bc' + str(bug_node) + '-h1' + str(hidden_layer_1))
        h1_ls.append('bc' + str(bug_node) + '-h1' + str(hidden_layer_1))
    for hidden_layer_2 in np.arange(7):
        dot.node('bc' + str(bug_node) + '-h2' + str(hidden_layer_2), '')
        for h1_node in h1_ls:
            dot.edge(h1_node, 'bc' + str(bug_node) + '-h2' + str(hidden_layer_2))
        dot.edge('bc' + str(bug_node) + '-h2' + str(hidden_layer_2), met_node)

dot.render(directory='/Users/jendawk/Dropbox (MIT)/M2M/figures').replace('\\', '/')

dot = graphviz.Digraph('NN_sm', comment='NN_sm')

bug_ls = []
for microbe in np.arange(3):
    dot.node( 'bc' + str(microbe), 'Microbe\nCluster ' + str(microbe))
    bug_ls.append('bc' + str(microbe))

hl1_ls = []
for hidden_layer_1 in np.arange(5):
    dot.node('h1' + str(hidden_layer_1), '')
    hl1_ls.append('h1' + str(hidden_layer_1))

for bug_node, hidden_node in product(bug_ls, hl1_ls):
    dot.edge(bug_node, hidden_node)

hl2_ls = []
for hidden_layer_2 in np.arange(7):
    dot.node('h2' + str(hidden_layer_2), '')
    hl2_ls.append('h2' + str(hidden_layer_2))

for hidden1, hidden2 in product(hl1_ls, hl2_ls):
    dot.edge(hidden1, hidden2)

met_ls = []
for metabolite in np.arange(3):
    dot.node('mc' + str(metabolite), 'Metabolite\nCluster ' + str(metabolite))
    met_ls.append('mc' + str(metabolite))

for hidden2, met_node in product(hl2_ls, met_ls):
    dot.edge(hidden2, met_node)

dot.render(directory='/Users/jendawk/Dropbox (MIT)/M2M/figures').replace('\\', '/')
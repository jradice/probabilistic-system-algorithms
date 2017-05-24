"""Testing pbnt. Run this before anything else to get pbnt to work!"""
import sys
if('pbnt/combined' not in sys.path):
    sys.path.append('pbnt/combined')
from exampleinference import inferenceExample
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)
from Inference import JunctionTreeEngine
from Inference import EnumerationEngine
from numpy import zeros, float32
import Distribution
from Distribution import DiscreteDistribution, ConditionalDiscreteDistribution
from Node import BayesNode
from Graph import BayesNet

def make_net():
    nodes = []
    R_node = BayesNode(0, 2, name="rain")
    W_node = BayesNode(1, 2, name="wet")
    S_node = BayesNode(2, 2, name="sick")
    
    R_node.add_child(W_node)
    
    W_node.add_parent(R_node)
    W_node.add_child(S_node)
    
    S_node.add_parent(R_node)
    
    nodes = [R_node, W_node, S_node]
    
    return BayesNet(nodes)

def set_distribution(bayes_net):
    R_node = bayes_net.get_node_by_name("rain")
    W_node = bayes_net.get_node_by_name("wet")
    S_node = bayes_net.get_node_by_name("sick")
    
    nodes = [R_node, W_node, S_node]
    
    # P(R)
    R_dist = DiscreteDistribution(R_node)
    index = R_dist.generate_index([], [])
    R_dist[index] = [0.8, 0.2]
    R_node.set_dist(R_dist)
    
    # P(W|R)
    # R | P(W|R)
    # F | 0.0
    # T | 0.25
    dist = zeros([R_node.size(), W_node.size()], dtype=float32)
    dist[0,:] = [1.0, 0.0]
    dist[1,:] = [0.75, 0.25]
    W_dist = ConditionalDiscreteDistribution(nodes=[R_node, W_node], table=dist)
    W_node.set_dist(W_dist)
    
    # P(S|R,W)
    # R | W | P(S|R,W)
    # F | F | 0.1 
    # F | T | 0.2
    # T | F | 0.33
    # T | T | 0.6
    dist = zeros([R_node.size(), W_node.size(), S_node.size()], dtype=float32)
    dist[0,0,:] = [0.9, 0.1]
    dist[0,1,:] = [0.8, 0.2]
    dist[1,0,:] = [0.67, 0.33]
    dist[1,1,:] = [0.4, 0.6]
    S_dist = ConditionalDiscreteDistribution(nodes=[R_node, W_node, S_node], table=dist)
    
    S_node.set_dist(S_dist)
    
    return bayes_net

def get_prob_s_given_w(bayes_net, sick, wet):
    S_node = bayes_net.get_node_by_name('sick')
    W_node = bayes_net.get_node_by_name('wet')
    engine = EnumerationEngine(bayes_net)
    engine.evidence[W_node] = wet
    Q = engine.marginal(S_node)[0]
    #index = Q.generate_index([sick], range(Q.nDims))
    return Q.table

def get_prob_s_given_w_r(bayes_net, sick, wet, rain):
    S_node = bayes_net.get_node_by_name('sick')
    W_node = bayes_net.get_node_by_name('wet')
    R_node = bayes_net.get_node_by_name('rain')
    engine = EnumerationEngine(bayes_net)
    engine.evidence[W_node] = wet
    engine.evidence[R_node] = rain
    Q = engine.marginal(S_node)[0]
    #index = Q.generate_index([sick], range(Q.nDims))
    return Q.table
    
test_net = make_net()
set_distribution(test_net)
print get_prob_s_given_w(test_net, True, True)
print get_prob_s_given_w_r(test_net, True, True, True)
print get_prob_s_given_w_r(test_net, True, True, False)


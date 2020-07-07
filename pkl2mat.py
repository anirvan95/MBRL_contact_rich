import pickle
import scipy.io
name = 'data/ioc_data_exp_7b'
p = open('%s.pkl'%name,'rb')
data = pickle.load(p)
dict = {}
dict['data'] = data
scipy.io.savemat('%s.mat'%name, dict)
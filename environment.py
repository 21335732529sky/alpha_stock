import pickle
from tqdm import tqdm
import random
import numpy as np
import torch
import os
import pandas as pd

class FXEnv(object):
	def __init__(self, instruments=[], holding_period=5, trading_period=60, look_back=12, test_size=0.2):
		assert len(instruments) > 0, "At least one of instruments must be given."
		assert holding_period < trading_period, "holding_period must be lesser than trading_period"
		self.data = []
		if not os.path.exists('datasets.pkl'):
			self.data, self.m_mean, self.m_std = self.read_data(instruments, holding_period)
			pickle.dump([self.data, self.m_mean, self.m_std], open('datasets.pkl', 'wb'))
		else:
			self.data, self.m_mean, self.m_std = pickle.load(open('datasets.pkl', 'rb'))
		# print(df_base.iloc[[False if d in all_null else True for d in df_base.index], :].shape)

		self.data = np.array(self.data)


		self.ins_data = instruments
		self.trading_period = trading_period
		self.holding_period = holding_period
		self.look_back = look_back
		self.split_point = int(len(self.data[0]) * (1 - test_size))

	def reset(self, mode='train', TC=1e-7):
		assert mode in ['train', 'test', 'test_private']
		if mode == 'train':
			self.t = random.randint(self.holding_period * self.look_back, self.split_point - self.trading_period)
			self.end_period = self.t + self.trading_period
		elif mode == 'test':
			self.t = self.split_point
			self.end_period = self.t + 450*self.holding_period
		elif mode == 'test_private':
			self.t = self.split_point
			self.end_period = len(self.data[0]) - self.holding_period
		self.current_ptf = torch.zeros(len(self.ins_data))
		self.TC = TC

	def get_state(self):
		assert self.t > self.holding_period * self.look_back
		tmp = self.data[:, self.t - (self.holding_period * self.look_back):self.t, :]
		tmp = tmp.reshape((tmp.shape[0], self.look_back, self.holding_period, tmp.shape[-1]))
		price = (tmp * self.m_std[0] + self.m_mean[0]).prod(axis=2)[:, :, 0:1]
		price = (price - self.m_mean[0]) / self.m_std[0]
		vol = tmp[:, :, -1, 1:2]
		tv = tmp[:, :, -1, 2:]
		return np.concatenate([price, vol, tv], axis=-1)

	def _return_rate_in_hp(self):
		assert self.t > self.holding_period - 1
		ret = np.array([1.0,]*len(self.ins_data))
		for i in range(self.t - self.holding_period + 1, self.t):
			ret *= self.data[:, i, 0] * self.m_std[0] + self.m_mean[0]
		return ret

	def is_end(self):
		return self.t >= self.end_period

	def step(self, portfolio):
		assert len(portfolio) == len(self.data)
		diff = sum(abs(portfolio.detach().cpu() - self.current_ptf)).numpy()
		self.current_ptf = portfolio.detach().cpu()
		self.t += self.holding_period

		z = self._return_rate_in_hp()
		Rt = (torch.tensor(z).type(torch.FloatTensor).cuda() * portfolio).sum()

		return self.get_state(), Rt, self.TC, self.is_end()

	def read_data(self, instruments, holding_period, ):
		ret = []
		ret_ratio = []
		now = '2020-03-10T09:00:00.000000000Z'
		all_null = set(pd.date_range(end=now, periods=80000, freq='H'))

		for name in instruments:
			pkl = pickle.load(open('{}.pkl'.format(name), 'rb'))[::-1]
			pkl = [(p[0], float(p[1]), p[2]) for p in pkl]
			row_data = []
			dindex = []
			print('processing {} ...'.format(name))
			for i in tqdm(range(holding_period - 1, len(pkl))):
				mean = sum([p[1] for p in pkl[i-holding_period+1:i+1]]) / holding_period
				vol = sum((p[1] - mean)**2 for p in pkl[i-holding_period+1:i+1]) / holding_period
				r = pkl[i][1] / pkl[i - 1][1]
				tv = sum(p[2] for p in pkl[i-holding_period+1:i+1])
				row_data.append((r, vol, tv))
				dindex.append(pkl[i][0])

			df = pd.DataFrame(row_data, index=dindex)
			df_base = pd.DataFrame([], index=pd.date_range(end=now, periods=80000, freq='H'))
			df_base = df_base.join(df)
			is_null = df_base.isnull().prod(axis=1)
			null_index = set([idx for idx, f in zip(df_base.index, is_null) if f])
			all_null = all_null & null_index
			ret.append(df_base)

		for i in range(len(ret)):
			tmp_df = ret[i]
			tmp_df = tmp_df.iloc[[False if d in all_null else True for d in df_base.index], :]
			tmp_df.iloc[:, 0] = tmp_df.iloc[:, 0].fillna(1.0)
			tmp_df = tmp_df.fillna(method='ffill').fillna(method='bfill')
			ret_ratio.append(tmp_df.values[:, 0])
			ret[i] = ((tmp_df - tmp_df.mean(axis=0)) / tmp_df.std(axis=0)).values

		return ret, tmp_df.mean(axis=0), tmp_df.std(axis=0)






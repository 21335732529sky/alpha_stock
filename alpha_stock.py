import torch
import torch.nn.functional as F
import numpy as np

class AlphaStock(torch.nn.Module):
	def __init__(self, num_stocks=2, num_features=2, rank_emb_dim=3, preset_size=1):
		super(AlphaStock, self).__init__()
		assert num_stocks // 2 >= preset_size
		self.lstm = torch.nn.LSTM(num_features, num_features, 1, batch_first=True)
		self.linear1 = torch.nn.Linear(num_features, num_features)
		self.linear2 = torch.nn.Linear(num_features, num_features)
		self.w_alpha = torch.nn.Linear(num_features, 1)
		# self.w_alpha = torch.Tensor(np.randn(num_stocks), requires_grad=True)
		self.linear_query = torch.nn.Linear(num_features, num_features)
		self.linear_key = torch.nn.Linear(num_features, num_features)
		self.linear_value = torch.nn.Linear(num_features, num_features)
		self.rank_embedding = torch.nn.Embedding(num_stocks, rank_emb_dim)
		self.linear_rank = torch.nn.Linear(rank_emb_dim, 1, bias=False)
		self.linear_winner = torch.nn.Linear(num_features, 1)
		self.preset_size = preset_size
		self.dropout = torch.nn.Dropout(0.1)
		self.instruments = ['USD_JPY', 'GBP_JPY', 'AUD_JPY', 'CAD_JPY', 'NZD_JPY', 'EUR_JPY', 'AUD_USD', 'EUR_USD', 'GBP_USD', 'NZD_USD',
			   'USD_CHF', 'EUR_CHF', 'GBP_CHF', 'EUR_GBP', 'AUD_NZD', 'AUD_CAD', 'AUD_CHF', 'CAD_CHF']


	def forward(self, state, stock_rank):
		state = torch.tensor(state).type(torch.FloatTensor).cuda()
		# print(state)
		S, L = state.shape[:2]
		stock_rank = torch.tensor(stock_rank).cuda()

		stock_rep, _ = self.lstm(state) # [S, L, H]
		stock_rep = self.dropout(stock_rep)
		alpha_rep1 = self.linear1(stock_rep) # [S, L, H]
		alpha_rep2 = self.linear2(stock_rep[:, -1, :]).unsqueeze(1) # [S, 1, H]

		# alpha = torch.sum(self.w_alpha * F.tanh(alpha_rep1 + alpha_rep2), dim=-1) # [S, L]
		alpha = self.w_alpha(F.tanh(alpha_rep1 + alpha_rep2)).squeeze(-1)
		alpha = F.softmax(alpha, dim=-1).unsqueeze(-1) # [S, L, 1]
		stock_rep = torch.sum(stock_rep * alpha, dim=1) # [S, H]

		query = self.linear_query(stock_rep) # [S, H]
		key = self.linear_key(stock_rep)
		value = self.linear_value(stock_rep)

		rank_ex_h = torch.cat([stock_rank.unsqueeze(1),] * stock_rank.shape[0], dim=1) # [S, S]
		rank_ex_v = torch.cat([stock_rank.unsqueeze(0),] * stock_rank.shape[0], dim=0) # [S, S]
		dis = abs(rank_ex_h - rank_ex_v)
		rank_emb = self.rank_embedding(dis) # [S, S, R]
		rank_emb = self.dropout(rank_emb)
		rank_emb = F.sigmoid(self.linear_rank(rank_emb)).squeeze() # [S, S]

		beta = torch.matmul(query, key.transpose(0, 1)) * rank_emb / torch.sqrt(torch.tensor(float(query.shape[-1]))) # [S, S]
		beta = F.softmax(beta, dim=-1).unsqueeze(-1) # [S, S, 1]

		stock_rep = torch.sum(value.unsqueeze(0) * beta, dim=1) # [S, H]

		winner_score = F.sigmoid(self.linear_winner(stock_rep).squeeze()) # [S]
		# print(winner_score)
		# winner_score = self.linear_winner(stock_rep).squeeze()
		rank = torch.argsort(winner_score)

		winners = set(rank.detach().cpu().numpy()[-self.preset_size:])
		losers = set(rank.detach().cpu().numpy()[:self.preset_size])

		winners_mask = torch.Tensor([0 if i in winners else 1 for i in range(rank.shape[0])]).cuda()
		loser_mask = torch.Tensor([0 if i in losers else 1 for i in range(rank.shape[0])]).cuda()

		winner_prop = F.softmax(winner_score - 1e9 * winners_mask, dim=0)
		loser_prop = F.softmax((1 - winner_score) - 1e9 * loser_mask, dim=0)

		portfolio = winner_prop - loser_prop

		chosen_ins = list(winners) + list(losers)

		return portfolio, chosen_ins

	def predict(self, query):
		portfolio = self.forward(query['state'], query['stock_rank'])
		return {ins: value for ins, value in zip(self.instruments, portfolio)}













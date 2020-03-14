from environment import *
from alpha_stock import *
import torch
import argparse
import matplotlib.pyplot as pl

instruments = ['USD_JPY', 'GBP_JPY', 'AUD_JPY', 'CAD_JPY', 'NZD_JPY', 'EUR_JPY', 'AUD_USD', 'EUR_USD', 'GBP_USD', 'NZD_USD',
			   'USD_CHF', 'EUR_CHF', 'GBP_CHF', 'EUR_GBP', 'AUD_NZD', 'AUD_CAD', 'AUD_CHF', 'CAD_CHF']

def evaluate(env, model):
	env.reset(mode='test_private', TC=1e-4)
	is_end = False
	state = env.get_state()
	rewards = []
	TCs = []
	accum_rewards = []
	buf = 0
	while not is_end:
		print(max(state[:, :, 0].flatten()))
		stock_rank = state.shape[0] - np.argsort(state[:, -1, 0]) - 1
		portfolio, chosen = model(state, stock_rank)
		print(portfolio)
		state, reward, TC, is_end = env.step(portfolio)
		rewards.append(reward)
		mask_tensor = torch.tensor([1 if i in chosen else 0 for i in range(portfolio.shape[0])]).type(torch.FloatTensor).cuda()
		TCs.append(TC)
		buf += reward.cpu().numpy() - TC
		accum_rewards.append(buf)
	excess_returns = [r.cpu().numpy() - t for r, t in zip(rewards, TCs)]
	print('Test rewards = {}'.format(sum(excess_returns)))
	pl.hist(excess_returns, bins=100)
	pl.show()
	pl.plot(accum_rewards)
	pl.show()

def main(args):
	env = FXEnv(instruments=instruments,
				holding_period=12,
				trading_period=12*12,
				look_back=12)
	env.reset(mode='train')

	model = AlphaStock(num_features=3, num_stocks=len(instruments), preset_size=4)
	model.eval()
	model.cuda()


	model.load_state_dict(torch.load(args.model_path))

	with torch.no_grad():
		evaluate(env, model)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--model_path', type=str)

	args = parser.parse_args()
	main(args)

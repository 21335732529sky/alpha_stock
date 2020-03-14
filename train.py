import torch
import argparse
from environment import *
from alpha_stock import *
from pprint import pprint

instruments = ['USD_JPY', 'GBP_JPY', 'AUD_JPY', 'CAD_JPY', 'NZD_JPY', 'EUR_JPY', 'AUD_USD', 'EUR_USD', 'GBP_USD', 'NZD_USD',
			   'USD_CHF', 'EUR_CHF', 'GBP_CHF', 'EUR_GBP', 'AUD_NZD', 'AUD_CAD', 'AUD_CHF', 'CAD_CHF']

def calc_sharp_ratio(rewards, TCs):
	rewards = [r.detach().cpu().numpy() for r in rewards]
	mean = sum(rewards) / len(rewards)
	At = sum(r - t for r, t in zip(rewards, TCs)) / len(rewards)
	vol = sum((r - mean) ** 2 for r in rewards) / len(rewards)
	vol = vol ** 0.5

	return (At - 1e-7) / (vol + 1e-9)


def evaluate(env, model):
	model.eval()
	env.reset(mode='test', TC=1e-4)
	is_end = False
	state = env.get_state()
	rewards = []
	TCs = []
	while not is_end:
		stock_rank = state.shape[0] - np.argsort(state[:, -1, 0]) - 1
		portfolio, chosen = model(state, stock_rank)
		print(portfolio)
		state, reward, TC, is_end = env.step(portfolio)
		rewards.append(reward)
		mask_tensor = torch.tensor([1 if i in chosen else 0 for i in range(portfolio.shape[0])]).type(torch.FloatTensor).cuda()
		TCs.append(TC)
	print(sorted(rewards)[:3], sorted(rewards)[-3:])
	At = sum(r - t for r, t in zip(rewards, TCs)) / len(rewards)
	vol = sum((r - r_mean) ** 2 for r in rewards) / len(rewards)
	vol = vol ** 0.5
	H = (At - 1e-7) / (vol + 1e-9)

	model.train()
	return H

def train(args):
	env = FXEnv(instruments=instruments,
				holding_period=args.holding_period,
				trading_period=args.holding_period*args.trading_period,
				look_back=args.look_back)
	env.reset(mode='train')

	model = AlphaStock(num_features=3, num_stocks=len(instruments), preset_size=args.preset_size)
	model.cuda()
	model.train()

	optim = torch.optim.RMSprop(model.parameters(), lr=args.lr, momentum=args.momentum)

	loss_buf = 0
	count = 0
	max_rewards = -1
	for i in range(args.num_training_steps):
		is_end = False
		state = env.get_state()
		rewards = []
		nlls = []
		TCs = []
		baselines = []
		env.reset(mode='train', TC=1e-4)

		while not is_end:
			# baselines.append((sum(state[:, :, 0].prod(axis=-1)) - state.shape[0]) / state.shape[0])
			stock_rank = np.argsort(state[:, -1, 0])
			stock_rank = [next(i for i in range(stock_rank.shape[0]) if stock_rank[i] == l) for l in range(stock_rank.shape[0])]
			portfolio, chosen = model(state, stock_rank)
			state, reward, TC, is_end = env.step(portfolio)
			rewards.append(reward)
			mask_tensor = torch.tensor([1 if i in chosen else 0 for i in range(portfolio.shape[0])]).type(torch.FloatTensor).cuda()
			nlls.append(torch.log(portfolio.abs() / 2 + 1e-9) * mask_tensor)
			TCs.append(TC)

		H_pi = calc_sharp_ratio(rewards, TCs)
		sharp_ratio = -H_pi * sum([e.sum() for e in nlls])
		sharp_ratio.backward(retain_graph=True)

		if (i + 1) % 32 == 0:
			print('Step {}: last loss = {:.5f}\r'.format(i, sharp_ratio), end='')
			# pprint([(n, e.grad) for n, e in model.named_parameters()])
			optim.step()
			optim.zero_grad()
			count = 0
		if (i + 1) % args.eval_step == 0:
			with torch.no_grad():
				reward_val = evaluate(env, model)
				print('Step {}: val_rewards = {}'.format(i, reward_val))
				if max_rewards < reward_val:
					max_rewards = reward_val
					torch.save(model.state_dict(), args.model_path)
				# torch.save(model.state_dict(), "AlphaStock_last2.bin")

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--holding_period', type=int, default=12)
	parser.add_argument('--trading_period', type=int, default=12)
	parser.add_argument('--look_back', type=int, default=12)
	parser.add_argument('--preset_size', type=int, default=4)
	parser.add_argument('--lr', type=float, default=0.01)
	parser.add_argument('--momentum', type=float, default=1e-4)
	parser.add_argument('--num_training_steps', type=int, default=1000000)
	parser.add_argument('--eval_step', type=int, default=2000)
	parser.add_argument('--model_path', type=str, default='AlphaStock_best.bin')
	args = parser.parse_args()

	train(args)



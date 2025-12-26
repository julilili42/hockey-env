import torch
import numpy as np
import gymnasium as gym
import optparse
from train import Train

def main():
    optParser = optparse.OptionParser()
    optParser.add_option('-e', '--env',action='store', type='string',
                         dest='env_name',default="Pendulum-v1",
                         help='Environment (default %default)')
    optParser.add_option('-n', '--eps',action='store',  type='float',
                         dest='eps',default=0.1,
                         help='Policy noise (default %default)')
    optParser.add_option('-t', '--train',action='store',  type='int',
                         dest='train',default=32,
                         help='number of training batches per episode (default %default)')
    optParser.add_option('-l', '--lr',action='store',  type='float',
                         dest='lr',default=0.0001,
                         help='learning rate for actor/policy (default %default)')
    optParser.add_option('-m', '--maxepisodes',action='store',  type='float',
                         dest='max_episodes',default=2000,
                         help='number of episodes (default %default)')
    optParser.add_option('-u', '--update',action='store',  type='float',
                         dest='update_every',default=100,
                         help='number of episodes between target network updates (default %default)')
    optParser.add_option('-s', '--seed',action='store',  type='int',
                         dest='seed',default=None,
                         help='random seed (default %default)')
    optParser.add_option('-c', '--checkpoint', action='store', type='string',
                         dest='checkpoint', default=None, 
                         help='Path to checkpoint (.pth). If set: skip training and render only.')
    opts, args = optParser.parse_args()

    
    random_seed = opts.seed
    env_name = opts.env_name
    checkpoint = opts.checkpoint



    # creating environment
    if env_name == "LunarLander-v2":
        env = gym.make(env_name, continuous = True)
    else:
        env = gym.make(env_name)

    
    if random_seed is not None:
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
    


    trainer = Train(opts, env)

    if checkpoint is not None:
        print(f"Loading checkpoint: {opts.checkpoint}")
        trainer.load_checkpoint(opts.checkpoint)
        trainer.render_env()
    else:
        trainer.train_loop()
        trainer.save_statistics()
        trainer.plot_rewards()
        trainer.render_env()

if __name__ == '__main__':
    main()

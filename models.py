import pandas as pd
import numpy as np

class Model:
    
    def __init__(self, env):
        self.env = env

    def predict(self, pred_args={}):
        test_obs = self.env.reset()
        for _ in range(self.env.expiry):
            action = self.model.predict(test_obs, **pred_args)
            test_obs, rewards, dones, info = self.env.step(action)
            if dones[0]:
                print("end!")
                break
                
        return info["output"]
    
    def train_model(self, model, total_timesteps=5000, n_eval_episodes=5):
        model = model.learn(total_timesteps=total_timesteps, n_eval_episodes=n_eval_episodes)
        return model
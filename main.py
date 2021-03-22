import numpy as np
from collections import namedtuple
import gym



class PGCartPoleAgent:
    def __init__(self):
        self.INPUT_DIMENSION = 4
        self.HIDDEN_LAYER = 200
        self.transition_ = namedtuple('transition', ('state', 'hidden', 'probability', 'reward'))
        self.model = self.create_model()
        self.env = gym.make('CartPole-v0')
        obs = self.env.reset()
        self.batch_size = 50
        self.learning_rate = 1e-3
        self.gamma = 0.99
        self.current_epoch = 0


    def create_model(self):
        model = dict()
        model['W1'] = np.random.randn(self.HIDDEN_LAYER, self.INPUT_DIMENSION) / np.sqrt(self.INPUT_DIMENSION)
        model['W2'] = np.random.randn(self.HIDDEN_LAYER) / np.sqrt(self.HIDDEN_LAYER)
        return model

    @staticmethod
    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))

    @staticmethod
    def discount_rewards(r, gamma=0.99):
        """ take 1D float array of rewards and compute discounted reward """
        running_add = 0
        discounted_r = np.zeros_like(r)
        for idx in reversed(range(0, r.size)):
            running_add = running_add * gamma + r[idx]
            discounted_r[idx] = running_add
        return discounted_r

    def policy_forward(self, obs):
        """ Return probability of taking action 1 (right), and the hidden state """
        hidden = np.dot(self.model['W1'], obs)
        hidden[hidden < 0] = 0  # ReLU nonlinearity
        logits = np.dot(self.model['W2'], hidden)
        probability = self.sigmoid(logits)
        return probability, hidden

    def policy_backward(self, episode_obs, episode_hidden, episode_probability):
        """ backward pass. (eph is array of intermediate hidden states) """
        dW2 = np.dot(episode_hidden.T, episode_probability)
        dh = np.outer(episode_probability, self.model['W2'])
        dh[episode_hidden < 0] = 0  # backprop relu
        dW1 = np.dot(dh.T, episode_obs)
        return {'W1': dW1, 'W2': dW2}

    def run_one_episode(self, training=True):
        done = False
        obs = self.env.reset()
        score = 0
        memory = list()
        while not done:
            prob, hidden = self.policy_forward(obs)
            if training:
                action = 1 if np.random.uniform() < prob else 0
            else:
                action = 1 if prob > 0.5 else 0  # deterministic in testing mode
            obs, rew, done, info = self.env.step(action)
            score += rew
            prob = action - prob
            memory.append(self.transition_(obs, hidden, prob, rew))
        return score, memory

    def train_one_epoch(self):
        running_reward = 0
        running_mean = 0
        memory = list()
        # update buffers that add up gradients over a batch and rmsprop memory
        gradient_buffer = {k: np.zeros_like(v, dtype=np.float) for k, v in self.model.items()}

        for episode in range(self.batch_size):
            score, transitions = self.run_one_episode()
            memory += transitions
            running_mean = running_mean * 0.99 + score * 0.01
            print(f"Episode {episode:6d}, score: {score: 4.0f}, running mean: {running_mean: 6.2f}")

            # Convert memory to a stack
            observations, hiddens, probabilities, rewards = np.array(list(zip(*memory)), dtype=np.object)
            observations = np.array(list(observations), dtype=np.float)
            hiddens = np.array(list(hiddens), dtype=np.float)

            # Calculate discounted rewards
            discounter_reward = self.discount_rewards(rewards, self.gamma)

            # standardize the rewards to be unit normal (helps control the gradient estimator variance)
            discounter_reward -= np.mean(discounter_reward)
            discounter_reward /= np.std(discounter_reward)

            # modulate the gradient with advantage (PG magic happens right here.)
            probabilities *= discounter_reward
            grad = self.policy_backward(observations, hiddens, probabilities)

            # accumulate grad over batch
            for weight in self.model:
                gradient_buffer[weight] += np.array(grad[weight], dtype=np.float)

            running_reward = running_reward * 0.99 + score * 0.01
            print(f"Epoch {self.current_epoch:6d}, Episode {episode:6d}, "
                  f"score: {score: 4.0f}, running mean: {running_reward: 6.2f}")

        for layer, weights in self.model.items():
            self.model[layer] += self.learning_rate * gradient_buffer[layer]
            gradient_buffer[layer] = np.zeros_like(weights)

    def train(self, nr_epochs=30):
        for epoch in range(1, nr_epochs+1):
            print("Starting epoch", epoch)
            self.current_epoch = epoch
            self.train_one_epoch()

    def evaluate(self, nr_games=100):
        """ Evaluate the model results.  """
        collected_scores = []
        for episode in range(1, nr_games + 1):
            score, _ = self.run_one_episode()
            print(f"\r\tGame {episode:3d}/{nr_games:3d} score: {score}", end='')
            collected_scores.append(score)
        average = sum(collected_scores) / nr_games
        print(f"\n\nThe model played: {nr_games} games, with an average score of: {average: 5.2f}")
        return average


if __name__ == '__main__':
    agent = PGCartPoleAgent()
    agent.train()
    agent.evaluate()














# Upper Confidence Bound

# Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing the dataset
dataset = pd.read_csv('Ads_CTR_Optimisation.csv')

# Implementing UCB
import math
N = 10000
d = 10

ads_selected = []
nums_selections = [0] * d
nums_rewards = [0] * d
max_confidence = 0
total_reward = 0

for n in range(0, N):
    ad = 0
    max_confidence = 0
    for i in range(0, d):
        if (nums_selections[i] > 0):
            number_of_selected_ads = nums_selections[i]
            sum_rewards = nums_rewards[i]
            average_reward = sum_rewards / number_of_selected_ads
            delta = math.sqrt(3/2 / math.log(N) / number_of_selected_ads)
            curr_confidence = average_reward + delta
        else:
            curr_confidence = 1e400
        if curr_confidence > max_confidence:
            max_confidence = curr_confidence
            ad = i
    ads_selected.append(ad)
    nums_selections[ad] += 1
    reward = dataset.values[n, ad]
    nums_rewards[ad] += reward
    total_reward += reward

# Visualizing the results
plt.hist(ads_selected)
plt.title('Histogram of Ad Selections')
plt.xlabel('Ad')
plt.ylabel('Number of Selections')
plt.show()
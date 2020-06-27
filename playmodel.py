import sys
import model

path = sys.argv[1]
episodes = 100

if len(sys.argv) > 2:
    episodes = int(sys.argv[2])

m = model.TrainedModel(path)

rewards = []

for i in range(episodes):
    r = m.play()
    print(r)
    rewards.append(r)


print(rewards)
print("Episodes: " + episodes)
print("Average reward: " + str(sum(rewards)/episodes))
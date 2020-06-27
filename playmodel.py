import sys
import model

path = sys.argv[1]
episodes = 100

if len(sys.argv) > 2:
    episodes = int(sys.argv[2])

m = model.TrainedModel(path)

rewards = []

for i in range(episodes):
    r = m.play(render=True)
    print(r)
    rewards.append(r)

m.env.close()


print(rewards)
print("Episodes: " + str(episodes))
print("Average reward: " + str(sum(list(map(lambda r: r[0], rewards)))/episodes))
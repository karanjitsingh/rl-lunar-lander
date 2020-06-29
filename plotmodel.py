from tensorboard.backend.event_processing import event_accumulator
import sys
import matplotlib.pyplot as plt
import matplotlib
import numpy
import model

plt.rcParams['image.cmap']='jet'

# path = ".\models\Jun27_18-38-13_KARSIN-PC\Jun27_18-38-13_KARSIN-PC.2700.model"
path = ".\models\Jun29_06-28-09_KARSIN-PC\Jun29_06-28-09_KARSIN-PC.3300.model"

episodes = 100

if len(sys.argv) > 2:
    episodes = int(sys.argv[2])

m = model.TrainedModel(path)

rspair = []

for i in range(episodes):
    r = m.play(render=False)
    print(i,r)
    rspair.append(r)

m.env.close()


rewards = list(map(lambda r: r[0], rspair))
avg = sum(rewards)/episodes
print(rewards)
print("Episodes: " + str(episodes))
print("Average reward: " + str(avg))


prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

plt.rcParams.update({'font.size': 22})



plt.figure(figsize=(10,5), dpi=120)
plt.subplots_adjust(left=0.150, bottom=0.17, right=0.99, top=0.97, wspace=0.2, hspace=0.2)

plt.plot(range(1,101),rewards,color=colors[0], alpha = 0.3)
plt.plot(range(1,101),[avg]*100,color=colors[0])
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.show()

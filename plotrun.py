from tensorboard.backend.event_processing import event_accumulator
import sys
import matplotlib.pyplot as plt
import matplotlib
import numpy

plt.rcParams['image.cmap']='jet'


prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

plt.rcParams.update({'font.size': 22})

def smooth(scalars, weight):  # Weight between 0 and 1
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)                        # Save it
        last = smoothed_val                                  # Anchor the last smoothed value

    return smoothed

def plot(models, param, smoothing, text=None, clip=None, ticks=None, xlabel="", ylabel=""):
    models = list(map(lambda m: ".\\runs\\" + m, models))

    plt.figure(figsize=(10,5), dpi=120)
    plt.subplots_adjust(left=0.16, bottom=0.17, right=0.99, top=0.97, wspace=0.2, hspace=0.2)
    for i in range(len(models)):
        model = models[i]
        
        ea = event_accumulator.EventAccumulator(model)
        ea.Reload()

        scalars = ea.Scalars(param)

        x = list(map(lambda x: x.step, scalars))[:clip]
        y = list(map(lambda x: x.value, scalars))[:clip]
        sy = smooth(y,smoothing)
        plt.ylim([-1000,400])

        plt.plot(x,y,color=colors[i], alpha=0.1)
        plt.plot(x,sy,color=colors[i], alpha=0.8, label= None if not text else text[i])

        y_ticks = plt.yticks()[0]

        if ticks is not None:
            for tick in ticks:
                y_ticks = numpy.append(y_ticks, tick)

        plt.yticks(y_ticks)


    # plt.yscale("log")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if text:
        plt.legend(loc=3, prop={'size': 12})
    plt.show()



trained_models = ["Jun27_18-38-13_KARSIN-PC"]
alpha_models = ["Jun28_18-21-28_KARSIN-PC", "Jun27_18-38-13_KARSIN-PC", "Jun28_18-21-20_KARSIN-PC", "Jun28_18-21-25_KARSIN-PC", "Jun28_18-31-15_KARSIN-PC", "Jun28_18-36-36_KARSIN-PC"]
mem_models = ["Jun29_09-38-18_KARSIN-PC", "Jun28_22-34-49_KARSIN-PC", "Jun28_22-34-59_KARSIN-PC", "Jun28_22-35-18_KARSIN-PC", "Jun28_22-35-11_KARSIN-PC"]
bound_models = ["Jun27_18-38-13_KARSIN-PC", "Jun27_18-20-59_KARSIN-PC"]
on_policy = ["Jun29_06-28-09_KARSIN-PC"]
replay = ["Jun27_18-38-13_KARSIN-PC", "Jun28_22-34-59_KARSIN-PC", "Jun29_02-41-31_KARSIN-PC"]


alphalabels = ["0.00005", "0.0001", "0.0002", "0.0004", "0.0008", "0.0016"]
mem_labels = ["200", "1000", "10000", "50000", "100000"]
bound_labels = ["Bounded", "Unbounded"]
replay_labels = ["Target network and replay", "Replay only", "Neither replay nor target network"]

# plot(trained_models, 'Reward', 0.8, clip=2000, ticks=[200], xlabel="Episode", ylabel="Reward")
# plot(alpha_models, 'Reward', 0.9, clip=2000, xlabel="Episode", ylabel="Reward", text=alphalabels)
# plot(bound_models, 'Steps', 0.8, clip=2500, xlabel="Episode", ylabel="Steps (log scale)", text=bound_labels)
# plot(trained_models, 'Reward', 0.8, ticks=[200], xlabel="Episode", ylabel="Reward")
# plot(on_policy, 'Reward', 0.8, ticks=[200], clip=5000, xlabel="Episode", ylabel="Reward")

plot(replay, 'Reward', 0.8, clip=2100, xlabel="Episode", ylabel="Reward", text=replay_labels)

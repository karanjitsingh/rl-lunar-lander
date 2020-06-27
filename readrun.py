from tensorboard.backend.event_processing import event_accumulator
import sys

path = sys.argv[1]

ea = event_accumulator.EventAccumulator(path)
ea.Reload()

print(ea.Tags())
print(ea.Scalars('Loss')[0])
print(ea.Scalars('Loss')[1])


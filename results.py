import pickle as pkl
import matplotlib.pyplot as plt


# Replace 'your_file.pkl' with the actual file path
with open('/scratch/projects/fouheylab/dma9300/OSNOM/osnom_output/results.pkl', 'rb') as file:
    data = pkl.load(file)

# Print or use the loaded data

# Extract keys divided by 50 and corresponding values
x_values = [key / 50 for key in data.keys()]
y_values = [list(value_dict.values())[0] for value_dict in data.values()]

# Plot
plt.figure(figsize=(8, 5))
plt.plot(x_values, y_values, marker='o', linestyle='-', color='b')
plt.xlabel("Time (sec)")
plt.ylabel("Percentage of Correct Locations ")
plt.title("PCL versus Time")
plt.grid(True)
plt.savefig("osnom_eval.png")

import matplotlib.pyplot as plt

hpc = list(hpc_data.keys())
iteration_speed = 0
calculation_speed = 0

hpc_data = {
    'HPC1': {'iteration_speed': 90, 'calculation_speed': 120},
    'HPC1': {'iteration_speed': 120, 'calculation_speed': 80},
    'HPC1': {'iteration_speed': 100, 'calculation_speed': 100}
}



plt.bar(hpc, 'iteration_speed', label= 'Iteration Speed')
plt.bar(hpc, 'calculation_speed', label= 'Calculation Speed', bottom=iteration_speed)

plt.xl



import numpy as np

hpc_years = np.array([2020, 2021, 2022, 2023])
performance_data = {
    'HPC1': '',
    'HPC2': '',
    'HPC3': '',
}

plt.figure(figsize=(10,6))
for hpc, performance in performance_data.items():
    plt.plot(hpc_years, performance, label=hpc)
bubble_sizes = [500, 1000, 1500]
for i, hpc in enumerate(hpc_data):
    
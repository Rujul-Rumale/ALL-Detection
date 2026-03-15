import numpy as np
import matplotlib.pyplot as plt

# IEEE font settings
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12

# Sample data
t = np.linspace(0, 1, 500)
signal = np.sin(2*np.pi*5*t)

# Plot (double-column width)
plt.figure(figsize=(7.16, 3))   # IEEE double-column width
plt.plot(t, signal, linewidth=2)

plt.xlabel("Time (s)")
plt.ylabel("Amplitude (mV)")
plt.title("ECG Signal")

plt.grid(True)

# Improve axis appearance
plt.tick_params(direction='in')
plt.minorticks_on()

# Save figure (IEEE preferred vector format)
plt.savefig("ecg_signal_double.eps", format='eps',
            dpi=300, bbox_inches='tight')

# Save high-resolution PNG
plt.savefig("ecg_signal_double.png",
            dpi=600, bbox_inches='tight')

plt.show()
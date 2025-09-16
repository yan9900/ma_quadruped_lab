# Interactive upright orientation reward using Matplotlib sliders (no ipywidgets needed)
# r = exp(-((g_z + 1)^2) / (2 * ε^2))
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

def upright_reward(gz, epsilon):
    gz = np.asarray(gz, dtype=float)
    return np.exp(-((gz + 1.0) ** 2) / (2.0 * epsilon ** 2))

# Initial values
eps0 = 0.15
gz0 = -1.0
x = np.linspace(-2.0, 2.0, 800)

# Create a single chart for the curve
fig, ax = plt.subplots(figsize=(7, 4.5))
y = upright_reward(x, eps0)
[line] = ax.plot(x, y, label="r = exp(-((g_z + 1)^2)/(2 ε^2))")
vline = ax.axvline(gz0, linestyle="--", label="current g_z")
ax.set_xlim(-2, 2)
ax.set_ylim(0, 1.05)
ax.set_xlabel("g_z  (gravity z-component in base frame)")
ax.set_ylabel("upright reward")
ax.set_title("Upright Orientation Reward")
ax.legend(loc="best")
fig.tight_layout(rect=[0, 0.15, 1, 1])  # leave space at bottom for sliders

# Slider axes (not additional charts)
ax_eps = fig.add_axes([0.12, 0.06, 0.76, 0.03])
ax_gz  = fig.add_axes([0.12, 0.02, 0.76, 0.03])

slider_eps = Slider(ax=ax_eps, label="ε", valmin=0.01, valmax=2, valinit=eps0, valstep=0.01)
slider_gz  = Slider(ax=ax_gz,  label="g_z", valmin=-2.0, valmax=1.0, valinit=gz0,  valstep=0.01)

# Status text
fig.suptitle(f"ε = {eps0:.3f} | g_z = {gz0:.3f} | reward = {upright_reward(gz0, eps0):.4f}", y=0.98)

def on_change(val):
    eps = slider_eps.val
    gz  = slider_gz.val
    line.set_ydata(upright_reward(x, eps))
    vline.set_xdata([gz, gz])
    fig.suptitle(f"ε = {eps:.3f} | g_z = {gz:.3f} | reward = {upright_reward(gz, eps):.4f}", y=0.98)
    fig.canvas.draw_idle()

slider_eps.on_changed(on_change)
slider_gz.on_changed(on_change)

plt.show()
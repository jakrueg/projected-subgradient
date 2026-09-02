import proj_grad

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap



def project_smc(x):
    """Project a two-dimensional point onto the three simple sets.

    The implementation avoids building intermediate arrays and keeps the
    computation entirely in scalar form, which is faster for this small
    fixed-dimension projection.
    """
    z0 = float(x[0])
    z1 = float(x[1])

    # Candidate 1: [0, inf) x {0}
    p0 = max(z0, 0.0)
    p1 = 0.0
    d1square = (z0 - p0) * (z0 - p0) + z1 * z1

    # Candidate 2: {0} x [0, 2]
    q0 = 0.0
    q1 = min(max(z1, 0.0), 2.0)
    d2square = z0 * z0 + (z1 - q1) * (z1 - q1)

    # Candidate 3: {(t, t + 2): t >= 0}
    if z0 + z1 <= 2.0:
        r0 = 0.0
        r1 = 2.0
    else:
        r0 = 0.5 * (z0 + z1 - 2.0)
        r1 = 0.5 * (z0 + z1 + 2.0)
    d3square = (z0 - r0) * (z0 - r0) + (z1 - r1) * (z1 - r1)

    if d1square <= d2square and d1square <= d3square:
        return np.array([p0, p1], dtype=float)
    if d2square <= d3square:
        return np.array([q0, q1], dtype=float)
    return np.array([r0, r1], dtype=float)

def project_smc_tangential(x,g):
    """Project a two-dimensional point onto the tangential cone of three simple sets.
    """
    z0 = float(x[0])
    z1 = float(x[1])
    g0 = float(g[0])
    g1 = float(g[1])
    if z0 == 0 and z1 == 0:
        i_axis = g.copy()
        j_axis = g.copy()
        i_axis[0] = max(g0,0.0)
        i_axis[1] = 0.0
        j_axis[0] = 0.0
        j_axis[1] = max(g1,0.0)

        if ((i_axis[0] - g0) ** 2 + (i_axis[1] - g1) ** 2
                <= (j_axis[i] - g0) ** 2 + (j_axis[j] - g1) ** 2):
            z[i], z[j] = i_axis[i], i_axis[j]
        else:
            z[i], z[j] = j_axis[i], j_axis[j]
        return np.array([max(g0, 0.0), max(g1, 0.0)], dtype=float)


obj = lambda x: 0.5*(x[0]-1)**2 + 0.5*(x[1]-1)**2
grad = lambda x: np.array([x[0]-1, x[1]-1])

x_start = np.array([-1, 1])
res = proj_grad.pgd_ac(x0=x_start, f=obj, grad=grad, proj=project_smc, max_iter=1000, TOL=1e-6)
print(res.message)
print(f"Starting from {x_start}, converged to {res.x} with objective value {obj(res.x)}")


# Run grid search and collect results
interval = np.linspace(-1.0, 4.0, 100)
results = []

print("Running grid search over starting points...")
for i, x0_val in enumerate(interval):
    for j, y0_val in enumerate(interval):
        z0 = np.array([x0_val, y0_val], dtype=float)
        res = proj_grad.pgd_avg(x0=z0, f=obj, grad=grad, proj=project_smc, max_iter=1000, TOL=1e-6)
        #print(res.message)
        x_final = res.x
        results.append({
            'start_x': x0_val,
            'start_y': y0_val,
            'final_x': x_final[0],
            'final_y': x_final[1],
            'final_obj': obj(x_final),
            'grid_i': i,
            'grid_j': j
        })

df = pd.DataFrame(results)

# Identify unique attractors using rounding tolerance
tolerance = 1e-3
df['attractor'] = (df[['final_x', 'final_y']].round(int(-np.log10(tolerance)))).\
    apply(tuple, axis=1)

# Analyze attractors
attractor_analysis = df.groupby('attractor').agg({
    'start_x': 'count',
    'final_obj': 'first'
}).rename(columns={'start_x': 'num_converged'})

print(f"\nTotal unique attractors: {len(attractor_analysis)}")
print("\nAttractor summary:")
print(attractor_analysis)
print("\nConvergence distribution:")
for attractor, count in df['attractor'].value_counts().items():
    print(f"  Attractor {attractor}: {count} starting points converge")

# Create attractor ID mapping
attractor_ids = {att: i for i, att in enumerate(sorted(df['attractor'].unique()))}
df['attractor_id'] = df['attractor'].map(attractor_ids)

# Reshape data for heatmap
attractor_grid = np.zeros((len(interval), len(interval)))
for idx, row in df.iterrows():
    attractor_grid[row['grid_i'], row['grid_j']] = row['attractor_id']

# Save the heatmap of attractors separately
fig_attractor = plt.figure(figsize=(6, 6))
ax_attractor = fig_attractor.add_subplot(111)
im_attractor = ax_attractor.imshow(attractor_grid, extent=[-1, 4, -1, 4], origin='lower', cmap='tab20', aspect='auto')
ax_attractor.set_xlabel('Initial $x_1$')
ax_attractor.set_ylabel('Initial $x_2$')
ax_attractor.set_title('Basins of Attraction (LS)')
# Add integer grid and coordinate system
ax_attractor.set_xticks(np.arange(-1, 5, 1))
ax_attractor.set_yticks(np.arange(-1, 5, 1))
ax_attractor.grid(True, linestyle='--', alpha=0.4, color='black')
ax_attractor.axhline(y=0, color='black', linewidth=2)
ax_attractor.axvline(x=0, color='black', linewidth=2)

# Mark the attractors on the heatmap
unique_attractors = df.drop_duplicates(subset=['attractor'])
for idx, row in unique_attractors.iterrows():
    ax_attractor.plot(row['final_x'], row['final_y'], 'r*', markersize=15, 
                     markeredgecolor='black', markeredgewidth=1.5, label='Attractors' if idx == unique_attractors.index[0] else '')
    #ax_attractor.text(row['final_x'], row['final_y'] + 0.15, f"{row['attractor_id']}", 
    #                 ha='center', fontsize=9, fontweight='bold', color='black',
    #                 bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))

#if len(unique_attractors) > 0:
#    ax_attractor.legend(loc='upper left', fontsize=9)

fig_attractor.tight_layout()
fig_attractor.savefig('basins_of_attraction.pdf')
print("\nHeatmap of attractors saved to 'basins_of_attraction.pdf'")
plt.close(fig_attractor)
"""
# Create visualization (combined plot)
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Heatmap of attractors
im = axes[0].imshow(attractor_grid, extent=[-1, 4, -1, 4], origin='lower', cmap='tab20', aspect='auto')
axes[0].set_xlabel('Initial $x_1$')
axes[0].set_ylabel('Initial $x_2$')
axes[0].set_title('Basins of Attraction')
plt.colorbar(im, ax=axes[0], label='Attractor ID')

# Heatmap of objective values at convergence
obj_grid = np.zeros((len(interval), len(interval)))
for idx, row in df.iterrows():
    obj_grid[row['grid_i'], row['grid_j']] = row['final_obj']

im2 = axes[1].imshow(obj_grid, extent=[-1, 4, -1, 4], origin='lower', cmap='viridis', aspect='auto')
axes[1].set_xlabel('Initial $x_1$')
axes[1].set_ylabel('Initial $x_2$')
axes[1].set_title('Objective Value at Convergence')
plt.colorbar(im2, ax=axes[1], label='Objective Value')

plt.tight_layout()
plt.show()
"""
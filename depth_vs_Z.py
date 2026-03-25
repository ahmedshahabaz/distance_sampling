import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import get_cmap

# ---------- Global Style ----------
plt.rcParams.update({
    "font.size": 15,
    "axes.labelsize": 13,
    "legend.fontsize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12
})

cmap = get_cmap("viridis")

fig, ax = plt.subplots(figsize=(6, 6))

cam_x, cam_z = 0, 0
X, Y, Z = 3, 2, 8
obj_x, obj_z = X, Z
distance = np.sqrt(X**2 + Y**2 + Z**2)

# Colors from colormap
cam_color = cmap(0.1)
obj_color = cmap(0.6)
depth_color = cmap(0.85)
dist_color = "black"

# ---------- Camera ----------
ax.plot(cam_x, cam_z, marker='s', color=cam_color,
        markersize=12, label='Camera')
ax.text(cam_x-0.4, cam_z-0.5, 'Camera (0,0,0)',
        ha='right')

# ---------- Object ----------
ax.plot(obj_x, obj_z, marker='o',
        color=obj_color, markersize=10,
        label='Object (X,Y,Z)')
ax.text(obj_x+0.25, obj_z+0.25,
        f'({X}, {Y}, {Z})',
        color=obj_color)

# ---------- Z-axis (Depth) ----------
ax.arrow(cam_x, cam_z, 0, Z,
         head_width=0.25,
         head_length=0.6,
         fc=depth_color,
         ec=depth_color,
         linewidth=2)

ax.text(cam_x+0.3, Z/2,
        f'Z = {Z}',
        color=depth_color,
        rotation=90,
        va='center')

# ---------- Actual Distance ----------
ax.plot([cam_x, obj_x],
        [cam_z, obj_z],
        linestyle='--',
        linewidth=2,
        color=dist_color,
        label='Euclidean Distance')

dx = obj_x - cam_x
dz = obj_z - cam_z
angle = np.degrees(np.arctan2(dz, dx))

mid_x, mid_z = (cam_x + obj_x)/2, (cam_z + obj_z)/2
calc_str = (r"$\sqrt{%d^2 + %d^2 + %d^2}$" % (X, Y, Z) +
            f"\n= {distance:.2f}")

ax.text(mid_x, mid_z,
        calc_str,
        rotation=angle,
        rotation_mode='anchor',
        ha='center',
        va='center',
        bbox=dict(facecolor='white',
                  edgecolor='none',
                  alpha=0.85))

# ---------- Projections ----------
ax.plot([obj_x, obj_x], [0, obj_z],
        linestyle=':',
        linewidth=1.5,
        color='gray')

ax.plot([cam_x, obj_x], [cam_z, cam_z],
        linestyle=':',
        linewidth=1.5,
        color='gray')

ax.text(obj_x/2, cam_z-0.4,
        f"X = {X}",
        ha='center')

ax.text(obj_x+0.3, cam_z-0.4,
        f"Y = {Y}")

# ---------- Axes & Layout ----------
ax.axhline(0, color='gray', linestyle='--', linewidth=1)

ax.set_xlabel('X-axis')
ax.set_ylabel('Z-axis (Depth)')

ax.set_aspect('equal', adjustable='box')
ax.grid(True, alpha=0.3)

ax.legend(frameon=False,
          loc='upper left',
          bbox_to_anchor=(1, 0.8))

plt.tight_layout()
plt.savefig('REPORTS/depth_vs_Z.svg',
            format="svg",
            bbox_inches="tight",
            dpi=300)
plt.show()


# fig = plt.figure(figsize=(10, 8))
# ax = fig.add_subplot(111, projection='3d')

# # Camera position (origin)
# cam = np.array([0, 0, 0])

# # Object position
# X, Y, Z = 3, 2, 8
# obj = np.array([X, Y, Z])
# distance = np.sqrt(X**2 + Y**2 + Z**2)

# # Draw Camera
# ax.scatter(*cam, color='k', s=80, label='Camera (0,0,0)')
# ax.text(-0.5, -0.5, -0.5, "Camera (0,0,0)", fontsize=12, ha='right')

# # Draw Object Point
# ax.scatter(*obj, color='b', s=60, label=f'Object ({X},{Y},{Z})')
# ax.text(X+0.2, Y+0.2, Z, f'({X}, {Y}, {Z})', color='blue', fontsize=12)

# # Draw Z axis (Depth)
# ax.plot([0, 0], [0, 0], [0, Z], color='g', linewidth=2, label='Z (Depth)')
# ax.text(0.25, 0, Z/2, 'Z (Depth)', color='g', fontsize=12, rotation=90)

# # Draw X axis
# ax.plot([0, X], [0, 0], [0, 0], color='purple', linewidth=2, label='X axis')
# ax.text(X/2, -0.3, 0, 'X', color='purple', fontsize=12)

# # Draw Y axis
# ax.plot([0, 0], [0, Y], [0, 0], color='orange', linewidth=2, label='Y axis')
# ax.text(0, Y/2, -0.3, 'Y', color='orange', fontsize=12)

# # Draw distance line
# ax.plot([0, X], [0, Y], [0, Z], 'r--', linewidth=2, label='Distance')
# # Place the distance calculation label along the line
# ax.text(X/2, Y/2, Z/2, 
#         r'Distance = $\sqrt{%d^2 + %d^2 + %d^2}$'"\n= %.2f" % (X, Y, Z, distance), 
#         color='r', fontsize=12, rotation=20, ha='center')

# # Axes labels
# ax.set_xlabel('X-axis (horizontal image axis)')
# ax.set_ylabel('Y-axis (vertical image axis)')
# ax.set_zlabel('Z-axis (depth, out from camera)')

# ax.set_title('3D Visualization: Depth (Z), X, Y, and Actual Distance')
# ax.legend(loc='upper left', bbox_to_anchor=(1.13, 0.9))

# # Set limits for better viewing
# ax.set_xlim(-1, X+3)
# ax.set_ylim(-1, Y+3)
# ax.set_zlim(-1, Z+3)

# plt.tight_layout()
# plt.show()

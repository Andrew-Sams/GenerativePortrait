# Required Libraries
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, RegularPolygon
from matplotlib.patches import Ellipse, Polygon, Circle, Arc
from matplotlib.collections import PatchCollection
from matplotlib.path import Path
from matplotlib.animation import FuncAnimation
import ipywidgets as widgets
from IPython.display import display

# Define the main facial boundaries more precisely
backgrounds = {
    #'background0': Ellipse((0.5, 0.55), 0.5*3.5, 0.7*3.5, fill=False),
    'background': Ellipse((0.5, 0.55), 0.5*2.5, 0.7*2.5, fill=False),
    'background2': Ellipse((0.5, 0.55), 0.5*1.5, 0.7*1.5, fill=False),
    #'background3': Ellipse((0.5, 0.55), 0.5*1.0, 0.7*1.0, fill=False)
}

face_backgrounds = {
    'head': Ellipse((0.5, 0.55), 0.5, 0.7, fill=False),
    'face': Ellipse((0.5, 0.55), 0.4, 0.7, fill=False),
    #'background0': Ellipse((0.5, 0.55), 0.5*3.5, 0.7*3.5, fill=False)
}

hair_boundaries = {
    'hair_left': Ellipse((0.3, 0.68), 0.1, 0.225, fill=False),
    'hair_right': Ellipse((0.7, 0.68), 0.1, 0.225, fill=False),
    'hair_top_left': Ellipse((0.375, 0.8), 0.1, 0.1, fill=False),
    'hair_right_right': Ellipse((0.625, 0.8), 0.1, 0.1, fill=False),
    'hair_center': Ellipse((0.5, 0.65+0.2), 0.2, 0.1, fill=False),
    'hair_top_left_top': Ellipse((0.375, 0.8+0.025), 0.15, 0.15, fill=False),
    'hair_right_right_top': Ellipse((0.625, 0.8+0.025), 0.15, 0.15, fill=False),
    'hair_center_top': Ellipse((0.5, 0.65+0.2+0.025), 0.25, 0.15, fill=False),
    'left_inner_circle': Ellipse((0.5-0.22, 0.63), 0.1, 0.4, fill=False),
    'right_inner_circle': Ellipse((0.5+0.22, 0.63), 0.1, 0.4, fill=False),
    #'outer_circle': Ellipse((0.5, 0.65), 0.5*7/6, 0.5*7/6, fill=False)
}

face_boundaries = {
    #'head': Ellipse((0.5, 0.55), 0.5, 0.7, fill=False),
    #'face': Ellipse((0.5, 0.55), 0.4, 0.7, fill=False),
    #'outer_circle': Ellipse((0.5, 0.65), 0.5*7/6, 0.5*7/6, fill=False),
    #'inner_circle': Ellipse((0.5, 0.65), 0.5, 0.5, fill=False),
    #'left_inner_circle': Ellipse((0.5-0.22, 0.63), 0.1, 0.4, fill=False),
    #'right_inner_circle': Ellipse((0.5+0.22, 0.63), 0.1, 0.4, fill=False),
    'left_cheek': Ellipse((0.41, 0.465), 0.12, 0.12, fill=False),
    'right_cheek': Ellipse((0.59, 0.465), 0.12, 0.12, fill=False),
    'left_temple': Ellipse((0.25+0.075, 0.6), 0.03, 0.18, fill=False),
    'right_temple': Ellipse((0.25+0.15+0.2+0.075, 0.6), 0.03, 0.18, fill=False),
    'left_side': Ellipse((0.25+0.1, 0.425), 0.03, 0.18, fill=False),
    'right_side': Ellipse((0.25+0.15+0.2+0.05, 0.425), 0.03, 0.18, fill=False),
    'forehead': Ellipse((0.5, 0.65), 0.3, 0.1, fill=False),
    'top_forehead': Ellipse((0.5, 0.65+0.1), 0.3, 0.1, fill=False),
    'left_temple': Ellipse((0.25+0.075, 0.6), 0.03, 0.18, fill=False),
    'right_temple': Ellipse((0.25+0.15+0.2+0.075, 0.6), 0.03, 0.18, fill=False),
    'left_forehead': Ellipse((0.25+0.4, 0.6+0.1), 0.1, 0.3, fill=False),
    'right_forehead': Ellipse((0.25+0.4+0.2-0.5, 0.6+0.1), 0.1, 0.3, fill=False),
    'left_side': Ellipse((0.25+0.1, 0.425), 0.03, 0.18, fill=False),
    'right_side': Ellipse((0.25+0.15+0.2+0.05, 0.425), 0.03, 0.18, fill=False),
    'left_cheek': Ellipse((0.41, 0.465), 0.12, 0.12, fill=False),
    'right_cheek': Ellipse((0.59, 0.465), 0.12, 0.12, fill=False),
    'neck': Ellipse((0.5, 0.1), 0.2, 0.2, fill=False),
    'neck_left': Ellipse((0.4, 0.15), 0.06, 0.2, fill=False),
    'neck_right': Ellipse((0.6, 0.15), 0.06, 0.2, fill=False),
    'left_jaw': Ellipse((0.4, 0.31), 0.06, 0.12, fill=False),
    'right_jaw': Ellipse((0.6, 0.31), 0.06, 0.12, fill=False),
    'chin': Ellipse((0.5, 0.24), 0.1, 0.075, fill=False),
    'chin_left': Ellipse((0.5-0.05, 0.24), 0.075, 0.075, fill=False),
    'chin_right': Ellipse((0.5+0.05, 0.24), 0.075, 0.075, fill=False),
    'left_eye': Ellipse((0.25+0.15, 0.55), 0.1, 0.0382, fill=False),
    'right_eye': Ellipse((0.25+0.15+0.2, 0.55), 0.1, 0.0382, fill=False),
    'left_ear': Ellipse((0.275, 0.5), 0.05, 0.175, fill=False),
    'right_ear': Ellipse((0.725, 0.5), 0.05, 0.175, fill=False),
    'left_iris': Ellipse((0.25+0.15, 0.55), 0.05, 0.05, fill=False),
    'right_iris': Ellipse((0.25+0.15+0.2, 0.55), 0.05, 0.05, fill=False),
    'left_pupil': Ellipse((0.25+0.15, 0.55), 0.02, 0.02, fill=False),
    'right_pupil': Ellipse((0.25+0.15+0.2, 0.55), 0.02, 0.02, fill=False),
    'center_brow': Ellipse((0.25+0.15+0.1, 0.6), 0.21, 0.03, fill=False),
    'left_brow': Ellipse((0.25+0.15, 0.6), 0.11, 0.02, fill=False),
    'right_brow': Ellipse((0.25+0.15+0.2, 0.6), 0.11, 0.02, fill=False),
    'nose_bridge': Ellipse((0.5, 0.51), 0.05, 0.125, fill=False),
    'left_nose_bridge': Ellipse((0.5-0.03, 0.5), 0.04, 0.125, fill=False),
    'right_nose_bridge': Ellipse((0.5+0.03, 0.5), 0.04, 0.125, fill=False),
    'left_nostril': Ellipse((0.475, 0.55-0.13), 0.025, 0.025, fill=False),
    'right_nostril': Ellipse((0.525, 0.55-0.13), 0.025, 0.025, fill=False),
    'nose': Ellipse((0.5, 0.55-0.125), 0.05, 0.05, fill=False),
    'top_lip': Ellipse((0.5, 0.55-0.2), 0.15, 0.02, fill=False),
    'bottom_lip': Ellipse((0.5, 0.55-0.225), 0.11, 0.03, fill=False),
    'lip_divit': Ellipse((0.5, 0.55-0.17), 0.03, 0.04, fill=False),
    'left_upper_lip': Ellipse((0.5-0.07, 0.55-0.17), 0.06+0.03, 0.04, fill=False),
    'right_upper_lip': Ellipse((0.5+0.07, 0.55-0.17), 0.06+0.03, 0.04, fill=False),
    'left_bottom_lip': Ellipse((0.5-0.06, 0.55-0.17-0.08), 0.06+0.03, 0.04, fill=False),
    'right_bottom_lip': Ellipse((0.5+0.06, 0.55-0.17-0.08), 0.06+0.03, 0.04, fill=False)
}

# Define the facial features more precisely
features = {
    'left_eye': Ellipse((0.25+0.15, 0.55), 0.1, 0.04, fill=False),
    'right_eye': Ellipse((0.25+0.15+0.2, 0.55), 0.1, 0.04, fill=False),
    'left_iris': Ellipse((0.25+0.15, 0.55), 0.03, 0.03, fill=False),
    'right_iris': Ellipse((0.25+0.15+0.2, 0.55), 0.03, 0.03, fill=False),
    'left_pupil': Ellipse((0.25+0.15, 0.55), 0.025, 0.025, fill=False),
    'right_pupil': Ellipse((0.25+0.15+0.2, 0.55), 0.02, 0.02, fill=False),
    'left_brow': Ellipse((0.25+0.15, 0.6), 0.11, 0.02, fill=False),
    'right_brow': Ellipse((0.25+0.15+0.2, 0.6), 0.11, 0.02, fill=False),
    #'nose_bridge': Ellipse((0.5, 0.51), 0.05, 0.125, fill=False),
    'left_nostril': Ellipse((0.475, 0.55-0.13), 0.025, 0.025, fill=False),
    'right_nostril': Ellipse((0.525, 0.55-0.13), 0.025, 0.025, fill=False),
    'nose': Ellipse((0.5, 0.55-0.125), 0.05, 0.05, fill=False),
    'top_lip': Ellipse((0.5, 0.55-0.2), 0.15, 0.02, fill=False),
    'bottom_lip': Ellipse((0.5, 0.55-0.225), 0.11, 0.03, fill=False),
    'lip_divit': Ellipse((0.5, 0.55-0.17), 0.03, 0.04, fill=False),
}

# Function to fill the background of the face
def fill_face_background(ax, face_ellipse, color):
    face_bg = Ellipse((face_ellipse.center), face_ellipse.width, face_ellipse.height,
                      facecolor=color, edgecolor='none')
    ax.add_patch(face_bg)

# Function to generate rectangles within a shape
def generate_rectangles_in_shape(ax, ellipses, depth=1):
    if depth == 0:
        return

    # Ensure ellipses is a list
    if not isinstance(ellipses, list):
        ellipses = [ellipses]

    new_ellipses = []
    for ellipse in ellipses:
        center, width, height = ellipse.center, ellipse.width, ellipse.height
        half_w, half_h = width * np.random.uniform (0.2, 0.8), height * np.random.uniform (0.2,0.8)
        num_ellipses = 2
        for _ in range(num_ellipses):
            rect_w = np.random.uniform(np.random.uniform (0.5, 1)* half_w, half_w)
            rect_h = np.random.uniform(np.random.uniform (0.5, 1) * half_h, half_h)
            rect_x = np.random.uniform(center[0] - half_w + rect_w / 2, center[0] + half_w - rect_w / 2)
            rect_y = np.random.uniform(center[1] - half_h + rect_h / 2, center[1] + half_h - rect_h / 2)
            color = np.random.choice(['black', 'black'])
            alpha = np.random.uniform(0.333, 0.5)
            rect = Rectangle((rect_x - rect_w / 2, rect_y - rect_h / 2), rect_w, rect_h, facecolor=color, alpha=alpha, edgecolor='none')
            ax.add_patch(rect)
            new_ellipses.append(Ellipse((rect_x, rect_y), rect_w, rect_h, fill=False))

    generate_rectangles_in_shape(ax, new_ellipses, depth - 1)

# Function to generate colored layers in a shape
def generate_colored_layers_in_shape(ax, ellipses, color_palette, depth=1):
    if depth == 0:
        return

    # Ensure ellipses is a list
    if not isinstance(ellipses, list):
        ellipses = [ellipses]

    new_ellipses = []
    for ellipse in ellipses:
        center, width, height = ellipse.center, ellipse.width, ellipse.height
        half_w, half_h = width * np.random.uniform (0.2, 0.8), height * np.random.uniform (0.2, 0.8)
        num_ellipses = 2
        for _ in range(num_ellipses):
            rect_w = np.random.uniform(np.random.uniform (0.5, 1) * half_w, half_w)
            rect_h = np.random.uniform(np.random.uniform (0.5, 1) * half_h, half_h)
            rect_x = np.random.uniform(center[0] - half_w + rect_w / 2, center[0] + half_w - rect_w / 2)
            rect_y = np.random.uniform(center[1] - half_h + rect_h / 2, center[1] + half_h - rect_h / 2)
            color = np.random.choice(color_palette)
            alpha = np.random.uniform(0.1, 0.333)
            rect = Rectangle((rect_x - rect_w / 2, rect_y - rect_h / 2), rect_w, rect_h, facecolor=color, alpha=alpha, edgecolor='none')
            ax.add_patch(rect)
            new_ellipses.append(Ellipse((rect_x, rect_y), rect_w, rect_h, fill=False))

    generate_colored_layers_in_shape(ax, new_ellipses, color_palette, depth - 1)

# Color palettes
background_colors = ['#FFFFFF','#000000']
skin_tones = ['#FFC0CB', '#FFDBAC']
eye_colors = ['#008080']
lip_colors = ['#FFC0CB', '#F08080']
hair_colors = ['#000000', '#8B4513', '#A9A9A9','#FFFFFF']
white = ['#FFFFFF']
black = ['#000000']
black_and_white = ['#000000', '#FFFFFF']

# Color palettes
#skin_tones = ['#FFEEF0', '#FFDBAC', '#8D5524', '#000000']
#eye_colors = ['#8D5524','#008080']
#lip_colors = ['#FFC0CB', '#F08080']
#hair_colors = ['#000000','#FFFFFF', '#8B4513', '#A52A2A', '#A9A9A9']

# Mapping of feature names to color palettes
feature_color_palettes = {
    'left_eye': white, 'right_eye': white,
    'left_iris': black_and_white, 'right_iris': black_and_white,
    'left_pupil': black_and_white, 'right_pupil': black_and_white,
    'left_brow': hair_colors, 'right_brow': hair_colors, #'nose_bridge': skin_tones,
    'left_nostril': skin_tones, 'right_nostril': skin_tones,
    'lip_divit': skin_tones, #'chin': skin_tones,
    'nose': skin_tones, 'top_lip': lip_colors, 'bottom_lip': lip_colors
    #'left_ear': skin_tones, 'right_ear': skin_tones
}

# Updated Update Function for the Animation
def updated_update_animation(frame):
    ax.clear()
    ax.set_aspect('equal', adjustable='box')
    ax.axis('off')
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)


    # Applying face background color
    for background in backgrounds.values():
        fill_face_background(ax, background, "black")

    # Applying background color
    for background in backgrounds.values():
        generate_colored_layers_in_shape(ax, background, black_and_white, depth=2)

    for boundary in face_boundaries.values():
        generate_rectangles_in_shape(ax, boundary, depth=1)
        #generate_rectangles_in_shape(ax, boundary, depth=1)
        #generate_rectangles_in_shape(ax, boundary, depth=3)
    for background in face_backgrounds.values():
        generate_colored_layers_in_shape(ax, background, black_and_white, depth=1)
        generate_colored_layers_in_shape(ax, background, skin_tones, depth=1)
        #generate_colored_layers_in_shape(ax, face, white, depth=1)
        #generate_colored_layers_in_shape(ax, face, white, depth=3)
    #for face in face_backgrounds.values():
        #generate_colored_layers_in_shape(ax, face, skin_tones, depth=1)

    # Applying colored layers with reduced depth
    #for boundary in face_boundaries.values():
        #generate_colored_layers_in_shape(ax, boundary, white, depth=1)
    for boundary in face_boundaries.values():
        #generate_colored_layers_in_shape(ax, boundary, black_and_white, depth=2)
        generate_colored_layers_in_shape(ax, boundary, skin_tones, depth=2)
        #generate_colored_layers_in_shape(ax, boundary, skin_tones, depth=3)
    #for feature_name in features.values():
        #generate_colored_layers_in_shape(ax, feature_name, black, depth=1)
    for hair in hair_boundaries.values():
        generate_colored_layers_in_shape(ax, hair, white, depth=1)
        generate_colored_layers_in_shape(ax, hair, white, depth=1)
        #generate_colored_layers_in_shape(ax, hair, black_and_white, depth=1)
        #generate_colored_layers_in_shape(ax, hair, black_and_white, depth=3)
    #for hair in hair_boundaries.values():
        #generate_rectangles_in_shape(ax, black_and_white, depth=3)
        #generate_rectangles_in_shape(ax, boundary, depth=3)
        #generate_rectangles_in_shape(ax, boundary, depth=3)
        #generate_rectangles_in_shape(ax, boundary, depth=2)
        #generate_rectangles_in_shape(ax, boundary, depth=1)

    #for boundary in face_boundaries.values():
        #generate_rectangles_in_shape(ax, boundary, depth=2)
        #generate_rectangles_in_shape(ax, boundary, depth=1)
        #generate_rectangles_in_shape(ax, boundary, depth=3)

    for boundary in face_boundaries.values():
        #generate_colored_layers_in_shape(ax, boundary, black_and_white, depth=2)
        generate_colored_layers_in_shape(ax, boundary, black_and_white, depth=1)
        #generate_colored_layers_in_shape(ax, boundary, skin_tones, depth=3)
    for feature_shape in features.values():
        #generate_rectangles_in_shape(ax, feature_shape, depth=2)
        #generate_rectangles_in_shape(ax, feature_shape, depth=3)
        #generate_rectangles_in_shape(ax, feature_shape, depth=1)
        #generate_rectangles_in_shape(ax, feature_shape, depth=4)
        generate_rectangles_in_shape(ax, feature_shape, depth=3)

    for feature_name, feature_shape in features.items():
        generate_colored_layers_in_shape(ax, feature_shape, feature_color_palettes[feature_name], depth=1)
        #generate_colored_layers_in_shape(ax, feature_shape, feature_color_palettes[feature_name], depth=2)
        #generate_colored_layers_in_shape(ax, feature_shape, feature_color_palettes[feature_name], depth=3)

    # Applying background color
    for background in backgrounds.values():
        generate_colored_layers_in_shape(ax, background, black_and_white, depth=1)
        #generate_colored_layers_in_shape(ax, background, black_and_white, depth=1)
        #generate_colored_layers_in_shape(ax, background, black_and_white, depth=3)

    return ax

# Create and Save the Animation
fig, apx = plt.subplots(figsize=(5, 5), dpi=100)
num_frames = 8  # Number of frames in the animation

# The rest of your code remains the same
updated_animation = FuncAnimation(fig, updated_update_animation, frames=num_frames, interval=300)
updated_animation_path = 'updated_integrated_portrait_animation.gif'
updated_animation.save(updated_animation_path, writer='Pillow', fps=2)
plt.close(fig)

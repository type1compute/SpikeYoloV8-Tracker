#!/usr/bin/env python3
"""
Generate a clean model architecture diagram for eTraM SpikeYOLO with Tracking.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# Set up the figure with very high resolution - larger canvas to prevent overlap
fig = plt.figure(figsize=(32, 60), dpi=400)
ax = fig.add_subplot(111)
ax.set_xlim(0, 120)  # Wider canvas for better horizontal spacing
ax.set_ylim(0, 200)  # Extended canvas height for better spacing
ax.axis('off')

# Color scheme
colors = {
    'input': '#E3F2FD',  # Light blue
    'backbone': '#C8E6C9',  # Light green
    'detection': '#FFF9C4',  # Light yellow
    'text': '#212121',  # Dark gray
    'arrow': '#424242',  # Medium gray
    'border': '#757575'  # Light gray
}

# Font sizes - optimized for clarity without overlap
title_font = 32
section_font = 22
detail_font = 18
small_font = 15

def draw_box(x, y, width, height, text, color, fontsize=detail_font, text_color='black', alpha=0.8):
    """Draw a rounded rectangle box with text."""
    box = FancyBboxPatch(
        (x, y), width, height,
        boxstyle="round,pad=0.4",
        facecolor=color,
        edgecolor=colors['border'],
        linewidth=2.5,
        alpha=alpha
    )
    ax.add_patch(box)
    # Split text into lines for better readability
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    if not lines:
        return
    total_height = height
    line_height = total_height / len(lines) if len(lines) > 0 else total_height
    for i, line in enumerate(lines):
        # Calculate vertical position with padding
        y_pos_text = y + height - (i + 0.5) * line_height
        ax.text(x + width/2, y_pos_text, line, 
                ha='center', va='center', fontsize=fontsize, 
                color=text_color, weight='bold' if fontsize >= section_font else 'normal',
                wrap=False)  # Disable wrapping to prevent overlap

def draw_arrow(x1, y1, x2, y2, color=colors['arrow'], style='->', linewidth=3):
    """Draw an arrow between two points."""
    arrow = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle=style,
        color=color,
        linewidth=linewidth,
        mutation_scale=25,
        alpha=0.9
    )
    ax.add_patch(arrow)

# Title
ax.text(60, 198, 'eTraM SpikeYOLO Architecture', 
        ha='center', va='top', fontsize=title_font, weight='bold', color=colors['text'])

# ==================== INPUT ====================
y_pos = 193
box_height = 3.2
draw_box(40, y_pos, 40, box_height, 'Input Events\n[B, T, H, W]\n[batch, 8, 720, 1280]', 
         colors['input'], fontsize=detail_font)
# Arrow: start below current box, end above next box
# Current box bottom: y_pos, next box will be at y_pos-6 with height 3, so top is y_pos-6+3 = y_pos-3
# Add 0.5 gap: start at y_pos-0.5, end at y_pos-3-0.5 = y_pos-3.5
draw_arrow(60, y_pos-0.5, 60, y_pos-3.5)

# ==================== INPUT LAYER ====================
y_pos -= 6
box_height = 3
draw_box(40, y_pos, 40, box_height, 'Input Layer\n(Identity)\nPass-through', 
         colors['input'], fontsize=detail_font)
# Next box at y_pos-6 with height 3.5, top is y_pos-6+3.5 = y_pos-2.5
draw_arrow(60, y_pos-0.5, 60, y_pos-2.5-0.5)

# ==================== TEMPORAL EXPANSION ====================
y_pos -= 6
box_height = 3.5
draw_box(40, y_pos, 40, box_height, 'Temporal & Channel Expansion\nSplit ON/OFF Channels\n[T, B, 2, H, W]\n[8, batch, 2, 720, 1280]', 
         colors['input'], fontsize=detail_font)
# Next section starts at y_pos-6, then block 0 at y_pos-6-4 = y_pos-10 with height 3, top is y_pos-10+3 = y_pos-7
draw_arrow(60, y_pos-0.5, 60, y_pos-7-0.5)

# ==================== BACKBONE ====================
y_pos -= 6
ax.text(60, y_pos+2, 'BACKBONE (9 Sequential Blocks)', 
        ha='center', va='top', fontsize=section_font, weight='bold', color=colors['text'])

# Block 0
y_pos -= 4
box_height = 3
draw_box(35, y_pos, 50, box_height, 'Block 0: MS_DownSampling\n2 → 128 channels, stride=4\n[8, B, 2, 720, 1280] → [8, B, 128, 180, 320]', 
         colors['backbone'], fontsize=small_font)
# Next box at y_pos-5 with height 3, top is y_pos-5+3 = y_pos-2
draw_arrow(60, y_pos-0.5, 60, y_pos-2-0.5)

# Block 1
y_pos -= 5
draw_box(35, y_pos, 50, box_height, 'Block 1: MS_AllConvBlock\n128 channels\n[8, B, 128, 180, 320]', 
         colors['backbone'], fontsize=small_font)
draw_arrow(60, y_pos-0.5, 60, y_pos-2-0.5)

# Block 2
y_pos -= 5
draw_box(35, y_pos, 50, box_height, 'Block 2: MS_DownSampling\n128 → 256 channels, stride=2\n[8, B, 128, 180, 320] → [8, B, 256, 90, 160]', 
         colors['backbone'], fontsize=small_font)
draw_arrow(60, y_pos-0.5, 60, y_pos-2-0.5)

# Block 3 - Save position for P3 fusion connection
y_pos -= 5
block3_y = y_pos
draw_box(35, y_pos, 50, box_height, 'Block 3: MS_AllConvBlock\n256 channels (P3 Backbone)\n[8, B, 256, 90, 160]', 
         colors['backbone'], fontsize=small_font)
draw_arrow(60, y_pos-0.5, 60, y_pos-2-0.5)

# Block 4
y_pos -= 5
draw_box(35, y_pos, 50, box_height, 'Block 4: MS_DownSampling\n256 → 512 channels, stride=2\n[8, B, 256, 90, 160] → [8, B, 512, 45, 80]', 
         colors['backbone'], fontsize=small_font)
draw_arrow(60, y_pos-0.5, 60, y_pos-2-0.5)

# Block 5 - Save position for P4 fusion connection
y_pos -= 5
block5_y = y_pos
draw_box(35, y_pos, 50, box_height, 'Block 5: MS_ConvBlock\n512 channels (P4 Backbone)\n[8, B, 512, 45, 80]', 
         colors['backbone'], fontsize=small_font)
draw_arrow(60, y_pos-0.5, 60, y_pos-2-0.5)

# Block 6
y_pos -= 5
draw_box(35, y_pos, 50, box_height, 'Block 6: MS_DownSampling\n512 → 1024 channels, stride=2\n[8, B, 512, 45, 80] → [8, B, 1024, 23, 40]', 
         colors['backbone'], fontsize=small_font)
draw_arrow(60, y_pos-0.5, 60, y_pos-2-0.5)

# Block 7
y_pos -= 5
draw_box(35, y_pos, 50, box_height, 'Block 7: MS_ConvBlock\n1024 channels\n[8, B, 1024, 23, 40]', 
         colors['backbone'], fontsize=small_font)
draw_arrow(60, y_pos-0.5, 60, y_pos-2-0.5)

# Block 8 - Save position for P5 connection
y_pos -= 5
block8_y = y_pos
draw_box(35, y_pos, 50, box_height, 'Block 8: SpikeSPPF\n1024 channels (P5 Backbone)\nkernel=5\n[8, B, 1024, 23, 40]', 
         colors['backbone'], fontsize=small_font)
# Extra gap before detection heads section (y_pos-6 for section title, then y_pos-6-4 = y_pos-10 for boxes)
# Detection head boxes at y_pos-10 with height 5, top is y_pos-10+5 = y_pos-5
draw_arrow(60, y_pos-0.5, 60, y_pos-5-0.5)

# ==================== DETECTION HEADS ====================
y_pos -= 6
ax.text(60, y_pos+2, 'DETECTION HEADS (Multi-Scale - FPN Style)', 
        ha='center', va='top', fontsize=section_font, weight='bold', color=colors['text'])

y_pos -= 4
# P5, P4, P3 in SEQUENCE (vertical stack) - centered horizontally
# Each box width 50, centered at x=60 (so x=35, width=50)
# P5 Head (top)
p5_y = y_pos
draw_box(35, p5_y, 50, 5, 'P5 Head\nInput: [8, B, 1024, 23, 40]\n1024 → 512 channels\nOutput: [8, B, 512, 23, 40]\nSpatial: 23×40 (920 anchors)', 
         colors['detection'], fontsize=small_font)

# Arrow from Block 8 (P5 backbone) to P5
# Block 8 bottom is at block8_y + box_height = block8_y + 3
# P5 top is at p5_y + 5
draw_arrow(60, block8_y + 3 - 0.5, 60, p5_y + 5)
ax.text(60+3, (block8_y + 3 + p5_y + 5)/2, 'P5 Backbone', ha='left', va='center', fontsize=small_font, 
        color=colors['arrow'], style='italic', weight='bold')

# P4 Head (middle) - with FPN upsampling from P5
p4_y = p5_y - 6.5  # Gap between P5 and P4
draw_box(35, p4_y, 50, 5, 'P4 Head (FPN)\nUpsample P5 (2×)\nFuse with Backbone P4\nOutput: [8, B, 256, 45, 80]\nSpatial: 45×80 (3600 anchors)', 
         colors['detection'], fontsize=small_font)

# Arrow from P5 to P4 (vertical, with upsampling indicator)
draw_arrow(60, p5_y-0.5, 60, p4_y+5)
ax.text(60+3, (p5_y + p4_y+5)/2, '2× Upsample', ha='left', va='center', fontsize=small_font, 
        color=colors['arrow'], weight='bold', style='italic')

# Arrow from Block 5 (P4 backbone) to P4 (for fusion) - dashed
# Block 5 bottom is at block5_y + box_height = block5_y + 3
# P4 top is at p4_y + 5
# Draw diagonal arrow from Block 5 to P4
p4_fusion_arrow = FancyArrowPatch(
    (60, block5_y + 3 - 0.5), (60, p4_y + 5),
    arrowstyle='->', color=colors['arrow'], linewidth=2.5,
    linestyle='--', alpha=0.7
)
ax.add_patch(p4_fusion_arrow)
ax.text(60+3, (block5_y + 3 + p4_y + 5)/2, 'P4 Backbone\nFuse', ha='left', va='center', fontsize=small_font, 
        color=colors['arrow'], style='italic', weight='bold')

# P3 Head (bottom) - with FPN upsampling from P4
p3_y = p4_y - 6.5  # Gap between P4 and P3
draw_box(35, p3_y, 50, 5, 'P3 Head (FPN)\nUpsample P4 (2×)\nFuse with Backbone P3\nOutput: [8, B, 128, 90, 160]\nSpatial: 90×160 (14400 anchors)', 
         colors['detection'], fontsize=small_font)

# Arrow from P4 to P3 (vertical, with upsampling indicator)
draw_arrow(60, p4_y-0.5, 60, p3_y+5)
ax.text(60+3, (p4_y + p3_y+5)/2, '2× Upsample', ha='left', va='center', fontsize=small_font, 
        color=colors['arrow'], weight='bold', style='italic')

# Arrow from Block 3 (P3 backbone) to P3 (for fusion) - dashed
# Block 3 bottom is at block3_y + box_height = block3_y + 3
# P3 top is at p3_y + 5
# Draw diagonal arrow from Block 3 to P3
p3_fusion_arrow = FancyArrowPatch(
    (60, block3_y + 3 - 0.5), (60, p3_y + 5),
    arrowstyle='->', color=colors['arrow'], linewidth=2.5,
    linestyle='--', alpha=0.7
)
ax.add_patch(p3_fusion_arrow)
ax.text(60+3, (block3_y + 3 + p3_y + 5)/2, 'P3 Backbone\nFuse', ha='left', va='center', fontsize=small_font, 
        color=colors['arrow'], style='italic', weight='bold')

# Update y_pos to bottom of P3 for next section
y_pos = p3_y

# Arrows from heads to detection module - each head connects to its corresponding scale
# Store current y_pos before detection module section
detection_module_y_start = y_pos
# Detection module: y_pos -= 7 for title, y_pos -= 4 for boxes
# So detection module boxes are at: detection_module_y_start - 7 - 4 = detection_module_y_start - 11
# Boxes have height 5, so top is at: (detection_module_y_start - 11) + 5 = detection_module_y_start - 6
# Detection module scales: P5 at x=5 (center 20), P4 at x=40 (center 55), P3 at x=75 (center 90)
# All heads are centered at x=60, so we need diagonal arrows
# P5 Head (x=60, y=p5_y) → P5 Scale top (x=20, y=detection_module_y_start-6)
draw_arrow(60, p5_y-0.5, 20, detection_module_y_start-6)
# P4 Head (x=60, y=p4_y) → P4 Scale top (x=55, y=detection_module_y_start-6)
draw_arrow(60, p4_y-0.5, 55, detection_module_y_start-6)
# P3 Head (x=60, y=p3_y) → P3 Scale top (x=90, y=detection_module_y_start-6)
draw_arrow(60, p3_y-0.5, 90, detection_module_y_start-6)

# ==================== DETECTION MODULE ====================
y_pos -= 7  # Increased gap before detection module
ax.text(60, y_pos+2, 'DETECTION MODULE (SpikeDetectWithTracking)', 
        ha='center', va='top', fontsize=section_font, weight='bold', color=colors['text'])

y_pos -= 4
# Three branches for each scale - better spacing and layout aligned with Detection Heads
# P5 Scale at x=5 (matching P5 Head), P4 at x=40, P3 at x=75, each with width 30
# P5 Scale
draw_box(5, y_pos, 30, 5, 'P5 Scale\nInput: [8, B, 512, 23, 40]', 
         colors['detection'], fontsize=small_font, alpha=0.6)

draw_box(5, y_pos-2, 9, 1.3, 'Box (cv2)\n64 DFL', 
         colors['detection'], fontsize=small_font)
draw_box(14.5, y_pos-2, 9, 1.3, 'Class (cv3)\n3 classes', 
         colors['detection'], fontsize=small_font)
draw_box(24, y_pos-2, 6, 1.3, 'Track\n(cv4)\n128 dim', 
         colors['detection'], fontsize=small_font)

draw_box(5, y_pos-3.8, 30, 1.3, 'Output: [8, B, 920, 67]\nTrack: [8, B, 920, 128]', 
         colors['detection'], fontsize=small_font)

# P4 Scale
draw_box(40, y_pos, 30, 5, 'P4 Scale\nInput: [8, B, 256, 45, 80]', 
         colors['detection'], fontsize=small_font, alpha=0.6)

draw_box(40, y_pos-2, 9, 1.3, 'Box (cv2)\n64 DFL', 
         colors['detection'], fontsize=small_font)
draw_box(49.5, y_pos-2, 9, 1.3, 'Class (cv3)\n3 classes', 
         colors['detection'], fontsize=small_font)
draw_box(59, y_pos-2, 6, 1.3, 'Track\n(cv4)\n128 dim', 
         colors['detection'], fontsize=small_font)

draw_box(40, y_pos-3.8, 30, 1.3, 'Output: [8, B, 3600, 67]\nTrack: [8, B, 3600, 128]', 
         colors['detection'], fontsize=small_font)

# P3 Scale
draw_box(75, y_pos, 30, 5, 'P3 Scale\nInput: [8, B, 128, 90, 160]', 
         colors['detection'], fontsize=small_font, alpha=0.6)

draw_box(75, y_pos-2, 9, 1.3, 'Box (cv2)\n64 DFL', 
         colors['detection'], fontsize=small_font)
draw_box(84.5, y_pos-2, 9, 1.3, 'Class (cv3)\n3 classes', 
         colors['detection'], fontsize=small_font)
draw_box(94, y_pos-2, 6, 1.3, 'Track\n(cv4)\n128 dim', 
         colors['detection'], fontsize=small_font)

draw_box(75, y_pos-3.8, 30, 1.3, 'Output: [8, B, 14400, 67]\nTrack: [8, B, 14400, 128]', 
         colors['detection'], fontsize=small_font)

# ==================== OUTPUT ====================
# Detection module output boxes are at y_pos-3.8 with height 1.3, so bottom is y_pos-5.1
# Store previous y_pos for arrow calculations
prev_y_pos = y_pos
y_pos -= 7  # Increased gap before output section
# Output box is at y_pos-6 with height 3.5, so top is y_pos-6+3.5 = y_pos-2.5
# Output box center is at x=60 (35 + 50/2 = 60)
# Arrows start from detection module output bottom: prev_y_pos-5.1-0.5 = prev_y_pos-5.6
# Arrows end at output box top: y_pos-2.5 (top of the box)
draw_arrow(20, prev_y_pos-5.6, 60, y_pos-2.5)
draw_arrow(55, prev_y_pos-5.6, 60, y_pos-2.5)
draw_arrow(90, prev_y_pos-5.6, 60, y_pos-2.5)

draw_box(35, y_pos-6, 50, 3.5, 'Final Output\nP5: [8, B, 920, 67] + [8, B, 920, 128]\nP4: [8, B, 3600, 67] + [8, B, 3600, 128]\nP3: [8, B, 14400, 67] + [8, B, 14400, 128]', 
         colors['detection'], fontsize=detail_font)

# Footer
ax.text(60, 2, 'eTraM SpikeYOLO Architecture v2.1 (2-Channel ON/OFF Input)', 
        ha='center', va='bottom', fontsize=14, style='italic', color='gray')

plt.tight_layout()
# Save with very high DPI for maximum clarity - increased to 600 DPI
plt.savefig('architecture_diagram.png', dpi=600, bbox_inches='tight', facecolor='white', pad_inches=0.3)
plt.savefig('architecture_diagram.pdf', bbox_inches='tight', facecolor='white', pad_inches=0.3)
print("Architecture diagram saved as 'architecture_diagram.png' (600 DPI) and 'architecture_diagram.pdf'")

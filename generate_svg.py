import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Set up figure size and canvas dimensions
fig, ax = plt.subplots(figsize=(14, 8), dpi=300)
ax.axis('off')
ax.set_xlim(0, 14)
ax.set_ylim(0, 8)

# Color palette definition matching research standards
COLOR_BLUE_BG = '#E8EAF6'
COLOR_BLUE_BORDER = '#3F51B5'
COLOR_BLUE_TEXT = '#1A237E'

COLOR_GRAY_BG = '#F5F5F5'
COLOR_GRAY_BORDER = '#757575'
COLOR_GRAY_TEXT = '#212121'

COLOR_ORANGE_BG = '#FFF3E0'
COLOR_ORANGE_BORDER = '#E65100'
COLOR_ORANGE_TEXT = '#BF360C'

COLOR_PURPLE_BG = '#F3E5F5'
COLOR_PURPLE_BORDER = '#7B1FA2'
COLOR_PURPLE_TEXT = '#4A148C'

# Helper function to draw rounded boxes with labels and details
def draw_box(x, y, w, h, title, details, bg, border, text_color):
    rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1,rounding_size=0.15",
                                 facecolor=bg, edgecolor=border, linewidth=2)
    ax.add_patch(rect)
    ax.text(x + w/2, y + h*0.68, title, ha='center', va='center', fontsize=9.5, fontweight='bold', color=text_color)
    ax.text(x + w/2, y + h*0.32, details, ha='center', va='center', fontsize=7.5, color='#424242')

# Helper function for drawing directional flow arrows
def draw_arrow(start, end, text=""):
    ax.annotate(text, xy=end, xytext=start,
                arrowprops=dict(arrowstyle='->', lw=1.8, color='#333333', mutation_scale=14),
                ha='center', va='center', fontsize=7.5, color='#616161')

# Draw Sub-system Containers / Grouping Boxes
rect_stage1 = patches.FancyBboxPatch((0.3, 4.3), 13.4, 3.4, boxstyle="round,pad=0.1,rounding_size=0.1",
                                     facecolor='#FAFAFA', edgecolor='#E0E0E0', linewidth=1.5, linestyle='--')
ax.add_patch(rect_stage1)
ax.text(0.6, 7.4, "STAGE 1 & 2: INPUT VALIDATION & FEATURE EXTRACTION PIPELINE", fontsize=9, fontweight='bold', color='#9E9E9E')

rect_stage2 = patches.FancyBboxPatch((0.3, 0.3), 13.4, 3.5, boxstyle="round,pad=0.1,rounding_size=0.1",
                                     facecolor='#FAFAFA', edgecolor='#E0E0E0', linewidth=1.5, linestyle='--')
ax.add_patch(rect_stage2)
ax.text(0.6, 3.5, "STAGE 3 & 4: CLASSIFICATION HEAD & EXPLAINABLE AI (GRAD-CAM)", fontsize=9, fontweight='bold', color='#9E9E9E')

# --- TOP ROW: STAGES 1 & 2 (Left to Right) ---
draw_box(0.6, 5.0, 2.2, 1.8, "Raw CT Scan", "Abdominal DICOM/PNG\n(2D Slice)", COLOR_BLUE_BG, COLOR_BLUE_BORDER, COLOR_BLUE_TEXT)
draw_arrow((2.8, 5.9), (3.4, 5.9))

draw_box(3.4, 5.0, 2.2, 1.8, "Input Guardrail", "OOD Quality Filter &\nAspect Verification", COLOR_GRAY_BG, COLOR_GRAY_BORDER, COLOR_GRAY_TEXT)
draw_arrow((5.6, 5.9), (6.2, 5.9))

draw_box(6.2, 5.0, 2.3, 1.8, "Preprocessing", "Min-Max Rescaling &\nResize to (224x224x3)", COLOR_GRAY_BG, COLOR_GRAY_BORDER, COLOR_GRAY_TEXT)
draw_arrow((8.5, 5.9), (9.1, 5.9))

draw_box(9.1, 5.0, 2.4, 1.8, "MobileNetV2", "Inverted Residuals &\nDepthwise Convolutions", COLOR_ORANGE_BG, COLOR_ORANGE_BORDER, COLOR_ORANGE_TEXT)
draw_arrow((10.3, 5.0), (10.3, 2.8))

# --- BOTTOM ROW: STAGES 3 & 4 (Right to Left) ---
draw_box(9.1, 1.0, 2.4, 1.8, "Target Conv Layer", "Extracted Feature Map\n(out_relu: 7x7x1280)", COLOR_ORANGE_BG, COLOR_ORANGE_BORDER, COLOR_ORANGE_TEXT)
draw_arrow((9.1, 1.9), (8.5, 1.9))

draw_box(6.2, 1.0, 2.3, 1.8, "Classifier Head", "GAP (1280-d Vector) +\nDropout (p=0.5) + Dense", COLOR_GRAY_BG, COLOR_GRAY_BORDER, COLOR_GRAY_TEXT)
draw_arrow((6.2, 1.9), (5.6, 1.9))

draw_box(3.4, 1.0, 2.2, 1.8, "Grad-CAM XAI", "Gradient Computation &\nSpatial Heatmap", COLOR_PURPLE_BG, COLOR_PURPLE_BORDER, COLOR_PURPLE_TEXT)
draw_arrow((3.4, 1.9), (2.8, 1.9))

draw_box(0.6, 1.0, 2.2, 1.8, "Diagnostic Result", "Tumor / Normal Class +\nOverlay Heatmap", COLOR_BLUE_BG, COLOR_BLUE_BORDER, COLOR_BLUE_TEXT)

plt.tight_layout()

# Save output directly as vector .svg
svg_path = 'architecture_diagram.svg'
plt.savefig(svg_path, format='svg', bbox_inches='tight')
plt.close()

print(f"Saved vector file: {svg_path}")
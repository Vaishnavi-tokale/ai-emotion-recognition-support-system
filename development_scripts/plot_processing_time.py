import matplotlib.pyplot as plt

# ===========================
# Average Processing Time (ms)
# ===========================

modules = [
    "OpenCV",
    "CNN\n(VGG19)",
    "MediaPipe",
    "LSTM",
    "Response\nGeneration",
    "TTS"
]

processing_time = [
    22,
    48,
    28,
    41,
    18,
    47
]

# ===========================
# Plot
# ===========================

plt.figure(figsize=(9,6))

bars = plt.bar(
    modules,
    processing_time,
    color="steelblue",
    edgecolor="black"
)

# Display values on bars
for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width()/2,
        height + 1,
        f"{height}",
        ha='center',
        fontsize=11,
        fontweight='bold'
    )

plt.title(
    "Average Processing Time of Proposed AI Therapist Modules",
    fontsize=16,
    fontweight="bold"
)

plt.xlabel("Modules", fontsize=13)
plt.ylabel("Processing Time (ms)", fontsize=13)

plt.ylim(0, 60)

plt.grid(axis="y", linestyle="--", alpha=0.4)

plt.tight_layout()

# Save figure
plt.savefig(
    "processing_time_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("Graph saved as processing_time_comparison.png")
import numpy as np

# Colours and types

Types = np.array(["T4", "T5"])
Type_colours = np.array(["#17becf", "#ff7f0e"])
Subtypes = np.array(["T4a", "T4b", "T4c", "T4d", "T5a", "T5b", "T5c", "T5d"])
Subtype_colours = np.array(
    [
        "#1f77b4",
        "#9edae5",
        "#98df8a",
        "#bcbd22",
        "#d62728",
        "#ffbb78",
        "#ff9896",
        "#9467bd",
    ]
)


def subtype_contour_plot(ax, dictionary):
    for i in range(len(Subtypes)):

        s = Subtypes[i]

        x = dictionary[s]["x"]
        y = dictionary[s]["y"]
        xx = dictionary[s]["xx"]
        yy = dictionary[s]["yy"]
        zz = dictionary[s]["zz"]

        # Plot the contour
        ax.contour(xx, yy, zz, levels=5, cmap="viridis")
        ax.scatter(
            x, y, s=0.3, color="gray", alpha=0.01, rasterized=True
        )  # Optional: plot original points
        ax.scatter(
            dictionary["offsets"][i][0], dictionary["offsets"][i][1], c="r", zorder=500
        )

    ax.set_aspect("equal")

    ax.grid(True)
    ax.set_axis_off()
    add_scale_bar(ax, length=1, label=r"$1 \sigma$", location = 'lower left')
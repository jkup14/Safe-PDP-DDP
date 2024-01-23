import matplotlib.pyplot as plt
import matplotlib
import numpy as np
# import systems_constraints.distance_2d
import plotly.express as px
import plotly.graph_objects as go
import jax.numpy as jnp
import jax


def plot2dvehicle_multi(
    n_agents,
    n1,
    delta_collide,
    x0,
    xf,
    times,
    X,
    fig=None,
    ax=None,
    color=None,
    label=None,
    same=False,
    delta_connect=None,
    n_teams=None,
    verbose=False,
):
    if not fig:
        fig, ax = plt.subplots(1, subplot_kw={"aspect": "equal"})

    for ii in range(n_agents):
        if color is None:
            colori = colors[ii]
        else:
            colori = colors[color]
        labeli = None
        if same and ii == 0:
            labeli = label
        ax.plot(X[:, ii * n1], X[:, ii * n1 + 1], color=colori, label=labeli)
        ax.plot(x0[ii * n1], x0[ii * n1 + 1], color=colori, marker="o")
        # ax.plot(X[ii * n1, 0], X[ii * n1 + 1, 0], color=colors[ii], marker='o')
        # ax.plot(X[ii * n1, -1], X[ii * n1 + 1, -1], color=colors[ii], marker='o')
        ax.plot(xf[ii * n1], xf[ii * n1 + 1], color=colori, marker="X")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    # ax.set_xticks([-10, -7.5, -5, -2.5, 0, 2.5, 5, 7.5, 10])
    # ax.set_yticks([-10, -7.5, -5, -2.5, 0, 2.5, 5, 7.5, 10])
    # ax[0].set_xlim([-agent_radius-1, agent_radius+1])
    # ax[0].set_ylim([-agent_radius-1, agent_radius+1])
    collided = False
    disconnected = False
    time_collide = []
    time_disconnect = []
    collided_agents = []
    disconnected_agents = []
    for ii in range(n_agents):
        for jj in range(ii + 1, n_agents):
            dists_ij = np.sqrt(
                np.sum(
                    (X[:, ii * n1 : ii * n1 + 2] - X[:, jj * n1 : jj * n1 + 2]) ** 2, 1
                )
            )
            # ax[1].plot(T, dists_ij)
            if (np.array(dists_ij) < delta_collide).any():
                collided = True
                time_collide.append(
                    times[np.where((np.array(dists_ij) < delta_collide))]
                )
                collided_agents.append((ii, jj))
            if int(ii / int(n_agents / n_teams)) == int(jj / int(n_agents / n_teams)):
                if (np.array(dists_ij) > delta_connect).any():
                    disconnected = True
                    time_disconnect.append(
                        times[np.where((np.array(dists_ij) > delta_connect))]
                    )
                    disconnected_agents.append((ii, jj))

    if verbose:
        if collided:
            for i, (ii, jj) in enumerate(collided_agents):
                dists_ij = np.sqrt(
                    np.sum(
                        (X[:, ii * n1 : ii * n1 + 2] - X[:, jj * n1 : jj * n1 + 2])
                        ** 2,
                        1,
                    )
                )
                print(
                    "Agents",
                    ii,
                    "and",
                    jj,
                    "Collided at time = ",
                    time_collide[i],
                    "with distance",
                    dists_ij[np.where((np.array(dists_ij) < delta_collide))],
                )
        else:
            print("Did not collide")
    if verbose:
        if disconnected:
            for i, (ii, jj) in enumerate(disconnected_agents):
                dists_ij = np.sqrt(
                    np.sum(
                        (X[:, ii * n1 : ii * n1 + 2] - X[:, jj * n1 : jj * n1 + 2])
                        ** 2,
                        1,
                    )
                )
                print(
                    "Agents",
                    ii,
                    "and",
                    jj,
                    "Disconnected at time = ",
                    time_disconnect[i],
                    "with distance",
                    dists_ij[np.where((np.array(dists_ij) > delta_connect))],
                )
        else:
            print("Did not Disconnect")
    return fig, ax


# def plot2dvehicle(x0, xf, T, X, U, class_constraint):
#     if class_constraint and isinstance(
#         class_constraint[0], systems_constraints.distance_2d.Distance2D
#     ):
#         n_agents = class_constraint[0].n_agents
#         n1 = class_constraint[0].n1
#         delta = class_constraint[0].d
#         fig, ax = plot2dvehicle_multi(n_agents, n1, delta, x0, xf, T, X)
#     else:
#         fig, ax = plt.subplots(1, subplot_kw={"aspect": "equal"})
#         ax.plot(X[0, :], X[1, :])
#         ax.plot(x0[0], x0[1], "bo")
#         ax.plot(X[0, 0], X[1, 0], "bo")
#         ax.plot(X[0, -1], X[1, -1], "ro")
#         ax.plot(xf[0], xf[1], "go")

#     for i in range(len(class_constraint)):
#         if hasattr(class_constraint[i], "obstacles"):
#             obstacles_info = class_constraint[i].obstacles()
#             ox = obstacles_info[:, 0]
#             oy = obstacles_info[:, 1]
#             r = obstacles_info[:, 2]
#             for ii in range(obstacles_info.shape[0]):
#                 ax.add_patch(plt.Circle((ox[ii], oy[ii]), r[ii], color="k", alpha=0.75))

#     # fig2, ax2 = plt.subplots(1)
#     # ax2.plot(T[0:-1], U.T)
#     return fig, ax


def animate2dVehicle_Multi(n_agents, n1, delta, x0, xf, T, X, U):
    fig2, ax2 = plt.subplots(1)
    ax2.plot(T[0:-1], U.T)

    collided = False
    disconnected = False
    time_collide = []
    time_disconnect = []
    collided_agents = []
    disconnected_agents = []
    for ii in range(n_agents):
        for jj in range(ii + 1, n_agents):
            dists_ij = np.sqrt(
                np.sum(
                    (X[ii * n1 : ii * n1 + 2, :] - X[jj * n1 : jj * n1 + 2, :]) ** 2, 0
                )
            )
            if (np.array(dists_ij) < delta).any():
                collided = True
                time_collide.append(T[np.where((np.array(dists_ij) < delta))])
                collided_agents.append((ii, jj))

    if collided:
        for i, (ii, jj) in enumerate(collided_agents):
            dists_ij = np.sqrt(
                np.sum(
                    (X[ii * n1 : ii * n1 + 2, :] - X[jj * n1 : jj * n1 + 2, :]) ** 2, 0
                )
            )
            print(
                "Agents",
                ii,
                "and",
                jj,
                "Collided at time = ",
                time_collide[i],
                "with distance",
                dists_ij[np.where((np.array(dists_ij) < delta))],
            )
    else:
        print("Did not collide")

    drawing_dt = 0.1
    dt = T[1] - T[0]
    step = int(drawing_dt / dt)
    indices = np.array(list(range(len(T))[::step]) + [len(T) - 1])

    data = dict()
    data["t"] = np.array([])
    data["x"] = np.array([])
    data["y"] = np.array([])
    data["type"] = []
    data["size"] = []
    category_orders = []
    for n in range(n_agents):
        data["t"] = np.hstack([data["t"], T[indices]])
        data["x"] = np.hstack([data["x"], X[n * n1, indices]])
        data["y"] = np.hstack([data["y"], X[n * n1 + 1, indices]])
        data["type"] = data["type"] + ["Agent " + str(n)] * len(indices)
        category_orders = category_orders + ["Agent " + str(n)]

    minx, maxx = np.min(data["x"]), np.max(data["x"])
    miny, maxy = np.min(data["y"]), np.max(data["y"])
    xrange = 2 + maxx - minx
    yrange = 2 + maxy - miny
    xfactor = 480  # needs to be tuned for your plot
    yfactor = 480  # needs to be tuned for your plot
    plotter_aspect_ratio = 2  # width/height
    for n in range(n_agents):  # To make marker size accurate
        if xrange / yrange >= plotter_aspect_ratio:
            data["size"] = data["size"] + [
                delta * 1 / (2 + maxx - minx) * xfactor
            ] * len(indices)
        else:
            data["size"] = data["size"] + [
                delta * 1 / (2 + maxy - miny) * yfactor
            ] * len(indices)

    fig = px.scatter(
        data,
        x="x",
        y="y",
        animation_frame="t",
        animation_group="type",
        color="type",
        category_orders={"type": category_orders},
        color_discrete_sequence=colorsHex[0:n_agents],
        size="size",
        size_max=data["size"][-1],
        hover_name="type",
        template="plotly_white",
        range_x=(-3, 3),
        range_y=(-3, 3),
        height=750,
        width=750,
    )
    # Make equal one meter grid
    fig.update_xaxes(dtick=1.0, showline=True)
    fig.update_yaxes(scaleanchor="x", scaleratio=1, showline=False, dtick=1.0)

    for n in range(n_agents):
        subject_line = px.line(x=X[n * n1, :], y=X[n * n1 + 1, :]).data[0]
        subject_line.line["color"] = colorsHex[n]
        subject_line.line["width"] = 1
        fig.add_trace(subject_line)
        xf_i = go.Scatter(
            x=xf[n * n1],
            y=xf[n * n1 + 1],
            marker_color=colorsHex[n],
            marker_symbol="x",
            marker_size=12,
        )
        fig.add_trace(xf_i)
        x0_i = go.Scatter(
            x=x0[n * n1],
            y=x0[n * n1 + 1],
            marker_color=colorsHex[n],
            marker_symbol="circle-open",
            marker_size=17,
        )
        fig.add_trace(x0_i)

    fig.show()
    return collided, fig2


def animate2dVehicle_Multi_Connect(
    n_agents, n1, delta_collide, delta_connect, n_teams, x0, xf, T, X, U
):
    fig2, ax2 = plt.subplots(1)
    ax2.plot(T[0:-1], U.T)
    collided = False
    disconnected = False
    time_collide = []
    time_disconnect = []
    collided_agents = []
    disconnected_agents = []
    for ii in range(n_agents):
        for jj in range(ii + 1, n_agents):
            dists_ij = np.sqrt(
                np.sum(
                    (X[ii * n1 : ii * n1 + 2, :] - X[jj * n1 : jj * n1 + 2, :]) ** 2, 0
                )
            )
            # ax[1].plot(T, dists_ij)
            if (np.array(dists_ij) < delta_collide).any():
                collided = True
                time_collide.append(T[np.where((np.array(dists_ij) < delta_collide))])
                collided_agents.append((ii, jj))
            if int(ii / int(n_agents / n_teams)) == int(jj / int(n_agents / n_teams)):
                if (np.array(dists_ij) > delta_connect).any():
                    disconnected = True
                    time_disconnect.append(
                        T[np.where((np.array(dists_ij) > delta_connect))]
                    )
                    disconnected_agents.append((ii, jj))

    if collided:
        for i, (ii, jj) in enumerate(collided_agents):
            dists_ij = np.sqrt(
                np.sum(
                    (X[ii * n1 : ii * n1 + 2, :] - X[jj * n1 : jj * n1 + 2, :]) ** 2, 0
                )
            )
            print(
                "Agents",
                ii,
                "and",
                jj,
                "Collided at time = ",
                time_collide[i],
                "with distance",
                dists_ij[np.where((np.array(dists_ij) < delta_collide))],
            )
    else:
        print("Did not collide")

    if disconnected:
        for i, (ii, jj) in enumerate(disconnected_agents):
            dists_ij = np.sqrt(
                np.sum(
                    (X[ii * n1 : ii * n1 + 2, :] - X[jj * n1 : jj * n1 + 2, :]) ** 2, 0
                )
            )
            print(
                "Agents",
                ii,
                "and",
                jj,
                "Disconnected at time = ",
                time_disconnect[i],
                "with distance",
                dists_ij[np.where((np.array(dists_ij) > delta_connect))],
            )
    else:
        print("Did not Disconnect")

    drawing_dt = 0.1
    dt = T[1] - T[0]
    step = int(drawing_dt / dt)
    indices = np.array(list(range(len(T))[::step]) + [len(T) - 1])

    data = dict()
    data["t"] = np.array([])
    data["x"] = np.array([])
    data["y"] = np.array([])
    data["type"] = []
    data["size"] = []
    data["symbol"] = []
    symbol_map = {1: "circle", 2: "circle-open"}
    category_orders = []
    for n in range(n_agents):
        data["t"] = np.hstack([data["t"], T[indices]])
        data["x"] = np.hstack([data["x"], X[n * n1, indices]])
        data["y"] = np.hstack([data["y"], X[n * n1 + 1, indices]])
        data["type"] = data["type"] + ["Agent " + str(n)] * len(indices)
        category_orders = category_orders + ["Agent " + str(n)]
        data["symbol"] = data["symbol"] + [1] * len(indices)
    for n in range(n_agents):
        data["t"] = np.hstack([data["t"], T[indices]])
        data["x"] = np.hstack([data["x"], X[n * n1, indices]])
        data["y"] = np.hstack([data["y"], X[n * n1 + 1, indices]])
        data["type"] = data["type"] + ["Agent circle " + str(n)] * len(indices)
        category_orders = category_orders + ["Agent circle" + str(n)]
        data["symbol"] = data["symbol"] + [2] * len(indices)

    minx, maxx = np.min(data["x"]), np.max(data["x"])
    miny, maxy = np.min(data["y"]), np.max(data["y"])
    xrange = 2 + maxx - minx
    yrange = 2 + maxy - miny
    xfactor = 50  # needs to be tuned for your plot
    yfactor = 50  # needs to be tuned for your plot
    plotter_aspect_ratio = 1  # width/height
    for n in range(n_agents):  # To make marker size accurate
        if xrange / yrange >= plotter_aspect_ratio:
            data["size"] = data["size"] + [
                delta_collide * 1 / (2 + maxx - minx) * xfactor
            ] * len(indices)
        else:
            data["size"] = data["size"] + [
                delta_collide * 1 / (2 + maxy - miny) * yfactor
            ] * len(indices)
    for n in range(n_agents):  # To make marker size accurate
        if xrange / yrange >= plotter_aspect_ratio:
            data["size"] = data["size"] + [
                delta_connect * 16 / (2 + maxx - minx) * xfactor
            ] * len(indices)
        else:
            data["size"] = data["size"] + [
                delta_connect * 16 / (2 + maxy - miny) * yfactor
            ] * len(indices)
    fig = px.scatter(
        data,
        x="x",
        y="y",
        animation_frame="t",
        animation_group="type",
        color="type",
        category_orders={"type": category_orders},
        color_discrete_sequence=colorsHex[0:n_agents],
        size="size",
        size_max=data["size"][-1],
        hover_name="type",
        template="plotly_white",
        symbol=data["symbol"],
        symbol_map=symbol_map,
        # range_x=(minx-.5, maxx+.5),
        # range_y=(miny-.5, maxy+.5),
        range_x=(-1, 4),
        range_y=(-1, 4),
        height=750,
        width=750,
    )

    # Make equal one meter grid
    fig.update_xaxes(dtick=1, showline=False)
    fig.update_yaxes(scaleanchor="x", scaleratio=1, showline=False, dtick=1)

    fig.add_shape(
        type="circle",
        xref="x",
        yref="y",
        fillcolor="black",
        x0=1.8 - 0.5,
        y0=1.3 - 0.5,
        x1=1.8 + 0.5,
        y1=1.3 + 0.5,
    )

    for n in range(n_agents):
        subject_line = px.line(x=X[n * n1, :], y=X[n * n1 + 1, :]).data[0]
        subject_line.line["color"] = colorsHex[n]
        subject_line.line["width"] = 1
        fig.add_trace(subject_line)
        xf_i = go.Scatter(
            x=xf[n * n1],
            y=xf[n * n1 + 1],
            marker_color=colorsHex[n],
            marker_symbol="x",
            marker_size=12,
        )
        fig.add_trace(xf_i)
        x0_i = go.Scatter(
            x=x0[n * n1],
            y=x0[n * n1 + 1],
            marker_color=colorsHex[n],
            marker_symbol="circle-open",
            marker_size=12,
        )
        fig.add_trace(x0_i)

    fig.show()
    return collided, fig2


colors = [
    "dimgrey",
    "rosybrown",
    "maroon",
    "sienna",
    "darkorange",
    "gold",
    "olive",
    "darkseagreen",
    "lime",
    "darkslategrey",
    "aqua",
    "dodgerblue",
    "slateblue",
    "blueviolet",
    "fuchsia",
    "cyan",
    "lavender",
    "orchid",
    "indianred",
    "yellow",
    "orangered",
    "black",
    "forestgreen",
    "darkgoldenrod",
    "tan",
]
colorsHex = [matplotlib.colors.cnames[colors[i]] for i in range(len(colors))]


def visualize_safety_function(
    h_, xlim, ylim, start=None, goal=None, n1=None, resolution=100, contour=False
):
    xs = np.linspace(xlim[0], xlim[1], resolution)
    ys = np.linspace(ylim[0], ylim[1], resolution)
    xv, yv = np.meshgrid(xs, ys)
    xy = np.vstack([xv.flatten(), yv.flatten()])
    # hs = np.array([h_(xy[:,i], 0)[0] for i in range(resolution*resolution)])
    hs = []

    def closest(xy_i):
        vals = h_(xy_i, 0)
        closest = jnp.argmin(vals)
        return vals[closest]

    hs = jax.vmap(closest, 1)(xy)

    fig, ax = plt.subplots()
    ax.contour(
        xv,
        yv,
        hs.reshape(resolution, resolution),
        levels=[0],
        linewidths=1,
        colors="blue",
    )
    if contour:
        im = ax.imshow(
            np.flipud(hs.reshape(resolution, resolution)),
            extent=(xlim[0], xlim[1], ylim[0], ylim[1]),
        )
        cbar = ax.figure.colorbar(im, ax=ax)
    if start is not None:
        if n1 is not None:
            for i in range(0, start.shape[0], n1):
                ax.scatter(start[i], start[i + 1], color="green", s=50)
        else:
            ax.scatter(start[0], start[1], color="green", s=50)
    if goal is not None:
        if n1 is not None:
            for i in range(0, start.shape[0], n1):
                ax.scatter(goal[i], goal[i + 1], color="red", s=50, marker="x")
        else:
            ax.scatter(goal[0], goal[1], color="red", s=50, marker="x")
    ax.set_aspect("equal")
    return fig, ax

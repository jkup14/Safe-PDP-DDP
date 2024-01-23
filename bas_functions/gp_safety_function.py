import jax.numpy as jnp

# import jaxlib.xla_extension.DeviceArray as DA
import jax

# jax.Device = jax.xla.Device # Need this for gpjax to work on my device, comment if not necessary
import numpy as np
import matplotlib.pyplot as plt
import time
from functools import partial
import gpjax as gpx
import jax.random as jr
from jaxutils import Dataset
import jaxkern as jk
from jax import jit
import optax as ox
from pprint import PrettyPrinter

pp = PrettyPrinter(indent=4)
import matplotlib.pyplot as plt


# TODO make it output separate functions for separate hs so that you don't need multiple objects
# TODO make time dependent safety function viable too, right now high H makes the prediction less accurate since there is another
# input dimension and normalize input and output gives high variance
class GP_Safety:
    def __init__(
        self,
        h,
        x_dim: int,
        x_lower,
        x_upper,
        H=None,
        n_samples=100,
        seed=345,
        kernel=jk.Polynomial(degree=2),
        lr=0.01,
        sampling="random",
        active_dims=None,
    ) -> None:
        key = jr.PRNGKey(seed)  # Seed for sample rng
        if active_dims is None:
            self.active_dims = np.arange(x_dim)
        else:
            self.active_dims = active_dims
            kernel.active_dims = active_dims
        H = None
        self.H = H
        self.x_dim = x_dim  # sample dim
        x_lower, x_upper = jnp.array(x_lower), jnp.array(x_upper)
        x_lower_full = (
            jnp.zeros((1, self.x_dim)).at[0, active_dims].set(x_lower.squeeze())
        )
        x_upper_full = (
            jnp.zeros((1, self.x_dim)).at[0, active_dims].set(x_upper.squeeze())
        )
        x_lower = x_lower.squeeze().reshape(1, -1)
        x_upper = x_upper.squeeze().reshape(1, -1)
        assert x_lower.shape == (1, len(self.active_dims)) and x_upper.shape == (
            1,
            len(self.active_dims),
        ), "Lower and Upper Sample lims must have same dim as samples"
        assert x_lower_full.shape == (1, x_dim) and x_upper_full.shape == (
            1,
            x_dim,
        ), "Lower and Upper Sample lims must have same dim as samples"

        self.x_lower, self.x_upper = x_lower, x_upper
        self.x_lower_full, self.x_upper_full = x_lower_full, x_upper_full

        if H is not None:
            self.x_lower = jnp.hstack([self.x_lower, [[0]]])
            self.x_upper = jnp.hstack(
                [self.x_upper, [[H]]]
            )  # Sample timestep from 0 to H
        assert sampling in [
            "random",
            "grid",
        ], "Must choose sampling strategy within ['random,grid']"
        if sampling == "random":
            samples = (
                jr.uniform(key=key, shape=(n_samples, x_dim + (0 if H is None else 1)))
                * (self.x_upper_full - self.x_lower_full)
                + self.x_lower_full
            )  # Create samples within range
        else:
            lspaces = []
            for i in range(x_dim):
                lspaces += [
                    np.linspace(
                        x_lower[0, i],
                        x_upper[0, i],
                        int(jnp.round(n_samples ** (1 / x_dim))),
                    )
                ]
            meshes = jnp.meshgrid(*lspaces)
            samples = np.hstack([mesh.reshape(-1, 1) for mesh in meshes])
        k = 0 if H is None else samples[:, -1]  # sample timesteps
        self.hmap = jax.vmap(
            h, in_axes=(0, None) if H is None else (0, 0)
        )  # vectorize safety function
        y = self.hmap(samples, k).reshape(n_samples, -1)  # get h values at samples
        self.out_dim = y.shape[1]
        self.samples_min, self.samples_range, self.y_min, self.y_range = (
            samples.min(axis=0),
            samples.max(axis=0) - samples.min(axis=0),
            y.min(axis=0),
            y.max(axis=0) - y.min(axis=0),
        )
        # samples, y = self.normalize(samples, self.samples_min, self.samples_range), self.normalize(y, self.y_min, self.y_range)
        self.posteriors = []
        self.learned_params_list = []
        self.likelihoods = []
        self.priors = []
        self.Ds = []
        for i in range(self.out_dim):
            self.Ds += [Dataset(X=samples, y=y[:, [i]])]

            self.priors += [gpx.Prior(kernel=kernel)]
            parameter_state = gpx.initialise(self.priors[-1], key)
            self.likelihoods += [gpx.Gaussian(num_datapoints=self.Ds[-1].n)]

            self.posteriors += [self.priors[-1] * self.likelihoods[-1]]
            parameter_state = gpx.initialise(self.posteriors[-1], key)
            params, trainable, bijectors = parameter_state.unpack()

            negative_mll = jit(
                self.posteriors[-1].marginal_log_likelihood(self.Ds[-1], negative=True)
            )

            optimiser = ox.adam(learning_rate=lr)
            inference_state = gpx.fit(
                objective=negative_mll,
                parameter_state=parameter_state,
                optax_optim=optimiser,
                n_iters=1000,
            )
            learned_params, training_history = inference_state.unpack()
            self.learned_params_list += [learned_params]

            assert not jnp.isnan(
                negative_mll(learned_params)
            ), "GP failed it seems, try lower learning rate maybe?"

        self.stochastic_safety = (
            self.__stochastic_safety
            if H is None
            else self.stochastic_safety_time_dependent
        )
        self.h_x = jax.vmap(
            jax.jacrev(self.stochastic_safety),
            in_axes=(0, None) if H is None else (0, 0),
        )
        self.h_est = jax.vmap(
            self.stochastic_safety, in_axes=(0, None) if H is None else (0, 0)
        )
        # self.h_mean = jax.vmap(self.__mean_safety, in_axes = (0,None) if H is None else (0,0))
        self.h_mean = self.__mean_safety

    # Work in progress
    def __stochastic_safety_time_dependent(self, x, k):
        x = jnp.append(x, k).reshape(-1, self.x_dim + 1)
        x_normalized = self.normalize_x(x)
        latent_dist = self.posterior(self.learned_params, self.D)(x_normalized)
        predictive_dist = self.likelihood(self.learned_params, latent_dist)
        return self.scale_y_dist(predictive_dist.mean(), predictive_dist.stddev())

    # def __stochastic_safety(self, x, k):
    #     x = x[:self.x_dim].reshape(-1,self.x_dim)
    #     # x = self.scale_x(x)
    #     latent_dist = self.posterior(self.learned_params, self.D)(x)
    #     predictive_dist = self.likelihood(self.learned_params, latent_dist)
    #     # return jnp.hstack([predictive_dist.mean(), predictive_dist.stddev()])
    #     return predictive_dist.mean(), predictive_dist.stddev()

    def __stochastic_safety(self, x, k):
        x = x[: self.x_dim].reshape(-1, self.x_dim)
        # x = self.scale_x(x)
        predictive_dist = [
            self.likelihoods[i](
                self.learned_params_list[i],
                self.posteriors[i](self.learned_params_list[i], self.Ds[i])(x),
            )
            for i in range(self.out_dim)
        ]
        return jnp.hstack(
            [predictive_dist[i].mean() for i in range(self.out_dim)]
        ), jnp.hstack([predictive_dist[i].stddev() for i in range(self.out_dim)])

    # def __mean_safety(self, x, k):
    #     x = x[:self.x_dim].reshape(-1,self.x_dim)
    #     latent_dist = self.posterior(self.learned_params, self.D)(x)
    #     predictive_dist = self.likelihood(self.learned_params, latent_dist)
    #     return predictive_dist.mean()

    def __mean_safety(self, x, k):
        x = x[: self.x_dim].reshape(-1, self.x_dim)
        predictive_dist = [
            self.likelihoods[i](
                self.learned_params_list[i],
                self.posteriors[i](self.learned_params_list[i], self.Ds[i])(x),
            )
            for i in range(self.out_dim)
        ]
        return jnp.hstack([predictive_dist[i].mean() for i in range(self.out_dim)])

    def normalize(self, val, min, range):
        return (val - min) / range

    def normalize_x(self, x):
        return (x - self.samples_min) / self.samples_range

    def scale_y_dist(self, mean, std):
        return jnp.hstack([mean * self.y_range + self.y_min, std * self.y_range**2])

    def scale_y(self, y):
        return y * self.y_range + self.y_min

    def test_estimation(self, res=100, border=5, option="error", twod=True, figax=None):
        assert option in ["error", "est", "std"]

        lspaces = [
            np.linspace(self.x_lower[0, i] - border, self.x_upper[0, i] + border, res)
            for i in range(len(self.active_dims))
        ]
        meshes = jnp.meshgrid(*lspaces)
        points = jnp.hstack([mesh.reshape(-1, 1) for mesh in meshes])
        full_mesh = jnp.zeros([points.shape[0], self.x_dim])
        full_mesh = full_mesh.at[:, self.active_dims].set(points)

        # with jax.disable_jit(True):
        est_mesh, std_mesh = self.h_est(full_mesh, 0)
        n_h = est_mesh.shape[1]
        est_mesh = np.array(
            [
                est_mesh[:, i].reshape(*([res] * len(self.active_dims)))
                for i in range(n_h)
            ]
        )
        std_mesh = np.array(
            [
                std_mesh[:, i].reshape(*([res] * len(self.active_dims)))
                for i in range(n_h)
            ]
        )
        labels = self.hmap(full_mesh, 0).reshape(full_mesh.shape[0], -1)
        labels = np.array(
            [labels[:, i].reshape(*([res] * len(self.active_dims))) for i in range(n_h)]
        )
        error_im = np.abs(est_mesh - labels)

        fig, axs = plt.subplots(1, n_h) if figax is None else figax

        if twod:
            if option == "error":
                cbar_min, cbar_max = error_im.min(), error_im.max()
                toplot = error_im
                cbar_label = "Estimation Error"
            elif option == "est":
                cbar_min, cbar_max = est_mesh.min(), est_mesh.max()
                toplot = est_mesh
                cbar_label = "Safety Estimation"
            else:
                cbar_min, cbar_max = std_mesh.min(), std_mesh.max()
                toplot = est_mesh
                cbar_label = "Safety Estimation Variance"

            # ax.set_xlim([self.x_lower[0,0], self.x_upper[0,0]])
            # ax.set_xlim([self.x_lower[1,0], self.x_upper[1,0]])
            try:
                for i, ax in enumerate(axs):
                    im = ax.pcolormesh(
                        lspaces[0],
                        lspaces[1],
                        (toplot[i]),
                        cmap="gray",
                        vmin=cbar_min,
                        vmax=cbar_max,
                    )
                    # plt.scatter(xymesh[:,0], xymesh[:,1], s=0.0001/np.abs(est_mesh-labels.squeeze()), c='red', alpha=1)
                    # cbar = ax.figure.colorbar(im, ax=ax)
                    ax.contour(
                        meshes[0],
                        meshes[1],
                        est_mesh[i],
                        colors="red",
                        levels=[0],
                        linewidths=3,
                        alpha=0.5,
                    )
                    ax.set_aspect("equal")
            except TypeError:
                # im = axs.pcolormesh(lspaces[0], lspaces[1], (toplot[0]), cmap='gray', vmin=cbar_min, vmax=cbar_max)
                # plt.scatter(xymesh[:,0], xymesh[:,1], s=0.0001/np.abs(est_mesh-labels.squeeze()), c='red', alpha=1)
                # cbar = ax.figure.colorbar(im, ax=ax)
                axs.contour(
                    meshes[0],
                    meshes[1],
                    est_mesh[0],
                    colors="red",
                    levels=[0],
                    linewidths=3,
                    alpha=0.5,
                )
                axs.set_aspect("equal")
            if figax is None:
                fig.subplots_adjust(right=0.8)
                cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
                cbar = cbar_ax.figure.colorbar(im, cax=cbar_ax)

                cbar.set_label(cbar_label)
        return fig, axs, np.mean(error_im)

    def plot_traj_uncertainty(self, X, timesteps, num_std=3, figax=None):
        if figax is None:
            fig, ax = plt.subplots(1, 1)
            ax.axhline(y=0, color="r", linestyle="--")
        else:
            fig, ax = figax
        mus, sigmas = self.h_est(X, 0)

        ax.plot(timesteps, mus, c="black")

        for i in range(self.out_dim):
            sigma = sigmas[:, i]
            for s in range(1, num_std + 1):
                ax.fill_between(
                    timesteps,
                    mus[:, i] - sigma * s,
                    mus[:, i] + sigma * s,
                    alpha=0.2,
                    color="tab:blue",
                    label=str(s) + " sigma",
                )
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("$\mu_h\pm$ " + str(num_std) + "$\sigma_h$")
        ax.set_xlim([0, timesteps[-1]])
        # ax.legend()
        return fig, ax

from collections.abc import Callable
import argparse
import pathlib
import h5py

import numpy as np
import scipy
import opinf


def generate_training_data(
    n_samples: int,
    n_timesteps: int,
    q_0: Callable[[np.ndarray], np.ndarray],
    u: Callable[[int], np.ndarray] | None = None,
    mu: float | None = None,
):
    """Generate sample data to be used for Operator Inference.

    Args:
    n_samples: Number of spatial samples.
    n_timesteps: Number of time steps.
    q_0: Initial condition function. Accepts an array of spatial locations and
        returns an array of initial condition values.
    u: External input function. Accepts a time value and returns the input
        values for that time step. If None, u defaults is the zero function
        (for a model that does not use external inputs)
    mu: Model parameter. If none, a non-parametric model is used.

    Note: 'u' and 'mu' cannot both be specified
    (the model can use external inputs or parameters or neither, but not both)

    Returns:
    t: Array of time points.
    Q: Array of "observed" snapshots, shape (n_samples, n_timesteps).
    x: Array of spatial points (for parametric models)
    """
    # construct the spatial and temporal domains
    x = np.linspace(0, 1, n_samples + 2)[1:-1]
    dx = x[1] - x[0]
    t = np.linspace(0, 1, n_timesteps)

    # construct the matrix of linear operators
    diags = np.array([1, -2, 1]) / dx**2
    A = scipy.sparse.diags(diags, [-1, 0, 1], (n_samples, n_samples))

    if mu is None:
        # non-parametric

        external_inputs = True
        if u is None:
            external_inputs = False

            def u(t):
                return 0

        # construct the matrix of external input operators
        B = np.zeros_like(x)
        B[0], B[-1] = 1 / dx**2, 1 / dx**2

        fom = opinf.models.ContinuousModel(
            operators=[
                opinf.operators.LinearOperator(A),
                opinf.operators.InputOperator(B),
            ]
        )

        initial_values = q_0(x)
        initial_values = (
            initial_values if not external_inputs else initial_values * u(0)
        )

        return t, fom.predict(initial_values, t, input_func=u, method="BDF")
    else:
        # parametric

        # construct the constant term dependent on mu
        c0 = np.zeros_like(x)
        c0[0], c0[-1] = 1 / dx**2, 1 / dx**2

        return (
            t,
            scipy.integrate.solve_ivp(
                fun=lambda t, q: mu * (c0 + A @ q),
                y0=q_0(x),
                t_span=[t[0], t[-1]],
                t_eval=t,
                method="BDF",
            ).y,
            x,
        )


def generate_basics_data(filepath: str = "basics_data.h5"):
    """Generate training data with various initial conditions
    for the basics tutorial.

    Creates datasets with solutions to the advection-diffusion equation using
    six different initial condition functions. Results are saved to an HDF5
    file with separate datasets for each initial condition.

    Parameters
    ----------
    filepath : str, optional
        Path where the generated HDF5 file will be saved.
        Default is "basics_data.h5".

    Returns
    -------
    None

    Notes
    -----
    The generated file contains:
    - "t": time points for all simulations
    - "default": snapshots for the default initial condition q_0(x) = x(1-x)
    - "Experiment 1" through "Experiment 6": snapshots for alternative
      initial conditions, each with a "title" attribute for reference

    The spatial domain is [0, 1] with 512 interior grid points, and the
    temporal domain extends from t=0 to t=1 with 401 time steps.
    """
    # set up basic parameters
    n_samples = 512
    n_timesteps = 401

    # define the various initial conditions used in the tutorial
    def q_0_default(x):
        return x * (1 - x)

    initial_conditions = [
        (r"$q_{0}(x) = 10 x (1 - x)$", lambda x: 10 * x * (1 - x)),
        (
            r"$q_{0}(x) = 5 x^{2} (1 - x)^{2}$",
            lambda x: 5 * x**2 * (1 - x) ** 2,
        ),
        (
            r"$q_{0}(x) = 50 x^{4} (1 - x)^{4}$",
            lambda x: 50 * x**4 * (1 - x) ** 4,
        ),
        (
            r"$q_{0}(x) = \frac{1}{2}\sqrt{x (1 - x)}$",
            lambda x: 0.5 * np.sqrt(x * (1 - x)),
        ),
        (
            r"$q_{0}(x) = \frac{1}{4}\sqrt[4]{x (1 - x)}$",
            lambda x: 0.25 * np.sqrt(np.sqrt(x * (1 - x))),
        ),
        (
            r"$q_{0}(x) = \frac{1}{3}\sin(\pi x) + \frac{1}{5}\sin(5\pi x)$",
            lambda x: np.sin(np.pi * x) / 3 + np.sin(5 * np.pi * x) / 5,
        ),
    ]

    # initialize the file we will write the data to
    with h5py.File(filepath, "w") as f:
        # generate and save data for the default initial condition
        # also save the data for the time dimension
        t, Q_default = generate_training_data(
            n_samples, n_timesteps, q_0_default
        )
        f.create_dataset("t", data=t)
        f.create_dataset("default", data=Q_default)

        f.attrs["num_experiments"] = len(initial_conditions)
        for idx, (title, func) in enumerate(initial_conditions):
            # for each initial condition,
            # generate the data for that condition
            # and save it as a new dataset
            _, Q = generate_training_data(n_samples, n_timesteps, func)
            dset = f.create_dataset(f"Experiment {idx+1}", data=Q)
            dset.attrs["title"] = title

    print(f"Training data saved to {filepath}")


def generate_external_inputs_data(filepath: str = "inputs_data.h5"):
    """Generate training data with external input functions for
    the inputs tutorial.

    Creates datasets with solutions to the advection-diffusion equation subject
    to three different external input functions. Each input function has both
    training and test variants to facilitate learning external input operators.
    Results are saved to an HDF5 file organized in training and test groups.

    Parameters
    ----------
    filepath : str, optional
        Path where the generated HDF5 file will be saved.
        Default is "inputs_data.h5".

    Returns
    -------
    None

    Notes
    -----
    The generated file contains:
    - "t": time points for all simulations
    - "Q": snapshots for the default external input function
    - "U": the default external input values
    - "train" group: contains Q_0, U_0, Q_1, U_1, Q_2, U_2 datasets
    - "test" group: contains Q_0, U_0, Q_1, U_1, Q_2, U_2 datasets

    The spatial domain is [0, 1] with 1023 interior grid points, and the
    temporal domain extends from t=0 to t=1 with 1001 time steps. The initial
    condition depends on an exponential-based function with
    parameter alpha=100.
    """
    n_samples = 1023
    n_timesteps = 1001

    alpha = 100

    # the part of the initial condition independent of u(t)
    def q_0(x):
        return np.exp(alpha * (x - 1)) + np.exp(-alpha * x) - np.exp(-alpha)

    # the define the external inputs functions
    def u(t):
        return np.ones_like(t) + np.sin(4 * np.pi * t) / 4

    def u_test(t):
        return 1 + t * (1 - t)

    train_inputs = [
        lambda t: np.exp(-t),
        lambda t: 1 + t**2 / 2,
        lambda t: 1 - np.sin(np.pi * t) / 2,
    ]
    test_inputs = [
        lambda t: 1 - np.sin(3 * np.pi * t) / 3,
        lambda t: 1 + 25 * (t * (t - 1)) ** 3,
        lambda t: 1 + np.exp(-2 * t) * np.sin(np.pi * t),
    ]

    # initialize the h5 file to write to
    with h5py.File(filepath, "w") as f:
        # generate training data for the beginning of the tutorial
        t, Q = generate_training_data(n_samples, n_timesteps, q_0, u)
        U = u(t)

        _, Q_test = generate_training_data(n_samples, n_timesteps, q_0, u_test)
        U_test = u_test(t)

        f.create_dataset("t", data=t)
        f.create_dataset("Q", data=Q)
        f.create_dataset("U", data=U)
        f.create_dataset("Q_test", data=Q_test)
        f.create_dataset("U_test", data=U_test)

        train_grp = f.create_group("train")
        test_grp = f.create_group("test")

        train_grp.attrs["num_input_functions"] = len(train_inputs)
        test_grp.attrs["num_input_functions"] = len(test_inputs)

        # for each input function, generate data for the inputs and snapshots
        # then, save that data to a new dataset in the file
        for idx, [train_input, test_input] in enumerate(
            zip(train_inputs, test_inputs)
        ):
            _, Q_train = generate_training_data(
                n_samples, n_timesteps, q_0, train_input
            )
            U_train = train_input(t)
            _, Q_test = generate_training_data(
                n_samples, n_timesteps, q_0, test_input
            )
            U_test = test_input(t)

            train_grp.create_dataset(f"Q_{idx}", data=Q_train)
            train_grp.create_dataset(f"U_{idx}", data=U_train)
            test_grp.create_dataset(f"Q_{idx}", data=Q_test)
            test_grp.create_dataset(f"U_{idx}", data=U_test)

    print(f"Training data saved to {filepath}")


def generate_parametric_data(filepath: str = "parametric_data.h5"):
    n_samples = 1023
    n_timesteps = 401

    alpha = 100

    # the part of the initial condition independent of u(t)
    def q_0(x):
        return np.exp(alpha * (x - 1)) + np.exp(-alpha * x) - np.exp(-alpha)

    # initialize the h5 file to write to
    f = h5py.File(filepath, "w")

    # generate and write training data
    num_training_parameters = 10
    training_parameters = np.logspace(-1, 1, num_training_parameters)

    train_grp = f.create_group("train")
    train_grp.attrs["num_mu_values"] = num_training_parameters

    for idx, mu in enumerate(training_parameters):
        t, Q, x = generate_training_data(n_samples, n_timesteps, q_0, mu=mu)

        if idx == 0:
            # on the first iteration,
            # also save the temporal and spatial dimensions
            f.create_dataset("t", data=t)
            f.create_dataset("x", data=x)
            f.create_dataset("q_0", data=q_0(x))

        dset = train_grp.create_dataset(f"Step {idx+1}", data=Q)
        dset.attrs["mu"] = mu

    # generate and write test data
    test_parameters = np.sqrt(
        training_parameters[:-1] * training_parameters[1:]
    )
    test_grp = f.create_group("test")
    test_grp.attrs["num_mu_values"] = len(test_parameters)

    for idx, mu in enumerate(test_parameters):
        _, Q, _ = generate_training_data(n_samples, n_timesteps, q_0, mu=mu)

        dset = test_grp.create_dataset(f"Step {idx+1}", data=Q)
        dset.attrs["mu"] = mu

    f.close()
    print(f"Data saved to {filepath}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate training data for Operator Inference tutorials."
    )
    parser.add_argument(
        "dataset",
        nargs="?",
        default="all",
        choices=["basics", "inputs", "parametric", "all"],
        help="Dataset to generate (default: all)",
    )

    args = parser.parse_args()

    BASE_DIR = pathlib.Path(__file__).resolve().parent

    if args.dataset == "basics" or args.dataset == "all":
        generate_basics_data(
            str(BASE_DIR / "source" / "tutorials" / "basics_data.h5")
        )
    if args.dataset == "inputs" or args.dataset == "all":
        generate_external_inputs_data(
            str(BASE_DIR / "source" / "tutorials" / "inputs_data.h5")
        )
    if args.dataset == "parametric" or args.dataset == "all":
        generate_parametric_data(
            str(BASE_DIR / "source" / "tutorials" / "parametric_data.h5")
        )

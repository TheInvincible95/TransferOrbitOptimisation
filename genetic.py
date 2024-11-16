import math as m
import os
import random
import numpy as np
import plotly.graph_objects as go


class Ellipse:
    def __init__(self, p, e, omega) -> None:
        self.p = p
        self.e = e
        self.omega = omega % (2 * m.pi)
        self.a = calc_a(p, e)

    # These functions only make sense for a particular transfer orbit solution, not all cases
    def setCandidateParameters(
        self, theta1: float, theta2: float, r1: float, r2: float
    ):
        self.theta1 = theta1
        self.theta2 = theta2
        self.r1 = r1
        self.r2 = r2

    def getGenes(self):
        return [self.theta1, self.theta2, self.p]

    def isValid(self):
        return not (
            self.r1 <= 0
            or self.r2 <= 0
            or (self.e < 0 or self.e >= 1)
            or (self.a <= 0 or 2 * self.a < self.r1 or 2 * self.a < self.r2)
        )


def calc_p(a, e):
    return a * (1 - e * e)


def calc_a(p, e):
    return p / (1 - e * e)


def calc_r(p: float, e: float, theta: float, omega: float):
    return p / (1 + e * m.cos(theta - omega))


def calc_omega_t(theta_1, theta_2, p_t, r_1, r_2):
    numerator = (m.cos(theta_2) * (p_t - r_1) / r_1) - (
        m.cos(theta_1) * (p_t - r_2) / r_2
    )
    denominator = (m.sin(theta_1) * (p_t - r_2) / r_2) - (
        m.sin(theta_2) * (p_t - r_1) / r_1
    )
    return m.atan2(numerator, denominator)


def calc_e_t(theta_1, p_t, r_1, omega_t):
    return (p_t - r_1) / (r_1 * m.cos(theta_1 - omega_t))


def calc_gamma(e, theta, omega):
    if e >= 1 or e < 0:
        raise ValueError("Eccentricity of the ellipse must be in range [0,1)")
    return m.atan2(e * m.sin(theta - omega), 1 + e * m.cos(theta - omega))


def calculate_orbit(ellipse: Ellipse):
    a, e, omega = ellipse.a, ellipse.e, ellipse.omega

    theta = np.linspace(0, 2 * np.pi, 100)
    r = a * (1 - e**2) / (1 + e * np.cos(theta))
    # r = a
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    x_rotated = x * np.cos(omega) - y * np.sin(omega)
    y_rotated = x * np.sin(omega) + y * np.cos(omega)
    return x_rotated, y_rotated


# The standard gravitational parameter μ of a celestial body
mu = 1  # Since it is constant for a system, it will get cancelled out where we're comparing between two solutions anyway.


# 2*a _ r
# r *  a
def calc_visviva_vel(r, a):
    return m.sqrt(mu * (2 / r - 1 / a))


def calc_delta_v(V_1, V_2, gamma_1, gamma_2):
    return m.sqrt(V_1**2 + V_2**2 - 2 * V_1 * V_2 * m.cos(gamma_1 - gamma_2))


class Problem:
    def __init__(self, initial: Ellipse, final: Ellipse) -> None:
        self.initial = initial
        self.final = final

    @property
    def max_p_t(self) -> float:
        return 2 * max(self.initial.a, self.final.a)

    def _getCandidateFromGenes(self, theta_1: float, theta_2: float, p_t: float):
        r_1 = calc_r(self.initial.p, self.initial.e, theta_1, self.initial.omega)
        r_2 = calc_r(self.final.p, self.final.e, theta_2, self.final.omega)

        omega_t = calc_omega_t(theta_1, theta_2, p_t, r_1, r_2)
        e_t = calc_e_t(p_t, r_1, theta_1, omega_t)
        candidate = Ellipse(p_t, e_t, omega_t)

        candidate.setCandidateParameters(theta_1, theta_2, r_1, r_2)
        return candidate

    def getCandidate(self) -> Ellipse:
        while True:
            theta_1, theta_2, p_t = (
                random.random() * 2 * m.pi,
                random.random() * 2 * m.pi,
                random.random() * problem.max_p_t,
            )
            candidate = self._getCandidateFromGenes(theta_1, theta_2, p_t)
            if candidate.isValid:
                return candidate

    def fitness(self, candidate: Ellipse) -> float:
        if not candidate.isValid():
            return 0

        gamma_1_i = calc_gamma(self.initial.e, candidate.theta1, self.initial.omega)
        gamma_1_t = calc_gamma(candidate.e, candidate.theta1, candidate.omega)
        gamma_2_t = calc_gamma(candidate.e, candidate.theta2, candidate.omega)
        gamma_2_f = calc_gamma(self.final.e, candidate.theta2, self.final.omega)

        V_1_i = calc_visviva_vel(candidate.r1, self.initial.a)
        V_1_t = calc_visviva_vel(candidate.r1, candidate.a)
        V_2_f = calc_visviva_vel(candidate.r2, self.final.a)
        V_2_t = calc_visviva_vel(candidate.r2, candidate.a)
        deltaV_1 = calc_delta_v(V_1_i, V_1_t, gamma_1_i, gamma_1_t)
        deltaV_2 = calc_delta_v(V_2_f, V_2_t, gamma_2_f, gamma_2_t)

        return 100 - (deltaV_1 + deltaV_2)

    # Will never return the same parents
    def crossover(self, parent1: Ellipse, parent2: Ellipse) -> tuple[Ellipse, Ellipse]:
        random_indexes = [random.randint(0, 1) for _ in range(3)]
        genes_parent1 = parent1.getGenes()
        genes_parent2 = parent2.getGenes()
        genes_child1 = [
            genes_parent1[i] if random_indexes[i] == 0 else genes_parent2[i]
            for i in range(3)
        ]
        genes_child2 = [
            genes_parent1[i] if random_indexes[i] == 1 else genes_parent2[i]
            for i in range(3)
        ]
        child1 = self._getCandidateFromGenes(*genes_child1)
        child2 = self._getCandidateFromGenes(*genes_child2)
        return child1, child2

    def mutate(self, candidate: Ellipse, angle_mutation_rate=0.5, p_mutation_rate=0.1):
        genes = candidate.getGenes()
        r = random.random() * (angle_mutation_rate * 2 + p_mutation_rate)

        if r < angle_mutation_rate:
            angle_mutation = random.uniform(-m.pi / 4, m.pi / 4)
            genes[0] = (genes[0] + angle_mutation) % (2 * m.pi)

        if r < angle_mutation_rate + angle_mutation_rate:
            angle_mutation = random.uniform(-m.pi / 4, m.pi / 4)
            genes[1] = (genes[1] + angle_mutation) % (2 * m.pi)

        if r >= angle_mutation_rate + angle_mutation_rate:

            p_mutation = (
                random.uniform(-self.max_p_t / 4, self.max_p_t / 4) 
            )
            genes[2] = (genes[2] +p_mutation)% self.max_p_t


        return self._getCandidateFromGenes(*genes)


def roulette_wheel_selection_generator(s):
    total_fitness = sum(fitness for _, fitness in s)
    selection_probs = [fitness / total_fitness for _, fitness in s]
    cumulative_probs = [
        sum(selection_probs[: i + 1]) for i in range(len(selection_probs))
    ]

    while True:
        rand1 = random.random()
        for i, cum_prob in enumerate(cumulative_probs):
            if rand1 < cum_prob:
                parent1 = s[i][0]
                break
        rand2 = random.random()
        for i, cum_prob in enumerate(cumulative_probs):
            if rand2 < cum_prob:
                parent2 = s[i][0]
                break

        yield parent1, parent2


def genetic_algorithm(
    problem: Problem,
    *,
    population=100,
    generations=50,
    k=20,  # GA Parameters
    _to_log: bool,
    _logs: list[list[Ellipse, float]],
    _log_limit: int,  # Logging Parameters
):
    if _to_log:
        assert _logs is not None, "Provide a storage array for the logs"
        assert isinstance(
            _log_limit, int
        ), "Max number of logs per generation must be provided"
        _log_limit = _log_limit if _log_limit <= population else population

    solution_set: list[Ellipse] = [problem.getCandidate() for _ in range(population)]

    for generation in range(generations):
        s: list[Ellipse, float] = sorted(
            [(x, problem.fitness(x)) for x in solution_set],
            key=lambda x: x[1],
            reverse=True,
        )

        if _to_log:
            _logs.append(s[:_log_limit])

        parent_generator = roulette_wheel_selection_generator(s)
        children = []

        while len(children) < population - len(s) // 3:
            parent1, parent2 = next(parent_generator)
            child1, child2 = problem.crossover(parent1, parent2)
            if child1.isValid() and len(children) < population - len(s) // 3:
                children.append(child1)
            if child2.isValid() and len(children) < population - len(s) // 3:
                children.append(child2)
            mutated_child1 = problem.mutate(child1)
            if mutated_child1.isValid() and len(children) < population - len(s) // 3:
                children.append(mutated_child1)
            mutated_child2 = problem.mutate(child2)
            if mutated_child2.isValid() and len(children) < population - len(s) // 3:
                children.append(mutated_child2)
        # keeps best third of the parents
        solution_set = [x[0] for x in s[: population // 3]] + children

        print(solution_set[0].getGenes(), problem.fitness(solution_set[0]))
    best_solution = max(solution_set, key=lambda x: problem.fitness(x))
    return best_solution


# problem = Problem(5, 5,
#                   0.5, 0.5,
#                  m.pi / 2, 3 * m.pi / 2)
ini = Ellipse(2, 0.2, 0)
fin = Ellipse(2, 0.4, m.pi / 3)
problem = Problem(ini, fin)
logs: list[list[Ellipse, float]] = []
log_limit = 5
sol = genetic_algorithm(problem, _to_log=True, _logs=logs, _log_limit=log_limit)

for generation, log in enumerate(logs):
    fig = go.Figure()

    # Plot the initial and final ellipses
    x_i, y_i = calculate_orbit(problem.initial)
    x_f, y_f = calculate_orbit(problem.final)
    fig.add_trace(
        go.Scatter(
            x=x_i,
            y=y_i,
            mode="lines",
            name="Initial Orbit",
            line=dict(color="black"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x_f, y=y_f, mode="lines", name="Final Orbit", line=dict(color="black")
        )
    )
    fig.add_trace(go.Scatter(x=[0], y=[0], name="Center", line=dict(color="black")))

    # Plot children with color based on fitness
    for i, (child, fitness_value) in enumerate(log):
        color = f"rgba({255 * (log_limit-i / log_limit)}, 0, {255 * (i / log_limit)}, 0.8)"  # Color from blue to red
        x_c, y_c = calculate_orbit(child)
        fig.add_trace(
            go.Scatter(
                x=x_c, y=y_c, mode="lines", name=f"Child_{i}", line=dict(color=color)
            )
        )

    fig.update_layout(
        title=f"Generation: {generation + 1}",
        xaxis_title="X",
        yaxis_title="Y",
        showlegend=True,
    )
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    fig.write_image(f"./frames/generation_{generation + 1:03d}.png")

os.system(
    "ffmpeg -framerate 1 -i frames/generation_%03d.png -c:v libx264 -pix_fmt yuv420p genetic_algorithm_evolution.mp4"
)

print(sol, problem.fitness(sol))

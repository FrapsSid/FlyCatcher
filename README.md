# Fly Catcher — Genetic Algorithm Training

A real-time simulation where teams of frogs evolve to catch flies using a
genetic algorithm. Watch as frog placement, tongue mechanics, and hunting
strategies improve across generations — all rendered live with Pygame.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Controls](#controls)
- [How It Works](#how-it-works)
  - [Simulation Arena](#simulation-arena)
  - [Frog Mechanics](#frog-mechanics)
  - [Genome & Traits](#genome--traits)
  - [Genetic Algorithm](#genetic-algorithm)
  - [Fitness Function](#fitness-function)
- [Configuration](#configuration)
- [Output](#output)
- [Performance Notes](#performance-notes)
- [License](#license)

---

## Overview

Fly Catcher is an evolutionary simulation that demonstrates how a genetic
algorithm can optimize multi-agent cooperative behavior. A team of frogs is
placed in a 2D arena where flies spawn on the left and attempt to escape to
the right. Each generation, the best-performing frog teams are selected,
crossed over, and mutated to produce increasingly effective hunters

The simulation runs in real-time with full visualization, allowing you to
observe the evolutionary process as it happens

---

## Features

- **Genetic Algorithm Optimization** — Evolves frog teams across 40+
  generations using tournament selection, blend crossover, and Gaussian
  mutation
- **Multi-Agent Cooperation** — Frogs coordinate target selection to avoid
  redundant attacks
- **Intelligent Fly AI** — Flies exhibit flocking and frog detection
- **Real-Time Visualization** — Full Pygame rendering with arena zones, tongue
  animations, vision ranges, and stat panels
- **Spatial Hashing** — Efficient neighbor queries via a spatial hash grid for
  scalable performance
- **Comprehensive Fitness** — Multi-objective fitness combining catch count,
  speed, consistency, cooperation, and spatial coverage
- **Data Export** — Training metrics saved to CSV; optional Matplotlib
  evolution plots

---

## Installation

### Prerequisites

- Python 3.8 or higher
- pip

### Install Dependencies

    pip install pygame numpy

Optional (for evolution plots after training):

    pip install matplotlib

---

## Usage

Run the simulation:

    python run.py

The program will:

1. Initialize a random population of frog teams
2. Evaluate each team by running the simulation
3. Display the best team of each generation
4. Evolve the population and repeat
5. Save results when complete

---

## Controls

| Key            | Action                        |
|----------------|-------------------------------|
| SPACE          | Pause / Resume simulation     |
| Right Arrow    | Skip to next generation       |
| Up Arrow       | Increase simulation speed     |
| Down Arrow     | Decrease simulation speed     |
| ESC            | Quit                          |

---

## How It Works

### Simulation Arena

The arena is a 500x150 unit space (rendered at 3x scale):

- Flies spawn in the left green zone and move rightward
- Frogs are positioned within the central zone (35-85% of width, 10-90%
  of height)
- Flies that reach x >= 460 are counted as escaped
- Each simulation runs for a maximum of 600 steps

### Frog Mechanics

Each frog has the following attack cycle:

1. **Scan** — Search for flies within vision range using the spatial grid
2. **Target Select** — Score visible flies by distance and escape progress
3. **Cooperate** — Optionally avoid targeting flies already targeted by
   teammates
4. **Ambush Decision** — Wait for optimal range based on the ambush_mode trait
5. **Predictive Strike** — Extend tongue toward the fly's predicted position
6. **Catch Check** — Any fly within 10 units of the tongue tip is caught
7. **Retract & Cooldown** — Tongue retracts, then the frog rests for
   rest_cycles steps

### Genome & Traits

Each frog's behavior is controlled by an 8-parameter genome:

| Gene               | Range      | Description                                          |
|--------------------|------------|------------------------------------------------------|
| tongue_power       | 0.5 – 2.0  | Multiplier on base tongue length (20 units)          |
| position_x         | 0.0 – 1.0  | Horizontal position within frog zone                 |
| position_y         | 0.0 – 1.0  | Vertical position within frog zone                   |
| vision_range       | 15 – 50    | Detection radius for spotting flies                  |
| target_preference  | 0.0 – 1.0  | 0 = nearest fly, 1 = fly closest to escaping         |
| cooperation        | 0.0 – 1.0  | Probability of avoiding teammates' targets           |
| ambush_mode        | 0.0 – 1.0  | 0 = strike at max range, 1 = wait for close range    |
| rest_cycles        | 5 – 30     | Cooldown steps between tongue strikes                |

### Genetic Algorithm

The evolution process follows a standard GA loop:

       Initialize random population (24 teams x 3 frogs each)
           |
           v
       Evaluate all teams (2 trials each, averaged)
           |
           v
       Record statistics & update best team
           |
           v
       Selection (tournament, size 3)
           |
           v
       Crossover (blend crossover, alpha=0.3, rate=80%)
           |
           v
       Mutation (Gaussian, sigma=15% of range, rate=20%)
           |
           v
       Elitism (top 3 teams copied unchanged)
           |
           v
       Next Generation (repeat x 40)

Key operators:

- **Tournament Selection** — Pick 3 random teams, select the one with highest
  fitness
- **Blend Crossover (BLX-alpha)** — For each gene, sample uniformly from an
  interval extended 30% beyond the parents' range
- **Gaussian Mutation** — Add noise from N(0, 0.15 x gene_range) with 20%
  probability per gene
- **Elitism** — Top 3 teams survive unchanged to the next generation

### Fitness Function

The fitness function is multi-objective, combining several factors:

    fitness = (catches x 100)
            - (escaped x 25)
            + perfect_clear_bonus      (200 if all flies caught)
            + speed_bonus              (faster completion = higher score)
            + catch_rate_bonus         (catches per 100 steps x 10)
            + first_catch_bonus        (early first catch = bonus up to 30)
            + consistency_bonus        (low interval between catches = up to 20)
            + cooperation_bonus        (multiple active frogs = 20 per frog)
            + spacing_bonus            (frogs spread apart = up to 30)

This encourages teams that:
- Catch as many flies as possible
- Do it quickly and consistently
- Use all frogs effectively
- Position frogs with good spatial coverage

---

## Configuration

All parameters can be adjusted at the top of run.py:

### Arena & Entities

    ARENA_WIDTH = 500      # Arena width in simulation units
    ARENA_HEIGHT = 150     # Arena height in simulation units
    SCALE = 3              # Pixel scale for rendering
    NUM_FLIES = 50         # Number of flies per simulation (<=50 recommended)
    NUM_FROGS = 3          # Number of frogs per team

### Genetic Algorithm

    POPULATION_SIZE = 24   # Number of teams per generation
    NUM_GENERATIONS = 40   # Total training generations
    MUTATION_RATE = 0.2    # Per-gene mutation probability
    CROSSOVER_RATE = 0.8   # Probability of crossover vs cloning
    ELITE_COUNT = 3        # Teams preserved unchanged

### Gameplay Mechanics

    FLY_SPEED = 2.2            # Maximum fly speed
    TONGUE_BASE_LENGTH = 20    # Base tongue reach (modified by genome)
    TONGUE_SPEED = 14          # Tongue extension/retraction speed
    MAX_STEPS = 600            # Maximum steps per simulation

### Fly AI

    FLY_FROG_DETECTION_RANGE = 80    # Range at which flies notice frogs
    FLY_FROG_DANGER_RANGE = 40       # Range at which flies enter panic mode
    FLY_LATERAL_AVOIDANCE = 0.4      # Strength of sideways evasion

WARNING: Setting NUM_FLIES above 50 may cause performance issues.

---

## Output

After training completes, the program generates:

### results/training_data.csv

A CSV file with per-generation statistics:

| Column               | Description                                 |
|----------------------|---------------------------------------------|
| generation           | Generation number                           |
| best_fitness         | Highest fitness in generation               |
| avg_fitness          | Mean fitness across population              |
| avg_catches          | Average catches per team                    |
| avg_escaped          | Average escapes per team                    |
| avg_completion_time  | Average steps to clear or timeout           |
| perfect_clears       | Number of teams that caught all flies       |
| avg_catch_speed      | Average catch rate (per 100 steps)          |

### results/evolution_plot.png

If Matplotlib is installed, a 2x2 plot is generated and displayed:

1. **Fitness Evolution** — Best and average fitness over generations
2. **Catches vs Escaped** — Catch and escape counts over time
3. **Catch Speed** — Rate of catches per 100 steps (area chart)
4. **Perfect Clears** — Bar chart of perfect clears per generation

---

### Code Architecture

| Class              | Responsibility                                              |
|--------------------|-------------------------------------------------------------|
| SpatialGrid        | Spatial hash grid for O(1) neighbor lookups                 |
| Genome             | 8-parameter genome with mutation, crossover, and clamping   |
| Fly                | Fly entity with flocking, avoidance, and panic behaviors    |
| Frog               | Frog entity with vision, targeting, and tongue mechanics    |
| Simulation         | Orchestrates a single simulation run with all entities      |
| GeneticAlgorithm   | Population management, selection, crossover, mutation       |
| Game               | Pygame visualization, rendering, and input handling         |

---

## Performance Notes

- **Spatial hashing** is used for both fly-fly neighbor queries and frog-fly
  vision checks, keeping performance manageable with 50 flies and 3 frogs
- The simulation is intentionally **single-threaded**; population evaluation
  is sequential
- Each team is evaluated over **2 trials** and averaged to reduce noise
- Use Up/Down arrows to adjust visualization speed (1x-15x) during training
- Press Right Arrow to skip the current generation's visualization entirely
- For faster training, the visualization loop could be bypassed by calling
  the GA loop directly without the Game class

---

## License

This project is provided as-is for educational and experimental purposes.
Feel free to use, modify, and distribute.

"""
Fly Catcher - Genetic Algorithm Training
By Maksim Batuev, Karim Fajzahanov and Alina Shadrina
"""

import pygame
import numpy as np
import random
import csv
import os
from dataclasses import dataclass
from typing import List, Tuple, Optional

# ============================================================
# CONFIGURATION
# ============================================================

# Arena
ARENA_WIDTH = 500
ARENA_HEIGHT = 150
SCALE = 3

# Flies
NUM_FLIES = 50
FLY_SPEED = 2.2
FLY_SIZE = 3

# Frogs
NUM_FROGS = 3
FROG_SIZE = 12

# Tongue
TONGUE_BASE_LENGTH = 20
TONGUE_SPEED = 14
BASE_COOLDOWN = 15

# Position constraints
FROG_ZONE_X_MIN = 0.35
FROG_ZONE_X_MAX = 0.85
FROG_ZONE_Y_MIN = 0.10
FROG_ZONE_Y_MAX = 0.90

# Zones
ESCAPE_ZONE_X = ARENA_WIDTH - 40
SPAWN_ZONE_WIDTH = 60
MAX_STEPS = 600

# Genetic Algorithm
POPULATION_SIZE = 24
NUM_GENERATIONS = 40
MUTATION_RATE = 0.2
CROSSOVER_RATE = 0.8
ELITE_COUNT = 3

# Time bonus
PERFECT_CLEAR_BONUS = 200
SPEED_BONUS_MULTIPLIER = 0.5

# Fly avoidance behavior
FLY_FROG_DETECTION_RANGE = 80
FLY_FROG_DANGER_RANGE = 40
FLY_LATERAL_AVOIDANCE = 0.4

# Performance settings
SPATIAL_GRID_SIZE = 30

# Colors
WHITE = (255, 255, 255)
SKY_BLUE = (135, 206, 235)
GREEN = (50, 205, 50)
DARK_GREEN = (34, 139, 34)
PINK = (255, 105, 180)
BLACK = (20, 20, 20)
RED_ZONE = (255, 200, 200)
SPAWN_ZONE_COLOR = (200, 255, 200)
GRAY = (100, 100, 100)
BLUE = (50, 50, 200)
GOLD = (255, 215, 0)
ORANGE = (255, 165, 0)

FROG_COLORS = [
    (50, 205, 50),
    (50, 150, 50),
    (100, 200, 100),
    (30, 180, 80),
    (80, 220, 80),
]

# ============================================================
# SPATIAL GRID
# ============================================================

class SpatialGrid:
    """Spatial hash grid for efficient neighbor queries."""
    
    def __init__(self, width: float, height: float, cell_size: float):
        self.cell_size = cell_size
        self.cols = int(np.ceil(width / cell_size)) + 1
        self.rows = int(np.ceil(height / cell_size)) + 1
        self.grid = {}
    
    def clear(self):
        self.grid.clear()
    
    def _get_cell(self, x: float, y: float) -> Tuple[int, int]:
        return (int(x / self.cell_size), int(y / self.cell_size))
    
    def insert(self, obj, x: float, y: float):
        cell = self._get_cell(x, y)
        if cell not in self.grid:
            self.grid[cell] = []
        self.grid[cell].append(obj)
    
    def get_nearby(self, x: float, y: float, radius: float) -> List:
        """Get all objects within radius of point."""
        nearby = []
        cell_radius = int(np.ceil(radius / self.cell_size))
        center_cell = self._get_cell(x, y)
        
        for dx in range(-cell_radius, cell_radius + 1):
            for dy in range(-cell_radius, cell_radius + 1):
                cell = (center_cell[0] + dx, center_cell[1] + dy)
                if cell in self.grid:
                    nearby.extend(self.grid[cell])
        
        return nearby

# ============================================================
# GENOME
# ============================================================

@dataclass 
class Genome:
    tongue_power: float
    position_x: float
    position_y: float
    vision_range: float
    target_preference: float
    cooperation: float
    ambush_mode: float
    rest_cycles: float
    
    RANGES = {
        'tongue_power': (0.5, 2.0),
        'position_x': (0.0, 1.0),
        'position_y': (0.0, 1.0),
        'vision_range': (15.0, 50.0),
        'target_preference': (0.0, 1.0),
        'cooperation': (0.0, 1.0),
        'ambush_mode': (0.0, 1.0),
        'rest_cycles': (5.0, 30.0)
    }
    
    def __post_init__(self):
        for attr, (min_v, max_v) in self.RANGES.items():
            val = getattr(self, attr)
            setattr(self, attr, max(min_v, min(max_v, val)))
    
    @classmethod
    def random(cls) -> 'Genome':
        return cls(
            tongue_power=random.uniform(0.5, 2.0),
            position_x=random.uniform(0.0, 1.0),
            position_y=random.uniform(0.0, 1.0),
            vision_range=random.uniform(30, 150),
            target_preference=random.uniform(0.0, 1.0),
            cooperation=random.uniform(0.0, 1.0),
            ambush_mode=random.uniform(0.0, 1.0),
            rest_cycles=random.uniform(5, 30)
        )
    
    def get_actual_position(self) -> Tuple[float, float]:
        x = FROG_ZONE_X_MIN + self.position_x * (FROG_ZONE_X_MAX - FROG_ZONE_X_MIN)
        x *= ARENA_WIDTH
        y = FROG_ZONE_Y_MIN + self.position_y * (FROG_ZONE_Y_MAX - FROG_ZONE_Y_MIN)
        y *= ARENA_HEIGHT
        return x, y
    
    def get_tongue_length(self) -> float:
        return TONGUE_BASE_LENGTH * self.tongue_power
    
    def mutate(self, rate: float = None) -> 'Genome':
        if rate is None:
            rate = MUTATION_RATE
        
        params = {}
        for attr, (min_v, max_v) in self.RANGES.items():
            val = getattr(self, attr)
            if random.random() < rate:
                std = (max_v - min_v) * 0.15
                val += random.gauss(0, std)
                val = max(min_v, min(max_v, val))
            params[attr] = val
        
        return Genome(**params)
    
    @classmethod
    def blend_crossover(cls, p1: 'Genome', p2: 'Genome', alpha: float = 0.3) -> 'Genome':
        params = {}
        for attr, (min_v, max_v) in cls.RANGES.items():
            v1 = getattr(p1, attr)
            v2 = getattr(p2, attr)
            low, high = min(v1, v2), max(v1, v2)
            range_ext = (high - low) * alpha
            val = random.uniform(low - range_ext, high + range_ext)
            val = max(min_v, min(max_v, val))
            params[attr] = val
        return cls(**params)
    
    def to_dict(self) -> dict:
        return {attr: getattr(self, attr) for attr in self.RANGES.keys()}
    
    def __repr__(self):
        x, y = self.get_actual_position()
        return (f"Genome(pos=({x:.0f},{y:.0f}), tongue={self.get_tongue_length():.0f}, "
                f"vision={self.vision_range:.0f}, pref={self.target_preference:.2f}, "
                f"coop={self.cooperation:.2f}, ambush={self.ambush_mode:.2f}, "
                f"rest={self.rest_cycles:.0f})")

# ============================================================
# FLY
# ============================================================

class Fly:
    
    __slots__ = ['x', 'y', 'vx', 'vy', 'alive', 'escaped', 'marked_escaped', 
                 'caught_at_step', 'panic_timer', 'avoidance_side']
    
    def __init__(self, x: float = None, y: float = None):
        self.x = x if x is not None else random.uniform(10, SPAWN_ZONE_WIDTH)
        self.y = y if y is not None else random.uniform(15, ARENA_HEIGHT - 15)
        
        angle = random.uniform(-0.4, 0.4)
        speed = random.uniform(1.0, FLY_SPEED)
        self.vx = np.cos(angle) * speed
        self.vy = np.sin(angle) * speed
        
        self.alive = True
        self.escaped = False
        self.marked_escaped = False
        self.caught_at_step = -1
        
        self.panic_timer = 0
        self.avoidance_side = random.choice([-1, 1])    
    def update_vectorized(self, nearby_flies: List['Fly'], 
                          frog_data: np.ndarray,
                          frog_tongue_states: List[bool]):
        
        if not self.alive or self.escaped:
            return
        
        ax, ay = 0.0, 0.0
        
        if self.panic_timer > 0:
            self.panic_timer -= 1
        
        for other in nearby_flies:
            if other is self or not other.alive or other.escaped:
                continue
            dx = self.x - other.x
            dy = self.y - other.y
            dist_sq = dx*dx + dy*dy
            if 0 < dist_sq < 144:
                dist = np.sqrt(dist_sq)
                ax += (dx / dist) * 0.25
                ay += (dy / dist) * 0.25
        
        if len(nearby_flies) > 1 and random.random() < 0.3:
            other = random.choice(nearby_flies)
            if other is not self and other.alive and not other.escaped:
                ax += (other.vx - self.vx) * 0.02
                ay += (other.vy - self.vy) * 0.02
        
        ax += 0.04
        
        if len(frog_data) > 0:
            for i, (fx, fy) in enumerate(frog_data):
                dx = self.x - fx
                dy = self.y - fy
                dist_sq = dx*dx + dy*dy
                
                tongue_danger = 1.5 if frog_tongue_states[i] else 1.0
                detection_range = FLY_FROG_DETECTION_RANGE * tongue_danger
                
                if dist_sq < detection_range * detection_range:
                    dist = np.sqrt(dist_sq)
                    
                    if dist < 1:
                        dist = 1
                    
                    if dist < FLY_FROG_DANGER_RANGE:
                        self.panic_timer = 20
                        
                        flee_strength = (FLY_FROG_DANGER_RANGE - dist) / FLY_FROG_DANGER_RANGE
                        
                        ax += (dx / dist) * flee_strength * 0.4
                        ay += (dy / dist) * flee_strength * 0.4
                        
                        perp_x = -dy / dist * self.avoidance_side
                        perp_y = dx / dist * self.avoidance_side
                        
                        ax += perp_x * flee_strength * 0.3 * FLY_LATERAL_AVOIDANCE
                        ay += perp_y * flee_strength * 0.3 * FLY_LATERAL_AVOIDANCE
                        
                        if random.random() < 0.1:
                            self.avoidance_side *= -1
                    
                    elif dist < detection_range:
                        avoid_strength = (detection_range - dist) / detection_range
                        
                        angle_to_fly = np.arctan2(dy, dx)
                        
                        curve_angle = angle_to_fly + (np.pi / 3) * self.avoidance_side
                        
                        curve_x = np.cos(curve_angle)
                        curve_y = np.sin(curve_angle)
                        
                        ax += curve_x * avoid_strength * 0.15 * FLY_LATERAL_AVOIDANCE
                        ay += curve_y * avoid_strength * 0.15 * FLY_LATERAL_AVOIDANCE
                        
                        ax += (dx / dist) * avoid_strength * 0.05
                        ay += (dy / dist) * avoid_strength * 0.05
                        
                        if abs(dy) < 20 and dx < 0:
                            if self.y < ARENA_HEIGHT / 2:
                                self.avoidance_side = 1
                            else:
                                self.avoidance_side = -1
        
        margin = 15
        if self.y < margin:
            ay += (margin - self.y) / margin * 0.15
            self.avoidance_side = 1
        if self.y > ARENA_HEIGHT - margin:
            ay -= (self.y - (ARENA_HEIGHT - margin)) / margin * 0.15
            self.avoidance_side = -1
        
        if self.panic_timer > 0:
            ax += random.uniform(-0.1, 0.1)
            ay += random.uniform(-0.1, 0.1)
            speed_mult = 1.3
        else:
            speed_mult = 1.0
        
        self.vx += ax
        self.vy += ay
        
        max_speed = FLY_SPEED * speed_mult
        speed = np.sqrt(self.vx*self.vx + self.vy*self.vy)
        if speed > max_speed:
            self.vx = (self.vx / speed) * max_speed
            self.vy = (self.vy / speed) * max_speed
        
        if self.vx < 0.3:
            self.vx = 0.3
        
        self.x += self.vx
        self.y += self.vy
        
        self.y = max(5, min(ARENA_HEIGHT - 5, self.y))
        self.x = max(0, self.x)
        
        if self.x >= ESCAPE_ZONE_X and not self.escaped:
            self.escaped = True

# ============================================================
# FROG
# ============================================================

class Frog:
    __slots__ = ['x', 'y', 'start_x', 'start_y', 'genome', 'frog_id',
                 'tongue_length', 'vision_range', 'cooldown_time',
                 'tongue_out', 'tongue_x', 'tongue_y', 
                 'tongue_target_x', 'tongue_target_y', 'tongue_extending',
                 'cooldown', 'current_target', 'catches', 'attempts',
                 'catch_positions', 'catch_times']
    
    def __init__(self, genome: Genome, frog_id: int = 0):
        self.x, self.y = genome.get_actual_position()
        self.start_x = self.x
        self.start_y = self.y
        
        self.genome = genome
        self.frog_id = frog_id
        
        self.tongue_length = genome.get_tongue_length()
        self.vision_range = genome.vision_range
        self.cooldown_time = int(genome.rest_cycles)
        
        self.tongue_out = False
        self.tongue_x = self.x
        self.tongue_y = self.y
        self.tongue_target_x = self.x
        self.tongue_target_y = self.y
        self.tongue_extending = True
        self.cooldown = 0
        
        self.current_target: Optional[Fly] = None
        
        self.catches = 0
        self.attempts = 0
        self.catch_positions = []
        self.catch_times = []

    def update(self, flies: List[Fly], other_frogs: List['Frog'], 
               current_step: int, spatial_grid: SpatialGrid):
        if self.tongue_out:
            self._update_tongue(flies, current_step)
            return
        
        if self.cooldown > 0:
            self.cooldown -= 1
            return
        
        nearby = spatial_grid.get_nearby(self.x, self.y, self.vision_range)
        visible_flies = []
        
        for fly in nearby:
            if not fly.alive or fly.escaped:
                continue
            dx = fly.x - self.x
            dy = fly.y - self.y
            dist_sq = dx*dx + dy*dy
            if dist_sq <= self.vision_range * self.vision_range:
                visible_flies.append((fly, np.sqrt(dist_sq)))
        
        if not visible_flies:
            self.current_target = None
            return
        
        other_targets = set(f.current_target for f in other_frogs 
                          if f is not self and f.current_target)
        
        if self.genome.cooperation > 0.3 and other_targets:
            if random.random() < self.genome.cooperation:
                filtered = [(f, d) for f, d in visible_flies if f not in other_targets]
                if filtered:
                    visible_flies = filtered
        
        def score_fly(fly_dist):
            fly, dist = fly_dist
            dist_score = dist / self.vision_range
            progress_score = fly.x / ESCAPE_ZONE_X
            pref = self.genome.target_preference
            score = (1 - pref) * dist_score + pref * (1 - progress_score)
            if fly.x > ESCAPE_ZONE_X - 50:
                score -= 0.5
            return score
        
        visible_flies.sort(key=score_fly)
        self.current_target = visible_flies[0][0]
        target = self.current_target
        
        dx = target.x - self.x
        dy = target.y - self.y
        dist = np.sqrt(dx*dx + dy*dy)
        
        optimal_dist = self.tongue_length * (0.5 + 0.4 * (1 - self.genome.ambush_mode))
        
        should_attack = dist <= optimal_dist
        if target.x > ESCAPE_ZONE_X - 40:
            should_attack = dist <= self.tongue_length * 0.95
        
        if should_attack:
            travel_time = dist / TONGUE_SPEED
            pred_x = target.x + target.vx * travel_time * 0.7
            pred_y = target.y + target.vy * travel_time * 0.7
            
            self.tongue_out = True
            self.tongue_extending = True
            self.tongue_x = self.x
            self.tongue_y = self.y
            self.tongue_target_x = pred_x
            self.tongue_target_y = pred_y
            self.attempts += 1
    
    def _update_tongue(self, flies: List[Fly], current_step: int):
        if self.tongue_extending:
            dx = self.tongue_target_x - self.x
            dy = self.tongue_target_y - self.y
            dist = np.sqrt(dx*dx + dy*dy)
            
            if dist > 0:
                self.tongue_x += (dx / dist) * TONGUE_SPEED
                self.tongue_y += (dy / dist) * TONGUE_SPEED
            
            tongue_dist = np.sqrt((self.tongue_x - self.x)**2 + 
                                 (self.tongue_y - self.y)**2)
            
            if tongue_dist >= self.tongue_length or tongue_dist >= dist:
                self.tongue_extending = False
            
            for fly in flies:
                if not fly.alive or fly.escaped:
                    continue
                
                fly_dist = (fly.x - self.tongue_x)**2 + (fly.y - self.tongue_y)**2
                if fly_dist < 100:  # 10^2
                    fly.alive = False
                    fly.caught_at_step = current_step
                    self.catches += 1
                    self.catch_positions.append(fly.x)
                    self.catch_times.append(current_step)
        else:
            dx = self.x - self.tongue_x
            dy = self.y - self.tongue_y
            dist = np.sqrt(dx*dx + dy*dy)
            
            if dist > TONGUE_SPEED:
                self.tongue_x += (dx / dist) * TONGUE_SPEED
                self.tongue_y += (dy / dist) * TONGUE_SPEED
            else:
                self.tongue_out = False
                self.tongue_x = self.x
                self.tongue_y = self.y
                self.cooldown = self.cooldown_time
                self.current_target = None

# ============================================================
# SIMULATION
# ============================================================

class Simulation:
    def __init__(self, genomes: List[Genome], num_frogs: int = None):
        self.genomes = genomes
        self.num_frogs = num_frogs if num_frogs else min(NUM_FROGS, len(genomes))
        
        self.flies: List[Fly] = []
        self.frogs: List[Frog] = []
        
        self.spatial_grid = SpatialGrid(ARENA_WIDTH, ARENA_HEIGHT, SPATIAL_GRID_SIZE)
        
        self.frog_positions = np.zeros((self.num_frogs, 2))
        self.frog_tongue_states = [False] * self.num_frogs
        
        self.step_count = 0
        self.escaped_count = 0
        self.running = True
        
        self.first_catch_step = -1
        self.last_catch_step = -1
        self.all_caught_step = -1
        self.clear_step = -1
        self.perfect_clear = False
    
    def reset(self):
        self.flies = [Fly() for _ in range(NUM_FLIES)]
        self.frogs = []
        
        for i in range(self.num_frogs):
            genome = self.genomes[i % len(self.genomes)]
            frog = Frog(genome, frog_id=i)
            self.frogs.append(frog)
            self.frog_positions[i] = [frog.x, frog.y]
        
        self.step_count = 0
        self.escaped_count = 0
        self.running = True
        
        self.first_catch_step = -1
        self.last_catch_step = -1
        self.all_caught_step = -1
        self.clear_step = -1
        self.perfect_clear = False
    
    def step(self):
        if not self.running:
            return
        
        self.step_count += 1
        
        for i, frog in enumerate(self.frogs):
            self.frog_positions[i] = [frog.x, frog.y]
            self.frog_tongue_states[i] = frog.tongue_out
        
        self.spatial_grid.clear()
        for fly in self.flies:
            if fly.alive and not fly.escaped:
                self.spatial_grid.insert(fly, fly.x, fly.y)
        
        catches_before = sum(f.catches for f in self.frogs)
        
        for fly in self.flies:
            if not fly.alive or fly.escaped:
                continue
            
            nearby = self.spatial_grid.get_nearby(fly.x, fly.y, 30)
            
            was_escaped = fly.escaped
            fly.update_vectorized(nearby, self.frog_positions, self.frog_tongue_states)
            
            if fly.escaped and not was_escaped and not fly.marked_escaped:
                self.escaped_count += 1
                fly.marked_escaped = True
        
        for frog in self.frogs:
            frog.update(self.flies, self.frogs, self.step_count, self.spatial_grid)
        
        catches_after = sum(f.catches for f in self.frogs)
        
        if catches_after > catches_before:
            if self.first_catch_step == -1:
                self.first_catch_step = self.step_count
            self.last_catch_step = self.step_count
        
        active_flies = sum(1 for f in self.flies if f.alive and not f.escaped)
        caught_flies = sum(1 for f in self.flies if not f.alive)
        
        if caught_flies == NUM_FLIES and self.all_caught_step == -1:
            self.all_caught_step = self.step_count
            self.perfect_clear = True
        
        if active_flies == 0 or self.step_count >= MAX_STEPS:
            self.running = False
            self.clear_step = self.step_count
            if caught_flies == NUM_FLIES:
                self.perfect_clear = True
    
    def get_total_catches(self) -> int:
        return sum(f.catches for f in self.frogs)
    
    def get_time_stats(self) -> dict:
        total_catches = self.get_total_catches()
        
        if self.last_catch_step > 0:
            catch_speed = (total_catches / self.last_catch_step) * 100
        else:
            catch_speed = 0
        
        time_to_first = self.first_catch_step if self.first_catch_step > 0 else MAX_STEPS
        
        if self.perfect_clear:
            completion_time = self.all_caught_step
        else:
            completion_time = self.clear_step if self.clear_step > 0 else MAX_STEPS
        
        all_catch_times = []
        for frog in self.frogs:
            all_catch_times.extend(frog.catch_times)
        all_catch_times.sort()
        
        if len(all_catch_times) > 1:
            intervals = [all_catch_times[i+1] - all_catch_times[i] 
                        for i in range(len(all_catch_times)-1)]
            avg_interval = np.mean(intervals)
        else:
            avg_interval = MAX_STEPS
        
        return {
            'total_catches': total_catches,
            'escaped': self.escaped_count,
            'perfect_clear': self.perfect_clear,
            'first_catch_step': time_to_first,
            'last_catch_step': self.last_catch_step,
            'all_caught_step': self.all_caught_step,
            'completion_time': completion_time,
            'catch_speed': catch_speed,
            'avg_catch_interval': avg_interval,
            'steps_used': self.step_count
        }
    
    def get_team_fitness(self) -> float:
        catches = self.get_total_catches()
        time_stats = self.get_time_stats()
        
        fitness = catches * 100
        fitness -= self.escaped_count * 25
        
        if time_stats['perfect_clear']:
            fitness += PERFECT_CLEAR_BONUS
            speed_factor = max(0, (MAX_STEPS - time_stats['completion_time']) / MAX_STEPS)
            fitness += speed_factor * 150 * SPEED_BONUS_MULTIPLIER
        
        catch_speed = time_stats['catch_speed']
        fitness += catch_speed * 10 * SPEED_BONUS_MULTIPLIER
        
        if time_stats['first_catch_step'] > 0:
            first_catch_bonus = max(0, (100 - time_stats['first_catch_step']) / 100) * 30
            fitness += first_catch_bonus * SPEED_BONUS_MULTIPLIER
        
        if time_stats['avg_catch_interval'] < MAX_STEPS:
            consistency_bonus = max(0, (50 - time_stats['avg_catch_interval']) / 50) * 20
            fitness += consistency_bonus * SPEED_BONUS_MULTIPLIER
        
        active_catchers = sum(1 for f in self.frogs if f.catches > 0)
        if active_catchers > 1:
            fitness += active_catchers * 20
        
        if self.num_frogs > 1:
            positions = [(f.x, f.y) for f in self.frogs]
            min_dist = float('inf')
            for i in range(len(positions)):
                for j in range(i + 1, len(positions)):
                    dx = positions[i][0] - positions[j][0]
                    dy = positions[i][1] - positions[j][1]
                    dist = np.sqrt(dx*dx + dy*dy)
                    min_dist = min(min_dist, dist)
            if min_dist > 30:
                fitness += min(30, min_dist - 30)
        
        return max(0, fitness)

# ============================================================
# GENETIC ALGORITHM
# ============================================================

class GeneticAlgorithm:
    def __init__(self, population_size: int = POPULATION_SIZE, 
                 num_frogs: int = NUM_FROGS):
        self.population_size = population_size
        self.num_frogs = num_frogs
        
        self.population: List[List[Genome]] = []
        self.fitness_scores: List[float] = []
        
        self.generation = 0
        self.best_fitness = 0
        self.best_team: List[Genome] = []
        
        self.history = {
            'generation': [],
            'best_fitness': [],
            'avg_fitness': [],
            'total_catches': [],
            'total_escaped': [],
            'avg_completion_time': [],
            'perfect_clears': [],
            'avg_catch_speed': []
        }
    
    def initialize(self):
        self.population = []
        for _ in range(self.population_size):
            team = []
            for i in range(self.num_frogs):
                genome = Genome.random()
                genome.position_y = (i + 0.5) / self.num_frogs
                genome.position_y = max(0.1, min(0.9, 
                    genome.position_y + random.uniform(-0.1, 0.1)))
                team.append(genome)
            self.population.append(team)
        self.generation = 0
    
    def evaluate_team(self, team: List[Genome], num_trials: int = 2) -> Tuple[float, dict]:
        total_fitness = 0
        total_catches = 0
        total_escaped = 0
        total_completion_time = 0
        total_catch_speed = 0
        perfect_count = 0
        
        for _ in range(num_trials):
            sim = Simulation(team, num_frogs=self.num_frogs)
            sim.reset()
            
            while sim.running:
                sim.step()
            
            total_fitness += sim.get_team_fitness()
            stats = sim.get_time_stats()
            total_catches += stats['total_catches']
            total_escaped += stats['escaped']
            total_completion_time += stats['completion_time']
            total_catch_speed += stats['catch_speed']
            if stats['perfect_clear']:
                perfect_count += 1
        
        return (
            total_fitness / num_trials,
            {
                'catches': total_catches / num_trials,
                'escaped': total_escaped / num_trials,
                'completion_time': total_completion_time / num_trials,
                'catch_speed': total_catch_speed / num_trials,
                'perfect_clears': perfect_count
            }
        )
    
    def evaluate_population(self):
        self.fitness_scores = []
        total_catches = 0
        total_escaped = 0
        total_completion_time = 0
        total_catch_speed = 0
        total_perfect = 0
        
        for team in self.population:
            fitness, stats = self.evaluate_team(team)
            self.fitness_scores.append(fitness)
            total_catches += stats['catches']
            total_escaped += stats['escaped']
            total_completion_time += stats['completion_time']
            total_catch_speed += stats['catch_speed']
            total_perfect += stats['perfect_clears']
        
        best_idx = np.argmax(self.fitness_scores)
        if self.fitness_scores[best_idx] > self.best_fitness:
            self.best_fitness = self.fitness_scores[best_idx]
            self.best_team = [Genome(**g.to_dict()) for g in self.population[best_idx]]
        
        n = self.population_size
        self.history['generation'].append(self.generation)
        self.history['best_fitness'].append(max(self.fitness_scores))
        self.history['avg_fitness'].append(np.mean(self.fitness_scores))
        self.history['total_catches'].append(total_catches / n)
        self.history['total_escaped'].append(total_escaped / n)
        self.history['avg_completion_time'].append(total_completion_time / n)
        self.history['perfect_clears'].append(total_perfect)
        self.history['avg_catch_speed'].append(total_catch_speed / n)
    
    def select_parent(self) -> List[Genome]:
        tournament = random.sample(range(self.population_size), 3)
        winner = max(tournament, key=lambda i: self.fitness_scores[i])
        return self.population[winner]
    
    def evolve(self):
        sorted_indices = np.argsort(self.fitness_scores)[::-1]
        new_population = []
        
        for i in range(ELITE_COUNT):
            elite = [Genome(**g.to_dict()) for g in self.population[sorted_indices[i]]]
            new_population.append(elite)
        
        while len(new_population) < self.population_size:
            p1 = self.select_parent()
            p2 = self.select_parent()
            
            child = []
            for i in range(self.num_frogs):
                if random.random() < CROSSOVER_RATE:
                    c = Genome.blend_crossover(p1[i], p2[i])
                else:
                    c = Genome(**random.choice([p1[i], p2[i]]).to_dict())
                child.append(c.mutate())
            
            new_population.append(child)
        
        self.population = new_population
        self.generation += 1
    
    def get_best_team(self) -> List[Genome]:
        if self.best_team:
            return self.best_team
        best_idx = np.argmax(self.fitness_scores)
        return self.population[best_idx]

# ============================================================
# VISUALIZATION
# ============================================================

class Game:
    def __init__(self, num_frogs: int = NUM_FROGS):
        pygame.init()
        
        self.num_frogs = num_frogs
        self.width = ARENA_WIDTH * SCALE
        self.height = ARENA_HEIGHT * SCALE + 120
        
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption(f"Fly Catcher - {NUM_FLIES} Flies, {num_frogs} Frogs")
        
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 26)
        self.small_font = pygame.font.Font(None, 20)
        self.tiny_font = pygame.font.Font(None, 16)
        
        self.fly_surface = pygame.Surface((10, 10), pygame.SRCALPHA)
        pygame.draw.circle(self.fly_surface, BLACK, (5, 5), 4)
        pygame.draw.ellipse(self.fly_surface, GRAY, (0, 3, 4, 3))
        pygame.draw.ellipse(self.fly_surface, GRAY, (6, 3, 4, 3))
        
        self.running = True
        self.paused = False
    
    def scale(self, x: float, y: float) -> Tuple[int, int]:
        return int(x * SCALE), int(y * SCALE)
    
    def handle_events(self) -> str:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                return 'quit'
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                    return 'quit'
                elif event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                    return 'pause'
                elif event.key == pygame.K_RIGHT:
                    return 'skip'
                elif event.key == pygame.K_UP:
                    return 'speed_up'
                elif event.key == pygame.K_DOWN:
                    return 'speed_down'
        return 'ok'
    
    def draw(self, sim: Simulation, generation: int, best_fitness: float, speed: int = 1):
        
        self.screen.fill(SKY_BLUE)
        
        zone_x1 = int(FROG_ZONE_X_MIN * ARENA_WIDTH * SCALE)
        zone_x2 = int(FROG_ZONE_X_MAX * ARENA_WIDTH * SCALE)
        pygame.draw.rect(self.screen, (220, 240, 255), 
                        (zone_x1, 0, zone_x2 - zone_x1, ARENA_HEIGHT * SCALE))
        
        pygame.draw.rect(self.screen, SPAWN_ZONE_COLOR, 
                        (0, 0, SPAWN_ZONE_WIDTH * SCALE, ARENA_HEIGHT * SCALE))
        
        escape_x = int(ESCAPE_ZONE_X * SCALE)
        pygame.draw.rect(self.screen, RED_ZONE, 
                        (escape_x, 0, self.width - escape_x, ARENA_HEIGHT * SCALE))
        
        pygame.draw.rect(self.screen, DARK_GREEN, 
                        (0, ARENA_HEIGHT * SCALE - 10, self.width, 10))
        
        self.screen.blit(self.tiny_font.render("SPAWN", True, DARK_GREEN), (5, 5))
        self.screen.blit(self.tiny_font.render("ESCAPE", True, (150, 50, 50)), (escape_x + 5, 5))
        
        for fly in sim.flies:
            if fly.alive and not fly.escaped:
                pos = self.scale(fly.x, fly.y)
                color = ORANGE if fly.panic_timer > 0 else BLACK
                pygame.draw.circle(self.screen, color, pos, 3)
        
        for i, frog in enumerate(sim.frogs):
            color = FROG_COLORS[i % len(FROG_COLORS)]
            pos = self.scale(frog.x, frog.y)
            
            pygame.draw.circle(self.screen, (*color, 30), pos, 
                             int(frog.vision_range * SCALE), 1)
            
            if frog.tongue_out:
                tongue_pos = self.scale(frog.tongue_x, frog.tongue_y)
                pygame.draw.line(self.screen, PINK, pos, tongue_pos, 4)
                pygame.draw.circle(self.screen, PINK, tongue_pos, 6)
            
            body_radius = int(FROG_SIZE * SCALE / 2) + 3
            pygame.draw.circle(self.screen, color, pos, body_radius)
            pygame.draw.circle(self.screen, DARK_GREEN, pos, body_radius, 2)
            
            eye_offset = body_radius // 2
            for dx in [-eye_offset, eye_offset]:
                eye_pos = (pos[0] + dx, pos[1] - eye_offset)
                pygame.draw.circle(self.screen, WHITE, eye_pos, 4)
                pygame.draw.circle(self.screen, BLACK, eye_pos, 2)
            
            catch_text = self.small_font.render(str(frog.catches), True, WHITE)
            pygame.draw.circle(self.screen, color, 
                             (pos[0], pos[1] - body_radius - 12), 10)
            self.screen.blit(catch_text, (pos[0] - 5, pos[1] - body_radius - 18))
        
        panel_y = ARENA_HEIGHT * SCALE + 5
        total_catches = sim.get_total_catches()
        
        self.screen.blit(self.font.render(f"Gen: {generation}", True, BLACK), (10, panel_y))
        self.screen.blit(self.font.render(f"Catches: {total_catches}/{NUM_FLIES}", 
                                         True, DARK_GREEN), (100, panel_y))
        self.screen.blit(self.font.render(f"Escaped: {sim.escaped_count}", 
                                         True, (180, 50, 50)), (280, panel_y))
        self.screen.blit(self.font.render(f"Step: {sim.step_count}/{MAX_STEPS}", 
                                         True, GRAY), (420, panel_y))
        self.screen.blit(self.font.render(f"Speed: {speed}x", True, GRAY), (580, panel_y))
        
        panel_y2 = panel_y + 24
        time_stats = sim.get_time_stats()
        
        speed_color = GOLD if time_stats['catch_speed'] > 5 else BLACK
        self.screen.blit(self.font.render(f"Rate: {time_stats['catch_speed']:.1f}/100steps", 
                                         True, speed_color), (10, panel_y2))
        
        if time_stats['first_catch_step'] > 0:
            self.screen.blit(self.small_font.render(
                f"1st: step {time_stats['first_catch_step']}", True, BLUE), (200, panel_y2 + 3))
        
        if sim.perfect_clear or total_catches == NUM_FLIES:
            self.screen.blit(self.font.render("⭐ PERFECT!", True, GOLD), (350, panel_y2))
        
        active = sum(1 for f in sim.flies if f.alive and not f.escaped)
        panicking = sum(1 for f in sim.flies if f.alive and not f.escaped and f.panic_timer > 0)
        self.screen.blit(self.small_font.render(
            f"Active: {active} (Panicking: {panicking})", True, GRAY), (480, panel_y2 + 3))
        
        panel_y3 = panel_y2 + 24
        fitness = sim.get_team_fitness()
        self.screen.blit(self.font.render(f"Fitness: {fitness:.0f}", True, BLUE), (10, panel_y3))
        self.screen.blit(self.font.render(f"Best: {best_fitness:.0f}", 
                                         True, (50, 150, 50)), (150, panel_y3))
        
        frog_stats = " | ".join([f"F{i+1}:{f.catches}/{f.attempts}" 
                                for i, f in enumerate(sim.frogs)])
        self.screen.blit(self.small_font.render(frog_stats, True, DARK_GREEN), (320, panel_y3 + 3))
        
        panel_y4 = panel_y3 + 24
        self.screen.blit(self.small_font.render(
            "SPACE=Pause  →=Skip  ↑↓=Speed  ESC=Quit", True, GRAY), (10, panel_y4))
        
        bar_w, bar_h = 200, 10
        bar_x = self.width - bar_w - 20
        pygame.draw.rect(self.screen, GRAY, (bar_x, panel_y4, bar_w, bar_h))
        progress = sim.step_count / MAX_STEPS
        prog_color = DARK_GREEN if total_catches == NUM_FLIES else GREEN
        pygame.draw.rect(self.screen, prog_color, (bar_x, panel_y4, int(bar_w * progress), bar_h))
        pygame.draw.rect(self.screen, BLACK, (bar_x, panel_y4, bar_w, bar_h), 1)
        
        if self.paused:
            pause_surf = self.font.render("⏸ PAUSED", True, (200, 50, 50))
            rect = pause_surf.get_rect(center=(self.width // 2, ARENA_HEIGHT * SCALE // 2))
            pygame.draw.rect(self.screen, WHITE, rect.inflate(20, 10))
            pygame.draw.rect(self.screen, (200, 50, 50), rect.inflate(20, 10), 2)
            self.screen.blit(pause_surf, rect)
        
        pygame.display.flip()
    
    def close(self):
        pygame.quit()

# ============================================================
# MAIN
# ============================================================

def main():
    print("\n" + "=" * 70)
    print("          🐸 FLY CATCHER - GENETIC ALGORITHM TRAINING 🪰")
    print("=" * 70)
    print(f"\n  Configuration:")
    print(f"    • Flies: {NUM_FLIES}")
    print(f"    • Frogs: {NUM_FROGS}")
    print(f"    • Population: {POPULATION_SIZE}")
    print(f"    • Generations: {NUM_GENERATIONS}")
    print(f"\n  Fly Behavior:")
    print(f"    • Detection range: {FLY_FROG_DETECTION_RANGE}")
    print(f"    • Danger zone: {FLY_FROG_DANGER_RANGE}")
    print(f"    • Lateral avoidance: {FLY_LATERAL_AVOIDANCE}")
    print(f"    • Panicking flies shown in ORANGE")
    print(f"\n  Controls: SPACE=Pause  →=Skip  ↑↓=Speed  ESC=Quit")
    print("=" * 70 + "\n")
    
    game = Game(num_frogs=NUM_FROGS)
    ga = GeneticAlgorithm(population_size=POPULATION_SIZE, num_frogs=NUM_FROGS)
    ga.initialize()
    
    speed = 2
    
    for gen in range(NUM_GENERATIONS):
        if not game.running:
            break
        
        print(f"\n{'─'*60}")
        print(f"  GENERATION {gen + 1}/{NUM_GENERATIONS}")
        print(f"{'─'*60}")
        
        ga.evaluate_population()
        
        best_fitness = max(ga.fitness_scores)
        avg_fitness = np.mean(ga.fitness_scores)
        avg_time = ga.history['avg_completion_time'][-1]
        perfect_count = ga.history['perfect_clears'][-1]
        catch_speed = ga.history['avg_catch_speed'][-1]
        
        print(f"  📈 Fitness   - Best: {best_fitness:.1f}  Avg: {avg_fitness:.1f}")
        print(f"  ⏱️  Speed     - Completion: {avg_time:.0f}steps  Rate: {catch_speed:.2f}/100")
        print(f"  ⭐ Perfects  - {perfect_count} teams")
        
        best_idx = np.argmax(ga.fitness_scores)
        best_team = ga.population[best_idx]
        
        for i, genome in enumerate(best_team):
            print(f"  🐸 F{i+1}: {genome}")
        
        sim = Simulation(best_team, num_frogs=NUM_FROGS)
        sim.reset()
        
        while game.running and sim.running:
            event = game.handle_events()
            
            if event == 'quit':
                break
            elif event == 'skip':
                break
            elif event == 'speed_up':
                speed = min(15, speed + 1)
            elif event == 'speed_down':
                speed = max(1, speed - 1)
            
            if not game.paused:
                for _ in range(speed):
                    sim.step()
                    if not sim.running:
                        break
            
            game.draw(sim, gen + 1, ga.best_fitness, speed)
            game.clock.tick(60)
        
        if not game.running:
            break
        
        final_stats = sim.get_time_stats()
        if final_stats['perfect_clear']:
            print(f"  ✅ Perfect clear in {final_stats['completion_time']} steps!")
        else:
            print(f"  📊 {final_stats['total_catches']}/{NUM_FLIES} caught, "
                  f"{final_stats['escaped']} escaped")
        
        pygame.time.wait(150)
        ga.evolve()
    
    game.close()
    
    print("\n" + "=" * 70)
    print("                    ✅ TRAINING COMPLETE!")
    print("=" * 70)
    print(f"\n  📊 Results:")
    print(f"    Best fitness: {ga.best_fitness:.1f}")
    print(f"    Best catch speed: {max(ga.history['avg_catch_speed']):.2f}/100 steps")
    print(f"    Most perfect clears: {max(ga.history['perfect_clears'])}")
    
    print(f"\n  🏆 Best Team:")
    for i, genome in enumerate(ga.get_best_team()):
        x, y = genome.get_actual_position()
        print(f"    Frog {i+1}: pos=({x:.0f},{y:.0f}) tongue={genome.get_tongue_length():.0f}")
    
    os.makedirs('results', exist_ok=True)
    
    with open('results/training_data.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['generation', 'best_fitness', 'avg_fitness', 
                        'avg_catches', 'avg_escaped', 'avg_completion_time',
                        'perfect_clears', 'avg_catch_speed'])
        for i in range(len(ga.history['generation'])):
            writer.writerow([
                ga.history['generation'][i],
                ga.history['best_fitness'][i],
                ga.history['avg_fitness'][i],
                ga.history['total_catches'][i],
                ga.history['total_escaped'][i],
                ga.history['avg_completion_time'][i],
                ga.history['perfect_clears'][i],
                ga.history['avg_catch_speed'][i]
            ])
    print(f"\n  💾 Saved to results/training_data.csv")
    
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        gens = ga.history['generation']
        
        axes[0,0].plot(gens, ga.history['best_fitness'], 'g-', lw=2, label='Best')
        axes[0,0].plot(gens, ga.history['avg_fitness'], 'b--', lw=2, label='Avg')
        axes[0,0].set_title('Fitness Evolution')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        
        axes[0,1].plot(gens, ga.history['total_catches'], 'g-', lw=2, label='Catches')
        axes[0,1].plot(gens, ga.history['total_escaped'], 'r-', lw=2, label='Escaped')
        axes[0,1].axhline(NUM_FLIES, color='gray', ls=':', alpha=0.5)
        axes[0,1].set_title('Catches vs Escaped')
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)
        
        axes[1,0].plot(gens, ga.history['avg_catch_speed'], 'orange', lw=2)
        axes[1,0].fill_between(gens, 0, ga.history['avg_catch_speed'], alpha=0.3, color='orange')
        axes[1,0].set_title('Catch Speed (per 100 steps)')
        axes[1,0].grid(True, alpha=0.3)
        
        axes[1,1].bar(gens, ga.history['perfect_clears'], color='gold', edgecolor='orange')
        axes[1,1].set_title('Perfect Clears per Generation')
        axes[1,1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig('results/evolution_plot.png', dpi=150)
        print(f"  📊 Saved to results/evolution_plot.png")
        plt.show()
    except ImportError:
        pass
    
    print("\n" + "=" * 70 + "\n")


if __name__ == '__main__':
    main()

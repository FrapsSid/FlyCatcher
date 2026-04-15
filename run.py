"""
Fly Catcher - Genetic Algorithm Training

A simulation where genetically-evolved frogs learn to catch flies in a 2D arena.
Frogs use tongues to catch flies that spawn on the left and try to escape to the right.
A genetic algorithm optimizes frog parameters (position, tongue length, vision, etc.)
over multiple generations to maximize fly catches.
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

# --- Arena dimensions ---
ARENA_WIDTH = 500       # Width of the simulation arena (in simulation units)
ARENA_HEIGHT = 150      # Height of the simulation arena (in simulation units)
SCALE = 3               # Pixel scale factor for rendering (sim units - pixels)

# --- Fly parameters ---
NUM_FLIES = 50          # Number of flies per simulation (keep <=50 to avoid lag)
FLY_SPEED = 2.2         # Maximum fly movement speed per step
FLY_SIZE = 3            # Visual size of flies (for rendering)

# --- Frog parameters ---
NUM_FROGS = 3           # Number of frogs per team
FROG_SIZE = 12          # Visual size of frogs (for rendering)

# --- Tongue mechanics ---
TONGUE_BASE_LENGTH = 20   # Base tongue reach before genome multiplier
TONGUE_SPEED = 14         # Speed at which the tongue extends/retracts per step
BASE_COOLDOWN = 15         # Base cooldown frames between tongue attacks

# --- Frog placement constraints (as fractions of arena dimensions) ---
FROG_ZONE_X_MIN = 0.35   # Leftmost x-position frogs can be placed (fraction)
FROG_ZONE_X_MAX = 0.85   # Rightmost x-position frogs can be placed (fraction)
FROG_ZONE_Y_MIN = 0.10   # Topmost y-position frogs can be placed (fraction)
FROG_ZONE_Y_MAX = 0.90   # Bottommost y-position frogs can be placed (fraction)

# --- Zone boundaries ---
ESCAPE_ZONE_X = ARENA_WIDTH - 40   # X-coordinate where flies escape (right side)
SPAWN_ZONE_WIDTH = 60               # Width of the left-side spawn area for flies
MAX_STEPS = 600                      # Maximum simulation steps before timeout

# --- Genetic Algorithm hyperparameters ---
POPULATION_SIZE = 24      # Number of frog teams in each generation
NUM_GENERATIONS = 40      # Total number of generations to evolve
MUTATION_RATE = 0.2        # Probability of mutating each gene
CROSSOVER_RATE = 0.8       # Probability of crossover vs. cloning a parent
ELITE_COUNT = 3            # Number of top teams carried unchanged to next generation

# --- Time/speed bonus settings for fitness calculation ---
PERFECT_CLEAR_BONUS = 200        # Bonus fitness for catching ALL flies
SPEED_BONUS_MULTIPLIER = 0.5    # Multiplier for all speed-related fitness bonuses

# --- Fly avoidance behavior (how flies react to frogs) ---
FLY_FROG_DETECTION_RANGE = 80   # Distance at which flies start noticing frogs
FLY_FROG_DANGER_RANGE = 40      # Distance at which flies enter panic mode
FLY_LATERAL_AVOIDANCE = 0.4     # Strength of sideways dodging when avoiding frogs

# --- Performance optimization ---
SPATIAL_GRID_SIZE = 30   # Cell size for spatial hash grid (neighbor lookups)

# --- Color definitions (RGB tuples) ---
WHITE = (255, 255, 255)
SKY_BLUE = (135, 206, 235)          # Background color
GREEN = (50, 205, 50)                # General green
DARK_GREEN = (34, 139, 34)           # Darker green for outlines/ground
PINK = (255, 105, 180)               # Tongue color
BLACK = (20, 20, 20)                 # Fly color (normal state)
RED_ZONE = (255, 200, 200)           # Escape zone background
SPAWN_ZONE_COLOR = (200, 255, 200)   # Spawn zone background
GRAY = (100, 100, 100)               # UI text color
BLUE = (50, 50, 200)                 # Fitness display color
GOLD = (255, 215, 0)                 # Perfect clear / high performance color
ORANGE = (255, 165, 0)               # Panicking fly color

# Colors for each frog
FROG_COLORS = [
    (50, 205, 50),     # Lime green
    (50, 150, 50),     # Forest green
    (100, 200, 100),   # Light green
    (30, 180, 80),     # Teal green
    (80, 220, 80),     # Bright green
]

# ============================================================
# SPATIAL GRID
# ============================================================

class SpatialGrid:
    """
    Spatial hash grid for efficient neighbor queries.
    
    Divides the arena into a grid of cells. Objects are inserted into their
    corresponding cell, enabling fast "find nearby objects" queries without
    checking every object in the simulation. This is critical for performance
    when many flies need to check distances to each other and to frogs.
    """
    
    def __init__(self, width: float, height: float, cell_size: float):
        """
        Initialize the spatial grid.
        
        Args:
            width: Total width of the space to partition.
            height: Total height of the space to partition.
            cell_size: Size of each grid cell (larger = fewer cells but less precise).
        """
        self.cell_size = cell_size
        self.cols = int(np.ceil(width / cell_size)) + 1   # Number of columns
        self.rows = int(np.ceil(height / cell_size)) + 1  # Number of rows
        self.grid = {}  # Dictionary mapping (col, row) -> list of objects
    
    def clear(self):
        """Remove all objects from the grid. Called each simulation step."""
        self.grid.clear()
    
    def _get_cell(self, x: float, y: float) -> Tuple[int, int]:
        """Convert world coordinates to grid cell indices."""
        return (int(x / self.cell_size), int(y / self.cell_size))
    
    def insert(self, obj, x: float, y: float):
        """
        Insert an object into the grid at the given position.
        
        Args:
            obj: Any object to store (typically a Fly instance).
            x, y: World-space coordinates of the object.
        """
        cell = self._get_cell(x, y)
        if cell not in self.grid:
            self.grid[cell] = []
        self.grid[cell].append(obj)
    
    def get_nearby(self, x: float, y: float, radius: float) -> List:
        """
        Get all objects within a given radius of a point.
        
        Checks all grid cells that could potentially contain objects within
        the radius, then returns all objects in those cells. Note: this returns
        objects in nearby cells (a rough filter); callers should do exact
        distance checks if needed.
        
        Args:
            x, y: Center point of the query.
            radius: Search radius.
            
        Returns:
            List of all objects in cells overlapping the search area.
        """
        nearby = []
        # How many cells in each direction we need to check
        cell_radius = int(np.ceil(radius / self.cell_size))
        center_cell = self._get_cell(x, y)
        
        # Iterate over all cells in the square neighborhood
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
    """
    Represents the genetic blueprint for a single frog.
    
    Each gene controls a different aspect of the frog's behavior and placement.
    Genomes are evolved through crossover and mutation by the genetic algorithm.
    
    Attributes:
        tongue_power: Multiplier for tongue reach (0.5-2.0x base length).
        position_x: Normalized x-position within the frog zone (0.0-1.0).
        position_y: Normalized y-position within the frog zone (0.0-1.0).
        vision_range: How far the frog can detect flies (15-50 units).
        target_preference: Bias between targeting nearest fly (0.0) vs.
                          most-advanced fly (1.0).
        cooperation: Tendency to avoid targeting flies another frog is after (0.0-1.0).
        ambush_mode: Patience level - higher values wait for flies to get closer (0.0-1.0).
        rest_cycles: Cooldown frames between tongue attacks (5-30).
    """
    tongue_power: float
    position_x: float
    position_y: float
    vision_range: float
    target_preference: float
    cooperation: float
    ambush_mode: float
    rest_cycles: float
    
    # Valid ranges for each gene (used for clamping and mutation)
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
        """Clamp all gene values to their valid ranges after initialization."""
        for attr, (min_v, max_v) in self.RANGES.items():
            val = getattr(self, attr)
            setattr(self, attr, max(min_v, min(max_v, val)))
    
    @classmethod
    def random(cls) -> 'Genome':
        """Create a genome with all genes randomly initialized within valid ranges."""
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
        """
        Convert normalized genome position to actual arena coordinates.
        
        Maps position_x and position_y from [0,1] range into the
        defined frog placement zone within the arena.
        
        Returns:
            (x, y) tuple in arena coordinates.
        """
        # Interpolate within the frog zone boundaries
        x = FROG_ZONE_X_MIN + self.position_x * (FROG_ZONE_X_MAX - FROG_ZONE_X_MIN)
        x *= ARENA_WIDTH
        y = FROG_ZONE_Y_MIN + self.position_y * (FROG_ZONE_Y_MAX - FROG_ZONE_Y_MIN)
        y *= ARENA_HEIGHT
        return x, y
    
    def get_tongue_length(self) -> float:
        """Calculate actual tongue reach by applying the power multiplier to base length."""
        return TONGUE_BASE_LENGTH * self.tongue_power
    
    def mutate(self, rate: float = None) -> 'Genome':
        """
        Create a mutated copy of this genome.
        
        Each gene has an independent chance (determined by rate) of being
        perturbed by Gaussian noise. The noise standard deviation is 15%
        of each gene's valid range.
        
        Args:
            rate: Mutation probability per gene (defaults to global MUTATION_RATE).
            
        Returns:
            A new Genome instance with potentially mutated values.
        """
        if rate is None:
            rate = MUTATION_RATE
        
        params = {}
        for attr, (min_v, max_v) in self.RANGES.items():
            val = getattr(self, attr)
            if random.random() < rate:
                # Gaussian perturbation scaled to 15% of the gene's range
                std = (max_v - min_v) * 0.15
                val += random.gauss(0, std)
                val = max(min_v, min(max_v, val))  # Clamp to valid range
            params[attr] = val
        
        return Genome(**params)
    
    @classmethod
    def blend_crossover(cls, p1: 'Genome', p2: 'Genome', alpha: float = 0.3) -> 'Genome':
        """
        Create a child genome by blending two parents (BLX-α crossover).
        
        For each gene, the child's value is sampled uniformly from an interval
        that extends slightly beyond the parents' values (controlled by alpha).
        This allows exploration beyond the parents' gene ranges.
        
        Args:
            p1, p2: Parent genomes.
            alpha: Extension factor beyond the parents' interval (0.3 = 30%).
            
        Returns:
            A new child Genome.
        """
        params = {}
        for attr, (min_v, max_v) in cls.RANGES.items():
            v1 = getattr(p1, attr)
            v2 = getattr(p2, attr)
            low, high = min(v1, v2), max(v1, v2)
            # Extend the interval by alpha * range on each side
            range_ext = (high - low) * alpha
            val = random.uniform(low - range_ext, high + range_ext)
            val = max(min_v, min(max_v, val))  # Clamp to valid range
            params[attr] = val
        return cls(**params)
    
    def to_dict(self) -> dict:
        """Convert genome to a dictionary of gene name -> value pairs."""
        return {attr: getattr(self, attr) for attr in self.RANGES.keys()}
    
    def __repr__(self):
        """Human-readable representation showing key genome parameters."""
        x, y = self.get_actual_position()
        return (f"Genome(pos=({x:.0f},{y:.0f}), tongue={self.get_tongue_length():.0f}, "
                f"vision={self.vision_range:.0f}, pref={self.target_preference:.2f}, "
                f"coop={self.cooperation:.2f}, ambush={self.ambush_mode:.2f}, "
                f"rest={self.rest_cycles:.0f})")

# ============================================================
# FLY
# ============================================================

class Fly:
    """
    Represents a single fly in the simulation.
    
    Flies spawn on the left side and move rightward, trying to reach the
    escape zone. They exhibit flocking behavior (separation from other flies,
    velocity alignment) and actively avoid frogs - especially those with
    extended tongues. Panicking flies move faster and more erratically.
    
    Uses __slots__ for memory efficiency when managing many fly instances.
    """
    
    __slots__ = ['x', 'y', 'vx', 'vy', 'alive', 'escaped', 'marked_escaped', 
                 'caught_at_step', 'panic_timer', 'avoidance_side']
    
    def __init__(self, x: float = None, y: float = None):
        """
        Initialize a fly at a given or random position in the spawn zone.
        
        Args:
            x: Initial x-position (random within spawn zone if None).
            y: Initial y-position (random within arena if None).
        """
        # Spawn within the left-side spawn zone
        self.x = x if x is not None else random.uniform(10, SPAWN_ZONE_WIDTH)
        self.y = y if y is not None else random.uniform(15, ARENA_HEIGHT - 15)
        
        # Initial velocity: mostly rightward with slight random angle
        angle = random.uniform(-0.4, 0.4)  # Small angle spread around 0 (rightward)
        speed = random.uniform(1.0, FLY_SPEED)
        self.vx = np.cos(angle) * speed
        self.vy = np.sin(angle) * speed
        
        # State flags
        self.alive = True           # False once caught by a frog
        self.escaped = False        # True once the fly reaches the escape zone
        self.marked_escaped = False # Prevents double-counting escapes
        self.caught_at_step = -1    # Step number when caught (-1 if not caught)
        
        # Avoidance behavior state
        self.panic_timer = 0                       # Countdown timer for panic state
        self.avoidance_side = random.choice([-1, 1])  # Which side to dodge (up or down)
    
    def update_vectorized(self, nearby_flies: List['Fly'], 
                          frog_data: np.ndarray,
                          frog_tongue_states: List[bool]):
        """
        Update fly position and velocity for one simulation step.
        
        Applies multiple behavioral forces:
        1. Separation from nearby flies (avoid crowding)
        2. Velocity alignment with nearby flies (weak flocking)
        3. Rightward drift (flies want to escape right)
        4. Frog avoidance (flee from nearby frogs, especially with tongues out)
        5. Wall avoidance (stay away from top/bottom edges)
        6. Panic behavior (faster, more erratic movement when in danger)
        
        Args:
            nearby_flies: List of flies in adjacent spatial grid cells.
            frog_data: Nx2 numpy array of frog (x, y) positions.
            frog_tongue_states: List of booleans indicating if each frog's tongue is out.
        """
        if not self.alive or self.escaped:
            return
        
        # Accumulated acceleration for this step
        ax, ay = 0.0, 0.0
        
        # Decrement panic timer each step
        if self.panic_timer > 0:
            self.panic_timer -= 1
        
        # --- Force 1: Separation from nearby flies ---
        # Push away from flies that are too close (within 12 units)
        for other in nearby_flies:
            if other is self or not other.alive or other.escaped:
                continue
            dx = self.x - other.x
            dy = self.y - other.y
            dist_sq = dx*dx + dy*dy
            if 0 < dist_sq < 144:  # 144 = 12^2 (separation radius)
                dist = np.sqrt(dist_sq)
                ax += (dx / dist) * 0.25  # Push away proportionally
                ay += (dy / dist) * 0.25
        
        # --- Force 2: Velocity alignment (weak flocking) ---
        # Occasionally match velocity with a random nearby fly
        if len(nearby_flies) > 1 and random.random() < 0.3:
            other = random.choice(nearby_flies)
            if other is not self and other.alive and not other.escaped:
                ax += (other.vx - self.vx) * 0.02  # Weak alignment force
                ay += (other.vy - self.vy) * 0.02
        
        # --- Force 3: Rightward drift (escape motivation) ---
        ax += 0.04  # Constant gentle push to the right
        
        # --- Force 4: Frog avoidance ---
        if len(frog_data) > 0:
            for i, (fx, fy) in enumerate(frog_data):
                dx = self.x - fx  # Vector from frog to fly
                dy = self.y - fy
                dist_sq = dx*dx + dy*dy
                
                # Frogs with tongue out are detected from further away (1.5x range)
                tongue_danger = 1.5 if frog_tongue_states[i] else 1.0
                detection_range = FLY_FROG_DETECTION_RANGE * tongue_danger
                
                if dist_sq < detection_range * detection_range:
                    dist = np.sqrt(dist_sq)
                    
                    if dist < 1:
                        dist = 1  # Prevent division by zero
                    
                    if dist < FLY_FROG_DANGER_RANGE:
                        # === DANGER ZONE: Strong panic response ===
                        self.panic_timer = 20  # Enter panic state for 20 steps
                        
                        # Flee strength increases as fly gets closer to frog
                        flee_strength = (FLY_FROG_DANGER_RANGE - dist) / FLY_FROG_DANGER_RANGE
                        
                        # Direct flee: push directly away from frog
                        ax += (dx / dist) * flee_strength * 0.4
                        ay += (dy / dist) * flee_strength * 0.4
                        
                        # Lateral dodge: move perpendicular to frog direction
                        # This makes flies curve around frogs instead of just backing up
                        perp_x = -dy / dist * self.avoidance_side
                        perp_y = dx / dist * self.avoidance_side
                        
                        ax += perp_x * flee_strength * 0.3 * FLY_LATERAL_AVOIDANCE
                        ay += perp_y * flee_strength * 0.3 * FLY_LATERAL_AVOIDANCE
                        
                        # Occasionally switch dodge direction (unpredictability)
                        if random.random() < 0.1:
                            self.avoidance_side *= -1
                    
                    elif dist < detection_range:
                        # === DETECTION ZONE: Gentle avoidance with curved path ===
                        avoid_strength = (detection_range - dist) / detection_range
                        
                        # Calculate a curved avoidance path (angled away from frog)
                        angle_to_fly = np.arctan2(dy, dx)
                        curve_angle = angle_to_fly + (np.pi / 3) * self.avoidance_side
                        
                        curve_x = np.cos(curve_angle)
                        curve_y = np.sin(curve_angle)
                        
                        # Apply curved avoidance force
                        ax += curve_x * avoid_strength * 0.15 * FLY_LATERAL_AVOIDANCE
                        ay += curve_y * avoid_strength * 0.15 * FLY_LATERAL_AVOIDANCE
                        
                        # Slight direct push away from frog
                        ax += (dx / dist) * avoid_strength * 0.05
                        ay += (dy / dist) * avoid_strength * 0.05
                        
                        # Smart dodge direction: if frog is directly ahead and close
                        # vertically, dodge toward the nearest arena edge
                        if abs(dy) < 20 and dx < 0:  # Frog is ahead and roughly same height
                            if self.y < ARENA_HEIGHT / 2:
                                self.avoidance_side = 1   # Dodge downward
                            else:
                                self.avoidance_side = -1  # Dodge upward
        
        # --- Force 5: Wall avoidance (top/bottom boundaries) ---
        margin = 15  # Distance from edge where wall avoidance kicks in
        if self.y < margin:
            ay += (margin - self.y) / margin * 0.15  # Push away from top
            self.avoidance_side = 1  # Prefer dodging downward near top
        if self.y > ARENA_HEIGHT - margin:
            ay -= (self.y - (ARENA_HEIGHT - margin)) / margin * 0.15  # Push away from bottom
            self.avoidance_side = -1  # Prefer dodging upward near bottom
        
        # --- Force 6: Panic jitter and speed boost ---
        if self.panic_timer > 0:
            # Add random jitter for unpredictable panic movement
            ax += random.uniform(-0.1, 0.1)
            ay += random.uniform(-0.1, 0.1)
            speed_mult = 1.3  # 30% speed boost when panicking
        else:
            speed_mult = 1.0
        
        # --- Apply accumulated acceleration to velocity ---
        self.vx += ax
        self.vy += ay
        
        # Clamp speed to maximum (with panic multiplier)
        max_speed = FLY_SPEED * speed_mult
        speed = np.sqrt(self.vx*self.vx + self.vy*self.vy)
        if speed > max_speed:
            self.vx = (self.vx / speed) * max_speed
            self.vy = (self.vy / speed) * max_speed
        
        # Ensure flies always move at least slightly rightward
        if self.vx < 0.3:
            self.vx = 0.3
        
        # --- Update position ---
        self.x += self.vx
        self.y += self.vy
        
        # Clamp to arena boundaries (with padding)
        self.y = max(5, min(ARENA_HEIGHT - 5, self.y))
        self.x = max(0, self.x)
        
        # Check if fly has reached the escape zone
        if self.x >= ESCAPE_ZONE_X and not self.escaped:
            self.escaped = True

# ============================================================
# FROG
# ============================================================

class Frog:
    """
    Represents a single frog controlled by a genome.
    
    Frogs are stationary and use their tongues to catch flies that come within
    range. They select targets based on their genome's preferences (nearest vs.
    most-advanced, cooperation with other frogs, ambush patience) and predict
    fly movement to aim their tongue attacks.
    
    Uses __slots__ for memory efficiency.
    """
    __slots__ = ['x', 'y', 'start_x', 'start_y', 'genome', 'frog_id',
                 'tongue_length', 'vision_range', 'cooldown_time',
                 'tongue_out', 'tongue_x', 'tongue_y', 
                 'tongue_target_x', 'tongue_target_y', 'tongue_extending',
                 'cooldown', 'current_target', 'catches', 'attempts',
                 'catch_positions', 'catch_times']
    
    def __init__(self, genome: Genome, frog_id: int = 0):
        """
        Initialize a frog from a genome.
        
        Args:
            genome: The genetic blueprint defining this frog's traits.
            frog_id: Unique identifier for this frog within its team.
        """
        # Position derived from genome
        self.x, self.y = genome.get_actual_position()
        self.start_x = self.x  # Remember starting position
        self.start_y = self.y
        
        self.genome = genome
        self.frog_id = frog_id
        
        # Derived stats from genome
        self.tongue_length = genome.get_tongue_length()  # Max tongue reach
        self.vision_range = genome.vision_range            # Detection radius
        self.cooldown_time = int(genome.rest_cycles)       # Steps between attacks
        
        # Tongue state machine
        self.tongue_out = False           # Whether tongue is currently active
        self.tongue_x = self.x            # Current tongue tip x-position
        self.tongue_y = self.y            # Current tongue tip y-position
        self.tongue_target_x = self.x     # Where the tongue is aiming
        self.tongue_target_y = self.y     # Where the tongue is aiming
        self.tongue_extending = True      # True = extending, False = retracting
        self.cooldown = 0                 # Remaining cooldown frames
        
        # Currently targeted fly (None if no target)
        self.current_target: Optional[Fly] = None
        
        # Performance tracking statistics
        self.catches = 0           # Total flies caught
        self.attempts = 0          # Total tongue attacks launched
        self.catch_positions = []  # X-positions where catches occurred
        self.catch_times = []      # Step numbers when catches occurred

    def update(self, flies: List[Fly], other_frogs: List['Frog'], 
               current_step: int, spatial_grid: SpatialGrid):
        """
        Update frog behavior for one simulation step.
        
        State machine with three states:
        1. Tongue out -> update tongue (extend/retract, check for catches)
        2. On cooldown -> decrement cooldown timer
        3. Ready -> scan for flies, select target, decide whether to attack
        
        Args:
            flies: All flies in the simulation (for tongue collision checking).
            other_frogs: Other frogs on the team (for cooperation logic).
            current_step: Current simulation step number.
            spatial_grid: Spatial hash grid for efficient fly lookups.
        """
        # If tongue is currently out, handle its movement
        if self.tongue_out:
            self._update_tongue(flies, current_step)
            return
        
        # If on cooldown, just wait
        if self.cooldown > 0:
            self.cooldown -= 1
            return
        
        # --- Target acquisition phase ---
        
        # Use spatial grid for efficient nearby fly lookup
        nearby = spatial_grid.get_nearby(self.x, self.y, self.vision_range)
        visible_flies = []
        
        # Filter to flies actually within vision range
        for fly in nearby:
            if not fly.alive or fly.escaped:
                continue
            dx = fly.x - self.x
            dy = fly.y - self.y
            dist_sq = dx*dx + dy*dy
            if dist_sq <= self.vision_range * self.vision_range:
                visible_flies.append((fly, np.sqrt(dist_sq)))
        
        # No visible flies -> nothing to do
        if not visible_flies:
            self.current_target = None
            return
        
        # --- Cooperation: avoid targeting flies other frogs are already after ---
        other_targets = set(f.current_target for f in other_frogs 
                          if f is not self and f.current_target)
        
        if self.genome.cooperation > 0.3 and other_targets:
            # With probability proportional to cooperation gene, filter out
            # flies that other frogs are targeting
            if random.random() < self.genome.cooperation:
                filtered = [(f, d) for f, d in visible_flies if f not in other_targets]
                if filtered:
                    visible_flies = filtered
        
        # --- Target scoring and selection ---
        def score_fly(fly_dist):
            """
            Score a fly for targeting priority (lower = higher priority).
            
            Blends distance-based targeting (catch nearby flies) with
            progress-based targeting (catch flies about to escape).
            The blend is controlled by the target_preference gene.
            """
            fly, dist = fly_dist
            dist_score = dist / self.vision_range          # 0=close, 1=far
            progress_score = fly.x / ESCAPE_ZONE_X         # 0=just spawned, 1=escaping
            pref = self.genome.target_preference
            # Blend: low pref = prioritize close flies; high pref = prioritize advanced flies
            score = (1 - pref) * dist_score + pref * (1 - progress_score)
            # Emergency: heavily prioritize flies about to escape
            if fly.x > ESCAPE_ZONE_X - 50:
                score -= 0.5
            return score
        
        # Sort flies by score (lowest = best target)
        visible_flies.sort(key=score_fly)
        self.current_target = visible_flies[0][0]
        target = self.current_target
        
        # --- Attack decision ---
        dx = target.x - self.x
        dy = target.y - self.y
        dist = np.sqrt(dx*dx + dy*dy)
        
        # Calculate optimal attack distance based on ambush_mode gene
        # High ambush_mode = wait for flies to get very close before striking
        optimal_dist = self.tongue_length * (0.5 + 0.4 * (1 - self.genome.ambush_mode))
        
        should_attack = dist <= optimal_dist
        
        # Override: always attack flies about to escape if in tongue range
        if target.x > ESCAPE_ZONE_X - 40:
            should_attack = dist <= self.tongue_length * 0.95
        
        if should_attack:
            # --- Predictive aiming ---
            # Estimate where the fly will be when the tongue reaches it
            travel_time = dist / TONGUE_SPEED
            pred_x = target.x + target.vx * travel_time * 0.7  # 70% prediction (conservative)
            pred_y = target.y + target.vy * travel_time * 0.7
            
            # Launch tongue attack
            self.tongue_out = True
            self.tongue_extending = True
            self.tongue_x = self.x
            self.tongue_y = self.y
            self.tongue_target_x = pred_x
            self.tongue_target_y = pred_y
            self.attempts += 1
    
    def _update_tongue(self, flies: List[Fly], current_step: int):
        """
        Update tongue position during an attack.
        
        Two phases:
        1. Extending: tongue tip moves toward target at TONGUE_SPEED.
           Checks for fly collisions (catch radius of 10 units).
        2. Retracting: tongue tip returns to frog's mouth at TONGUE_SPEED.
        
        Args:
            flies: All flies to check for tongue-tip collisions.
            current_step: Current step for recording catch times.
        """
        if self.tongue_extending:
            # --- Extension phase ---
            dx = self.tongue_target_x - self.x
            dy = self.tongue_target_y - self.y
            dist = np.sqrt(dx*dx + dy*dy)
            
            # Move tongue tip toward target
            if dist > 0:
                self.tongue_x += (dx / dist) * TONGUE_SPEED
                self.tongue_y += (dy / dist) * TONGUE_SPEED
            
            # Check if tongue has reached max length or target
            tongue_dist = np.sqrt((self.tongue_x - self.x)**2 + 
                                 (self.tongue_y - self.y)**2)
            
            if tongue_dist >= self.tongue_length or tongue_dist >= dist:
                self.tongue_extending = False  # Switch to retraction
            
            # --- Collision detection with flies ---
            for fly in flies:
                if not fly.alive or fly.escaped:
                    continue
                
                # Check if fly is within catch radius (10 units) of tongue tip
                fly_dist = (fly.x - self.tongue_x)**2 + (fly.y - self.tongue_y)**2
                if fly_dist < 100:  # 100 = 10^2 (catch radius squared)
                    fly.alive = False
                    fly.caught_at_step = current_step
                    self.catches += 1
                    self.catch_positions.append(fly.x)
                    self.catch_times.append(current_step)
        else:
            # --- Retraction phase ---
            # Move tongue tip back toward frog's mouth
            dx = self.x - self.tongue_x
            dy = self.y - self.tongue_y
            dist = np.sqrt(dx*dx + dy*dy)
            
            if dist > TONGUE_SPEED:
                # Still retracting
                self.tongue_x += (dx / dist) * TONGUE_SPEED
                self.tongue_y += (dy / dist) * TONGUE_SPEED
            else:
                # Tongue fully retracted -> enter cooldown
                self.tongue_out = False
                self.tongue_x = self.x
                self.tongue_y = self.y
                self.cooldown = self.cooldown_time
                self.current_target = None

# ============================================================
# SIMULATION
# ============================================================

class Simulation:
    """
    Runs a single simulation episode with a team of frogs trying to catch flies.
    
    Manages the game loop: spawning flies, updating positions, detecting catches
    and escapes, and tracking statistics for fitness evaluation.
    """
    
    def __init__(self, genomes: List[Genome], num_frogs: int = None):
        """
        Initialize a simulation with a set of frog genomes.
        
        Args:
            genomes: List of Genome objects defining each frog's traits.
            num_frogs: Number of frogs to use (defaults to NUM_FROGS or genome count).
        """
        self.genomes = genomes
        self.num_frogs = num_frogs if num_frogs else min(NUM_FROGS, len(genomes))
        
        self.flies: List[Fly] = []
        self.frogs: List[Frog] = []
        
        # Spatial grid for efficient fly neighbor lookups
        self.spatial_grid = SpatialGrid(ARENA_WIDTH, ARENA_HEIGHT, SPATIAL_GRID_SIZE)
        
        # Cached frog data arrays (updated each step for fly avoidance)
        self.frog_positions = np.zeros((self.num_frogs, 2))
        self.frog_tongue_states = [False] * self.num_frogs
        
        # Simulation state
        self.step_count = 0
        self.escaped_count = 0
        self.running = True
        
        # Timing statistics
        self.first_catch_step = -1    # Step of the first catch
        self.last_catch_step = -1     # Step of the most recent catch
        self.all_caught_step = -1     # Step when all flies were caught (-1 if not achieved)
        self.clear_step = -1          # Step when simulation ended
        self.perfect_clear = False    # True if all flies were caught (none escaped)
    
    def reset(self):
        """Reset the simulation to initial state with fresh flies and frogs."""
        # Spawn all flies in the spawn zone
        self.flies = [Fly() for _ in range(NUM_FLIES)]
        
        # Create frogs from genomes
        self.frogs = []
        for i in range(self.num_frogs):
            genome = self.genomes[i % len(self.genomes)]  # Cycle if fewer genomes than frogs
            frog = Frog(genome, frog_id=i)
            self.frogs.append(frog)
            self.frog_positions[i] = [frog.x, frog.y]
        
        # Reset counters
        self.step_count = 0
        self.escaped_count = 0
        self.running = True
        
        self.first_catch_step = -1
        self.last_catch_step = -1
        self.all_caught_step = -1
        self.clear_step = -1
        self.perfect_clear = False
    
    def step(self):
        """
        Advance the simulation by one time step.
        
        Order of operations:
        1. Update cached frog position/tongue data for fly AI
        2. Rebuild spatial grid with active flies
        3. Update all flies (movement, avoidance, escape detection)
        4. Update all frogs (target selection, tongue attacks)
        5. Track catch/escape statistics
        6. Check termination conditions
        """
        if not self.running:
            return
        
        self.step_count += 1
        
        # Cache frog positions and tongue states for fly AI
        for i, frog in enumerate(self.frogs):
            self.frog_positions[i] = [frog.x, frog.y]
            self.frog_tongue_states[i] = frog.tongue_out
        
        # Rebuild spatial grid with currently active flies
        self.spatial_grid.clear()
        for fly in self.flies:
            if fly.alive and not fly.escaped:
                self.spatial_grid.insert(fly, fly.x, fly.y)
        
        # Snapshot total catches before updates (to detect new catches)
        catches_before = sum(f.catches for f in self.frogs)
        
        # --- Update all flies ---
        for fly in self.flies:
            if not fly.alive or fly.escaped:
                continue
            
            # Get nearby flies for flocking behavior
            nearby = self.spatial_grid.get_nearby(fly.x, fly.y, 30)
            
            was_escaped = fly.escaped
            fly.update_vectorized(nearby, self.frog_positions, self.frog_tongue_states)
            
            # Track newly escaped flies (prevent double-counting)
            if fly.escaped and not was_escaped and not fly.marked_escaped:
                self.escaped_count += 1
                fly.marked_escaped = True
        
        # --- Update all frogs ---
        for frog in self.frogs:
            frog.update(self.flies, self.frogs, self.step_count, self.spatial_grid)
        
        # --- Track catch timing statistics ---
        catches_after = sum(f.catches for f in self.frogs)
        
        if catches_after > catches_before:
            if self.first_catch_step == -1:
                self.first_catch_step = self.step_count  # Record first catch
            self.last_catch_step = self.step_count       # Update most recent catch
        
        # --- Check termination conditions ---
        active_flies = sum(1 for f in self.flies if f.alive and not f.escaped)
        caught_flies = sum(1 for f in self.flies if not f.alive)
        
        # Check for perfect clear (all flies caught)
        if caught_flies == NUM_FLIES and self.all_caught_step == -1:
            self.all_caught_step = self.step_count
            self.perfect_clear = True
        
        # End simulation if no active flies remain or time limit reached
        if active_flies == 0 or self.step_count >= MAX_STEPS:
            self.running = False
            self.clear_step = self.step_count
            if caught_flies == NUM_FLIES:
                self.perfect_clear = True
    
    def get_total_catches(self) -> int:
        """Return the total number of flies caught by all frogs combined."""
        return sum(f.catches for f in self.frogs)
    
    def get_time_stats(self) -> dict:
        """
        Compute detailed timing and performance statistics for this simulation.
        
        Returns:
            Dictionary containing:
            - total_catches: Number of flies caught
            - escaped: Number of flies that escaped
            - perfect_clear: Whether all flies were caught
            - first_catch_step: When the first catch occurred
            - last_catch_step: When the last catch occurred
            - all_caught_step: When all flies were caught (-1 if not)
            - completion_time: Total steps used
            - catch_speed: Catches per 100 steps
            - avg_catch_interval: Average steps between consecutive catches
            - steps_used: Total simulation steps
        """
        total_catches = self.get_total_catches()
        
        # Calculate catch rate (catches per 100 simulation steps)
        if self.last_catch_step > 0:
            catch_speed = (total_catches / self.last_catch_step) * 100
        else:
            catch_speed = 0
        
        # Time to first catch (penalty if never caught anything)
        time_to_first = self.first_catch_step if self.first_catch_step > 0 else MAX_STEPS
        
        # Completion time depends on whether it was a perfect clear
        if self.perfect_clear:
            completion_time = self.all_caught_step
        else:
            completion_time = self.clear_step if self.clear_step > 0 else MAX_STEPS
        
        # Calculate average interval between consecutive catches
        all_catch_times = []
        for frog in self.frogs:
            all_catch_times.extend(frog.catch_times)
        all_catch_times.sort()
        
        if len(all_catch_times) > 1:
            intervals = [all_catch_times[i+1] - all_catch_times[i] 
                        for i in range(len(all_catch_times)-1)]
            avg_interval = np.mean(intervals)
        else:
            avg_interval = MAX_STEPS  # Penalty for only catching 0-1 flies
        
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
        """
        Calculate the fitness score for this frog team.
        
        Fitness components (higher = better):
        1. Base catch reward: +100 per catch
        2. Escape penalty: -25 per escaped fly
        3. Perfect clear bonus: +200 for catching all flies
        4. Speed bonus: Reward for finishing quickly (perfect clear only)
        5. Catch rate bonus: Higher catches-per-100-steps = more fitness
        6. First catch bonus: Reward for catching the first fly quickly
        7. Consistency bonus: Reward for steady catching (low interval between catches)
        8. Team participation: Bonus when multiple frogs contribute catches
        9. Spacing bonus: Reward frogs for spreading out across the arena
        
        Returns:
            Non-negative float fitness score.
        """
        catches = self.get_total_catches()
        time_stats = self.get_time_stats()
        
        # --- Component 1: Base catch reward ---
        fitness = catches * 100
        
        # --- Component 2: Escape penalty ---
        fitness -= self.escaped_count * 25
        
        # --- Component 3 & 4: Perfect clear and speed bonuses ---
        if time_stats['perfect_clear']:
            fitness += PERFECT_CLEAR_BONUS
            # Speed bonus: more reward for faster perfect clears
            speed_factor = max(0, (MAX_STEPS - time_stats['completion_time']) / MAX_STEPS)
            fitness += speed_factor * 150 * SPEED_BONUS_MULTIPLIER
        
        # --- Component 5: Catch rate bonus ---
        catch_speed = time_stats['catch_speed']
        fitness += catch_speed * 10 * SPEED_BONUS_MULTIPLIER
        
        # --- Component 6: First catch bonus (reward early aggression) ---
        if time_stats['first_catch_step'] > 0:
            first_catch_bonus = max(0, (100 - time_stats['first_catch_step']) / 100) * 30
            fitness += first_catch_bonus * SPEED_BONUS_MULTIPLIER
        
        # --- Component 7: Consistency bonus (steady catching) ---
        if time_stats['avg_catch_interval'] < MAX_STEPS:
            consistency_bonus = max(0, (50 - time_stats['avg_catch_interval']) / 50) * 20
            fitness += consistency_bonus * SPEED_BONUS_MULTIPLIER
        
        # --- Component 8: Team participation bonus ---
        active_catchers = sum(1 for f in self.frogs if f.catches > 0)
        if active_catchers > 1:
            fitness += active_catchers * 20
        
        # --- Component 9: Frog spacing bonus ---
        # Reward frogs for spreading out (better coverage of the arena)
        if self.num_frogs > 1:
            positions = [(f.x, f.y) for f in self.frogs]
            min_dist = float('inf')
            for i in range(len(positions)):
                for j in range(i + 1, len(positions)):
                    dx = positions[i][0] - positions[j][0]
                    dy = positions[i][1] - positions[j][1]
                    dist = np.sqrt(dx*dx + dy*dy)
                    min_dist = min(min_dist, dist)
            # Bonus if minimum inter-frog distance exceeds 30 units
            if min_dist > 30:
                fitness += min(30, min_dist - 30)
        
        return max(0, fitness)  # Fitness is always non-negative

# ============================================================
# GENETIC ALGORITHM
# ============================================================

class GeneticAlgorithm:
    """
    Evolves teams of frog genomes to maximize fly-catching performance.
    
    Uses tournament selection, BLX-α crossover, Gaussian mutation, and
    elitism to evolve a population of frog teams over multiple generations.
    Each "individual" in the population is a team of NUM_FROGS genomes.
    """
    
    def __init__(self, population_size: int = POPULATION_SIZE, 
                 num_frogs: int = NUM_FROGS):
        """
        Initialize the genetic algorithm.
        
        Args:
            population_size: Number of teams in each generation.
            num_frogs: Number of frogs per team.
        """
        self.population_size = population_size
        self.num_frogs = num_frogs
        
        # Current population: list of teams, where each team is a list of Genomes
        self.population: List[List[Genome]] = []
        self.fitness_scores: List[float] = []
        
        # Best-ever tracking
        self.generation = 0
        self.best_fitness = 0
        self.best_team: List[Genome] = []
        
        # Historical data for plotting and analysis
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
        """
        Create the initial random population.
        
        Frogs within each team are given vertically-spaced starting positions
        to encourage coverage of the arena (with small random perturbation).
        """
        self.population = []
        for _ in range(self.population_size):
            team = []
            for i in range(self.num_frogs):
                genome = Genome.random()
                # Distribute frogs evenly across the vertical axis
                genome.position_y = (i + 0.5) / self.num_frogs
                # Add slight random offset to avoid perfectly uniform placement
                genome.position_y = max(0.1, min(0.9, 
                    genome.position_y + random.uniform(-0.1, 0.1)))
                team.append(genome)
            self.population.append(team)
        self.generation = 0
    
    def evaluate_team(self, team: List[Genome], num_trials: int = 2) -> Tuple[float, dict]:
        """
        Evaluate a team's fitness by running multiple simulation trials.
        
        Multiple trials reduce variance from random fly spawning and movement.
        
        Args:
            team: List of Genome objects defining the frog team.
            num_trials: Number of independent simulations to average over.
            
        Returns:
            Tuple of (average_fitness, stats_dictionary).
        """
        total_fitness = 0
        total_catches = 0
        total_escaped = 0
        total_completion_time = 0
        total_catch_speed = 0
        perfect_count = 0
        
        for _ in range(num_trials):
            sim = Simulation(team, num_frogs=self.num_frogs)
            sim.reset()
            
            # Run simulation to completion
            while sim.running:
                sim.step()
            
            # Accumulate results
            total_fitness += sim.get_team_fitness()
            stats = sim.get_time_stats()
            total_catches += stats['total_catches']
            total_escaped += stats['escaped']
            total_completion_time += stats['completion_time']
            total_catch_speed += stats['catch_speed']
            if stats['perfect_clear']:
                perfect_count += 1
        
        # Return averages across trials
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
        """
        Evaluate all teams in the current population and record statistics.
        
        Also updates the all-time best team if a new record is achieved.
        """
        self.fitness_scores = []
        total_catches = 0
        total_escaped = 0
        total_completion_time = 0
        total_catch_speed = 0
        total_perfect = 0
        
        # Evaluate each team
        for team in self.population:
            fitness, stats = self.evaluate_team(team)
            self.fitness_scores.append(fitness)
            total_catches += stats['catches']
            total_escaped += stats['escaped']
            total_completion_time += stats['completion_time']
            total_catch_speed += stats['catch_speed']
            total_perfect += stats['perfect_clears']
        
        # Update all-time best team
        best_idx = np.argmax(self.fitness_scores)
        if self.fitness_scores[best_idx] > self.best_fitness:
            self.best_fitness = self.fitness_scores[best_idx]
            # Deep copy the best team's genomes
            self.best_team = [Genome(**g.to_dict()) for g in self.population[best_idx]]
        
        # Record generation statistics for history tracking
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
        """
        Select a parent team using tournament selection (size 3).
        
        Randomly picks 3 teams and returns the one with the highest fitness.
        This balances selection pressure with diversity preservation.
        
        Returns:
            The winning team (list of Genomes).
        """
        tournament = random.sample(range(self.population_size), 3)
        winner = max(tournament, key=lambda i: self.fitness_scores[i])
        return self.population[winner]
    
    def evolve(self):
        """
        Create the next generation through selection, crossover, and mutation.
        
        Process:
        1. Elitism: Top ELITE_COUNT teams pass unchanged to next generation.
        2. For remaining slots: select two parents via tournament selection,
           apply BLX-α crossover (or clone) per frog, then mutate.
        3. Replace old population with new population.
        4. Increment generation counter.
        """
        # Sort teams by fitness (descending)
        sorted_indices = np.argsort(self.fitness_scores)[::-1]
        new_population = []
        
        # --- Elitism: preserve top teams unchanged ---
        for i in range(ELITE_COUNT):
            elite = [Genome(**g.to_dict()) for g in self.population[sorted_indices[i]]]
            new_population.append(elite)
        
        # --- Fill remaining population with offspring ---
        while len(new_population) < self.population_size:
            # Select two parent teams
            p1 = self.select_parent()
            p2 = self.select_parent()
            
            # Create child team by crossing over/mutating each frog independently
            child = []
            for i in range(self.num_frogs):
                if random.random() < CROSSOVER_RATE:
                    # BLX-α crossover between corresponding frogs from each parent team
                    c = Genome.blend_crossover(p1[i], p2[i])
                else:
                    # No crossover: clone one parent's frog
                    c = Genome(**random.choice([p1[i], p2[i]]).to_dict())
                # Apply mutation to the child genome
                child.append(c.mutate())
            
            new_population.append(child)
        
        # Replace old population and advance generation
        self.population = new_population
        self.generation += 1
    
    def get_best_team(self) -> List[Genome]:
        """
        Return the best team found so far.
        
        Returns the all-time best if available, otherwise the best
        in the current generation.
        """
        if self.best_team:
            return self.best_team
        best_idx = np.argmax(self.fitness_scores)
        return self.population[best_idx]

# ============================================================
# VISUALIZATION
# ============================================================

class Game:
    """
    Pygame-based visualization for the fly catcher simulation.
    
    Renders the arena with flies, frogs, tongues, zones, and an information
    panel showing real-time statistics. Supports pausing, speed control,
    and generation skipping via keyboard controls.
    """
    
    def __init__(self, num_frogs: int = NUM_FROGS):
        """
        Initialize the Pygame display and rendering resources.
        
        Args:
            num_frogs: Number of frogs (shown in window title).
        """
        pygame.init()
        
        self.num_frogs = num_frogs
        self.width = ARENA_WIDTH * SCALE          # Window width in pixels
        self.height = ARENA_HEIGHT * SCALE + 120  # Window height (arena + info panel)
        
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption(f"Fly Catcher - {NUM_FLIES} Flies, {num_frogs} Frogs")
        
        self.clock = pygame.time.Clock()
        
        # Font sizes for different UI elements
        self.font = pygame.font.Font(None, 26)        # Main stats
        self.small_font = pygame.font.Font(None, 20)   # Secondary info
        self.tiny_font = pygame.font.Font(None, 16)    # Zone labels
        
        # Pre-render a fly sprite surface (body + wings)
        self.fly_surface = pygame.Surface((10, 10), pygame.SRCALPHA)
        pygame.draw.circle(self.fly_surface, BLACK, (5, 5), 4)       # Body
        pygame.draw.ellipse(self.fly_surface, GRAY, (0, 3, 4, 3))    # Left wing
        pygame.draw.ellipse(self.fly_surface, GRAY, (6, 3, 4, 3))    # Right wing
        
        # Application state
        self.running = True
        self.paused = False
    
    def scale(self, x: float, y: float) -> Tuple[int, int]:
        """Convert simulation coordinates to screen pixel coordinates."""
        return int(x * SCALE), int(y * SCALE)
    
    def handle_events(self) -> str:
        """
        Process Pygame events and return an action string.
        
        Returns:
            'quit': User closed window or pressed ESC
            'pause': Space was pressed (toggle pause)
            'skip': Right arrow pressed (skip to next generation)
            'speed_up': Up arrow pressed (increase simulation speed)
            'speed_down': Down arrow pressed (decrease simulation speed)
            'ok': No special action
        """
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
        """
        Render one frame of the simulation.
        
        Draws (in order):
        1. Background and zone overlays (spawn zone, frog zone, escape zone)
        2. Ground strip at bottom
        3. Zone labels
        4. Active flies (orange if panicking, black otherwise)
        5. Frogs with vision rings, tongues, eyes, and catch counters
        6. Information panel with stats, controls, and progress bar
        7. Pause overlay if paused
        
        Args:
            sim: Current simulation state to render.
            generation: Current generation number (for display).
            best_fitness: All-time best fitness (for display).
            speed: Current simulation speed multiplier (for display).
        """
        # --- Background ---
        self.screen.fill(SKY_BLUE)
        
        # Frog placement zone (light blue tint)
        zone_x1 = int(FROG_ZONE_X_MIN * ARENA_WIDTH * SCALE)
        zone_x2 = int(FROG_ZONE_X_MAX * ARENA_WIDTH * SCALE)
        pygame.draw.rect(self.screen, (220, 240, 255), 
                        (zone_x1, 0, zone_x2 - zone_x1, ARENA_HEIGHT * SCALE))
        
        # Spawn zone (green tint, left side)
        pygame.draw.rect(self.screen, SPAWN_ZONE_COLOR, 
                        (0, 0, SPAWN_ZONE_WIDTH * SCALE, ARENA_HEIGHT * SCALE))
        
        # Escape zone (red tint, right side)
        escape_x = int(ESCAPE_ZONE_X * SCALE)
        pygame.draw.rect(self.screen, RED_ZONE, 
                        (escape_x, 0, self.width - escape_x, ARENA_HEIGHT * SCALE))
        
        # Ground strip at bottom of arena
        pygame.draw.rect(self.screen, DARK_GREEN, 
                        (0, ARENA_HEIGHT * SCALE - 10, self.width, 10))
        
        # Zone labels
        self.screen.blit(self.tiny_font.render("SPAWN", True, DARK_GREEN), (5, 5))
        self.screen.blit(self.tiny_font.render("ESCAPE", True, (150, 50, 50)), (escape_x + 5, 5))
        
        # --- Draw flies ---
        for fly in sim.flies:
            if fly.alive and not fly.escaped:
                pos = self.scale(fly.x, fly.y)
                # Panicking flies are drawn in orange for visual feedback
                color = ORANGE if fly.panic_timer > 0 else BLACK
                pygame.draw.circle(self.screen, color, pos, 3)
        
        # --- Draw frogs ---
        for i, frog in enumerate(sim.frogs):
            color = FROG_COLORS[i % len(FROG_COLORS)]
            pos = self.scale(frog.x, frog.y)
            
            # Vision range indicator (semi-transparent circle outline)
            pygame.draw.circle(self.screen, (*color, 30), pos, 
                             int(frog.vision_range * SCALE), 1)
            
            # Tongue (pink line with circle at tip) - drawn behind frog body
            if frog.tongue_out:
                tongue_pos = self.scale(frog.tongue_x, frog.tongue_y)
                pygame.draw.line(self.screen, PINK, pos, tongue_pos, 4)
                pygame.draw.circle(self.screen, PINK, tongue_pos, 6)
            
            # Frog body (filled circle with dark outline)
            body_radius = int(FROG_SIZE * SCALE / 2) + 3
            pygame.draw.circle(self.screen, color, pos, body_radius)
            pygame.draw.circle(self.screen, DARK_GREEN, pos, body_radius, 2)
            
            # Frog eyes (two white circles with black pupils)
            eye_offset = body_radius // 2
            for dx in [-eye_offset, eye_offset]:
                eye_pos = (pos[0] + dx, pos[1] - eye_offset)
                pygame.draw.circle(self.screen, WHITE, eye_pos, 4)
                pygame.draw.circle(self.screen, BLACK, eye_pos, 2)
            
            # Catch counter badge (colored circle with number above frog)
            catch_text = self.small_font.render(str(frog.catches), True, WHITE)
            pygame.draw.circle(self.screen, color, 
                             (pos[0], pos[1] - body_radius - 12), 10)
            self.screen.blit(catch_text, (pos[0] - 5, pos[1] - body_radius - 18))
        
        # ========== INFORMATION PANEL (below arena) ==========
        panel_y = ARENA_HEIGHT * SCALE + 5
        total_catches = sim.get_total_catches()
        
        # --- Row 1: Generation, catches, escaped, step, speed ---
        self.screen.blit(self.font.render(f"Gen: {generation}", True, BLACK), (10, panel_y))
        self.screen.blit(self.font.render(f"Catches: {total_catches}/{NUM_FLIES}", 
                                         True, DARK_GREEN), (100, panel_y))
        self.screen.blit(self.font.render(f"Escaped: {sim.escaped_count}", 
                                         True, (180, 50, 50)), (280, panel_y))
        self.screen.blit(self.font.render(f"Step: {sim.step_count}/{MAX_STEPS}", 
                                         True, GRAY), (420, panel_y))
        self.screen.blit(self.font.render(f"Speed: {speed}x", True, GRAY), (580, panel_y))
        
        # --- Row 2: Catch rate, first catch time, perfect clear, active flies ---
        panel_y2 = panel_y + 24
        time_stats = sim.get_time_stats()
        
        # Catch rate (gold color if high performance)
        speed_color = GOLD if time_stats['catch_speed'] > 5 else BLACK
        self.screen.blit(self.font.render(f"Rate: {time_stats['catch_speed']:.1f}/100steps", 
                                         True, speed_color), (10, panel_y2))
        
        # First catch timing
        if time_stats['first_catch_step'] > 0:
            self.screen.blit(self.small_font.render(
                f"1st: step {time_stats['first_catch_step']}", True, BLUE), (200, panel_y2 + 3))
        
        # Perfect clear indicator
        if sim.perfect_clear or total_catches == NUM_FLIES:
            self.screen.blit(self.font.render("⭐ PERFECT!", True, GOLD), (350, panel_y2))
        
        # Active and panicking fly counts
        active = sum(1 for f in sim.flies if f.alive and not f.escaped)
        panicking = sum(1 for f in sim.flies if f.alive and not f.escaped and f.panic_timer > 0)
        self.screen.blit(self.small_font.render(
            f"Active: {active} (Panicking: {panicking})", True, GRAY), (480, panel_y2 + 3))
        
        # --- Row 3: Fitness scores, per-frog stats ---
        panel_y3 = panel_y2 + 24
        fitness = sim.get_team_fitness()
        self.screen.blit(self.font.render(f"Fitness: {fitness:.0f}", True, BLUE), (10, panel_y3))
        self.screen.blit(self.font.render(f"Best: {best_fitness:.0f}", 
                                         True, (50, 150, 50)), (150, panel_y3))
        
        # Per-frog catch/attempt breakdown
        frog_stats = " | ".join([f"F{i+1}:{f.catches}/{f.attempts}" 
                                for i, f in enumerate(sim.frogs)])
        self.screen.blit(self.small_font.render(frog_stats, True, DARK_GREEN), (320, panel_y3 + 3))
        
        # --- Row 4: Controls help text and progress bar ---
        panel_y4 = panel_y3 + 24
        self.screen.blit(self.small_font.render(
            "SPACE=Pause  →=Skip  ↑↓=Speed  ESC=Quit", True, GRAY), (10, panel_y4))
        
        # Progress bar (shows simulation time elapsed)
        bar_w, bar_h = 200, 10
        bar_x = self.width - bar_w - 20
        pygame.draw.rect(self.screen, GRAY, (bar_x, panel_y4, bar_w, bar_h))
        progress = sim.step_count / MAX_STEPS
        prog_color = DARK_GREEN if total_catches == NUM_FLIES else GREEN
        pygame.draw.rect(self.screen, prog_color, (bar_x, panel_y4, int(bar_w * progress), bar_h))
        pygame.draw.rect(self.screen, BLACK, (bar_x, panel_y4, bar_w, bar_h), 1)
        
        # --- Pause overlay ---
        if self.paused:
            pause_surf = self.font.render("⏸ PAUSED", True, (200, 50, 50))
            rect = pause_surf.get_rect(center=(self.width // 2, ARENA_HEIGHT * SCALE // 2))
            pygame.draw.rect(self.screen, WHITE, rect.inflate(20, 10))
            pygame.draw.rect(self.screen, (200, 50, 50), rect.inflate(20, 10), 2)
            self.screen.blit(pause_surf, rect)
        
        # Push everything to the display
        pygame.display.flip()
    
    def close(self):
        """Shut down Pygame and release display resources."""
        pygame.quit()

# ============================================================
# MAIN
# ============================================================

def main():
    """
    Main entry point for the Fly Catcher genetic algorithm training program.
    
    Workflow:
    1. Print configuration summary
    2. Initialize Pygame visualization and genetic algorithm
    3. For each generation:
       a. Evaluate all teams (headless simulations)
       b. Print generation statistics
       c. Visualize the best team's simulation
       d. Handle user input (pause, speed, skip, quit)
       e. Evolve the population for the next generation
    4. Print final results
    5. Save training history to CSV
    6. Generate and save matplotlib plots (if available)
    """
    # --- Print startup banner and configuration ---
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
    
    # --- Initialize systems ---
    game = Game(num_frogs=NUM_FROGS)
    ga = GeneticAlgorithm(population_size=POPULATION_SIZE, num_frogs=NUM_FROGS)
    ga.initialize()
    
    speed = 2  # Initial simulation speed multiplier
    
    # ===== MAIN GENERATION LOOP =====
    for gen in range(NUM_GENERATIONS):
        if not game.running:
            break
        
        # Print generation header
        print(f"\n{'─'*60}")
        print(f"  GENERATION {gen + 1}/{NUM_GENERATIONS}")
        print(f"{'─'*60}")
        
        # Evaluate all teams in the population (headless)
        ga.evaluate_population()
        
        # Extract generation statistics
        best_fitness = max(ga.fitness_scores)
        avg_fitness = np.mean(ga.fitness_scores)
        avg_time = ga.history['avg_completion_time'][-1]
        perfect_count = ga.history['perfect_clears'][-1]
        catch_speed = ga.history['avg_catch_speed'][-1]

        # Print generation stats
        print(f"  📈 Fitness   - Best: {best_fitness:.1f}  Avg: {avg_fitness:.1f}")
        print(f"  ⏱️  Speed     - Completion: {avg_time:.0f}steps  Rate: {catch_speed:.2f}/100")
        print(f"  ⭐ Perfects  - {perfect_count} teams")
        
        # Print best team's genome details
        best_idx = np.argmax(ga.fitness_scores)
        best_team = ga.population[best_idx]
        
        for i, genome in enumerate(best_team):
            print(f"  🐸 F{i+1}: {genome}")
        
        # --- Visualize the best team's simulation ---
        sim = Simulation(best_team, num_frogs=NUM_FROGS)
        sim.reset()
        
        # Run visualization loop until simulation ends or user skips/quits
        while game.running and sim.running:
            event = game.handle_events()
            
            if event == 'quit':
                break
            elif event == 'skip':
                break  # Skip to next generation
            elif event == 'speed_up':
                speed = min(15, speed + 1)
            elif event == 'speed_down':
                speed = max(1, speed - 1)
            
            # Advance simulation (multiple steps per frame for speed > 1)
            if not game.paused:
                for _ in range(speed):
                    sim.step()
                    if not sim.running:
                        break
            
            # Render the current state
            game.draw(sim, gen + 1, ga.best_fitness, speed)
            game.clock.tick(60)  # Cap at 60 FPS
        
        if not game.running:
            break
        
        # Print end-of-generation summary
        final_stats = sim.get_time_stats()
        if final_stats['perfect_clear']:
            print(f"  ✅ Perfect clear in {final_stats['completion_time']} steps!")
        else:
            print(f"  📊 {final_stats['total_catches']}/{NUM_FLIES} caught, "
                  f"{final_stats['escaped']} escaped")
        
        # Brief pause between generations for readability
        pygame.time.wait(150)
        
        # Evolve population for next generation
        ga.evolve()
    
    # --- Cleanup visualization ---
    game.close()
    
    # ===== PRINT FINAL RESULTS =====
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
    
    # ===== SAVE TRAINING DATA TO CSV =====
    os.makedirs('results', exist_ok=True)
    
    with open('results/training_data.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        # Write header row
        writer.writerow(['generation', 'best_fitness', 'avg_fitness', 
                        'avg_catches', 'avg_escaped', 'avg_completion_time',
                        'perfect_clears', 'avg_catch_speed'])
        # Write one row per generation
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
    
    # ===== GENERATE MATPLOTLIB PLOTS (if matplotlib is available) =====
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        gens = ga.history['generation']
        
        # Plot 1: Fitness evolution (best and average over generations)
        axes[0,0].plot(gens, ga.history['best_fitness'], 'g-', lw=2, label='Best')
        axes[0,0].plot(gens, ga.history['avg_fitness'], 'b--', lw=2, label='Avg')
        axes[0,0].set_title('Fitness Evolution')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        
        # Plot 2: Catches vs. escapes over generations
        axes[0,1].plot(gens, ga.history['total_catches'], 'g-', lw=2, label='Catches')
        axes[0,1].plot(gens, ga.history['total_escaped'], 'r-', lw=2, label='Escaped')
        axes[0,1].axhline(NUM_FLIES, color='gray', ls=':', alpha=0.5)  # Total flies reference
        axes[0,1].set_title('Catches vs Escaped')
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)
        
        # Plot 3: Catch speed (efficiency) over generations
        axes[1,0].plot(gens, ga.history['avg_catch_speed'], 'orange', lw=2)
        axes[1,0].fill_between(gens, 0, ga.history['avg_catch_speed'], alpha=0.3, color='orange')
        axes[1,0].set_title('Catch Speed (per 100 steps)')
        axes[1,0].grid(True, alpha=0.3)
        
        # Plot 4: Number of perfect clears per generation
        axes[1,1].bar(gens, ga.history['perfect_clears'], color='gold', edgecolor='orange')
        axes[1,1].set_title('Perfect Clears per Generation')
        axes[1,1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig('results/evolution_plot.png', dpi=150)
        print(f"  📊 Saved to results/evolution_plot.png")
        plt.show()
    except ImportError:
        # matplotlib not installed - skip plotting silently
        pass
    
    print("\n" + "=" * 70 + "\n")


if __name__ == '__main__':
    main()
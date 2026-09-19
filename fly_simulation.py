"""
Fly Arena - A 2D Multi-Agent Insect Simulation
================================================
Uses pygame to simulate FemaleFly agents performing a random walk
inside a bounded arena, and a MaleFly that senses the nearest female
through simulated LC10-style eye nodes, feeds those values straight
into left/right motor nodes, and steers/pursues accordingly.

Run with:  python fly_simulation.py
Requires:  pip install pygame
"""

import pygame
import random
import math

# ----------------------------------------------------------------------
# Global configuration constants
# ----------------------------------------------------------------------
ARENA_WIDTH = 600
ARENA_HEIGHT = 600
FPS = 60
MAX_FRAME_TIME = 0.1

BACKGROUND_COLOR = (20, 20, 25)       # near-black arena background
FEMALE_FLY_COLOR = (60, 220, 90)      # green
MALE_FLY_COLOR = (60, 130, 240)       # blue

FEMALE_FLY_RADIUS = 4
MALE_FLY_RADIUS = 5

NUM_FEMALE_FLIES = 15

# Random-walk tuning parameters
FEMALE_SPEED = 90.0             # pixels moved per second
HEADING_CHANGE_INTERVAL = 1 / 6  # re-randomize heading every N seconds
MAX_TURN_ANGLE = math.radians(45)  # max heading change per adjustment, in radians

# Visual/neural tracking tuning parameters (for MaleFly's LC10-style nodes)
MAX_SENSING_DISTANCE = 400   # beyond this distance, stimulus is treated as 0
STIMULUS_GAIN = 1.0          # scales the final 0-1 stimulus value

# Motor / steering tuning parameters (for MaleFly's pursuit behavior)
MOTOR_TURN_GAIN = math.radians(600)  # max heading change per second from a fully-fired motor
MALE_PURSUIT_SPEED = 120.0           # forward speed while actively tracking a female
MALE_DRIFT_SPEED = 24.0              # slow forward speed while hovering/drifting with no target
DRIFT_HEADING_CHANGE_INTERVAL = 1 / 3  # how often the drift heading wanders, in seconds
DRIFT_MAX_TURN_ANGLE = math.radians(20)  # max random turn per drift adjustment


class FemaleFly:
    """
    Represents a female fly that performs a continuous random walk.

    Movement logic:
    - The fly has a `heading` (an angle in radians) that determines the
      direction it's currently moving in.
    - Every `HEADING_CHANGE_INTERVAL` seconds, the fly nudges its heading
      by a small random amount (up to +/- MAX_TURN_ANGLE). This produces
      smooth, organic-looking wandering instead of jerky teleporting,
      because the fly doesn't pick a brand new random direction every
      simulation step -- it gradually steers.
    - On every simulation step (regardless of whether the heading just changed),
      the fly moves at FEMALE_SPEED pixels per second
      in the direction its current heading points.
    - If that step would carry the fly outside the arena walls, the
      fly "bounces": its position is clamped back inside the boundary
      and the relevant component of its heading is mirrored (reflected),
      similar to how light reflects off a mirror.
    """

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.radius = FEMALE_FLY_RADIUS
        self.speed = FEMALE_SPEED
        # Start facing a random direction (0 to 2*pi radians)
        self.heading = random.uniform(0, 2 * math.pi)
        # Countdown timer until the next heading adjustment
        self.time_until_turn = random.uniform(0, HEADING_CHANGE_INTERVAL)

    def update(self, dt):
        """Advance the fly's random walk by one simulation step."""
        # --- Step 1: Occasionally adjust heading (turning) ---
        self.time_until_turn -= dt
        if self.time_until_turn <= 0:
            # Pick a small random turn, left or right, within +/- MAX_TURN_ANGLE
            turn = random.uniform(-MAX_TURN_ANGLE, MAX_TURN_ANGLE)
            self.heading += turn
            # Reset the countdown for the next turn
            self.time_until_turn = HEADING_CHANGE_INTERVAL

        # --- Step 2: Move forward at constant speed along current heading ---
        dx = math.cos(self.heading) * self.speed * dt
        dy = math.sin(self.heading) * self.speed * dt
        self.x += dx
        self.y += dy

        # --- Step 3: Bounce off arena walls ---
        # If the fly's center goes past the left/right walls (accounting
        # for its radius so it visually bounces at the wall, not the edge
        # of the window), clamp its position and flip the horizontal
        # component of the heading (mirror reflection off a vertical wall).
        if self.x - FEMALE_FLY_RADIUS < 0:
            self.x = FEMALE_FLY_RADIUS
            self.heading = math.pi - self.heading
        elif self.x + FEMALE_FLY_RADIUS > ARENA_WIDTH:
            self.x = ARENA_WIDTH - FEMALE_FLY_RADIUS
            self.heading = math.pi - self.heading

        # Same idea for the top/bottom walls (mirror reflection off a
        # horizontal wall flips the sign of the vertical component).
        if self.y - FEMALE_FLY_RADIUS < 0:
            self.y = FEMALE_FLY_RADIUS
            self.heading = -self.heading
        elif self.y + FEMALE_FLY_RADIUS > ARENA_HEIGHT:
            self.y = ARENA_HEIGHT - FEMALE_FLY_RADIUS
            self.heading = -self.heading

    def draw(self, surface):
        """Draw this fly as a small filled circle."""
        pygame.draw.circle(
            surface,
            FEMALE_FLY_COLOR,
            (int(self.x), int(self.y)),
            FEMALE_FLY_RADIUS,
        )


class MaleFly:
    """
    Male fly with a simplified simulated visual-tracking network that
    is now wired directly into his own steering (motor) control.

    Pipeline, run fresh every simulation step:

        FemaleFly positions
              |
              v
        Left_Eye_Node / Right_Eye_Node   (sensory layer -- "how close/
              |                            which side is she on?")
              v
        Turn_Left_Motor / Turn_Right_Motor  (motor layer -- receives
              |                               the eye values directly)
              v
        heading adjustment + forward step   (actual movement)

    This is a deliberately simplified stand-in for the kind of
    sensory-to-motor "vector" pathway found in real fly visual
    circuits (e.g. LC10 neurons feeding steering circuitry): strong
    stimulus on one eye pulls the fly's heading toward that side.
    """

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.radius = MALE_FLY_RADIUS
        self.speed = MALE_DRIFT_SPEED
        # The male's facing/movement direction. This now actually
        # drives his motion (previously it only existed as a
        # reference angle for the stationary placeholder).
        self.heading = -math.pi / 2

        # The two simulated visual input nodes (LC10-style). Each is
        # a scalar stimulus value in the range [0, 1], where 0 means
        # "no stimulus" and 1 means "target as close/centered as
        # possible on that side."
        self.left_eye_node = 0.0
        self.right_eye_node = 0.0

        # The two downstream motor nodes. Each simulation step they simply take
        # on the value of their corresponding eye node -- the eyes
        # feed straight into the motors with no extra processing.
        self.turn_left_motor = 0.0
        self.turn_right_motor = 0.0

        # Countdown timer used only for the "no target" drifting
        # behavior (a slow, gentle random wander, same technique as
        # FemaleFly's random walk but much slower and gentler).
        self.time_until_drift_turn = random.uniform(0, DRIFT_HEADING_CHANGE_INTERVAL)

        # Cached diagnostics from the last tracking update, useful for
        # printing/debugging.
        self.nearest_distance = None
        self.relative_angle_deg = None
        self.tracking_label = "NONE"

    def find_nearest_female(self, female_flies):
        """Return the FemaleFly closest to this male, or None if empty."""
        nearest = None
        nearest_dist_sq = float("inf")
        for female in female_flies:
            dist_sq = (female.x - self.x) ** 2 + (female.y - self.y) ** 2
            if dist_sq < nearest_dist_sq:
                nearest_dist_sq = dist_sq
                nearest = female
        return nearest

    def update_visual_tracking(self, female_flies):
        """
        Core "neural vector" tracking function.

        Step 1: Find the nearest FemaleFly and compute the absolute
                distance to her.
        Step 2: Compute the angle from the male's position to her,
                then subtract the male's current heading to get the
                *relative* angle (how far off to the left/right of
                straight-ahead she is).
        Step 3: Classify that relative angle as LEFT, RIGHT, or
                STRAIGHT AHEAD.
        Step 4: Convert distance into a 0-1 "closeness" stimulus
                (closer = stronger), and feed that stimulus into
                whichever eye node (Left_Eye_Node / Right_Eye_Node)
                corresponds to her side. The node on the opposite
                side is driven to 0 for this frame.
        """
        nearest = self.find_nearest_female(female_flies)

        if nearest is None:
            # No females in the arena -- both nodes go quiet.
            self.left_eye_node = 0.0
            self.right_eye_node = 0.0
            self.nearest_distance = None
            self.relative_angle_deg = None
            self.tracking_label = "NONE"
            return

        # --- Step 1: absolute distance to the nearest female ---
        dx = nearest.x - self.x
        dy = nearest.y - self.y
        distance = math.hypot(dx, dy)

        # --- Step 2: relative angle (her bearing minus our heading) ---
        # atan2(dy, dx) gives the absolute angle (in radians) from the
        # male fly to the female, in standard screen coordinates.
        absolute_angle_to_female = math.atan2(dy, dx)
        relative_angle = absolute_angle_to_female - self.heading

        # Normalize the relative angle into the range (-pi, pi] so
        # "left" vs "right" is well-defined regardless of which way
        # the raw angle subtraction wrapped around.
        relative_angle = (relative_angle + math.pi) % (2 * math.pi) - math.pi

        # --- Step 3: classify left / right / straight ahead ---
        # In screen coordinates, a positive relative angle means the
        # target is below/clockwise from our heading, which we treat
        # as being to our RIGHT; a negative relative angle means she's
        # to our LEFT. A small dead-zone around 0 counts as "straight
        # ahead" (neither eye node fires more than the other).
        straight_ahead_deadzone = math.radians(3)

        # --- Step 4: distance -> stimulus strength (closer = stronger) ---
        # Linearly falls off from 1.0 (right on top of her) to 0.0
        # (at or beyond MAX_SENSING_DISTANCE), then scaled by GAIN.
        closeness = max(0.0, 1.0 - (distance / MAX_SENSING_DISTANCE))
        stimulus = closeness * STIMULUS_GAIN

        if abs(relative_angle) <= straight_ahead_deadzone:
            # Straight ahead: split evenly between both eyes.
            self.left_eye_node = stimulus
            self.right_eye_node = stimulus
            self.tracking_label = "STRAIGHT AHEAD"
        elif relative_angle < 0:
            # Female is to the LEFT -> feed the Left_Eye_Node.
            self.left_eye_node = stimulus
            self.right_eye_node = 0.0
            self.tracking_label = "LEFT"
        else:
            # Female is to the RIGHT -> feed the Right_Eye_Node.
            self.left_eye_node = 0.0
            self.right_eye_node = stimulus
            self.tracking_label = "RIGHT"

        self.nearest_distance = distance
        self.relative_angle_deg = math.degrees(relative_angle)

    def update_motor_control(self, dt):
        """
        Feed the eye (sensory) nodes into the motor nodes, then use
        the motor nodes to steer and move the fly.

        Step 1: The motor nodes simply mirror their corresponding eye
                nodes -- Left_Eye_Node -> Turn_Left_Motor,
                Right_Eye_Node -> Turn_Right_Motor. No extra math,
                just a direct pass-through connection.
        Step 2: motor_diff = Turn_Right_Motor - Turn_Left_Motor.
                - If the RIGHT motor is firing harder, motor_diff is
                  positive, and (in our screen-coordinate convention)
                  a positive heading change turns the fly toward the
                  female on his right.
                - If the LEFT motor is firing harder, motor_diff is
                  negative, turning him left instead.
                - If neither motor is firing (no female in range),
                  motor_diff is 0 and he doesn't turn toward anything.
                This single subtraction is the fly's entire steering
                decision every frame.
        Step 3: Scale motor_diff by MOTOR_TURN_GAIN to get the actual
                heading adjustment for this frame, and apply it.
        Step 4: Decide forward speed:
                - If either motor has a nonzero reading, a female is
                  being tracked -> move forward at MALE_PURSUIT_SPEED
                  along the (now-updated) heading, chasing her down.
                - If both motors read 0 (no female nearby), fall back
                  to a slow, gentle random drift/hover instead of
                  freezing in place.
        """
        # --- Step 1: eye nodes feed directly into the motor nodes ---
        self.turn_left_motor = self.left_eye_node
        self.turn_right_motor = self.right_eye_node

        is_tracking = (self.turn_left_motor > 0.0) or (self.turn_right_motor > 0.0)

        if is_tracking:
            # --- Step 2 & 3: steer using the motor differential ---
            motor_diff = self.turn_right_motor - self.turn_left_motor
            self.heading += motor_diff * MOTOR_TURN_GAIN * dt

            # --- Step 4a: pursue -- move forward toward the target ---
            speed = MALE_PURSUIT_SPEED
        else:
            # --- Step 4b: no target -- gentle random drift/hover ---
            self.time_until_drift_turn -= dt
            if self.time_until_drift_turn <= 0:
                drift_turn = random.uniform(-DRIFT_MAX_TURN_ANGLE, DRIFT_MAX_TURN_ANGLE)
                self.heading += drift_turn
                self.time_until_drift_turn = DRIFT_HEADING_CHANGE_INTERVAL
            speed = MALE_DRIFT_SPEED

        self.speed = speed

        # --- Apply the movement step ---
        self.x += math.cos(self.heading) * self.speed * dt
        self.y += math.sin(self.heading) * self.speed * dt

        # --- Bounce off arena walls (same mirrored-reflection logic
        #     used by FemaleFly) so pursuit never drives him off-screen ---
        if self.x - MALE_FLY_RADIUS < 0:
            self.x = MALE_FLY_RADIUS
            self.heading = math.pi - self.heading
        elif self.x + MALE_FLY_RADIUS > ARENA_WIDTH:
            self.x = ARENA_WIDTH - MALE_FLY_RADIUS
            self.heading = math.pi - self.heading

        if self.y - MALE_FLY_RADIUS < 0:
            self.y = MALE_FLY_RADIUS
            self.heading = -self.heading
        elif self.y + MALE_FLY_RADIUS > ARENA_HEIGHT:
            self.y = ARENA_HEIGHT - MALE_FLY_RADIUS
            self.heading = -self.heading

    def print_eye_nodes(self):
        """Print the current live eye-node and motor-node values to the console."""
        if self.nearest_distance is None:
            print(
                "MaleFly LC10 nodes -> L: 0.000 | R: 0.000 || "
                "Motors -> L: 0.000 | R: 0.000 | (no female detected, drifting)"
            )
        else:
            print(
                f"MaleFly LC10 nodes -> "
                f"L: {self.left_eye_node:.3f} | R: {self.right_eye_node:.3f} || "
                f"Motors -> L: {self.turn_left_motor:.3f} | R: {self.turn_right_motor:.3f} | "
                f"dist: {self.nearest_distance:6.1f}px | "
                f"rel_angle: {self.relative_angle_deg:6.1f} deg | "
                f"side: {self.tracking_label}"
            )

    def update(self, female_flies, dt):
        # 1. Sense: update the LC10-style eye nodes from the nearest female.
        self.update_visual_tracking(female_flies)
        # 2. Act: feed those eye values into the motor nodes and move.
        self.update_motor_control(dt)

    def draw(self, surface):
        pygame.draw.circle(
            surface,
            MALE_FLY_COLOR,
            (int(self.x), int(self.y)),
            MALE_FLY_RADIUS,
        )


class FlyArena:
    """Owns the pygame window and runs the main simulation loop."""

    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((ARENA_WIDTH, ARENA_HEIGHT))
        pygame.display.set_caption("Fly Arena")
        self.clock = pygame.time.Clock()

        # Spawn the female flies at random positions within the arena.
        self.female_flies = [
            FemaleFly(
                random.uniform(FEMALE_FLY_RADIUS, ARENA_WIDTH - FEMALE_FLY_RADIUS),
                random.uniform(FEMALE_FLY_RADIUS, ARENA_HEIGHT - FEMALE_FLY_RADIUS),
            )
            for _ in range(NUM_FEMALE_FLIES)
        ]

        # Spawn the single male fly at the exact center of the arena.
        self.male_fly = MaleFly(ARENA_WIDTH / 2, ARENA_HEIGHT / 2)

        self.running = True

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

    def update(self, dt):
        for fly in self.female_flies:
            fly.update(dt)
        # Pass the current female flies so the male can locate the
        # nearest one and update his simulated visual-tracking nodes.
        self.male_fly.update(self.female_flies, dt)
        self.resolve_collisions()
        self.keep_flies_in_bounds()

    def resolve_collisions(self):
        flies = [*self.female_flies, self.male_fly]
        for index, first in enumerate(flies):
            for second in flies[index + 1:]:
                self.resolve_collision(first, second)

    def resolve_collision(self, first, second):
        dx = second.x - first.x
        dy = second.y - first.y
        minimum_distance = first.radius + second.radius
        distance_squared = dx * dx + dy * dy

        if distance_squared >= minimum_distance * minimum_distance:
            return

        distance = math.sqrt(distance_squared)
        if distance == 0:
            angle = random.uniform(0, 2 * math.pi)
            normal_x = math.cos(angle)
            normal_y = math.sin(angle)
            distance = 1e-6
        else:
            normal_x = dx / distance
            normal_y = dy / distance

        overlap = minimum_distance - distance
        first.x -= normal_x * overlap / 2
        first.y -= normal_y * overlap / 2
        second.x += normal_x * overlap / 2
        second.y += normal_y * overlap / 2

        first_vx = math.cos(first.heading) * first.speed
        first_vy = math.sin(first.heading) * first.speed
        second_vx = math.cos(second.heading) * second.speed
        second_vy = math.sin(second.heading) * second.speed

        first_normal_velocity = first_vx * normal_x + first_vy * normal_y
        second_normal_velocity = second_vx * normal_x + second_vy * normal_y
        if second_normal_velocity - first_normal_velocity >= 0:
            return

        first_vx += (second_normal_velocity - first_normal_velocity) * normal_x
        first_vy += (second_normal_velocity - first_normal_velocity) * normal_y
        second_vx += (first_normal_velocity - second_normal_velocity) * normal_x
        second_vy += (first_normal_velocity - second_normal_velocity) * normal_y

        first_speed = math.hypot(first_vx, first_vy)
        second_speed = math.hypot(second_vx, second_vy)
        if first_speed > 1e-6:
            first.heading = math.atan2(first_vy, first_vx)
        if second_speed > 1e-6:
            second.heading = math.atan2(second_vy, second_vx)

    def keep_flies_in_bounds(self):
        flies = [*self.female_flies, self.male_fly]
        for fly in flies:
            if fly.x - fly.radius < 0:
                fly.x = fly.radius
                if math.cos(fly.heading) < 0:
                    fly.heading = math.pi - fly.heading
            elif fly.x + fly.radius > ARENA_WIDTH:
                fly.x = ARENA_WIDTH - fly.radius
                if math.cos(fly.heading) > 0:
                    fly.heading = math.pi - fly.heading

            if fly.y - fly.radius < 0:
                fly.y = fly.radius
                if math.sin(fly.heading) < 0:
                    fly.heading = -fly.heading
            elif fly.y + fly.radius > ARENA_HEIGHT:
                fly.y = ARENA_HEIGHT - fly.radius
                if math.sin(fly.heading) > 0:
                    fly.heading = -fly.heading

    def draw(self):
        self.screen.fill(BACKGROUND_COLOR)
        for fly in self.female_flies:
            fly.draw(self.screen)
        self.male_fly.draw(self.screen)
        pygame.display.flip()

    def run(self):
        """Main simulation loop: handle input, update state, render."""
        while self.running:
            dt = min(self.clock.tick(FPS) / 1000.0, MAX_FRAME_TIME)
            self.handle_events()
            self.update(dt)
            self.draw()
        pygame.quit()


if __name__ == "__main__":
    arena = FlyArena()
    arena.run()
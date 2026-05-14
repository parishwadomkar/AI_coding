import math
import pygame
import numpy as np
import gymnasium as gym
from gymnasium import spaces


WIDTH = 1000
HEIGHT = 600
FPS = 60

BG = (0, 0, 0)
FG = (245, 245, 245)

BORDER = 8

PADDLE_W = 12
PADDLE_H = 80
PADDLE_MARGIN = 18
PADDLE_SPEED = 6

BALL_SIZE = 12
BALL_SPEED_X = 6
BALL_SPEED_Y = 4

BASE_BALL_SPEED = math.hypot(BALL_SPEED_X, BALL_SPEED_Y)
MAX_BALL_SPEED = 1.8 * BASE_BALL_SPEED
MIN_BALL_SPEED = BASE_BALL_SPEED
LEFT_PADDLE_REBOUND_DAMPING = 0.8

MAX_REBOUND_ANGLE_DEG = 60
MAX_REBOUND_ANGLE_RAD = math.radians(MAX_REBOUND_ANGLE_DEG)

CENTER_HIT_THRESHOLD = 0.20
MIDDLE_HIT_THRESHOLD = 0.60

CENTER_HIT_SPEED_MULTIPLIER = 1.0
MIDDLE_HIT_SPEED_MULTIPLIER = 1.5
EDGE_HIT_SPEED_MULTIPLIER = 2.0

LEFT_X = PADDLE_MARGIN
RIGHT_X = WIDTH - PADDLE_MARGIN - PADDLE_W

DASH_H = 14
DASH_GAP = 10
SUCCESSFUL_RETURN_REWARD = 0.02
TARGET_ALIGNMENT_REWARD_MAX = 0.05
TARGET_ALIGNMENT_SIGMA = 120.0

def clamp(v, lo, hi):
    return max(lo, min(hi, v))


class Paddle:
    def __init__(self, x, y):
        self.x = float(x)
        self.y = float(y)
        self.target_y = y

    def reset(self, x, y):
        self.x = float(x)
        self.y = float(y)
        self.target_y = self.center_y

    @property
    def rect(self):
        return pygame.Rect(int(self.x), int(self.y), PADDLE_W, PADDLE_H)

    @property
    def center_y(self):
        return self.y + PADDLE_H / 2

    def update(self):
        dy = self.target_y - self.center_y

        if abs(dy) < PADDLE_SPEED:
            self.y += dy
        else:
            self.y += PADDLE_SPEED if dy > 0 else -PADDLE_SPEED

        self.y = clamp(self.y, BORDER, HEIGHT - BORDER - PADDLE_H)


class Ball:
    def __init__(self):
        self.x = WIDTH / 2
        self.y = HEIGHT / 2
        self.vx = BALL_SPEED_X
        self.vy = BALL_SPEED_Y

    @property
    def rect(self):
        return pygame.Rect(int(self.x), int(self.y), BALL_SIZE, BALL_SIZE)

    @property
    def center_y(self):
        return self.y + BALL_SIZE / 2

    def reset(self, rng):
        self.x = WIDTH / 2
        self.y = float(rng.uniform(HEIGHT * 0.2, HEIGHT * 0.8))

        self.vx = abs(BALL_SPEED_X)

        vy = float(rng.uniform(-BALL_SPEED_Y, BALL_SPEED_Y))
        if abs(vy) < 1.5:
            vy = 1.5 if vy >= 0 else -1.5
        self.vy = vy

    def update(self):
        self.x += self.vx
        self.y += self.vy

        if self.y <= BORDER:
            self.y = BORDER
            self.vy = abs(self.vy)

        if self.y + BALL_SIZE >= HEIGHT - BORDER:
            self.y = HEIGHT - BORDER - BALL_SIZE
            self.vy = -abs(self.vy)


def unbeatable_ai(paddle, ball):
    if ball.vx < 0:
        paddle.target_y = clamp(
            ball.center_y,
            BORDER + PADDLE_H / 2,
            HEIGHT - BORDER - PADDLE_H / 2,
        )
    else:
        paddle.target_y = HEIGHT / 2

def apply_right_paddle_rebound(ball, paddle):
    half_paddle = PADDLE_H / 2

    normalized_offset = (ball.center_y - paddle.center_y) / half_paddle
    normalized_offset = clamp(normalized_offset, -1.0, 1.0)

    abs_offset = abs(normalized_offset)

    if abs_offset <= CENTER_HIT_THRESHOLD:
        speed_multiplier = CENTER_HIT_SPEED_MULTIPLIER
    elif abs_offset <= MIDDLE_HIT_THRESHOLD:
        speed_multiplier = MIDDLE_HIT_SPEED_MULTIPLIER
    else:
        speed_multiplier = EDGE_HIT_SPEED_MULTIPLIER

    incoming_speed = math.hypot(ball.vx, ball.vy)
    outgoing_speed = min(incoming_speed * speed_multiplier, MAX_BALL_SPEED)

    rebound_angle = normalized_offset * MAX_REBOUND_ANGLE_RAD

    ball.vx = -abs(outgoing_speed * math.cos(rebound_angle))
    ball.vy = outgoing_speed * math.sin(rebound_angle)

    return normalized_offset, speed_multiplier, outgoing_speed


def apply_left_paddle_rebound_damping(ball):
    incoming_speed = math.hypot(ball.vx, ball.vy)

    if incoming_speed == 0:
        ball.vx = BALL_SPEED_X
        ball.vy = BALL_SPEED_Y
        return MIN_BALL_SPEED

    outgoing_speed = max(
        incoming_speed * LEFT_PADDLE_REBOUND_DAMPING,
        MIN_BALL_SPEED,
    )

    speed_scale = outgoing_speed / incoming_speed

    ball.vx = abs(ball.vx) * speed_scale
    ball.vy = ball.vy * speed_scale

    return outgoing_speed


def reflect_y_into_bounds(y, top, bottom):
    span = bottom - top

    if span <= 0:
        return HEIGHT / 2

    shifted = y - top
    folded = shifted % (2 * span)

    if folded <= span:
        return top + folded
    else:
        return bottom - (folded - span)


def predict_ball_y_at_next_right_return(ball):
    x = ball.x
    y = ball.center_y
    vx = ball.vx
    vy = ball.vy

    left_contact_x = LEFT_X + PADDLE_W
    right_contact_x = RIGHT_X - BALL_SIZE

    if vx < 0:
        time_to_left = max((x - left_contact_x) / abs(vx), 0.0)
        time_left_to_right = max(
            (right_contact_x - left_contact_x) / abs(vx),
            0.0,
        )
        total_time = time_to_left + time_left_to_right

    elif vx > 0:
        total_time = max((right_contact_x - x) / vx, 0.0)

    else:
        return HEIGHT / 2

    predicted_y = y + vy * total_time

    top = BORDER + BALL_SIZE / 2
    bottom = HEIGHT - BORDER - BALL_SIZE / 2

    return reflect_y_into_bounds(predicted_y, top, bottom)


def compute_target_alignment_reward(target_y, predicted_return_y):
    alignment_error = abs(target_y - predicted_return_y)

    reward_fraction = math.exp(
        -0.5 * (alignment_error / TARGET_ALIGNMENT_SIGMA) ** 2
    )

    alignment_reward = TARGET_ALIGNMENT_REWARD_MAX * reward_fraction

    return alignment_error, alignment_reward
    

def draw_background(screen):
    screen.fill(BG)

    pygame.draw.rect(screen, FG, (0, 0, WIDTH, BORDER))
    pygame.draw.rect(screen, FG, (0, HEIGHT - BORDER, WIDTH, BORDER))

    x = WIDTH // 2
    y = BORDER + 10
    while y < HEIGHT - BORDER:
        pygame.draw.rect(screen, FG, (x - 2, y, 4, DASH_H))
        y += DASH_H + DASH_GAP


class ModifiedPongEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": FPS}
    def __init__(self, n_action_bins=11, render_mode=None, max_internal_frames=5000, max_episode_decisions=50):
        super().__init__()
        self.n_action_bins = n_action_bins
        self.render_mode = render_mode
        self.max_internal_frames = max_internal_frames
        self.max_episode_decisions = max_episode_decisions
        self.decision_count = 0
        self.last_right_hit_offset = 0.0
        self.last_right_speed_multiplier = 1.0
        self.last_right_outgoing_speed = BASE_BALL_SPEED    
        self.action_space = spaces.Discrete(self.n_action_bins)
        self.last_left_outgoing_speed = BASE_BALL_SPEED
        self.pending_decision_reward = 0.0
        self.successful_agent_returns = 0
        self.last_selected_target_y = HEIGHT / 2
        self.last_predicted_return_y = HEIGHT / 2
        self.last_target_alignment_error = 0.0
        self.last_target_alignment_reward = 0.0
        self.cumulative_target_alignment_reward = 0.0
    
        low = np.array([0.0, 0.0, -1.0, -1.0, 0.0, 0.0, -1.0, -1.0, 0.0], dtype=np.float32)
        high = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
    
        self.left = Paddle(LEFT_X, HEIGHT / 2)
        self.right = Paddle(RIGHT_X, HEIGHT / 2)
        self.ball = Ball()
        self.left_score = 0
        self.right_score = 0
        self.first_phase = True
        self.agent_can_act = False
        self.total_frames = 0
        self.rally_hits = 0
        self.screen = None
        self.clock = None
        self.font = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.left.reset(LEFT_X, HEIGHT / 2)
        self.right.reset(RIGHT_X, HEIGHT / 2)
        self.ball.reset(self.np_random)
        self.left_score = 0
        self.right_score = 0
        self.first_phase = True
        self.agent_can_act = False
        self.total_frames = 0
        self.rally_hits = 0
        self.decision_count = 0
        self.last_right_hit_offset = 0.0
        self.last_right_speed_multiplier = 1.0
        self.last_right_outgoing_speed = BASE_BALL_SPEED
        self.last_left_outgoing_speed = BASE_BALL_SPEED
        self.pending_decision_reward = 0.0
        self.successful_agent_returns = 0
        self.last_selected_target_y = HEIGHT / 2
        self.last_predicted_return_y = HEIGHT / 2
        self.last_target_alignment_error = 0.0
        self.last_target_alignment_reward = 0.0
        self.cumulative_target_alignment_reward = 0.0
        reward, terminated, truncated = self._advance_to_decision_or_score()
        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    def step(self, action):
        action = int(action)
        target_y = self._action_to_target_y(action)
        predicted_return_y = predict_ball_y_at_next_right_return(self.ball)
        (alignment_error, alignment_reward) = compute_target_alignment_reward(
            target_y=target_y,predicted_return_y=predicted_return_y)
    
        self.last_selected_target_y = target_y
        self.last_predicted_return_y = predicted_return_y
        self.last_target_alignment_error = alignment_error
        self.last_target_alignment_reward = alignment_reward
        self.cumulative_target_alignment_reward += alignment_reward
        self.right.target_y = target_y
        self.agent_can_act = False
        self.decision_count += 1
        reward, terminated, truncated = self._advance_to_decision_or_score()
        reward += alignment_reward
    
        if not terminated and not truncated:
            if self.decision_count >= self.max_episode_decisions:
                truncated = True
    
        obs = self._get_obs()
        info = self._get_info()
    
        if truncated and not terminated:
            if self.decision_count >= self.max_episode_decisions: info["truncation_reason"] = "max_episode_decisions"
            else: info["truncation_reason"] = "max_internal_frames"
        return obs, reward, terminated, truncated, info

    def _action_to_target_y(self, action):
        lo = BORDER + PADDLE_H / 2
        hi = HEIGHT - BORDER - PADDLE_H / 2
        if self.n_action_bins == 1: return HEIGHT / 2
        frac = action / (self.n_action_bins - 1)
        return lo + frac * (hi - lo)

    def _get_obs(self):
        ball_x = self.ball.x / WIDTH
        ball_y = self.ball.center_y / HEIGHT
        ball_vx = self.ball.vx / MAX_BALL_SPEED
        ball_vy = self.ball.vy / MAX_BALL_SPEED
        right_y = self.right.center_y / HEIGHT
        left_y = self.left.center_y / HEIGHT
        rel_right = (self.ball.center_y - self.right.center_y) / HEIGHT
        rel_left = (self.ball.center_y - self.left.center_y) / HEIGHT
        predicted_return_y = predict_ball_y_at_next_right_return(self.ball) / HEIGHT
        obs = np.array([ball_x, ball_y, ball_vx, ball_vy, right_y, left_y, rel_right, rel_left, predicted_return_y], dtype=np.float32)
        return np.clip(obs, self.observation_space.low, self.observation_space.high)

    def _get_info(self):
        return {
            "left_score": self.left_score,
            "right_score": self.right_score,
            "score_diff": self.right_score - self.left_score,
            "total_frames": self.total_frames,
            "rally_hits": self.rally_hits,
            "decision_count": self.decision_count,
            "first_phase": self.first_phase,
            "agent_can_act": self.agent_can_act,
            "last_right_hit_offset": self.last_right_hit_offset,
            "last_right_speed_multiplier": self.last_right_speed_multiplier,
            "last_right_outgoing_speed": self.last_right_outgoing_speed,
            "current_ball_speed": math.hypot(self.ball.vx, self.ball.vy),
            "last_left_outgoing_speed": self.last_left_outgoing_speed,
            "successful_agent_returns": self.successful_agent_returns,
            "last_selected_target_y": self.last_selected_target_y,
            "last_predicted_return_y": self.last_predicted_return_y,
            "last_target_alignment_error": self.last_target_alignment_error,
            "last_target_alignment_reward": self.last_target_alignment_reward,
            "cumulative_target_alignment_reward": self.cumulative_target_alignment_reward,
        }

    def _advance_to_decision_or_score(self):
        internal_frames = 0
        while True:
            reward, terminated = self._simulate_one_frame()
            internal_frames += 1
            self.total_frames += 1
            if self.render_mode == "human": self.render()
            if terminated: return reward, True, False
            if self.agent_can_act and not self.first_phase:
                reward = self.pending_decision_reward
                self.pending_decision_reward = 0.0
                return reward, False, False
            if internal_frames >= self.max_internal_frames: return 0.0, False, True

    def _simulate_one_frame(self):
        unbeatable_ai(self.left, self.ball)
        if self.first_phase:
            self.right.y = self.ball.center_y - PADDLE_H / 2
            self.right.y = clamp(self.right.y, BORDER, HEIGHT - BORDER - PADDLE_H)
    
        self.left.update()
        self.right.update()
        self.ball.update()

        if self.ball.rect.colliderect(self.left.rect):
            self.ball.x = self.left.x + PADDLE_W
            self.last_left_outgoing_speed = apply_left_paddle_rebound_damping(self.ball)
    
        if self.ball.rect.colliderect(self.right.rect):
            was_scripted_first_hit = self.first_phase
            self.ball.x = self.right.x - BALL_SIZE
        
            (
                self.last_right_hit_offset,
                self.last_right_speed_multiplier,
                self.last_right_outgoing_speed,
            ) = apply_right_paddle_rebound(self.ball, self.right)
        
            self.rally_hits += 1
            if was_scripted_first_hit:
                self.pending_decision_reward = 0.0
                self.first_phase = False
            else:
                self.pending_decision_reward = SUCCESSFUL_RETURN_REWARD
                self.successful_agent_returns += 1
            self.agent_can_act = True
    
        if self.ball.x < 0:
            self.right_score += 1
            return 1.0, True
    
        if self.ball.x > WIDTH:
            self.left_score += 1
            return -1.0, True
    
        return 0.0, False

    def set_rendering(self, enabled):
        if enabled: self.render_mode = "human"
        else:
            self.render_mode = None
            self.close()

    def render(self):
        if self.render_mode != "human": return
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
            pygame.display.set_caption("Modified Pong RL Environment")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.SysFont("consolas", 60)
        self.clock.tick(FPS)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.set_rendering(False)
                return
                
        draw_background(self.screen)
        pygame.draw.rect(self.screen, FG, self.left.rect)
        pygame.draw.rect(self.screen, FG, self.right.rect)
        pygame.draw.rect(self.screen, FG, self.ball.rect)
        left_s = self.font.render(str(self.left_score), True, FG)
        right_s = self.font.render(str(self.right_score), True, FG)
        self.screen.blit(left_s, (WIDTH // 2 - 120, 20))
        self.screen.blit(right_s, (WIDTH // 2 + 60, 20))
        pygame.display.flip()

    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None
            self.clock = None
            self.font = None
            
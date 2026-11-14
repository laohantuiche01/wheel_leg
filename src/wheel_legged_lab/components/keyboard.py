# Copyright (c) 2026 SCUTRobotLab
# SPDX-License-Identifier: MIT
"""Keyboard playback adapted from scripts/rsl_rl/keyboard_controller.py in the old project.

Key semantics are retained; the input callback only edits CPU state. Environment
commands are written once per control step through the public command methods.
"""
from dataclasses import dataclass
import json
import math
from pathlib import Path
import time
import weakref


@dataclass(frozen=True)
class KeyboardCommand:
    vx: float
    wz: float
    height: float
    jump: bool = False


class KeyboardCommandState:
    """Simulator-independent held keys, bounded height integration and jump edges."""
    keys = frozenset(("W", "S", "A", "D", "Z", "X", "L", "Q"))
    motion_keys = frozenset(("W", "S", "A", "D", "Z", "X"))

    def __init__(self, settings, initial_height, *, speed=2., turn_rate=2., height_rate=.2,
                 fixed_vx=None, fixed_wz=None, fixed_height=None, pulse_hold=.2,
                 unified=False, height_range=(.20, .38)):
        values = (speed, turn_rate, height_rate, initial_height, fixed_vx, fixed_wz, fixed_height, pulse_hold)
        if (any(v is not None and not math.isfinite(v) for v in values)
                or min(speed, turn_rate, height_rate) <= 0 or pulse_hold < 0):
            raise ValueError("Keyboard rates must be finite and positive; commands must be finite")
        self.vx_range, self.wz_range = settings.commands.vx, settings.commands.wz
        self.height_range = settings.commands.height
        self.unified = unified
        if unified:
            if (len(height_range) != 2 or not all(math.isfinite(v) for v in height_range)
                    or not 0 < height_range[0] < height_range[1]):
                raise ValueError("Keyboard height range must be finite, positive and increasing")
            self.vx_range, self.wz_range = (-speed, speed), (-turn_rate, turn_rate)
            self.height_range = tuple(height_range)
        self.speed, self.turn_rate, self.height_rate = speed, turn_rate, height_rate
        self.fixed_vx, self.fixed_wz, self.fixed_height = fixed_vx, fixed_wz, fixed_height
        self.can_translate = unified or (settings.task != "jump" and self.vx_range != (0., 0.))
        self.can_rotate = unified or (settings.task != "jump" and self.wz_range != (0., 0.))
        self.can_jump = unified or (settings.task == "jump" and settings.jump.enabled)
        if not self.can_translate and fixed_vx not in (None, 0., 0):
            raise ValueError("This task does not accept forward velocity commands")
        if not self.can_rotate and fixed_wz not in (None, 0., 0):
            raise ValueError("This task does not accept yaw rate commands")
        self.default_height = float(fixed_height if fixed_height is not None else initial_height)
        self.height = self.default_height
        if unified and not self.height_range[0] <= self.height <= self.height_range[1]:
            raise ValueError(f"Initial keyboard target height must lie in {self.height_range}")
        self.can_change_height = (unified or fixed_height is None) and self.height_range[0] < self.height_range[1]
        self.pressed = set()
        self.pulse_hold = pulse_hold
        self._unobserved_presses = set()
        self._pulses = {}
        self._jump_pending = False
        self.clear_epoch = 0

    def reset(self):
        """L or an episode reset: clear held commands and restore initial height."""
        self.release_all()
        self.height = self.default_height

    def release_all(self):
        """Focus loss: stop motion/height changes and discard queued jump edges."""
        self.pressed.clear()
        self._unobserved_presses.clear()
        self._pulses.clear()
        self._jump_pending = False
        self.clear_epoch += 1
        if self.unified:
            self.fixed_vx = self.fixed_wz = None

    def _take_axis(self, key):
        if self.unified:
            if key in ('W', 'S'):
                self.fixed_vx = None
            elif key in ('A', 'D'):
                self.fixed_wz = None

    def handle_key(self, key, pressed, *, repeat=False):
        key = key.upper()
        if key not in self.keys:
            return False
        if repeat:
            return True
        self._take_axis(key)
        if not pressed:
            # Some remote/IME input paths send press AND release between two
            # policy ticks. Preserve those short pulses rather than losing them.
            if key in self._unobserved_presses and key in self.motion_keys:
                self._pulses[key] = self.pulse_hold
            else:
                self._pulses.pop(key, None)
            self._unobserved_presses.discard(key)
            self.pressed.discard(key)
        elif key == "L":
            self.reset()
        elif key not in self.pressed:
            self.pressed.add(key)
            self._unobserved_presses.add(key)
            if key == "Q" and self.can_jump:
                self._jump_pending = True
        return True

    def handle_character(self, character):
        """Support character-only remote input without repeating L/Q actions."""
        key = character.upper()
        self._take_axis(key)
        if key in self.motion_keys and key not in self.pressed:
            self._pulses[key] = self.pulse_hold

    def advance(self, dt):
        if not math.isfinite(dt) or dt < 0:
            raise ValueError("Keyboard dt must be finite and nonnegative")
        active = self.pressed | {key for key, remaining in self._pulses.items() if remaining > 1e-9}
        def axis(positive, negative):
            return int(positive in active) - int(negative in active)
        def bounded(value, bounds):
            return max(min(bounds[0], 0.), min(max(bounds[1], 0.), value))
        vx = bounded(axis("W", "S") * self.speed, self.vx_range) if self.can_translate else 0.
        wz = bounded(axis("A", "D") * self.turn_rate, self.wz_range) if self.can_rotate else 0.
        if self.can_change_height:
            self.height = max(self.height_range[0], min(self.height_range[1],
                self.height + axis("Z", "X") * self.height_rate * dt))
        jump, self._jump_pending = self._jump_pending, False
        self._unobserved_presses.clear()
        self._pulses = {key: remaining - dt for key, remaining in self._pulses.items() if remaining - dt > 1e-9}
        return KeyboardCommand(float(vx if self.fixed_vx is None else self.fixed_vx),
            float(wz if self.fixed_wz is None else self.fixed_wz), self.height, jump)


class PlaybackKeyboard:
    """Omniverse input adapter; instantiate only after AppLauncher has started."""
    def __init__(self, env, state, log_path=None):
        import carb.input
        import omni.appwindow
        self.env, self.state = env, state
        self._types = carb.input.KeyboardEventType
        self._window = omni.appwindow.get_default_app_window()
        if self._window is None or self._window.get_keyboard() is None:
            raise RuntimeError("Keyboard playback requires an Isaac Sim GUI window")
        self._keyboard = self._window.get_keyboard()
        self._input = carb.input.acquire_input_interface()
        self._subscription = None
        self._last_command = None
        self._last_print = -math.inf
        self._last_trace = -math.inf
        self._last_trace_command = None
        self._last_jump_status = None
        self._last_clear_epoch = state.clear_epoch
        self._log = None
        if log_path is not None:
            log_path = Path(log_path)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            self._log = log_path.open("w", buffering=1)
            print(f"[INFO] Keyboard input log: {log_path}", flush=True)
        reference = weakref.ref(self)
        def callback(event, *args):
            obj = reference()
            return obj._on_event(event) if obj is not None else True
        self._subscription = self._input.subscribe_to_keyboard_events(self._keyboard, callback)
        self._trace("start", focused=self._window.is_focused(), pulse_hold=state.pulse_hold,
                    unified=state.unified, height_range=state.height_range)

    def _trace(self, kind, **data):
        if self._log is not None:
            self._log.write(json.dumps(dict(time=time.time(), kind=kind, **data)) + "\n")

    def _on_event(self, event):
        key = event.input if isinstance(event.input, str) else event.input.name
        self._trace("input", key=key, event=str(event.type), focused=self._window.is_focused())
        if event.type == self._types.KEY_PRESS:
            # Never latch a key received while the application is unfocused.
            if self._window.is_focused():
                self.state.handle_key(key, True)
        elif event.type == self._types.KEY_RELEASE:
            self.state.handle_key(key, False)
        elif event.type == getattr(self._types, "CHAR", None) and self._window.is_focused():
            self.state.handle_character(key)
        return True  # preserve old Carb event propagation behavior

    def reset(self):
        self.state.reset()
        self._trace("reset")

    def advance(self, dt):
        if not self._window.is_focused():
            self.state.release_all()
        if self.state.clear_epoch != self._last_clear_epoch:
            cancel = getattr(self.env, "cancel_jump_request", None)
            if cancel is not None:
                cancel()
            self._last_clear_epoch = self.state.clear_epoch
        command = self.state.advance(dt)
        self.env.set_play_velocity_command(vx=command.vx, vy=0., wz=command.wz)
        self.env.set_height_command(command.height)
        if command.jump:
            self.env.request_jump_takeoff()
            self._trace("jump_request")
        now = time.monotonic()
        if now - self._last_trace >= 1. or command != self._last_trace_command:
            data = dict(focused=self._window.is_focused(), held=sorted(self.state.pressed),
                        pulses=sorted(self.state._pulses), requested=[command.vx, command.wz, command.height])
            if hasattr(self.env, "commands"):
                data["applied_velocity"] = self.env.commands.velocity[0].tolist()
            if hasattr(self.env, "reader"):
                measured = self.env.reader.read()
                if measured.axle_linear_velocity_heading is not None:
                    data["measured_axle_velocity"] = measured.axle_linear_velocity_heading[0].tolist()
                if measured.yaw_rate_world is not None:
                    data["measured_yaw_rate"] = float(measured.yaw_rate_world[0])
                if measured.axle_position_world is not None:
                    data["axle_position_world"] = measured.axle_position_world[0].tolist()
            self._trace("command", **data)
            self._last_trace, self._last_trace_command = now, command
        if command != self._last_command and (now - self._last_print >= .25 or command.jump):
            print(f"[Keyboard] vx={command.vx:+.3f} m/s, wz={command.wz:+.3f} rad/s, "
                  f"height={command.height:.3f} m" + (", jump requested" if command.jump else ""), flush=True)
            self._last_print = now
            self._last_command = command
        return command

    def report_jump_status(self):
        jump = getattr(self.env, "jump", None)
        if not self.state.can_jump or not hasattr(jump, "diagnostics"):
            return
        status = jump.diagnostics()
        key = (status['phase'], status['pending'], status['attempts'], status['releases'],
               status['takeoffs'], status['successes'], status['failures'],
               tuple(status['blocked']) if status['pending'] else ())
        if key == self._last_jump_status:
            return
        previous = self._last_jump_status
        self._last_jump_status = key
        self._trace("jump_status", **status)
        if status['pending']:
            detail = 'request queued; waiting for ' + (', '.join(status['blocked']) or 'next control step')
        elif previous is not None and status['failures'] > previous[6]:
            detail = (f"attempt failed; max upward speed={status['max_upward_speed']:.3f} m/s, "
                      f"release requires >= {status['release_speed_required']:.3f} m/s "
                      "and height/unloading gate")
        else:
            detail = f"attempts={status['attempts']}, releases={status['releases']}, takeoffs={status['takeoffs']}, landings={status['successes']}"
        print(f"[Jump] {status['phase']}: {detail}", flush=True)

    def help_text(self):
        lines = ["[INFO] Keyboard enabled: click the Isaac Sim viewport to focus it.",
                 "[INFO] Hold keys to move; release to request zero velocity. L clears keyboard commands."]
        if self.state.unified:
            lines.append("[INFO] Unified manual control: W/S, A/D, Q and Z/X are enabled for every task.")
            lines.append("[INFO] Command availability does not add behaviors that the loaded policy was never trained to perform.")
        if self.state.pulse_hold:
            lines.append(f"[INFO] Short/character input pulses are held for {self.state.pulse_hold:g} s; L clears immediately.")
        lines.append(f"[INFO] W/S: forward/backward up to {self.state.speed:g} m/s" if self.state.can_translate
                     else "[INFO] W/S disabled: this task has zero translation commands.")
        lines.append(f"[INFO] A/D: positive/negative yaw up to {self.state.turn_rate:g} rad/s" if self.state.can_rotate
                     else "[INFO] A/D disabled: this task has zero yaw commands.")
        lines.append("[INFO] Z/X: raise/lower target height; release to hold it." if self.state.can_change_height
                     else f"[INFO] Target height locked at {self.state.height:.3f} m.")
        if self.state.can_jump:
            lines.append("[INFO] Q: request one jump when ready (hold does not repeat).")
        if any(v is not None for v in (self.state.fixed_vx, self.state.fixed_wz, self.state.fixed_height)):
            lines.append("[INFO] CLI commands are initial targets; matching keys take over, L clears velocity targets."
                         if self.state.unified else "[INFO] Explicit --vx/--wz/--command_height override keyboard input on those axes.")
        return "\n".join(lines)

    def close(self):
        subscription = self._subscription
        self._subscription = None
        if subscription is not None:
            self._input.unsubscribe_to_keyboard_events(self._keyboard, subscription)
        self.state.release_all()
        if self._log is not None:
            self._trace("close")
            self._log.close()
            self._log = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass  # Kit interfaces may already have been unloaded at shutdown.

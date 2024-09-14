import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

class Motor:
    def __init__(self, motor_type="SPM", pole_pairs=4, Rs=0.01, Lq_base=0.005, Ld_base=0.005,
                 bemf_const_base=0.1, inertia=0.01, friction_coeff=0.001):
        self.motor_type = motor_type
        self.pole_pairs = pole_pairs
        self.Rs = Rs
        self.Lq_base = Lq_base
        self.Ld_base = Ld_base
        self.bemf_const_base = bemf_const_base
        self.inertia = inertia
        self.friction_coeff = friction_coeff

    def inductance(self, Iq, Id):
        Is = np.sqrt(Iq**2 + Id**2)  # Total current magnitude
        Lq = self.Lq_base * (1 + 0.01 * Is)
        Ld = self.Ld_base * (1 + 0.01 * Is)
        return Lq, Ld

    def bemf_constant(self, angle, harmonics=None):
        bemf = self.bemf_const_base * np.sin(self.pole_pairs * angle)
        if harmonics:
            for h, mag in harmonics.items():
                bemf += mag * np.sin(h * self.pole_pairs * angle)
        return bemf

    def phase_bemf(self, angle, phase_shift):
        return self.bemf_const_base * np.sin(self.pole_pairs * (angle + phase_shift))

    def torque(self, Iq, Id):
        Lq, Ld = self.inductance(Iq, Id)
        if self.motor_type == "SPM":
            torque = 1.5 * self.pole_pairs * self.bemf_const_base * Iq
        elif self.motor_type == "IPM":
            torque = 1.5 * self.pole_pairs * (self.bemf_const_base * Iq + (Lq - Ld) * Iq * Id)
        elif self.motor_type == "Reluctance":
            torque = 1.5 * self.pole_pairs * (Lq - Ld) * Iq * Id
        return torque

class Simulation:
    def __init__(self, time_step=100e-9, total_time=0.01):
        self.time_step = time_step      # Simulation time_step must be the divisor of the sampling_time with no remainder.
        self.total_time = total_time
        self.time_points = np.arange(0, total_time, time_step)

class Application:
    def __init__(self, speed_control=True, commanded_speed=1000, commanded_iq=1.0, commanded_id=0.0,
                 speed_ramp_rate=1000000, current_ramp_rate=10000, vBus = 48):
        self.speed_control = speed_control
        self.commanded_speed = commanded_speed
        self.commanded_iq = commanded_iq
        self.commanded_id = commanded_id
        self.speed_ramp_rate = speed_ramp_rate
        self.current_ramp_rate = current_ramp_rate
        self.vBus = vBus        

class MotorControl:
    def __init__(self, Kp=100.0, Ki=1, sampling_time=62.5e-6, deadTime = 300e-9):
        self.Kp = Kp
        self.Ki = Ki
        self.sampling_time = sampling_time
        self.integral_error_iq = 0
        self.integral_error_id = 0
        self.last_update_time = 0
        self.deadTime = deadTime        # deadTime must be divided by simulation time_step with no remainder.

    def pi_control(self, error_iq, error_id, current_time, Vq, Vd):
        # Only update the control at the specified sampling time step
        if current_time - self.last_update_time >= self.sampling_time:
            self.integral_error_iq += error_iq * self.sampling_time
            self.integral_error_id += error_id * self.sampling_time            
            Vq = self.Kp * error_iq + self.Ki * self.integral_error_iq
            Vd = self.Kp * error_id + self.Ki * self.integral_error_id
            self.last_update_time = current_time
        else:
            return Vq, Vd
        
        return Vq, Vd

def inverse_park_transform(Vq, Vd, angle):
    V_alpha = Vq * np.cos(angle) - Vd * np.sin(angle)
    V_beta = Vq * np.sin(angle) + Vd * np.cos(angle)
    
    Va = V_alpha
    Vb = -0.5 * V_alpha + (np.sqrt(3) / 2) * V_beta
    Vc = -0.5 * V_alpha - (np.sqrt(3) / 2) * V_beta    
    return Va, Vb, Vc

def park_transform(Ia, Ib, Ic, angle):
    Iq = np.sqrt(2/3) * (Ia * np.cos(angle) + Ib * np.cos(angle - 2*np.pi/3) + Ic * np.cos(angle + 2*np.pi/3))
    Id = np.sqrt(2/3) * (-Ia * np.sin(angle) - Ib * np.sin(angle - 2*np.pi/3) - Ic * np.sin(angle + 2*np.pi/3))
    return Iq, Id

def phase_current_ode(t, currents, Va, Vb, Vc, motor, angle):
    Ia, Ib, Ic = currents
    Is = np.sqrt(Ia**2 + Ib**2 + Ic**2)  # Total current magnitude
    Lq, Ld = motor.inductance(Is, 0)
    bemf_a = motor.phase_bemf(angle, 0)
    bemf_b = motor.phase_bemf(angle, -2 * np.pi / 3)
    bemf_c = motor.phase_bemf(angle, 2 * np.pi / 3)

    dIa_dt = (Va - motor.Rs * Ia - bemf_a) / Lq
    dIb_dt = (Vb - motor.Rs * Ib - bemf_b) / Ld
    dIc_dt = (Vc - motor.Rs * Ic - bemf_c) / Ld

    return [dIa_dt, dIb_dt, dIc_dt]


def center_aligned_pwm_with_deadtime(Va, Vb, Vc, Vbus, t, switching_freq, deadtime):
    """
    Generate center-aligned PWM signals with symmetrical dead time for each phase based on voltage commands.
    Va, Vb, Vc: phase voltages
    Vbus: bus voltage
    t: current simulation time
    switching_freq: PWM switching frequency
    deadtime: dead time between switching events (seconds)
    """
    switch_period = 1 / switching_freq
    half_period = switch_period / 2

    # Time in the switching period
    time_in_period = t % switch_period

    # Create two triangular carrier waveforms with dead time shifts
    # Top transistors carrier: shifted by +deadtime/2
    # Bottom transistors carrier: shifted by -deadtime/2
    top_carrier_wave = np.abs((time_in_period - half_period - deadtime / 2) / half_period)
    bottom_carrier_wave = np.abs((time_in_period - half_period + deadtime / 2) / half_period)

    # Calculate duty cycle for each phase voltage (normalized by bus voltage)
    duty_a = Va / Vbus + 0.5  # Duty cycle between 0 and 1
    duty_b = Vb / Vbus + 0.5
    duty_c = Vc / Vbus + 0.5

    # Generate the PWM signals (-1 for low, 1 for high)
    pwm_a_top = 1 if top_carrier_wave < duty_a else -1
    pwm_b_top = 1 if top_carrier_wave < duty_b else -1
    pwm_c_top = 1 if top_carrier_wave < duty_c else -1

    pwm_a_bottom = 1 if bottom_carrier_wave < duty_a else -1
    pwm_b_bottom = 1 if bottom_carrier_wave < duty_b else -1
    pwm_c_bottom = 1 if bottom_carrier_wave < duty_c else -1

    return np.array([pwm_a_top, pwm_b_top, pwm_c_top]), np.array([pwm_a_bottom, pwm_b_bottom, pwm_c_bottom])


def apply_voltage_during_deadtime(Ia, Ib, Ic, pwm_signals_top, pwm_signals_bottom, motor):
    """
    Adjust the applied voltage during the dead time based on current direction.
    """
    # Initialize applied voltages
    Va_applied, Vb_applied, Vc_applied = 0, 0, 0

    # Phase A
    if pwm_signals_top[0] == -1 and pwm_signals_bottom[0] == -1:
        if Ia > 0:  # Positive current
            Va_applied = 0  # Bottom transistor's voltage (ground)
        else:  # Negative current
            Va_applied = app.vBus  # Top transistor's voltage (bus voltage)
    else:
        Va_applied = (pwm_signals_top[0] - pwm_signals_bottom[0]) * app.vBus / 2

    # Phase B
    if pwm_signals_top[1] == -1 and pwm_signals_bottom[1] == -1:
        if Ib > 0:  # Positive current
            Vb_applied = 0  # Bottom transistor's voltage (ground)
        else:  # Negative current
            Vb_applied = app.vBus  # Top transistor's voltage (bus voltage)
    else:
        Vb_applied = (pwm_signals_top[1] - pwm_signals_bottom[1]) * app.vBus / 2

    # Phase C
    if pwm_signals_top[2] == -1 and pwm_signals_bottom[2] == -1:
        if Ic > 0:  # Positive current
            Vc_applied = 0  # Bottom transistor's voltage (ground)
        else:  # Negative current
            Vc_applied = app.vBus  # Top transistor's voltage (bus voltage)
    else:
        Vc_applied = (pwm_signals_top[2] - pwm_signals_bottom[2]) * app.vBus / 2

    return Va_applied, Vb_applied, Vc_applied


iq_list = []
id_list = []

iq_cmd_list = []
id_cmd_list = []

def simulate_motor(motor, sim, app, control):
    speed = 0
    angle = 0
    iq_ramped = 0
    id_ramped = 0
    Vq = 0
    Vd = 0
    torque = []
    bemf = []
    currents = []    

    Ia, Ib, Ic = 0, 0, 0

    for t in sim.time_points:
        if app.speed_control:
            if speed < app.commanded_speed:
                speed += app.speed_ramp_rate * sim.time_step
        else:
            speed += torque_current / motor.inertia * sim.time_step

        if iq_ramped < app.commanded_iq:
            iq_ramped += app.current_ramp_rate * sim.time_step
        if id_ramped > app.commanded_id:   # id is always negative
            id_ramped -= app.current_ramp_rate * sim.time_step

        iq_cmd_list.append(iq_ramped)
        id_cmd_list.append(id_ramped)


        Iq_sensed, Id_sensed = park_transform(Ia, Ib, Ic, angle)
        error_iq = iq_ramped - Iq_sensed
        error_id = id_ramped - Id_sensed

        iq_list.append(Iq_sensed)
        id_list.append(Id_sensed)

        Vq, Vd = control.pi_control(error_iq, error_id, t, Vq, Vd)

        # Transform Vq and Vd to Va, Vb, Vc using inverse Park-Clarke
        Va, Vb, Vc = inverse_park_transform(Vq, Vd, angle)

        # Generate center-aligned PWM signals for each phase with dead time compensation
        pwm_signals_top, pwm_signals_bottom = center_aligned_pwm_with_deadtime(Va, Vb, Vc, app.vBus, t, (1/control.sampling_time), control.deadTime)

        # Adjust the applied voltages during dead time based on current direction
        Va_applied, Vb_applied, Vc_applied = apply_voltage_during_deadtime(Ia, Ib, Ic, pwm_signals_top, pwm_signals_bottom, motor)

        # Solve the ODE for phase currents over one time step
        sol = solve_ivp(phase_current_ode, [t, t + sim.time_step], [Ia, Ib, Ic],
                        args=(Va_applied, Vb_applied, Vc_applied, motor, angle), method='RK45')


        Ia, Ib, Ic = sol.y[:, -1]
        currents.append([Ia, Ib, Ic])        

        torque_current = motor.torque(Iq_sensed, Id_sensed)
        torque.append(torque_current)

        bemf_a = motor.phase_bemf(angle, 0)
        bemf_b = motor.phase_bemf(angle, -2 * np.pi / 3)
        bemf_c = motor.phase_bemf(angle, 2 * np.pi / 3)
        bemf.append([bemf_a, bemf_b, bemf_c])

        angle += speed * sim.time_step

    return torque, bemf, currents

# Instantiate objects
motor = Motor()
sim = Simulation()
app = Application()
control = MotorControl()

# Run the simulation
torque, bemf, currents = simulate_motor(motor, sim, app, control)

# Plot results
time_points = sim.time_points
currents = np.array(currents)

plt.figure(figsize=(10, 8))

# Plot phase currents
plt.subplot(4, 1, 1)
plt.plot(time_points, currents[:, 0], label='Ia')
plt.plot(time_points, currents[:, 1], label='Ib')
plt.plot(time_points, currents[:, 2], label='Ic')
plt.title('Phase Currents')
plt.legend()

plt.subplot(4, 1, 2)
plt.plot(time_points, iq_list, label='iqSensed')
plt.plot(time_points, id_list, label='idSensed')
plt.plot(time_points, iq_cmd_list, label='iqCmd')
plt.plot(time_points, id_cmd_list, label='idCmd')
plt.title('Iq, Id Cmd + Sensed')
plt.legend()

# Plot torque
plt.subplot(4, 1, 3)
plt.plot(time_points, torque)
plt.title('Torque')

# Plot back EMF
plt.subplot(4, 1, 4)
plt.plot(time_points, bemf)
plt.title('Back EMF')

plt.tight_layout()
plt.show()

'''
TODO:
* Add bus voltage
* Add switching behavior
* Add ability to transition to short circuit

'''







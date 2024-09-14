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
    def __init__(self, time_step=62.5e-6, total_time=1.0):
        self.time_step = time_step
        self.total_time = total_time
        self.time_points = np.arange(0, total_time, time_step)

class Application:
    def __init__(self, speed_control=True, commanded_speed=1000, commanded_iq=100.0, commanded_id=0.0,
                 speed_ramp_rate=100, current_ramp_rate=10000):
        self.speed_control = speed_control
        self.commanded_speed = commanded_speed
        self.commanded_iq = commanded_iq
        self.commanded_id = commanded_id
        self.speed_ramp_rate = speed_ramp_rate
        self.current_ramp_rate = current_ramp_rate

class MotorControl:
    def __init__(self, Kp=100.0, Ki=1, sampling_time=62.5e-6):
        self.Kp = Kp
        self.Ki = Ki
        self.sampling_time = sampling_time
        self.integral_error_iq = 0
        self.integral_error_id = 0
        self.last_update_time = 0

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
    
    Va = np.sqrt(2/3) * (Vd * np.cos(angle) - Vq * np.sin(angle))
    Vb = np.sqrt(2/3) * (Vd * np.cos(angle - 2*np.pi/3) - Vq * np.sin(angle - 2*np.pi/3))
    Vc = np.sqrt(2/3) * (Vd * np.cos(angle + 2*np.pi/3) - Vq * np.sin(angle + 2*np.pi/3))
    
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

iq_list = []
id_list = []

iq_cmd_list = []
id_cmd_list = []

def simulate_motor(motor, sim, app, control):
    speed = 0
    angle = 0
    iq = 0
    id = 0
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

        if iq < app.commanded_iq:
            iq += app.current_ramp_rate * sim.time_step
        if id > app.commanded_id:   # id is always negative
            id -= app.current_ramp_rate * sim.time_step

        iq_cmd_list.append(iq)
        id_cmd_list.append(id)


        Iq_sensed, Id_sensed = park_transform(Ia, Ib, Ic, angle)
        error_iq = iq - Iq_sensed
        error_id = id - Id_sensed

        iq_list.append(Iq_sensed)
        id_list.append(Id_sensed)

        Vq, Vd = control.pi_control(error_iq, error_id, t, Vq, Vd)

        Va, Vb, Vc = inverse_park_transform(Vq, Vd, angle)

        # Solve the ODE for phase currents over one time step
        sol = solve_ivp(phase_current_ode, [t, t + sim.time_step], [Ia, Ib, Ic],
                        args=(Va, Vb, Vc, motor, angle), method='RK45')

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

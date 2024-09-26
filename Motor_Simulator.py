import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

class Motor:
    def __init__(self, motor_type="SPM", pole_pairs=4, Rs=0.005, Lq_base=0.0003, Ld_base=0.0003,
                 bemf_const_base=0.1, inertia=0.0, friction_coeff=0.0, i_max = 600):
        self.motor_type = motor_type
        self.pole_pairs = pole_pairs
        self.Rs = Rs
        self.Lq_base = Lq_base
        self.Ld_base = Ld_base
        self.Lq = Lq_base
        self.Ld = Ld_base        
        self.Laa = 0
        self.Lbb = 0
        self.Lcc = 0
        self.Lab = 0
        self.Lac = 0
        self.Lbc = 0
        self.Laa_dot = 0
        self.Lbb_dot = 0
        self.Lcc_dot = 0
        self.Lab_dot = 0
        self.Lac_dot = 0
        self.Lbc_dot = 0
        self.bemf_const_base = bemf_const_base
        self.bemf_a = 0
        self.bemf_b = 0
        self.bemf_c = 0
        self.harmonics = None
        # self.harmonics = {1: {'harmonic': 5, 'mag': bemf_const_base / 20},
        #                   2: {'harmonic': 7, 'mag': bemf_const_base / 20}}                
        self.inertia = inertia
        self.friction_coeff = friction_coeff
        self.i_max = i_max

        

    def inductance_dq(self, Iq, Id):
        Is = np.sqrt(Iq**2 + Id**2)  # Total current magnitude
        # Todo: Need to add magnetic saturation (Reduction in inductance as a function of current)
        self.Lq = self.Lq_base# * (1 - 0.5 * Is/self.i_max)
        self.Ld = self.Ld_base# * (1 - 0.5 * Is/self.i_max)
    
    def inductance_abc(self, theta):
        # Inductance matrix in the abc frame
        self.Laa = self.Ld# * (np.cos(theta))**2 + self.Lq * (np.sin(theta))**2
        self.Lbb = self.Ld# * (np.cos(theta - 2*np.pi/3))**2 + self.Lq * (np.sin(theta - 2*np.pi/3))**2
        self.Lcc = self.Ld# * (np.cos(theta + 2*np.pi/3))**2 + self.Lq * (np.sin(theta + 2*np.pi/3))**2

        self.Lab = 0#(self.Ld - self.Lq) * np.cos(theta) * np.cos(theta - 2*np.pi/3)
        self.Lac = 0#(self.Ld - self.Lq) * np.cos(theta) * np.cos(theta + 2*np.pi/3)
        self.Lbc = 0#(self.Ld - self.Lq) * np.cos(theta - 2*np.pi/3) * np.cos(theta + 2*np.pi/3)   

    def inductance_abc_dot(self, theta, speed):        
        # Derivatives for self-inductances
        self.Laa_dot = 0#(2 * (self.Lq - self.Ld) * np.sin(theta) * np.cos(theta)) * speed
        self.Lbb_dot = 0#(2 * (self.Lq - self.Ld) * np.sin(theta - 2*np.pi/3) * np.cos(theta - 2*np.pi/3)) * speed
        self.Lcc_dot = 0#(2 * (self.Lq - self.Ld) * np.sin(theta + 2*np.pi/3) * np.cos(theta + 2*np.pi/3)) * speed
        
        # Derivatives for mutual inductances
        self.Lab_dot = 0#((self.Ld - self.Lq) * (-np.sin(theta) * np.cos(theta - 2*np.pi/3) - np.cos(theta) * np.sin(theta - 2*np.pi/3))) * speed
        self.Lac_dot = 0#((self.Ld - self.Lq) * (-np.sin(theta) * np.cos(theta + 2*np.pi/3) - np.cos(theta) * np.sin(theta + 2*np.pi/3))) * speed
        self.Lbc_dot = 0#((self.Ld - self.Lq) * (-np.sin(theta - 2*np.pi/3) * np.cos(theta + 2*np.pi/3) - np.cos(theta - 2*np.pi/3) * np.sin(theta + 2*np.pi/3))) * speed        

    def phase_bemf(self, angle, phase_shift):
        bemf = self.bemf_const_base * np.cos(angle + phase_shift)
        if self.harmonics:
            for _, data in self.harmonics.items():
                bemf += data['mag'] * np.cos(data['harmonic'] * angle + phase_shift)        
        return bemf

    def torque(self, Iq, Id):
        if self.motor_type == "SPM":
            torque = 1.5 * self.pole_pairs * self.bemf_const_base * Iq
        elif self.motor_type == "IPM":
            torque = 1.5 * self.pole_pairs * (self.bemf_const_base * Iq + (self.Lq - self.Ld) * Iq * Id)
        elif self.motor_type == "Reluctance":
            torque = 1.5 * self.pole_pairs * (self.Lq - self.Ld) * Iq * Id
        return torque

class Simulation:
    def __init__(self, time_step=3125e-9, total_time=0.3):
        self.time_step = time_step      # Simulation time_step must be the divisor of the sampling_time with no remainder.
        self.total_time = total_time
        self.time_points = np.arange(0, total_time, time_step)

class Application:
    def __init__(self, speed_control=True, commanded_speed=100, commanded_iq=20.0, commanded_id=0.0,
                 speed_ramp_rate=0.0, current_ramp_rate=7000.0, vBus = 48):
        self.speed_control = speed_control
        self.commanded_speed = commanded_speed
        self.commanded_iq = commanded_iq
        self.commanded_id = commanded_id
        self.speed_ramp_rate = speed_ramp_rate
        self.current_ramp_rate = current_ramp_rate
        self.vBus = vBus        

class MotorControl:
    def __init__(self, Kp=1.0, Ki=50.0, sampling_time=62.5e-6, deadTime = 0):
        self.Kp = Kp
        self.Ki = Ki
        self.sampling_time = sampling_time
        self.integral_error_iq = 0
        self.integral_error_id = 0
        self.last_update_time = 0
        self.deadTime = deadTime        # deadTime must be divided by simulation time_step with no remainder.
        self.saturation = 0

    def pi_control(self, error_iq, error_id, current_time, Vq, Vd, vbus):
        # Only update the control at the specified sampling time step
        if (current_time - self.last_update_time) >= self.sampling_time:
            self.integral_error_iq += error_iq * self.sampling_time * (1 - self.saturation)
            self.integral_error_id += error_id * self.sampling_time * (1 - self.saturation)            
            Vq = self.Kp * error_iq + self.Ki * self.integral_error_iq
            Vd = self.Kp * error_id + self.Ki * self.integral_error_id
            self.last_update_time = current_time
            if (Vq**2 + Vd**2 > vbus**2):
                volt_amp_gain = vbus / np.sqrt(Vq**2 + Vd**2)
                self.saturation = 1
                Vq *= volt_amp_gain
            else:
                self.saturation = 0

        else:
            return Vq, Vd
        
        return Vq, Vd

# Inverse Direct DQ transformation
# Switched q and d from the original inverse transformation because q and d currents were switched.
def inverse_dq_transform(q, d, angle):
    a = q * np.cos(angle) - d * np.sin(angle)
    b = q * np.cos(angle - 2*np.pi/3) - d * np.sin(angle - 2*np.pi/3)
    c = q * np.cos(angle + 2*np.pi/3) - d * np.sin(angle + 2*np.pi/3)
    return a, b, c

# Direct DQ transformation
def dq_transform(a, b, c, angle):
    q = (2/3) * (a * np.cos(angle) + b * np.cos(angle - 2*np.pi/3) + c * np.cos(angle + 2*np.pi/3))
    d = (2/3) * (-a * np.sin(angle) - b * np.sin(angle - 2*np.pi/3) - c * np.sin(angle + 2*np.pi/3))
    return q, d

def phase_current_ode(t, currents, va, vb, vc, motor):
    ia, ib, ic = currents
    
    # A is the inductance matrix with neutral voltage handling
    A = np.array([
        [motor.Laa, motor.Lab, motor.Lac, 1],
        [motor.Lab, motor.Lbb, motor.Lbc, 1],
        [motor.Lac, motor.Lbc, motor.Lcc, 1],
        [1,         1,         1,         0]
        ])

    # The b vector (applied voltages minus resistive and flux terms)
    b = np.array([
        va - ia * motor.Rs - motor.bemf_a,# - motor.Laa_dot * ia - motor.Lab_dot * ib - motor.Lac_dot * ic,
        vb - ib * motor.Rs - motor.bemf_b,# - motor.Lab_dot * ia - motor.Lbb_dot * ib - motor.Lbc_dot * ic,
        vc - ic * motor.Rs - motor.bemf_c,# - motor.Lac_dot * ia - motor.Lbc_dot * ib - motor.Lcc_dot * ic,
        0   # KCL constraint i_a + i_b + i_c = 0
        ])

    # Solve for current derivatives and neutral voltage, Ax = b
    x = np.linalg.solve(A, b)

    di_a_dt, di_b_dt, di_c_dt, V_n = x

    return [di_a_dt, di_b_dt, di_c_dt]

def center_aligned_pwm_with_deadtime(Va, Vb, Vc, Vbus, t, switching_freq, dead_time):
    """
    Generates center-aligned PWM signals for top and bottom transistors with dead-time insertion.
    """
    pwm_period = 1 / switching_freq
    half_period = pwm_period / 2

    # Calculate the time in the current PWM period
    time_in_period = t % pwm_period

    # Calculate duty cycles for each phase (between 0 and 1)
    duty_a = (Va / Vbus + 1) / 2
    duty_b = (Vb / Vbus + 1) / 2
    duty_c = (Vc / Vbus + 1) / 2

    # Create two triangular carrier waveforms with dead time shifts
    carrier_wave = time_in_period / half_period if time_in_period < half_period else (pwm_period - time_in_period) / half_period

    # Generate the PWM signals (1 for high, -1 for low)
    pwm_a_top = 1 if carrier_wave > (1 - duty_a + (dead_time / 2) / half_period) else 0
    pwm_b_top = 1 if carrier_wave > (1 - duty_b + (dead_time / 2) / half_period) else 0
    pwm_c_top = 1 if carrier_wave > (1 - duty_c + (dead_time / 2) / half_period) else 0

    # The bottom transistors should be the inverse of the top ones, with dead time compensation
    pwm_a_bottom = 1 if carrier_wave < (1 - duty_a - (dead_time / 2) / half_period) else 0
    pwm_b_bottom = 1 if carrier_wave < (1 - duty_b - (dead_time / 2) / half_period) else 0
    pwm_c_bottom = 1 if carrier_wave < (1 - duty_c - (dead_time / 2) / half_period) else 0

    return np.array([pwm_a_top, pwm_b_top, pwm_c_top]), np.array([pwm_a_bottom, pwm_b_bottom, pwm_c_bottom])


def apply_voltage_during_deadtime(Ia, Ib, Ic, pwm_signals_top, pwm_signals_bottom, motor):
    """
    Adjust the applied voltage during the dead time based on current direction.
    """
    # Initialize applied voltages
    Va_applied, Vb_applied, Vc_applied = 0, 0, 0

    # # Phase A
    # if pwm_signals_top[0] == 0 and pwm_signals_bottom[0] == 0:
    #     if Ia > 0:  # Positive current
    #         Va_applied = 0  # Bottom transistor's voltage (ground)
    #     else:  # Negative current
    #         Va_applied = app.vBus  # Top transistor's voltage (bus voltage)
    # else:
    Va_applied = pwm_signals_top[0] * app.vBus

    # # Phase B
    # if pwm_signals_top[1] == 0 and pwm_signals_bottom[1] == 0:
    #     if Ib > 0:  # Positive current
    #         Vb_applied = 0  # Bottom transistor's voltage (ground)
    #     else:  # Negative current
    #         Vb_applied = app.vBus  # Top transistor's voltage (bus voltage)
    # else:
    Vb_applied = pwm_signals_top[1] * app.vBus

    # # Phase C
    # if pwm_signals_top[2] == 0 and pwm_signals_bottom[2] == 0:
    #     if Ic > 0:  # Positive current
    #         Vc_applied = 0  # Bottom transistor's voltage (ground)
    #     else:  # Negative current
    #         Vc_applied = app.vBus  # Top transistor's voltage (bus voltage)
    # else:
    Vc_applied = pwm_signals_top[2] * app.vBus

    return Va_applied, Vb_applied, Vc_applied


iq_list = []
id_list = []

iq_cmd_list = []
id_cmd_list = []

Va_Applied_list = []
Vb_Applied_list = []
Vc_Applied_list = []
Va_list = []
Vb_list = []
Vc_list = []
err_q_list = []
err_d_list = []
Vq_list = []
Vd_list = []
angle_e_list = []

def simulate_motor(motor, sim, app, control):
    speed_m = 0
    speed_e = 0
    angle_m = 0
    angle_e = 0
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
            if (speed_m < app.commanded_speed) and (app.speed_ramp_rate != 0):            
                speed_m += app.speed_ramp_rate * sim.time_step
            else:
                speed_m = app.commanded_speed
        else:
            speed_m += (torque_current / motor.inertia) * sim.time_step
        speed_e = speed_m * motor.pole_pairs

        if iq_ramped < app.commanded_iq:
            iq_ramped += app.current_ramp_rate * sim.time_step
        if id_ramped > app.commanded_id:   # id is always negative
            id_ramped -= app.current_ramp_rate * sim.time_step

        iq_cmd_list.append(iq_ramped)
        id_cmd_list.append(id_ramped)


        Iq_sensed, Id_sensed = dq_transform(Ia, Ib, Ic, angle_e)
        error_iq = iq_ramped - Iq_sensed
        error_id = id_ramped - Id_sensed
        
        err_q_list.append(error_iq)
        err_d_list.append(error_id)

        iq_list.append(Iq_sensed)
        id_list.append(Id_sensed)

        # Va = 0.5 * np.cos(angle_e + 0.1)
        # Vb = 0.5 * np.cos(angle_e + 0.1 - 2 * np.pi / 3)
        # Vc = 0.5 * np.cos(angle_e + 0.1 + 2 * np.pi / 3)
        # Vq, Vd = dq_transform(Va, Vb, Vc, angle_e)    

        Vq, Vd = control.pi_control(error_iq, error_id, t, Vq, Vd, app.vBus)
        Vq_list.append(Vq)
        Vd_list.append(Vd)

        # Transform Vq and Vd to Va, Vb, Vc using inverse Park-Clarke
        Va, Vb, Vc = inverse_dq_transform(Vq, Vd, angle_e)
        Va_list.append(Va)
        Vb_list.append(Vb)
        Vc_list.append(Vc)

        # Generate center-aligned PWM signals for each phase with dead time compensation
        pwm_signals_top, pwm_signals_bottom = center_aligned_pwm_with_deadtime(Va, Vb, Vc, app.vBus, t, (1/control.sampling_time), control.deadTime)

        # Adjust the applied voltages during dead time based on current direction
        Va_Applied, Vb_Applied, Vc_Applied = apply_voltage_during_deadtime(Ia, Ib, Ic, pwm_signals_top, pwm_signals_bottom, motor)
        Va_Applied_list.append(Va_Applied)
        Vb_Applied_list.append(Vb_Applied)
        Vc_Applied_list.append(Vc_Applied)

        # Update Lq, Ld
        motor.inductance_dq(Iq_sensed, Id_sensed)
        
        # Update phase self and mutual inductances
        motor.inductance_abc(angle_e)

        # Update phase self and mutual inductances time derivative 
        motor.inductance_abc_dot(angle_e, speed_e)

        # Calculate the phase bemf
        motor.bemf_a = speed_m * motor.phase_bemf(angle_e, 0)
        motor.bemf_b = speed_m * motor.phase_bemf(angle_e, -2 * np.pi / 3)
        motor.bemf_c = speed_m * motor.phase_bemf(angle_e, 2 * np.pi / 3)
        bemf.append([motor.bemf_a, motor.bemf_b, motor.bemf_c])

        # Solve the ODE for phase currents over one time step
        sol = solve_ivp(phase_current_ode, [t, t + sim.time_step], [Ia, Ib, Ic],
                        args=(Va_Applied, Vb_Applied, Vc_Applied, motor), method='RK45')    

        Ia, Ib, Ic = sol.y[:, -1]
        currents.append([Ia, Ib, Ic])        

        torque_current = motor.torque(Iq_sensed, Id_sensed)
        torque.append(torque_current)

        angle_m += speed_m * sim.time_step
        angle_e += speed_e * sim.time_step
        angle_e_list.append(angle_e)

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
plt.subplot(4, 1, 4)
plt.plot(time_points, currents[:, 0], label='Ia')
plt.plot(time_points, currents[:, 1], label='Ib')
plt.plot(time_points, currents[:, 2], label='Ic')
plt.title('Phase Currents')
plt.legend()

# plt.subplot(4, 1, 2)
# plt.plot(time_points, err_q_list, label='q_err')
# plt.plot(time_points, err_d_list, label='d_err')
# plt.title('Iq, Id errors')
# plt.legend()

plt.subplot(4, 1, 2)
plt.plot(time_points, Va_list, label='Va')
plt.plot(time_points, Vb_list, label='Vb')
plt.plot(time_points, Vc_list, label='Vc')
plt.title('V')
plt.legend()


# plt.subplot(4, 1, 2)
# plt.plot(time_points, Va_Applied_list, label='Va_App')
# plt.plot(time_points, Vb_Applied_list, label='Vb_App')
# plt.plot(time_points, Vc_Applied_list, label='Vc_App')
# plt.title('Applied V')
# plt.legend()
plt.subplot(4, 1, 1)
plt.plot(time_points, iq_list, label='iqSensed')
plt.plot(time_points, id_list, label='idSensed')
plt.plot(time_points, iq_cmd_list, label='iqCmd')
plt.plot(time_points, id_cmd_list, label='idCmd')
plt.title('Iq, Id Cmd + Sensed')
plt.legend()

# # Plot torque
# plt.subplot(4, 1, 3)
# plt.plot(time_points, torque)
# plt.title('Torque')
plt.subplot(4, 1, 3)
plt.plot(time_points, Vq_list, label='Vq')
plt.plot(time_points, Vd_list, label='Vd')
plt.title('Vq, Vd')
plt.legend()



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







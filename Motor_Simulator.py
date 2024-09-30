import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import control as ctrl

class Motor:
    def __init__(self, motor_type="SYNC", pole_pairs=4, Rs=0.005, Lq_base=0.0001, Ld_base=0.0001,
                 bemf_const_base=0.1, inertia=0.0, visc_fric_coeff=0.0, i_max = 600):
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
        #                   2: {'harmonic': 5, 'mag': bemf_const_base / 20},
        #                   3: {'harmonic': 9, 'mag': bemf_const_base / 40},
        #                   4: {'harmonic': 11, 'mag': bemf_const_base / 40}}
        self.inertia = inertia
        self.visc_fric_coeff = visc_fric_coeff
        self.i_max = i_max

        

    def inductance_dq(self, Iq, Id):
        """
        Update the Lq, Ld inductances based on current amplitude
        """
        Is = np.sqrt(Iq**2 + Id**2)  # Total current magnitude
        # Assuming inductance reduces by half at peak current.
        self.Lq = self.Lq_base * (1 - 0.5 * Is/self.i_max)
        self.Ld = self.Ld_base * (1 - 0.5 * Is/self.i_max)
    
    def inductance_abc(self, theta):
        """
        Update the inductances in the abc frame (Based on the dq transform)
        """        
        self.Laa = self.Ld * (np.cos(theta))**2 + self.Lq * (np.sin(theta))**2
        self.Lbb = self.Ld * (np.cos(theta - 2*np.pi/3))**2 + self.Lq * (np.sin(theta - 2*np.pi/3))**2
        self.Lcc = self.Ld * (np.cos(theta + 2*np.pi/3))**2 + self.Lq * (np.sin(theta + 2*np.pi/3))**2

        self.Lab = (self.Ld - self.Lq) * np.cos(theta) * np.cos(theta - 2*np.pi/3)
        self.Lac = (self.Ld - self.Lq) * np.cos(theta) * np.cos(theta + 2*np.pi/3)
        self.Lbc = (self.Ld - self.Lq) * np.cos(theta - 2*np.pi/3) * np.cos(theta + 2*np.pi/3)   

    def inductance_abc_dot(self, theta, speed):        
        """
        Update the inductances time derivatives
        """  
        # Derivatives for self inductances
        self.Laa_dot = (2 * (self.Lq - self.Ld) * np.sin(theta) * np.cos(theta)) * speed
        self.Lbb_dot = (2 * (self.Lq - self.Ld) * np.sin(theta - 2*np.pi/3) * np.cos(theta - 2*np.pi/3)) * speed
        self.Lcc_dot = (2 * (self.Lq - self.Ld) * np.sin(theta + 2*np.pi/3) * np.cos(theta + 2*np.pi/3)) * speed
        
        # Derivatives for mutual inductances
        self.Lab_dot = ((self.Ld - self.Lq) * (-np.sin(theta) * np.cos(theta - 2*np.pi/3) - np.cos(theta) * np.sin(theta - 2*np.pi/3))) * speed
        self.Lac_dot = ((self.Ld - self.Lq) * (-np.sin(theta) * np.cos(theta + 2*np.pi/3) - np.cos(theta) * np.sin(theta + 2*np.pi/3))) * speed
        self.Lbc_dot = ((self.Ld - self.Lq) * (-np.sin(theta - 2*np.pi/3) * np.cos(theta + 2*np.pi/3) - np.cos(theta - 2*np.pi/3) * np.sin(theta + 2*np.pi/3))) * speed        

    def phase_bemf(self, angle, phase_shift):
        """
        Calculate bemf, allow for harmonics.
        """          
        bemf = self.bemf_const_base * np.cos(angle + phase_shift)
        if self.harmonics:
            for _, data in self.harmonics.items():
                bemf += data['mag'] * np.cos(data['harmonic'] * angle + phase_shift)        
        return bemf

    def torque(self, Iq, Id):
        """
        Calculate Torque based on motor type: Synchronous or asynchronous
        """           
        if self.motor_type == "SYNC":
            torque = 1.5 * self.pole_pairs * (self.bemf_const_base * Iq + (self.Lq - self.Ld) * Iq * Id)
        else: # ASYNC motor
            torque = 1.5 * self.pole_pairs * (self.Lq - self.Ld) * Iq * Id
        return torque

class Simulation:
    def __init__(self, time_step=100e-9, total_time=0.05):
        self.time_step = time_step
        self.total_time = total_time
        self.time_points = np.arange(0, total_time, time_step)

class Application:
    def __init__(self, speed_control=True, commanded_speed=100, commanded_iq=50.0, commanded_id=-50.0,
                 speed_ramp_rate=10000.0, current_ramp_rate=7000.0, vBus = 48, init_speed = 0, short_circuit = True):
        self.speed_control = speed_control
        self.commanded_speed = commanded_speed
        self.commanded_iq = commanded_iq
        self.commanded_id = commanded_id
        self.speed_ramp_rate = speed_ramp_rate
        self.current_ramp_rate = current_ramp_rate
        self.vBus = vBus
        self.init_speed = init_speed
        self.short_circuit = short_circuit

class MotorControl:
    def __init__(self, Kp_d=5.0, Ki_d=200.0, Kp_q=5.0, Ki_q=200.0, sampling_time=62.5e-6, deadTime = 300e-9):
        self.Kp_d = Kp_d
        self.Ki_d = Ki_d
        self.Kp_q = Kp_q
        self.Ki_q = Ki_q
        self.sampling_time = sampling_time
        self.integral_error_iq = 0
        self.integral_error_id = 0
        self.last_update_time = 0
        self.deadTime = deadTime
        self.saturation = 0

    def pi_control(self, error_iq, error_id, current_time, Vq, Vd, vbus):
        """
        Parallel current loop PI controller.
        """          
        # Only update the control at the specified sampling time step
        if (current_time - self.last_update_time) >= self.sampling_time:
            self.integral_error_iq += error_iq * self.sampling_time * (1 - self.saturation)
            self.integral_error_id += error_id * self.sampling_time * (1 - self.saturation)            
            Vq = self.Kp_q * error_iq + self.Ki_q * self.integral_error_iq
            Vd = self.Kp_d * error_id + self.Ki_d * self.integral_error_id
            self.last_update_time = current_time
            # Saturation handling
            if ((Vq**2 + Vd**2) > vbus**2):
                volt_amp_gain = vbus / np.sqrt(Vq**2 + Vd**2)
                self.saturation = 1
                Vq *= volt_amp_gain
                Vd *= volt_amp_gain
            else:
                self.saturation = 0
        else:
            return Vq, Vd
        
        return Vq, Vd

def inverse_dq_transform(q, d, angle):
    '''
    Inverse Direct DQ transformation
    Switched q and d from the original inverse transformation because q and d currents were switched.    
    '''
    a = q * np.cos(angle) - d * np.sin(angle)
    b = q * np.cos(angle - 2*np.pi/3) - d * np.sin(angle - 2*np.pi/3)
    c = q * np.cos(angle + 2*np.pi/3) - d * np.sin(angle + 2*np.pi/3)
    return a, b, c

# Direct DQ transformation
def dq_transform(a, b, c, angle):
    '''
    Direct DQ transformation
    '''    
    q = (2/3) * (a * np.cos(angle) + b * np.cos(angle - 2*np.pi/3) + c * np.cos(angle + 2*np.pi/3))
    d = (2/3) * (-a * np.sin(angle) - b * np.sin(angle - 2*np.pi/3) - c * np.sin(angle + 2*np.pi/3))
    return q, d

def phase_current_ode(t, currents, va, vb, vc, motor):
    '''
    Solve for current time derivatives.
    '''    

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
        va - ia * motor.Rs - motor.bemf_a - motor.Laa_dot * ia - motor.Lab_dot * ib - motor.Lac_dot * ic,
        vb - ib * motor.Rs - motor.bemf_b - motor.Lab_dot * ia - motor.Lbb_dot * ib - motor.Lbc_dot * ic,
        vc - ic * motor.Rs - motor.bemf_c - motor.Lac_dot * ia - motor.Lbc_dot * ib - motor.Lcc_dot * ic,
        0   # KCL constraint i_a + i_b + i_c = 0
        ])

    # Solve for current derivatives and neutral voltage, Ax = b
    x = np.linalg.solve(A, b)

    di_a_dt, di_b_dt, di_c_dt, V_n = x

    return [di_a_dt, di_b_dt, di_c_dt]

def center_aligned_pwm_with_deadtime(Va, Vb, Vc, Vbus, t, switching_freq, dead_time):
    """
    Generates center-aligned PWM signals for top and bottom transistors with dead-time.
    """
    pwm_period = 1 / switching_freq
    half_period = pwm_period / 2

    # Calculate the time in the current PWM period
    time_in_period = t % pwm_period

    # Calculate duty cycles for each phase (between 0 and 1, default is 0.5)
    duty_a = (Va / Vbus + 1) / 2
    duty_b = (Vb / Vbus + 1) / 2
    duty_c = (Vc / Vbus + 1) / 2

    # Create a triangular carrier waveform
    carrier_wave = time_in_period / half_period if time_in_period < half_period else (pwm_period - time_in_period) / half_period

    # Generate the top and bottom PWM signals w/ dead time compensation (1 for high, 0 for low)
    pwm_a_top = 1 if carrier_wave > (1 - duty_a + (dead_time / 2) / half_period) else 0
    pwm_b_top = 1 if carrier_wave > (1 - duty_b + (dead_time / 2) / half_period) else 0
    pwm_c_top = 1 if carrier_wave > (1 - duty_c + (dead_time / 2) / half_period) else 0

    pwm_a_bottom = 1 if carrier_wave < (1 - duty_a - (dead_time / 2) / half_period) else 0
    pwm_b_bottom = 1 if carrier_wave < (1 - duty_b - (dead_time / 2) / half_period) else 0
    pwm_c_bottom = 1 if carrier_wave < (1 - duty_c - (dead_time / 2) / half_period) else 0

    return np.array([pwm_a_top, pwm_b_top, pwm_c_top]), np.array([pwm_a_bottom, pwm_b_bottom, pwm_c_bottom])


def terminal_voltage_with_deadtime(Ia, Ib, Ic, pwm_signals_top, pwm_signals_bottom, motor):
    """
    Set the terminal voltages while taking dead time into account based on current direction.
    """
    # Initialize applied voltages
    Va_Terminal, Vb_Terminal, Vc_Terminal = 0, 0, 0

    # Phase A
    if pwm_signals_top[0] == 0 and pwm_signals_bottom[0] == 0:
        if Ia > 0:
            Va_Terminal = 0  # Bottom transistor's voltage (ground)
        else:
            Va_Terminal = app.vBus  # Top transistor's voltage (bus voltage)
    else:
        Va_Terminal = pwm_signals_top[0] * app.vBus

    # Phase B
    if pwm_signals_top[1] == 0 and pwm_signals_bottom[1] == 0:
        if Ib > 0:
            Vb_Terminal = 0  # Bottom transistor's voltage (ground)
        else:
            Vb_Terminal = app.vBus  # Top transistor's voltage (bus voltage)
    else:
        Vb_Terminal = pwm_signals_top[1] * app.vBus

    # Phase C
    if pwm_signals_top[2] == 0 and pwm_signals_bottom[2] == 0:
        if Ic > 0:
            Vc_Terminal = 0  # Bottom transistor's voltage (ground)
        else:
            Vc_Terminal = app.vBus  # Top transistor's voltage (bus voltage)
    else:
        Vc_Terminal = pwm_signals_top[2] * app.vBus

    return Va_Terminal, Vb_Terminal, Vc_Terminal

def estimate_BW():
    # Transfer function: G(s) = 1 / (L * s + r)
    num_d = [1]
    den_d = [motor.Ld, motor.Rs]
    num_q = [1]
    den_q = [motor.Lq, motor.Rs]
    num_pi_d = [control.Kp_d, control.Ki_d]
    den_pi_d = [1, 0]
    num_pi_q = [control.Kp_q, control.Ki_q]
    den_pi_q = [1, 0]    

    # Create transfer function
    G_d = ctrl.TransferFunction(num_d, den_d)
    G_q = ctrl.TransferFunction(num_q, den_q)

    # Plot Bode plot
    mag, phase, omega = ctrl.bode(G_d, dB=True, Hz=False, omega_limits=(1e-1, 1e3), plot=True, label = 'D axis')
    mag, phase, omega = ctrl.bode(G_q, dB=True, Hz=False, omega_limits=(1e-1, 1e3), plot=True, label = 'Q axis')

    # Show the plot
    plt.show()    

# Lists for plotting:
speed_list = []
iqd_ramped_list = []
iqd_sensed_list = []
error_list = []
Vqd_list = []
Vabc_list = []
pwm_list = []
V_terminal = []
bemf = []
currents = []    
torque_list = []
angle_list = []

def simulate_motor(motor, sim, app, control):
    # Initializations
    speed_m = app.init_speed
    speed_e = speed_m * motor.pole_pairs
    angle_m = 0
    angle_e = 0
    iq_ramped = 0
    id_ramped = 0
    Vq = 0
    Vd = 0
    Ia, Ib, Ic = 0, 0, 0

    for t in sim.time_points:
        # Ramp handling
        # Speed ramp
        if app.speed_control:
            if (speed_m < app.commanded_speed) and (app.speed_ramp_rate != 0):            
                speed_m += app.speed_ramp_rate * sim.time_step
            else:
                speed_m = app.commanded_speed
        else:
            speed_m += ((torque - speed_m * motor.visc_fric_coeff) / motor.inertia) * sim.time_step
        speed_e = speed_m * motor.pole_pairs
        speed_list.append([speed_m, speed_e])

        # Current ramps
        if (abs(iq_ramped) < abs(app.commanded_iq)) and (app.current_ramp_rate != 0):
            iq_ramped += app.current_ramp_rate * sim.time_step * np.sign(app.commanded_iq)
        else:
            iq_ramped = app.commanded_iq

        if (abs(id_ramped) < abs(app.commanded_id)) and (app.current_ramp_rate != 0):
            id_ramped += app.current_ramp_rate * sim.time_step * np.sign(app.commanded_id)
        else:
            id_ramped = app.commanded_id        
        iqd_ramped_list.append([iq_ramped, id_ramped])

        # Convert abc frame currents to qd currents
        Iq_sensed, Id_sensed = dq_transform(Ia, Ib, Ic, angle_e)
        iqd_sensed_list.append([Iq_sensed, Id_sensed])
        
        # Errors
        error_iq = iq_ramped - Iq_sensed
        error_id = id_ramped - Id_sensed    
        error_list.append([error_iq, error_id])
        
        # Calculate DQ voltage commands
        Vq, Vd = control.pi_control(error_iq, error_id, t, Vq, Vd, app.vBus)
        Vqd_list.append([Vq, Vd])
        
        # Convert Vdq voltages to abc frame
        Va, Vb, Vc = inverse_dq_transform(Vq, Vd, angle_e)
        Vabc_list.append([Va, Vb, Vc])

        # Calculate transistor values including dead time        
        # Activate short circuiting at half the sim time if applicable
        if (app.short_circuit == False) or ((app.short_circuit == True) and (t < (sim.total_time / 2))):
            pwm_signals_top, pwm_signals_bottom = center_aligned_pwm_with_deadtime(Va, Vb, Vc, app.vBus, t, (1/control.sampling_time), control.deadTime)        
        else:
            pwm_signals_top = [0, 0, 0]
            pwm_signals_bottom = [1, 1, 1]                    

        pwm_list.append([pwm_signals_top, pwm_signals_bottom])

        # Calculate terminal voltages including dead time
        Va_Terminal, Vb_Terminal, Vc_Terminal = terminal_voltage_with_deadtime(Ia, Ib, Ic, pwm_signals_top, pwm_signals_bottom, motor)
        V_terminal.append([Va_Terminal, Vb_Terminal, Vc_Terminal])

        # Update Lq, Ld
        motor.inductance_dq(Iq_sensed, Id_sensed)
        
        # Update self and mutual phase inductances
        motor.inductance_abc(angle_e)

        # Update self and mutual phase inductances time derivatives
        motor.inductance_abc_dot(angle_e, speed_e)

        # Calculate the phases bemf
        motor.bemf_a = speed_m * motor.phase_bemf(angle_e, 0)
        motor.bemf_b = speed_m * motor.phase_bemf(angle_e, -2 * np.pi / 3)
        motor.bemf_c = speed_m * motor.phase_bemf(angle_e, 2 * np.pi / 3)
        bemf.append([motor.bemf_a, motor.bemf_b, motor.bemf_c])

        # Solve the ODE for phase currents over one time step
        sol = solve_ivp(phase_current_ode, [t, t + sim.time_step], [Ia, Ib, Ic],
                        args=(Va_Terminal, Vb_Terminal, Vc_Terminal, motor), method='RK45')    

        Ia, Ib, Ic = sol.y[:, -1]
        currents.append([Ia, Ib, Ic])        

        torque = motor.torque(Iq_sensed, Id_sensed)
        torque_list.append(torque)

        angle_m += speed_m * sim.time_step
        angle_e += speed_e * sim.time_step
        angle_list.append([angle_m, angle_e])

# Instantiate objects
motor = Motor()
sim = Simulation()
app = Application()
control = MotorControl()

# Plot system bode plots
estimate_BW()

# Run the simulation
simulate_motor(motor, sim, app, control)

# Plot results
time_points = sim.time_points
speed_list = np.array(speed_list)
iqd_ramped_list = np.array(iqd_ramped_list)
iqd_sensed_list = np.array(iqd_sensed_list)
error_list = np.array(error_list)
Vqd_list = np.array(Vqd_list)
Vabc_list = np.array(Vabc_list)
pwm_list = np.array(pwm_list)
V_terminal = np.array(V_terminal)
bemf = np.array(bemf)
currents = np.array(currents)
torque = np.array(torque_list)
angle_list = np.array(angle_list)

plt.figure(figsize=(10, 8))

# plt.subplot(4, 1, 1)
# plt.plot(time_points, iqd_sensed_list[:, 0], label='iqSensed')
# plt.plot(time_points, iqd_sensed_list[:, 1], label='idSensed')
# plt.plot(time_points, iqd_ramped_list[:, 0], label='iqCmd')
# plt.plot(time_points, iqd_ramped_list[:, 1], label='idCmd')
# plt.title('Iq, Id Cmd + Sensed')
# plt.legend()

plt.subplot(4, 1, 1)
plt.plot(time_points, currents[:, 0], label='ia')
plt.plot(time_points, currents[:, 1], label='ib')
plt.plot(time_points, currents[:, 2], label='ic')
plt.title('Phase currents')
plt.legend()

plt.subplot(4, 1, 2)
plt.plot(time_points, Vabc_list[:, 0], label='Va')
plt.plot(time_points, Vabc_list[:, 1], label='Vb')
plt.plot(time_points, Vabc_list[:, 2], label='Vc')
plt.title('Phase Voltages')
plt.legend()

plt.subplot(4, 1, 3)
plt.plot(time_points, Vqd_list[:, 0], label='Vq')
plt.plot(time_points, Vqd_list[:, 1], label='Vd')
plt.title('Vq, Vd')
plt.legend()

plt.subplot(4, 1, 4)
plt.plot(time_points, bemf[:, 0], label='bemf_a')
plt.plot(time_points, bemf[:, 1], label='bemf_b')
plt.plot(time_points, bemf[:, 2], label='bemf_c')
plt.title('Back-emf')
plt.legend()

plt.tight_layout()
plt.show()


'''
TODO:
Verify that injecting Id lowers voltage amplitude.
Add short circuit test option
Calculate bode of PI and system to know BW.
'''

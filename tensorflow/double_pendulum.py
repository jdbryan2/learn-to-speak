from numpy import sin, cos
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import matplotlib.animation as animation
import os

SHOW_ANIMATION = False
SHOW_TRACE = False
steps = 10000
batches = 10 
save_dir = './pendulum_val/'

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

class Pendulum(object):
    def __init__(self, step_size=0.01, period=5, L1=1.0, L2=1.0, speed_limit=np.radians(360.), initial_state=np.zeros(4)):
        self.L1 = L1 # length of pendulum 1 in m
        self.L2 = L2 # length of pendulum 2 in m
        self.step_size = step_size
        self.period = period

        self.speed_limit = speed_limit

        self.state = initial_state # theta1, omega1, theta2, omega2
        #print self.state

    def simulate_period(self, action):
        for k in range(self.period):
            self.update_state(action)

        return self.state

    def xy_out(self):

        x1 = self.L1*sin(self.state[0])
        y1 = -self.L1*cos(self.state[0])

        x2 = self.L2*sin(self.state[2]) + x1
        y2 = -self.L2*cos(self.state[2]) + y1

        return x1, y1, x2, y2

    def observed_out(self):
        x1 = self.L1*sin(self.state[0])
        y1 = -self.L1*cos(self.state[0])

        x2 = self.L2*sin(self.state[2]) + x1
        y2 = -self.L2*cos(self.state[2]) + y1
        return np.array([x2, y2])


    def limit_speed(self, action):
        action[action>np.abs(self.speed_limit)] = np.abs(self.speed_limit)
        action[action< -1.*np.abs(self.speed_limit)] = -1.*np.abs(self.speed_limit)
        return action
        

    def update_state(self, _action):
        action = self.limit_speed(_action)

        self.state[1] = action[0]
        self.state[3] = action[1]
        self.state[0] += self.step_size*self.state[1]
        self.state[2] += self.step_size*self.state[3]




# initial state
initial_state = np.radians(np.random.random(4)*360)
#steps = 1000
step_size = 0.01
if __name__ == '__main__':
# integrate your ODE using scipy.integrate.
    pendulum = Pendulum(initial_state=initial_state, step_size=step_size, period=5)

    for b in range(batches):
        state = np.zeros((steps, 4))
        observation = np.zeros((steps, 2))
        if SHOW_ANIMATION:
            x1 = np.zeros(steps)
            y1 = np.zeros(steps)
            x2 = np.zeros(steps)
            y2 = np.zeros(steps)

        action = np.radians(100*(np.random.random((steps, 2))-0.5))
        action[0, :] = 0. # initial action is set to zero

        for k in range(state.shape[0]):
            state[k] = pendulum.simulate_period(action[k,:])
            observation[k, :] = pendulum.observed_out()
            if SHOW_ANIMATION:
                x1[k], y1[k], x2[k], y2[k] = pendulum.xy_out()

        # save data to numpy file
        print "Saving batch ", b
        np.savez(save_dir+'batch_'+str(b), action = action, state=state, observation=observation)


        # show trace
        if SHOW_TRACE:
            plt.scatter(observation[:, 0], observation[:, 1])
            plt.show()


        # show animation
        if SHOW_ANIMATION:
            fig = plt.figure()
            ax = fig.add_subplot(111, autoscale_on=False, xlim=(-2, 2), ylim=(-2, 2))
            ax.grid()

            line, = ax.plot([], [], 'o-', lw=2)
            time_template = 'time = %.1fs'
            time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)


            def init():
                line.set_data([], [])
                time_text.set_text('')
                return line, time_text


            def animate(i):
                thisx = [0, x1[i], x2[i]]
                thisy = [0, y1[i], y2[i]]

                line.set_data(thisx, thisy)
                time_text.set_text(time_template % (i*step_size*pendulum.period))
                return line, time_text

            ani = animation.FuncAnimation(fig, animate, np.arange(1, len(state)),
                                          interval=25, blit=True, init_func=init)

# ani.save('double_pendulum.mp4', fps=15)
            plt.show()

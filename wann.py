import gym
class CartPoleSwingUpEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 50
    }

    def __init__(self):
        self.g = 9.82  #gravity
        self.m_c = 0.5 #cart mass
        self.m_p = 0.5  #pendulum mass
        self.total_m = (self.m_p + self.m_c)
        self.l = 0.6 #pole's length
        self.m_p_l = (self.m_p*self.l)
        self.force_mag = 10.0
        self.dt = 0.01  #seconds between state updates
        self.b = 0.1  #friction coefficient
        self.t = 0 #timestep
        self.t_limit = 3000
        self.hard_mode = False        
        self.theta_threshold_radians = 12 * 2 * math.pi / 360  #Angle at which to fail the episode
        self.x_threshold = 2.4
        self.action_space = 1
        self.observation_space = 5
        self.state = None

    def step(self, action):
        if (action < -1.0): action = -1.0
        if (action > 1.0): action = 1.0
        action *= self.force_mag
        state = self.state
        x = state[0]
        x_dot = state[1]
        theta = state[2]
        theta_dot = state[3]
        s = math.sin(theta)
        c = math.cos(theta)
        xdot_update = (-2*self.m_p_l*(theta_dot*theta_dot)*s + 3*self.m_p*self.g*s*c + 4*action - 4*self.b*x_dot)/(4*self.total_m - 3*self.m_p*c*c)
        thetadot_update = (-3*self.m_p_l*(theta_dot*theta_dot)*s*c + 6*self.total_m*self.g*s + 6*(action - self.b*x_dot)*c)/(4*self.l*self.total_m - 3*self.m_p_l*c*c)
        x = x + x_dot*self.dt
        theta = theta + theta_dot*self.dt
        x_dot = x_dot + xdot_update*self.dt
        theta_dot = theta_dot + thetadot_update*self.dt
        self.state = [x,x_dot,theta,theta_dot]
        done = False
        if ((x < -self.x_threshold) or (x > self.x_threshold)): done = True
        self.t += 1
        if (self.t >= self.t_limit): done = True
        reward_theta = (math.cos(theta)+1.0)/2.0
        reward_x = math.cos((x/self.x_threshold)*(math.pi/2.0))
        reward = reward_theta*reward_x
        obs = [x,x_dot,math.cos(theta),math.sin(theta),theta_dot]
        return [obs, reward, done]

    def reset(self):
        stdev = 0.1
        x = self.randn(0.0, stdev)
        x_dot = self.randn(0.0, stdev)
        theta = self.randn(math.pi, stdev)
        theta_dot = self.randn(0.0, stdev)
        x = self.randf(-1.0, 1.0)*self.x_threshold*0.75
        if (self.hard_mode):
            x = self.randf(-1.0, 1.0)*self.x_threshold*1.0
            x_dot = self.randf(-1.0, 1.0)*10.0*1.0
            theta = self.randf(-1.0, 1.0)*math.pi/2.0+math.pi
            theta_dot = self.randf(-1.0, 1.0)*10.0*1.0
        self.state = [x, x_dot, theta, theta_dot]
        self.t = 0
        obs = [x,x_dot,math.cos(theta),math.sin(theta),theta_dot]
        return obs

    def render(self, mode='human'):
        return

        screen_width = 800
        world_width = 5  #max visible position of cart
        scale = screen_width/world_width
        carty = screen_width/8 #TOP OF CART (assume screen_width == screen_height * 4)
        polewidth = 6.0*screen_width/600
        polelen = scale*self.l  #0.6 or self.l
        cartwidth = 40.0*screen_width/600
        cartheight = 20.0*screen_width/600
        state = self.state
        x = state[0]
        x_dot = state[1]
        theta = state[2]
        theta_dot = state[3]
        cartx = x*scale+screen_width/2.0 #MIDDLE OF CART
        self.p.stroke(0)
        self.p.strokeWeight(0.5)
        #track
        self.p.line(screen_width/2 - self.x_threshold*scale, carty + cartheight/2 + cartheight/4 + 1, screen_width/2 + self.x_threshold*scale, carty + cartheight/2 + cartheight/4 + 1)
        l=-cartwidth/2
        r=cartwidth/2
        t=cartheight/2
        b=-cartheight/2
        #cart
        self.p.fill(255, 64, 64)
        self.p.push()
        self.p.translate(cartx, carty)
        self.polygon(self.p, [[l,b], [l,t], [r,t], [r,b]])
        self.p.pop()
        #L and R wheels
        self.p.fill(192)
        self.p.circle(cartx-cartwidth/2, carty+cartheight/2, cartheight/2)
        self.p.circle(cartx+cartwidth/2, carty+cartheight/2, cartheight/2)
        #pole
        l=-polewidth/2
        r=polewidth/2
        t=polelen-polewidth/2
        b=-polewidth/2
        self.p.fill(64, 64, 255)
        self.p.push()
        self.p.translate(cartx, carty)
        self.p.rotate(math.pi-theta)
        self.polygon(self.p, [[l,b], [l,t], [r,t], [r,b]])
        self.p.pop()
        #axle
        self.p.fill(48)
        self.p.circle(cartx, carty, polewidth) #note: diameter, not radius.

    def polygon(self, p, points):
        p.beginShape()
        N = points.length
        for i in range(N):
            x = points[i][0]
            y = points[i][1]
            p.vertex(x, y)
        p.endShape(p.CLOSE)

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def gaussRandom(self):
        if not hasattr(self, 'return_v'):
            self.return_v = False
            self.v_val = 0.0

        if(self.return_v):
            self.return_v = False
            return self.v_val

        u = 2*random.random()-1
        v = 2*random.random()-1
        r = u*u + v*v
        if(r == 0 or r > 1):
            return self.gaussRandom()
        c = math.sqrt(-2*math.log(r)/r)
        v_val = v*c  #cache this
        self.return_v = True
        return u*c

    def randf(self, a, b):
        return random.random()*(b-a)+a

    def randi(self, a, b):
        return math.floor(random.random()*(b-a)+a)

    def randn(self, mu, std):
        return mu+self.gaussRandom()*std

    def birandn(self, mu1, mu2, std1, std2, rho):
        z1 = randn(0, 1)
        z2 = randn(0, 1)
        x = math.sqrt(1-rho*rho)*std1*z1 + rho*std1*z2 + mu1
        y = std2*z2 + mu2
        return [x, y]

import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
class CartPoleEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 50
    }

    def __init__(self):
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = (self.masspole + self.masscart)
        self.length = 0.5 #actually half the pole's length
        self.polemass_length = (self.masspole * self.length)
        self.force_mag = 10.0
        self.tau = 0.02  #seconds between state updates
        self.kinematics_integrator = 'euler'

        #Angle at which to fail the episode
        self.theta_threshold_radians = 12 * 2 * math.pi / 360
        self.x_threshold = 2.4

        #Angle limit set to 2 * theta_threshold_radians so failing observation is still within bounds
        high = np.array([
            self.x_threshold * 2,
            np.finfo(np.float32).max,
            self.theta_threshold_radians * 2,
            np.finfo(np.float32).max])

        self.action_space = spaces.Box(-100.0, +100.0, shape=(1,), dtype=np.float32)  #TPJ: spaces.Discrete(2)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self.seed()
        self.viewer = None
        self.state = None

        self.steps_beyond_done = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        #TPJ：assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        state = self.state
        x, x_dot, theta, theta_dot = state
        force = self.force_mag if action==1 else -self.force_mag
        costheta = math.cos(theta)
        sintheta = math.sin(theta)
        temp = (force + self.polemass_length * theta_dot * theta_dot * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta* temp) / (self.length * (4.0/3.0 - self.masspole * costheta * costheta / self.total_mass))
        xacc  = temp - self.polemass_length * thetaacc * costheta / self.total_mass
        if self.kinematics_integrator == 'euler':
            x  = x + self.tau * x_dot
            x_dot = x_dot + self.tau * xacc
            theta = theta + self.tau * theta_dot
            theta_dot = theta_dot + self.tau * thetaacc
        else: #semi-implicit euler
            x_dot = x_dot + self.tau * xacc
            x  = x + self.tau * x_dot
            theta_dot = theta_dot + self.tau * thetaacc
            theta = theta + self.tau * theta_dot
        self.state = (x,x_dot,theta,theta_dot)
        done =  x < -self.x_threshold \
                or x > self.x_threshold \
                or theta < -self.theta_threshold_radians \
                or theta > self.theta_threshold_radians
        done = bool(done)

        if not done:
            reward = 1.0
        elif self.steps_beyond_done is None:
            #Pole just fell!
            self.steps_beyond_done = 0
            reward = 1.0
        else:
            if self.steps_beyond_done == 0:
                logger.warn("You are calling 'step()' even though this environment has already returned done = True. You should always call 'reset()' once you receive 'done = True' -- any further steps are undefined behavior.")
            self.steps_beyond_done += 1
            reward = 0.0

        return np.array(self.state), reward, done, {}

    def reset(self):
        self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(4,))
        self.steps_beyond_done = None
        return np.array(self.state)

    def render(self, mode='human'):
        screen_width = 600
        screen_height = 400

        world_width = self.x_threshold*2
        scale = screen_width/world_width
        carty = 100 #TOP OF CART
        polewidth = 10.0
        polelen = scale * (2 * self.length)
        cartwidth = 50.0
        cartheight = 30.0

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            l,r,t,b = -cartwidth/2, cartwidth/2, cartheight/2, -cartheight/2
            axleoffset =cartheight/4.0
            cart = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            self.carttrans = rendering.Transform()
            cart.add_attr(self.carttrans)
            self.viewer.add_geom(cart)
            l,r,t,b = -polewidth/2,polewidth/2,polelen-polewidth/2,-polewidth/2
            pole = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            pole.set_color(.8,.6,.4)
            self.poletrans = rendering.Transform(translation=(0, axleoffset))
            pole.add_attr(self.poletrans)
            pole.add_attr(self.carttrans)
            self.viewer.add_geom(pole)
            self.axle = rendering.make_circle(polewidth/2)
            self.axle.add_attr(self.poletrans)
            self.axle.add_attr(self.carttrans)
            self.axle.set_color(.5,.5,.8)
            self.viewer.add_geom(self.axle)
            self.track = rendering.Line((0,carty), (screen_width,carty))
            self.track.set_color(0,0,0)
            self.viewer.add_geom(self.track)

            self._pole_geom = pole

        if self.state is None: return None

        #Edit the pole polygon vertex
        pole = self._pole_geom
        l,r,t,b = -polewidth/2,polewidth/2,polelen-polewidth/2,-polewidth/2
        pole.v = [(l,b), (l,t), (r,t), (r,b)]

        x = self.state
        cartx = x[0]*scale+screen_width/2.0 #MIDDLE OF CART
        self.carttrans.set_translation(cartx, carty)
        self.poletrans.set_rotation(-x[2])

        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None


import torch
import math
import random
class DRL(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.001):
        super(DRL, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.mapping = torch.nn.Sequential(torch.nn.Linear(input_size, hidden_size), torch.nn.ReLU(), torch.nn.Linear(hidden_size, hidden_size), torch.nn.ReLU(), torch.nn.Linear(hidden_size, output_size))
        self.apply(self.__class__.weights_init)  #TPJ
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        self.criterion = torch.nn.MSELoss()
        self.steps = 0
        self.buffer = []
        self.epsi_low = 0.05
        self.epsi_high = 0.9
        self.gamma = 0.8
        self.decay = 200
        self.capacity = 10000
        self.batch_size = 64

    def weights_init(m):
        if m.__class__.__name__.find('Linear') != -1:
            torch.nn.init.normal_(m.weight.data, mean=0.0, std=0.1)
            torch.nn.init.constant_(m.bias.data, val=0.0)
        
    def explore(self, state):
        self.steps += 1
        if random.random() < self.epsi_low + (self.epsi_high-self.epsi_low) * (math.exp(-1.0 * self.steps/self.decay)):
            action = random.randrange(self.output_size)
        else:
            state = torch.tensor(state, dtype=torch.float).view(1,-1)
            action = torch.argmax(self.mapping(state)).item()
        return action

    def remember(self, *transition):
        if len( self.buffer)==self.capacity:
            self.buffer.pop(0)
        self.buffer.append(transition)
        
    def rethink(self):
        if len(self.buffer) >= self.batch_size:
            state_old, action_now, reward_now, state_new = zip(*random.sample(self.buffer, self.batch_size))
            state_old = torch.tensor(state_old, dtype=torch.float)
            action_now = torch.tensor(action_now, dtype=torch.long).view(self.batch_size, -1)
            reward_now = torch.tensor(reward_now, dtype=torch.float).view(self.batch_size, -1)
            state_new = torch.tensor(state_new, dtype=torch.float)
            y_true = reward_now + self.gamma * torch.max( self.mapping(state_new).detach(), dim=1)[0].view(self.batch_size, -1)
            y_pred = self.mapping.forward(state_old).gather(1, action_now)
            loss = self.criterion(y_pred, y_true)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

import numpy
class WAN(object):  #Weight Agnostic Neural
    def __init__(self, init_shared_weight):
        self.num_hidden = 10
        self.input_size = 5
        self.output_size = 1
        self.shape_in = [self.input_size, self.num_hidden]
        self.shape_out = [self.num_hidden, self.output_size]
        self.aVec = [1,1,1,1,1,1,7,7,5,1,5,5,4,1,7,3,9,1,3,7,9,5,4,3,9,7,1,7,1]
        self.wKey = [10,35,36,41,64,69,95,97,108,125,128,142,157,202,231,257,289,302,331,361,362,363,364,367,368,373,374,376,394,395,398,401,403,425,461,484,517,543,574,576,602,603,604,606,633,662,692,722,723,753,782,811]
        self.weights = [-0.1783,-0.0303,1.5435,1.8088,-0.857,1.024,-0.3872,0.2639,-1.138,-0.2857,0.3797,-0.199,1.3008,-1.4126,-1.3841,7.1232,-1.5903,-0.6301,0.8013,-1.1348,-0.7306,0.006,1.4754,1.1144,-1.5251,-1.277,1.0933,0.1666,-0.5483,2.6779,-1.2728,0.4593,-0.2608,0.1183,-2.1036,-0.3119,-1.0469,0.2662,0.7156,0.0328,0.3441,-0.1147,-0.0553,-0.4123,-3.2276,2.5201,1.7362,-2.9654,0.9641,-1.7355,-0.1573,2.9135]
        self.weight_bias = -1.5
        nNodes = len(self.aVec)
        self.wVec = [0] * (nNodes*nNodes)
        for i in range(nNodes*nNodes):
            self.wVec[i] = 0
        self.set_weight(init_shared_weight, 0)

    def set_weight(self, weight, weight_bias):
        nValues = len(self.wKey)
        if type(weight_bias).__name__ not in ['int','long','float']:
            weight_bias = 0
        if type(weight).__name__ == 'list':
            weights = weight
        else:
            weights = [weight] * nValues
        for i in range(nValues):
            k = self.wKey[i]
            self.wVec[k] = weights[i] + weight_bias

    def tune_weights(self):
        self.set_weight(self.weights, self.weight_bias)

    def get_action(self, old_state):
        nNodes = len(self.aVec)
        wMat = numpy.array(self.wVec).reshape((nNodes, nNodes))
        nodeAct = [0] * nNodes
        nodeAct[0] = 1
        for i in range(len(old_state)):
            nodeAct[i+1] = old_state[i]
        for iNode in range(self.input_size+1, nNodes):
            rawAct = numpy.dot(nodeAct, wMat[:, iNode:iNode+1])  #TPJ
            rawAct = self.applyActSimple(self.aVec[iNode], rawAct.tolist()[0])
            nodeAct[iNode] = rawAct
        return nodeAct[-self.output_size:][0]

    def applyActSimple(self, actId, x):
        if actId == 1:
            return x
        elif actId == 2:
            return 0.0 if x<=0.0 else 1.0  #unsigned step
        elif actId == 3:
            return math.sin(math.pi*x)
        elif actId == 4:  
            return math.exp(-(x*x)/2.0)  #gaussian with mean zero and unit variance 1
        elif actId == 5:
            return math.tanh(x)
        elif actId == 6:
            return (math.tanh(x/2.0) + 1.0)/2.0  #sigmoid
        elif actId == 7:
            return -x
        elif actId == 8:
            return math.abs(x)
        elif actId == 9:
            return max(x, 0)  #relu
        elif actId == 10:
            return math.cos(math.pi*x)
        else:
            print('unsupported actionvation type: ',actId)
            return None

def drl(environment):
    environment = CartPoleEnv()  #import gym environment = gym.make('CartPole-v0')
    drl = DRL(environment.observation_space.shape[0], 256, environment.action_space.shape[0])  #.n
    for epoch in range(1000):
        state_old = environment.reset()
        rewards = 1
        while True:
            environment.render()
            action_now = drl.explore(state_old)            
            state_new, reward_now, done, _ = environment.step(action_now)
            if done:
                reward_now = -1
            drl.remember(state_old, action_now, reward_now, state_new)
            if done:
                break
            rewards += reward_now
            state_old = state_new
            drl.rethink()   #TPJ: the chance to rethink is very import: not-done
        print('epoch=%04d'%(epoch),'  ','rewards=%d'%(rewards))

def tpj():
    #1.) Initialize：Create population of minimal networks.
    #2.) Evaluate：Test with range of shared weight values.
    #3.) Rank：Rank by performance and complexity 
    #4.) Vary：Create new population by varying best networks.
    #TODO

def wan():
    environment = CartPoleSwingUpEnv()
    drl = WAN(-1.5)
    for epoch in range(20):
        if epoch == 0:
            print('init_weights:')
        elif epoch == 10:
            print()
            print('tune_weights:')
            drl.tune_weights()

        state_old = environment.reset()
        rewards = 1
        for step in range(100000000):
            environment.render()
            action_now = drl.get_action(state_old)
            state_new, reward_now, done = environment.step(action_now)
            if done:
                reward_now = -1
                break
            rewards += reward_now
            state_old = state_new
        print('epoch=%04d'%(epoch),'  ','rewards=%d'%(rewards),'  ','step=%d'%(step))

if __name__ == '__main__':    
    #drl()
    #wan()
    tpj()

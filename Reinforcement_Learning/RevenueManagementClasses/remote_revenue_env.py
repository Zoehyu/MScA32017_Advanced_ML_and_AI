# Remote Revenue Management environment for both test & training modes
# 14.01.2019
# iLykei (c)

import socket
import struct
from datetime import  datetime

import numpy as np

import revenue_pb2


SERVER_ADDR = 'datastream.ilykei.com'
#SERVER_ADDR = '127.0.0.1'
SERVER_PORT = 30077
SERVER_TIME_LIMIT = 100  # RemoteRevenueEnv throws exception if server does not respond for SERVER_TIME_LIMIT seconds


class TcpClient:
    def __init__(self, addr, port, timeout):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.settimeout(timeout)
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.sock.connect((addr, port))
        self.readbuf_sz = 256*256
        self.buffer = memoryview(bytearray(self.readbuf_sz))

    def write_msg(self, bytes):
        self.sock.sendall(struct.pack("!H", len(bytes)) + bytes)

    def _receive_msg(self, msg_len):
        assert msg_len <= self.readbuf_sz
        rcvd = self.sock.recv_into(self.buffer, msg_len)
        while rcvd < msg_len:
            rcvd += self.sock.recv_into(self.buffer[rcvd:], msg_len - rcvd)
        return self.buffer[:msg_len]

    def read_msg(self):
        header_bytes = self._receive_msg(2)
        msg_len = struct.unpack("!H", header_bytes)[0]
        return self._receive_msg(msg_len)


class YieldFigure:
    def __init__(self, time_horizon, total_capacity, action_space, redraw_seconds=0.1):
        import matplotlib.pyplot as plt
        self.time_horizon = time_horizon
        self.total_capacity = total_capacity
        self.action_space = action_space
        self.fig = plt.figure()
        self.ax1 = self.fig.add_subplot(111)
        self.ax2 = self.ax1.twinx()
        self.prices_color = 'tab:green'
        self.clients_color = 'tab:blue'
        self.ax1.set_xlabel('time (subepisodes)')
        self.ax1.set_ylabel('price', color=self.prices_color)
        self.ax1.tick_params(axis='y', labelcolor=self.prices_color)
        self.ax1.set_xlim(0, self.time_horizon)
        self.ax1.set_xticks(range(self.time_horizon + 1))
        self.ax1.set_ylim(0, max(self.action_space))
        self.ax1.set_yticks(self.action_space)
        self.ax1.grid(axis='x', color='tab:gray', linestyle='-', linewidth=0.5)
        self.ax2.tick_params(axis='y', labelcolor=self.clients_color)
        self.ax2.set_ylabel('clients', color=self.clients_color)
        self.ax2.set_ylim(0, self.total_capacity)
        self.ax2.set_title(f'Revenue: $ {0:05d}, Capacity: {self.total_capacity:03d}')
        self.last_plottime = datetime.now()
        self.redraw_seconds = redraw_seconds

    def add_price(self, subepisode, price, force_plot=True):
        self.ax1.bar(subepisode, price, color=self.prices_color,
                     alpha=0.3, align='edge', width=1)  # width=0.95
        now = datetime.now()
        if force_plot or (now - self.last_plottime).total_seconds() > self.redraw_seconds:
            self.fig.canvas.draw()
            self.last_plottime = now

    def plot_clients(self, arrival_times, cum_clients, revenue, force_plot=False):
        self.ax2.plot(arrival_times[-2:], cum_clients[-2:], color=self.clients_color)
        self.ax2.set_title(f'Revenue: $ {revenue:05d}, Capacity: {self.total_capacity-cum_clients[-1]:03d}')
        #self.ax2.clear()
        #self.ax2.tick_params(axis='y', labelcolor=self.clients_color)
        #self.ax2.set_ylabel('clients', color=self.clients_color)
        #self.ax2.set_ylim(0, self.total_capacity)
        #self.ax2.plot(arrival_times, cum_clients, color=self.clients_color)
        now = datetime.now()
        if force_plot or (now - self.last_plottime).total_seconds() > self.redraw_seconds:
            self.fig.canvas.draw()
            self.last_plottime = now


class RemoteRevenueEnv:
    def __init__(self, episode_duration=100,
                 addr=SERVER_ADDR, port=SERVER_PORT,
                 timeout=SERVER_TIME_LIMIT, plotting=True, redraw_seconds=0.05):
        assert episode_duration > 0
        self.episode_duration = episode_duration
        self.plotting = plotting
        self.redraw_seconds = redraw_seconds
        self.fig = None
        self.info_msg = revenue_pb2.InfoMsg()
        self.tcp_client = TcpClient(addr, port, timeout)
        init_request = revenue_pb2.InitRequestMsg(id='student_test', episode_duration=self.episode_duration)
        self.tcp_client.write_msg(init_request.SerializeToString())
        init_reply = revenue_pb2.InitReplyMsg()
        init_reply.ParseFromString(self.tcp_client.read_msg())
        self.total_capacity = init_reply.total_capacity
        self.time_horizon = init_reply.n_subepisodes
        self.action_space = init_reply.prices
        self.capacity = None
        self.timeleft = None
        self.done = None
        self.revenue = None  # total score
        self.arrival_times = None
        self.cum_clients = None

    def reset(self):
        reset_msg = revenue_pb2.SetPriceMsg(reset_episode=True)
        self.tcp_client.write_msg(reset_msg.SerializeToString())
        # receive first InfoMsg (or multiple until initial InfoMsg rcvd)
        while True:
            self.info_msg.ParseFromString(self.tcp_client.read_msg())
            if self.info_msg.subepisode_end and self.info_msg.subepisodes_left == self.time_horizon:
                break
        self.capacity = self.info_msg.capacity
        self.timeleft = self.info_msg.subepisodes_left
        self.done = self.info_msg.episode_end
        self.revenue = self.info_msg.total_revenue
        self.arrival_times = [0]
        self.cum_clients = [0]
        if self.plotting:
            self.fig = YieldFigure(self.time_horizon, self.total_capacity, self.action_space, self.redraw_seconds)
        return self.get_obs()

    def get_obs(self):
        return self.capacity, self.timeleft

    def step(self, action):
        self.tcp_client.write_msg(revenue_pb2.SetPriceMsg(price=action).SerializeToString())
        subepisode_start = datetime.now()
        if self.plotting:
            self.fig.add_price(self.time_horizon - self.timeleft, action)
        else:
            print(f"Setting price = ${action} for subepisode # {self.time_horizon - self.timeleft+1}")
        reward = 0
        clients = 0
        while True:
            self.info_msg.ParseFromString(self.tcp_client.read_msg())
            self.capacity = self.info_msg.capacity
            self.timeleft = self.info_msg.subepisodes_left
            self.done = self.info_msg.episode_end
            self.revenue = self.info_msg.total_revenue
            reward += self.info_msg.latest_reward
            clients += self.info_msg.latest_clients
            msg_time = (datetime.now() - subepisode_start).total_seconds()
            msg_time = msg_time / self.episode_duration * self.time_horizon  # in fractions of subepisode
            msg_time = min(msg_time, 1)
            msg_time += (self.time_horizon - self.timeleft - 1)
            self.arrival_times.append(msg_time)
            self.cum_clients.append(self.total_capacity - self.capacity)
            if self.plotting:
                self.fig.plot_clients(self.arrival_times, self.cum_clients, self.revenue, force_plot=self.done)
            else:
                if self.info_msg.latest_clients > 0:
                    print(f"Sold {self.info_msg.latest_clients} tickets, reward is ${self.info_msg.latest_reward}")

            if self.info_msg.subepisode_end:
                info = {'revenue': self.revenue,
                        'last_action': action,
                        'new_clients': clients}
                if self.plotting:
                    pass
                else:
                    print(f"After subepisode # {self.time_horizon - self.timeleft} "
                          f"revenue = {self.revenue}, capacity = {self.capacity}")
                return self.get_obs(), reward, self.done, info


class RemoteRevenueTrainEnv:
    def __init__(self, addr=SERVER_ADDR, port=SERVER_PORT, timeout=SERVER_TIME_LIMIT, prune_overbook=True):
        self.prune_overbook = prune_overbook
        self.sinfo_msg = revenue_pb2.SimpleInfoMsg()
        self.tcp_client = TcpClient(addr, port, timeout)
        init_request = revenue_pb2.InitRequestMsg(id='student_train', episode_duration=0)
        self.tcp_client.write_msg(init_request.SerializeToString())
        init_reply = revenue_pb2.InitReplyMsg()
        init_reply.ParseFromString(self.tcp_client.read_msg())
        self.total_capacity = init_reply.total_capacity
        self.time_horizon = init_reply.n_subepisodes
        self.action_space = init_reply.prices
        self.idx_actions = {self.action_space[i]: i for i in range(len(self.action_space))}
        self.capacity = None
        self.timeleft = None
        self.done = None
        self.revenue = None  # total score

    def reset(self):
        self.capacity = self.total_capacity
        self.timeleft = self.time_horizon
        self.done = False
        self.revenue = 0
        return self.get_obs()

    def get_obs(self):
        return self.capacity, self.timeleft

    def step(self, action):
        assert not self.done and action in self.action_space
        self.sinfo_msg.ParseFromString(self.tcp_client.read_msg())
        new_clients = self.sinfo_msg.clients[self.idx_actions[action]]
        if self.prune_overbook:
            new_clients = min(new_clients, self.capacity)
        self.capacity -= new_clients
        self.timeleft -= 1
        if self.timeleft == 0 or (self.capacity == 0 and not self.prune_overbook):
            self.done = True
        reward = new_clients * action
        self.revenue += reward
        info = {'revenue': self.revenue,
                'last_action': action,
                'new_clients': new_clients}
        return self.get_obs(), reward, self.done, info


if __name__ == '__main__':
    from datetime import datetime
    # train mode
    train_env = RemoteRevenueTrainEnv()
    n_games = 1000
    t0 = datetime.now()
    for i in range(n_games):
        train_env.reset()
        while True:
            a = np.random.choice(train_env.action_space)
            obs, reward, done, info = train_env.step(a)
            if done:
                break
        print(f'episode # {i}, revenue: {info["revenue"]}, capacity left: {obs[0]}')
    t1 = datetime.now()
    print(t1-t0)
    # test mode
    env = RemoteRevenueEnv(episode_duration=10, plotting=False)
    n_games = 1
    t0 = datetime.now()
    for i in range(n_games):
        env.reset()
        while True:
            a = np.random.choice(env.action_space)
            obs, reward, done, info = env.step(a)
            #print(obs, reward, done, info)
            if done:
                break
        print(f'episode # {i}, revenue: {info["revenue"]}, capacity left: {obs[0]}')
    t1 = datetime.now()
    print(t1-t0)

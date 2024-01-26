import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from matplotlib import transforms
import scipy
import math
from SequenceGenerator import MultiSequenceGenerator

def pareto_forward_recurrence_times_seq_generator(alpha, beta, size):
    vals=[]
    a=1-1.0/alpha
    for i in range(size):
        y=np.random.random()
        x=None
        if (y<a):# reverse of formula
            x = beta*(alpha*a-1.0+y)/(a*alpha-1)
        else: # reverse of formula
           x = beta*np.power(alpha*(1-y), 1.0/(1-alpha))
        vals.append(x)
    return vals

class PoissonParetoBurstProcessSimulator():
    def set_alpha_list(self):
        self.alpha_list = [3.0-2.0*h for h in self.hurst_list]

    def set_beta_list(self):
        self.beta_list = [(alpha-1.0)*self.burst_duration_mean/alpha for alpha in self.alpha_list]

    def set_theta(self):
        self.theta_list = [1.0/lam for lam in self.lam_list]
    
    def __init__(self, total_time=2000, lam_list=None, hurst_list=None, burst_duration_mean=0.4, has_pre_burst=True, least_interval_length=None):
        self.lam_list = lam_list
        self.hurst_list = hurst_list
        self.burst_duration_mean = burst_duration_mean
        self.total_time = total_time
        self.has_pre_burst = has_pre_burst
        self.set_alpha_list()
        self.set_beta_list()
        self.set_theta()
        if least_interval_length == None:
            self.least_interval_length = min(self.beta_list)/4.0
        else:
            self.least_interval_length = least_interval_length

    def get_interval_for_burst_duration(self, start_time, finish_time): # finding indexes that this burst is in burst_list (active burst_list)
        start = int((start_time/self.least_interval_length)+1.0) if np.fmod(start_time, self.least_interval_length) else int(start_time/self.least_interval_length)
        end = int(finish_time/self.least_interval_length) if np.fmod(finish_time, self.least_interval_length) else int((finish_time/self.least_interval_length)-1.0)
        return start, end

    def single_simulate(self, index, plot_result):
        time_list = np.arange(0, self.total_time, self.least_interval_length)
        active_burst_list = np.zeros(len(time_list), dtype=float)

        exponential_generator = scipy.stats.expon(scale=self.theta_list[index])
        pareto_generator = scipy.stats.pareto(self.alpha_list[index])

        if self.has_pre_burst:
            initial_burst_number = scipy.stats.poisson.rvs(self.lam_list[index] * self.burst_duration_mean)
            initial_burst_duration_time = pareto_forward_recurrence_times_seq_generator(self.alpha_list[index], self.beta_list[index], initial_burst_number)
            for i in range(initial_burst_number):
                burst_list_start_index, burst_list_end_index = self.get_interval_for_burst_duration(start_time=0, finish_time=min(initial_burst_duration_time[i],self.total_time))
                active_burst_list[burst_list_start_index:burst_list_end_index+1]+=1.0

        start_time = 0
        while start_time <= self.total_time:
            duration_between_arrivals = exponential_generator.rvs(size=1)
            # burst start time is after duration_between_arrivals
            start_time += duration_between_arrivals #this is the moment that next burst is comming
            if start_time > self.total_time:
                break
            burst_duration_time = self.beta_list[index] * pareto_generator.rvs(size=1)
            burst_finish_time = start_time + burst_duration_time
            burst_list_start_index, burst_list_end_index = self.get_interval_for_burst_duration(start_time=start_time, finish_time=min(burst_finish_time,self.total_time))
            active_burst_list[burst_list_start_index:burst_list_end_index+1] += 1.0

        if plot_result:
            self.plot_results(time_list, active_burst_list, index)
        return time_list, active_burst_list

    def make_dataframe(self, time_intervals, burst_list):
        df = pd.DataFrame([[time_intervals[i], burst_list[i]] for i in range(len(burst_list))], columns = ['time', 'active-burst'])
        lower, upper = scipy.stats.t.interval(0.95, df=len(burst_list)-1, loc=np.mean(burst_list), scale=np.std(burst_list))
        print(f'lower-ci: {lower}, upper-ci: {upper}')
        df['active-burst-mean'] = np.mean(burst_list)
        df['active-burst-upper'] = upper
        df['active-burst-lower'] = lower
        return df

    def plot_results(self, time_intervals, burst_list, index):
        #plt.rcParams['figure.figsize'] = (15.0, 6.0)
        df = self.make_dataframe(time_intervals, burst_list)
        sns.lineplot(data=df, x="time", y='active-burst', label='active burst')
        sns.lineplot(data=df, x="time", y='active-burst-mean', label='active burst mean', linestyle='--')
        sns.lineplot(data=df, x="time", y='active-burst-upper', label='active burst upper ci, alpha: 0.95', linestyle=':')
        sns.lineplot(data=df, x="time", y='active-burst-lower', label='active burst lower ci, alpha: 0.95', linestyle=':')
        plt.fill_between(time_intervals, df['active-burst-upper'], df['active-burst-mean'], color="lightcyan")
        plt.fill_between(time_intervals, df['active-burst-lower'], df['active-burst-mean'], color="pink")
        plt.title(f'Poisson Pareto Burst Process. hurst: {self.hurst_list[index]}, alpha: {self.alpha_list[index]}, lam: {self.lam_list[index]}')
        plt.show()

    def plot_std_compare(self, stat_function, name):
        for index in range(len(self.alpha_list)):
            time, burst = self.result[index]
            std_per_time = [stat_function(burst[:i]) for i in range(1, 1+len(burst))]
            sns.lineplot(x=time, y=std_per_time, label=f'hurst: {self.hurst_list[index]}, alpha: {self.alpha_list[index]}, lam: {self.lam_list[index]}')
        plt.title(f'{name} of Poisson Pareto Burst Process. hurst: {self.hurst_list[index]}, alpha: {self.alpha_list[index]}, lam: {self.lam_list[index]}')
        plt.xlabel('time')
        plt.ylabel(f'active burst {name}')
        plt.show()
    
    def simulate(self, plot_result=True):
        self.result = [self.single_simulate(i, plot_result) for i in range(len(self.alpha_list))]
        if plot_result:
            self.plot_std_compare(np.var, 'var')
            self.plot_std_compare(np.mean, 'mean')
            self.plot_std_compare(np.std, 'std')
        return self.result
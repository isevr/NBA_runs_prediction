# Necessary libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)

class Preprocessing():
    def __init__(self, data):
        self.data = data

        # Feature selection
        self.factors = ['Quarter','ShotType','ShotDist','FoulType',
                'TimeoutTeam','EnterGame','LeaveGame', 'Shooter',
                'Rebounder', 'Blocker','Fouler',
                'Fouled','ReboundType','ViolationPlayer', 'ViolationType',
                'FreeThrowShooter','TurnoverPlayer','TurnoverType','TurnoverCause',
                'TurnoverCauser']

        self.fact_cols = [col + str((i // 20) % 10 + 1) for i, col in enumerate(self.factors * 10)]
        self.fact_cols.append('class')

    # Label encoding
    def encoders(self):
        for factor in self.factors:
            globals()[f'{factor}_le'] = LabelEncoder()
            self.data[factor] = globals()[f'{factor}_le'].fit_transform(self.data[factor])
        return self.data

    # Creating arrays of runs
    def home_runner(self):
        # global home_runs
        run = []
        self.home_runs = []
        for idx in self.data.index:
            if self.data.at[idx,'HomePlay'] is not np.nan:
                    if 'makes' in self.data.at[idx,'HomePlay']:
                        run.append(idx)
            elif self.data.at[idx,'AwayPlay'] is not np.nan:
                    if 'makes' in self.data.at[idx,'AwayPlay']:
                        run.clear()
            if len(run) == 4:
                self.home_runs.append(run.copy())
                run.clear()
        return self.home_runs
                    
    def away_runner(self):
        # global away_runs
        run = []
        self.away_runs = []
        for idx in self.data.index:
            if self.data.at[idx,'AwayPlay'] is not np.nan:
                    if 'makes' in self.data.at[idx,'AwayPlay']:
                        run.append(idx)
            elif self.data.at[idx,'HomePlay'] is not np.nan:
                    if 'makes' in self.data.at[idx,'HomePlay']:
                        run.clear()
            if len(run) == 4:
                self.away_runs.append(run.copy())
                run.clear()
        return self.away_runs

    def all_runner(self):
        self.all_runs = []
        self.all_runs.extend(self.home_runs)
        self.all_runs.extend(self.away_runs)
        return self.all_runs

    # Flattening runs
    def runs_iter(self):
        # global runs_df
        self.runs_df = pd.DataFrame()
        for run in self.all_runs:
            a = self.data.loc[run[0]-10:run[0]-1, self.factors].values.ravel()
            a = np.append(a,1)
            self.runs_df = pd.concat([self.runs_df,pd.DataFrame([a.copy()])])
        self.runs_df.columns = self.fact_cols
        return self.runs_df

    # Function to remove runs from original Dataframe
    def no_runs_preprocessing(self):
        # global no_runs_split
        
        # find the first index of a run
        r = [i[0] for i in self.all_runs]  

        # create a list of runs
        r_x = []
        for num in r:
            r_x.extend(range(num - 10, num + 1))

        # mask the df without runs
        self.no_runs_df = self.data[~self.data.index.isin(r_x)].reset_index(drop=True)

        # segment the df and keep those that are length of 10
        segment_size = 10
        segments = len(self.no_runs_df) // segment_size

        self.no_runs_split = np.array_split(self.no_runs_df, segments)

        self.no_runs_split = [x for x in self.no_runs_split if len(x) != 11]

        return self.no_runs_split

    # Flattening no runs
    def no_runs_optimized(self):
        # global no_runs_df
        self.no_runs_df = pd.DataFrame([np.append(segment.loc[:, self.factors].values.ravel(), int(0)) for segment in self.no_runs_split])
        self.no_runs_df.columns = self.fact_cols
        self.no_runs_df=self.no_runs_df.sample(frac=0.14)
        return self.no_runs_df

    # Preparing final Dataframe for training
    def final(self):
        self.final_df = pd.concat([self.runs_df,self.no_runs_df],ignore_index=True).dropna().astype(int)
        self.scaler = MinMaxScaler((0,255))
        self.values = pd.DataFrame(self.scaler.fit_transform(self.final_df.iloc[:,:-1]))
        # print(self.values)
        # print(type(self.values))
        self.labels = self.final_df.iloc[:,-1]
        self.values.to_csv('data.csv')
        self.labels.to_csv('labels.csv')

    # Run everything
    def preprocess(self):
        self.encoders()
        self.home_runner()
        self.away_runner()
        self.all_runner()
        self.runs_iter()
        self.no_runs_preprocessing()
        self.no_runs_optimized()
        self.final()

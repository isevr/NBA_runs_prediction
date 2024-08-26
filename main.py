from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
import pandas as pd
pd.options.mode.chained_assignment = None
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report
from collections import defaultdict

app = FastAPI()
templates = Jinja2Templates(directory="templates")

def sequence_mining(team, opponent, df):
    combined_df = df
    combined_df = combined_df.replace({str(team):'same', str(opponent):'other'}, regex=True)
    
    df = pd.DataFrame()
    encoders = []

    for column in combined_df.columns[:-1]:
        le = LabelEncoder()
        encoders.append(le)
        df[column] = le.fit_transform(combined_df[column])

    df = pd.concat([df,combined_df.iloc[:,-1]], axis=1)

    undersample_len = len(df[df['class'] == 1])

    undersample_df = df[df['class'] == 0 ].sample(n=undersample_len, random_state=43)
    df = pd.concat([df[df['class'] == 1], undersample_df])

    run_events = {}
    run_max_counts = []

    sequence_mining_html = ""
    for j, event in zip(range(12,112,11), range(10, 0, -1)):
        a = combined_df.iloc[:, -j:-1][combined_df['class'] == 1]

        # count occurrences of each row
        row_counts = defaultdict(int)
        for i in range(len(a)):
            row_tuple = tuple(a.iloc[i])
            row_counts[row_tuple] += 1

        # row with the maximum count
        max_count = 0
        mc_row = None
        for row, count in row_counts.items():
            if count > max_count:
                max_count = count
                mc_row = row

        # index of the row 
        mc_idx = a.apply(lambda row: tuple(row) == mc_row, axis=1).idxmax()

        sequence_mining_html += f"<p><strong>Event: {event}</strong></p>"
        sequence_mining_html += f"<p>Index: {mc_idx}, Max Count: {max_count}</p>"
        sequence_mining_html += f"<table class='table table-striped'>{combined_df.iloc[mc_idx, -j:-1].to_frame().dropna().T.to_html()}</table>"
    
    return sequence_mining_html

@app.get("/", response_class=HTMLResponse)
async def landing_page(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/analyze", response_class=HTMLResponse)
async def analyze_team(request: Request, team: str = Form(...)):
    # Load and preprocess data
    df = pd.read_csv('all_seasons.csv')

    def team_selection(input, df):
        if input in df.HomeTeam.unique():
            df = df[df.HomeTeam == input]
            return df
        else:
            return None

    new_df = team_selection(team, df)

    if new_df is None:
        return templates.TemplateResponse("error.html", {"request": request, "message": "Team not found."})

    # Your existing preprocessing code
    factors = ['ShotDist','TimeoutTeam','Substitution', 'Shooter',
               'Rebounder', 'Blocker','Fouler',
               'ReboundType','ViolationPlayer',
               'FreeThrowShooter','TurnoverPlayer']

    fact_cols = [col + str((i // 11) % 10 + 1) for i, col in enumerate(factors * 10)]
    fact_cols.append('class')

    new_df['ShotDist'] = new_df.ShotDist.apply(lambda x: 'close' if x <= 10 else '3pt' if x >= 22 else 'mid' if pd.notna(x) else x)
    
    new_df['TimeoutTeam'] = new_df.apply(
        lambda row: 'timeout_home' if pd.notna(row['TimeoutTeam']) and row['TimeoutTeam'] == row['HomeTeam'] 
        else 'timeout_away' if pd.notna(row['TimeoutTeam']) 
        else row['TimeoutTeam'], 
        axis=1
    )

    new_df['Shooter'] = new_df.apply(lambda row: 'shooter_home' if pd.notna(row['Shooter']) and pd.notna(row['HomePlay'])
                                         else 'shooter_away' if pd.notna(row['Shooter']) and pd.notna(row['AwayPlay'])
                                         else np.nan,
                                         axis=1)

    new_df['Rebounder'] = new_df.apply(lambda row: 'rebounder_home' if pd.notna(row['Rebounder']) and pd.notna(row['HomePlay'])
                                         else 'rebounder_away' if pd.notna(row['Rebounder']) and pd.notna(row['AwayPlay'])
                                         else np.nan,
                                         axis=1)

    new_df['Blocker'] = new_df.apply(lambda row: 'blocker_home' if pd.notna(row['Blocker']) and pd.notna(row['HomePlay'])
                                         else 'blocker_away' if pd.notna(row['Blocker']) and pd.notna(row['AwayPlay'])
                                         else np.nan,
                                         axis=1)

    new_df['Fouler'] = new_df.apply(lambda row: 'fouler_home' if pd.notna(row['Fouler']) and pd.notna(row['HomePlay'])
                                         else 'fouler_away' if pd.notna(row['Fouler']) and pd.notna(row['AwayPlay'])
                                         else np.nan,
                                         axis=1)

    new_df['ViolationPlayer'] = new_df.apply(lambda row: 'violator_home' if pd.notna(row['ViolationPlayer']) and pd.notna(row['HomePlay'])
                                         else 'violator_away' if pd.notna(row['ViolationPlayer']) and pd.notna(row['AwayPlay'])
                                         else np.nan,
                                         axis=1)

    new_df['FreeThrowShooter'] = new_df.apply(lambda row: 'ft_home' if pd.notna(row['FreeThrowShooter']) and pd.notna(row['HomePlay'])
                                         else 'ft_away' if pd.notna(row['FreeThrowShooter']) and pd.notna(row['AwayPlay'])
                                         else np.nan,
                                         axis=1)

    new_df['TurnoverPlayer'] = new_df.apply(lambda row: 'to_player_home' if pd.notna(row['TurnoverPlayer']) and pd.notna(row['HomePlay'])
                                         else 'to_player_away' if pd.notna(row['TurnoverPlayer']) and pd.notna(row['AwayPlay'])
                                         else np.nan,
                                         axis=1)

    new_df['Substitution'] = new_df.apply(lambda row: 'sub_home' if pd.notna(row['EnterGame']) and pd.notna(row['HomePlay'])
                                                  else 'sub_away' if pd.notna(row['EnterGame']) and pd.notna(row['AwayPlay'])
                                                  else np.nan,
                                                  axis=1)

    def home_runner(data):
        global home_runs
        run = []
        home_runs = []
        for idx in data.index:
            if data.at[idx,'HomePlay'] is not np.nan:
                    if 'makes' in data.at[idx,'HomePlay']:
                        run.append(idx)
            elif data.at[idx,'AwayPlay'] is not np.nan:
                    if 'makes' in data.at[idx,'AwayPlay']:
                        run.clear()
            if len(run) == 4:
                home_runs.append(run.copy())
                run.clear()
        return home_runs
                
    home_runner(new_df)

    def away_runner(data):
        global away_runs
        run = []
        away_runs = []
        for idx in data.index:
            if data.at[idx,'AwayPlay'] is not np.nan:
                    if 'makes' in data.at[idx,'AwayPlay']:
                        run.append(idx)
            elif data.at[idx,'HomePlay'] is not np.nan:
                    if 'makes' in data.at[idx,'HomePlay']:
                        run.clear()
            if len(run) == 4:
                away_runs.append(run.copy())
                run.clear()
        return away_runs

    away_runner(new_df)

    all_runs = []
    all_runs.extend(home_runs)
    all_runs.extend(away_runs)

    new_df = new_df[factors]

    def runs_iter(data, runs):
        global runs_df
        runs_df = pd.DataFrame()
        for run in runs:
            a = data.loc[run[0]-10:run[0]-1, factors].values.ravel()
            a = np.append(a,1)
            runs_df = pd.concat([runs_df,pd.DataFrame([a.copy()])])
        return runs_df

    runs_iter(new_df, home_runs)
    runs_df.columns = fact_cols

    def no_runs_preprocessing(data, runs):
        global no_runs_split

        # find the first index of a run
        r = [i[0] for i in runs]  

        # create a list of runs
        r_x = []
        for num in r:
            r_x.extend(range(num - 10, num + 1))

        # mask the df without runs
        no_runs_df = data[~data.index.isin(r_x)].reset_index(drop=True)

        # segment the df and keep those that are length of 10
        segment_size = 10
        segments = len(no_runs_df) // segment_size

        no_runs_split = np.array_split(no_runs_df, segments)

        no_runs_split = [x for x in no_runs_split if len(x) != 11]

        return no_runs_split

    def no_runs_optimized(data, factors, fact_cols):
        global no_runs_df
        no_runs_df = pd.DataFrame([np.append(segment.loc[:, factors].values.ravel(), int(0)) for segment in data])
        no_runs_df.columns = fact_cols
        return no_runs_df

    no_runs_optimized(no_runs_preprocessing(new_df, home_runs), factors, fact_cols)

    combined_df = pd.concat([runs_df,no_runs_df],ignore_index=True)
    combined_df.to_csv(str(team)+'_runs.csv', index=False)
    combined_df = pd.read_csv(str(team)+'_runs.csv')


    df = pd.DataFrame()
    encoders = []

    for column in combined_df.columns[:-1]:
        le = LabelEncoder()
        encoders.append(le)
        df[column] = le.fit_transform(combined_df[column])

    df = pd.concat([df,combined_df.iloc[:,-1]],axis=1)
    model = load_model('pretrained_model_le.keras')
    X = df.iloc[:,:-1].values.reshape(-1,11,10)
    preds = np.argmax(model.predict(X), axis=1)

    df['class'] = df['class'].fillna(0) # to work for now

    report_dict = classification_report(df.iloc[:,-1], preds, output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose()
    report_html = report_df.to_html(classes='table table-striped', header="true", table_id="report_table")

    sequence_mining_html = sequence_mining('home', 'away',df)

    # Render the template with both the classification report and the sequence mining results
    return templates.TemplateResponse("report.html", {
        "request": request,
        "report_html": report_html,
        "sequence_mining_html": sequence_mining_html
    })



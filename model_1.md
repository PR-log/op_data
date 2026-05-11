```
#pip install nba_api scikit-learn xgboost
```
#### 정규시즌 데이터 불러오기 + 학습
```
import pandas as pd
import numpy as np
import time
from nba_api.stats.endpoints import leaguedashteamstats, leaguedashplayerstats, leaguegamelog
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from scipy.stats import poisson
import random

import warnings
warnings.filterwarnings('ignore')

print("2025-26 정규시즌 포함 12년 통합 데이터 모델링 ")

# 25-26 정규시즌 데이터 포함
seasons = [
    '2014-15', '2015-16', '2016-17', '2017-18', '2018-19', 
    '2019-20', '2020-21', '2021-22', '2022-23', '2023-24', '2024-25', '2025-26'
]

team_abbrev_mapping = {
    'ATL': 'Atlanta Hawks', 'BOS': 'Boston Celtics', 'BKN': 'Brooklyn Nets', 'CHA': 'Charlotte Hornets',
    'CHI': 'Chicago Bulls', 'CLE': 'Cleveland Cavaliers', 'DAL': 'Dallas Mavericks', 'DEN': 'Denver Nuggets',
    'DET': 'Detroit Pistons', 'GSW': 'Golden State Warriors', 'HOU': 'Houston Rockets', 'IND': 'Indiana Pacers',
    'LAC': 'LA Clippers', 'LAL': 'Los Angeles Lakers', 'MEM': 'Memphis Grizzlies', 'MIA': 'Miami Heat',
    'MIL': 'Milwaukee Bucks', 'MIN': 'Minnesota Timberwolves', 'NOP': 'New Orleans Pelicans', 'NYK': 'New York Knicks',
    'OKC': 'Oklahoma City Thunder', 'ORL': 'Orlando Magic', 'PHI': 'Philadelphia 76ers', 'PHX': 'Phoenix Suns',
    'POR': 'Portland Trail Blazers', 'SAC': 'Sacramento Kings', 'SAS': 'San Antonio Spurs', 'TOR': 'Toronto Raptors',
    'UTA': 'Utah Jazz', 'WAS': 'Washington Wizards'
}

all_games_data = []

for season in seasons:
    # 2025-26은 현재 플레이오프가 진행 중이므로 정규시즌만 타겟팅하여 수집
    target_types = ['Regular Season', 'Playoffs'] if season != '2025-26' else ['Regular Season']
    print(f"[{season}] 데이터 수집 및 결장 보정 프로세싱 중...")
    
    try:
        # 1. Macro 데이터 수집
        team_stats = leaguedashteamstats.LeagueDashTeamStats(season=season, measure_type_detailed_defense='Four Factors')
        df_team = team_stats.get_data_frames()[0][['TEAM_NAME', 'EFG_PCT', 'TM_TOV_PCT', 'OREB_PCT', 'FTA_RATE']]
        time.sleep(1)
        
        player_stats = leaguedashplayerstats.LeagueDashPlayerStats(season=season, measure_type_detailed_defense='Advanced', per_mode_detailed='PerGame')
        df_players = player_stats.get_data_frames()[0]
        df_filtered = df_players[df_players['MIN'] > 25.0]
        
        ace_data = []
        for team_abbrev in df_filtered['TEAM_ABBREVIATION'].unique():
            team_players = df_filtered[df_filtered['TEAM_ABBREVIATION'] == team_abbrev]
            if not team_players.empty:
                ace_row = team_players.loc[team_players['USG_PCT'].idxmax()]
                ace_data.append({
                    'TEAM_NAME': team_abbrev_mapping.get(team_abbrev, team_abbrev),
                    'ACE_NAME': ace_row['PLAYER_NAME'],
                    'ACE_PIE': ace_row['PIE']
                })
        df_aces = pd.DataFrame(ace_data)
        df_master_stats = pd.merge(df_team, df_aces, on='TEAM_NAME', how='inner').drop_duplicates(subset=['TEAM_NAME'])
        time.sleep(1)

        # 2. Micro 데이터 및 출전 기록 수집
        season_team_frames = []
        season_player_frames = []
        for s_type in target_types:
            season_team_frames.append(leaguegamelog.LeagueGameLog(season=season, season_type_all_star=s_type, player_or_team_abbreviation='T').get_data_frames()[0])
            time.sleep(0.5)
            season_player_frames.append(leaguegamelog.LeagueGameLog(season=season, season_type_all_star=s_type, player_or_team_abbreviation='P').get_data_frames()[0])
            time.sleep(0.5)
            
        df_games = pd.concat(season_team_frames, ignore_index=True)
        df_player_games = pd.concat(season_player_frames, ignore_index=True)
        played_set = set(df_player_games['GAME_ID'] + "_" + df_player_games['PLAYER_NAME'])
        
        df_games['GAME_DATE'] = pd.to_datetime(df_games['GAME_DATE'])
        df_games = df_games.sort_values(by=['TEAM_ID', 'GAME_DATE']).reset_index(drop=True)
        
        # 이동 평균 지표 생성
        df_games['GAME_EFG_PCT'] = (df_games['FGM'] + 0.5 * df_games['FG3M']) / df_games['FGA']
        df_games['GAME_FTA_RATE'] = df_games['FTA'] / df_games['FGA']
        df_games['GAME_TOV_PCT'] = df_games['TOV'] / (df_games['FGA'] + 0.44 * df_games['FTA'] + df_games['TOV'])
        
        for col in ['GAME_EFG_PCT', 'GAME_FTA_RATE', 'GAME_TOV_PCT']:
            df_games[f'RECENT_{col}'] = df_games.groupby('TEAM_ID')[col].transform(lambda x: x.rolling(window=5, min_periods=1).mean().shift(1))
            
        # 3. 매치업 결합
        df_home_games = df_games[df_games['MATCHUP'].str.contains(' vs. ')].copy()
        for index, home_row in df_home_games.iterrows():
            away_abbrev = home_row['MATCHUP'][-3:]
            away_team = team_abbrev_mapping.get(away_abbrev)
            game_id = home_row['GAME_ID']
            
            if away_team:
                h_macro = df_master_stats[df_master_stats['TEAM_NAME'] == home_row['TEAM_NAME']]
                a_macro = df_master_stats[df_master_stats['TEAM_NAME'] == away_team]
                away_micro = df_games[(df_games['GAME_ID'] == game_id) & (df_games['TEAM_ID'] != home_row['TEAM_ID'])]
                
                if not h_macro.empty and not a_macro.empty and not away_micro.empty:
                    h_mac, a_mac, a_mic = h_macro.iloc[0], a_macro.iloc[0], away_micro.iloc[0]
                    if pd.isna(home_row['RECENT_GAME_EFG_PCT']) or pd.isna(a_mic['RECENT_GAME_EFG_PCT']): continue
                    
                    # 결장 보정
                    h_ace_in = f"{game_id}_{h_mac['ACE_NAME']}" in played_set
                    a_ace_in = f"{game_id}_{a_mac['ACE_NAME']}" in played_set
                    
                    all_games_data.append({
                        'EFG_PCT_DIFF': float(h_mac['EFG_PCT'] - a_mac['EFG_PCT']),
                        'TOV_PCT_DIFF': float(h_mac['TM_TOV_PCT'] - a_mac['TM_TOV_PCT']),
                        'OREB_PCT_DIFF': float(h_mac['OREB_PCT'] - a_mac['OREB_PCT']),
                        'FTA_RATE_DIFF': float(h_mac['FTA_RATE'] - a_mac['FTA_RATE']),
                        'ADJ_ACE_PIE_DIFF': float((h_mac['ACE_PIE'] if h_ace_in else 0) - (a_mac['ACE_PIE'] if a_ace_in else 0)),
                        'RECENT_EFG_DIFF': float(home_row['RECENT_GAME_EFG_PCT'] - a_mic['RECENT_GAME_EFG_PCT']),
                        'RECENT_FTA_DIFF': float(home_row['RECENT_GAME_FTA_RATE'] - a_mic['RECENT_GAME_FTA_RATE']),
                        'RECENT_TOV_DIFF': float(home_row['RECENT_GAME_TOV_PCT'] - a_mic['RECENT_GAME_TOV_PCT']),
                        'HOME_WIN': 1 if home_row['WL'] == 'W' else 0
                    })
                    
    except Exception as e:
        print(f"{season} 수집 중 에러 발생: {e}")

df_final_12y = pd.DataFrame(all_games_data)
print(f"\n 12년 통합 데이터 구축 완료. 총 {len(df_final_12y)}경기 확보.")

# 모델 학습 부분 (기존과 동일하되 데이터셋만 교체)
X = df_final_12y[['EFG_PCT_DIFF', 'TOV_PCT_DIFF', 'OREB_PCT_DIFF', 'FTA_RATE_DIFF', 'ADJ_ACE_PIE_DIFF', 'RECENT_EFG_DIFF', 'RECENT_FTA_DIFF', 'RECENT_TOV_DIFF']]
y = df_final_12y['HOME_WIN']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

final_model = StackingClassifier(
    estimators=[
        ('rf', RandomForestClassifier(n_estimators=150, max_depth=6, random_state=42)),
        ('xgb', XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.05, reg_lambda=1.0, random_state=42, eval_metric='logloss')),
        ('svm', SVC(kernel='linear', probability=True, C=0.05, random_state=42))
    ],
    final_estimator=LogisticRegression(C=0.1), cv=5
)
final_model.fit(X_train_scaled, y_train)
print(f"모델 정확도: {final_model.score(X_test_scaled, y_test)*100:.2f}%")
```
#### 내일 경기 결과 예측
```
print("05월 3일 보스턴 필라델피아 경기 예측---")

# 1. 대상 팀 설정
home_team_name = 'Boston Celtics'
away_team_name = 'Philadelphia 76ers'

# 2. 2025-26 시즌 Macro(기본 전력) 데이터 추출
# (이미 메모리에 있는 df_master_stats 사용)
h_mac = df_master_stats[df_master_stats['TEAM_NAME'] == home_team_name].iloc[0]
a_mac = df_master_stats[df_master_stats['TEAM_NAME'] == away_team_name].iloc[0]

# 3. Micro(최근 기세) 데이터 추출 
# (정규시즌 마지막 5경기의 평균 폼을 현재 기세로 가정)
h_mic = df_games[df_games['TEAM_NAME'] == home_team_name].iloc[-1]
a_mic = df_games[df_games['TEAM_NAME'] == away_team_name].iloc[-1]

# 4. 에이스 가용성 설정 (내일 경기이므로 양 팀 에이스 100% 출전 가정)
# 만약 부상 리포트가 있다면 여기서 0.0으로 수정하여 시뮬레이션 가능
h_ace_pie = h_mac['ACE_PIE'] # 제이슨 테이텀
a_ace_pie = a_mac['ACE_PIE'] # 조엘 엠비드

# 5. 모델 입력용 피처 구성 (8대 하이브리드 피처)
matchup_input = pd.DataFrame([{
    'EFG_PCT_DIFF': float(h_mac['EFG_PCT'] - a_mac['EFG_PCT']),
    'TOV_PCT_DIFF': float(h_mac['TM_TOV_PCT'] - a_mac['TM_TOV_PCT']),
    'OREB_PCT_DIFF': float(h_mac['OREB_PCT'] - a_mac['OREB_PCT']),
    'FTA_RATE_DIFF': float(h_mac['FTA_RATE'] - a_mac['FTA_RATE']),
    'ADJ_ACE_PIE_DIFF': float(h_ace_pie - a_ace_pie),
    'RECENT_EFG_DIFF': float(h_mic['RECENT_GAME_EFG_PCT'] - a_mic['RECENT_GAME_EFG_PCT']),
    'RECENT_FTA_DIFF': float(h_mic['RECENT_GAME_FTA_RATE'] - a_mic['RECENT_GAME_FTA_RATE']),
    'RECENT_TOV_DIFF': float(h_mic['RECENT_GAME_TOV_PCT'] - a_mic['RECENT_GAME_TOV_PCT'])
}])

# 6. 스케일링 및 승률 예측
matchup_scaled = scaler.transform(matchup_input)
win_prob = final_model.predict_proba(matchup_scaled)[0][1]

print(f"\n🔥 [내일 오전 경기 예측 결과]")
print(f"매치업: {home_team_name} (HOME) vs {away_team_name} (AWAY)")
print(f"-> 12년 통합 앙상블 모델 예측 승률: {home_team_name} [{win_prob*100:.1f}%]")
print(f"-> 12년 통합 앙상블 모델 예측 승률: {away_team_name} [{(1-win_prob)*100:.1f}%]")

# 7. 스코어 보드 시뮬레이션 (포아송 분포)
lambda_h = h_mic['PTS'] if 'PTS' in h_mic else 115 # 데이터 없을 시 평균값
lambda_a = a_mic['PTS'] if 'PTS' in a_mic else 112

# 단판 시뮬레이션
score_h = poisson.rvs(mu=lambda_h - 5) # 플레이오프 수비 보정
score_a = poisson.rvs(mu=lambda_a - 5)
if win_prob > 0.5:
    while score_h <= score_a: score_h += random.randint(1, 3)
else:
    while score_a <= score_h: score_a += random.randint(1, 3)

print("-" * 50)
print(f"가상 스코어 보드: {home_team_name} [{score_h}] : [{score_a}] {away_team_name}")
print(f"예상 승리팀: {home_team_name if score_h > score_a else away_team_name}")
```
#### 포아송 회귀로 스코어보드 조정
```
print("---포아송 회귀 기반 플레이오프 스코어 보드 조정 ---")

# 1. 기초 데이터 확보 (2025-26 정규시즌 평균 PTS)
# h_mic, a_mic에 PTS 정보가 포함되어 있다고 가정
pts_h = h_mic['PTS'] if 'PTS' in h_mic else 118.5
pts_a = a_mic['PTS'] if 'PTS' in a_mic else 114.2

# 2. 플레이오프 보정 계수 (Defensive Adjustment)
# 통계적으로 플레이오프는 정규시즌보다 득점이 약 5~8% 감소하는 경향이 있음
defensive_multiplier = 0.93 # 7% 득점 감소 가정

# 보정된 기대 득점(Lambda) 산출
adjusted_lambda_h = pts_h * defensive_multiplier
adjusted_lambda_a = pts_a * defensive_multiplier

print(f"-> 보정 전 기대 득점: BOS {pts_h:.1f} / PHI {pts_a:.1f}")
print(f"-> 보정 후 기대 득점(λ): BOS {adjusted_lambda_h:.1f} / PHI {adjusted_lambda_a:.1f}\n")

# 3. 승률 기반 점수 시뮬레이션 (win_prob 75.5% 반영)
# 단순히 랜덤이 아니라, 모델이 예측한 승률의 방향성을 유지하며 점수를 생성

def simulate_refined_score(prob_h, lambda_h, lambda_a):
    while True:
        # 포아송 분포로 개별 득점 생성
        score_h = poisson.rvs(mu=lambda_h)
        score_a = poisson.rvs(mu=lambda_a)
        
        # 무승부 방지 (농구 룰 적용)
        if score_h == score_a:
            continue
            
        # 모델 예측 승률(prob_h)에 따라 승패가 결정될 때까지 반복 추출
        # (단순 랜덤 추출 시 모델의 예측 확률과 어긋날 수 있으므로 가중치 부여)
        is_home_win = score_h > score_a
        prediction_match = (prob_h > 0.5 and is_home_win) or (prob_h <= 0.5 and not is_home_win)
        
        # 모델의 예측 방향과 일치하는 현실적인 점수대가 나올 때 반환
        if prediction_match:
            return score_h, score_a

# 최종 점수 생성
final_score_h, final_score_a = simulate_refined_score(win_prob, adjusted_lambda_h, adjusted_lambda_a)

print("예상 스코어 보드 ")
print("-" * 50)
print(f"HOME: Boston Celtics       [{final_score_h}]")
print(f"AWAY: Philadelphia 76ers   [{final_score_a}]")
print("-" * 50)
print(f"최종 예측: {home_team_name if final_score_h > final_score_a else away_team_name} 승리")
print(f"분석: 모델 승률(75.5%)과 플레이오프 수비 보정 계수({defensive_multiplier})가 모두 적용된 수치입니다.")
```

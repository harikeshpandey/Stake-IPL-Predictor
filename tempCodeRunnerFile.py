import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier

df = pd.read_csv('deliveries.csv')

# Step 1: Mark sixes in first 2 overs
df['is_six'] = ((df['batsman_runs'] == 6) & (df['over'] < 2)).astype(int)

# Step 2: Compute total runs per team per match
team_runs = df.groupby(['match_id', 'batting_team'])['total_runs'].sum().reset_index()
team_runs.rename(columns={'total_runs': 'team_score'}, inplace=True)

# Step 3: Compute if team hit six in first 2 overs
six_summary = df.groupby(['match_id', 'batting_team'])['is_six'].max().reset_index()
six_summary.rename(columns={'is_six': 'six_in_first_2_overs'}, inplace=True)

# Step 4: Merge runs and six info
team_summary = pd.merge(team_runs, six_summary, on=['match_id', 'batting_team'])

# Step 5: Prepare opponent runs
opponent = team_summary.copy()
opponent.rename(columns={'batting_team': 'bowling_team', 'team_score': 'opponent_score', 'six_in_first_2_overs': 'opponent_six'}, inplace=True)

# Step 6: Merge to create full summary
match_summary = pd.merge(team_summary, opponent, on='match_id')
match_summary = match_summary[match_summary['batting_team'] != match_summary['bowling_team']]

# Step 7: Create 'team_lost' flag
match_summary['team_lost'] = (match_summary['team_score'] < match_summary['opponent_score']).astype(int)

# Step 8: Encode team names
le = LabelEncoder()
match_summary['batting_team_encoded'] = le.fit_transform(match_summary['batting_team'])
match_summary['bowling_team_encoded'] = le.transform(match_summary['bowling_team'])

# Step 9: Build Loss Prediction Model
X_loss = match_summary[['batting_team_encoded', 'bowling_team_encoded']]
y_loss = match_summary['team_lost']

X_train, X_test, y_train, y_test = train_test_split(X_loss, y_loss, test_size=0.3, random_state=42)

# Hyperparameter Tuning for Loss Prediction Model
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10]
}

loss_model = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(estimator=loss_model, param_grid=param_grid, cv=3, scoring='accuracy')
grid_search.fit(X_train, y_train)
best_loss_model = grid_search.best_estimator_

print(f"Best Parameters for Loss Prediction Model: {grid_search.best_params_}")
print(f"Loss Prediction Accuracy: {accuracy_score(y_test, best_loss_model.predict(X_test)) * 100:.2f}%")
print(classification_report(y_test, best_loss_model.predict(X_test)))

# Step 10: Build Six Prediction Model (Only on losing teams)
losers = match_summary[match_summary['team_lost'] == 1]
X_six = losers[['batting_team_encoded', 'bowling_team_encoded']]
y_six = losers['six_in_first_2_overs']

X_train_six, X_test_six, y_train_six, y_test_six = train_test_split(X_six, y_six, test_size=0.3, random_state=42)

# Using XGBoost for Six Prediction Model
six_model = XGBClassifier(random_state=42)
six_model.fit(X_train_six, y_train_six)
print(f"Six Prediction Accuracy (on losing teams): {accuracy_score(y_test_six, six_model.predict(X_test_six)) * 100:.2f}%")
print(classification_report(y_test_six, six_model.predict(X_test_six)))

# ================= PREDICTION FUNCTION =================
def predict_result(team1, team2):
    if team1 not in le.classes_ or team2 not in le.classes_:
        print("Error: One or both team names not in training data.")
        return

    team1_encoded = le.transform([team1])[0]
    team2_encoded = le.transform([team2])[0]

    # Predict team1 losing
    team1_loses_prob = best_loss_model.predict_proba([[team1_encoded, team2_encoded]])[0][1]
    # Predict team2 losing
    team2_loses_prob = best_loss_model.predict_proba([[team2_encoded, team1_encoded]])[0][1]

    # Decide who is more likely to lose
    if team1_loses_prob > team2_loses_prob:
        losing_team = team1
        losing_encoded = team1_encoded
        opponent_encoded = team2_encoded
    else:
        losing_team = team2
        losing_encoded = team2_encoded
        opponent_encoded = team1_encoded

    print(f"\nPredicted Losing Team: {losing_team}")

    # Predict if losing team will hit a six early
    will_six = six_model.predict([[losing_encoded, opponent_encoded]])[0]
    if will_six:
        print(f"{losing_team} is likely to hit a SIX in the first 2 overs.")
    else:
        print(f"{losing_team} is NOT likely to hit a SIX in the first 2 overs.")

# ================== USER INPUT ==================
team1_input = input("Enter Team 1: ")
team2_input = input("Enter Team 2: ")

predict_result(team1_input, team2_input)
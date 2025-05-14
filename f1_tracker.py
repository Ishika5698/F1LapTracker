import eventlet
eventlet.monkey_patch()

import random
import time
import os
import shutil
import stat
import pickle
import pandas as pd
import numpy as np
from datetime import timedelta, datetime
from flask import Flask, render_template_string, request, redirect, url_for, send_from_directory, abort
from flask_socketio import SocketIO, emit
import aiohttp
import asyncio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from textblob import TextBlob
import json

# Flask and SocketIO setup
app = Flask(__name__, static_folder='/Users/tara/Desktop/F1LapTracker/static')
app.config['SECRET_KEY'] = 'f1-tracker-2025'
socketio = SocketIO(app, async_mode='eventlet', cors_allowed_origins="*", ping_timeout=60, ping_interval=25)
global_leaderboard = []
global_race = {}
live_updates = []
ai_predictions = []
strategy_recommendations = []
fan_comments = []

# OpenF1 API setup
BASE_URL = "https://api.openf1.org/v1"
SESSION_KEY = "latest"  # Replace with specific session_key for Saudi GP 2025
CACHE_FILE = "/Users/tara/Desktop/F1LapTracker/saudi_gp_cache.json"

# Drivers and teams (2025 Saudi Arabian GP)
drivers = [
    {"name": "Max Verstappen", "team": "Red Bull", "color": "\033[94m", "dry_skill": 1.2, "wet_skill": 1.0, "number": 1},
    {"name": "Lewis Hamilton", "team": "Ferrari", "color": "\033[31m", "dry_skill": 1.1, "wet_skill": 1.1, "number": 44},
    {"name": "Lando Norris", "team": "McLaren", "color": "\033[91m", "dry_skill": 1.0, "wet_skill": 1.0, "number": 4},
    {"name": "Charles Leclerc", "team": "Ferrari", "color": "\033[31m", "dry_skill": 1.0, "wet_skill": 1.2, "number": 16},
    {"name": "Carlos Sainz", "team": "Williams", "color": "\033[96m", "dry_skill": 0.9, "wet_skill": 0.9, "number": 55},
    {"name": "Oscar Piastri", "team": "McLaren", "color": "\033[91m", "dry_skill": 0.95, "wet_skill": 0.95, "number": 81},
    {"name": "Kimi Antonelli", "team": "Mercedes", "color": "\033[37m", "dry_skill": 0.85, "wet_skill": 0.9, "number": 27},
    {"name": "Oliver Bearman", "team": "Haas", "color": "\033[90m", "dry_skill": 0.8, "wet_skill": 0.85, "number": 38},
]

# Tire compounds
tires = [
    {"type": "Soft", "speed_bonus": -1.0, "pit_chance_increase": 0.1, "wear_rate": 0.2},
    {"type": "Medium", "speed_bonus": 0.0, "pit_chance_increase": 0.0, "wear_rate": 0.1},
    {"type": "Hard", "speed_bonus": 1.0, "pit_chance_increase": -0.1, "wear_rate": 0.05},
    {"type": "Wet", "speed_bonus": 2.0, "pit_chance_increase": 0.05, "wear_rate": 0.15}
]

# 2025 Saudi Arabian GP actual results (simplified for integration)
saudi_2025_results = [
    {"name": "Oscar Piastri", "team": "McLaren", "position": 1, "best_lap": 88.500, "tire": "Hard", "penalties": 0},
    {"name": "Max Verstappen", "team": "Red Bull", "position": 2, "best_lap": 88.267, "tire": "Hard", "penalties": 5},
    {"name": "Charles Leclerc", "team": "Ferrari", "position": 3, "best_lap": 88.749, "tire": "Hard", "penalties": 0},
    {"name": "Lando Norris", "team": "McLaren", "position": 4, "best_lap": 88.246, "tire": "Medium", "penalties": 0},
    {"name": "George Russell", "team": "Mercedes", "position": 5, "best_lap": 88.973, "tire": "Hard", "penalties": 0},
    {"name": "Lewis Hamilton", "team": "Ferrari", "position": 6, "best_lap": 89.371, "tire": "Hard", "penalties": 0},
    {"name": "Carlos Sainz", "team": "Williams", "position": 7, "best_lap": 88.942, "tire": "Hard", "penalties": 0},
    {"name": "Kimi Antonelli", "team": "Mercedes", "position": 8, "best_lap": 89.242, "tire": "Hard", "penalties": 0},
]

async def fetch_openf1_data(endpoint, params=None):
    async with aiohttp.ClientSession() as session:
        url = f"{BASE_URL}/{endpoint}"
        try:
            async with session.get(url, params=params) as response:
                if response.status == 429:
                    print("Rate limit hit, waiting...")
                    await asyncio.sleep(30)
                    return await fetch_openf1_data(endpoint, params)
                return await response.json()
        except Exception as e:
            print(f"Error fetching {url}: {e}")
            return []

def load_cache():
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r") as f:
            return json.load(f)
    return {}

def save_cache(data):
    with open(CACHE_FILE, "w") as f:
        json.dump(data, f)

async def get_real_lap_times():
    cache = load_cache()
    cache_time = cache.get("timestamp", "")
    if cache_time and (datetime.now() - datetime.fromisoformat(cache_time)) < timedelta(minutes=5):
        return cache.get("laps", [])

    laps = []
    for driver in drivers:
        data = await fetch_openf1_data("laps", {"session_key": SESSION_KEY, "driver_number": driver["number"]})
        if data:
            best_lap = min([float(lap["lap_duration"]) for lap in data if lap.get("lap_duration")], default=None)
            if best_lap:
                result = next((r for r in saudi_2025_results if r["name"] == driver["name"]), None)
                laps.append({
                    "driver": driver["name"],
                    "team": driver["team"],
                    "best_lap": best_lap,
                    "tire": result["tire"] if result else "Medium",
                    "penalties": result["penalties"] if result else 0,
                    "position": result["position"] if result else len(drivers)
                })
    cache = {"timestamp": datetime.now().isoformat(), "laps": laps}
    save_cache(cache)
    return laps

def format_time(seconds):
    if seconds is None:
        return "N/A"
    td = timedelta(seconds=seconds)
    minutes, sec = divmod(td.seconds, 60)
    millis = int(td.microseconds / 1000)
    return f"{minutes}:{sec:02d}.{millis:03d}"

def train_ai_model():
    model_path = "/Users/tara/Desktop/F1LapTracker/rf_model.pkl"
    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            return pickle.load(f), LabelEncoder().fit([t['type'] for t in tires])
    
    data = []
    for _ in range(100):
        race = {"sprint": False, "rainy": False}  # Simplified for training
        for driver in drivers:
            tire = random.choice(tires[:-1])
            lap_time = random.uniform(88, 90)  # Approximate real lap times
            data.append({
                'dry_skill': driver['dry_skill'],
                'wet_skill': driver['wet_skill'],
                'is_rainy': 0,
                'is_sprint': 0,
                'tire_type': tire['type'],
                'lap_time': lap_time
            })
    
    df = pd.DataFrame(data)
    le = LabelEncoder()
    df['tire_type'] = le.fit_transform(df['tire_type'])
    
    X = df[['dry_skill', 'wet_skill', 'is_rainy', 'is_sprint', 'tire_type']]
    y = df['lap_time']
    
    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(X, y)
    
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    os.chmod(model_path, 0o644)
    return model, le

def train_strategy_model():
    model_path = "/Users/tara/Desktop/F1LapTracker/strategy_model.pkl"
    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            return pickle.load(f), LabelEncoder().fit([t['type'] for t in tires])
    
    data = []
    actions = ['Stay', 'Pit for Soft', 'Pit for Medium', 'Pit for Hard']
    for _ in range(100):
        race = {"sprint": False, "rainy": False}
        for lap in range(1, 6):
            for driver in drivers:
                tire = random.choice(tires[:-1])
                position = random.randint(1, len(drivers))
                tire_wear = random.uniform(0, 1)
                recent_lap_time = random.uniform(88, 90)
                position_delta = random.randint(-2, 2)
                action = 'Pit for Medium' if tire_wear > 0.8 else 'Stay'
                data.append({
                    'is_rainy': 0,
                    'lap_number': lap,
                    'total_laps': 5,
                    'position': position,
                    'tire_type': tire['type'],
                    'tire_wear': tire_wear,
                    'recent_lap_time': recent_lap_time,
                    'position_delta': position_delta,
                    'action': action
                })
    
    df = pd.DataFrame(data)
    le = LabelEncoder()
    df['tire_type'] = le.fit_transform(df['tire_type'])
    X = df[['is_rainy', 'lap_number', 'total_laps', 'position', 'tire_type', 'tire_wear', 'recent_lap_time', 'position_delta']]
    y = df['action']
    
    model = DecisionTreeClassifier(max_depth=3, random_state=42)
    model.fit(X, y)
    
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    os.chmod(model_path, 0o644)
    return model, le

def predict_race_outcomes(race, model, label_encoder):
    global ai_predictions
    ai_predictions = []
    for driver in drivers:
        result = next((r for r in saudi_2025_results if r["name"] == driver["name"]), None)
        tire = result["tire"] if result else "Medium"
        features = pd.DataFrame([{
            'dry_skill': driver['dry_skill'],
            'wet_skill': driver['wet_skill'],
            'is_rainy': 0,
            'is_sprint': 0,
            'tire_type': label_encoder.transform([tire])[0]
        }])
        predicted_lap = model.predict(features)[0]
        ai_predictions.append({
            'driver': driver['name'],
            'team': driver['team'],
            'predicted_lap': predicted_lap,
            'tire': tire
        })
    ai_predictions.sort(key=lambda x: x['predicted_lap'])
    return ai_predictions

def recommend_strategy(race, lap, driver, tire, tire_wear, team_metrics, leaderboard, strategy_model, label_encoder):
    global strategy_recommendations
    position = next((i + 1 for i, d in enumerate(leaderboard) if d['name'] == driver['name']), len(drivers))
    recent_lap_time = team_metrics[driver['team']]['lap_times'][-1] if team_metrics[driver['team']]['lap_times'] else 88.5
    prev_position = team_metrics[driver['team']].get('last_position', position)
    position_delta = prev_position - position
    team_metrics[driver['team']]['last_position'] = position
    
    features = pd.DataFrame([{
        'is_rainy': 0,
        'lap_number': lap,
        'total_laps': 5,
        'position': position,
        'tire_type': label_encoder.transform([tire['type']])[0],
        'tire_wear': tire_wear,
        'recent_lap_time': recent_lap_time,
        'position_delta': position_delta
    }])
    
    action = strategy_model.predict(features)[0]
    priority = "Urgent" if tire_wear > 0.8 else "Normal"
    recommendation = {
        'driver': driver['name'],
        'team': driver['team'],
        'lap': lap,
        'action': action,
        'priority': priority,
        'message': f"AI Strategy ({priority}): {driver['name']} should {action.lower()} on Lap {lap}."
    }
    strategy_recommendations.append(recommendation)
    socketio.emit('strategy_update', recommendation)
    return recommendation

def process_fan_comment(comment, leaderboard, ai_predictions, race):
    global fan_comments
    blob = TextBlob(comment)
    sentiment = blob.sentiment.polarity
    sentiment_label = "positive" if sentiment > 0 else "negative" if sentiment < 0 else "neutral"
    
    keywords = [word.lower() for word, _ in blob.tags if word.lower() in [d['name'].lower() for d in drivers] + ['fast', 'slow', 'crash', 'win', 'tire', 'strategy']]
    response = ""
    if any(d['name'].lower() in keywords for d in drivers):
        driver_name = next((d['name'] for d in drivers if d['name'].lower() in keywords), None)
        driver_data = next((d for d in leaderboard if d['name'] == driver_name), None)
        pred = next((p for p in ai_predictions if p['driver'] == driver_name), None)
        if driver_data:
            if 'fast' in keywords or 'win' in keywords:
                response = f"{driver_name}'s P{driver_data['position']} finish with a {format_time(driver_data['best_lap'])} lap shows strong pace."
            elif 'slow' in keywords:
                response = f"{driver_name}'s {format_time(driver_data['best_lap'])} lap placed them P{driver_data['position']}."
            elif 'tire' in keywords or 'strategy' in keywords:
                rec = next((r for r in strategy_recommendations if r['driver'] == driver_name), None)
                response = f"{driver_name}'s tire: {driver_data['tire']}. Latest strategy: {rec['message'] if rec else 'Stay on current tires.'}"
            else:
                response = f"{driver_name} finished P{driver_data['position']} with a lap of {format_time(driver_data['best_lap'])}."
        else:
            response = f"No race data for {driver_name}. Predicted lap: {format_time(pred['predicted_lap'])}."
    else:
        response = f"{'Great enthusiasm!' if sentiment > 0 else 'Thanks for the feedback!'} Current leader: {leaderboard[0]['driver'] if leaderboard else 'TBD'}."
    
    comment_data = {
        'comment': comment,
        'sentiment': sentiment_label,
        'response': response,
        'timestamp': time.strftime("%H:%M:%S")
    }
    fan_comments.append(comment_data)
    socketio.emit('fan_comment', comment_data)
    return comment_data

async def run_race(race, laps=5):
    global global_leaderboard, global_race, live_updates, ai_predictions, strategy_recommendations, fan_comments
    live_updates = []
    strategy_recommendations = []
    fan_comments = []
    race_type = "Sprint" if race["sprint"] else "Grand Prix"
    
    model, label_encoder = train_ai_model()
    strategy_model, strategy_le = train_strategy_model()
    predict_race_outcomes(race, model, label_encoder)
    
    real_laps = await get_real_lap_times()
    leaderboard = []
    team_metrics = {driver['team']: {'lap_times': [], 'sectors': [[], [], []], 'incidents': 0, 'tire_wear': 0.0, 'last_position': len(drivers)} for driver in drivers}
    
    for driver in drivers:
        real_data = next((lap for lap in real_laps if lap["driver"] == driver["name"]), None)
        if real_data:
            lap_time = real_data["best_lap"]
            tire = next((t for t in tires if t["type"] == real_data["tire"]), tires[1])
            penalties = real_data["penalties"]
            sectors = [lap_time / 3] * 3  # Approximate sectors (OpenF1 sector data requires additional endpoint)
            leaderboard.append({
                "name": driver["name"],
                "team": driver["team"],
                "best_lap": lap_time,
                "color": driver["color"],
                "best_sectors": sectors,
                "tire": tire["type"],
                "penalties": penalties,
                "dnf": False,
                "position": real_data["position"]
            })
            team_metrics[driver["team"]]["lap_times"].append(lap_time)
            for i, sector in enumerate(sectors):
                team_metrics[driver["team"]]["sectors"][i].append(sector)
            team_metrics[driver["team"]]["tire_wear"] += tire["wear_rate"]
            recommendation = recommend_strategy(race, 1, driver, tire, team_metrics[driver["team"]]["tire_wear"], team_metrics, leaderboard, strategy_model, strategy_le)
            live_updates.append(recommendation)
            update = {
                'driver': driver["name"],
                'team': driver["team"],
                'lap': 1,
                'lap_time': format_time(lap_time),
                'sectors': [round(s, 3) for s in sectors],
                'tire_wear': round(team_metrics[driver["team"]]["tire_wear"], 2)
            }
            live_updates.append(update)
            socketio.emit('lap_update', update)
    
    leaderboard.sort(key=lambda x: x['position'])
    global_leaderboard = leaderboard
    global_race = race
    global_race['weather_history'] = [(1, False)]  # Saudi GP was dry
    return leaderboard, team_metrics

def analyze_performance(leaderboard, team_metrics, race):
    analysis = {"cars": {}, "drivers": {}, "ai_accuracy": {}, "clusters": {}, "clusters_comment": ""}
    
    for team in team_metrics:
        lap_times = team_metrics[team]['lap_times']
        avg_lap = sum(lap_times) / len(lap_times) if lap_times else None
        analysis["cars"][team] = {
            "avg_lap": format_time(avg_lap),
            "comment": f"{team}'s car performed {'well' if avg_lap and avg_lap < 89 else 'averagely'} in dry conditions."
        }
    
    for driver in leaderboard:
        team = driver['team']
        avg_team_lap = sum(team_metrics[team]['lap_times']) / len(team_metrics[team]['lap_times']) if team_metrics[team]['lap_times'] else None
        analysis["drivers"][driver['name']] = {
            "lap_time": format_time(driver['best_lap']),
            "comment": f"{driver['name']} finished P{driver['position']}."
        }
    
    for driver in leaderboard:
        pred = next((p for p in ai_predictions if p['driver'] == driver['name']), None)
        if pred and driver['best_lap'] is not None:
            error = abs(driver['best_lap'] - pred['predicted_lap'])
            analysis["ai_accuracy"][driver['name']] = {
                "predicted_lap": format_time(pred['predicted_lap']),
                "actual_lap": format_time(driver['best_lap']),
                "error": f"{error:.3f}s",
                "comment": f"AI predicted {driver['name']}'s lap time as {format_time(pred['predicted_lap'])}."
            }
    
    valid_drivers = [d for d in leaderboard if d['best_lap'] is not None]
    if len(valid_drivers) >= 3:
        features = [[d['best_lap'], 0] for d in valid_drivers]  # Simplified clustering
        kmeans = KMeans(n_clusters=3, random_state=42)
        labels = kmeans.fit_predict(features)
        
        clusters = {"Elite": [], "Competitive": [], "Struggling": []}
        for driver, label in zip(valid_drivers, labels):
            cluster_name = ["Elite", "Competitive", "Struggling"][label]
            clusters[cluster_name].append(driver['name'])
        
        analysis["clusters"] = clusters
        analysis["clusters_comment"] = "Drivers clustered by lap times."
    
    return analysis

def plot_lap_times(leaderboard, race, team_metrics, analysis):
    plots_dir = os.path.normpath("/Users/tara/Desktop/F1LapTracker/plots")
    static_plots_dir = os.path.normpath("/Users/tara/Desktop/F1LapTracker/static/plots")
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(static_plots_dir, exist_ok=True)
    
    race_name_safe = race["name"].replace(" ", "_").lower()
    lap_plot_path = os.path.join(plots_dir, f"{race_name_safe}_lap_times.png")
    
    if not os.path.exists(lap_plot_path):
        valid_drivers = [d for d in leaderboard if not d.get('dnf', False)]
        driver_names = [d['name'] for d in valid_drivers]
        lap_times = [d['best_lap'] for d in valid_drivers]
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(driver_names, lap_times, color='red', edgecolor='black')
        plt.title(f"{race['name']} Best Lap Times")
        plt.xlabel("Driver")
        plt.ylabel("Lap Time (seconds)")
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(lap_plot_path)
        plt.close()
        
        static_lap_plot_path = os.path.join(static_plots_dir, f"{race_name_safe}_lap_times.png")
        shutil.copy(lap_plot_path, static_lap_plot_path)
        os.chmod(static_lap_plot_path, 0o644)
    
    return lap_plot_path, "", ""  # Only generate lap time plot for simplicity

def display_leaderboard(leaderboard, race, team_metrics):
    race_type = "Sprint" if race["sprint"] else "Grand Prix"
    analysis = analyze_performance(leaderboard, team_metrics, race)
    lap_plot_path, sector_plot_path, cluster_plot_path = plot_lap_times(leaderboard, race, team_metrics, analysis)
    
    with open("/Users/tara/Desktop/F1LapTracker/race_results.txt", "w") as f:
        f.write(f"F1 Lap Time Tracker - {race['name']} {race_type}\n")
        for i, driver in enumerate(leaderboard, 1):
            line = f"P{i}: {driver['name']} ({driver['team']}) - Best Lap: {format_time(driver['best_lap'])} (Tire: {driver['tire']})"
            if driver['penalties'] > 0:
                line += f", +{driver['penalties']}s Penalty"
            f.write(line + "\n")
    
    return analysis, lap_plot_path, sector_plot_path, cluster_plot_path

@app.route('/plots/<path:filename>')
def serve_plots(filename):
    plots_dir = os.path.normpath('/Users/tara/Desktop/F1LapTracker/plots')
    file_path = os.path.normpath(os.path.join(plots_dir, filename))
    if not os.path.exists(file_path):
        abort(404, description=f"Plot file {filename} not found")
    return send_from_directory(plots_dir, filename)

@socketio.on('fan_comment')
def handle_fan_comment(data):
    comment = data.get('comment', '')
    if comment:
        response = process_fan_comment(comment, global_leaderboard, ai_predictions, global_race)
        socketio.emit('fan_comment', response)

@app.route('/')
def dashboard():
    race = global_race
    leaderboard = global_leaderboard
    race_type = "Sprint" if race.get('sprint', False) else "Grand Prix"
    is_rainy = race.get('rainy', False)
    weather_history = race.get('weather_history', [(1, is_rainy)])
    analysis = getattr(dashboard, 'last_analysis', {})
    lap_plot_path = getattr(dashboard, 'last_lap_plot', '')
    timestamp = int(time.time())
    race_name_safe = race.get('name', '').replace(' ', '_').lower()
    lap_plot_url = f"{race_name_safe}_lap_times.png" if lap_plot_path else ''
    return render_template_string('''
        <!DOCTYPE html>
        <html>
        <head>
            <title>F1 Lap Time Tracker 2025</title>
            <script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.5.1/socket.io.js"></script>
            <style>
                body { font-family: Arial, sans-serif; background: #f1f1f1; text-align: center; margin: 0; padding: 10px; }
                h1, h2 { color: #e10600; font-size: 1.5em; }
                table { width: 100%; max-width: 1000px; margin: 10px auto; border-collapse: collapse; font-size: 0.9em; }
                th, td { border: 1px solid #ddd; padding: 6px; text-align: left; }
                th { background: #e10600; color: white; }
                tr:nth-child(even) { background: #fff; }
                .button { background: #e10600; color: white; padding: 8px 16px; text-decoration: none; border-radius: 5px; display: inline-block; font-size: 0.9em; }
                .button:hover { background: #b50500; }
                #live-feed, #analysis, #fan-interaction { margin: 10px auto; width: 100%; max-width: 1000px; text-align: left; font-size: 0.85em; border: 1px solid #ddd; padding: 10px; background: #fff; }
                #live-feed, #fan-interaction { max-height: 200px; overflow-y: auto; }
                .fan-input { width: 100%; padding: 5px; margin: 5px 0; }
                .fan-submit { background: #e10600; color: white; border: none; padding: 5px 10px; cursor: pointer; }
                img.plot { max-width: 100%; height: auto; margin: 10px auto; display: block; }
            </style>
        </head>
        <body>
            <h1>🏎️ F1 Lap Time Tracker - {{ race['name'] if race else '2025 Season' }} {{ race_type if race else '' }} 🏁</h1>
            {% if race %}
            <div id="live-feed">
                <strong>Live Race Feed:</strong><br>
            </div>
            <div id="fan-interaction">
                <strong>Fan Interaction:</strong><br>
                <input type="text" id="fan-comment" class="fan-input" placeholder="Ask about the race">
                <button class="fan-submit" onclick="submitComment()">Submit</button>
                <div id="fan-responses"></div>
            </div>
            <table>
                <tr>
                    <th>Position</th>
                    <th>Driver</th>
                    <th>Team</th>
                    <th>Best Lap</th>
                    <th>Tire</th>
                    <th>Status</th>
                </tr>
                {% for driver in leaderboard %}
                <tr>
                    <td>P{{ driver['position'] }}</td>
                    <td>{{ driver['name'] }}</td>
                    <td>{{ driver['team'] }}</td>
                    <td>{{ format_time(driver['best_lap']) }}</td>
                    <td>{{ driver['tire'] }}</td>
                    <td>{{ 'DNF' if driver['dnf'] else ('+' ~ driver['penalties'] ~ 's Penalty' if driver['penalties'] > 0 else 'Running') }}</td>
                </tr>
                {% endfor %}
            </table>
            <div id="analysis">
                <h2>Performance Analysis</h2>
                {% if lap_plot_url %}
                <img src="{{ url_for('static', filename='plots/' + lap_plot_url) }}?t={{ timestamp }}"
                     class="plot" alt="Lap Times Plot">
                {% endif %}
                <strong>Car Performance:</strong>
                <ul>
                    {% for team, data in analysis.cars.items() %}
                    <li>{{ team }}: {{ data.comment }}</li>
                    {% endfor %}
                </ul>
                <strong>Driver Performance:</strong>
                <ul>
                    {% for driver, data in analysis.drivers.items() %}
                    <li>{{ driver }}: {{ data.comment }}</li>
                    {% endfor %}
                </ul>
            </div>
            {% else %}
            <p>No race simulated yet. Select a race below.</p>
            {% endif %}
            <p><a href="{{ url_for('select_race') }}" class="button">Select Another Race</a></p>
            <script>
                var socket = io('http://127.0.0.1:5000', {transports: ['websocket', 'polling'], reconnectionAttempts: 5});
                socket.on('lap_update', function(data) {
                    var feed = document.getElementById('live-feed');
                    feed.innerHTML += `<div>Lap ${data.lap} - ${data.driver} (${data.team}): ${data.lap_time}</div>`;
                    feed.scrollTop = feed.scrollHeight;
                });
                socket.on('strategy_update', function(data) {
                    var feed = document.getElementById('live-feed');
                    feed.innerHTML += `<div>${data.message}</div>`;
                    feed.scrollTop = feed.scrollHeight;
                });
                socket.on('fan_comment', function(data) {
                    var responses = document.getElementById('fan-responses');
                    responses.innerHTML += `<div>${data.timestamp}: "${data.comment}" -> ${data.response}</div>`;
                    responses.scrollTop = responses.scrollHeight;
                });
                function submitComment() {
                    var comment = document.getElementById('fan-comment').value;
                    if (comment.trim()) {
                        socket.emit('fan_comment', {comment: comment});
                        document.getElementById('fan-comment').value = '';
                    }
                }
            </script>
        </body>
        </html>
    ''', race=race, leaderboard=leaderboard, race_type=race_type, is_rainy=is_rainy, 
    weather_history=weather_history, format_time=format_time, analysis=analysis, 
    lap_plot_url=lap_plot_url, timestamp=timestamp, ai_predictions=ai_predictions)

@app.route('/select-race')
def select_race():
    return render_template_string('''
        <!DOCTYPE html>
        <html>
        <head>
            <title>F1 2025 Race Selection</title>
            <style>
                body { font-family: Arial, sans-serif; background: #f1f1f1; text-align: center; margin: 0; padding: 10px; }
                h1 { color: #e10600; font-size: 1.5em; }
                ul { list-style: none; padding: 0; width: 100%; max-width: 600px; margin: 10px auto; }
                li { margin: 8px 0; }
                .button { background: #e10600; color: white; padding: 8px 16px; text-decoration: none; border-radius: 5px; display: block; font-size: 0.9em; }
                .button:hover { background: #b50500; }
            </style>
        </head>
        <body>
            <h1>🏎️ F1 2025 Race Selection 🏁</h1>
            <ul>
                <li><a href="{{ url_for('run_race_route', race_index=4) }}" class="button">Saudi Arabian GP (Apr 18-20, GP, Dry ☀️)</a></li>
            </ul>
        </body>
        </html>
    ''')

@app.route('/run-race/<int:race_index>')
async def run_race_route(race_index):
    try:
        if race_index == 4:  # Saudi Arabian GP
            race = {"name": "Saudi Arabian GP", "date": "Apr 18-20", "sprint": False, "rainy": False}
            leaderboard, team_metrics = await run_race(race, laps=1)  # Single lap for simplicity
            analysis, lap_plot_path, sector_plot_path, cluster_plot_path = display_leaderboard(leaderboard, race, team_metrics)
            dashboard.last_analysis = analysis
            dashboard.last_lap_plot = lap_plot_path
            dashboard.last_sector_plot = sector_plot_path
            dashboard.last_cluster_plot = cluster_plot_path
        else:
            raise ValueError("Invalid race index")
        return redirect(url_for('dashboard'))
    except Exception as e:
        print(f"Error in run_race: {e}")
        return render_template_string('<h1>Error</h1><p>Sorry, something went wrong.</p><p><a href="{{ url_for('select_race') }}">Back</a></p>'), 500

def main():
    print("🏎️ Starting F1 Lap Time Tracker 2025! 🏁")
    print("Open http://127.0.0.1:5000 in your browser.")
    socketio.run(app, debug=False, port=5000, host='0.0.0.0')

if __name__ == "__main__":
    main()
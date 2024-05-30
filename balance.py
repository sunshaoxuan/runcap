import pandas as pd
import requests
from datetime import datetime, timedelta
from tqdm import tqdm
import curses
import re
import time
import sqlite3
import hashlib
import os
import json

# 更新的模型价格表
MODEL_PRICES = {
    "gpt-4o": {"context_token_cost": 0.000005, "generated_token_cost": 0.000015},
    "gpt-4o-2024-05-13": {"context_token_cost": 0.000005, "generated_token_cost": 0.000015},
    "gpt-4-turbo": {"context_token_cost": 0.00001, "generated_token_cost": 0.00003},
    "gpt-4-turbo-2024-04-09": {"context_token_cost": 0.00001, "generated_token_cost": 0.00003},
    "gpt-4-0613": {"context_token_cost": 0.00003, "generated_token_cost": 0.00006},
    "gpt-4-32k": {"context_token_cost": 0.00006, "generated_token_cost": 0.00012},
    "gpt-4-0125-preview": {"context_token_cost": 0.00001, "generated_token_cost": 0.00003},
    "gpt-4-1106-preview": {"context_token_cost": 0.00001, "generated_token_cost": 0.00003},
    "gpt-4": {"context_token_cost": 0.00003, "generated_token_cost": 0.00006},
    "gpt-4-vision-preview": {"context_token_cost": 0.00001, "generated_token_cost": 0.00003},
    "gpt-3.5-turbo-0125": {"context_token_cost": 0.0000005, "generated_token_cost": 0.0000015},
    "gpt-3.5-turbo-1106": {"context_token_cost": 0.000001, "generated_token_cost": 0.000002},
    "gpt-3.5-turbo-0613": {"context_token_cost": 0.0000015, "generated_token_cost": 0.000002},
    "gpt-3.5-turbo-16k-0613": {"context_token_cost": 0.000003, "generated_token_cost": 0.000004},
    "gpt-3.5-turbo-0301": {"context_token_cost": 0.0000015, "generated_token_cost": 0.000002},
    "gpt-3.5-turbo-instruct": {"context_token_cost": 0.0000015, "generated_token_cost": 0.000002},
}

def initialize_database(db_path='openai_usage_cache.db'):
    """Initialize the SQLite database and create the table if it does not exist."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS usage_cache (
            id TEXT PRIMARY KEY,
            api_key TEXT,
            date TEXT,
            usage_data TEXT
        )
    ''')
    # 检查并添加新的字段
    cursor.execute('PRAGMA table_info(usage_cache)')
    columns = [info[1] for info in cursor.fetchall()]
    if 'update_date' not in columns:
        cursor.execute('ALTER TABLE usage_cache ADD COLUMN update_date TEXT')
    
    conn.commit()
    conn.close()

def update_existing_records(db_path='openai_usage_cache.db'):
    """Update existing records with a default update_date in the correct format."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('SELECT id, date FROM usage_cache WHERE update_date IS NULL OR update_date NOT LIKE "____-__-__"')
    records = cursor.fetchall()
    
    for record in records:
        id, date = record
        cursor.execute('UPDATE usage_cache SET update_date = ? WHERE id = ?', (date, id))
    
    conn.commit()
    conn.close()


def generate_id(api_key, date):
    """Generate a unique ID using MD5 hash of the API key and date."""
    unique_string = f"{api_key}_{date}"
    return hashlib.md5(unique_string.encode()).hexdigest()

def get_cached_usage(api_key, date, db_path='openai_usage_cache.db'):
    """Retrieve cached usage data from the database."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    id = generate_id(api_key, date)
    cursor.execute('SELECT usage_data, update_date FROM usage_cache WHERE id = ?', (id,))
    result = cursor.fetchone()
    conn.close()
    if result:
        return result
    return None

def cache_usage_data(api_key, date, usage_data, db_path='openai_usage_cache.db'):
    """Cache usage data in the database."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    id = generate_id(api_key, date)
    update_date = datetime.now().strftime('%Y-%m-%d')
    cursor.execute('''
        INSERT OR REPLACE INTO usage_cache (id, api_key, date, usage_data, update_date)
        VALUES (?, ?, ?, ?, ?)
    ''', (id, api_key, date, usage_data, update_date))
    conn.commit()
    conn.close()

def read_api_keys(file_path):
    """Read API keys from a file."""
    df = pd.read_csv(file_path, skiprows=1, names=['Key Name', 'API Key', 'Creation Date'])
    df['Creation Date'] = pd.to_datetime(df['Creation Date'], format='%Y-%m-%d')
    return df

def fetch_daily_usage(api_key, date, stdscr, current_progress, db_path='openai_usage_cache.db'):
    """Fetches usage data for a specific date with retry logic and caching."""
    usage_data = None
    cached_data = get_cached_usage(api_key, date, db_path)
    if cached_data:
        usage_data, update_date = cached_data
        try:
            if update_date is None or datetime.strptime(update_date, '%Y-%m-%d') < datetime.strptime(date, '%Y-%m-%d'):
                usage_data = None  # Force update if the update date is invalid or outdated
            else:
                usage_data = json.loads(usage_data)
                if len(usage_data["data"]) == 0:
                    usage_data = None
                else:
                    return usage_data
        except ValueError:
            usage_data = None  # Handle potential format issues, force update

    if usage_data is None:
        url = f"https://api.openai.com/v1/usage?date={date}"
        headers = {
            "Authorization": f"Bearer {api_key}"
        }
        
        max_retries = 5
        backoff_factor = 1

        for attempt in range(max_retries):
            response = requests.get(url, headers=headers)
            
            if response.status_code == 200:
                usage_data = response.json()
                cache_usage_data(api_key, date, json.dumps(usage_data), db_path)
                time.sleep(3)  # Limit request frequency, only for remote calls
                return usage_data
            elif response.status_code == 429:  # Rate limit exceeded
                retry_after = int(response.headers.get("Retry-After", backoff_factor * (2 ** attempt)))
                retry_message = f"Rate limit exceeded. Retrying after {retry_after} seconds..."
                max_y, max_x = stdscr.getmaxyx()
                if current_progress + 2 < max_y:
                    stdscr.addstr(current_progress + 2, 0, retry_message[:max_x-1], curses.color_pair(2) | curses.A_BLINK)
                    stdscr.refresh()
                    time.sleep(retry_after)
                    stdscr.addstr(current_progress + 2, 0, " " * (len(retry_message[:max_x-1])))
                    stdscr.refresh()
                else:
                    print(retry_message)  # Print to console if screen lines exceed limit
                    time.sleep(retry_after)
            else:
                error_message = f"Error fetching data for {date}: {response.status_code} {response.text}"
                max_y, max_x = stdscr.getmaxyx()
                if current_progress + 2 < max_y:
                    stdscr.addstr(current_progress + 2, 0, error_message[:max_x-1])
                    stdscr.refresh()
                else:
                    print(error_message)  # Print to console if screen lines exceed limit
                break

    return usage_data


def match_model_prices(model_name):
    """Match model name with its prices using regular expressions."""
    for pattern, prices in MODEL_PRICES.items():
        if re.search(pattern, model_name):
            return prices
    return {"context_token_cost": 0.1, "generated_token_cost": 0.2}  # 默认价格，单位是每百万token的费用

def collect_monthly_usage(api_key, creation_date, year, month, stdscr):
    """Collects usage data for each day in a specified month and year."""
    start_date = max(datetime(year, month, 1), creation_date)
    end_date = datetime(year, month + 1, 1) if month < 12 else datetime(year + 1, 1, 1)

    records = []
    date_range = list(pd.date_range(start=start_date, end=end_date-timedelta(days=1)))
    for idx, current_date in enumerate(tqdm(date_range, desc="Processing")):
        formatted_date = current_date.strftime('%Y-%m-%d')
        stdscr.addstr(1, 0, f"Processing: {int((idx + 1) / len(date_range) * 100)}%", curses.color_pair(1))
        stdscr.refresh()
        daily_usage = fetch_daily_usage(api_key, formatted_date, stdscr, idx + 1)
        if daily_usage and len(daily_usage["data"]) >= 0:
            for item in daily_usage['data']:
                model = item.get('snapshot_id')
                context_tokens = item.get('n_context_tokens_total')
                generated_tokens = item.get('n_generated_tokens_total')
                model_prices = match_model_prices(model)
                context_cost = round(context_tokens * model_prices["context_token_cost"], 4)
                generated_cost = round(generated_tokens * model_prices["generated_token_cost"], 4)
                total_cost = round(context_cost + generated_cost, 4)

                records.append({
                    'Date': formatted_date,
                    'Operation': item.get('operation'),
                    'Model': model,
                    'Context Tokens': context_tokens,
                    'Context Price (USD/1M)': round(model_prices["context_token_cost"] * 1000000, 2),
                    'Context Cost (USD)': context_cost,
                    'Generated Tokens': generated_tokens,
                    'Generated Price (USD/1M)': round(model_prices["generated_token_cost"] * 1000000, 2),
                    'Generated Cost (USD)': generated_cost,
                    'Total Cost (USD)': total_cost
                })
    return records


def format_dataframe(df):
    """Format dataframe for better readability."""
    output = []
    header = df.columns
    max_lens = [max(df[col].astype(str).apply(len).max(), len(col)) for col in header]
    separator = "=" * (sum(max_lens) + 3 * (len(header) - 1))
    header_line = " | ".join(f"{header[i]:<{max_lens[i]}}" for i in range(len(header)))
    output.append(header_line)
    output.append(separator)
    for _, row in df.iterrows():
        formatted_row = " | ".join(f"{str(row.iloc[i]):>{max_lens[i]}}" for i in range(len(header)))
        output.append(formatted_row)
    return "\n".join(output)

def menu(stdscr, options, prompt="Use arrow keys to select and press Enter:"):
    curses.curs_set(0)
    current_row = 0

    while True:
        stdscr.clear()
        height, width = stdscr.getmaxyx()
        prompt_x = max(0, (width - len(prompt)) // 2)
        prompt_y = max(0, height // 2 - len(options) // 2 - 1)
        stdscr.addstr(prompt_y, prompt_x, prompt[:width-1])

        for idx, row in enumerate(options):
            x = max(0, (width - len(row)) // 2)
            y = prompt_y + idx + 2
            if y < height:  # 确保行不会超过终端高度
                if idx == current_row:
                    stdscr.attron(curses.color_pair(1))
                    stdscr.addstr(y, x, row[:width-3])
                    stdscr.attroff(curses.color_pair(1))
                else:
                    stdscr.addstr(y, x, row[:width-3])

        stdscr.refresh()

        key = stdscr.getch()

        if key == curses.KEY_UP and current_row > 0:
            current_row -= 1
        elif key == curses.KEY_DOWN and current_row < len(options) - 1:
            current_row += 1
        elif key == curses.KEY_ENTER or key in [10, 13]:
            return current_row
        elif key == 27:  # ESC key
            return -1



def main(stdscr):
    curses.start_color()
    curses.init_pair(1, curses.COLOR_BLACK, curses.COLOR_WHITE)
    curses.init_pair(2, curses.COLOR_RED, curses.COLOR_BLACK)

    while True:
        api_keys_df = read_api_keys('api_keys.txt')
        key_names = api_keys_df['Key Name'].tolist()
        key_names.append("Exit")
        selected_index = menu(stdscr, key_names, prompt="Select an API Key:")
        if selected_index == -1 or key_names[selected_index] == "Exit":
            break
        
        selected_key_name = key_names[selected_index]
        selected_key_info = api_keys_df.loc[api_keys_df['Key Name'] == selected_key_name]
        selected_key = selected_key_info['API Key'].values[0]
        creation_date = pd.to_datetime(selected_key_info['Creation Date'].values[0]).to_pydatetime()
        creation_year = creation_date.year
        creation_month = creation_date.month

        while True:
            current_year = datetime.now().year
            current_month = datetime.now().month
            months = [datetime(year, month, 1).strftime('%B %Y') for year in range(creation_year, current_year + 1)
                      for month in range(1, 13) if (year > creation_year or month >= creation_month) and (year < current_year or month <= current_month)]
            months.append("Back")
            selected_month_index = menu(stdscr, months, prompt="Select a month or press ESC to go back:")
            if selected_month_index == -1 or months[selected_month_index] == "Back":
                break
            
            selected_month_str = months[selected_month_index]
            selected_month = datetime.strptime(selected_month_str, '%B %Y').month
            selected_year = datetime.strptime(selected_month_str, '%B %Y').year
            
            stdscr.clear()
            stdscr.refresh()
            curses.endwin()  # 退出curses模式以显示进度条
            print(f"\nCollecting data for {selected_key_name} for {selected_month_str}...\n")
            curses.initscr()  # 重新进入curses模式
            stdscr = curses.initscr()
            usage_records = collect_monthly_usage(selected_key, creation_date, selected_year, selected_month, stdscr)
            
            df = pd.DataFrame(usage_records)
            if not df.empty:
                grouped_df = df.groupby(['Date', 'Model']).agg({
                    'Operation': lambda x: ', '.join(set(x)),
                    'Context Tokens': 'sum',
                    'Context Price (USD/1M)': 'first',
                    'Context Cost (USD)': 'sum',
                    'Generated Tokens': 'sum',
                    'Generated Price (USD/1M)': 'first',
                    'Generated Cost (USD)': 'sum',
                    'Total Cost (USD)': 'sum'
                }).reset_index()
                
                total_row = grouped_df.sum(numeric_only=True)
                total_row['Date'] = 'Total'
                total_row['Model'] = ''
                total_row['Operation'] = ''
                total_row['Context Price (USD/1M)'] = ''
                total_row['Generated Price (USD/1M)'] = ''
                total_row['Context Cost (USD)'] = round(total_row['Context Cost (USD)'], 4)
                total_row['Generated Cost (USD)'] = round(total_row['Generated Cost (USD)'], 4)
                total_row['Total Cost (USD)'] = round(total_row['Total Cost (USD)'], 4)
                total_row = pd.DataFrame(total_row).T

                final_df = pd.concat([grouped_df, total_row], ignore_index=True)

                final_df['Context Cost (USD)'] = final_df['Context Cost (USD)'].apply(lambda x: f"{float(x):.4f}" if pd.notna(x) else '')
                final_df['Generated Cost (USD)'] = final_df['Generated Cost (USD)'].apply(lambda x: f"{float(x):.4f}" if pd.notna(x) else '')
                final_df['Total Cost (USD)'] = final_df['Total Cost (USD)'].apply(lambda x: f"{float(x):.4f}" if pd.notna(x) else '')

                stdscr.clear()
                stdscr.addstr(0, 0, f"Usage for {selected_key_name} - {selected_month_str}\n")
                formatted_data = format_dataframe(final_df)
                max_y, max_x = stdscr.getmaxyx()
                for idx, line in enumerate(formatted_data.split("\n")):
                    if idx + 1 < max_y:
                        stdscr.addstr(idx + 1, 0, line[:max_x-1])  # 确保行不会超过终端宽度
                if len(formatted_data.split("\n")) + 2 < max_y:
                    stdscr.addstr(len(formatted_data.split("\n")) + 2, 0, f"Total Cost (USD): {float(total_row['Total Cost (USD)'].values[0]):.2f}", curses.A_REVERSE)
                stdscr.addstr(len(formatted_data.split("\n")) + 3, 0, "\nPress any key to return to menu...")
                stdscr.refresh()
                stdscr.getch()
            else:
                stdscr.clear()
                stdscr.addstr(0, 0, "No data available for the specified period.\nPress any key to return to menu...")
                stdscr.refresh()
                stdscr.getch()

if __name__ == '__main__':
    os.system('cls' if os.name == 'nt' else 'clear')
    initialize_database()
    update_existing_records()  # 更新现有记录的 update_date 字段
    curses.wrapper(main)

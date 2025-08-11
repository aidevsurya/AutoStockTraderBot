import gspread
import pandas as pd
from oauth2client.service_account import ServiceAccountCredentials

# --- Connect to Google Sheets ---
def connect_gsheet():
    scope = [
        "https://spreadsheets.google.com/feeds",
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive.file",
        "https://www.googleapis.com/auth/drive"
    ]
    creds = ServiceAccountCredentials.from_json_keyfile_name("credentials.json", scope)
    return gspread.authorize(creds)

# --- Create a new spreadsheet ---
def create_spreadsheet(client, title):
    return client.create(title)

# --- Add a new sheet tab ---
def add_sheet(spreadsheet, title, rows=100, cols=26):
    return spreadsheet.add_worksheet(title=title, rows=str(rows), cols=str(cols))

# --- Delete a sheet tab ---
def delete_sheet(spreadsheet, worksheet):
    spreadsheet.del_worksheet(worksheet)

# --- Rename a sheet tab ---
def rename_sheet(worksheet, new_title):
    worksheet.update_title(new_title)

# --- List all sheet tabs ---
def list_sheets(spreadsheet):
    return [ws.title for ws in spreadsheet.worksheets()]

# --- Open an existing spreadsheet by name ---
def open_spreadsheet(client, title):
    return client.open(title)

# --- Open a specific sheet tab by name ---
def open_sheet_tab(spreadsheet, tab_name):
    return spreadsheet.worksheet(tab_name)

# --- Read all data into DataFrame ---
def read_sheet(worksheet):
    data = worksheet.get_all_values()
    return pd.DataFrame(data[1:], columns=data[0]) if data else pd.DataFrame()

# --- Update a specific cell ---
def update_cell(worksheet, row, col, value):
    worksheet.update_cell(row, col, value)

# --- Append rows ---
def append_rows(worksheet, rows):
    worksheet.append_rows(rows, value_input_option="USER_ENTERED")

# --- Overwrite sheet with DataFrame ---
def write_dataframe(worksheet, df):
    data = [df.columns.tolist()] + df.astype("str").values.tolist()
    worksheet.update(data)
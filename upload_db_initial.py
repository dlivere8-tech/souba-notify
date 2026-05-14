"""
初回のみ実行: ローカルDuckDBをGoogle Driveにアップロードしてfile_idを得る。

使い方:
  1. Google Drive上に「souba-db」フォルダを作成し、サービスアカウントと共有する
  2. そのフォルダのIDを FOLDER_ID に設定する
  3. python upload_db_initial.py を実行する
  4. 出力された file_id を GitHub Secrets の GDRIVE_DB_FILE_ID に登録する
"""
import os
import sys
from pathlib import Path

# フォルダIDをここに設定（Google DriveのURLの末尾）
FOLDER_ID = "ここにGoogle DriveフォルダIDを入力"

DB_PATH = Path(__file__).parent.parent / "stock_db" / "stock_prices.duckdb"

if not DB_PATH.exists():
    print(f"DBが見つかりません: {DB_PATH}")
    sys.exit(1)

creds_json = os.environ.get('GOOGLE_CREDS', '')
if not creds_json:
    print("環境変数 GOOGLE_CREDS が未設定です")
    print("GitHub Secretsからコピーするか、サービスアカウントJSONを設定してください")
    sys.exit(1)

from gdrive import upload_db_new
file_id = upload_db_new(creds_json, FOLDER_ID, str(DB_PATH))
print(f"\nGitHub Secrets に追加: GDRIVE_DB_FILE_ID = {file_id}")

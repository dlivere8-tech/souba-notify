"""Google Drive上のDuckDBファイルを管理するヘルパー"""
import json
import io
from pathlib import Path
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload
from google.oauth2 import service_account

SCOPES = ['https://www.googleapis.com/auth/drive']


def _get_service(creds_json_str: str):
    creds_info = json.loads(creds_json_str)
    creds = service_account.Credentials.from_service_account_info(
        creds_info, scopes=SCOPES
    )
    return build('drive', 'v3', credentials=creds, cache_discovery=False)


def download_db(creds_json_str: str, file_id: str, local_path: str):
    """Google DriveからDBファイルをダウンロード"""
    service = _get_service(creds_json_str)
    request = service.files().get_media(fileId=file_id)
    with open(local_path, 'wb') as f:
        downloader = MediaIoBaseDownload(f, request, chunksize=4 * 1024 * 1024)
        done = False
        while not done:
            status, done = downloader.next_chunk()
            if status:
                print(f"  DB DL: {int(status.progress() * 100)}%", flush=True)
    print(f"DB downloaded to {local_path} ({Path(local_path).stat().st_size // 1024 // 1024}MB)")


def upload_db(creds_json_str: str, file_id: str, local_path: str):
    """ローカルDBファイルをGoogle Driveにアップロード（既存ファイルを更新）"""
    service = _get_service(creds_json_str)
    media = MediaFileUpload(
        local_path,
        mimetype='application/octet-stream',
        resumable=True,
        chunksize=4 * 1024 * 1024,
    )
    request = service.files().update(fileId=file_id, media_body=media)
    response = None
    while response is None:
        status, response = request.next_chunk()
        if status:
            print(f"  DB UL: {int(status.progress() * 100)}%", flush=True)
    print(f"DB uploaded (file_id={file_id})")


def upload_db_new(creds_json_str: str, folder_id: str, local_path: str) -> str:
    """初回アップロード用: 新規ファイルとしてアップロードしてfile_idを返す"""
    service = _get_service(creds_json_str)
    file_metadata = {
        'name': 'stock_prices.duckdb',
        'parents': [folder_id],
    }
    media = MediaFileUpload(
        local_path,
        mimetype='application/octet-stream',
        resumable=True,
        chunksize=4 * 1024 * 1024,
    )
    request = service.files().create(body=file_metadata, media_body=media, fields='id')
    response = None
    while response is None:
        status, response = request.next_chunk()
        if status:
            print(f"  初回UL: {int(status.progress() * 100)}%", flush=True)
    file_id = response.get('id')
    print(f"初回アップロード完了: file_id={file_id}")
    print(f"→ この file_id を GitHub Secrets の GDRIVE_DB_FILE_ID に設定してください")
    return file_id

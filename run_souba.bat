@echo off
set GMAIL_ADDRESS=dlive.re8@gmail.com
set GMAIL_APP_PASS=gyjrdhlrsnkifskm
set SEND_TO=dlive.re8@gmail.com
python "C:\Users\dlive\OneDrive\ドキュメント\ClaudeCode用\souba-notify\fetch_shinyo.py"
python "C:\Users\dlive\OneDrive\ドキュメント\ClaudeCode用\souba-notify\souba.py"
git -C "C:\Users\dlive\OneDrive\ドキュメント\ClaudeCode用\souba-notify" add docs\data\
git -C "C:\Users\dlive\OneDrive\ドキュメント\ClaudeCode用\souba-notify" commit -m "Update results"
git -C "C:\Users\dlive\OneDrive\ドキュメント\ClaudeCode用\souba-notify" push
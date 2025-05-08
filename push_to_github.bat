@echo off
echo Running git operations...
git add app.py
git add -f uploads/*
git commit -m "Add port change and ensure uploads are included"
git push origin main
echo Done!
pause 
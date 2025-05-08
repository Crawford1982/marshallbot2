Write-Host "Killing all Python processes..." -ForegroundColor Red
taskkill /F /IM python.exe -ErrorAction SilentlyContinue
Start-Sleep -Seconds 2

Write-Host "Running git commands..." -ForegroundColor Green
git add -A
git commit -m "Full repository transfer including all files"
git push -f origin main

Write-Host "Done!" -ForegroundColor Green
Write-Host "Press any key to exit..."
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown") 